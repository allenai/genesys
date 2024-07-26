# Copyright (c) 2024, Simran Arora

import math
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
from train.src.generation import InferenceParams

try:
    from fla.ops.based import fused_chunk_based, parallel_based
    from fla.ops.based.naive import naive_parallel_based
    print(f"Successfully imported the FLA triton kernels! ")
except:
    print(f"Could not import the FLA triton kernels... ")
    

class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(
            self, 
            input_dim: int, 
            scale_dim: Optional[int] = None, 
            eps: float = 1e-12,
            head_dim_idx: int = -1, 
            **kwargs: any
        ):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx     
        self.eps = eps
        self.r2  = math.sqrt(2)
        if scale_dim is None:
            scale_dim = self.input_dim
        self.rd  = math.sqrt(scale_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        return torch.cat([x[..., :1] ** 0, 
                          x / self.rrd, x2 / self.rd], dim=self.head_dim_idx)


class PrefixLinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=None,
        feature_dim: int = 16,
        eps: float = 1e-12,
        scale_dim: Optional[int] = None,
        enc_length = None,
        dec_length = None,
        device=None,
        dtype=None,
        layer_idx=-1,
        implementation="default",
        use_mask=True,
        **kwargs,
    ) -> None:
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}

        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads_kv = num_heads
        assert (self.num_heads % self.num_heads_kv == 0), "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        self.enc_length = enc_length
        self.dec_length = dec_length
    
        # For linear attention
        self.d_model = embed_dim
        self.feature_dim = feature_dim
        feature_map_kwargs = {
            'input_dim': self.feature_dim,
            'head_dim_idx': -1,
            'eps': 1e-12,
            "scale_dim": scale_dim,
        }
        self.feature_map = TaylorExp(**feature_map_kwargs)
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_heads_kv * self.head_dim, bias=False)

        self.proj_k_enc = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v_enc = nn.Linear(self.d_model, self.num_heads_kv * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()
        self.eps = eps

        self.implementation = implementation
        self.use_mask = use_mask

        print(f"{self.enc_length=}, {self.dec_length=}, {self.implementation=}, {self.use_mask=}")

    def forward(
        self,
        x,
        inference_params: InferenceParams = None,
        **kwargs,
    ):
        batch, seqlen = x.shape[:2]
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.view(batch, seqlen, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(batch, seqlen, self.num_heads, self.feature_dim).transpose(1, 2)
        v = v.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        mask = None
        num_zeros = None
        if 'mask' in kwargs:
            mask = kwargs['mask']
        elif 'attn_mask' in kwargs:
            mask = kwargs['attn_mask']
        elif inference_params is not None:
            try:
                mask = inference_params.mask 
            except:
                mask = None

        # Encoder part
        k_enc, v_enc = None, None
        if inference_params is None or inference_params.seqlen_offset == 0: # We only do encoder in this region
            enc_length = min(self.enc_length, seqlen) # In inference, enc_length could be > seqlen
            x_enc = x[:, :enc_length]
            k_enc, v_enc = self.proj_k_enc(x_enc), self.proj_v_enc(x_enc)
            k_enc = k_enc.view(batch, enc_length, self.num_heads, self.feature_dim).transpose(1, 2)
            v_enc = v_enc.view(batch, enc_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Applying masks 
        # if mask is not None and q.shape[2] > 1 and mask.shape[1] <= enc_length and self.use_mask: 
        #     # Second condition checks that we're in prefill
        #     # num_zeros = mask.shape[1] - torch.sum(mask[0], dim=-1) 
        #     if len(mask.shape) == 4:
        #         lin_attn_mask = (mask == 0)[:, :1, -1, :][..., None]  # b, 1, k_len, 1
        #     else:
        #         lin_attn_mask = mask[:, None, :, None]  # b, 1, k_len, 1
        #     lin_attn_mask = lin_attn_mask.to(torch.bool)
        #     k = k.masked_fill(~lin_attn_mask, 0)
        #     k_enc = k_enc.masked_fill(~lin_attn_mask, 0)

        if inference_params is None:
            y = self.parallel_forward(x, q, k, v, k_enc, v_enc, mask=mask)
            return y
        
        else:
            # check if we're doing prefill or generation
            if inference_params.seqlen_offset > 0: 
                
                # recurrent generation
                kv_state_dec, k_state_dec = self._get_inference_cache(inference_params)
                q, k = self.feature_map(q), self.feature_map(k)
                if k_enc is not None: 
                    assert 0, print(f"Need to left pad your prefill!")
                return self.recurrent_forward(x, kv_state_dec, k_state_dec, q, k, v)
            
            else:  
                y = self.parallel_forward(x, q, k, v, k_enc, v_enc, mask=mask)
                in_dt = q.dtype 
                dt = q.dtype #torch.float32

                # Encoder part
                k_enc = self.feature_map(k_enc)
                kv_state_enc = torch.einsum("bhnd,bhnf->bhfd", k_enc.to(dt), v_enc.to(dt))[:, :, None]
                k_state_enc = k_enc.sum(dim=2)[:, :, None, None]

                # Decoder part
                q, k = self.feature_map(q), self.feature_map(k)
                kv_state_dec = torch.einsum("bhnd,bhnf->bhfd", k.to(dt), v.to(dt))[:, :, None]
                k_state_dec = k.to(dt).sum(dim=2)[:, :, None, None]

                # Combined State
                kv_state_dec += kv_state_enc
                k_state_dec += k_state_enc

                if self.layer_idx in inference_params.key_value_memory_dict:
                    # update the state in-place when graph caching is enabled
                    inference_params.key_value_memory_dict[self.layer_idx][0].copy_(kv_state_dec.to(in_dt))
                    inference_params.key_value_memory_dict[self.layer_idx][1].copy_(k_state_dec.to(in_dt))
                else: 
                    inference_params.key_value_memory_dict[self.layer_idx] = (kv_state_dec.to(in_dt), k_state_dec.to(in_dt))
                return y


    def parallel_forward(self, x, q, k, v, k_enc=None, v_enc=None, mask=None):
        batch, seqlen = x.shape[:2]
        k_enc = self.feature_map(k_enc)
        in_dt = q.dtype 
        dt = q.dtype #torch.float32

        # Standard attention
        if self.implementation == "default": 
            q, k = self.feature_map(q), self.feature_map(k)
            A_qk = torch.einsum("bhnd,bhmd->bhnm", q.to(dt), k.to(dt)) 
            try:
                A_qk = torch.tril(A_qk)
            except:
                cumsum_matrix = torch.tril(torch.ones((seqlen, seqlen), device=A_qk.device))
                A_qk = A_qk * cumsum_matrix
            y1 = torch.einsum("bhnm,bhme->bhne", A_qk.to(dt), v.to(dt))

        elif self.implementation == "fla_parallel": 
            y1, z1 = parallel_based(q, k, v, use_scale=True, use_normalize=False, return_both=True) # false to handle denom seperately
            q = self.feature_map(q)
        else:
            assert 0, print(f"Invalid implementation: {self.implementation}")

        # Cross attention
        A_qk_2 = torch.einsum("bhnd,bhmd->bhnm", q.to(dt), k_enc.to(dt))
        y2 = torch.einsum("bhnm,bhme->bhne", A_qk_2.to(dt), v_enc.to(dt))

        # Denominator
        if self.implementation == "default": 
            k_state = k_enc.to(dt).sum(dim=2, keepdim=True) +  k.to(dt).cumsum(2)
            z = 1 / ((q.to(dt) * k_state.to(dt)).sum(dim=-1)  + self.eps)
            y1 = y1 * z[..., None]
            output_1 = rearrange(y1, 'b h l d -> b l (h d)')
            y2 = y2 * z[..., None]
            output_2 = rearrange(y2, 'b h l d -> b l (h d)')
            output = output_1 + output_2

        elif self.implementation == "fla_parallel": 
            z2 = (q * k_enc.to(dt).sum(dim=2, keepdim=True)).sum(dim=-1)
            z = 1 / ((z2.to(dt) + z1.to(dt)) + self.eps)
            output = (y1 +  y2) * z[..., None]
            output = rearrange(output, 'b h l d -> b l (h d)')

        else:
            assert 0, print(f"Invalid implementation: {self.implementation}")
            
        output = self.out_proj(output)
        return output.to(in_dt)


    def recurrent_forward(self, x, kv_state, k_state, q, k, v):
        b, h, l, d = q.shape
        assert l == 1, f'q.shape is {q.shape} but should be ({b}, {h}, 1, {d})'
        in_dt = q.dtype 
        dt = q.dtype #torch.float32

        # Expand dims for broadcasting to compute linear attention
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        kv_state += k[:, :, -1:] * v[:, :, -1:]
        k_state  += k[:, :, -1:]

        num_attn = (q.to(dt) * kv_state.to(dt)).sum(dim=-1)
        y = num_attn / ((q.to(dt) * k_state.to(dt)).sum(dim=-1) + self.eps) 
        y = rearrange(y, 'b h l d -> b l (h d)')
        return self.out_proj(y).to(in_dt)

    
    def expanded_size(self):
        return self.feature_dim ** 2 + self.feature_dim + 1
    

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Creates a state tensor of shape ..."""

        kv_shape = (
            batch_size, self.num_heads, 1, self.head_dim, self.expanded_size()
        )
        k_shape = (
            batch_size, self.num_heads, 1, 1, self.expanded_size()
        )
        
        kv_state_dec = torch.zeros(*kv_shape, dtype=dtype, device=self.out_proj.weight.device)
        k_state_dec = torch.zeros(*k_shape, dtype=dtype, device=self.out_proj.weight.device)

        return (kv_state_dec, k_state_dec)
     
    
    def _get_inference_cache(self, inference_params: InferenceParams):
        return inference_params.key_value_memory_dict[self.layer_idx]
