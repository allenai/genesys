import math
from typing import Dict, Optional, Tuple, List
import logging
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import uuid
from einops import rearrange
from efficient_attention.attn_utils import pad_to_multiple,FairseqDropout,T5RelativePositionBias

logger = logging.getLogger(__name__)


def prm_projection(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    normalize: bool=True
    ):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    eps: numerical stabilizer.
    Returns:
    Random features for fast softmax attention.
    """
    # data : [b, h, lk, d]
    # proj : [b, h, lc, d]
    data_normalizer = (data.shape[-1] ** -0.5)
    data_dash = torch.einsum('...nd,...md->...nm', 
                            projection_matrix,
                            (data_normalizer * data),
                            ) # [b, h, lq, lk]
    # norm = (data_normalizer ** 2) * torch.sum(data ** 2, dim=-1).unsqueeze(-2) / 2.0# [b, h, 1, lk]
    norm = data_normalizer * torch.sum(data ** 2, dim=-1).unsqueeze(-2) / 2.0# [b, h, 1, lk]
    if normalize:
        proj_data = F.softmax(data_dash - norm, dim=-1)  # [b, h, l_c, l_k]      
    else:
        proj_data = data_dash - norm
    return proj_data

def window_1d_merge(x):
    return rearrange(x, '... g w d ->... (g w) d')

def causal_window_1d_partition(x, window_size, ext_window_size=0, pad_val=0):
    b, h, n, d = x.shape
    n_groups = n // window_size
    if ext_window_size > 0:
        ext_len = ext_window_size
        x = F.pad(x, (0, 0, ext_len, 0), value=pad_val)
        out_shape = (b, h, n_groups, ext_len + window_size, d)
        strides = x.stride()
        out_stride = (strides[0], strides[1], window_size * strides[2], strides[2], strides[3])
        return torch.as_strided(x, size=out_shape, stride=out_stride)
    else:
        return rearrange(x, '... (g w) d -> ... g w d', w=window_size)

def default(val, d):
    return val if val is not None else d

def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class CausalEVAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        attn_args=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.window_size = attn_args.window_size
        if attn_args.overlap_window:
            self.ext_size = max(1, self.window_size)
        else:
            self.ext_size = 0
        
        self.causal = attn_args.causal
        self.num_chunks = attn_args.num_chunks
        self.chunk_size = attn_args.chunk_size
        if self.chunk_size is not None:
            assert self.window_size >= self.chunk_size and self.window_size % self.chunk_size == 0
            # chunk_size overrides the number of landmarks
            self.num_chunks = None

        self.use_t5_rpe = (attn_args.use_t5_rpe) if attn_args.window_size > 0 else False

        if self.use_t5_rpe:
            self.rel_pos_bias = T5RelativePositionBias(
                self.scaling, 
                causal = self.causal, 
                num_buckets=max(min(int((self.window_size + self.ext_size) / 2), 64), 16), 
                max_distance=attn_args.window_size + self.ext_size
            )
        else:
            self.rel_pos_bias = None

        self.adaptive_proj = attn_args.adaptive_proj
        if self.adaptive_proj in ['qk']:
            self.adaptive_mu_q = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
        elif self.adaptive_proj in ['no-ln']:
            self.adaptive_mu_q = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
            )
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
            )    

    def _process_input(self, x, key_padding_mask):
        # this function re-implements the parent method.
        B, N, C = x.shape
        if self.window_size > 0:
            if key_padding_mask is None:
                x, key_padding_mask = pad_to_multiple(x, self.window_size, dim=-2, create_mask=True)
            else:
                x = pad_to_multiple(x, self.window_size, dim=-2)
                key_padding_mask = pad_to_multiple(key_padding_mask, self.window_size, dim=-1, value=True)
            N = x.shape[-2]
        return x, key_padding_mask


    def window_partition(self, x, shape, ext_window_size, pad_val=0, window_size=None):
        window_size = default(window_size, self.window_size)
        return causal_window_1d_partition(
            x, 
            window_size=window_size, 
            ext_window_size=ext_window_size, 
            pad_val=pad_val
            )
    
    def window_merge(self, x, shape, window_size=None):
        window_size = default(window_size, self.window_size)
        return window_1d_merge(x)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        # static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        # before_softmax: bool = False,
        # need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        mask_val = -5e4
        query = query.transpose(0, 1)
        
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [bsz, tgt_len, embed_dim]
        if key is not None:
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            key_bsz, src_len , _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert bsz, src_len == value.shape[:2]

        if incremental_state is None:
            # pad the whole seq only when incremental_state is None.
            B, tgt_len, C = query.shape
            query, key_padding_mask = self._process_input(query, key_padding_mask)
            B, N, C = query.shape
        seq_shape = (N,)
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q = (
            q.contiguous()
            .view(bsz, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        # Training & evaluation only. No incremental state is used.
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(B, N, dtype=k.dtype, device=k.device)
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]
    
        w_q = self.window_partition(q, seq_shape, ext_window_size=0) # [b, h, w, i, d]
        w_k = self.window_partition(k, seq_shape, ext_window_size=self.ext_size)
        w_v = self.window_partition(v, seq_shape, ext_window_size=self.ext_size) # [b, h, w, j, d]

        if self.chunk_size is not None:
            rf_chunk_size = self.chunk_size
        else:
            rf_chunk_size = int(N // self.num_chunks)
        if rf_chunk_size >= N:
            rf_w_q = q
            rf_w_k = k
            rf_w_v = v
        else:
            # [b, h, c, j, d]
            rf_w_q = self.window_partition(q, seq_shape, window_size=rf_chunk_size, ext_window_size=0)
            # [b, h, c, j, d]
            rf_w_k = self.window_partition(k, seq_shape, window_size=rf_chunk_size, ext_window_size=0)
            # [b, h, c, j, d]
            rf_w_v = self.window_partition(v, seq_shape, window_size=rf_chunk_size, ext_window_size=0)
            # compute local attention
            # [b, 1, c, j, 1]
            rf_w_mask = self.window_partition(
                key_padding_mask, 
                seq_shape, 
                window_size=rf_chunk_size,
                ext_window_size=0,
                pad_val=1
                ).to(torch.bool)
            # print(rf_w_mask)
            rf_w_q = rf_w_q.masked_fill(rf_w_mask, 0.)
            rf_w_k = rf_w_k.masked_fill(rf_w_mask, 0.)
            rf_w_v = rf_w_v.masked_fill(rf_w_mask, 0.)

            rf_q_bar = self.adaptive_mu_q(rf_w_q.mean(dim=-2))
            rf_k_bar = self.adaptive_mu_k(rf_w_k.mean(dim=-2))
            # [b, h, c, d]
            mu = rf_q_bar + rf_k_bar
            
            ######################## Sampling from proposal ###############################
            if self.training:
                weights = mu + torch.randn_like(mu)
            else:
                weights = mu    
            # [b, h, c, j, d], [b, h, c, 1, d] -> [b, h, c, j]
            log_proj_w_k = prm_projection(rf_w_k, weights.unsqueeze(-2), normalize=False).squeeze(-2)
            log_proj_w_k = log_proj_w_k.masked_fill(rf_w_mask.squeeze(-1), mask_val)

            # [b, h, c, j] [b, h, c, j, d] -> [b, h, c, d]
            beta = torch.einsum('...cj,...cjd->...cd', torch.softmax(log_proj_w_k, dim=-1), rf_w_v)
        
            # compute approx. expectation of CVs.
            # [b, h, c, d]
            approx_expected_cv = torch.einsum('...wid,...cd->...wic', w_q, self.scaling * rf_k_bar)
            if self.causal:
                # [b, h, j, c, c]
                b, h, j, c = q.shape[0], q.shape[1], rf_w_k.shape[-2], rf_w_k.shape[-3]
                if self.adaptive_proj in ['no-ln', 'qk']:
                    causal_mask = torch.ones(b, h, j, c, c, dtype=q.dtype, device=q.device).triu(0).transpose(-2, -3) # [b, h, c, j, c]
                    # NOTE: .triu(0) is used to remove the context of the current chunk from localized RFA.
                    # since we compute `rf_q_bar` for each chunk for random features, 
                    # it requires the future information if we compute it on the current chunk.
                    # however, note that the current chunk's information is still retained through
                    # the local attention module.
                else:
                    raise NotImplementedError("Other adaptive projection methods are not implemented yet.")
                causal_mask = self.window_merge(causal_mask, seq_shape) # [b, h, n, c]
                causal_mask = self.window_partition(causal_mask, seq_shape, ext_window_size=0).to(torch.bool) # [b, h, w, i, c]
                approx_expected_cv = approx_expected_cv.masked_fill(causal_mask, mask_val)

            # compute local attention
            mask_q = self.window_partition(
                key_padding_mask, 
                seq_shape, 
                ext_window_size=0,
                pad_val=1
                ).to(torch.bool) # [b, 1, w, i, 1]
            mask_k = self.window_partition(
                key_padding_mask, 
                seq_shape, 
                ext_window_size=self.ext_size,
                pad_val=1
                ).to(torch.bool).transpose(-1, -2) # [b, 1, w, 1, j] 
            local_dots_mask = torch.logical_or(mask_q, mask_k)
            log_qk_local_dot = torch.einsum('bhwie,bhwje->bhwij', w_q, w_k) * self.scaling # [b, h, w, i, j]
            # if self.use_headed_t5_rpe:
                # here the t5-rpe-bias has already been scaled by \sqrt{d}
                # log_qk_local_dot = log_qk_local_dot + self.headed_rel_pos_bias(log_qk_local_dot)
            if self.use_t5_rpe:
                # here the t5-rpe-bias has already been scaled by \sqrt{d}
                log_qk_local_dot = log_qk_local_dot + self.rel_pos_bias(log_qk_local_dot)

            log_qk_local_dot = log_qk_local_dot.masked_fill(local_dots_mask, mask_val)

            if self.causal:
                # e.g., if window_size = 3 and ext_size = 3, then it creates a causal_mask as follows:
                # [0 0 0 0 1 1]
                # [0 0 0 0 0 1]
                # [0 0 0 0 0 0]
                causal_mask = torch.ones_like(log_qk_local_dot).triu(1 + self.ext_size).to(torch.bool)
                log_qk_local_dot = log_qk_local_dot.masked_fill(causal_mask, mask_val)

            local_len = log_qk_local_dot.shape[-1]
            num_rfa_chunks = approx_expected_cv.shape[-1]

            # compute attention weights along with normalizing constant.
            attn = torch.softmax(torch.cat([log_qk_local_dot, approx_expected_cv], dim=-1), dim=-1)
            attn = self.dropout_module(attn)
            local_attn, ra_attn = torch.split(attn, [local_len, num_rfa_chunks], dim=-1)
            output_local = torch.einsum('bhwij,bhwjd->bhwid', local_attn, w_v)
            output_snis = torch.einsum('bhwic,bhcd->bhwid', ra_attn, beta) 
            ######################## Combine them together ############################
            output = self.window_merge(output_snis + output_local, seq_shape) # [b, h, n, d]
            x = output.permute(0, 2, 1, 3).reshape((B,) + tuple(seq_shape) + (C,))
            x = self.out_proj(x)
            if tgt_len is not None and tgt_len != N:
                x = x[..., :tgt_len, :]
            return x.transpose(0, 1).contiguous(), None


    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights
