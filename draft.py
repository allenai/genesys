import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase
from transformers.activations import ACT2FN
from typing import Dict, List, Optional, Tuple, Union
from torchtune.modules import RMSNorm


class RetNetRelPos(nn.Module):

    def __init__(self, embed_dim, chunk_size, retention_heads, device=None,
        dtype=None):
        super().__init__()
        num_heads = retention_heads
        angle = 1.0 / 10000 ** torch.linspace(0, 1, embed_dim // num_heads //
            2, device=device, dtype=dtype)
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads, dtype=
            dtype, device=device)))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        self.recurrent_chunk_size = chunk_size
        self.device = device
        self.dtype = dtype

    def forward(self, slen, forward_impl='parallel', recurrent_chunk_size=
        None, retention_mask=None, get_decay_scale=True):
        if forward_impl == 'recurrent':
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = (sin, cos), self.decay.view(1, -1, 1, 1).exp()
        elif forward_impl == 'chunkwise':
            if recurrent_chunk_size is None:
                recurrent_chunk_size = self.recurrent_chunk_size
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            block_index = torch.arange(recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(recurrent_chunk_size,
                recurrent_chunk_size)).to(self.decay)
            mask = torch.masked_fill(block_index[:, None] - block_index[
                None, :], ~mask.bool(), float('inf'))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask.unsqueeze(0)
            value_inner_decay = mask[:, :, -1] / mask[:, :, -1].sum(dim=-1,
                keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale
            cross_decay = torch.exp(self.decay * recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (
                block_index + 1))
            cross_decay = cross_decay[None, :, None, None]
            query_inner_decay = query_inner_decay[None, :, :, None] / (scale /
                mask[:, :, -1].sum(dim=-1)[:, :, None, None])
            if get_decay_scale:
                decay_scale = self.compute_decay_scale(slen, retention_mask
                    ).to(self.device, self.dtype)
            else:
                decay_scale = None
            retention_rel_pos = (sin.to(self.device, self.dtype), cos.to(
                self.device, self.dtype)), (inner_mask.to(self.device, self
                .dtype), cross_decay.to(self.device, self.dtype),
                query_inner_decay.to(self.device, self.dtype),
                value_inner_decay.to(self.device, self.dtype), decay_scale)
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen)).to(self.decay)
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask
                .bool(), float('inf'))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask.unsqueeze(0)
            if retention_mask is not None:
                mask = mask * retention_mask.float().view(-1, 1, 1, slen).to(
                    mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            mask = torch.nan_to_num(mask, nan=0.0)
            if get_decay_scale:
                decay_scale = self.compute_decay_scale(slen, retention_mask
                    ).to(self.device, self.dtype)
            else:
                decay_scale = None
            if retention_mask is not None:
                max_non_zero = torch.cumsum(retention_mask, dim=-1).max(dim=-1
                    ).indices
                intra_decay = mask[range(mask.shape[0]), :, max_non_zero]
            else:
                intra_decay = mask[:, :, -1]
            retention_rel_pos = (sin.to(self.device, self.dtype), cos.to(
                self.device, self.dtype)), (mask.to(self.device, self.dtype
                ), intra_decay.to(self.device, self.dtype), decay_scale)
        return retention_rel_pos

    def compute_decay_scale(self, slen, retention_mask=None):
        exponent = torch.arange(slen, device=self.decay.device).float()
        decay_scale = self.decay.exp().view(-1, 1) ** exponent.view(1, -1)
        if retention_mask is not None:
            seqlen = retention_mask.sum(dim=-1)
            bsz = seqlen.size(0)
            decay_scale = decay_scale.unsqueeze(0).repeat(bsz, 1, 1)
            for i, pos in enumerate(seqlen):
                decay_scale[i, :, pos.item():] = 0
        else:
            bsz = 1
        decay_scale = decay_scale.sum(-1).view(bsz, -1, 1, 1)
        return decay_scale


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return x * cos + rotate_every_two(x) * sin


def split_heads(tensors, bsz, seqlen, num_heads):
    assert isinstance(tensors, (tuple, list))
    return [x.view(bsz, seqlen, num_heads, -1).transpose(1, 2) for x in tensors
        ]


class MultiScaleRetention(nn.Module):

    def __init__(self, embed_dim, heads, use_bias=False, tensor_parallel=
        False, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = int(embed_dim * 1.5)
        self.num_heads = heads
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.gate_fn = ACT2FN['swish']
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=
            use_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=
            use_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.value_dim, bias=
            use_bias, device=device, dtype=dtype)
        self.g_proj = nn.Linear(self.embed_dim, self.value_dim, bias=
            use_bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(self.value_dim, self.embed_dim, bias=
            use_bias, device=device, dtype=dtype)
        self.group_norm = RMSNorm(self.head_dim).to(device=device, dtype=dtype)
        self.reset_parameters()
        if tensor_parallel:
            self.decay_proj = nn.Linear(self.num_heads, self.num_heads,
                bias=False, device=device, dtype=dtype)
        else:
            self.decay_proj = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -1)

    def parallel_retention(self, q, k, v, decay_mask):
        """
        q,  # bsz * num_head * len * qk_dim
        k,  # bsz * num_head * len * qk_dim
        v,  # bsz * num_head * len * v_dim
        decay_mask,  # (1 or bsz) * num_head * len * len
        """
        decay_mask, intra_decay, scale = decay_mask
        if self.decay_proj is not None:
            decay_mask = self.decay_proj(decay_mask.transpose(-1, -3)
                ).transpose(-3, -1)
        retention = q @ k.transpose(-1, -2)
        retention = retention * decay_mask
        retention = retention / retention.detach().abs().sum(dim=-1,
            keepdim=True).clamp(min=1, max=50000.0)
        output = retention @ v
        output = output.transpose(1, 2)
        if self.training:
            return output, None, retention
        if self.decay_proj is not None:
            intra_decay = self.decay_proj(intra_decay.transpose(-1, -2)
                ).transpose(-2, -1)
        current_kv = k.unsqueeze(-2) * v.unsqueeze(-1)
        intra_decay = intra_decay[:, :, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)
        cache = {'prev_key_value': current_kv, 'scale': scale}
        return output, cache, retention

    def recurrent_retention(self, q, k, v, decay, past_key_value=None,
        retention_mask=None):
        """
        q, k, v, # bsz * num_head * 1 * qkv_dim
        past_key_value:
            - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
            - "scale"  # (1 or bsz) * num_head * 1 * 1
        decay # (1 or bsz) * num_head * 1 * 1
        retention_mask # bsz * 1
        """
        if retention_mask is not None:
            retention_mask = retention_mask.float().view(-1, 1, 1, 1).to(decay)
        else:
            retention_mask = torch.ones(k.size(0), 1, 1, 1).to(decay).to(k.
                device, k.dtype)
        current_kv = k * v.transpose(-1, -2) * retention_mask
        if past_key_value is not None and 'prev_key_value' in past_key_value:
            prev_kv = past_key_value['prev_key_value']
            prev_scale = past_key_value['scale']
            scale = torch.where(retention_mask == 0, prev_scale, prev_scale *
                decay + 1)
            decay_amount = prev_scale.sqrt() * decay / scale.sqrt()
            decay_amount = torch.where(retention_mask == 0, 1, decay_amount)
            prev_kv = prev_kv * decay_amount
            current_kv = current_kv / scale.sqrt()
            current_kv = torch.nan_to_num(current_kv, nan=0.0)
            current_kv = prev_kv + current_kv
        else:
            scale = torch.ones_like(decay).to(k.device, k.dtype)
            scale = torch.where(retention_mask == 0, torch.zeros_like(decay
                ), scale)
        output = torch.sum(q * current_kv, dim=3).unsqueeze(1)
        cache = {'prev_key_value': current_kv, 'scale': scale}
        return output, cache

    def chunkwise_retention(self, q, k, v, decay_mask):
        """
        q, k, v,  # bsz * num_head * seqlen * qkv_dim
        past_key_value:
            - "prev_key_value"  # bsz * num_head * v_dim * qk_dim
            - "scale"  # (1 or bsz) * num_head * 1 * 1
        decay_mask,  # 1 * num_head * chunk_size * chunk_size
        cross_decay,  # 1 * num_head * 1 * 1
        inner_decay,  # 1 * num_head * chunk_size * 1
        """
        (decay_mask, cross_decay, query_inner_decay, value_inner_decay,
            decay_scale) = decay_mask
        bsz, _, tgt_len, _ = v.size()
        chunk_len = decay_mask.size(-1)
        assert tgt_len % chunk_len == 0
        num_chunks = tgt_len // chunk_len
        q = q.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim
            ).transpose(1, 2)
        k = k.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim
            ).transpose(1, 2)
        v = v.view(bsz, self.num_heads, num_chunks, chunk_len, self.head_dim
            ).transpose(1, 2)
        k_t = k.transpose(-1, -2)
        qk_mat = q @ k_t
        qk_mat = qk_mat * decay_mask.unsqueeze(1)
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min
            =1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(qk_mat, v)
        kv = k_t @ (v * value_inner_decay)
        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim
            ).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(
                dim=-1, keepdim=True).values.clamp(min=1)
        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)
        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale
        cross_output = q * query_inner_decay.unsqueeze(1) @ kv_recurrent
        output = (inner_output / align_inner_scale + cross_output /
            align_cross_scale)
        output = output.transpose(2, 3)
        cache = {'prev_key_value': kv_state.transpose(-2, -1), 'scale':
            decay_scale}
        return output, cache

    def forward(self, hidden_states: torch.Tensor, rel_pos: Tuple[Tuple[
        torch.Tensor]], retention_mask: Optional[torch.Tensor]=None,
        past_key_value: Optional[Tuple[torch.Tensor]]=None, forward_impl:
        str='parallel') ->Tuple[torch.FloatTensor, torch.FloatTensor,
        Optional[torch.FloatTensor]]:
        B, T, H = hidden_states.size()
        (sin, cos), decay_mask = rel_pos
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)
        q, k, v = split_heads((q, k, v), B, T, self.num_heads)
        k *= self.scaling
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)
        if forward_impl == 'parallel':
            retention_out, curr_kv, retention_weights = (self.
                parallel_retention(qr, kr, v, decay_mask))
        elif forward_impl == 'recurrent':
            retention_out, curr_kv = self.recurrent_retention(qr, kr, v,
                decay_mask, past_key_value=past_key_value, retention_mask=
                retention_mask)
        elif forward_impl == 'chunkwise':
            retention_out, curr_kv = self.chunkwise_retention(qr, kr, v,
                decay_mask)
        else:
            raise ValueError(f'forward_impl {forward_impl} not supported.')
        normed = self.group_norm(retention_out).reshape(B, T, self.value_dim)
        out = self.gate_fn(g) * normed
        out = self.out_proj(out)
        outputs = out, curr_kv
        return outputs


class GAB(GABBase):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """

    def __init__(self, embed_dim: int, device=None, dtype=None, heads=3,
        chunk_size=64, mode='chunkwise', **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim)
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.mode = mode
        self.heads = heads
        ffn_size = 4 * embed_dim
        self.retnet_rel_pos = RetNetRelPos(embed_dim, chunk_size=chunk_size,
            retention_heads=heads, device=device, dtype=dtype)
        self.retention = MultiScaleRetention(embed_dim, heads, device=
            device, dtype=dtype)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_size), nn.GELU(),
            nn.Linear(ffn_size, embed_dim)).to(device=device, dtype=dtype)
        self.layer_norm_1 = nn.LayerNorm(embed_dim).to(device=device, dtype
            =dtype)
        self.layer_norm_2 = nn.LayerNorm(embed_dim).to(device=device, dtype
            =dtype)

    def _forward(self, X, **kwargs):
        slen = X.shape[1]
        retention_rel_pos = self.retnet_rel_pos(slen, get_decay_scale=not
            self.training, forward_impl=self.mode)
        X = self.layer_norm_1(X)
        Y, _ = self.retention(X, retention_rel_pos, forward_impl=self.mode)
        Y = Y + X
        X = self.ffn(self.layer_norm_2(Y)) + Y
        return X


""" The dictionary of hyperparameters for constructing a GAB layer
    embed_dim, device, dtype should NOT be included in gab_config
"""
gab_config = {'heads': 8, 'chunk_size': 64, 'mode': 'recurrent'}
