# -*- coding: utf-8 -*-

# "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence"[https://arxiv.org/abs/2404.05892]

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from transformers.activations import ACT2FN


# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class RWKV6Config(PretrainedConfig):

    model_type = 'rwkv6'

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 4,
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        hidden_act: str = "sqrelu",
        norm_eps: float = 1e-5,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps

        super().__init__(
            **kwargs,
        )


def pad_seq(X, chunk_size):
    pad_len = (chunk_size - X.shape[1] % chunk_size) % chunk_size
    if pad_len > 0:
        padding = torch.zeros(X.shape[0], pad_len, *X.shape[2:], dtype=X.dtype, device=X.device)
        X = torch.cat([X, padding], dim=1)
    return X

def naive_chunk_rwkv6(
    q,
    k,
    v,
    w,
    u,
    chunk_size=32,
):
    assert q.shape[-2] % chunk_size == 0
    orig_dtype = q.dtype
    num_chunk = q.shape[-2] // chunk_size
    u = u.unsqueeze(0)

    q, k, v, w = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size).float(), (q, k, v, w))

    w_cumsum = w.cumsum(-2)

    kw = k * (w_cumsum[..., -1, None, :] - w_cumsum).exp()
    wkv = kw.transpose(-1, -2) @ v

    wkv_new = torch.zeros_like(wkv)

    for i in range(num_chunk - 1):
        wkv_new[:, :, i+1] = (wkv_new[:, :, i] * w_cumsum[:, :, i, -1, :, None].exp()) + wkv[:, :, i]

    o_inter = torch.einsum('b h n d p, b h n c d -> b h n c p', wkv_new, (q * (w_cumsum - w).exp()))

    o_intra = torch.zeros_like(o_inter)
    for i in range(chunk_size):
        attn = (q[:, :, :, i, None] * k * (w_cumsum[:, :, :, i, None] - w[:, :, :, i, None] - w_cumsum).exp()).sum(-1)
        mask = (torch.arange(0, chunk_size) < i).to(attn.device)
        attn.masked_fill_(~mask, 0)
        intra_inter_o = (attn.unsqueeze(-1) * v).sum(-2)
        intra_intra_o = (q[:, :, :, i] * u.unsqueeze(2) * k[:, :, :, i]).sum(-1).unsqueeze(-1) * v[:, :, :, i]
        o_intra[:, :, :, i] = intra_inter_o + intra_intra_o
    o = o_inter + o_intra
    return rearrange(o, 'b h n c d -> b h (n c) d').to(orig_dtype)

class RWKV6Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        num_heads: int = 4,
        gate_fn: str = 'swish',
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        chunk_size: int = 32,
        **kwargs
    ) -> RWKV6Attention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.chunk_size = chunk_size

        self.key_dim = int(hidden_size * 0.5)
        self.value_dim = int(hidden_size * 1.0)
        self.layer_idx = layer_idx

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_proj = nn.Sequential(
            LerpLinear(hidden_size, proj_low_rank_dim * 5),
            nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 5, hidden_size, bias=False)
        )
        self.x_bias = nn.Parameter(torch.zeros(5, hidden_size))

        self.r_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.w_proj = DDLerpLinear(hidden_size, self.key_dim, low_rank_dim=gate_low_rank_dim)
        self.k_proj = DDLerpLinear(hidden_size, self.key_dim)
        self.v_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.g_proj = DDLerpLinear(hidden_size, self.value_dim)
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_qk_dim))

        # TODO: fuse GroupNorm and output gate
        self.g_norm = nn.GroupNorm(self.num_heads, self.value_dim, affine=elementwise_affine, bias=True, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_fn = ACT2FN[gate_fn]

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module, gain=2 ** -2.5)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        _seqlen = hidden_states.shape[1]
        hidden_states = pad_seq(hidden_states, self.chunk_size)

        batch_size, seq_len, hidden_size = hidden_states.shape
        # launching the triton kernel for just one token will actually be slower
        last_state = None

        if attention_mask is not None:
            hidden_states = hidden_states.mul_(attention_mask.unsqueeze(-1))
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state[0].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state[0]

        delta = shifted - hidden_states
        x = self.x_proj[0](hidden_states, delta).view(batch_size, seq_len, -1, self.proj_low_rank_dim)
        x = torch.einsum('b l n r, h n r-> b l n h', self.x_proj[1](x), self.x_proj[2].weight.view(hidden_size, 5, -1))

        r, w, k, v, g = x.add_(self.x_bias).unbind(-2)
        r = self.r_proj(hidden_states, r, delta)
        w = self.w_proj(hidden_states, w, delta)
        k = self.k_proj(hidden_states, k, delta)
        v = self.v_proj(hidden_states, v, delta)
        g = self.g_proj(hidden_states, g, delta)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))
        r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (r, w, k, v))
        w = -torch.exp(w)
        u = self.bonus

        o = naive_chunk_rwkv6(r, k, v, w, u, self.chunk_size)

        o = self.g_norm(rearrange(o, 'b h l d -> b l (h d)')) * self.gate_fn(g)
        o = self.o_proj(o)

        o = o[:, :_seqlen]
        return o

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = [param.new_zeros(batch_size, self.hidden_size),
                 param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim)]
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        return state_size


class LoRA(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: Optional[bool] = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            nn.Tanh(),
            nn.Linear(low_rank_dim, output_dim, bias=bias)
        )

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"input_dim={self.input_dim}, low_rank_dim={self.low_rank_dim}, output_dim={self.output_dim}"
        if not self.bias:
            s += f", bias={self.bias}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


class LerpLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: Optional[int] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)
        self.mu = nn.Parameter(torch.zeros(input_dim))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}"
        if self.low_rank_dim is not None:
            s += f", low_rank_dim={self.low_rank_dim}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * self.mu)


class DDLerpLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: Optional[int] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}"
        if self.low_rank_dim is not None:
            s += f", low_rank_dim={self.low_rank_dim}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor, mu: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * mu)
    



class RWKV6FeedForward(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_act: str = 'sqrelu',
    ) -> RWKV6FeedForward:
        super().__init__()

        self.hidden_size = hidden_size
        hidden_ratio = 3.5
        intermediate_size = int(hidden_size * hidden_ratio)
        intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = LerpLinear(hidden_size, intermediate_size)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.receptance = LerpLinear(hidden_size, hidden_size)
        self.act_fn = ACT2FN[hidden_act]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            x = x.mul_(attention_mask.unsqueeze(-1))
        shifted = self.time_shift(x)
        delta = shifted - x
        key = self.act_fn(self.key(x, delta))
        value = self.value(key)
        receptance = self.receptance(x, delta)

        return receptance.sigmoid() * value


class RWKV6Block(nn.Module):
    def __init__(self, config: RWKV6Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.config = config
        self.layer_idx = layer_idx

        if layer_idx == 0:
            self.pre_norm = nn.LayerNorm(config.hidden_size, bias=True, eps=config.norm_eps)
        self.attn_norm = nn.LayerNorm(config.hidden_size, bias=True, eps=config.norm_eps)
        self.attn = RWKV6Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            proj_low_rank_dim=config.proj_low_rank_dim,
            gate_low_rank_dim=config.gate_low_rank_dim,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx
        )
        self.ffn_norm = nn.LayerNorm(hidden_size=config.hidden_size, bias=True, eps=config.norm_eps)
        self.ffn = RWKV6FeedForward(
            hidden_size=config.hidden_size,
            hidden_act=config.hidden_act,
            layer_idx=layer_idx
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states
        hidden_states = self.attn_norm(residual)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states, residual = self.ffn_norm(hidden_states, residual, True)
        hidden_states = self.ffn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        return hidden_states
