import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = RWKV6(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


import torch.nn.functional as F
from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl


class RWKV6(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        norm_eps: float=1e-05, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.hidden_size = embed_dim
        self.attn_norm = nn.LayerNorm(self.hidden_size, bias=True, eps=
            norm_eps, **self.factory_kwargs)
        self.attn = RWKV6Attention(embed_dim=self.embed_dim, block_loc=self
            .block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, bias=True, eps=
            norm_eps, **self.factory_kwargs)
        self.ffn = RWKV6FeedForward(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)

    def _forward(self, X, **Z):
        X1, _ = self.attn(self.attn_norm(X), **Z)
        X = X1 + X
        X2, _ = self.ffn(self.ffn_norm(X), **Z)
        X = X2 + X
        return X


import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from typing import Optional


class RWKV6Attention(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        num_heads: int=4, gate_fn: str='swish', proj_low_rank_dim: int=32,
        gate_low_rank_dim: int=64, elementwise_affine: Optional[bool]=True,
        norm_eps: float=1e-05, chunk_size: int=32, device=None, dtype=None,
        **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.hidden_size = embed_dim
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.chunk_size = chunk_size
        self.key_dim = embed_dim // 2
        self.value_dim = embed_dim
        assert self.key_dim % num_heads == 0, f'key dim must be divisible by num_heads of {num_heads}'
        assert self.value_dim % num_heads == 0, f'value dim must be divisible by num_heads of {num_heads}'
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        kwarg_all['output_dim'] = proj_low_rank_dim * 5
        self.x_proj = nn.Sequential(LerpLinear(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all), nn.Tanh(), nn.Linear(
            proj_low_rank_dim * 5, embed_dim, bias=False, device=device,
            dtype=dtype))
        self.x_bias = nn.Parameter(torch.zeros(5, embed_dim, device=device,
            dtype=dtype))
        kwarg_all['output_dim'] = self.key_dim
        self.r_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self
            .block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        kwarg_all['low_rank_dim'] = gate_low_rank_dim
        self.w_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self
            .block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        kwarg_all.pop('low_rank_dim')
        self.k_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self
            .block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        kwarg_all['output_dim'] = self.value_dim
        self.v_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self
            .block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        kwarg_all['low_rank_dim'] = gate_low_rank_dim
        self.g_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self
            .block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_qk_dim,
            device=device, dtype=dtype))
        self.g_norm = nn.LayerNorm(self.value_dim, elementwise_affine=
            elementwise_affine, eps=norm_eps, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.value_dim, embed_dim, bias=False,
            device=device, dtype=dtype)
        self.gate_fn = ACT2FN[gate_fn]
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, '_is_hf_initialized', False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module, gain=2 ** -2.5)
        module._is_hf_initialized = True

    def naive_chunk_rwkv6(self, q: torch.Tensor, k: torch.Tensor, v: torch.
        Tensor, w: torch.Tensor, u: torch.Tensor, chunk_size: int=32):
        assert q.shape[-2] % chunk_size == 0
        orig_dtype = q.dtype
        num_chunk = q.shape[-2] // chunk_size
        u = u.unsqueeze(0)
        q, k, v, w = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d',
            c=chunk_size).float(), (q, k, v, w))
        w_cumsum = w.cumsum(-2)
        kw = k * (w_cumsum[..., -1, None, :] - w_cumsum).exp()
        wkv = kw.transpose(-1, -2) @ v
        wkv_new = torch.zeros_like(wkv)
        for i in range(num_chunk - 1):
            wkv_new[:, :, i + 1] = wkv_new[:, :, i].clone() * w_cumsum[:, :,
                i, -1, :, None].exp() + wkv[:, :, i]
        o_inter = torch.einsum('b h n d p, b h n c d -> b h n c p', wkv_new,
            q * (w_cumsum - w).exp())
        o_intra = torch.zeros_like(o_inter)
        for i in range(chunk_size):
            attn = (q[:, :, :, i, None] * k * (w_cumsum[:, :, :, i, None] -
                w[:, :, :, i, None] - w_cumsum).exp()).sum(-1)
            mask = (torch.arange(0, chunk_size) < i).to(attn.device)
            attn.masked_fill_(~mask, 0)
            intra_inter_o = (attn.unsqueeze(-1) * v).sum(-2)
            intra_intra_o = (q[:, :, :, i] * u.unsqueeze(2) * k[:, :, :, i]
                ).sum(-1).unsqueeze(-1) * v[:, :, :, i]
            o_intra[:, :, :, i] = intra_inter_o + intra_intra_o
        o = o_inter + o_intra
        return rearrange(o, 'b h n c d -> b h (n c) d').to(orig_dtype)

    def pad_input(self, X):
        _seq_len = X.shape[-2]
        pad_len = (X.shape[-2] + self.chunk_size - 1
            ) // self.chunk_size * self.chunk_size - X.shape[-2]
        return F.pad(X, (0, 0, 0, pad_len)), _seq_len

    def _forward(self, X: torch.Tensor):
        X, _seq_len = self.pad_input(X)
        batch_size, seq_len, hidden_size = X.shape
        last_state = None
        if X.shape[1] == 1 and last_state is not None:
            shifted = last_state[0].unsqueeze(1)
        else:
            shifted = self.time_shift(X)
            if last_state is not None:
                shifted[:, 0] = last_state[0]
        delta = shifted - X
        x = self.x_proj[0](X, **{'delta': delta})[1]['o'].view(batch_size,
            seq_len, -1, self.proj_low_rank_dim)
        x = torch.einsum('b l n r, h n r-> b l n h', self.x_proj[1](x),
            self.x_proj[2].weight.view(hidden_size, 5, -1))
        r, w, k, v, g = x.add_(self.x_bias).unbind(-2)
        r = self.r_proj(X, **{'mu': r, 'delta': delta})[1]['o']
        w = self.w_proj(X, **{'mu': w, 'delta': delta})[1]['o']
        k = self.k_proj(X, **{'mu': k, 'delta': delta})[1]['o']
        v = self.v_proj(X, **{'mu': v, 'delta': delta})[1]['o']
        g = self.g_proj(X, **{'mu': g, 'delta': delta})[1]['o']
        r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=
            self.num_heads), (r, w, k, v))
        w = -torch.exp(w)
        u = self.bonus
        o = self.naive_chunk_rwkv6(r, k, v, w, u, chunk_size=self.chunk_size)
        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.g_norm(o)
        o = o * self.gate_fn(g)
        o = self.o_proj(o)
        o = o[:, :_seq_len]
        return o


import torch.nn.functional as F
from typing import Optional


class LerpLinear(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        output_dim: int, low_rank_dim: Optional[int]=None, device=None,
        dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.input_dim = embed_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if self.low_rank_dim is None:
            self.linear = nn.Linear(embed_dim, output_dim, bias=False,
                device=device, dtype=dtype)
        else:
            kwarg_all['output_dim'] = output_dim
            kwarg_all['low_rank_dim'] = low_rank_dim
            self.linear = LoRA(embed_dim=self.embed_dim, block_loc=self.
                block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
                **self.kwarg_all)
        self.mu = nn.Parameter(torch.zeros(embed_dim, device=device, dtype=
            dtype))

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}({self.input_dim}, {self.output_dim}'
        if self.low_rank_dim is not None:
            s += f', low_rank_dim={self.low_rank_dim}'
        s += ')'
        return s

    def _forward(self, X: torch.Tensor, delta: Optional[torch.Tensor]=None
        ) ->torch.Tensor:
        if delta is None:
            shifted = self.time_shift(X)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - X
        if self.low_rank_dim is None:
            o = self.linear(X + delta * self.mu)
        else:
            o = self.linear(X + delta * self.mu)[1]['o']
        return X, {'o': o}


import torch.nn.functional as F
from typing import Optional


class LoRA(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        output_dim: int, low_rank_dim: int, bias: Optional[bool]=True,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.input_dim = embed_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias
        self.lora = nn.Sequential(nn.Linear(embed_dim, low_rank_dim, bias=
            False, device=device, dtype=dtype), nn.Tanh(), nn.Linear(
            low_rank_dim, output_dim, bias=bias, device=device, dtype=dtype))

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}('
        s += (
            f'input_dim={self.input_dim}, low_rank_dim={self.low_rank_dim}, output_dim={self.output_dim}'
            )
        if not self.bias:
            s += f', bias={self.bias}'
        s += ')'
        return s

    def _forward(self, X, **Z):
        return X, {'o': self.lora(X)}


import torch.nn.functional as F
from typing import Optional


class DDLerpLinear(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        output_dim: int, low_rank_dim: Optional[int]=None, device=None,
        dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.input_dim = embed_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(embed_dim, output_dim, bias=False,
                device=device, dtype=dtype)
        else:
            kwarg_all['output_dim'] = output_dim
            kwarg_all['low_rank_dim'] = low_rank_dim
            self.linear = LoRA(embed_dim=self.embed_dim, block_loc=self.
                block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
                **self.kwarg_all)

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}({self.input_dim}, {self.output_dim}'
        if self.low_rank_dim is not None:
            s += f', low_rank_dim={self.low_rank_dim}'
        s += ')'
        return s

    def forward(self, x: torch.Tensor, mu: torch.Tensor, delta: Optional[
        torch.Tensor]=None) ->torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        if self.low_rank_dim is None:
            o = self.linear(x + delta * mu)
        else:
            o = self.linear(x + delta * mu)[1]['o']
        return x, {'o': o}


import torch.nn.functional as F


class RWKV6FeedForward(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.hidden_size = embed_dim
        hidden_ratio = 3.5
        intermediate_size = int(embed_dim * hidden_ratio)
        intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        kwarg_all['output_dim'] = intermediate_size
        self.key = LerpLinear(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.value = nn.Linear(intermediate_size, embed_dim, bias=False,
            device=device, dtype=dtype)
        kwarg_all['output_dim'] = embed_dim
        self.receptance = LerpLinear(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)
        self.relu = nn.ReLU()

    def _forward(self, X, **Z):
        shifted = self.time_shift(X)
        delta = shifted - X
        _key = self.key(X, **{'delta': delta})[1]['o']
        r = self.relu(_key)
        key = r * r
        value = self.value(key)
        receptance = self.receptance(X, **{'delta': delta})[1]['o']
        return receptance.sigmoid() * value


gab_config = {'norm_eps': 1e-05, 'num_heads': 4, 'gate_fn': 'swish',
    'proj_low_rank_dim': 32, 'gate_low_rank_dim': 64, 'elementwise_affine':
    True, 'chunk_size': 32}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from model_discovery.model.block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)