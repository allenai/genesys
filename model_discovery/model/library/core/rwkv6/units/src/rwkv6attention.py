# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #

from einops import rearrange

from transformers.activations import ACT2FN
from typing import Optional




class RWKV6Attention(GAUBase):

    def __init__(
        self,
        embed_dim: int, 
        block_loc: tuple, 
        kwarg_all: dict,
        num_heads: int = 4,
        gate_fn: str = 'swish',
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
        **kwargs
    ):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

        self.hidden_size = embed_dim
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim

        self.key_dim = embed_dim // 2
        self.value_dim = embed_dim

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        kwarg_all['output_dim']=proj_low_rank_dim * 5
        self.x_proj = nn.Sequential(
            LerpLinear(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all),
            nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 5, embed_dim, bias=False, device=device, dtype=dtype)
        )
        self.x_bias = nn.Parameter(torch.zeros(5, embed_dim, device=device, dtype=dtype))

        kwarg_all['output_dim']=self.key_dim
        self.r_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        kwarg_all['low_rank_dim']=gate_low_rank_dim
        self.w_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        kwarg_all.pop('low_rank_dim')
        self.k_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        kwarg_all['output_dim']=self.value_dim
        self.v_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        kwarg_all['low_rank_dim']=gate_low_rank_dim
        self.g_proj = DDLerpLinear(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_qk_dim, device=device, dtype=dtype))

        # self.g_norm = nn.GroupNorm(self.num_heads, self.value_dim, affine=elementwise_affine, eps=norm_eps, device=device, dtype=dtype) # buggy now
        self.g_norm = nn.LayerNorm(self.value_dim, elementwise_affine=elementwise_affine, eps=norm_eps, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.value_dim, embed_dim, bias=False, device=device, dtype=dtype)
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

    def naive_recurrent_rwkv6(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        scale: Optional[float] = None,
    ):
        orig_dtype = q.dtype
        B, H, T, K, V = *q.shape, v.shape[-1]
        q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        o = torch.zeros_like(v)

        if scale is None:
            scale = K ** -0.5

        for i in range(T):
            q_i = q[:, :, i, :] * scale
            k_i = k[:, :, i]
            v_i = v[:, :, i, :]
            w_i = w[:, :, i].exp()
            kv_i = k_i[..., None] * v_i[..., None, :]
            o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None]
            o[:, :, i] = o_i.sum(-2)
            h = h * w_i[..., None] + kv_i
        return o.to(orig_dtype)

    def _forward(self,X: torch.Tensor):
        batch_size, seq_len, hidden_size = X.shape
        # launching the triton kernel for just one token will actually be slower
        last_state = None

        if X.shape[1] == 1 and last_state is not None:
            shifted = last_state[0].unsqueeze(1)
        else:
            shifted = self.time_shift(X)
            if last_state is not None:
                shifted[:, 0] = last_state[0]

        delta = shifted - X
        x = self.x_proj[0](X, **{'delta':delta})[1]['o'].view(batch_size, seq_len, -1, self.proj_low_rank_dim)
        x = torch.einsum('b l n r, h n r-> b l n h', self.x_proj[1](x), self.x_proj[2].weight.view(hidden_size, 5, -1))

        r, w, k, v, g = x.add_(self.x_bias).unbind(-2)
        r = self.r_proj(X, **{'mu': r, 'delta':delta})[1]['o']
        w = self.w_proj(X, **{'mu': w, 'delta':delta})[1]['o']
        k = self.k_proj(X, **{'mu': k, 'delta':delta})[1]['o']
        v = self.v_proj(X, **{'mu': v, 'delta':delta})[1]['o']
        g = self.g_proj(X, **{'mu': g, 'delta':delta})[1]['o']

        r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (r, w, k, v))
        w = -torch.exp(w)
        u = self.bonus

        o = self.naive_recurrent_rwkv6(r, k, v, w, u, scale=1.0)

        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.g_norm(o)
        o = o * self.gate_fn(g)
        o = self.o_proj(o)

        return o


@gau_test
def test_rwkv6attention(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    rwkv6attention = RWKV6Attention(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    y,_=rwkv6attention(x)
    assert y.shape==(1,100,128)


CHILDREN_DECLARATIONS = [
    UnitDecl(
        unitname="LerpLinear",
        requirements="",
        inputs=['X','delta'],
        outputs=['Y'],
    ),
    UnitDecl(
        unitname="DDLerpLinear",
        requirements="",
        inputs=['X','mu','delta'],
        outputs=['Y'],
    ),
]

SPEC = {
    "unitname": "RWKV6Attention",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
RWKV6Attention
''',
}

ARGS = {
    'num_heads': 4,
    'gate_fn': 'swish',
    'proj_low_rank_dim': 32,
    'gate_low_rank_dim': 64,
    'elementwise_affine': True,
}

CHILDREN = ['LerpLinear','DDLerpLinear']
DESC='''
''' 
