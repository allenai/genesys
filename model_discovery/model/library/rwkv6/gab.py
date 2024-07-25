import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase
from einops import rearrange
from transformers.activations import ACT2FN
from typing import TYPE_CHECKING, Optional, Tuple


def naive_recurrent_rwkv6(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    w: torch.Tensor, u: torch.Tensor, scale: Optional[float]=None):
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


class RWKV6Attention(nn.Module):

    def __init__(self, hidden_size: int=1024, num_heads: int=4, gate_fn:
        str='swish', proj_low_rank_dim: int=32, gate_low_rank_dim: int=64,
        elementwise_affine: Optional[bool]=True, norm_eps: float=1e-05,
        device=None, dtype=None, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.key_dim = hidden_size // 2
        self.value_dim = hidden_size
        assert self.key_dim % num_heads == 0, f'key dim must be divisible by num_heads of {num_heads}'
        assert self.value_dim % num_heads == 0, f'value dim must be divisible by num_heads of {num_heads}'
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_proj = nn.Sequential(LerpLinear(hidden_size, 
            proj_low_rank_dim * 5, device=device, dtype=dtype), nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 5, hidden_size, bias=False,
            device=device, dtype=dtype))
        self.x_bias = nn.Parameter(torch.zeros(5, hidden_size, device=
            device, dtype=dtype))
        self.r_proj = DDLerpLinear(hidden_size, self.key_dim, device=device,
            dtype=dtype)
        self.w_proj = DDLerpLinear(hidden_size, self.key_dim, low_rank_dim=
            gate_low_rank_dim, device=device, dtype=dtype)
        self.k_proj = DDLerpLinear(hidden_size, self.key_dim, device=device,
            dtype=dtype)
        self.v_proj = DDLerpLinear(hidden_size, self.value_dim, device=
            device, dtype=dtype)
        self.g_proj = DDLerpLinear(hidden_size, self.value_dim,
            low_rank_dim=gate_low_rank_dim, device=device, dtype=dtype)
        self.bonus = nn.Parameter(torch.zeros(num_heads, self.head_qk_dim,
            device=device, dtype=dtype))
        self.g_norm = nn.LayerNorm(self.value_dim, elementwise_affine=
            elementwise_affine, eps=norm_eps, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False,
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

    def forward(self, hidden_states: torch.Tensor, **kwargs) ->Tuple[torch.
        Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        last_state = None
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state[0].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state[0]
        delta = shifted - hidden_states
        x = self.x_proj[0](hidden_states, delta).view(batch_size, seq_len, 
            -1, self.proj_low_rank_dim)
        x = torch.einsum('b l n r, h n r-> b l n h', self.x_proj[1](x),
            self.x_proj[2].weight.view(hidden_size, 5, -1))
        r, w, k, v, g = x.add_(self.x_bias).unbind(-2)
        r = self.r_proj(hidden_states, r, delta)
        w = self.w_proj(hidden_states, w, delta)
        k = self.k_proj(hidden_states, k, delta)
        v = self.v_proj(hidden_states, v, delta)
        g = self.g_proj(hidden_states, g, delta)
        r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=
            self.num_heads), (r, w, k, v))
        w = -torch.exp(w)
        u = self.bonus
        o = naive_recurrent_rwkv6(r, k, v, w, u, scale=1.0)
        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.g_norm(o)
        o = o * self.gate_fn(g)
        o = self.o_proj(o)
        return o

    def init_state(self, batch_size: int) ->Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = [param.new_zeros(batch_size, self.hidden_size), param.
            new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.
            head_v_dim)]
        return state

    def state_size(self, **kwargs) ->int:
        state_size = self.key_dim * self.head_v_dim
        return state_size


class LoRA(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, low_rank_dim: int,
        bias: Optional[bool]=True, device=None, dtype=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias
        self.lora = nn.Sequential(nn.Linear(input_dim, low_rank_dim, bias=
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

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.lora(x)


class LerpLinear(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, low_rank_dim:
        Optional[int]=None, device=None, dtype=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False,
                device=device, dtype=dtype)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim, device=
                device, dtype=dtype)
        self.mu = nn.Parameter(torch.zeros(input_dim, device=device, dtype=
            dtype))

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}({self.input_dim}, {self.output_dim}'
        if self.low_rank_dim is not None:
            s += f', low_rank_dim={self.low_rank_dim}'
        s += ')'
        return s

    def forward(self, x: torch.Tensor, delta: Optional[torch.Tensor]=None
        ) ->torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * self.mu)


class DDLerpLinear(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, low_rank_dim:
        Optional[int]=None, device=None, dtype=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False,
                device=device, dtype=dtype)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim, device=
                device, dtype=dtype)

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
        return self.linear(x + delta * mu)


class RWKV6FeedForward(nn.Module):

    def __init__(self, hidden_size: int, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        hidden_ratio = 3.5
        intermediate_size = int(hidden_size * hidden_ratio)
        intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = LerpLinear(hidden_size, intermediate_size, device=device,
            dtype=dtype)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False,
            device=device, dtype=dtype)
        self.receptance = LerpLinear(hidden_size, hidden_size, device=
            device, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        shifted = self.time_shift(x)
        delta = shifted - x
        _key = self.key(x, delta)
        r = self.relu(_key)
        key = r * r
        value = self.value(key)
        receptance = self.receptance(x, delta)
        return receptance.sigmoid() * value


class GAB(GABBase):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """

    def __init__(self, embed_dim: int, device=None, dtype=None, num_heads:
        int=4, proj_low_rank_dim: int=32, gate_low_rank_dim: int=64,
        norm_eps: float=1e-05, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim)
        self.hidden_size = embed_dim
        self.attn_norm = nn.LayerNorm(self.hidden_size, bias=True, eps=
            norm_eps, **factory_kwargs)
        self.attn = RWKV6Attention(hidden_size=self.hidden_size, num_heads=
            num_heads, proj_low_rank_dim=proj_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim, norm_eps=norm_eps, **
            factory_kwargs)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, bias=True, eps=
            norm_eps, **factory_kwargs)
        self.ffn = RWKV6FeedForward(hidden_size=self.hidden_size, **
            factory_kwargs)

    def _forward(self, X, **kwargs):
        hidden_states = self.attn_norm(X)
        X = self.attn(hidden_states) + X
        hidden_states = self.ffn_norm(X)
        X = self.ffn(hidden_states) + X
        return X


""" The dictionary of hyperparameters for constructing a GAB layer
    embed_dim, device, dtype should NOT be included in gab_config
"""
gab_config = {'num_heads': 4, 'proj_low_rank_dim': 32, 'gate_low_rank_dim':
    64, 'norm_eps': 1e-05}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)