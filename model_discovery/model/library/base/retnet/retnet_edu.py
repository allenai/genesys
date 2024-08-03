# gab.py

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #


# YOU CAN IMPORT MORE MODULES HERE #

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers.activations import ACT2FN
import torch.utils.checkpoint
from torchtune.modules import RotaryPositionalEmbeddings,RMSNorm

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #



def naive_retention(q, k, v): 
    orig_type = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    _, n_heads, seq_len, d_head = q.shape
    s = (1 - q.new_tensor(2., dtype=torch.float).pow(-5. - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
    n = q.new_tensor(range(seq_len), dtype=torch.float)
    n = torch.exp2((n.unsqueeze(-1) - n) * s.view(-1, 1, 1)) * n.unsqueeze(-1).ge(n)
    s = torch.einsum('bhqd,bhkd,hqk->bhqk', q * d_head ** -0.5, k, n.to(q.dtype))
    o = torch.einsum('bhqk,bhkd->bhqd', s, v)
    return o.to(orig_type)

class MultiScaleRetention(nn.Module):
    r"""
    The layer implementaion for [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/pdf/2307.08621.pdf).  # noqa

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 1.0.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        num_heads (int, Optional):
            The number of heads. Default: 8.
        gate_fn (str, Optional):
            The activation function for the output gate. Default: `swish`.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.key_dim = hidden_size 
        self.value_dim = hidden_size * 2
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False, device=device, dtype=dtype)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False, device=device, dtype=dtype)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False, device=device, dtype=dtype)

        self.g_norm = RMSNorm(self.head_v_dim, eps=norm_eps).to(device=device, dtype=dtype)
        self.gate_fn = ACT2FN['swish']

        # assert self.head_qk_dim <= 256, "head_qk_dim must be less than or equal to 256"
        self.rotary = RotaryPositionalEmbeddings(dim=self.head_qk_dim).to(device=device, dtype=dtype)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(k, '... (h d) -> ... h d', h=self.num_kv_heads)

        q = self.rotary(q)
        k = self.rotary(k)
        q = q.transpose(1, 2)
        if self.num_kv_groups > 1:
            k = repeat(k, 'b t h d -> b (h g) t d', h=self.num_kv_heads, g=self.num_kv_groups)
            v = repeat(v, 'b t (h d) -> b (h g) t d', h=self.num_kv_heads, g=self.num_kv_groups)
        else:
            k, v = rearrange(k, 'b t h d -> b h t d'), rearrange(v, 'b t (h d) -> b h t d', h=self.num_kv_heads)
        
        o = naive_retention(q, k, v)

        o = rearrange(o, 'b h l d -> b l h d')
        g = self.g_proj(hidden_states)
        o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')
        o = o * self.gate_fn(g)
        o = self.o_proj(o)

        return o

    
class RetNetMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        hidden_ratio = 2
        intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
        intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = ACT2FN['swish']

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z=self.act_fn(gate)*y
        x = self.down_proj(z)
        return x


class GAB(GABBase):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """
    def __init__(
            self,
            embed_dim: int, 
            device=None,
            dtype=None,
            expand_k: int = 1,
            expand_v: int = 2,
            num_heads: int = 8,
            norm_eps: float = 1e-6,
            **kwargs): # YOU CAN ADD MORE ARGUMENTS, BUT YOU HAVE TO HAVE embed_dim, device, dtype AS THE ARGUTMENTS #
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim) # DO NOT CHANGE THIS LINE #
        
        # COMPLETING THE CODE HERE #
        self.hidden_size = embed_dim

        self.attn_norm = RMSNorm(self.hidden_size, eps=norm_eps).to(device=device, dtype=dtype)
        self.attn = MultiScaleRetention(
            hidden_size=self.hidden_size,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            norm_eps=norm_eps,
            **factory_kwargs
        )
        self.mlp_norm = RMSNorm(self.hidden_size, eps=norm_eps).to(device=device, dtype=dtype)
        self.mlp = RetNetMLP(
            hidden_size=self.hidden_size, **factory_kwargs
        )

    # YOU CAN ADD MORE FUNCTIONS HERE #


    def _forward(self,X,**kwargs): # type hints are optional but recommended

        # THE CODE HERE MUST BE COMPLETED #
        hidden_states = self.attn_norm(X)
        X = self.attn(hidden_states=hidden_states)+X
        hidden_states = self.mlp_norm(X)
        X = self.mlp(hidden_states)+X
        return X
    
    
""" The dictionary of hyperparameters for constructing a GAB layer
    embed_dim, device, dtype should NOT be included in gab_config
"""
gab_config = {
    # THE HYPERPARAMETERS OF ADDITIONAL ARGUMENTS IN GAB CLASS #
    'expand_k': 1,
    'expand_v': 2,
    'num_heads': 8,
    'norm_eps': 1e-6,
}
