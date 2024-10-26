# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #

from transformers.activations import ACT2FN
from einops import rearrange, repeat
from torchtune.modules import RotaryPositionalEmbeddings,RMSNorm



class MultiScaleRetention(GAUBase): 
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None,
            hidden_size=None,num_heads: int = 8,norm_eps: float = 1e-5,**kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        
        hidden_size = hidden_size if hidden_size is not None else embed_dim
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


    def naive_retention(self, q, k, v): 
        orig_type = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        _, n_heads, seq_len, d_head = q.shape
        s = (1 - q.new_tensor(2., dtype=torch.float).pow(-5. - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
        n = q.new_tensor(range(seq_len), dtype=torch.float)
        n = torch.exp2((n.unsqueeze(-1) - n) * s.view(-1, 1, 1)) * n.unsqueeze(-1).ge(n)
        s = torch.einsum('bhqd,bhkd,hqk->bhqk', q * d_head ** -0.5, k, n.to(q.dtype))
        o = torch.einsum('bhqk,bhkd->bhqd', s, v)
        return o.to(orig_type)

    def _forward(self, X,**Z):
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)

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
        
        o = self.naive_retention(q, k, v)

        o = rearrange(o, 'b h l d -> b l h d')
        g = self.g_proj(X)
        o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')
        o = o * self.gate_fn(g)
        o = self.o_proj(o)

        return o


@gau_test
def test_multiscaleretention(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={'hidden_size':128}
    multiscaleretention=MultiScaleRetention(embed_dim,block_loc,kwarg_all,device=device,dtype=dtype,**kwarg_all)
    x=torch.randn(1,100,128).to(device=device,dtype=dtype)
    Z={}
    y,Z_=multiscaleretention(x,**Z)
    assert y.shape==(1,100,128)


CHILDREN_DECLARATIONS = []


SPEC ={
    "unitname": "MultiScaleRetention",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
RetNet MultiScaleRetention
''',
}
ARGS = {
    "hidden_size":None,
    "num_heads":8,
    "norm_eps":1e-5,
}
CHILDREN = []
DESC='''
'''

