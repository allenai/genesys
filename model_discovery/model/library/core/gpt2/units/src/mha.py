import torch    
import torch.nn as nn

from model_discovery.model.utils.modules import GAUBase,gau_test # DO NOT CHANGE THIS IMPORT STATEMENT #

import torch.nn.functional as F
import math
from einops import rearrange, repeat




class MHA(GAUBase):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim: int,
        block_loc: tuple,
        kwarg_all: dict,
        n_heads: int=8, 
        causal: bool=True,
        num_heads_kv: int=None,
        head_dim: int=None,  # If None, use embed_dim // num_heads
        mlp_dim: int=0,
        qkv_proj_bias: bool=True,
        out_proj_bias: bool=True,
        softmax_scale: float=None,
        rotary_emb_base=10000.0,
        d_conv: int=0,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.d_conv = d_conv
        self.softmax_scale = softmax_scale
        self.causal = causal

        self.num_heads = n_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else n_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        if head_dim is None:
            assert self.embed_dim % n_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = head_dim if head_dim is not None else self.embed_dim // n_heads
        self.mlp_dim = math.ceil(mlp_dim / 256) * 256
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads

        kwarg_all['rotary_emb_dim'] = self.head_dim 
        self.rotary_emb = RotaryPositionalEmbeddings(
            embed_dim=self.embed_dim,
            block_loc=self.block_loc,
            kwarg_all=self.kwarg_all,
            **self.factory_kwargs,
            **self.kwarg_all
        )

        self.in_proj = nn.Linear(embed_dim, qkv_dim + self.mlp_dim, bias=qkv_proj_bias, **self.factory_kwargs)
        if self.d_conv > 0:
            self.conv1d = nn.Conv1d(
                qkv_dim, qkv_dim, kernel_size=self.d_conv, padding=self.d_conv - 1, groups=qkv_dim,
                **self.factory_kwargs
            )
        self.out_proj = nn.Linear(out_dim + self.mlp_dim // 2, embed_dim, bias=out_proj_bias, **self.factory_kwargs)

    def _forward(self, X, **Z):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        qkv = self.in_proj(X)
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)
        if self.d_conv > 0:
            qkv = rearrange(
                self.conv1d(rearrange(qkv, "b s d -> b d s"))[..., :-(self.d_conv - 1)], "b d s -> b s d"
            ).contiguous()

        q,k,v=qkv.split([self.num_heads * self.head_dim]*3, dim=-1)
        q=rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k=rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v=rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        Z['input_emb']=q
        _,Z=self.rotary_emb(X,**Z)
        q=Z['output_emb']

        Z['input_emb']=k
        _,Z=self.rotary_emb(X,**Z)
        k=Z['output_emb']

        k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)
        v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)
        context = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
        ).transpose(1, 2)
        context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        return out


@gau_test
def test_mha(device=None,dtype=None):   
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    mha = MHA(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    Z={}
    y,Z_=mha(x,**Z)   
    assert y.shape==(1,100,128)

SPEC = {
    "unitname": "MHA",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
MHA
''',
}   
ARGS = {
    "n_heads":8,
    "causal":True,
    "num_heads_kv":None,
    "head_dim":None,
    "mlp_dim":0,
    "qkv_proj_bias":True,
    "out_proj_bias":True,
    "softmax_scale":None,   
    "rotary_emb_base":10000.0,
    "d_conv":0,
}
CHILDREN = ['RotaryPositionalEmbeddings']
DESC='''
''' 

