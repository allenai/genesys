import torch    

from torch import nn, Tensor

from model_discovery.model.utils.modules import GABBase,GAUBase # DO NOT CHANGE THIS IMPORT STATEMENT #

import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange, repeat




class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = GPT2(embed_dim=embed_dim,
            block_loc=block_loc, kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


class GPT2(GAUBase):
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.mha=MHA(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.mlp=GatedMLP(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.norm1=RMSNorm(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.norm2=RMSNorm(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)   

    def _forward(self,X,**Z): # type hints are optional but recommended
        X,Z=self.mha(X,**Z)
        X,Z=self.norm1(X,**Z)
        X,Z=self.mlp(X,**Z)
        X,Z=self.norm2(X,**Z)
        return X,Z


class GatedMLP(GAUBase):
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None,
            hidden_features=None,out_features=None,activation=None,bias=False,multiple_of=128,**kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        out_features = out_features if out_features is not None else embed_dim
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * embed_dim / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(embed_dim, 2 * hidden_features, bias=bias, **self.factory_kwargs)
        self.activation = activation if activation is not None else F.silu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **self.factory_kwargs)

    def _forward(self, X, **Z):
        y = self.fc1(X)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


class RMSNorm(GAUBase):
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None,
            eps=1e-5, **kwargs):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.weight = nn.Parameter(torch.ones(embed_dim,**self.factory_kwargs))
        self.variance_epsilon = eps

    def _forward(self, X, **Z):
        input_dtype = X.dtype
        X = X.to(torch.float32)
        variance = X.pow(2).mean(-1, keepdim=True)
        X = X * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * X.to(input_dtype)


class RotaryPositionalEmbeddings(GAUBase):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        embed_dim: int,
        block_loc: tuple,
        kwarg_all: dict,
        device=None,
        dtype=None,
        rotary_emb_base: int = 10_000,
        rotary_emb_dim: int = None,
        max_seq_len: int = 4096,
        **kwargs
    ) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.dim = rotary_emb_dim
        self.base = rotary_emb_base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2,**self.factory_kwargs)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def _forward(self, X: Tensor, *, input_emb: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = input_emb.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = input_emb.float().reshape(*input_emb.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        output_emb = x_out.type_as(input_emb)
        return X,{"output_emb":output_emb}



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

        kwarg_all['rotary_emb_base'] = rotary_emb_base
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


    
gab_config = {'n_heads':8}

