import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = GPT2(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


import torch.nn.functional as F
from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl


class GPT2(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.mha = HierarchicalAdaptiveAttention(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.mlp = GatedMLP(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm1 = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm2 = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)

    def _forward(self, X, **Z):
        X1, Z = self.norm1(X, **Z)
        X2, Z = self.mha(X1, **Z)
        X = X + X2
        X3, Z = self.norm2(X, **Z)
        X4, Z = self.mlp(X3, **Z)
        X = X + X4
        return X, Z


import torch.nn.functional as F
from torch import Tensor


class RMSNorm(GAUBase):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    This layer applies a variant of layer normalization that uses only the root mean square
    statistics, without centering. It's computationally more efficient than standard
    layer normalization and has been shown to be effective in various NLP tasks.

    Args:
        embed_dim (int): The size of the input feature dimension.
        block_loc (tuple): The location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the parent class.
        device (torch.device, optional): The device on which to allocate the module's parameters.
        dtype (torch.dtype, optional): The dtype of the module's parameters.
        eps (float, optional): A small constant added to the denominator for numerical stability.
            Default: 1e-5.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter of shape (embed_dim,).
        variance_epsilon (float): The epsilon value used in the normalization formula.

    Shape:
        - Input: (*, embed_dim)
        - Output: (*, embed_dim) (same shape as input)

    Examples:
        >>> rmsnorm = RMSNorm(128, (0, 6), {})
        >>> x = torch.randn(1, 100, 128)
        >>> output = rmsnorm(x)
        >>> print(output.shape)
        torch.Size([1, 100, 128])

    References:
        - Paper: "Root Mean Square Layer Normalization" by Biao Zhang and Rico Sennrich
          https://arxiv.org/abs/1910.07467
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, eps=1e-05, **kwargs):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.weight = nn.Parameter(torch.ones(embed_dim, **self.factory_kwargs)
            )
        self.variance_epsilon = eps

    def _forward(self, X, **Z):
        input_dtype = X.dtype
        X = X.to(torch.float32)
        variance = X.pow(2).mean(-1, keepdim=True)
        X = X * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * X.to(input_dtype)


import torch.nn.functional as F


class GatedMLP(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, hidden_features=None, out_features=None,
        activation=None, bias=False, multiple_of=128, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        out_features = out_features if out_features is not None else embed_dim
        hidden_features = (hidden_features if hidden_features is not None else
            int(8 * embed_dim / 3))
        hidden_features = (hidden_features + multiple_of - 1
            ) // multiple_of * multiple_of
        self.fc1 = nn.Linear(embed_dim, 2 * hidden_features, bias=bias, **
            self.factory_kwargs)
        self.activation = activation if activation is not None else F.silu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **
            self.factory_kwargs)

    def _forward(self, X, **Z):
        y = self.fc1(X)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


import torch.nn.functional as F
import math


class HierarchicalAdaptiveAttention(GAUBase):
    """
    Hierarchical Adaptive Multi-Head Attention (HA-MHA)

    This module implements a hierarchical adaptive multi-head attention mechanism that
    captures multi-scale dependencies in the input sequence. It organizes attention heads
    into hierarchical groups, each responsible for capturing dependencies at different scales
    (e.g., local, medium, global). An adaptive gating mechanism dynamically allocates attention
    resources based on the input context, allowing the model to focus on the most relevant
    information at each scale.

    **Main Features:**
    - **Hierarchical Structure**: Attention heads are grouped into multiple scales to capture
      dependencies at different levels.
    - **Multi-Scale Linear Attention**: Reduces computational complexity from O(N^2) to O(N)
      within each hierarchical group using linear attention mechanisms.
    - **Adaptive Gating Mechanism**: Dynamically scales the contribution of each hierarchical group
      based on the input context using a gating function.
    - **Dynamic Composition**: Composes attention outputs from all hierarchical groups adaptively.
    - **Rotary Positional Embeddings**: Incorporates positional information using rotary embeddings.
    - **Causal Attention**: Ensures autoregressive property by masking future positions.

    Args:
        embed_dim (int): Total embedding dimension.
        block_loc (tuple): Location of the block within the network.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): The device to use.
        dtype (torch.dtype, optional): The data type to use.
        num_heads (int): Total number of attention heads.
        num_scales (int): Number of hierarchical scales.
        dropout (float): Dropout probability.
        rotary_emb_base (float): Base for rotary positional embeddings.
        **kwargs: Additional keyword arguments.

    Shape:
        - Input: X of shape (batch_size, seq_len, embed_dim)
        - Output: Y of shape (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_heads: int=8, num_scales: int=2,
        dropout: float=0.1, rotary_emb_base: float=10000.0, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        assert embed_dim % (num_heads * num_scales
            ) == 0, f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}) * num_scales ({num_scales})'
        assert num_heads > 0, f'num_heads must be positive, got {num_heads}'
        assert num_scales > 0, f'num_scales must be positive, got {num_scales}'
        assert 0 <= dropout < 1, f'dropout must be in [0,1), got {dropout}'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.head_dim = embed_dim // (num_heads * num_scales)
        self.dropout = dropout
        self.scaling_factor = 1.0 / math.sqrt(self.head_dim)
        self.query_projs = nn.ModuleList([nn.Linear(embed_dim, num_heads *
            self.head_dim, bias=False, **self.factory_kwargs) for _ in
            range(num_scales)])
        self.key_projs = nn.ModuleList([nn.Linear(embed_dim, num_heads *
            self.head_dim, bias=False, **self.factory_kwargs) for _ in
            range(num_scales)])
        self.value_projs = nn.ModuleList([nn.Linear(embed_dim, num_heads *
            self.head_dim, bias=False, **self.factory_kwargs) for _ in
            range(num_scales)])
        self.gate_proj = nn.Linear(embed_dim, num_scales, bias=False, **
            self.factory_kwargs)
        self.out_proj = nn.Linear(num_heads * self.head_dim * num_scales,
            embed_dim, **self.factory_kwargs)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        kwarg_all['rotary_emb_dim'] = self.head_dim
        self.rotary_emb = RotaryPositionalEmbeddings(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        if X.size(1) == 0:
            return X, Z
        B, L, D = X.size()
        assert D == self.embed_dim, f'Expected input embedding dimension: {self.embed_dim}, got: {D}'
        causal_mask = torch.triu(torch.ones(L, L, device=X.device), diagonal=1
            ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        gate_scores = torch.sigmoid(self.gate_proj(X))
        attn_outputs = []
        for scale in range(self.num_scales):
            Q = self.query_projs[scale](X).view(B, L, self.num_heads, self.
                head_dim)
            K = self.key_projs[scale](X).view(B, L, self.num_heads, self.
                head_dim)
            V = self.value_projs[scale](X).view(B, L, self.num_heads, self.
                head_dim)
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            Z['input_emb'] = Q
            _, Z = self.rotary_emb(X, **Z)
            Q = Z['output_emb']
            Z['input_emb'] = K
            _, Z = self.rotary_emb(X, **Z)
            K = Z['output_emb']
            Q = Q * self.scaling_factor
            attn_weights = torch.matmul(Q, K.transpose(-2, -1))
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout_layer(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
            attn_outputs.append(attn_output)
        attn_output = torch.cat(attn_outputs, dim=-1)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, -1)
        gate_scores = gate_scores.unsqueeze(-1).expand(-1, -1, -1, self.
            num_heads * self.head_dim)
        attn_output = attn_output.view(B, L, self.num_scales, -1)
        attn_output = attn_output * gate_scores
        attn_output = attn_output.reshape(B, L, -1)
        Y = self.out_proj(attn_output)
        return Y, Z


import torch.nn.functional as F
from torch import Tensor
from typing import Optional


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

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, rotary_emb_base: int=10000, rotary_emb_dim:
        int=None, max_seq_len: int=4096, **kwargs) ->None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.dim = rotary_emb_dim
        self.base = rotary_emb_base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / self.base ** (torch.arange(0, self.dim, 2, **self.
            factory_kwargs)[:self.dim // 2].float() / self.dim)
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int=4096) ->None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=
            self.theta.device)
        idx_theta = torch.einsum('i, j -> ij', seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)],
            dim=-1)
        self.register_buffer('cache', cache, persistent=False)

    def _forward(self, X: Tensor, input_emb: Tensor, input_pos: Optional[
        Tensor]=None) ->Tensor:
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
        seq_len = input_emb.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[
            input_pos]
        xshaped = input_emb.float().reshape(*input_emb.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2
            )
        x_out = torch.stack([xshaped[..., 0] * rope_cache[..., 0] - xshaped
            [..., 1] * rope_cache[..., 1], xshaped[..., 1] * rope_cache[...,
            0] + xshaped[..., 0] * rope_cache[..., 1]], -1)
        x_out = x_out.flatten(3)
        output_emb = x_out.type_as(input_emb)
        return X, {'output_emb': output_emb}


gab_config = {'dropout': 0.1, 'num_scales': 2, 'num_heads': 8,
    'rotary_emb_base': 10000.0, 'eps': 1e-05, 'max_seq_len': 4096, 'bias': 
    False, 'multiple_of': 128, 'hidden_features': None, 'out_features':
    None, 'activation': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)