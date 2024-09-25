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
        self.mha = CondSparseMHA(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
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
from einops import rearrange


class CondSparseMHA(GAUBase):
    """
    Conditional Sparse Multi-Head Attention (CS-MHA)

    This GAU implements a multi-head self-attention mechanism with conditional sparse attention.
    It introduces a scoring network that dynamically assigns importance weights to each attention
    head based on the input context. The attention outputs of different heads are scaled by these
    weights, allowing the model to focus on the most relevant heads while maintaining differentiability.

    Args:
        embed_dim (int): The embedding dimension.
        block_loc (tuple): The location of the block within the network.
        kwarg_all (dict): Dictionary of all keyword arguments for initializing child units.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        causal (bool): Whether to apply causal masking.
        softmax_scale (float): Scaling factor for softmax.
        rotary_emb_base (float): Base value for rotary positional embeddings.
        **kwargs: Additional keyword arguments.

    Inputs:
        X (Tensor): Input embeddings of shape (B, L, D).

    Outputs:
        Y (Tensor): Output embeddings of shape (B, L, D).

    Example:
        >>> cond_sparse_mha = CondSparseMHA(embed_dim=512, block_loc=(0, 12), kwarg_all={}, n_heads=8)
        >>> Y, _ = cond_sparse_mha(X)
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        n_heads: int=8, dropout: float=0.1, causal: bool=True,
        softmax_scale: float=None, rotary_emb_base: float=10000.0, device=
        None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == self.embed_dim, 'embed_dim must be divisible by n_heads'
        self.query = nn.Linear(embed_dim, embed_dim, **self.factory_kwargs)
        self.key = nn.Linear(embed_dim, embed_dim, **self.factory_kwargs)
        self.value = nn.Linear(embed_dim, embed_dim, **self.factory_kwargs)
        self.scoring_network = nn.Linear(embed_dim, n_heads, **self.
            factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **self.factory_kwargs)
        kwarg_all['rotary_emb_dim'] = self.head_dim
        self.rotary_emb = RotaryPositionalEmbeddings(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        """
        Forward pass of the Conditional Sparse Multi-Head Attention.

        Args:
            X (Tensor): Input tensor of shape (B, L, D).
            **Z: Additional intermediate variables.

        Returns:
            Y (Tensor): Output tensor of shape (B, L, D).
            Z (dict): Updated intermediate variables.
        """
        B, L, D = X.size()
        Q, K, V = self._project_qkv(X)
        Q, K, Z = self._apply_rotary_embeddings(Q, K, X, Z)
        head_weights = self._compute_head_weights(X)
        Q = Q * head_weights
        K = K * head_weights
        V = V * head_weights
        attn_output = self._compute_attention(Q, K, V)
        Y = self.out_proj(attn_output)
        return Y, Z

    def _project_qkv(self, X):
        """
        Projects the input tensor X to queries, keys, and values.

        Args:
            X (Tensor): Input tensor of shape (B, L, D).

        Returns:
            Q (Tensor): Queries of shape (B, L, n_heads, head_dim).
            K (Tensor): Keys of shape (B, L, n_heads, head_dim).
            V (Tensor): Values of shape (B, L, n_heads, head_dim).
        """
        Q = self.query(X).view(X.size(0), -1, self.n_heads, self.head_dim)
        K = self.key(X).view(X.size(0), -1, self.n_heads, self.head_dim)
        V = self.value(X).view(X.size(0), -1, self.n_heads, self.head_dim)
        return Q, K, V

    def _apply_rotary_embeddings(self, Q, K, X, Z):
        """
        Applies Rotary Positional Embeddings to queries and keys.

        Args:
            Q (Tensor): Queries.
            K (Tensor): Keys.
            X (Tensor): Input tensor.
            Z (dict): Intermediate variables.

        Returns:
            Q (Tensor): Queries with rotary embeddings applied.
            K (Tensor): Keys with rotary embeddings applied.
            Z (dict): Updated intermediate variables.
        """
        Z['input_emb'] = Q
        _, Z = self.rotary_emb(X, **Z)
        Q = Z['output_emb']
        Z['input_emb'] = K
        _, Z = self.rotary_emb(X, **Z)
        K = Z['output_emb']
        return Q, K, Z

    def _compute_head_weights(self, X):
        """
        Computes importance weights for each attention head.

        Args:
            X (Tensor): Input tensor of shape (B, L, D).

        Returns:
            head_weights (Tensor): Head weights of shape (B, L, n_heads, 1).
        """
        scores = self.scoring_network(X)
        head_weights = F.softmax(scores, dim=-1)
        head_weights = head_weights.unsqueeze(-1)
        return head_weights

    def _compute_attention(self, Q, K, V):
        """
        Computes the scaled dot-product attention.

        Args:
            Q (Tensor): Queries of shape (B, L, n_heads, head_dim).
            K (Tensor): Keys of shape (B, L, n_heads, head_dim).
            V (Tensor): Values of shape (B, L, n_heads, head_dim).

        Returns:
            attn_output (Tensor): Attention output of shape (B, L, D).
        """
        B, L, n_heads, head_dim = Q.size()
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        scale = self.softmax_scale or 1 / math.sqrt(head_dim)
        Q = Q * scale
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        if self.causal:
            causal_mask = torch.triu(torch.ones(L, L, device=Q.device,
                dtype=Q.dtype), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask == 1, float(
                '-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L,
            self.embed_dim)
        return attn_output


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


gab_config = {'hidden_features': None, 'out_features': None, 'activation':
    None, 'bias': False, 'multiple_of': 128, 'eps': 1e-05,
    'rotary_emb_base': 10000.0, 'max_seq_len': 4096, 'n_heads': 8,
    'dropout': 0.1, 'causal': True, 'softmax_scale': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)