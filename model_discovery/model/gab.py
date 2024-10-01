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
        self.mha = AdaptiveMHA(embed_dim=self.embed_dim, block_loc=self.
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
from einops import rearrange


class AdaptiveMHA(GAUBase):
    """
    Adaptive Multi-Head Attention (AdaptiveMHA)

    This GAU implements the Adaptive Hybrid Attention Network (AHAN) design by incorporating multiple attention mechanisms:

    - **Global Attention**: Captures long-range dependencies using standard full-range self-attention.
    - **Causal Linear Attention**: Efficiently handles very long sequences using a causal linearized attention mechanism.
    - **Adaptive Attention Router**: Dynamically computes routing weights for attention types based on input characteristics, allowing the model to adaptively combine attention outputs.

    **Inputs**:
        - **X** (*Tensor*): Input sequence embeddings of shape (B, L, D), where B is batch size, L is sequence length, D is embedding dimension.
        - **Z** (*dict*): Intermediate variables.

    **Outputs**:
        - **Y** (*Tensor*): Output sequence embeddings of the same shape as X.
        - **Z\\_** (*dict*): Updated intermediate variables.

    **Example**:

        >>> attention = AdaptiveMHA(embed_dim=512, block_loc=(0, 0), kwarg_all={})
        >>> X = torch.randn(2, 1024, 512)
        >>> Y, Z = attention(X)

    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        n_heads: int=8, device=None, dtype=None, num_attention_types=2, **
        kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.global_attention = GlobalAttention(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.linear_attention = CausalLinearAttention(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)
        self.attention_router = AdaptiveAttentionRouter(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **self.factory_kwargs)
        num_attention_types = num_attention_types

    def _forward(self, X, **Z):
        """
        Forward pass of AdaptiveMHA.

        Args:
            X (Tensor): Input sequence of shape (B, L, D).

        Returns:
            Y (Tensor): Output sequence of shape (B, L, D).
            Z (dict): Updated intermediate variables.
        """
        _, Z = self.attention_router(X, **Z)
        routing_weights = Z.get('routing_weights', None)
        if routing_weights is None:
            routing_weights = torch.full((X.shape[0], X.shape[1], 2), 0.5,
                device=X.device, dtype=X.dtype)
        global_context, Z = self.global_attention(X, **Z)
        linear_context, Z = self.linear_attention(X, **Z)
        contexts = torch.stack([global_context, linear_context], dim=2)
        routing_weights = routing_weights.unsqueeze(-1)
        weighted_contexts = contexts * routing_weights
        combined_context = weighted_contexts.sum(dim=2)
        Y = self.out_proj(combined_context)
        return Y, Z


import torch.nn.functional as F
import math


class CausalLinearAttention(GAUBase):
    """
    Causal Linear Attention module for efficient sequence processing.

    This module implements a causal linear attention mechanism, which approximates
    the standard softmax attention in linear time complexity with respect to the
    sequence length. It ensures that the attention computation is causal, meaning
    each position only attends to positions up to and including itself.

    **Mathematical Overview:**

    The standard attention is computed as:

        A = softmax(Q K^T / sqrt(d_k)) V

    In linear attention, we approximate the softmax function using a positive
    definite kernel feature mapping φ(.), so that:

        Attention(Q, K, V) ≈ φ(Q) [φ(K)^T V]

    To ensure causality, cumulative sums are used during the computation.

    **Implementation Details:**

    - Queries, keys, and values are linear projections of the input.
    - A feature mapping (e.g., ELU + 1) is applied to queries and keys.
    - Causality is enforced through cumulative sums over the sequence length.
    - The output is passed through an output projection layer.

    **Args:**

        embed_dim (int): The dimension of the input embeddings.
        block_loc (tuple): The location of this block within the network.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): The device to use.
        dtype (torch.dtype, optional): The data type.

    **Example:**

        attention = CausalLinearAttention(embed_dim=512, block_loc=(0, 12), kwarg_all={})
        Y, Z = attention(X)

    **Note:**

    - This module is designed to handle sequences of shape (B, L, D).
    - It uses cumulative sums to enforce causality.

    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.feature_map = lambda x: F.elu(x) + 1

    def _forward(self, X, **Z):
        B, L, D = X.shape
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        Q = self.feature_map(Q)
        K = self.feature_map(K)
        K_cumsum = torch.cumsum(K, dim=1)
        KV_cumsum = torch.cumsum(K * V, dim=1)
        numerator = (Q * KV_cumsum).sum(dim=2, keepdim=True)
        denominator = (Q * K_cumsum).sum(dim=2, keepdim=True) + 1e-06
        Y = numerator / denominator
        Y = Y.expand(-1, -1, D)
        Y = self.out_proj(Y)
        Z_ = {}
        return Y, Z_


import torch.nn.functional as F
from einops import rearrange


class GlobalAttention(GAUBase):
    """
    Global Attention

    This GAU implements standard multi-head self-attention to capture long-range dependencies.

    Inputs:
        - X: Input sequence embeddings of shape (B, L, D).

    Outputs:
        - Y: Output sequence embeddings of the same shape as X.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        n_heads: int=8, causal: bool=True, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.causal = causal
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, **self.
            factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **self.factory_kwargs)

    def _forward(self, X, **Z):
        """
        Forward pass of GlobalAttention.

        Args:
            X (Tensor): Input sequence of shape (B, L, D).

        Returns:
            Y (Tensor): Output sequence of shape (B, L, D).
        """
        B, L, D = X.shape
        qkv = self.qkv_proj(X)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.n_heads)
        attn_scores = torch.einsum('b h i d, b h j d -> b h i j', q, k
            ) / self.head_dim ** 0.5
        if self.causal:
            mask = torch.triu(torch.ones(L, L, device=X.device), diagonal=1
                ).bool()
            attn_scores = attn_scores.masked_fill(mask[None, None, :, :],
                float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
        Y = self.out_proj(attn_output)
        return Y, Z


import torch.nn.functional as F
import math


class AdaptiveAttentionRouter(GAUBase):
    """
    AdaptiveAttentionRouter computes routing weights for different attention types for each token.

    **Inputs**:
    - **X** (*Tensor*): Input embeddings of shape `(batch_size, seq_length, embed_dim)`.
    - **Z** (*dict*, optional): Contains optional intermediate variables:
        - **'positional_ids'** (*Tensor*, optional): Positional indices of shape `(batch_size, seq_length)`.
          If not provided, defaults to `[0, 1, ..., seq_length - 1]` for each sequence in the batch.
        - **'sequence_lengths'** (*Tensor*, optional): Actual lengths of sequences in the batch of shape `(batch_size,)`.
          If not provided, defaults to `seq_length` for all sequences.

    **Outputs**:
    - **Y** (*Tensor*): Output embeddings (same as input `X`), shape `(batch_size, seq_length, embed_dim)`.
    - **Z_** (*dict*): Updated intermediate variables containing:
        - **'routing_weights'** (*Tensor*): Routing weights for attention types,
          shape `(batch_size, seq_length, num_attention_types)`.

    **Purpose**:
    The AdaptiveAttentionRouter dynamically computes routing weights for multiple attention mechanisms
    (e.g., global, linear) for each token in the input sequence. It considers the token embeddings,
    positional information, and sequence lengths to produce a softmax distribution over attention types,
    enabling the model to adaptively select the most appropriate attention mechanism for each token.

    **Details**:
    - Uses a small neural network (MLP) to compute routing weights.
    - Concatenates input features: token embeddings, positional embeddings, and sequence length encoding.
    - Normalizes routing weights using softmax to ensure they sum to 1 across attention types.
    - Handles missing positional encodings and sequence lengths by providing default values.

    **Example**:

        >>> router = AdaptiveAttentionRouter(embed_dim=512, block_loc=(0, 0), kwarg_all={})
        >>> X = torch.randn(2, 1024, 512)
        >>> Y, Z = router(X)  

    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_attention_types: int=2,
        router_hidden_dim: int=None, positional_dim: int=None, length_dim:
        int=1, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        if router_hidden_dim is None:
            router_hidden_dim = embed_dim // 2
        if positional_dim is None:
            positional_dim = embed_dim // 4
        self.num_attention_types = num_attention_types
        self.positional_dim = positional_dim
        self.length_dim = length_dim
        hidden_dim = router_hidden_dim
        input_dim = embed_dim + positional_dim + length_dim
        self.routing_network = nn.Sequential(nn.Linear(input_dim,
            hidden_dim, **self.factory_kwargs), nn.ReLU(), nn.Linear(
            hidden_dim, num_attention_types, **self.factory_kwargs))

    def _forward(self, X, **Z):
        batch_size, seq_length, _ = X.shape
        positional_ids = Z.get('positional_ids', None)
        if positional_ids is None:
            positional_ids = torch.arange(seq_length, device=X.device
                ).unsqueeze(0).expand(batch_size, seq_length)
        positional_embeddings = self.get_positional_embeddings(positional_ids,
            X.dtype)
        sequence_lengths = Z.get('sequence_lengths', None)
        if sequence_lengths is None:
            sequence_lengths = torch.full((batch_size,), seq_length, device
                =X.device, dtype=torch.long)
        sequence_length_encoding = sequence_lengths.unsqueeze(1).repeat(1,
            seq_length).unsqueeze(-1).to(dtype=X.dtype)
        input_features = torch.cat([X, positional_embeddings,
            sequence_length_encoding], dim=-1)
        routing_logits = self.routing_network(input_features)
        routing_weights = torch.softmax(routing_logits, dim=-1)
        Y = X
        Z_ = {'routing_weights': routing_weights}
        return Y, Z_

    def get_positional_embeddings(self, positional_ids, dtype):
        """
        Generates sinusoidal positional embeddings.

        Args:
            positional_ids (Tensor): Tensor of shape (batch_size, seq_length) containing positional indices.
            dtype (torch.dtype): Data type for the embeddings.

        Returns:
            Tensor: Positional embeddings of shape (batch_size, seq_length, positional_dim).
        """
        positions = positional_ids
        position_encodings = self.sinusoidal_embedding(positions, dtype)
        return position_encodings

    def sinusoidal_embedding(self, positions, dtype):
        """
        Creates sinusoidal positional embeddings as described in "Attention is All You Need".

        Args:
            positions (Tensor): Tensor of shape (batch_size, seq_length) containing positional indices.
            dtype (torch.dtype): Data type for the embeddings.

        Returns:
            Tensor: Sinusoidal positional embeddings of shape (batch_size, seq_length, positional_dim).
        """
        batch_size, seq_length = positions.shape
        dim = self.positional_dim
        positions = positions.to(dtype=dtype).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=positions.
            device, dtype=dtype) * -(math.log(10000.0) / dim))
        pe = torch.zeros(batch_size, seq_length, dim, device=positions.
            device, dtype=dtype)
        pe[..., 0::2] = torch.sin(positions * div_term)
        pe[..., 1::2] = torch.cos(positions * div_term)
        return pe


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


gab_config = {'hidden_features': None, 'out_features': None, 'activation':
    None, 'bias': False, 'multiple_of': 128, 'eps': 1e-05, 'n_heads': 8,
    'causal': True, 'num_attention_types': 2, 'router_hidden_dim': None,
    'positional_dim': None, 'length_dim': 1}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)