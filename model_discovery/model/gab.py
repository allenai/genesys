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


class AdaptiveMHA(GAUBase):
    """
    Adaptive Multi-Head Attention (AdaptiveMHA)

    This GAU implements the Adaptive Hybrid Attention Network (AHAN) design by incorporating multiple attention mechanisms:

    - Global Attention: Captures long-range dependencies using standard full-range self-attention.

    The Adaptive Attention Router dynamically computes routing weights for the attention type based on input characteristics,
    allowing the model to adaptively combine attention outputs.

    **Note:** Due to causality issues with `LinearAttention`, it has been temporarily excluded.

    Inputs:
        - X: Input sequence embeddings of shape (B, L, D), where B is batch size, L is sequence length, D is embedding dimension.

    Outputs:
        - Y: Output sequence embeddings of the same shape as X.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        n_heads: int=8, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.global_attention = GlobalAttention(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **self.factory_kwargs)

    def _forward(self, X, **Z):
        """
        Forward pass of AdaptiveMHA.

        Args:
            X (Tensor): Input sequence of shape (B, L, D).

        Returns:
            Y (Tensor): Output sequence of shape (B, L, D).
            Z (dict): Updated intermediate variables.
        """
        routing_weights = {'global': torch.ones(X.shape[:-1] + (1,), device
            =X.device, dtype=X.dtype)}
        global_context, Z = self.global_attention(X, **Z)
        Y = routing_weights['global'] * global_context
        Y = self.out_proj(Y)
        return Y, Z


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
    - **Z\\_** (*dict*): Updated intermediate variables containing:
        - **'routing_weights'** (*Tensor*): Routing weights for attention types,
          shape `(batch_size, seq_length, num_attention_types)`.

    **Purpose**:
    The AdaptiveAttentionRouter dynamically computes routing weights for multiple attention mechanisms
    (e.g., global, local, linear) for each token in the input sequence. It considers the token embeddings,
    positional information, and sequence lengths to produce a softmax distribution over attention types,
    enabling the model to adaptively select the most appropriate attention mechanism for each token.

    **Details**:
    - Uses a small neural network (MLP) to compute routing weights.
    - Concatenates input features: token embeddings, positional embeddings, and sequence length encoding.
    - Normalizes routing weights using softmax to ensure they sum to 1 across attention types.
    - Handles missing positional encodings and sequence lengths by providing default values.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_attention_types=3, router_hidden_dim=
        None, positional_dim=None, length_dim=1, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        if router_hidden_dim is None:
            router_hidden_dim = embed_dim
        if positional_dim is None:
            positional_dim = embed_dim
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
        positional_embeddings = self.get_positional_embeddings(positional_ids)
        sequence_lengths = Z.get('sequence_lengths', None)
        if sequence_lengths is None:
            sequence_lengths = torch.full((batch_size,), seq_length, device
                =X.device)
        sequence_length_encoding = sequence_lengths.unsqueeze(1).repeat(1,
            seq_length).unsqueeze(-1).float()
        input_features = torch.cat([X, positional_embeddings,
            sequence_length_encoding], dim=-1)
        routing_logits = self.routing_network(input_features)
        routing_weights = torch.softmax(routing_logits, dim=-1)
        Y = X
        Z_ = {'routing_weights': routing_weights}
        return Y, Z_

    def get_positional_embeddings(self, positional_ids):
        positions = positional_ids
        position_encodings = self.sinusoidal_embedding(positions)
        return position_encodings

    def sinusoidal_embedding(self, positions):
        batch_size, seq_length = positions.shape
        dim = self.positional_dim
        positions = positions.float().unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=positions.
            device, dtype=positions.dtype) * -(math.log(10000.0) / dim))
        pe = torch.zeros(batch_size, seq_length, dim, device=positions.
            device, dtype=positions.dtype)
        pe[..., 0::2] = torch.sin(positions * div_term)
        pe[..., 1::2] = torch.cos(positions * div_term)
        return pe


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


gab_config = {'hidden_features': None, 'out_features': None, 'activation':
    None, 'bias': False, 'multiple_of': 128, 'eps': 1e-05, 'n_heads': 8,
    'causal': True, 'num_attention_types': 3, 'router_hidden_dim': None,
    'positional_dim': None, 'length_dim': 1}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)