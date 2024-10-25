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
        self.mha = MemoryAugmentedMHA(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)
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
import math


class MemoryAugmentedMHA(GAUBase):
    """
    Memory-Augmented Multi-Head Attention (MAMHA)

    This GAU implements the Memory-Augmented Multi-Head Attention mechanism. It enhances
    the standard Multi-Head Attention (MHA) by integrating a gated memory mechanism with
    learnable memory tokens. The memory tokens allow the model to retain long-term dependencies,
    and the gating mechanism controls the influence of the memory tokens based on the current input.

    **Mathematical Formulation:**

    - Augmented keys and values:
      \\[
      	ilde{K} = egin{bmatrix} K \\ M \\end{bmatrix}, \\quad
      	ilde{V} = egin{bmatrix} V \\ M \\end{bmatrix}
      \\]
      where \\( M \\in \\mathbb{R}^{N_m 	imes d_{	ext{head}}} \\) are the memory tokens.

    - Gating mechanism:
      \\[
      G = \\sigma(Q W_g + b_g)
      \\]
      where \\( G \\in \\mathbb{R}^{B 	imes L 	imes N_m} \\).

    - Apply gating to the attention scores corresponding to memory tokens:
      \\[
      	ext{AttnScores}_{	ext{mem}} = 	ext{AttnScores}_{	ext{mem}} 	imes G
      \\]

    **Causal Masking:**

    - Apply causal masking to regular tokens to prevent attention to future positions.
    - Memory tokens are always attendable and are not masked.

    **Args:**
        embed_dim (int): Embedding dimension.
        block_loc (tuple): Location of the block within the network.
        kwarg_all (dict): Dictionary of all keyword arguments.
        num_heads (int, optional): Number of attention heads. Default: 8.
        num_memory_tokens (int, optional): Number of learnable memory tokens. Default: 4.
        device (torch.device, optional): Device to place the module. Default: None.
        dtype (torch.dtype, optional): Data type of the module. Default: None.

    **Attributes:**

        memory_tokens (nn.Parameter): Learnable memory tokens of shape (N_m, head_dim).
        Q_proj, K_proj, V_proj (nn.Linear): Linear layers for query, key, and value projections.
        gate_proj (nn.Linear): Linear layer for computing gating values.
        out_proj (nn.Linear): Output linear layer.

    **Example:**

        mha = MemoryAugmentedMHA(embed_dim=768, block_loc=(0, 12), kwarg_all={})
        X = torch.randn(8, 128, 768)  # Batch of 8, sequence length 128, embedding dim 768
        Y, Z = mha(X)

    **References:**
        - *Memory-Augmented Multi-Head Attention*, see proposal for details.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        num_heads: int=8, num_memory_tokens: int=4, device=None, dtype=None,
        **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        self.num_memory_tokens = num_memory_tokens
        self.memory_tokens = nn.Parameter(torch.randn(self.
            num_memory_tokens, self.head_dim, **self.factory_kwargs) * 0.01)
        self.Q_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.K_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.V_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.gate_proj = nn.Linear(embed_dim, self.num_heads * self.
            num_memory_tokens, **self.factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)

    def _forward(self, X, **Z):
        """
        Forward pass of the MemoryAugmentedMHA.

        Args:
            X (torch.Tensor): Input tensor of shape (B, L, embed_dim).
            **Z: Additional intermediate variables.

        Returns:
            output (torch.Tensor): Output tensor of shape (B, L, embed_dim).
            Z (dict): Updated intermediate variables.
        """
        B, L, _ = X.size()
        Q = self.Q_proj(X)
        K = self.K_proj(X)
        V = self.V_proj(X)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        M = self.memory_tokens.unsqueeze(0).unsqueeze(0).expand(B, self.
            num_heads, self.num_memory_tokens, self.head_dim)
        K = torch.cat([K, M], dim=2)
        V = torch.cat([V, M], dim=2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self
            .head_dim)
        attn_scores_reg = attn_scores[:, :, :, :L]
        attn_scores_mem = attn_scores[:, :, :, L:]
        causal_mask = torch.tril(torch.ones(L, L, device=X.device, dtype=
            torch.bool)).unsqueeze(0).unsqueeze(0)
        attn_scores_reg = attn_scores_reg.masked_fill(~causal_mask, float(
            '-inf'))
        gate = torch.sigmoid(self.gate_proj(X))
        gate = gate.view(B, L, self.num_heads, self.num_memory_tokens).permute(
            0, 2, 1, 3)
        attn_scores_mem = attn_scores_mem * gate
        attn_scores = torch.cat([attn_scores_reg, attn_scores_mem], dim=-1)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L,
            self.embed_dim)
        output = self.out_proj(attn_output)
        return output, Z


gab_config = {'num_memory_tokens': 4, 'num_heads': 8, 'eps': 1e-05, 'bias':
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