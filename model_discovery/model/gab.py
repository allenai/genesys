import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = EventVQ(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
import torch.nn.functional as F


class EventVQ(GAUBase):
    """
    EventVQ Block: Event-Driven Vector Quantized Language Model Block

    This block orchestrates the main components of the EventVQ design,
    integrating event detection, vector quantization, and attention mechanisms
    to create an efficient and adaptive language model block.

    **Core Components:**
    - **Event Detection and Quantization Module**: Prepares inputs for attention based on detected events.
    - **Hierarchical State Manager**: Manages state compression and updates.
    - **Selective Attention Computer**: Computes attention using quantized keys and values.

    **Inputs:**
        - **X**: Input tensor of shape (batch_size, seq_len, embed_dim).

    **Outputs:**
        - **Y**: Output tensor of shape (batch_size, seq_len, embed_dim).
        - **Z**: Dictionary of intermediate variables.

    **Args:**
        embed_dim (int): Embedding dimension.
        block_loc (tuple): Location of the block within the network.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): Device to place the model on.
        dtype (torch.dtype, optional): Data type of the model parameters.

    **Example Usage:**

        >>> eventvq_block = EventVQ(embed_dim=512, block_loc=(0, 1), kwarg_all={})
        >>> X = torch.randn(2, 1024, 512)
        >>> Y, Z = eventvq_block(X)

    **Note:**
    - This block is designed to operate within a stack of blocks in an autoregressive language model.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.hidden_size = embed_dim
        self.seq_norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.ffn_norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.attention = EDVQAttention(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)
        self.mlp = SwiGluMLP(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)

    def _forward(self, X, **Z):
        hidden_states = X
        residual = hidden_states
        hidden_states, _ = self.seq_norm(hidden_states, **Z)
        hidden_states, Z = self.attention(hidden_states, **Z)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states, _ = self.ffn_norm(hidden_states, **Z)
        hidden_states, _ = self.mlp(hidden_states, **Z)
        hidden_states = residual + hidden_states
        return hidden_states, Z


import torch.nn.functional as F


class RMSNorm(GAUBase):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    This layer applies a variant of layer normalization that uses only the root mean square
    statistics, without centering. It's computationally more efficient than standard
    layer normalization and has been shown to be effective in various NLP tasks.

    **Args:**
        embed_dim (int): The size of the input feature dimension.
        block_loc (tuple): The location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the parent class.
        device (torch.device, optional): The device on which to allocate the module's parameters.
        dtype (torch.dtype, optional): The dtype of the module's parameters.
        eps (float, optional): A small constant added to the denominator for numerical stability.
            Default: 1e-5.

    **Attributes:**
        weight (nn.Parameter): Learnable scale parameter of shape (embed_dim,).
        variance_epsilon (float): The epsilon value used in the normalization formula.

    **Inputs:**
        - **X**: Input tensor of shape (batch_size, seq_len, embed_dim).

    **Outputs:**
        - **Y**: Output tensor of shape (batch_size, seq_len, embed_dim).

    **Example Usage:**

        >>> rmsnorm = RMSNorm(128, (0, 6), {})
        >>> x = torch.randn(1, 100, 128)
        >>> output, _ = rmsnorm(x)
        >>> print(output.shape)
        torch.Size([1, 100, 128])

    **References:**

    - Paper: "Root Mean Square Layer Normalization" by Biao Zhang and Rico Sennrich
      https://arxiv.org/abs/1910.07467
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, eps=1e-05, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        """Initialize RMSNorm module."""
        self.weight = nn.Parameter(torch.ones(embed_dim, **self.factory_kwargs)
            )
        self.variance_epsilon = eps

    def _forward(self, X, **Z):
        input_dtype = X.dtype
        X = X.to(torch.float32)
        variance = X.pow(2).mean(-1, keepdim=True)
        X = X * torch.rsqrt(variance + self.variance_epsilon)
        Y = self.weight * X.to(input_dtype)
        return Y


import torch.nn.functional as F


class EDVQAttention(GAUBase):
    """
    EDVQAttention: Event-Driven Vector Quantized Attention Unit

    This unit integrates event detection, vector quantization, and attention computation
    to create an efficient and adaptive attention mechanism.

    **Core Components:**
    - **Event Detection**: Identifies important events in the input sequence.
    - **Vector Quantization**: Compresses inputs based on importance.
    - **Attention Mechanism**: Computes attention using quantized and original inputs with causal masking.

    **Mathematical Formulation:**
    1. Event Detection:
       \\[ e(x) = \\sigma(W_e x + b_e) \\]
       \\[ 	ext{importance} = e(x) \\]

    2. Vector Quantization:
       \\[ x_{q} = 	ext{VQ}(x) \\]

    3. Attention Computation:
       \\[ y = 	ext{Attention}(Q, K', V') \\]
       where \\( K' = 	ext{importance} \\cdot K + (1 - 	ext{importance}) \\cdot x_{q} \\)

    **Args:**
        embed_dim (int): Embedding dimension.
        block_loc (tuple): Location within the network.
        kwarg_all (dict): Additional keyword arguments.
        num_heads (int, optional): Number of attention heads. Default is 8.
        device (optional): Device to place the model on.
        dtype (optional): Data type of the model parameters.

    **Inputs:**
        - **X**: Input tensor of shape (batch_size, seq_len, embed_dim).

    **Outputs:**
        - **Y**: Output tensor of shape (batch_size, seq_len, embed_dim).
        - **Z'**: Dictionary containing intermediate variables, e.g., 'importance'.

    **Example Usage:**

        >>> edvq_attn = EDVQAttention(embed_dim=512, block_loc=(0, 1), kwarg_all={})
        >>> X = torch.randn(2, 128, 512)
        >>> Y, Z = edvq_attn(X)

    **Note:**
        - This unit is designed to be used within a stack of blocks in an autoregressive language model.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_heads=8, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        self.event_linear = nn.Linear(embed_dim, 1, **self.factory_kwargs)
        self.codebook = nn.Parameter(torch.randn(256, self.head_dim, **self
            .factory_kwargs) / self.head_dim ** 0.5)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)

    def _quantize(self, x):
        BNH, L, D = x.shape
        x_flat = x.view(-1, D)
        distances = torch.cdist(x_flat, self.codebook)
        indices = distances.argmin(dim=1)
        x_q = self.codebook[indices]
        x_q = x_q + (x_flat - x_q).detach()
        x_q = x_q.view(BNH, L, D)
        return x_q

    def _forward(self, X, **Z):
        B, L, D = X.shape
        importance = torch.sigmoid(self.event_linear(X))
        Z_ = {'importance': importance.squeeze(-1)}
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        B_heads = B * self.num_heads
        K_reshaped = K.contiguous().view(B_heads, L, self.head_dim)
        V_reshaped = V.contiguous().view(B_heads, L, self.head_dim)
        K_quantized = self._quantize(K_reshaped).view(B, self.num_heads, L,
            self.head_dim)
        V_quantized = self._quantize(V_reshaped).view(B, self.num_heads, L,
            self.head_dim)
        importance_expanded = importance.unsqueeze(1)
        importance_expanded = importance_expanded.expand(-1, self.num_heads,
            -1, self.head_dim)
        K_q = importance_expanded * K + (1 - importance_expanded) * K_quantized
        V_q = importance_expanded * V + (1 - importance_expanded) * V_quantized
        attn_scores = torch.matmul(Q, K_q.transpose(-2, -1)
            ) / self.head_dim ** 0.5
        seq_len = L
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=X.
            device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_q)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        Y = self.out_proj(attn_output)
        assert Y.shape == X.shape, f'Output shape {Y.shape} does not match input shape {X.shape}'
        return Y, Z_


import torch.nn.functional as F
from typing import Optional
from transformers.activations import ACT2FN


class SwiGluMLP(GAUBase):
    """
    SwiGluMLP: Feed-Forward Network with SwiGLU activation function.

    This unit implements a feed-forward neural network using the SwiGLU activation function,
    as described in the paper "GLU Variants Improve Transformer" by Shazeer (2020).

    **Mathematical Formulation:**

    .. math::

        Y = 	ext{DownProj}(	ext{SwiGLU}(	ext{GateProj}(X)) \\odot 	ext{UpProj}(X))

    where:

    - \\( X \\) is the input tensor of shape (batch, seq\\_len, embed\\_dim).
    - \\( 	ext{GateProj} \\), \\( 	ext{UpProj} \\), and \\( 	ext{DownProj} \\) are linear projections.
    - \\( \\odot \\) denotes element-wise multiplication.
    - \\( 	ext{SwiGLU}(x) = 	ext{SiLU}(x) \\).

    **Args:**

        embed_dim (int): Embedding dimension of the input and output.
        block_loc (tuple): Location of the block within the network.
        kwarg_all (dict): Dictionary of all keyword arguments.
        intermediate_size (int, optional): Dimension of the intermediate projection.
            If None, defaults to int(embed_dim * 2.5).
        device (optional): Device to place the model on.
        dtype (optional): Data type of the model parameters.

    **Inputs:**

        - **X**: Input tensor of shape (batch, seq\\_len, embed\\_dim).

    **Outputs:**

        - **Y**: Output tensor of the same shape as input X.

    **Example:**

        >>> swiglu_mlp = SwiGluMLP(embed_dim=512, block_loc=(0, 0), kwarg_all={})
        >>> X = torch.randn(2, 1024, 512)
        >>> Y, Z = swiglu_mlp(X)

    **References:**

    - Shazeer, N. (2020). "GLU Variants Improve Transformer". arXiv preprint arXiv:2002.05202.

    **Note:**

    - The activation function used is 'silu', which is also known as the SiLU or Swish function.

    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, intermediate_size: Optional[int]=None, **
        kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.hidden_size = embed_dim
        self.intermediate_size = (intermediate_size if intermediate_size is not
            None else int(embed_dim * 2.5))
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,
            bias=False, **self.factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,
            bias=False, **self.factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,
            bias=False, **self.factory_kwargs)
        self.act_fn = ACT2FN['silu']

    def _forward(self, X, **Z):
        gate_output = self.act_fn(self.gate_proj(X))
        up_output = self.up_proj(X)
        hidden = gate_output * up_output
        Y = self.down_proj(hidden)
        return Y, {}


gab_config = {'num_heads': 8, 'eps': 1e-05, 'intermediate_size': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)