import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = HybridMambaBlock(embed_dim=embed_dim, block_loc=
            block_loc, kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
import torch.nn.functional as F


class HybridMambaBlock(GAUBase):
    """
    HybridMamba block that combines SSM-inspired layers with modified Transformer blocks.
    
    This block implements the core architecture of HybridMamba, featuring:
    1. SSM Layer with matrix-valued states and dynamic recurrence
    2. Modified Transformer Layer with sparse attention
    3. Advanced normalization techniques (TCN and Adaptive Slice-level)
    
    The block processes input sequences through alternating SSM and Transformer 
    components with residual connections and normalization layers.
    
    Args:
        embed_dim (int): Dimension of input embeddings
        block_loc (tuple): Location of block in network (layer_idx, block_idx)
        kwarg_all (dict): Additional arguments dictionary
        n_heads (int, optional): Number of attention heads. Defaults to 8.
        mlp_ratio (float, optional): Ratio for MLP hidden dimension. Defaults to 4.0.
        ssm_ratio (float, optional): Ratio for SSM hidden dimension. Defaults to 2.0.
        norm_eps (float, optional): Epsilon for normalization. Defaults to 1e-5
        device (torch.device, optional): Device to place the module
        dtype (torch.dtype, optional): Data type of the module
        
    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        n_heads: int=8, mlp_ratio: float=4.0, ssm_ratio: float=2.0,
        norm_eps: float=1e-05, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.ssm_dim = int(embed_dim * ssm_ratio)
        self.mlp_dim = int(embed_dim * mlp_ratio)
        self.ssm_norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.ssm = SSMLayer(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.attn_norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.attn = SparseAttention(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)
        self.mlp_norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.mlp = GatedMLP(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.tcn = TransitionConstantNorm(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.adaptive_norm = AdaptiveSliceNorm(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        ssm_in = self.ssm_norm(X, **Z)[0]
        ssm_out, Z = self.ssm(ssm_in, **Z)
        X = X + ssm_out
        X, Z = self.tcn(X, **Z)
        attn_in = self.attn_norm(X, **Z)[0]
        attn_out, Z = self.attn(attn_in, **Z)
        X = X + attn_out
        mlp_in = self.mlp_norm(X, **Z)[0]
        mlp_out, Z = self.mlp(mlp_in, **Z)
        X = X + mlp_out
        X, Z = self.adaptive_norm(X, **Z)
        return X, Z


import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class RMSNorm(GAUBase):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    This layer applies a variant of layer normalization that uses only the root mean square
    statistics, without centering. It's computationally more efficient than standard
    layer normalization and has been shown to be effective in various NLP tasks.

    **Code Example:**

    .. code-block:: python

        rmsnorm = RMSNorm(128, (0, 6), {})
        x = torch.randn(1, 100, 128)
        output = rmsnorm(x)
        print(output.shape)  # torch.Size([1, 100, 128])

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
        eps (float): The epsilon value used in the normalization formula.

    **Shape:**

        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim) (same shape as input)

    **References:**

        - Paper: "Root Mean Square Layer Normalization" by Biao Zhang and Rico Sennrich
          https://arxiv.org/abs/1910.07467
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, eps: float=1e-05, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.ones(embed_dim, **self.factory_kwargs)
            )
        self.eps = eps

    def _forward(self, X, **Z):
        """
        Apply RMS normalization to the input tensor.

        Args:
            X (torch.Tensor): Input tensor of shape (..., embed_dim).
            **Z: Additional intermediate variables (unused).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as X.
        """
        input_dtype = X.dtype
        X_float = X.to(torch.float32)
        variance = X_float.pow(2).mean(dim=-1, keepdim=True)
        X_norm = X_float * torch.rsqrt(variance + self.eps)
        Y = self.weight * X_norm.to(input_dtype)
        return Y


import torch.nn.functional as F
import math


class AdaptiveSliceNorm(GAUBase):
    '\n    Adaptive Slice-level Normalization (AdaptiveSliceNorm).\n    \n    This GAU implements Adaptive Slice-level Normalization, which normalizes the input tensor\n    at the slice level, adapting scaling and shifting parameters dynamically based on the input.\n    \n    Key features:\n    - **Slice-level Normalization**: Divides the input along the sequence length and/or feature dimension\n      into slices and normalizes each slice independently.\n    - **Adaptive Scaling and Shifting**: Learnable parameters or mechanisms to scale and shift each slice\n      adaptively based on input data.\n    \n    **Mathematical Formulation:**\n    \n    .. math::\n    \n        Y_{b, s, d} = \\gamma_{d, l} \\left( \x0crac{X_{b, s, d}}{\\sqrt{\text{Var}(X_{b, s, \\cdot l}) + \\epsilon}} \right) + \x08eta_{d, l}\n    \n    where:\n    - :math:`Y` is the output tensor.\n    - :math:`X` is the input tensor.\n    - :math:`\\gamma` and :math:`\x08eta` are learnable scaling and shifting parameters for each slice.\n    - :math:`\\epsilon` is a small constant for numerical stability.\n    - :math:`l` indexes the slice.\n    \n    **Code Example:**\n    \n    .. code-block:: python\n    \n        adaptive_norm = AdaptiveSliceNorm(embed_dim=256, block_loc=(0,0), kwarg_all={})\n        x = torch.randn(32, 128, 256)  # (batch_size, seq_len, embed_dim)\n        y, Z = adaptive_norm(x)\n        print(y.shape)  # torch.Size([32, 128, 256])\n    \n    **Args:**\n    \n        embed_dim (int): Dimension of the input embeddings.\n        block_loc (tuple): Location of the block in the architecture (layer_idx, block_idx).\n        kwarg_all (dict): Additional keyword arguments.\n        device (torch.device, optional): Device to place the module.\n        dtype (torch.dtype, optional): Data type of the module.\n        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.\n        slice_size (int, optional): Size of each slice along the sequence or feature dimension. Defaults to 32.\n        slice_dim (int, optional): Dimension to slice on (1: sequence length, 2: embed_dim). Defaults to 2.\n        **kwargs: Additional keyword arguments.\n    \n    **Returns:**\n    \n        - Output tensor of the same shape as X.\n        - Updated intermediate variables Z.\n    \n    **References:**\n    \n        - Liu, Z., et al. (2023). "Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective"\n    '

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, eps: float=1e-05, slice_size: int=32,
        slice_dim: int=2, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        """
        Initialize the AdaptiveSliceNorm layer.
        
        Args:
            embed_dim (int): Dimension of the input embeddings.
            block_loc (tuple): Location of the block in the architecture (layer_idx, block_idx).
            kwarg_all (dict): Additional keyword arguments.
            device (torch.device, optional): Device to place the module.
            dtype (torch.dtype, optional): Data type of the module.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
            slice_size (int, optional): Size of each slice along the sequence or feature dimension. Defaults to 32.
            slice_dim (int, optional): Dimension to slice on (1: sequence length, 2: embed_dim). Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        self.embed_dim = embed_dim
        self.slice_size = slice_size
        self.slice_dim = slice_dim
        self.eps = eps
        if self.slice_dim == 2:
            self.num_slices = math.ceil(self.embed_dim / self.slice_size)
            self.actual_slice_size = math.ceil(self.embed_dim / self.num_slices
                )
            self.gamma = nn.Parameter(torch.ones(1, 1, self.num_slices,
                self.actual_slice_size, device=device, dtype=dtype))
            self.beta = nn.Parameter(torch.zeros(1, 1, self.num_slices,
                self.actual_slice_size, device=device, dtype=dtype))
        elif self.slice_dim == 1:
            self.gamma_network = nn.Linear(embed_dim, embed_dim, bias=True,
                **self.factory_kwargs)
            self.beta_network = nn.Linear(embed_dim, embed_dim, bias=True,
                **self.factory_kwargs)
        else:
            raise ValueError(
                'slice_dim must be 1 (sequence length) or 2 (embed_dim)')

    def _forward(self, X, **Z):
        """
        Forward pass for AdaptiveSliceNorm.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            **Z: Additional intermediate variables.
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as X.
            dict: Updated intermediate variables Z.
        """
        if self.slice_dim == 2:
            batch_size, seq_len, embed_dim = X.shape
            X_sliced = X.view(batch_size, seq_len, self.num_slices, self.
                actual_slice_size)
            mean = X_sliced.mean(dim=-1, keepdim=True)
            var = X_sliced.var(dim=-1, keepdim=True, unbiased=False)
            X_norm = (X_sliced - mean) / torch.sqrt(var + self.eps)
            Y = self.gamma * X_norm + self.beta
            Y = Y.view(batch_size, seq_len, embed_dim)
        elif self.slice_dim == 1:
            batch_size, seq_len, embed_dim = X.shape
            gamma = self.gamma_network(X).view(batch_size, seq_len, self.
                num_slices, self.slice_size)
            beta = self.beta_network(X).view(batch_size, seq_len, self.
                num_slices, self.slice_size)
            X_sliced = X.view(batch_size, seq_len, self.num_slices, self.
                slice_size)
            mean = X_sliced.mean(dim=-1, keepdim=True)
            var = X_sliced.var(dim=-1, keepdim=True, unbiased=False)
            X_norm = (X_sliced - mean) / torch.sqrt(var + self.eps)
            Y = gamma * X_norm + beta
            Y = Y.view(batch_size, seq_len, embed_dim)
        else:
            raise ValueError('slice_dim must be 1 or 2')
        return Y, Z


import torch.nn.functional as F
from math import sqrt


class SparseAttention(GAUBase):
    """
    Sparse Attention with SPARSEK Mechanism.

    This GAU implements an efficient sparse attention mechanism inspired by SPARSEK Attention. It dynamically selects
    the top-k relevant key-value pairs for each query to reduce computational complexity while maintaining
    performance.

    **Key Features:**
    - **Scoring Network:** Computes relevance scores between queries and keys.
    - **Differentiable Top-K Masking:** Selects the top-k relevant keys for each query in a differentiable manner.
    - **Sparse Attention Computation:** Performs attention only over the selected top-k keys and values.

    **Mathematical Formulation:**

    .. math::

        scores = QK^T / \\sqrt{d_k}
        mask = topk(scores, k)
        attention\\_weights = softmax(scores \\odot mask)
        Y = attention\\_weights V

    where:
    - :math:`Q, K, V` are the query, key, and value matrices.
    - :math:`d_k` is the dimension of the key vectors.
    - :math:`topk` selects the top-k scores for each query.

    **Code Example:**

    .. code-block:: python

        sparse_attn = SparseAttention(embed_dim=256, block_loc=(0,0), kwarg_all={}, top_k=32)
        x = torch.randn(32, 128, 256)  # (batch_size, seq_len, embed_dim)
        y, Z = sparse_attn(x)

    **Args:**

        embed_dim (int): Input embedding dimension.
        block_loc (tuple): Location of the block in the architecture.
        kwarg_all (dict): Additional keyword arguments.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        top_k (int, optional): Number of top elements to keep for each query. Defaults to 32.
        dropout (float, optional): Dropout probability on attention weights. Defaults to 0.1.
        device (torch.device, optional): Device to place the module.
        dtype (torch.dtype, optional): Data type of the module.
        rotary_emb (RotaryPositionalEmbeddings, optional): Rotary positional embeddings.
        **kwargs: Additional keyword arguments.

    **Returns:**

        - Output tensor of shape (batch_size, seq_len, embed_dim).
        - Updated intermediate variables Z.

    **References:**

        - SPARSEK Attention: Efficient Sparse Attention Mechanism for Long Sequences.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        num_heads: int=8, top_k: int=32, dropout: float=0.1, device=None,
        dtype=None, rotary_emb=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, f'embed_dim {embed_dim} not divisible by num_heads {num_heads}'
        self.scale = sqrt(self.head_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False, **
            self.factory_kwargs)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False, **
            self.factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = rotary_emb
        if self.rotary_emb is not None:
            self.rotary_emb = kwarg_all.get('rotary_emb_instance', None)
            if self.rotary_emb is None:
                raise ValueError(
                    'RotaryPositionalEmbeddings instance must be provided if rotary_emb is not None.'
                    )

    def _forward(self, X, **Z):
        """
        Forward pass for SparseAttention.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            **Z: Additional intermediate variables.

        Returns:
            Tuple[torch.Tensor, dict]: Output tensor and updated intermediate variables.
        """
        batch_size, seq_len, embed_dim = X.size()
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
        if self.rotary_emb is not None:
            Q, K = self.rotary_emb(Q, K, X=X, **Z)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=X.
            device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        scores_flat = scores.view(batch_size * self.num_heads, seq_len, seq_len
            )
        topk_scores, topk_indices = scores_flat.topk(self.top_k, dim=-1)
        mask_flat = torch.zeros_like(scores_flat).scatter_(-1, topk_indices,
            1.0)
        scores_flat = scores_flat.masked_fill(mask_flat == 0, float('-inf'))
        scores = scores_flat.view(batch_size, self.num_heads, seq_len, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size,
            seq_len, embed_dim)
        Y = self.out_proj(attn_output)
        return Y, Z


import torch.nn.functional as F


class GatedMLP(GAUBase):
    """
    Gated Multi-Layer Perceptron (GatedMLP) with SwiGLU activation function.

    This module implements a Gated MLP with the SwiGLU activation function as described in
    "Scaling Language Models with Pathways" (https://arxiv.org/abs/2204.02311).

    The SwiGLU activation function is defined as:
        SwiGLU(a, b) = a * silu(b)
    where silu(x) = x * sigmoid(x).

    **Code Example:**

        rmsnorm = RMSNorm(128, (0, 6), {})
        x = torch.randn(1, 100, 128)
        gated_mlp = GatedMLP(embed_dim=128, block_loc=(0, 0), kwarg_all={}, device=x.device)
        output = gated_mlp(x)
        print(output.shape)  # torch.Size([1, 100, 128])

    **Args:**

        embed_dim (int): Dimension of the input and output embeddings.
        block_loc (tuple): Location of the block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the parent class.
        device (torch.device, optional): Device to place the module. Default is None.
        dtype (torch.dtype, optional): Data type of the module. Default is None.
        hidden_features (int, optional): Dimension of the hidden layer. If None, calculated as int(embed_dim * mlp_ratio).
        mlp_ratio (float, optional): Ratio to determine hidden_features when hidden_features is None. Default is 4.0.
        multiple_of (int, optional): Adjusts hidden_features to be a multiple of this value. Default is 256.
        **kwargs: Additional keyword arguments.

    **Attributes:**

        fc1 (nn.Linear): First linear layer projecting from embed_dim to 2 * hidden_features.
        fc2 (nn.Linear): Second linear layer projecting from hidden_features to embed_dim.

    **Shape:**

        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)

    **References:**

        - Paper: "Scaling Language Models with Pathways", Pathways Transformer Section
          https://arxiv.org/abs/2204.02311
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, hidden_features: int=None, mlp_ratio:
        float=4.0, multiple_of: int=256, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        out_features = embed_dim
        if hidden_features is None:
            hidden_features = int(embed_dim * mlp_ratio)
        hidden_features = (hidden_features + multiple_of - 1
            ) // multiple_of * multiple_of
        self.fc1 = nn.Linear(embed_dim, 2 * hidden_features, bias=False, **
            self.factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False, **
            self.factory_kwargs)

    def _forward(self, X, **Z):
        y = self.fc1(X)
        a, b = y.chunk(2, dim=-1)
        y = a * F.silu(b)
        y = self.fc2(y)
        return y


import torch.nn.functional as F


class TransitionConstantNorm(GAUBase):
    '\n    Transition-Constant Normalization (TCN).\n    \n    This normalization layer adjusts the outputs from the SSM layer before they are input into the Transformer layer,\n    mitigating discrepancies caused by differing layer characteristics. It incorporates learnable scaling (`gamma`),\n    bias (`beta`), and a transition constant (`C`) to ensure stable information flow between layers.\n    \n    **Mathematical Formulation:**\n    \n    .. math::\n    \n        Y = \\gamma \\left( \x0crac{X}{\\sqrt{\text{variance} + \\epsilon}} \right) + \x08eta + C\n    \n    where:\n    - :math:`X` is the input tensor of shape `(batch_size, seq_len, embed_dim)`.\n    - :math:`\\gamma` is the learnable scaling parameter of shape `(embed_dim,)`.\n    - :math:`\x08eta` is the learnable bias parameter of shape `(embed_dim,)`.\n    - :math:`C` is the learnable transition constant of shape `(embed_dim,)`.\n    - :math:`\\epsilon` is a small constant for numerical stability.\n    \n    **Code Example:**\n    \n    .. code-block:: python\n    \n        tcn = TransitionConstantNorm(embed_dim=256, block_loc=(0,0), kwarg_all={})\n        x = torch.randn(32, 128, 256)  # (batch_size, seq_len, embed_dim)\n        y, Z = tcn(x)\n        print(y.shape)  # torch.Size([32, 128, 256])\n    \n    **Args:**\n    \n        embed_dim (int): Input embedding dimension.\n        block_loc (tuple): Location of the block in the architecture (layer_idx, block_idx).\n        kwarg_all (dict): Additional keyword arguments.\n        device (torch.device, optional): Device to place the module.\n        dtype (torch.dtype, optional): Data type of the module.\n        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.\n        **kwargs: Additional keyword arguments.\n    \n    **Returns:**\n    \n        - Output tensor of shape `(batch_size, seq_len, embed_dim)`.\n        - Updated intermediate variables `Z`.\n    \n    **References:**\n    \n        - Huang, J., et al. (2023). "Transition-Constant Normalization for Image Enhancement." Neural Information Processing Systems.\n    '

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, eps: float=1e-05, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        """
        Initializes the TransitionConstantNorm layer.
        
        Args:
            embed_dim (int): Input embedding dimension.
            block_loc (tuple): Location of the block in the architecture.
            kwarg_all (dict): Additional keyword arguments.
            device (torch.device, optional): Device to place the module. Defaults to None.
            dtype (torch.dtype, optional): Data type of the module. Defaults to None.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
            **kwargs: Additional keyword arguments.
        """
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim, device=device,
            dtype=dtype))
        self.beta = nn.Parameter(torch.zeros(embed_dim, device=device,
            dtype=dtype))
        self.C = nn.Parameter(torch.zeros(embed_dim, device=device, dtype=
            dtype))

    def _forward(self, X: torch.Tensor, **Z):
        """
        Forward pass for TransitionConstantNorm.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            **Z: Additional intermediate variables (unused).
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as X.
            dict: Updated intermediate variables (empty in this case).
        """
        input_dtype = X.dtype
        X_float = X.to(torch.float32)
        variance = X_float.pow(2).mean(dim=-1, keepdim=True)
        X_norm = X_float * torch.rsqrt(variance + self.eps)
        Y = self.gamma * X_norm + self.beta + self.C
        Y = Y.to(input_dtype)
        return Y, Z


import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class SSMLayer(GAUBase):
    """
    State Space Model (SSM) Layer with Matrix-Valued States and Dynamic Recurrence

    This layer implements a State Space Model layer with matrix-valued states and dynamic recurrence, inspired by RWKV6.

    The SSMLayer processes input sequences using matrix-valued hidden states that capture long-range dependencies efficiently.

    Key features:
    - **Matrix-Valued States**: Multi-headed states for improved expressivity.
    - **Dynamic Recurrence**: Efficient handling of long sequences via recurrence.
    - **Selective Compression**: Compresses input sequences into recurrent hidden states, inspired by the Samba architecture.

    **Mathematical Formulation:**

    .. math::

        S[t] = S[t-1] + K_t \\odot V_t

        Y_t = \\sigma(G_t) \\odot S[t]

    where:
    - :math:`S[t]` is the state at time t
    - :math:`K_t, V_t, G_t` are projections of the input :math:`X_t`
    - :math:`\\sigma` is the gate activation function

    **Code Example:**

    .. code-block:: python

        ssmlayer = SSMLayer(embed_dim=256, block_loc=(0,0), kwarg_all={})
        x = torch.randn(32, 128, 256)  # (batch_size, seq_len, embed_dim)
        y, Z = ssmlayer(x)

    **Args:**

        embed_dim (int): Input embedding dimension.
        block_loc (tuple): Location of the block in the architecture.
        kwarg_all (dict): Additional keyword arguments.
        num_heads (int, optional): Number of heads for multi-headed states. Defaults to 8.
        gate_fn (str, optional): Activation function for gating. Defaults to 'silu'.
        proj_dim (int, optional): Dimension of the projections. Defaults to embed_dim // 2.
        norm_eps (float, optional): Epsilon value for layer normalization. Defaults to 1e-5.
        device (torch.device, optional): Device to place the module.
        dtype (torch.dtype, optional): Data type of the module.
        **kwargs: Additional keyword arguments.

    **Returns:**

        - Output tensor of shape (batch_size, seq_len, embed_dim).
        - Updated intermediate variables Z.

    **References:**

        - RWKV6: "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence" (Peng et al., 2024)
        - Samba: "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling" (Chen et al., 2024)
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        num_heads: int=8, gate_fn: str='silu', proj_dim: Optional[int]=None,
        norm_eps: float=1e-05, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gate_fn = getattr(F, gate_fn)
        self.proj_dim = proj_dim if proj_dim is not None else embed_dim // 2
        assert self.proj_dim % num_heads == 0, 'proj_dim must be divisible by num_heads'
        self.head_dim = self.proj_dim // num_heads
        self.norm = nn.LayerNorm(self.embed_dim, eps=norm_eps, **self.
            factory_kwargs)
        self.linear_kvg = nn.Linear(self.embed_dim, self.proj_dim * 3, bias
            =False, **self.factory_kwargs)
        self.output_proj = nn.Linear(self.proj_dim, self.embed_dim, bias=
            False, **self.factory_kwargs)

    def _forward(self, X, **Z):
        batch_size, seq_len, _ = X.shape
        X_norm = self.norm(X)
        kvg = self.linear_kvg(X_norm)
        K, V, G = kvg.chunk(3, dim=-1)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        G = G.view(batch_size, seq_len, self.num_heads, self.head_dim)
        S = Z.get('S', None)
        if S is None:
            S = torch.zeros(batch_size, self.num_heads, self.head_dim, **
                self.factory_kwargs)
        Y = []
        for t in range(seq_len):
            K_t = K[:, t, :, :]
            V_t = V[:, t, :, :]
            G_t = G[:, t, :, :]
            S = S + K_t * V_t
            O_t = self.gate_fn(G_t) * S
            Y.append(O_t)
        Y = torch.stack(Y, dim=1)
        Y = Y.view(batch_size, seq_len, -1)
        Y = self.output_proj(Y)
        Z_ = {'S': S}
        return Y, Z_


gab_config = {'norm_eps': 1e-05, 'proj_dim': None, 'gate_fn': 'silu',
    'num_heads': 8, 'n_heads': 8, 'mlp_ratio': 4.0, 'ssm_ratio': 2.0, 'eps':
    1e-05, 'multiple_of': 256, 'hidden_features': None, 'top_k': 32,
    'dropout': 0.1, 'rotary_emb': None, 'slice_size': 32, 'slice_dim': 2}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)