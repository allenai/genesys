import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = TTT(embed_dim=embed_dim, block_loc=block_loc, kwarg_all
            =kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


import torch.nn.functional as F
from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.utils import logging


class TTT(GAUBase):
    """
    Problem Statement
This paper addresses the challenge of long context in recurrent neural networks (RNNs). While RNNs offer linear computational complexity, their performance suffers in long sequences due to the limited expressive power of their fixed-size hidden states. This limitation contrasts with Transformers, which excel in long-context scenarios but have quadratic complexity.

Main Claims
The paper proposes a new class of sequence modeling layers called Test-Time Training (TTT) layers that offer both linear complexity and expressive hidden states.
The key idea is to make the hidden state a machine learning model itself, where the update rule is a step of self-supervised learning. This allows for continuous training of the hidden state even on test sequences.
The paper introduces two instantiations of TTT layers: TTT-Linear, with a linear model as the hidden state, and TTT-MLP, with a two-layer multi-layer perceptron (MLP) as the hidden state.
Both TTT-Linear and TTT-MLP demonstrate competitive performance compared to strong Transformer and Mamba (a modern RNN) baselines across various model sizes.
Unlike Mamba, both TTT layers show a continuous decrease in perplexity as they condition on more tokens in long sequences.
TTT-Linear, with preliminary systems optimization, is faster than Transformers at 8k context and matches Mamba in wall-clock time.
Methodology
The paper introduces TTT layers, which use a self-supervised learning approach to update the hidden state. The update rule is effectively a gradient step on a self-supervised loss function, allowing for "training" of the hidden state at test time. Two implementations are explored: TTT-Linear, where the hidden state is a linear model, and TTT-MLP, where the hidden state is a two-layer MLP. The paper also proposes mini-batch TTT and a dual form to improve hardware efficiency and speed up computations.

Key Results
In short-context (2k and 8k tokens) experiments on the Pile dataset, both TTT-Linear and TTT-MLP demonstrate performance comparable to or exceeding Mamba and Transformer baselines.
In long-context (1k to 32k tokens) experiments on the Books3 subset of the Pile, both TTT-Linear and TTT-MLP outperform Mamba, especially at longer context lengths.
TTT-Linear with the Mamba backbone outperforms both Mamba and Transformers with the Transformer backbone across various model sizes.
With preliminary systems optimization, TTT-Linear is already faster than Transformers at 8k context and matches Mamba in wall-clock time.
TTT-MLP shows potential for even better performance in long-context scenarios but currently faces challenges in memory I/O.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.hidden_size = embed_dim
        kwarg_all['num_attention_heads'] = max(4, embed_dim // 64)
        self.seq_modeling_block = FastTTTLinear(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        kwarg_all['intermediate_size'] = int(embed_dim * 2.5)
        self.mlp = SwiGluMLP(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.conv = Conv(embed_dim=self.embed_dim, block_loc=self.block_loc,
            kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.seq_norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.ffn_norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)

    def _forward(self, X, **Z):
        hidden_states = X
        position_ids = torch.arange(0, X.shape[1], dtype=torch.long, device
            =X.device).unsqueeze(0)
        residual = hidden_states
        hidden_states = self.conv(hidden_states, **Z)[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.seq_norm(hidden_states, **Z)[0]
        Z['position_ids'] = position_ids
        hidden_states = self.seq_modeling_block(hidden_states, **Z)[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states, **Z)[0]
        hidden_states = self.mlp(hidden_states, **Z)[0]
        hidden_states = residual + hidden_states
        return hidden_states


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
from transformers.utils import logging


class FastTTTLinear(GAUBase):
    """
    **FastTTTLinear**

    FastTTTLinear is a modified version of TTTLinear that integrates Gated Linear Attention (GLA)
    and concepts from the RWKV architecture to enhance computational efficiency for long sequences.
    This implementation addresses inefficiency concerns by vectorizing operations, eliminating
    Python-level for-loops, and optimizing tensor computations.

    **Key Features:**

    - **Gated Linear Attention**: Uses data-dependent gates to modulate queries and keys, enabling linear attention computation.
    - **Vectorized Computations**: Eliminates Python for-loops by using efficient tensor operations.
    - **Normalization**: Applies LayerNorm to queries and keys to stabilize computations.
    - **Adjustments for Numerical Stability**: Uses appropriate scaling, activation functions, and safeguards.
    - **Local Convolutional Augmentation**: Applies causal convolution to prevent information leakage and enhance local context.

    **Args:**
        embed_dim (int): Embedding dimension.
        block_loc (tuple): Location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): Device on which to allocate tensors.
        dtype (torch.dtype, optional): Data type of the tensors.
        num_attention_heads (int, optional): Number of attention heads. Default: 4.

    **Inputs:**
        - **X**: Input tensor of shape (batch_size, seq_len, embed_dim).

    **Outputs:**
        - **Y**: Output tensor of shape (batch_size, seq_len, embed_dim).

    **Example:**

        >>> fast_ttt_linear = FastTTTLinear(embed_dim=512, block_loc=(0, 0), kwarg_all={})
        >>> X = torch.randn(2, 1024, 512)
        >>> Y, Z = fast_ttt_linear(X)

    **References:**

    - Yang, S., et al. (2023). *Gated Linear Attention Transformers with Hardware-Efficient Training*.
    - Peng, B., et al. (2023). *RWKV: Reinventing RNNs for the Transformer Era*.

    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_attention_heads=4, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.num_heads = num_attention_heads
        assert embed_dim % self.num_heads == 0, 'embed_dim must be divisible by num_attention_heads'
        self.head_dim = embed_dim // self.num_heads
        self.embed_dim = embed_dim
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.gate_Q = nn.Linear(embed_dim, embed_dim, bias=True, **self.
            factory_kwargs)
        self.gate_K = nn.Linear(embed_dim, embed_dim, bias=True, **self.
            factory_kwargs)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False, **
            self.factory_kwargs)
        self.local_conv = nn.Conv1d(in_channels=embed_dim, out_channels=
            embed_dim, kernel_size=3, padding=2, bias=True, **self.
            factory_kwargs)
        self.norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.q_norm = nn.LayerNorm(embed_dim, eps=1e-05, **self.factory_kwargs)
        self.k_norm = nn.LayerNorm(embed_dim, eps=1e-05, **self.factory_kwargs)
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.xavier_uniform_(self.gate_Q.weight)
        nn.init.zeros_(self.gate_Q.bias)
        nn.init.xavier_uniform_(self.gate_K.weight)
        nn.init.zeros_(self.gate_K.bias)
        nn.init.xavier_uniform_(self.local_conv.weight)
        nn.init.zeros_(self.local_conv.bias)

    def _forward(self, X, **Z):
        B, L, D = X.size()
        H = self.num_heads
        D_H = self.head_dim
        X_conv = self.local_conv(X.transpose(1, 2))
        X_conv = X_conv.transpose(1, 2)[:, :L, :]
        X = X + X_conv
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        G_Q = torch.sigmoid(self.gate_Q(X))
        G_K = torch.sigmoid(self.gate_K(X))
        Q = Q * G_Q
        K = K * G_K
        Q = Q.view(B, L, H, D_H).transpose(1, 2)
        K = K.view(B, L, H, D_H).transpose(1, 2)
        V = V.view(B, L, H, D_H).transpose(1, 2)
        Q_prime = F.elu(Q) + 1
        K_prime = F.elu(K) + 1
        QV = Q_prime * V
        K_cumsum = K_prime.cumsum(dim=2)
        QV_cumsum = (K_prime * V).cumsum(dim=2)
        denominator = torch.einsum('bhlf,bhlf->bhl', Q_prime, K_cumsum)
        numerator = torch.einsum('bhlf,bhlf->bhlf', Q_prime, QV_cumsum)
        epsilon = 1e-06
        denominator = denominator.unsqueeze(-1) + epsilon
        output = numerator / denominator
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.output_proj(output)
        output = X + output
        output, Z = self.norm(output, **Z)
        return output, Z


import torch.nn.functional as F


class Conv(GAUBase):
    """
    Dynamic Gated Convolutional Layer with RMSNorm.
    
    This layer implements a dynamic and gated convolution mechanism that enhances
    local context modeling while maintaining computational efficiency. It includes:
    
    1. RMSNorm for input normalization
    2. Dynamic kernel generation based on input content
    3. Gated convolution mechanism
    4. Causal padding to prevent information leakage
    
    Args:
        embed_dim (int): The embedding dimension
        block_loc (tuple): Location of this block in the model (layer_idx, block_idx)
        kwarg_all (dict): Additional keyword arguments
        device (torch.device, optional): Device to place the module on
        dtype (torch.dtype, optional): Data type of the module's parameters
        conv_kernel (int, optional): Size of the convolutional kernel. Default: 4
        rms_norm_eps (float, optional): Epsilon for RMSNorm stability. Default: 1e-6
        kernel_reduction_factor (int, optional): Reduction factor for kernel generator hidden dim. Default: 2
        use_cache (bool, optional): Whether to cache dynamic weights during inference. Default: True
        
    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
        
    Examples:
        >>> conv = Conv(embed_dim=512, block_loc=(0,0), kwarg_all={})
        >>> x = torch.randn(2, 128, 512)
        >>> y, z = conv(x)
        >>> print(y.shape)
        torch.Size([2, 128, 512])
        
    Note:
        The implementation uses several optimizations:
        1. Weight caching during inference
        2. Fused operations for dynamic convolution
        3. Efficient memory management for long sequences
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, conv_kernel=4, rms_norm_eps=1e-06,
        kernel_reduction_factor=2, use_cache=True, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        kwarg_all['eps'] = rms_norm_eps
        self.norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=conv_kernel,
            groups=embed_dim, padding=conv_kernel - 1, bias=True, **self.
            factory_kwargs)
        kernel_hidden_dim = embed_dim // kernel_reduction_factor
        self.kernel_gen = nn.Sequential(nn.Linear(embed_dim,
            kernel_hidden_dim, **self.factory_kwargs), nn.GELU(), nn.Linear
            (kernel_hidden_dim, conv_kernel, **self.factory_kwargs))
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim, **self.
            factory_kwargs), nn.Sigmoid())
        self.use_cache = use_cache
        self.weight_cache = None
        self.conv_kernel = conv_kernel
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform and zero biases."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        for module in self.kernel_gen.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_dynamic_weights(self, hidden_states):
        """Compute dynamic convolution weights with optional caching."""
        if (not self.training and self.use_cache and self.weight_cache is not
            None):
            return self.weight_cache
        avg_seq = hidden_states.mean(dim=1)
        dynamic_weights = self.kernel_gen(avg_seq)
        dynamic_weights = F.softmax(dynamic_weights, dim=-1)
        dynamic_weights = dynamic_weights.unsqueeze(1).unsqueeze(1)
        if not self.training and self.use_cache:
            self.weight_cache = dynamic_weights
        return dynamic_weights

    def _forward(self, X, **Z):
        """
        Forward pass of the Conv GAU.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            Z (dict): Additional inputs passed as keyword arguments
            
        Returns:
            tuple: (output tensor, updated Z dictionary)
        """
        seq_len = X.shape[1]
        hidden_states = self.norm(X, **Z)[0]
        dynamic_weights = self._compute_dynamic_weights(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        conv_out = self.conv(hidden_states)
        conv_out = conv_out.unsqueeze(-1) * dynamic_weights
        conv_out = conv_out.sum(dim=-1)[..., :seq_len]
        hidden_states = conv_out.transpose(1, 2)
        gate_values = self.gate(X)
        hidden_states = hidden_states * gate_values
        return hidden_states, Z

    def reset_cache(self):
        """Reset the weight cache."""
        self.weight_cache = None


import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.utils import logging
from transformers.activations import ACT2FN


class SwiGluMLP(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, intermediate_size=None, **kwargs):
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
        down_proj = self.down_proj(self.act_fn(self.gate_proj(X)) * self.
            up_proj(X))
        return down_proj


gab_config = {'eps': 1e-05, 'num_attention_heads': 4, 'conv_kernel': 4,
    'rms_norm_eps': 1e-06, 'use_cache': True, 'kernel_reduction_factor': 2,
    'intermediate_size': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)