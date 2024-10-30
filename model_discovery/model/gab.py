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
import math
try:
    from flash_attn import flash_attention_impl
    HAS_FLASH_ATTENTION = True
except ImportError:
    HAS_FLASH_ATTENTION = False


class FastTTTLinear(GAUBase):
    """
    FastTTTLinear with enhanced causality, memory efficiency, and performance optimizations.
    
    Key Features:
    - Causal attention with efficient chunked computation
    - Memory-efficient implementation with gradient checkpointing
    - Optional Flash Attention support for faster computation
    - Adaptive chunk sizing based on sequence length
    - Enhanced numerical stability through proper scaling and normalization
    
    Performance Guidelines:
    - Recommended maximum sequence length: 32K
    - Optimal chunk size: 1024 for 16GB GPU
    - Memory usage: O(N) where N is sequence length
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_attention_heads=4, dropout=0.0,
        attention_dropout=0.0, chunk_size=1024, max_position_embeddings=
        32768, layer_norm_eps=1e-05, use_flash_attention=True, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.num_heads = num_attention_heads
        self.head_dim = embed_dim // num_attention_heads
        assert embed_dim % num_attention_heads == 0, 'embed_dim must be divisible by num_attention_heads'
        self.embed_dim = embed_dim
        self.base_chunk_size = chunk_size
        self.chunk_size = chunk_size
        self.max_position_embeddings = max_position_embeddings
        self.use_flash_attention = use_flash_attention and HAS_FLASH_ATTENTION
        self.scale = 1.0 / math.sqrt(self.head_dim)
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
        self.norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.dropout = nn.Dropout(p=dropout)
        self.attention_dropout = nn.Dropout(p=attention_dropout)
        self.local_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3,
            padding=0, groups=embed_dim, bias=True, **self.factory_kwargs)
        self._init_weights()
        self.gradient_checkpointing = False

    def _init_weights(self):
        """Initialize weights with proper scaling for stability."""
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.W_Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_K.weight, gain=gain)
        nn.init.xavier_uniform_(self.W_V.weight, gain=gain)
        nn.init.xavier_uniform_(self.gate_Q.weight, gain=gain)
        nn.init.zeros_(self.gate_Q.bias)
        nn.init.xavier_uniform_(self.gate_K.weight, gain=gain)
        nn.init.zeros_(self.gate_K.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.xavier_uniform_(self.local_conv.weight)
        nn.init.zeros_(self.local_conv.bias)

    def _efficient_attention(self, Q, K, V, mask):
        """Efficient attention computation."""
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        return torch.matmul(attn_weights, V)

    def _causal_attention(self, Q, K, V, chunk_size):
        """Compute chunked causal attention with optional Flash Attention."""
        B, H, L, D = Q.shape
        if self.use_flash_attention and not self.training:
            return flash_attention_impl(Q, K, V, causal=True)
        outputs = []
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            Q_chunk = Q[:, :, chunk_start:chunk_end]
            K_chunk = K[:, :, :chunk_end]
            V_chunk = V[:, :, :chunk_end]
            causal_mask = torch.triu(torch.ones(chunk_end - chunk_start,
                chunk_end, device=Q.device, dtype=torch.bool), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            chunk_output = self._efficient_attention(Q_chunk * self.scale,
                K_chunk, V_chunk, causal_mask)
            outputs.append(chunk_output)
        return torch.cat(outputs, dim=2)

    def _forward_impl(self, X, **Z):
        """Main implementation of forward pass with all optimizations."""
        B, L, D = X.size()
        H = self.num_heads
        D_H = self.head_dim
        self.chunk_size = min(self.base_chunk_size, max(128, L // 8))
        X_pad = F.pad(X.transpose(1, 2), (2, 0), mode='replicate')
        X_conv = self.local_conv(X_pad)
        X_conv = X_conv.transpose(1, 2)
        X = X + self.dropout(X_conv)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        G_Q = torch.sigmoid(self.gate_Q(X))
        G_K = torch.sigmoid(self.gate_K(X))
        Q = Q * G_Q
        K = K * G_K
        Q = Q.view(B, L, H, D_H).transpose(1, 2)
        K = K.view(B, L, H, D_H).transpose(1, 2)
        V = V.view(B, L, H, D_H).transpose(1, 2)
        attn_output = self._causal_attention(Q, K, V, self.chunk_size)
        output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.output_proj(output)
        output = X + 0.1 * self.dropout(output)
        output, Z = self.norm(output, **Z)
        return output, Z

    def _forward(self, X, **Z):
        """Forward pass with optional gradient checkpointing."""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, X,
                *Z.values())
        return self._forward_impl(X, **Z)


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
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils._pytree import tree_map
from transformers.utils import logging
from transformers.activations import ACT2FN
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except:
    causal_conv1d_update, causal_conv1d_fn = None, None


class Conv(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, conv_kernel=4, rms_norm_eps=1e-06, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        kwarg_all['eps'] = rms_norm_eps
        self.norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.conv = nn.Conv1d(embed_dim, embed_dim, bias=True, kernel_size=
            conv_kernel, groups=embed_dim, padding=conv_kernel - 1, **self.
            factory_kwargs)

    def __call__(self, X, **Z):
        hidden_states = X
        seq_len = hidden_states.shape[1]
        hidden_states = self.norm(hidden_states, **Z)[0]
        hidden_states = hidden_states.transpose(1, 2)
        if causal_conv1d_fn is None:
            hidden_states = self.conv(hidden_states)[..., :seq_len]
        else:
            conv_weights = self.conv.weight.view(self.conv.weight.size(0),
                self.conv.weight.size(2))
            hidden_states = causal_conv1d_fn(hidden_states, conv_weights,
                self.conv.bias, activation=None)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


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


gab_config = {'eps': 1e-05, 'attention_dropout': 0.0, 'num_attention_heads':
    4, 'dropout': 0.0, 'layer_norm_eps': 1e-05, 'use_flash_attention': True,
    'max_position_embeddings': 32768, 'chunk_size': 1024, 'conv_kernel': 4,
    'rms_norm_eps': 1e-06, 'intermediate_size': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)