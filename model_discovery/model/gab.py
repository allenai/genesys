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
        self.seq_modeling_block = LightningTTTLinear(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)
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


class LightningTTTLinear(GAUBase):
    """
    Enhanced TTTLinear with Lightning Attention for efficient long-context modeling.
    
    This unit improves upon TTTLinear by:
    1. Integrating Lightning Attention for efficient processing of long sequences
    2. Using block-based attention computation (intra-block and inter-block)
    3. Adding gating mechanisms for better expressiveness
    4. Maintaining test-time training capabilities
    
    Args:
        embed_dim (int): The embedding dimension
        block_loc (tuple): Location of block in model (layer_idx, block_idx)
        kwarg_all (dict): Additional arguments
        device (torch.device, optional): Device to place tensors
        dtype (torch.dtype, optional): Data type of tensors
        num_attention_heads (int, optional): Number of attention heads. Default: 4
        block_size (int, optional): Size of attention blocks. Default: 128
        mini_batch_size (int, optional): Size of mini-batches for TTT. Default: 16
        rope_theta (float, optional): Base for rotary embeddings. Default: 10000.0
        ttt_base_lr (float, optional): Base learning rate for TTT. Default: 1.0
        dropout (float, optional): Dropout probability. Default: 0.0
        
    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_attention_heads=4, block_size=128,
        mini_batch_size=16, rope_theta=10000.0, ttt_base_lr=1.0, dropout=
        0.0, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.num_heads = num_attention_heads
        self.head_dim = embed_dim // num_attention_heads
        self.block_size = block_size
        self.mini_batch_size = mini_batch_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.ttt_base_lr = ttt_base_lr
        self.learnable_token_idx = nn.Parameter(torch.zeros(mini_batch_size,
            **self.factory_kwargs))
        self.W = nn.Parameter(torch.randn(num_attention_heads, self.
            head_dim, self.head_dim, **self.factory_kwargs) * 0.02)
        self.b = nn.Parameter(torch.zeros(num_attention_heads, 1, self.
            head_dim, **self.factory_kwargs))
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=True, **self.
            factory_kwargs)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-06, **self.factory_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-06, **self.factory_kwargs)
        kwargs['dim'] = self.head_dim
        kwargs['max_position_embeddings'] = block_size
        kwargs['base'] = rope_theta
        self.rotary_emb = RotaryEmbedding(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        token_idx = 1.0 / torch.arange(1, mini_batch_size + 1, **self.
            factory_kwargs)
        self.register_buffer('token_idx', token_idx, persistent=False)
        self.attention_scale = nn.Parameter(torch.ones(1, **self.
            factory_kwargs))

    def _forward(self, X, position_ids=None, **Z):
        B, L, D = X.shape
        if position_ids is None:
            position_ids = torch.arange(0, L, dtype=torch.long, device=X.device
                ).unsqueeze(0)
        X_norm = self.norm1(X)
        Q = self.q_proj(X_norm).view(B, L, self.num_heads, self.head_dim
            ).transpose(1, 2)
        K = self.k_proj(X_norm).view(B, L, self.num_heads, self.head_dim
            ).transpose(1, 2)
        V = self.v_proj(X_norm).view(B, L, self.num_heads, self.head_dim
            ).transpose(1, 2)
        K_transformed = torch.einsum('bhnd,hde->bhne', K, self.W) + self.b
        Z['position_ids'] = position_ids % self.block_size
        Z['input'] = V
        _, Z = self.rotary_emb(X, **Z)
        cos, sin = Z['cos'], Z['sin']
        Q, K_transformed = self.apply_rotary_pos_emb(Q, K_transformed, cos, sin
            )
        num_blocks = (L + self.block_size - 1) // self.block_size
        output = torch.zeros_like(Q)
        base_token_weights = F.softplus(self.learnable_token_idx)
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, L)
            block_size = end_idx - start_idx
            Q_block = Q[:, :, start_idx:end_idx]
            K_block = K_transformed[:, :, start_idx:end_idx]
            V_block = V[:, :, start_idx:end_idx]
            scores = torch.matmul(Q_block, K_block.transpose(-2, -1))
            scores = scores * self.attention_scale / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.ones(block_size, block_size,
                device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
            if block_size <= self.mini_batch_size:
                block_weights = base_token_weights[:block_size]
            else:
                block_weights = F.interpolate(base_token_weights.view(1, 1,
                    -1), size=block_size, mode='linear', align_corners=False
                    ).squeeze(0).squeeze(0)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = attn_weights * block_weights.view(1, 1, -1, 1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training
                =self.training)
            block_output = torch.matmul(attn_weights, V_block)
            if i > 0:
                K_prev = K_transformed[:, :, :start_idx]
                V_prev = V[:, :, :start_idx]
                KV_global = torch.einsum('bhnd,bhne->bhde', K_prev, V_prev)
                global_output = torch.einsum('bhmd,bhde->bhme', Q_block,
                    KV_global)
                block_output = block_output + global_output
            output[:, :, start_idx:end_idx] = block_output
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        gate = torch.sigmoid(self.gate(X_norm))
        output = gate * output
        output = self.norm2(output)
        output = self.o_proj(output)
        return output

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary positional embeddings to queries and keys."""

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)
        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin
        return q_embed, k_embed


import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.utils import logging


class RotaryEmbedding(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, dim=None, max_position_embeddings=16, base
        =10000, scaling_factor=1.0, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.scaling_factor = scaling_factor
        self.dim = dim if dim is not None else embed_dim // 4
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2, dtype=
            torch.int64).float().to(device) / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @torch.no_grad()
    def _forward(self, X, input, position_ids, **Z):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = input.device.type
        device_type = device_type if isinstance(device_type, str
            ) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        Z['cos'] = cos.to(**self.factory_kwargs)
        Z['sin'] = sin.to(**self.factory_kwargs)
        return X, Z


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


gab_config = {'intermediate_size': None, 'scaling_factor': 1.0, 'dim': None,
    'base': 10000, 'max_position_embeddings': 16, 'eps': 1e-05,
    'conv_kernel': 4, 'rms_norm_eps': 1e-06, 'rope_theta': 10000.0,
    'mini_batch_size': 16, 'num_attention_heads': 4, 'dropout': 0.0,
    'ttt_base_lr': 1.0, 'block_size': 128}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)