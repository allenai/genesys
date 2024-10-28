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
        self.seq_modeling_block = GatedTTTLinear(embed_dim=self.embed_dim,
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
from typing import Tuple, Dict


class GatedTTTLinear(GAUBase):
    """
    GatedTTTLinear GAU.

    This GAU enhances the existing TTTLinear GAU by integrating gating mechanisms and 
    graph-inspired convolutional operations. It allows the model to dynamically adapt 
    during test-time training by capturing both local and global dependencies efficiently.

    **Code Example:**

    ```python
        from gatedtttlinear import GatedTTTLinear

    embed_dim = 128
    block_loc = (0, 0)
    gau = GatedTTTLinear(embed_dim=embed_dim, block_loc=block_loc, kwarg_all={})

    X = torch.randn(2, 50, embed_dim)  # Batch size 2, sequence length 50
    Z = {}
    Y, Z = gau(X, **Z)
    print(Y.shape)  # Should output: torch.Size([2, 50, 128])
    ```

    **Todo:**
        * Implement dynamic adjacency matrix computation based on input embeddings.
        * Ensure computational efficiency for scalability.

    Args:
        embed_dim (int): The size of the input feature dimension.
        block_loc (tuple): The location of this GAU within the network.
        kwarg_all (dict): Dictionary of all keyword arguments for initializing child GAUs.
        device (torch.device, optional): Device to allocate the GAU's parameters.
        dtype (torch.dtype, optional): Data type of the GAU's parameters.

    Returns:
        Output embeddings Y and updated intermediate variables Z.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=True, **self.
            factory_kwargs)
        self.activation = nn.Sigmoid()
        self.linear_proj = nn.Linear(embed_dim, embed_dim, bias=False, **
            self.factory_kwargs)
        self.graph_conv = GraphConvolution(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)

    def _forward(self, X: torch.Tensor, **Z) ->Tuple[torch.Tensor, Dict]:
        """
        Forward pass of GatedTTTLinear GAU.

        Args:
            X (torch.Tensor): Input embeddings of shape (B, L, D).
            **Z: Intermediate variables.

        Returns:
            Tuple containing output embeddings Y and updated intermediate variables Z.
        """
        G = self.activation(self.gate_proj(X))
        Y_gated = G * self.linear_proj(X)
        Y_graph, Z = self.graph_conv(Y_gated, **Z)
        return Y_graph, Z


import torch.nn.functional as F
from typing import Tuple, Dict


class GraphConvolution(GAUBase):
    """
    Graph Convolutional GAU.

    This GAU performs graph convolution to capture global dependencies within the input sequence.
    It leverages multi-head attention to compute a dynamic adjacency matrix based on input embeddings
    and applies attention-based transformations to integrate global contextual information.

    Args:
        embed_dim (int): The size of the input and output feature dimensions.
        block_loc (tuple): The location of this GAU within the network.
        kwarg_all (dict): Dictionary of all keyword arguments for initializing child GAUs.
        device (torch.device, optional): Device to allocate the GAU's parameters.
        dtype (torch.dtype, optional): Data type of the GAU's parameters.
        num_heads (int, optional): Number of attention heads. Defaults to 4.
        dropout (float, optional): Dropout probability on attention weights. Defaults to 0.1.

    Returns:
        Output embeddings Y and updated intermediate variables Z.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_heads: int=4, dropout: float=0.1, **kwargs
        ):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f'embed_dim {embed_dim} must be divisible by num_heads {num_heads}'
                )
        self.scale = self.head_dim ** -0.5
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=
            num_heads, dropout=dropout, batch_first=True, **self.factory_kwargs
            )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-05, **self.
            factory_kwargs)

    def _forward(self, X: torch.Tensor, **Z) ->Tuple[torch.Tensor, Dict]:
        """
        Forward pass of GraphConvolution GAU.

        Args:
            X (torch.Tensor): Input embeddings of shape (B, L, D).
            **Z: Intermediate variables.

        Returns:
            Tuple containing output embeddings Y and updated intermediate variables Z.
        """
        residual = X
        X = self.layer_norm(X)
        B, L, _ = X.shape
        causal_mask = torch.triu(torch.ones(L, L, device=X.device, dtype=
            torch.bool), diagonal=1)
        attn_output, _ = self.attn(X, X, X, need_weights=False, attn_mask=
            causal_mask)
        attn_output = self.dropout_layer(attn_output)
        Y = self.out_proj(attn_output)
        Y = Y + residual
        return Y, Z


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


gab_config = {'conv_kernel': 4, 'rms_norm_eps': 1e-06, 'scaling_factor': 
    1.0, 'dim': None, 'base': 10000, 'max_position_embeddings': 16, 'eps': 
    1e-05, 'dropout': 0.1, 'num_heads': 4, 'intermediate_size': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)