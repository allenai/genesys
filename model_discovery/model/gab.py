import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = GatedMamba(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
import torch.nn.functional as F
from typing import Optional


class GatedMamba(GAUBase):
    """
    GatedMamba: Root unit that implements hierarchical processing with gated linear attention
    and state space modeling.

    **Architecture:**
    1. Input normalization using RMSNorm.
    2. Multiple GatedHierarchicalLayers with dense connections.
    3. Output projection to embedding dimension.

    **Args:**
        embed_dim (int): Embedding dimension of the inputs and outputs.
        block_loc (tuple): Location in the model (layer index, total layers).
        kwarg_all (dict): Additional keyword arguments for child modules.
        num_layers (int, optional): Number of hierarchical layers. Default is 3.
        device (torch.device, optional): Device for computation.
        dtype (torch.dtype, optional): Data type for computation.

    **Inputs:**
        - **X**: Input tensor of shape (batch_size, seq_len, embed_dim).
        - **Z**: Dictionary of intermediate variables.

    **Outputs:**
        - **Y**: Output tensor of same shape as **X**.
        - **Z**: Updated dictionary of intermediate variables.

    **Example:**

        >>> gated_mamba = GatedMamba(embed_dim=512, block_loc=(0, 12), kwarg_all={}, num_layers=3)
        >>> X = torch.randn(2, 1024, 512)
        >>> Y, Z = gated_mamba(X)

    **Note:**
        This class serves as the root unit in the GatedMamba architecture, orchestrating the hierarchical layers and managing inputs and outputs.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        num_layers: int=3, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        kwarg_all = kwarg_all.copy()
        kwarg_all.pop('layer_idx', None)
        self.norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        assert num_layers == 3, 'This implementation supports num_layers=3'
        self.layer0 = GatedHierarchicalLayer(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.layer1 = GatedHierarchicalLayer(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.layer2 = GatedHierarchicalLayer(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.input_proj1 = nn.Linear(embed_dim * 2, embed_dim, bias=False,
            **self.factory_kwargs)
        nn.init.xavier_uniform_(self.input_proj1.weight)
        self.input_proj2 = nn.Linear(embed_dim * 3, embed_dim, bias=False,
            **self.factory_kwargs)
        nn.init.xavier_uniform_(self.input_proj2.weight)
        self.output_proj = nn.Linear(embed_dim * num_layers, embed_dim,
            bias=False, **self.factory_kwargs)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def _forward(self, X, **Z):
        X_normed, Z = self.norm(X, **Z)
        u0 = X_normed
        h0, Z = self.layer0(u0, **Z)
        u1 = torch.cat([X_normed, h0], dim=-1)
        u1 = self.input_proj1(u1)
        h1, Z = self.layer1(u1, **Z)
        u2 = torch.cat([X_normed, h0, h1], dim=-1)
        u2 = self.input_proj2(u2)
        h2, Z = self.layer2(u2, **Z)
        concatenated_outputs = torch.cat([h0, h1, h2], dim=-1)
        Y = self.output_proj(concatenated_outputs)
        return Y, Z


import torch.nn.functional as F


class GatedHierarchicalLayer(GAUBase):
    """
    GatedHierarchicalLayer: Implements a single layer of hierarchical processing with:
        - Gated Linear Attention mechanism
        - State expansion
        - Dense connection handling

    **Args:**
        embed_dim (int): Embedding dimension of the inputs and outputs.
        block_loc (tuple): Location in the model (layer index, total layers).
        kwarg_all (dict): Additional keyword arguments for child modules.
        layer_idx (int): Index of this layer in the hierarchy.
        device (torch.device, optional): Device for computation.
        dtype (torch.dtype, optional): Data type for computation.

    **Inputs:**
        - **X**: Input tensor of shape (batch_size, seq_len, embed_dim).
        - **Z**: Dictionary of intermediate variables.

    **Outputs:**
        - **Y**: Output tensor of shape (batch_size, seq_len, embed_dim).
        - **Z**: Updated dictionary of intermediate variables.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        layer_idx: int, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.layer_idx = layer_idx
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
        self.state_expansion = nn.Linear(embed_dim, embed_dim, bias=False,
            **self.factory_kwargs)
        nn.init.xavier_uniform_(self.state_expansion.weight)
        self.norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)

    def _forward(self, X, **Z):
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        G_Q = torch.sigmoid(self.gate_Q(X))
        G_K = torch.sigmoid(self.gate_K(X))
        Q = Q * G_Q
        K = K * G_K
        Q_prime = F.elu(Q) + 1
        K_prime = F.elu(K) + 1
        attention_scores = torch.bmm(Q_prime, K_prime.transpose(1, 2))
        scaling_factor = 1.0 / self.embed_dim ** 0.5
        attention_scores = attention_scores * scaling_factor
        seq_len = X.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=X.
            device), diagonal=1).bool()
        attention_scores = attention_scores.masked_fill(causal_mask.
            unsqueeze(0), float('-inf'))
        attention_probs = F.softmax(attention_scores, dim=-1)
        H = torch.bmm(attention_probs, V)
        S = self.state_expansion(H)
        output = H + S
        Y, Z = self.norm(output, **Z)
        return Y, Z


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
        eps (float, optional): A small constant added to the denominator for numerical stability.
            Default: 1e-8.
        device (torch.device, optional): The device on which to allocate the module's parameters.
        dtype (torch.dtype, optional): The dtype of the module's parameters.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter of shape (embed_dim,).
        eps (float): The epsilon value used in the normalization formula.

    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim) (same shape as input)
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        eps: float=1e-08, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.weight = nn.Parameter(torch.ones(embed_dim, **self.factory_kwargs)
            )
        self.eps = eps

    def _forward(self, X, **Z):
        norm = torch.sqrt(torch.mean(X * X, dim=-1, keepdim=True) + self.eps)
        Y = X / norm * self.weight
        return Y, Z


gab_config = {'layer_idx': None, 'num_layers': 3, 'eps': 1e-08}



autoconfig = {
    'd_model': 128,
    'n_block': 5
}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)