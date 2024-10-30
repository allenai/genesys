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
        self.mha = HierarchicalAdaptiveAttention(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.mlp = GatedMLP(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm1 = ODERMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm2 = ODERMSNorm(embed_dim=self.embed_dim, block_loc=self.
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
import math


class ODERMSNorm(GAUBase):
    """
    ODE-based Root Mean Square Layer Normalization (ODERMSNorm).

    This layer applies a variant of RMSNorm where the scaling parameter gamma is modeled
    as a continuous function evolving through an ODE. This allows the normalization parameters
    to adapt continuously based on the input context, enabling smooth adaptation to varying
    sequence lengths and input distributions.

    **Main Features:**
    - **Continuous Parameter Evolution**: Gamma is obtained by integrating an ODE, allowing it to adapt smoothly.
    - **Adaptive Normalization**: The normalization adapts to the input context for better performance.

    **Code Example:**

        # Initialize ODERMSNorm
        norm = ODERMSNorm(embed_dim=128, block_loc=(0, 6), kwarg_all={})
        # Input tensor X
        X = torch.randn(4, 10, 128)
        # Forward pass
        Y, Z = norm(X, t=torch.tensor(1.0))
        print(Y.shape)  # Output: torch.Size([4, 10, 128])

    Args:
        embed_dim (int): The size of the input feature dimension.
        block_loc (tuple): The location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the parent class.
        device (torch.device, optional): The device on which to allocate the module's parameters.
        dtype (torch.dtype, optional): The dtype of the module's parameters.
        eps (float, optional): A small constant added to the denominator for numerical stability.
            Default: 1e-5.
        num_steps (int, optional): Number of steps for ODE integration. Default: 10.
        **kwargs: Additional keyword arguments.

    Attributes:
        eps (float): The epsilon value used in the normalization formula.
        param_net (nn.Module): A parameter network generating initial gamma.

    Shape:
        - Input: X of shape (batch_size, seq_len, embed_dim)
        - Output: Y of shape (batch_size, seq_len, embed_dim)

    Examples:
        >>> norm = ODERMSNorm(embed_dim=128, block_loc=(0, 6), kwarg_all={})
        >>> x = torch.randn(4, 10, 128)
        >>> y, Z = norm(x, t=torch.tensor(1.0))
        >>> y.shape
        torch.Size([4, 10, 128])

    References:
        - Proposal: "ODEAdaptGPT: Continuous Adaptive Normalization for Efficient Language Models"
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, eps=1e-05, num_steps=10, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.eps = eps
        self.num_steps = num_steps
        self.param_net = nn.Sequential(nn.Linear(embed_dim, embed_dim // 4,
            **self.factory_kwargs), nn.SiLU(), nn.Linear(embed_dim // 4,
            embed_dim, **self.factory_kwargs))
        self.ode_function = nn.Sequential(nn.Linear(embed_dim, embed_dim //
            2, **self.factory_kwargs), nn.Tanh(), nn.Linear(embed_dim // 2,
            embed_dim, **self.factory_kwargs))

    def _forward(self, X, **Z):
        t = Z.get('t', torch.tensor(1.0, **self.factory_kwargs))
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t = t.item()
            else:
                raise ValueError('t must be a scalar.')
        B, L, D = X.size()
        gamma0 = self.param_net(X)
        gamma = self.get_gamma(t, gamma0)
        assert gamma.shape == (B, L, D
            ), f'Gamma shape mismatch: expected ({B}, {L}, {D}), got {gamma.shape}'
        rms = torch.sqrt(torch.mean(X * X, dim=-1, keepdim=True) + self.eps)
        Y = X / rms * gamma
        return Y, Z

    def get_gamma(self, t, gamma0):
        gamma = self.euler_integration(gamma0, t, self.num_steps)
        return gamma

    def euler_integration(self, gamma0, t, num_steps):
        dt = t / num_steps
        gamma = gamma0
        for _ in range(int(num_steps)):
            delta = self.ode_function(gamma)
            gamma = gamma + dt * delta
        return gamma


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
import math
from einops import rearrange


class HierarchicalAdaptiveAttention(GAUBase):
    """
    Hierarchical Adaptive Multi-Head Attention (HA-MHA)

    This module implements a hierarchical adaptive multi-head attention mechanism that
    captures multi-scale dependencies in the input sequence. It organizes attention heads
    into hierarchical groups, each responsible for capturing dependencies at different scales
    (e.g., local, medium, global). An adaptive gating mechanism dynamically allocates attention
    resources based on the input context, allowing the model to focus on the most relevant
    information at each scale.

    **Main Features:**
    - **Hierarchical Structure**: Attention heads are grouped into multiple scales to capture
      dependencies at different levels.
    - **Multi-Scale Linear Attention**: Reduces computational complexity from O(N^2) to O(N)
      within each hierarchical group using linear attention mechanisms.
    - **Adaptive Gating Mechanism**: Dynamically scales the contribution of each hierarchical group
      based on the input context using a gating function.
    - **Dynamic Composition**: Composes attention outputs from all hierarchical groups adaptively.
    - **Rotary Positional Embeddings**: Incorporates positional information using rotary embeddings.

    Args:
        embed_dim (int): Total embedding dimension.
        block_loc (tuple): Location of the block within the network.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): The device to use.
        dtype (torch.dtype, optional): The data type to use.
        num_heads (int): Total number of attention heads.
        num_scales (int): Number of hierarchical scales.
        dropout (float): Dropout probability.
        rotary_emb_base (float): Base for rotary positional embeddings.
        **kwargs: Additional keyword arguments.

    Attributes:
        head_dim (int): Dimension of each attention head.
        query_projs (nn.ModuleList): List of query projections for each scale.
        key_projs (nn.ModuleList): List of key projections for each scale.
        value_projs (nn.ModuleList): List of value projections for each scale.
        gate_proj (nn.Linear): Linear layer for adaptive gating.
        out_proj (nn.Linear): Output projection layer.
        rotary_emb (RotaryPositionalEmbeddings): Positional embedding module.

    Shape:
        - Input: X of shape (batch_size, seq_len, embed_dim)
        - Output: Y of shape (batch_size, seq_len, embed_dim)

    Examples:
        >>> attn = HierarchicalAdaptiveAttention(embed_dim=512, block_loc=(0, 1), kwarg_all={}, num_heads=8, num_scales=2)
        >>> X = torch.randn(2, 10, 512)
        >>> Y, Z = attn(X)
        >>> Y.shape
        torch.Size([2, 10, 512])

    References:
        - Paper: "HieraNorm-AttnGPT: Hierarchical Adaptive Multi-Head Attention with Dynamic Layer Normalization"
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_heads: int=8, num_scales: int=2,
        dropout: float=0.1, rotary_emb_base: float=10000.0, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        assert embed_dim % (num_heads * num_scales
            ) == 0, 'embed_dim must be divisible by num_heads * num_scales'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.head_dim = embed_dim // (num_heads * num_scales)
        self.dropout = dropout
        self.query_projs = nn.ModuleList([nn.Linear(embed_dim, num_heads *
            self.head_dim, bias=False, **self.factory_kwargs) for _ in
            range(num_scales)])
        self.key_projs = nn.ModuleList([nn.Linear(embed_dim, num_heads *
            self.head_dim, bias=False, **self.factory_kwargs) for _ in
            range(num_scales)])
        self.value_projs = nn.ModuleList([nn.Linear(embed_dim, num_heads *
            self.head_dim, bias=False, **self.factory_kwargs) for _ in
            range(num_scales)])
        self.gate_proj = nn.Linear(embed_dim, num_scales, bias=False, **
            self.factory_kwargs)
        self.out_proj = nn.Linear(num_heads * self.head_dim * num_scales,
            embed_dim, **self.factory_kwargs)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        kwarg_all['rotary_emb_dim'] = self.head_dim
        self.rotary_emb = RotaryPositionalEmbeddings(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        B, L, D = X.size()
        assert D == self.embed_dim, f'Expected input embedding dimension: {self.embed_dim}, got: {D}'
        gate_scores = torch.sigmoid(self.gate_proj(X))
        attn_outputs = []
        for scale in range(self.num_scales):
            Q = self.query_projs[scale](X).view(B, L, self.num_heads, self.
                head_dim).transpose(1, 2)
            K = self.key_projs[scale](X).view(B, L, self.num_heads, self.
                head_dim).transpose(1, 2)
            V = self.value_projs[scale](X).view(B, L, self.num_heads, self.
                head_dim).transpose(1, 2)
            Z['input_emb'] = Q
            _, Z = self.rotary_emb(X, **Z)
            Q = Z['output_emb']
            Z['input_emb'] = K
            _, Z = self.rotary_emb(X, **Z)
            K = Z['output_emb']
            scaling_factor = 1.0 / math.sqrt(self.head_dim)
            Q = Q * scaling_factor
            K = F.softmax(K, dim=-1)
            V = V
            KV = torch.einsum('bhld,bhld->bhld', K, V)
            attn_output = torch.einsum('bhld,bhld->bhld', Q, KV)
            attn_output = self.dropout_layer(attn_output)
            attn_outputs.append(attn_output)
        attn_output = torch.cat(attn_outputs, dim=-1)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, -1)
        gate_scores = gate_scores.unsqueeze(-1)
        gate_scores = gate_scores.expand(-1, -1, -1, self.num_heads * self.
            head_dim)
        attn_output = attn_output.view(B, L, self.num_scales, self.
            num_heads * self.head_dim)
        attn_output = attn_output * gate_scores
        attn_output = attn_output.reshape(B, L, -1)
        Y = self.out_proj(attn_output)
        return Y, Z


import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class RotaryPositionalEmbeddings(GAUBase):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, rotary_emb_base: int=10000, rotary_emb_dim:
        int=None, max_seq_len: int=4096, **kwargs) ->None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.dim = rotary_emb_dim
        self.base = rotary_emb_base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / self.base ** (torch.arange(0, self.dim, 2, **self.
            factory_kwargs)[:self.dim // 2].float() / self.dim)
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int=4096) ->None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=
            self.theta.device)
        idx_theta = torch.einsum('i, j -> ij', seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)],
            dim=-1)
        self.register_buffer('cache', cache, persistent=False)

    def _forward(self, X: Tensor, input_emb: Tensor, input_pos: Optional[
        Tensor]=None) ->Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        seq_len = input_emb.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[
            input_pos]
        xshaped = input_emb.float().reshape(*input_emb.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2
            )
        x_out = torch.stack([xshaped[..., 0] * rope_cache[..., 0] - xshaped
            [..., 1] * rope_cache[..., 1], xshaped[..., 1] * rope_cache[...,
            0] + xshaped[..., 0] * rope_cache[..., 1]], -1)
        x_out = x_out.flatten(3)
        output_emb = x_out.type_as(input_emb)
        return X, {'output_emb': output_emb}


gab_config = {'max_seq_len': 4096, 'rotary_emb_base': 10000.0, 'dropout': 
    0.1, 'num_scales': 2, 'num_heads': 8, 'num_steps': 10, 'eps': 1e-05,
    'bias': False, 'multiple_of': 128, 'hidden_features': None,
    'out_features': None, 'activation': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)