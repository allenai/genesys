# gab.py    # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE #

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #


class GAB(GABBase):
    def __init__(self,embed_dim: int, block_loc: tuple, device=None,dtype=None,**kwargs): # YOU CAN ADD MORE ARGUMENTS, BUT YOU HAVE TO HAVE embed_dim, device, dtype AS THE ARGUTMENTS #
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim, block_loc) # DO NOT CHANGE THIS LINE #
        self.root = MemHierBlock(embed_dim=embed_dim, block_loc=block_loc, kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z): 
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
import torch.nn.functional as F


class MemHierBlock(GAUBase):
    """
    Memory-Augmented Hierarchical Transformer Block with Unified Resource Management.

    This block combines hierarchical normalization and attention through a shared memory 
    system, dynamically allocating computational resources based on input complexity.

    Features:
    - Memory-augmented hierarchical attention with paged attention cache
    - Dynamic layer normalization for adaptive scaling
    - Unified memory management across components
    - Resource-aware computation allocation

    Args:
        embed_dim (int): Embedding dimension
        block_loc (tuple): Block location in network (layer_idx, block_idx)
        kwarg_all (dict): Additional arguments
        device (torch.device, optional): Computation device
        dtype (torch.dtype, optional): Data type
        num_heads (int, optional): Number of attention heads. Default: 8
        num_scales (int, optional): Number of hierarchical scales. Default: 2
        dropout (float, optional): Dropout rate. Default: 0.1
        memory_size (int, optional): Memory cache size. Default: 1024

    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_heads: int=8, num_scales: int=2,
        dropout: float=0.1, memory_size: int=1024, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.dropout = dropout
        self.memory_size = memory_size
        self.attn = HierarchicalAdaptiveAttention(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        self.mlp = GatedMLP(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)
        self.norm1 = DynamicLayerNorm(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)
        self.norm2 = DynamicLayerNorm(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)
        self.memory_manager = MemoryManager(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        self.resource_allocator = ResourceAllocator(embed_dim=
            self.embed_dim, block_loc=self.block_loc, kwarg_all=
            self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        """Forward pass of the MemHierBlock."""
        X_mem, Z = self.memory_manager(X, **Z)
        X_res, Z = self.resource_allocator(X_mem, **Z)
        residual = X_res
        X_norm1, Z = self.norm1(X_res, **Z)
        X_attn, Z = self.attn(X_norm1, **Z)
        X_post_attn = residual + X_attn
        residual = X_post_attn
        X_norm2, Z = self.norm2(X_post_attn, **Z)
        X_mlp, Z = self.mlp(X_norm2, **Z)
        Y = residual + X_mlp
        return Y, Z

import torch.nn.functional as F


class DynamicLayerNorm(GAUBase):
    """
    Dynamic Layer Normalization with Adaptive Parameters.

    This layer extends RMSNorm by making the normalization parameters dynamic and input-dependent.
    It generates scaling and shifting parameters adaptively based on the input features,
    allowing the normalization behavior to change based on the context.

    Features:
    - Dynamic parameter generation through lightweight MLPs
    - Input-dependent scaling and shifting
    - Efficient computation through shared parameter networks
    - Stable gradient flow through residual connections

    Args:
        embed_dim (int): The size of the input feature dimension
        block_loc (tuple): Location of this block in the model architecture
        kwarg_all (dict): Additional keyword arguments
        device (torch.device, optional): Device for computation
        dtype (torch.dtype, optional): Data type for computation
        eps (float, optional): Small constant for numerical stability. Default: 1e-5
        reduction_factor (int, optional): Reduction factor for parameter generation MLPs. Default: 4

    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)

    Examples:
        >>> norm = DynamicLayerNorm(512, (0, 0), {})
        >>> x = torch.randn(2, 100, 512)
        >>> y, z = norm(x)
        >>> print(y.shape)
        torch.Size([2, 100, 512])

    References:
        - "Dynamic Layer Normalization for Adaptive Neural Acoustic Modeling in Speech Recognition"
        - "Root Mean Square Layer Normalization"
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, eps: float=1e-05, reduction_factor: int=4,
        **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.weight = nn.Parameter(torch.ones(embed_dim, **self.factory_kwargs)
            )
        self.bias = nn.Parameter(torch.zeros(embed_dim, **self.factory_kwargs))
        self.variance_epsilon = eps
        hidden_dim = max(embed_dim // reduction_factor, 32)
        mlp_kwargs = {'device': device, 'dtype': torch.float32}
        self.gamma_net = nn.Sequential(nn.Linear(embed_dim, hidden_dim, **
            mlp_kwargs), nn.ReLU(), nn.Linear(hidden_dim, embed_dim, **
            mlp_kwargs))
        self.beta_net = nn.Sequential(nn.Linear(embed_dim, hidden_dim, **
            mlp_kwargs), nn.ReLU(), nn.Linear(hidden_dim, embed_dim, **
            mlp_kwargs))
        self.gamma_net[-1].weight.data.zero_()
        self.gamma_net[-1].bias.data.zero_()
        self.beta_net[-1].weight.data.zero_()
        self.beta_net[-1].bias.data.zero_()

    def _forward(self, X, **Z):
        """
        Forward pass of Dynamic Layer Normalization.

        Args:
            X: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Y: Normalized tensor of shape (batch_size, seq_len, embed_dim)
        """
        input_dtype = X.dtype
        X_f32 = X.to(torch.float32)
        variance = X_f32.pow(2).mean(-1, keepdim=True)
        X_norm = X_f32 * torch.rsqrt(variance + self.variance_epsilon)
        seq_context = X_f32.mean(1)
        dynamic_gamma = 1.0 + self.gamma_net(seq_context)
        dynamic_beta = self.beta_net(seq_context)
        dynamic_gamma = dynamic_gamma.to(input_dtype).unsqueeze(1)
        dynamic_beta = dynamic_beta.to(input_dtype).unsqueeze(1)
        Y = self.weight * X_norm.to(input_dtype
            ) * dynamic_gamma + self.bias + dynamic_beta
        return Y, Z

import torch.nn.functional as F
import math


class ResourceAllocator(GAUBase):
    """
    ResourceAllocator

    The ResourceAllocator dynamically allocates computational resources based on the input complexity
    and memory state. It updates the resource allocation parameters in Z['resource_allocation'] that
    are used by other components such as attention, MLP, and normalization layers.

    **Core Idea:**

    - Analyze the input complexity (e.g., sequence length, variance)
    - Allocate computational resources proportionally based on input complexity
    - Update resource allocation parameters in Z['resource_allocation']
    - Ensure efficient usage of computational resources

    **Mathematical Formulation:**

        For input X:
            - Compute complexity metric C(X)
            - Determine scaling factors for different components:
                - attention_scale = f_attn(C(X))
                - mlp_scale = f_mlp(C(X))
                - norm_scale = f_norm(C(X))
            - Update Z['resource_allocation'] with scales

    **Args:**
        embed_dim (int): Embedding dimension.
        block_loc (tuple): Location of the block within the network.
        kwarg_all (dict): Dictionary of all keyword arguments.
        device (torch.device, optional): Device to use.
        dtype (torch.dtype, optional): Data type to use.
        **kwargs: Additional keyword arguments.

    **Inputs:**
        - X: Input tensor of shape (batch_size, seq_len, embed_dim)

    **Outputs:**
        - Y: Output tensor (same as input X)
        - Z: Updated intermediate variables with 'resource_allocation' key

    **Example:**

        >>> allocator = ResourceAllocator(embed_dim=512, block_loc=(0, 0), kwarg_all={})
        >>> X = torch.randn(32, 128, 512)
        >>> Y, Z = allocator(X)
        >>> print(Z['resource_allocation'])

    **Note:**
        This implementation uses simple heuristics to allocate resources based on input variance and sequence length.

    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        pass

    def analyze_complexity(self, X):
        seq_len = X.size(1)
        variance = X.var(dim=-1).mean()
        complexity = variance * seq_len
        return complexity

    def allocate_resources(self, complexity):
        normalized_complexity = torch.tanh(complexity / 1000.0)
        attention_scale = 1.0 - normalized_complexity * 0.5
        mlp_scale = 1.0 - normalized_complexity * 0.5
        norm_scale = 1.0
        resource_allocation = {'attention_scale': attention_scale.item(),
            'mlp_scale': mlp_scale.item(), 'norm_scale': norm_scale}
        return resource_allocation

    def _forward(self, X, **Z):
        complexity = self.analyze_complexity(X)
        resource_allocation = self.allocate_resources(complexity)
        Z['resource_allocation'] = resource_allocation
        Y = X
        return Y, Z

import torch.nn.functional as F
import math


class HierarchicalAdaptiveAttention(GAUBase):
    """
    Memory-Integrated Hierarchical Adaptive Multi-Head Attention (MI-HA-MHA)

    This module extends the hierarchical adaptive multi-head attention mechanism with memory integration.
    It captures multi-scale dependencies while efficiently utilizing cached memory states.

    Args:
        embed_dim (int): Total embedding dimension
        block_loc (tuple): Block location in network
        kwarg_all (dict): Additional keyword arguments
        device (torch.device, optional): Computation device
        dtype (torch.dtype, optional): Data type
        num_heads (int): Number of attention heads. Default: 8
        num_scales (int): Number of hierarchical scales. Default: 2
        dropout (float): Dropout probability. Default: 0.1
        rotary_emb_base (float): Base for rotary embeddings. Default: 10000.0

    Shape:
        - Input: X of shape (batch_size, seq_len, embed_dim)
        - Output: Y of shape (batch_size, seq_len, embed_dim)
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
        self.dropout_layer = nn.Dropout(p=dropout)
        kwarg_all['rotary_emb_dim'] = self.head_dim
        self.rotary_emb = RotaryPositionalEmbeddings(embed_dim=
            self.embed_dim, block_loc=self.block_loc, kwarg_all=
            self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        B, L, D = X.size()
        assert D == self.embed_dim, f'Expected input embedding dimension: {self.embed_dim}, got: {D}'
        resource_scale = Z.get('resource_allocation', {}).get('attention_scale'
            , 1.0)
        if resource_scale is None:
            resource_scale = 1.0
        cached_keys = Z.get('cached_keys', None)
        cached_values = Z.get('cached_values', None)
        gate_scores = torch.sigmoid(self.gate_proj(X))
        attn_outputs = []
        for scale in range(self.num_scales):
            Q = self.query_projs[scale](X).view(B, L, self.num_heads, self.
                head_dim).transpose(1, 2)
            K = self.key_projs[scale](X).view(B, L, self.num_heads, self.
                head_dim).transpose(1, 2)
            V = self.value_projs[scale](X).view(B, L, self.num_heads, self.
                head_dim).transpose(1, 2)
            Q_flat = Q.reshape(B * self.num_heads, L, self.head_dim)
            K_flat = K.reshape(B * self.num_heads, L, self.head_dim)
            Z['input_emb'] = Q_flat
            _, Z_q = self.rotary_emb(X, **Z)
            Q = Q_flat if Z_q.get('output_emb') is None else Z_q['output_emb']
            Q = Q.reshape(B, self.num_heads, L, self.head_dim)
            Z['input_emb'] = K_flat
            _, Z_k = self.rotary_emb(X, **Z)
            K = K_flat if Z_k.get('output_emb') is None else Z_k['output_emb']
            K = K.reshape(B, self.num_heads, L, self.head_dim)
            if cached_keys is not None and cached_values is not None:
                K_cache = self.key_projs[scale](cached_keys)
                V_cache = self.value_projs[scale](cached_values)
                K_cache = K_cache.view(B, -1, self.num_heads, self.head_dim
                    ).transpose(1, 2)
                V_cache = V_cache.view(B, -1, self.num_heads, self.head_dim
                    ).transpose(1, 2)
                K_cache_flat = K_cache.reshape(B * self.num_heads, -1, self
                    .head_dim)
                Z['input_emb'] = K_cache_flat
                _, Z_kc = self.rotary_emb(cached_keys, **Z)
                K_cache = K_cache_flat if Z_kc.get('output_emb'
                    ) is None else Z_kc['output_emb']
                K_cache = K_cache.reshape(B, self.num_heads, -1, self.head_dim)
                K = torch.cat([K_cache, K], dim=2)
                V = torch.cat([V_cache, V], dim=2)
            scaling_factor = 1.0 / math.sqrt(self.head_dim)
            Q = Q * scaling_factor * resource_scale
            K = F.softmax(K, dim=-1)
            KV = torch.einsum('bhld,bhld->bhld', K, V)
            attn_output = torch.einsum('bhld,bhld->bhld', Q, KV)
            attn_output = self.dropout_layer(attn_output)
            attn_outputs.append(attn_output)
        attn_output = torch.cat(attn_outputs, dim=-1)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, -1)
        gate_scores = gate_scores.unsqueeze(-1).expand(-1, -1, -1, self.
            num_heads * self.head_dim)
        attn_output = attn_output.view(B, L, self.num_scales, self.
            num_heads * self.head_dim)
        attn_output = attn_output * gate_scores
        attn_output = attn_output.reshape(B, L, -1)
        Y = self.out_proj(attn_output)
        Z['keys'] = X
        Z['values'] = X
        return Y, Z

import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math


class RotaryPositionalEmbeddings(GAUBase):
    """
    Rotary Positional Embeddings (RoPE) for Memory-Augmented Hierarchical Transformer.

    This implementation provides rotary positional embeddings that are used in the attention
    mechanism to encode relative positions. It caches embeddings for efficient computation
    and supports both training and inference modes.

    Features:
    - Efficient caching of position embeddings
    - Support for packed sequences through position IDs
    - Memory-efficient implementation with buffer reuse
    - Dynamic sequence length handling

    Mathematical Formulation:
        For position i and dimension d:
        θ_i,d = 1/10000^(2d/D)
        Rotation matrix R_θ = [cos(θ), -sin(θ)]
                             [sin(θ), cos(θ)]
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, rotary_emb_base: int=10000, rotary_emb_dim:
        int=None, max_seq_len: int=4096, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.dim = rotary_emb_dim
        if self.dim is None:
            self.dim = kwarg_all.get('rotary_emb_dim', None)
        if self.dim is None:
            self.dim = kwargs.get('rotary_emb_dim', None)
        if self.dim is None:
            self.dim = embed_dim
        self.dim = self.dim // 2 * 2
        if self.dim <= 0:
            raise ValueError(
                f'Rotary embedding dimension must be positive, got {self.dim}')
        self.base = rotary_emb_base
        self.max_seq_len = max_seq_len
        self._init_rotary()

    def _init_rotary(self):
        """Initialize rotary embeddings with geometric progression."""
        half_dim = self.dim // 2
        inv_freq = torch.arange(half_dim, **self.factory_kwargs)
        inv_freq = self.base ** (-2.0 * inv_freq / self.dim)
        self.register_buffer('inv_freq', inv_freq)
        pos = torch.arange(self.max_seq_len, **self.factory_kwargs)
        sincos = torch.outer(pos, inv_freq)
        self.register_buffer('cos_cached', torch.cos(sincos), persistent=False)
        self.register_buffer('sin_cached', torch.sin(sincos), persistent=False)

    def _forward(self, X: Tensor, input_emb: Tensor, input_pos: Optional[
        Tensor]=None):
        """
        Apply rotary position embeddings to input embeddings.

        Args:
            X: Original input tensor (unused, kept for interface compatibility)
            input_emb: Input embeddings to apply rotary embeddings to [batch, seq_len, dim]
            input_pos: Optional position IDs for packed sequences [batch, seq_len]
        """
        if input_emb is None:
            return X, {'output_emb': None}
        input_emb = input_emb.to(**self.factory_kwargs)
        batch_size, seq_len, dim = input_emb.shape
        half_dim = dim // 2
        if dim != self.dim:
            raise ValueError(
                f'Input embedding dimension {dim} does not match rotary dimension {self.dim}'
                )
        if seq_len > self.max_seq_len:
            pos = torch.arange(seq_len, **self.factory_kwargs)
            sincos = torch.outer(pos, self.inv_freq)
            cos = torch.cos(sincos)
            sin = torch.sin(sincos)
        else:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        if input_pos is not None:
            input_pos = input_pos.to(self.factory_kwargs['device'])
            cos = cos[input_pos]
            sin = sin[input_pos]
        else:
            cos = cos.unsqueeze(0).expand(batch_size, seq_len, half_dim)
            sin = sin.unsqueeze(0).expand(batch_size, seq_len, half_dim)
        x_split = input_emb.view(batch_size, seq_len, 2, half_dim)
        x1, x2 = x_split[..., 0, :], x_split[..., 1, :]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        output_emb = torch.stack((out1, out2), dim=-2).flatten(-2)
        return X, {'output_emb': output_emb}

import torch.nn.functional as F


class GatedMLP(GAUBase):
    """
    Gated Multi-Layer Perceptron with position-wise feed-forward network and gating mechanism.

    This implementation extends the base GatedMLP with:
    - Efficient memory usage through multiple-of-8 padding
    - Resource-aware computation with optional layer scaling
    - Adaptive activation gating

    Args:
        embed_dim (int): Input embedding dimension
        block_loc (tuple): Location of block in network (layer_idx, block_idx)
        kwarg_all (dict): Additional keyword arguments
        device (torch.device, optional): Computation device
        dtype (torch.dtype, optional): Data type
        hidden_features (int, optional): Hidden layer dimension. If None, computed as 8/3 * embed_dim
        out_features (int, optional): Output dimension. If None, same as embed_dim
        activation (callable, optional): Activation function. Default: F.silu
        bias (bool): Whether to use bias in linear layers. Default: False
        multiple_of (int): Pad hidden dimension to be multiple of this. Default: 128
        dropout (float): Dropout probability. Default: 0.0

    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)

    Examples:
        >>> mlp = GatedMLP(embed_dim=512, block_loc=(0,0), kwarg_all={})
        >>> x = torch.randn(2, 128, 512)
        >>> y, z = mlp(x)
        >>> print(y.shape)
        torch.Size([2, 128, 512])
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, hidden_features=None, out_features=None,
        activation=None, bias=False, multiple_of=128, dropout=0.0, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.out_features = (out_features if out_features is not None else
            embed_dim)
        if hidden_features is None:
            hidden_features = int(8 * embed_dim / 3)
        self.hidden_features = (hidden_features + multiple_of - 1
            ) // multiple_of * multiple_of
        self.fc1 = nn.Linear(embed_dim, 2 * self.hidden_features, bias=bias,
            **self.factory_kwargs)
        self.fc2 = nn.Linear(self.hidden_features, self.out_features, bias=
            bias, **self.factory_kwargs)
        self.activation = activation if activation is not None else F.silu
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution"""
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def _forward(self, X, **Z):
        """
        Forward pass of GatedMLP.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            **Z: Additional arguments passed through the network

        Returns:
            tuple: (output tensor, updated Z dictionary)
        """
        resource_scale = Z.get('resource_allocation', {}).get('mlp_scale', 1.0)
        y = self.fc1(X)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate) * resource_scale
        y = self.dropout(y)
        y = self.fc2(y)
        return y, Z

import torch.nn.functional as F


class MemoryManager(GAUBase):
    """
    MemoryManager

    This GAU manages the memory state for the MemHierGPT model, including the paged attention cache and blockwise processing state.

    It maintains and updates the memory state during the forward pass and provides it to other components as needed.

    **Code Example:**

        # Initialize MemoryManager
        memory_manager = MemoryManager(embed_dim=512, block_loc=(0, 0), kwarg_all={})

        # Forward pass
        X = torch.randn(32, 128, 512)
        Y, Z = memory_manager(X)

    Args:
        embed_dim (int): Embedding dimension.
        block_loc (tuple): Location of the block within the network.
        kwarg_all (dict): All keyword arguments.
        device (torch.device, optional): Device to use.
        dtype (torch.dtype, optional): Data type to use.
        memory_size (int, optional): Size of the memory cache. Default: 1024.

    Returns:
        Y: Output tensor (possibly modified input X).
        Z (dict): Updated intermediate variables, with 'memory_state' key updated.

    Raises:
        ValueError: If any of the inputs are invalid.

    Example:
        >>> memory_manager = MemoryManager(embed_dim=512, block_loc=(0, 0), kwarg_all={})
        >>> X = torch.randn(32, 128, 512)
        >>> Y, Z = memory_manager(X)

    Note:
        The MemoryManager uses child GAUs PagedAttentionCache, BlockwiseProcessor, and MemoryState to manage different aspects of the memory.

        The actual implementations of these components are declared as child GAUs and need to be implemented separately.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, memory_size: int=1024, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.paged_attention = PagedAttentionCache(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        self.block_processor = BlockwiseProcessor(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        self.memory_state_module = MemoryState(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        memory_state = Z.get('memory_state', {})
        Z['paged_attention_state'] = memory_state.get('paged_attention', {})
        Z['block_processor_state'] = memory_state.get('block_processor', {})
        Z['memory_state_state'] = memory_state.get('memory_state', {})
        _, Z = self.paged_attention(X, **Z)
        _, Z = self.block_processor(X, **Z)
        _, Z = self.memory_state_module(X, **Z)
        memory_state = {'paged_attention': Z.get('paged_attention_state', {
            }), 'block_processor': Z.get('block_processor_state', {}),
            'memory_state': Z.get('memory_state_state', {})}
        Z['memory_state'] = memory_state
        return X, Z

import torch.nn.functional as F


class PagedAttentionCache(GAUBase):
    """
    Paged Attention Cache for Memory-Augmented Hierarchical Transformers.

    This GAU handles the caching of attention keys and values in a paginated manner
    to facilitate memory-efficient attention computations for long sequences. It 
    manages the insertion, retrieval, and eviction of cache pages based on sequence 
    positions and predefined memory constraints.

    **Features:**
    - **Paged Caching:** Divides the attention cache into fixed-size pages to manage memory efficiently.
    - **Dynamic Eviction:** Implements an eviction policy to remove the oldest pages when the cache exceeds memory limits.
    - **Scalable Design:** Supports large-scale sequence processing by handling multiple pages seamlessly.
    - **Integration with Attention Mechanisms:** Interfaces with the attention mechanism to provide cached keys and values.

    **Code Example:**

    .. code-block:: python

        # Initialize PagedAttentionCache with a page size of 1024 tokens
        paged_cache = PagedAttentionCache(embed_dim=512, block_loc=(0, 1), kwarg_all={}, page_size=1024, max_pages=10)

        # Mock input keys and values for a batch
        X_keys = torch.randn(32, 128, 512)  # (batch_size, seq_len, embed_dim)
        X_values = torch.randn(32, 128, 512)

        # Forward pass to update the cache
        Y_keys, Z = paged_cache(X_keys, keys=X_keys, values=X_values)

        # Retrieve cached keys and values for attention
        cached_keys = Z.get('cached_keys')
        cached_values = Z.get('cached_values')

    Args:
        embed_dim (int): Dimensionality of the embeddings.
        block_loc (tuple): Location of this block within the network, (layer_idx, block_idx).
        kwarg_all (dict): Dictionary of all keyword arguments.
        device (torch.device, optional): Device for computation. Default: None.
        dtype (torch.dtype, optional): Data type for computation. Default: None.
        page_size (int, optional): Number of tokens per cache page. Default: 1024.
        max_pages (int, optional): Maximum number of pages to retain in the cache. Default: 10.

    Shape:
        - Input: 
            - X (Tensor): Tensor containing new keys or values to cache, shape (batch_size, seq_len, embed_dim).
            - keys (Tensor): Keys tensor to cache, shape (batch_size, seq_len, embed_dim).
            - values (Tensor): Values tensor to cache, shape (batch_size, seq_len, embed_dim).
        - Output: 
            - Y (Tensor): Same as input X, shape (batch_size, seq_len, embed_dim).
            - Z (dict): Updated cache state containing 'cached_keys' and 'cached_values'.

    Example:
        >>> paged_cache = PagedAttentionCache(embed_dim=512, block_loc=(0, 1), kwarg_all={}, page_size=1024, max_pages=10)
        >>> X_keys = torch.randn(32, 128, 512)
        >>> X_values = torch.randn(32, 128, 512)
        >>> Y_keys, Z = paged_cache(X_keys, keys=X_keys, values=X_values)
        >>> cached_keys = Z.get('cached_keys')
        >>> cached_values = Z.get('cached_values')
        >>> print(cached_keys.shape)
        torch.Size([32, 128, 512])
        >>> print(cached_values.shape)
        torch.Size([32, 128, 512])

    References:
        - Wu, Q., et al. (2020). "Memformer: A Memory-Augmented Transformer for Sequence Modeling."
        - Kitaev, N., et al. (2020). "Reformer: The Efficient Transformer."
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, page_size: int=1024, max_pages: int=10, **
        kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.current_page = 0
        self.cache_keys = {}
        self.cache_values = {}

    def _forward(self, X, **Z):
        """
        Forward pass for PagedAttentionCache.

        Args:
            X (Tensor): Tensor containing new keys or values to cache, shape (batch_size, seq_len, embed_dim).
            keys (Tensor): Keys tensor to cache, shape (batch_size, seq_len, embed_dim).
            values (Tensor): Values tensor to cache, shape (batch_size, seq_len, embed_dim).

        Returns:
            Y (Tensor): Same as input X.
            Z (dict): Updated cache state containing 'cached_keys' and 'cached_values'.
        """
        new_keys = Z.get('keys')
        new_values = Z.get('values')
        if new_keys is not None and new_values is not None:
            assert new_keys.shape == new_values.shape, 'Keys and values must have the same shape.'
            batch_size, seq_len, embed_dim = new_keys.shape
            assert embed_dim == self.embed_dim, f'Expected embed_dim={self.embed_dim}, got {embed_dim}.'
            tokens_per_batch = seq_len
            total_new_pages = (tokens_per_batch + self.page_size - 1
                ) // self.page_size
            for page in range(total_new_pages):
                start_idx = page * self.page_size
                end_idx = min((page + 1) * self.page_size, tokens_per_batch)
                actual_page_size = end_idx - start_idx
                page_keys = new_keys[:, start_idx:end_idx, :]
                page_values = new_values[:, start_idx:end_idx, :]
                self.cache_keys[self.current_page] = page_keys.detach()
                self.cache_values[self.current_page] = page_values.detach()
                self.current_page += 1
                if self.current_page > self.max_pages:
                    oldest_page = self.current_page - self.max_pages - 1
                    if oldest_page in self.cache_keys:
                        del self.cache_keys[oldest_page]
                    if oldest_page in self.cache_values:
                        del self.cache_values[oldest_page]
        Y = X
        Z_updated = {'cached_keys': self._get_cached_tensor(self.cache_keys
            ), 'cached_values': self._get_cached_tensor(self.cache_values)}
        return Y, Z_updated

    def _get_cached_tensor(self, cache_dict):
        """
        Concatenates cached tensors across all pages.

        Args:
            cache_dict (dict): Dictionary of cached tensors.

        Returns:
            Tensor: Concatenated tensor of shape (batch_size, total_cached_tokens, embed_dim).
        """
        if not cache_dict:
            return torch.empty(0, self.embed_dim, device=self.
                factory_kwargs['device'], dtype=self.factory_kwargs['dtype'])
        sorted_keys = sorted(cache_dict.keys())
        cached_tensor = torch.cat([cache_dict[page] for page in sorted_keys
            ], dim=1)
        return cached_tensor

import torch.nn.functional as F


class BlockwiseProcessor(GAUBase):
    """
    BlockwiseProcessor

    This GAU processes sequences in blocks. It is designed to handle long sequences efficiently by
    splitting them into smaller blocks, processing each block independently, and then combining the results.

    **Features:**
    - Splits the input sequence into blocks of a specified size
    - Processes each block individually
    - Maintains and updates a block_processor_state to handle stateful operations across blocks
    - Supports both sequential and parallel block processing

    **Args:**
        embed_dim (int): The embedding dimension of the input sequence.
        block_loc (tuple): The location of this block within the network, (layer_idx, block_idx).
        kwarg_all (dict): Dictionary containing all keyword arguments.
        device (torch.device, optional): Device to use for computation.
        dtype (torch.dtype, optional): Data type to use for computation.
        block_size (int, optional): Size of each block. Default: 128.
        **kwargs: Additional keyword arguments.

    **Shape:**
        - Input:
            - X: Tensor of shape (batch_size, seq_len, embed_dim)
            - block_processor_state: A dictionary containing the state of the block processor
        - Output:
            - Y: Tensor of the same shape as X
            - block_processor_state: Updated block processor state

    **Example:**
        >>> block_processor = BlockwiseProcessor(embed_dim=512, block_loc=(0,0), kwarg_all={}, block_size=128)
        >>> X = torch.randn(2, 1024, 512)
        >>> Z = {}
        >>> Y, Z = block_processor(X, **Z)
        >>> Y.shape
        torch.Size([2, 1024, 512])

    **Note:**
        The actual processing applied to each block can be defined by overriding the `process_block` method.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, block_size: int=128, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.block_size = block_size

    def _forward(self, X, block_processor_state=None, **Z):
        if block_processor_state is None:
            block_processor_state = Z.get('block_processor_state', {})
        B, L, D = X.size()
        blocks = X.split(self.block_size, dim=1)
        processed_blocks = []
        for block in blocks:
            processed_block = self.process_block(block, block_processor_state)
            processed_blocks.append(processed_block)
        Y = torch.cat(processed_blocks, dim=1)
        Z['block_processor_state'] = block_processor_state
        return Y, Z

    def process_block(self, block, block_processor_state):
        """
        Process a single block. This method can be overridden to apply specific operations to each block.

        Args:
            block (Tensor): Tensor of shape (batch_size, block_size, embed_dim)
            block_processor_state (dict): State dictionary for the block processor

        Returns:
            processed_block (Tensor): Tensor of the same shape as block
        """
        return block

import torch.nn.functional as F


class MemoryState(GAUBase):
    """
    MemoryState GAU for maintaining the overall memory state in MemHierGPT.

    This unit is responsible for maintaining and updating the overall memory state across forward passes.
    It interacts with other components like PagedAttentionCache and BlockwiseProcessor through the memory state.

    **Features:**
    - Maintains a persistent memory state across time steps
    - Provides methods for initializing, updating, and retrieving memory state
    - Integrates with MemoryManager and other units that require access to memory state

    **Mathematical Formulation:**

        The MemoryState maintains a state dictionary that can be updated and retrieved.
        In the forward pass, it updates the memory state based on the input X and the previous state.

    **Code Example:**

        # Initialize MemoryState
        memory_state = MemoryState(embed_dim=512, block_loc=(0, 0), kwarg_all={})

        # Forward pass
        X = torch.randn(32, 128, 512)
        memory_state_state = {"previous_state": ...}
        Y, Z = memory_state(X, memory_state_state=memory_state_state)

    **Args:**
        embed_dim (int): Embedding dimension.
        block_loc (tuple): Location of the block within the network.
        kwarg_all (dict): Dictionary of keyword arguments for initialization.
        device (torch.device, optional): Device to use.
        dtype (torch.dtype, optional): Data type to use.
        **kwargs: Additional keyword arguments.

    **Inputs:**
        - X: Input tensor of shape (batch_size, seq_len, embed_dim)
        - memory_state_state: Dictionary representing the previous memory state

    **Outputs:**
        - Y: Output tensor (can be the same as input X)
        - memory_state_state: Updated memory state dictionary

    **Example:**

        >>> memory_state = MemoryState(embed_dim=512, block_loc=(0, 0), kwarg_all={})
        >>> X = torch.randn(32, 128, 512)
        >>> Y, Z = memory_state(X, memory_state_state={})
        >>> print(Z['memory_state_state'])

    **Note:**
        This implementation initializes the memory state if it is not provided.
        The memory state can include any information needed to maintain state across time steps.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim

    def _forward(self, X, memory_state_state=None, **Z):
        if memory_state_state is None:
            memory_state_state = {}
        X_mean = X.mean(dim=1)
        memory_state_state['X_mean'] = X_mean
        Z['memory_state_state'] = memory_state_state
        Y = X
        return Y, Z


gab_config = {'rotary_emb_dim': None, 'max_seq_len': 4096, 'rotary_emb_base': 10000.0, 'reduction_factor': 4, 'eps': 1e-05, 'scales': [1, 2, 4], 'memory_size': 1024, 'page_size': 1024, 'max_pages': 10, 'block_size': 128, 'dropout': 0.1, 'num_scales': 2, 'num_heads': 8, 'bias': False, 'multiple_of': 128, 'hidden_features': None, 'out_features': None, 'activation': None}