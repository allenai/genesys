import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = HierTTT(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
import torch.nn.functional as F
from typing import List


class HierTTT(GAUBase):
    """
    HierTTT: Hierarchical Test-Time Training with Multi-Scale Linear Attention

    **Overview:**

    HierTTT introduces a hierarchical test-time training architecture that:
    - Processes features at multiple scales efficiently
    - Uses sparse attention patterns for linear complexity
    - Maintains test-time adaptability at each scale
    - Integrates features through adaptive normalization

    **Key Components:**
    - **SparseLinearAttention**: Applies sparse linear attention at multiple scales.
    - **ScaleIntegration**: Integrates outputs from different scales.
    - **HierarchicalRMSNorm**: Applies hierarchical normalization.

    **Args:**
        embed_dim (int): The embedding dimension.
        block_loc (tuple): The location of the block in the network.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): The device to run on.
        dtype (torch.dtype, optional): The data type.

    **Inputs:**
        - **X**: Input tensor of shape (batch_size, seq_length, embed_dim)

    **Outputs:**
        - **Y**: Output tensor of the same shape as X.

    **Example:**
        hier_ttt = HierTTT(embed_dim=512, block_loc=(0,0), kwarg_all={})
        X = torch.randn(8, 128, 512)
        Y, Z = hier_ttt(X)

    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.scales = [1, 2, 4]
        self.sparse_attention_s1 = RotaryPositionalEmbeddings(embed_dim=
            self.embed_dim, block_loc=self.block_loc, kwarg_all=self.
            kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.sparse_attention_s2 = RotaryPositionalEmbeddings(embed_dim=
            self.embed_dim, block_loc=self.block_loc, kwarg_all=self.
            kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.sparse_attention_s4 = RotaryPositionalEmbeddings(embed_dim=
            self.embed_dim, block_loc=self.block_loc, kwarg_all=self.
            kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.scale_integration = ScaleIntegration(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.norm = HierarchicalRMSNorm(embed_dim=self.embed_dim, block_loc
            =self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        scale_outputs = []
        for s in self.scales:
            x_s = self._downsample(X, s)
            Z[f'x_s_{s}'] = x_s
            if s == 1:
                y_s, Z = self.sparse_attention_s1(x_s, **Z)
            elif s == 2:
                y_s, Z = self.sparse_attention_s2(x_s, **Z)
            elif s == 4:
                y_s, Z = self.sparse_attention_s4(x_s, **Z)
            else:
                raise ValueError(f'Unsupported scale: {s}')
            y_s_upsampled = self._upsample(y_s, target_length=X.shape[1],
                scale=s)
            scale_outputs.append(y_s_upsampled)
        Z['scale_outputs'] = scale_outputs
        Y, Z = self.scale_integration(X, **Z)
        Y, Z = self.norm(Y, **Z)
        return Y, Z

    def _downsample(self, X, scale):
        if scale == 1:
            return X
        else:
            batch_size, seq_len, embed_dim = X.size()
            pad = scale - 1, 0
            X_padded = F.pad(X.transpose(1, 2), pad)
            weight = X.new_ones((embed_dim, 1, scale)) / scale
            x_s = F.conv1d(X_padded, weight, stride=scale, groups=embed_dim
                ).transpose(1, 2)
            return x_s

    def _upsample(self, X, target_length, scale):
        if scale == 1:
            return X
        else:
            X_upsampled = X.repeat_interleave(scale, dim=1)
            X_upsampled = X_upsampled[:, :target_length, :]
            return X_upsampled


import torch.nn.functional as F


class ScaleIntegration(GAUBase):
    """
    ScaleIntegration

    **Overview:**

    ScaleIntegration integrates outputs from multiple scales into a single output.
    It takes a list of scale outputs provided in `Z['scale_outputs']`, applies
    learnable weights to each scale output via softmax-normalized weights, concatenates
    the weighted outputs, and projects them back to the embedding dimension.

    **Key Features:**

    - Accepts multiple inputs corresponding to outputs from different scales.
    - Applies learnable weights to each scale output.
    - Combines the weighted outputs via concatenation and linear projection.
    - Ensures output shape is consistent with input shape.
    - Handles edge cases where scale outputs have varying sequence lengths.

    **Inputs:**

    - `X`: Tensor of shape `(batch_size, seq_length, embed_dim)`
    - `Z`: A dictionary containing:
        - `'scale_outputs'`: Optional list of tensors, each of shape `(batch_size, seq_length, embed_dim)`

    **Outputs:**

    - `Y`: Tensor of shape `(batch_size, seq_length, embed_dim)`

    **Example:**

        scale_integration = ScaleIntegration(embed_dim=512, block_loc=(0, 0), kwarg_all={'scales': [1, 2, 4]})
        X = torch.randn(8, 128, 512)
        Z = {'scale_outputs': [torch.randn(8, 128, 512) for _ in range(3)]}
        Y, Z = scale_integration(X, **Z)

    **Args:**

    - `embed_dim` (int): Embedding dimension.
    - `block_loc` (tuple): Location of the block within the network.
    - `kwarg_all` (dict): Additional keyword arguments.
    - `device` (torch.device, optional): Device to use.
    - `dtype` (torch.dtype, optional): Data type to use.

    **Note:**

    This unit ensures that the output `Y` has the same shape as the input `X`.
    If `scale_outputs` is not provided in `Z`, it defaults to using `X` for all scales.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.scales = kwargs.pop('scales', kwarg_all.get('scales', [1, 2, 4]))
        if not isinstance(self.scales, (list, tuple)):
            raise ValueError('scales must be a list or tuple')
        if not all(isinstance(s, int) and s > 0 for s in self.scales):
            raise ValueError('all scales must be positive integers')
        self.num_scales = len(self.scales)
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales, **
            self.factory_kwargs))
        self.proj = nn.Linear(embed_dim * self.num_scales, embed_dim, bias=
            False, **self.factory_kwargs)

    def _forward(self, X, **Z):
        scale_outputs = Z.get('scale_outputs', None)
        if not scale_outputs:
            scale_outputs = [X for _ in range(self.num_scales)]
        if not isinstance(scale_outputs, list) or len(scale_outputs
            ) != self.num_scales:
            raise ValueError(
                f"'scale_outputs' must be a list of length {self.num_scales}")
        target_length = X.shape[1]
        aligned_outputs = []
        for out in scale_outputs:
            if out.shape[1] != target_length:
                out = self._align_sequence_length(out, target_length)
            aligned_outputs.append(out.to(**self.factory_kwargs))
        weights = F.softmax(self.scale_weights, dim=0)
        weighted_outputs = [(out * w.view(1, 1, 1)) for out, w in zip(
            aligned_outputs, weights)]
        combined = torch.cat(weighted_outputs, dim=-1)
        Y = self.proj(combined)
        return Y, Z

    def _align_sequence_length(self, out, target_length):
        curr_length = out.shape[1]
        if curr_length > target_length:
            out = out[:, :target_length, :]
        elif curr_length < target_length:
            pad_size = target_length - curr_length
            pad = torch.zeros(out.shape[0], pad_size, out.shape[2], device=
                out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=1)
        return out


import torch.nn.functional as F
import math


class RotaryPositionalEmbeddings(GAUBase):
    """
    Rotary Positional Embeddings (RoPE) for transformers.
    
    This unit implements rotary position embeddings that:
    - Injects relative positional information through rotation matrices
    - Enables attention to consider token positions efficiently
    - Maintains linear complexity and causal properties
    
    **Key Features:**
    - Position-dependent rotation of token embeddings
    - Efficient cached computation of rotation matrices
    - Support for variable sequence lengths
    - Maintains gradients for end-to-end training
    
    **Args:**
        embed_dim (int): The embedding dimension
        block_loc (tuple): Location of this block in the network
        kwarg_all (dict): Additional keyword arguments
        device (torch.device, optional): Device to use
        dtype (torch.dtype, optional): Data type to use
        rotary_emb_dim (int, optional): Dimension for rotary embeddings. Default: embed_dim//4
        max_position_embeddings (int, optional): Maximum sequence length. Default: 4096
        base (int, optional): Base for the angle computation. Default: 10000
        
    **Shape:**
        - Input: (batch_size, seq_length, embed_dim)
        - Output: Rotated embeddings with same shape as input
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.dim = kwargs.pop('rotary_emb_dim', embed_dim // 4)
        self.max_seq_len = kwargs.pop('max_position_embeddings', 4096)
        self.base = kwargs.pop('base', 10000)
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2).float()
            .to(device) / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.build_cache()

    def build_cache(self):
        """Precompute rotation matrices for all possible positions."""
        seq_idx = torch.arange(self.max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', seq_idx.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)

    def _rotate_half(self, x: torch.Tensor) ->torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _forward(self, X: torch.Tensor, **Z) ->tuple:
        """Apply rotary embeddings to input tensor."""
        input_emb = Z.get('input_emb')
        if input_emb is None:
            return X, Z
        position_ids = Z.get('position_ids')
        if position_ids is None:
            position_ids = torch.arange(input_emb.size(1), device=input_emb
                .device)
            position_ids = position_ids.unsqueeze(0).expand(input_emb.size(
                0), -1)
        if position_ids.max() >= self.max_seq_len:
            raise ValueError(
                f'Position IDs must be less than max_seq_len ({self.max_seq_len})'
                )
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        input_rot = self._rotate_half(input_emb)
        output_emb = input_emb * cos + input_rot * sin
        Z['output_emb'] = output_emb.to(dtype=input_emb.dtype)
        return X, Z


import torch.nn.functional as F


class HierarchicalRMSNorm(GAUBase):
    """
    Hierarchical Root Mean Square Layer Normalization (HierarchicalRMSNorm).

    This layer extends RMSNorm by incorporating multi-scale normalization.
    It processes input embeddings at multiple scales and integrates them
    to produce the normalized output while ensuring causality.

    **Core Idea:**

    - The input embeddings are downsampled to multiple scales using causal operations.
    - Each scale has its own normalization parameters.
    - The normalized embeddings at each scale are upsampled causally and combined.

    **Mathematical Formulation:**

        For each scale s:

        x_s = causal_downsample(x, scale=s)

        rms_s(x) = sqrt(mean(x_s^2) + eps)

        y_s = x_s / rms_s(x) * gamma_s

        y = sum(causal_upsample(y_s) * w_s for s in scales)

    **Args:**
        embed_dim (int): Dimensionality of the input embeddings.
        block_loc (tuple): Location of the block within the network.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): Device to use.
        dtype (torch.dtype, optional): Data type to use.

    **Inputs:**
        - **X**: Input tensor of shape (batch_size, sequence_length, embed_dim)

    **Outputs:**
        - **Y**: Output tensor of the same shape as X.

    **Example:**

        >>> norm = HierarchicalRMSNorm(embed_dim=512, block_loc=(0, 0), kwarg_all={'scales': [1, 2, 4]})
        >>> x = torch.randn(32, 128, 512)
        >>> y, _ = norm(x)

    **References:**

        - Proposal for HierarchicalRMSNorm.
    
    **Note:**
        This implementation ensures causality by using causal downsampling and upsampling operations.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.scales = kwargs.pop('scales', kwarg_all.get('scales', [1, 2, 4]))
        self.eps = kwargs.pop('eps', kwarg_all.get('eps', 1e-05))
        self.gammas = nn.ParameterDict({f's{s}': nn.Parameter(torch.ones(
            embed_dim, **self.factory_kwargs)) for s in self.scales})
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales), **
            self.factory_kwargs))

    def _decompose_scales(self, X: torch.Tensor) ->dict:
        """
        Decompose the input tensor into multiple scales.

        Args:
            X (torch.Tensor): Input tensor of shape (B, L, D)

        Returns:
            dict: Dictionary mapping scale to downsampled tensor
        """
        x_scales = {}
        for s in self.scales:
            if s == 1:
                x_scales[s] = X
            else:
                x_s = self._causal_downsample(X, s)
                x_scales[s] = x_s
        return x_scales

    def _causal_downsample(self, X: torch.Tensor, scale: int) ->torch.Tensor:
        """
        Perform causal downsampling on the input tensor.

        Args:
            X (torch.Tensor): Input tensor of shape (B, L, D)
            scale (int): Downsampling scale factor

        Returns:
            torch.Tensor: Downsampled tensor of shape (B, L//scale, D)
        """
        batch_size, seq_length, embed_dim = X.size()
        padding = scale - 1, 0
        X_padded = F.pad(X.transpose(1, 2), padding)
        weight = X.new_ones((embed_dim, 1, scale)) / scale
        x_s = F.conv1d(X_padded, weight, stride=scale, groups=embed_dim
            ).transpose(1, 2)
        return x_s

    def _causal_upsample(self, y_s: torch.Tensor, scale: int, target_length:
        int) ->torch.Tensor:
        """
        Perform causal upsampling on the downsampled tensor.

        Args:
            y_s (torch.Tensor): Downsampled tensor of shape (B, L//scale, D)
            scale (int): Upsampling scale factor
            target_length (int): The target sequence length after upsampling

        Returns:
            torch.Tensor: Upsampled tensor of shape (B, target_length, D)
        """
        upsampled_y_s = y_s.repeat_interleave(scale, dim=1)
        upsampled_y_s = upsampled_y_s[:, :target_length, :]
        return upsampled_y_s

    def _integrate_scales(self, y_scales: dict) ->torch.Tensor:
        """
        Integrate the outputs from different scales.

        Args:
            y_scales (dict): Dictionary mapping scale to upsampled tensor

        Returns:
            torch.Tensor: Integrated tensor of shape (B, L, D)
        """
        weights = F.softmax(self.scale_weights, dim=0)
        Y = 0
        target_length = y_scales[1].size(1)
        for i, (s, y_s) in enumerate(y_scales.items()):
            if s == 1:
                upsampled_y_s = y_s
            else:
                upsampled_y_s = self._causal_upsample(y_s, scale=s,
                    target_length=target_length)
            Y = Y + upsampled_y_s * weights[i]
        return Y

    def _forward(self, X, **Z):
        """
        Forward pass of HierarchicalRMSNorm.

        Args:
            X (torch.Tensor): Input tensor of shape (B, L, D)
            **Z: Additional keyword arguments

        Returns:
            tuple: (Normalized tensor Y, Updated intermediate variables Z)
        """
        X = X.to(**self.factory_kwargs)
        x_scales = self._decompose_scales(X)
        y_scales = {}
        for s, x_s in x_scales.items():
            rms_s = torch.sqrt(torch.mean(x_s.pow(2), dim=-1, keepdim=True) +
                self.eps)
            gamma_s = self.gammas[f's{s}']
            y_s = x_s / rms_s * gamma_s
            y_scales[s] = y_s
        Y = self._integrate_scales(y_scales)
        return Y, Z


gab_config = {}



autoconfig = {
    'd_model': 256,
    'n_block': 11
}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)