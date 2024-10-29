import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = Mamba2(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


import torch.nn.functional as F
from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl


class Mamba2(GAUBase):
    """
    Mamba2: A Generalized Autoregressive Unit (GAU) implementing a double-layer Mamba architecture.

    This class represents a Mamba2 block, which consists of two Mamba layers with normalization.
    It's designed to process sequential data in a causal, differentiable, and parallelizable manner.

    Architecture:
        1. Input Normalization (RMSNorm)
        2. First Mamba Layer
        3. Residual Connection
        4. Second Normalization (RMSNorm)
        5. Second Mamba Layer
        6. Final Residual Connection

    Args:
        embed_dim (int): The dimensionality of the input and output embeddings.
        block_loc (tuple): The location of this block within the larger model architecture.
        kwarg_all (dict): Additional keyword arguments to be passed to child components.
        device (torch.device, optional): The device on which to allocate tensors.
        dtype (torch.dtype, optional): The default dtype for tensors in this module.

    Inputs:
        X (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).
        **Z: Additional keyword arguments for potential future extensions.

    Outputs:
        X (torch.Tensor): Output tensor of shape (batch_size, sequence_length, embed_dim).
        Z (dict): Updated keyword arguments.

    Note:
        This implementation adheres to the GAU (Generalized Autoregressive Unit) interface
        and maintains causal properties for autoregressive processing.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.mamba1 = Mamba2Layer(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.mamba2 = Mamba2Layer(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm1 = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm2 = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)

    def _forward(self, X, **Z):
        X1, Z = self.norm1(X, **Z)
        X2, Z = self.mamba1(X1, **Z)
        X = X + X2
        X3, Z = self.norm2(X, **Z)
        X4, Z = self.mamba2(X3, **Z)
        X = X + X4
        return X, Z


import torch.nn.functional as F
import math
from einops import rearrange, repeat


class Mamba2Layer(GAUBase):
    """
    Mamba2Layer: An implementation of the Mamba architecture layer.

    This layer is based on the Mamba architecture, which combines elements of
    State Space Models (SSMs) and attention mechanisms. It's designed for
    efficient processing of long sequences.

    Args:
        embed_dim (int): Dimension of the input embeddings.
        block_loc (tuple): Location of the block within the model.
        kwarg_all (dict): Additional keyword arguments.
        d_state (int, optional): Dimension of the state. Defaults to 64.
        d_conv (int, optional): Kernel size for the 1D convolution. Defaults to 4.
        expand (int, optional): Expansion factor for the inner dimension. Defaults to 2.
        headdim (int, optional): Dimension of each head. Defaults to 128.
        ngroups (int, optional): Number of groups for group linear operators. Defaults to 1.
        A_init_range (tuple, optional): Range for initializing the A parameter. Defaults to (1, 16).
        dt_min (float, optional): Minimum value for dt initialization. Defaults to 0.001.
        dt_max (float, optional): Maximum value for dt initialization. Defaults to 0.1.
        dt_init_floor (float, optional): Floor value for dt initialization. Defaults to 1e-4.
        chunk_size (int, optional): Size of chunks for processing. Defaults to 256.
        device (torch.device, optional): Device to use for computations.
        dtype (torch.dtype, optional): Data type to use for computations.

    The Mamba2Layer processes input sequences using a combination of linear projections,
    1D convolutions, and a selective scan operation (implemented in SSDMinimalDiscrete).
    It's designed to capture long-range dependencies efficiently.

    The layer includes several components:
    1. Input projection
    2. 1D Convolution
    3. Selective Scan Discrete operation
    4. Output projection

    The layer also implements a chunking mechanism to process long sequences efficiently.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        d_state=64, d_conv=4, expand=2, headdim=128, ngroups=1,
        A_init_range=(1, 16), dt_min=0.001, dt_max=0.1, dt_init_floor=
        0.0001, chunk_size=256, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.d_model = embed_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.chunk_size = chunk_size
        d_in_proj = (2 * self.d_inner + 2 * self.ngroups * self.d_state +
            self.nheads)
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=True, **self
            .factory_kwargs)
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim,
            bias=True, kernel_size=d_conv, groups=conv_dim, padding=d_conv -
            1, **self.factory_kwargs)
        self.act = nn.SiLU()
        dt = torch.exp(torch.rand(self.nheads, **self.factory_kwargs) * (
            math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device
            ).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.norm = nn.LayerNorm(self.d_inner, eps=1e-05, **self.factory_kwargs
            )
        self.silu = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=True, **
            self.factory_kwargs)
        self.ssd_minimal_discrete = GatedSSDMinimalDiscrete(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)

    def pad_to_block_length(self, X, block_len):
        pad_len = (block_len - X.shape[1] % block_len) % block_len
        if pad_len > 0:
            padding = torch.zeros(X.shape[0], pad_len, *X.shape[2:], dtype=
                X.dtype, device=X.device)
            X = torch.cat([X, padding], dim=1)
        return X

    def _forward(self, u, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, _seqlen, dim = u.shape
        u = self.pad_to_block_length(u, self.chunk_size)
        seqlen = u.shape[1]
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 *
            self.ngroups * self.d_state, self.nheads], dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        xBC = xBC[:, :seqlen, :]
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.
            d_state, self.ngroups * self.d_state], dim=-1)
        x = rearrange(x, 'b l (h p) -> b l h p', p=self.headdim)
        B = rearrange(B, 'b l (g n) -> b l g n', g=self.ngroups)
        C = rearrange(C, 'b l (g n) -> b l g n', g=self.ngroups)
        Z = {'x': x, 'A': A, 'B': B, 'C': C, 'dt': dt, 'chunk_size': self.
            chunk_size}
        _, Z_ = self.ssd_minimal_discrete(u, **Z)
        y = Z_.get('y')
        y = rearrange(y, 'b l h p -> b l (h p)')
        y = self.norm(y * self.silu(z))
        out = self.out_proj(y)
        out = out[:, :_seqlen, :]
        return out


import torch.nn.functional as F
from einops import rearrange, repeat


class GatedSSDMinimalDiscrete(GAUBase):
    """
    GatedSSDMinimalDiscrete implements a gated discrete-time state space model.
    
    This class enhances the SSDMinimalDiscrete with gated linear attention mechanisms,
    allowing selective information flow while maintaining the efficient SSM algorithm.
    The implementation processes sequential data in chunks with dynamic gating.
    
    Args:
        embed_dim (int): The embedding dimension of the input.
        block_loc (tuple): The location of the block within the larger model structure.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): The device to run the module on.
        dtype (torch.dtype, optional): The data type of the module's parameters.
        gate_eps (float, optional): Small constant for gate numerical stability. Default: 1e-5
        
    Inputs:
        X (torch.Tensor): The input tensor of shape (batch, length, n_heads, d_head).
        A (torch.Tensor): The state transition tensor of shape (batch, length, n_heads).
        B (torch.Tensor): The input-to-state tensor of shape (batch, length, n_heads, d_state).
        C (torch.Tensor): The state-to-output tensor of shape (batch, length, n_heads, d_state).
        dt (torch.Tensor): The time step tensor of shape (batch, length, n_heads).
        chunk_size (int): The size of chunks for processing the sequence.
        
    Outputs:
        Y (torch.Tensor): The gated output tensor of shape (batch, length, n_heads, d_head).
        
    The class implements an enhanced forward pass with:
    1. Gated intra-chunk computations
    2. Selective inter-chunk state propagation
    3. Gated state-to-output conversion
    4. Dynamic memory management through gating
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, gate_eps=1e-05, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.gate_eps = gate_eps
        self.gate_proj = nn.Linear(embed_dim, embed_dim, **self.factory_kwargs)
        self.gate_norm = nn.LayerNorm(embed_dim, eps=gate_eps, **self.
            factory_kwargs)

    def compute_gates(self, x, chunk_size):
        """Compute attention-based gates for selective information flow."""
        gates = self.gate_proj(x)
        gates = self.gate_norm(gates)
        gates = torch.sigmoid(gates)
        gates = rearrange(gates, 'b (c l) d -> b c l d', l=chunk_size)
        return gates

    def _forward(self, X, x, A, B, C, dt, chunk_size):
        gates = self.compute_gates(X, chunk_size)
        y, final_state = self.gated_ssd_minimal_discrete(x * dt.unsqueeze(-
            1), A * dt, B, C, chunk_size, gates)
        Z_ = {'y': y, 'final_state': final_state}
        return X, Z_

    def segsum(self, x):
        """More stable segment sum calculation with gating support."""
        T = x.size(-1)
        x = repeat(x, '... d -> ... d e', e=T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool),
            diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool),
            diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def gated_ssd_minimal_discrete(self, X, A, B, C, block_len, gates,
        initial_states=None):
        """
        Enhanced SSM computation with gating mechanisms.
        
        Arguments:
            X: (batch, length, n_heads, d_head)
            A: (batch, length, n_heads)
            B: (batch, length, n_heads, d_state)
            C: (batch, length, n_heads, d_state)
            gates: (batch, chunks, block_len, embed_dim)
            
        Returns:
            Y: (batch, length, n_heads, d_head)
            final_state: Final state tensor
        """
        assert X.dtype == A.dtype == B.dtype == C.dtype
        X, A, B, C = [rearrange(x, 'b (c l) ... -> b c l ...', l=block_len) for
            x in (X, A, B, C)]
        A = rearrange(A, 'b c l h -> b h c l')
        A_cumsum = torch.cumsum(A, dim=-1)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum('bclhn,bcshn,bhcls,bcshp->bclhp', C, B, L, X *
            gates.unsqueeze(-2))
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum('bclhn,bhcl,bclhp->bchpn', B, decay_states, X *
            gates.unsqueeze(-2))
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1,
            0))))
        new_states = torch.einsum('bhzc,bchpn->bzhpn', decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states,
            state_decay_out)
        Y = rearrange(Y_diag + Y_off, 'b c l h p -> b (c l) h p')
        return Y, final_state


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


gab_config = {'gate_eps': 1e-05, 'eps': 1e-05, 'chunk_size': 256,
    'dt_init_floor': 0.0001, 'd_conv': 4, 'A_init_range': [1, 16], 'dt_min':
    0.001, 'headdim': 128, 'ngroups': 1, 'dt_max': 0.1, 'd_state': 64,
    'expand': 2}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)