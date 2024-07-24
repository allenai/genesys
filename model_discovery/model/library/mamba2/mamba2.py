# gab.py

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #


# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

# try:
#     from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
# except ImportError:
#     RMSNormGated, LayerNorm = None, None

# from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn



def segsum_unstable(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def pad_to_block_length(X, block_len):
    pad_len = (block_len - X.shape[1] % block_len) % block_len
    if pad_len > 0:
        padding = torch.zeros(X.shape[0], pad_len, *X.shape[2:], dtype=X.dtype, device=X.device)
        X = torch.cat([X, padding], dim=1)
    return X

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    length = X.shape[1]
    # assert X.shape[1] % block_len == 0
    X = pad_to_block_length(X, block_len) # should be no problem as it is a causal model

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")

    Y = Y[:, :length, :, :]  # Trim to original length
    return Y, final_state



class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        # Fused kernel and sharding options
        chunk_size=256,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.chunk_size = chunk_size

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=True, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=True,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        # self.conv1d.weight._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # # D "skip" parameter
        # self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        # self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        # assert RMSNormGated is not None
        # self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=True, **factory_kwargs)

    def forward(self, u):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # 1D Convolution
        # if causal_conv1d_fn is None:
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * ngroups * d_state)
        xBC = xBC[:, :seqlen, :]
        # else: # BUGGY, see https://github.com/state-spaces/mamba/pull/362 
        #     xBC = causal_conv1d_fn(
        #         x=xBC.transpose(1, 2),
        #         weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
        #         bias=self.conv1d.bias,
        #         activation='swish',
        #     ).transpose(1, 2)

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
        
        y, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, self.chunk_size)
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out


class GAB(GABBase):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """
    def __init__(
            self,
            embed_dim: int, 
            device=None,
            dtype=None,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=128,
            ngroups=1,
            A_init_range=(1, 16),
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=1e-4,
            chunk_size=256,
            **kwargs): # YOU CAN ADD MORE ARGUMENTS, BUT YOU HAVE TO HAVE embed_dim, device, dtype AS THE ARGUTMENTS #
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim) # DO NOT CHANGE THIS LINE #
        
        # COMPLETING THE CODE HERE #
        self.mamba1=Mamba2Simple(embed_dim, d_state, d_conv, expand, headdim, ngroups, A_init_range, dt_min, dt_max, dt_init_floor, chunk_size, **factory_kwargs)
        self.mamba2=Mamba2Simple(embed_dim, d_state, d_conv, expand, headdim, ngroups, A_init_range, dt_min, dt_max, dt_init_floor, chunk_size, **factory_kwargs)
        # self.norm1 = RMSNorm(embed_dim, eps=1e-5, **factory_kwargs)
        # self.norm2 = RMSNorm(embed_dim, eps=1e-5, **factory_kwargs)
        self.norm1 = nn.LayerNorm(embed_dim, **factory_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim, **factory_kwargs)


    # YOU CAN ADD MORE FUNCTIONS HERE #


    def _forward(self,X,**kwargs): # type hints are optional but recommended
        # THE CODE HERE MUST BE COMPLETED #
        hidden_states = self.norm1(X.to(dtype=self.norm1.weight.dtype))
        X = self.mamba1(hidden_states) + X
        hidden_states = self.norm2(X.to(dtype=self.norm2.weight.dtype))
        X = self.mamba2(hidden_states) + X
        return X
        
    
    
""" The dictionary of hyperparameters for constructing a GAB layer
    embed_dim, device, dtype should NOT be included in gab_config
"""
gab_config = {
    # THE HYPERPARAMETERS OF ADDITIONAL ARGUMENTS IN GAB CLASS #
    'd_state': 64,
    'd_conv': 4,
    'expand': 2,
    'headdim': 128,
    'ngroups': 1,
    'A_init_range': (1, 16),
    'dt_min': 0.001,
    'dt_max': 0.1,
    'dt_init_floor': 1e-4,
    'chunk_size': 256,
}
