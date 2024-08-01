import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from torchtune.modules import RMSNorm


def segsum_unstable(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0
        )
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, '... d -> ... d e', e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool),
        diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0
        )
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def pad_to_block_length(X, block_len):
    pad_len = (block_len - X.shape[1] % block_len) % block_len
    if pad_len > 0:
        padding = torch.zeros(X.shape[0], pad_len, *X.shape[2:], dtype=X.
            dtype, device=X.device)
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
    X, A, B, C = [rearrange(x, 'b (c l) ... -> b c l ...', l=block_len) for
        x in (X, A, B, C)]
    A = rearrange(A, 'b c l h -> b h c l')
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum('bclhn,bcshn,bhcls,bcshp->bclhp', C, B, L, X)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum('bclhn,bhcl,bclhp->bchpn', B, decay_states, X)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum('bhzc,bchpn->bzhpn', decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    Y = rearrange(Y_diag + Y_off, 'b c l h p -> b (c l) h p')
    return Y, final_state


class Mamba2Simple(nn.Module):

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, headdim=128,
        ngroups=1, A_init_range=(1, 16), dt_min=0.001, dt_max=0.1,
        dt_init_floor=0.0001, chunk_size=256, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
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
        d_in_proj = (2 * self.d_inner + 2 * self.ngroups * self.d_state +
            self.nheads)
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=True, **
            factory_kwargs)
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(in_channels=conv_dim, out_channels=conv_dim,
            bias=True, kernel_size=d_conv, groups=conv_dim, padding=d_conv -
            1, **factory_kwargs)
        self.act = nn.SiLU()
        dt = torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.
            log(dt_max) - math.log(dt_min)) + math.log(dt_min))
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
        self.norm = nn.LayerNorm(self.d_inner, eps=1e-05, **factory_kwargs)
        self.silu = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=True, **
            factory_kwargs)

    def forward(self, u):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, _seqlen, dim = u.shape
        u = pad_to_block_length(u, self.chunk_size)
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
        y, _ = ssd_minimal_discrete(x * dt.unsqueeze(-1), A * dt, B, C,
            self.chunk_size)
        y = rearrange(y, 'b l h p -> b l (h p)')
        y = self.norm(y * self.silu(z))
        out = self.out_proj(y)
        out = out[:, :_seqlen, :]
        return out


class GAB(GABBase):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """

    def __init__(self, embed_dim: int, device=None, dtype=None, d_state=64,
        d_conv=4, expand=2, headdim=128, ngroups=1, A_init_range=(1, 16),
        dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001, chunk_size=256, **
        kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim)
        self.mamba1 = Mamba2Simple(embed_dim, d_state, d_conv, expand,
            headdim, ngroups, A_init_range, dt_min, dt_max, dt_init_floor,
            chunk_size, **factory_kwargs)
        self.mamba2 = Mamba2Simple(embed_dim, d_state, d_conv, expand,
            headdim, ngroups, A_init_range, dt_min, dt_max, dt_init_floor,
            chunk_size, **factory_kwargs)
        self.norm1 = RMSNorm(embed_dim, eps=1e-05).to(**factory_kwargs)
        self.norm2 = RMSNorm(embed_dim, eps=1e-05).to(**factory_kwargs)

    def _forward(self, X, **kwargs):
        X = self.mamba1(self.norm1(X)) + X
        X = self.mamba2(self.norm2(X)) + X
        return X


""" The dictionary of hyperparameters for constructing a GAB layer
    embed_dim, device, dtype should NOT be included in gab_config
"""
gab_config = {'d_state': 64, 'd_conv': 4, 'expand': 2, 'headdim': 128,
    'ngroups': 1, 'A_init_range': (1, 16), 'dt_min': 0.001, 'dt_max': 0.1,
    'dt_init_floor': 0.0001, 'chunk_size': 256}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)