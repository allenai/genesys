# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #


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

    def __init__(
        self,
        embed_dim: int,
        block_loc: tuple,
        kwarg_all: dict,
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
        device=None,
        dtype=None,
        **kwargs
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
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

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=True, **self.factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=True,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **self.factory_kwargs,
        )
        # self.conv1d.weight._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **self.factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
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
        self.A_log._no_weight_decay = True

        self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **self.factory_kwargs)
        self.silu = nn.SiLU()

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=True, **self.factory_kwargs)

        self.ssd_minimal_discrete = SSDMinimalDiscrete(
            embed_dim=self.embed_dim,
            block_loc=self.block_loc,
            kwarg_all=self.kwarg_all,
            **self.factory_kwargs,
            **self.kwarg_all
        )

        
    def pad_to_block_length(self, X, block_len):
        pad_len = (block_len - X.shape[1] % block_len) % block_len
        if pad_len > 0:
            padding = torch.zeros(X.shape[0], pad_len, *X.shape[2:], dtype=X.dtype, device=X.device)
            X = torch.cat([X, padding], dim=1)
        return X


    def _forward(self, u, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, _seqlen, dim = u.shape
        u=self.pad_to_block_length(u, self.chunk_size)
        seqlen = u.shape[1]

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # 1D Convolution
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * ngroups * d_state)
        xBC = xBC[:, :seqlen, :]

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
        
        Z={
            'x':x,
            'A':A,
            'B':B,
            'C':C,
            'dt':dt,
            'chunk_size':self.chunk_size,
        }
        _, Z_ = self.ssd_minimal_discrete(u,**Z)
        y = Z_.get('y')
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y=self.norm(y * self.silu(z))
        out = self.out_proj(y)
        out = out[:, :_seqlen, :]
        return out



@gau_test
def test_mamba2layer(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    mamba2layer = Mamba2Layer(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    y,_ = mamba2layer(x)
    assert y.shape==(1,100,128)



CHILDREN_DECLARATIONS = [
    UnitDecl(
        unitname="SSDMinimalDiscrete",
        requirements="",
        inputs=['X','x','A','B','C','dt','chunk_size'],
        outputs=['Y','y'],
    ),
]



SPEC = {
    "unitname": "Mamba2Layer",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": 
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
    """,
}
    
ARGS = {
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
CHILDREN = ['SSDMinimalDiscrete']
DESC='''
''' 

