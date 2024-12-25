import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from torchtune.modules import RMSNorm

from mamba_ssm.modules.mamba2 import Mamba2



class GAB(GABBase):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """

    def __init__(self, embed_dim: int, block_loc, device=None, dtype=None,
        d_state=64, d_conv=4, expand=2, headdim=128, ngroups=1,
        A_init_range=(1, 16), dt_min=0.001, dt_max=0.1, dt_init_floor=
        0.0001, chunk_size=256, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.mamba1 = Mamba2(embed_dim,d_state=d_state,d_conv=d_conv,conv_init=None,
            expand=expand,headdim=headdim,ngroups=ngroups,A_init_range=A_init_range,
            dt_min=dt_min,dt_max=dt_max,dt_init_floor=dt_init_floor,
            chunk_size=chunk_size,**factory_kwargs
        )
        self.mamba2 = Mamba2(embed_dim,d_state=d_state,d_conv=d_conv,conv_init=None,
            expand=expand,headdim=headdim,ngroups=ngroups,A_init_range=A_init_range,
            dt_min=dt_min,dt_max=dt_max,dt_init_floor=dt_init_floor,
            chunk_size=chunk_size,**factory_kwargs
        )
        self.norm1 = RMSNorm(embed_dim, eps=1e-05).to(**factory_kwargs)
        self.norm2 = RMSNorm(embed_dim, eps=1e-05).to(**factory_kwargs)

    def _forward(self, X, **kwargs):
        X = self.mamba1(self.norm1(X)) + X
        X = self.mamba2(self.norm2(X)) + X
        return X


gab_config = {'d_state': 64, 'd_conv': 4, 'expand': 2, 'headdim': 128,
    'ngroups': 1, 'A_init_range': (1, 16), 'dt_min': 0.001, 'dt_max': 0.1,
    'dt_init_floor': 0.0001, 'chunk_size': 256}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from model_discovery.model.block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)