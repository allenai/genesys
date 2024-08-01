import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase
from mamba_ssm.modules.mamba2 import Mamba2
from torchtune.modules import RMSNorm


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
        self.mamba1 = Mamba2(embed_dim, d_state, d_conv, expand, headdim,
            ngroups=ngroups, A_init_range=A_init_range, dt_min=dt_min,
            dt_max=dt_max, dt_init_floor=dt_init_floor, chunk_size=
            chunk_size, **factory_kwargs)
        self.mamba2 = Mamba2(embed_dim, d_state, d_conv, expand, headdim,
            ngroups=ngroups, A_init_range=A_init_range, dt_min=dt_min,
            dt_max=dt_max, dt_init_floor=dt_init_floor, chunk_size=
            chunk_size, **factory_kwargs)
        self.norm1 = RMSNorm(embed_dim, eps=1e-05).to(device=device, dtype=
            dtype)
        self.norm2 = RMSNorm(embed_dim, eps=1e-05).to(device=device, dtype=
            dtype)

    def _forward(self, X, **kwargs):
        hidden_states = self.norm1(X.to(dtype=self.norm1.weight.dtype))
        X = self.mamba1(hidden_states) + X
        hidden_states = self.norm2(X.to(dtype=self.norm2.weight.dtype))
        X = self.mamba2(hidden_states) + X
        return X


""" The dictionary of hyperparameters for constructing a GAB layer
    embed_dim, device, dtype should NOT be included in gab_config
"""
gab_config = {'d_state': 64, 'd_conv': 4, 'expand': 2, 'headdim': 128,
    'ngroups': 1, 'A_init_range': (1, 16), 'dt_min': 0.001, 'dt_max': 0.1,
    'dt_init_floor': 0.0001, 'chunk_size': 256}

