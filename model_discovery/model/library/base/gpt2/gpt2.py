import torch.nn as nn
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #

from torchtune.modules import RMSNorm

class GAB(GABBase):
    def __init__(self,embed_dim: int, n_heads, rotary_pct, device=None,dtype=None,**kwargs):
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim)
        self.fn = MHA(embed_dim, n_heads, causal=True, rotary_emb_dim=int(rotary_pct*embed_dim//n_heads), **factory_kwargs)
        self.fn2 = GatedMLP(embed_dim, **factory_kwargs)
        self.norm1 = RMSNorm(embed_dim, **factory_kwargs)
        self.norm2 = RMSNorm(embed_dim, **factory_kwargs)

    def _forward(self,X,**kwargs): # type hints are optional but recommended
        X = self.fn(self.norm1(X))+X
        X = self.fn2(self.norm2(X))+X
        return X
    
gab_config = {'n_heads':8, 'rotary_pct':0.25}