import torch.nn as nn
from mamba_ssm.modules.mha import MHA
from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #
from model_discovery.model.utils.modules import MLP 

class GAB(GABBase):
    def __init__(self,embed_dim: int, n_heads, device=None,dtype=None,**kwargs):
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim)
        self.fn = MHA(embed_dim, n_heads, causal=True, rotary_emb_dim=embed_dim, **factory_kwargs)
        self.fn2 = MLP(embed_dim, 4*embed_dim, embed_dim, **factory_kwargs)
        self.norm1 = nn.LayerNorm(embed_dim, **factory_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim, **factory_kwargs)

    def _forward(self,X,**kwargs): # type hints are optional but recommended
        X = self.fn(self.norm1(X))+X
        X = self.fn2(self.norm2(X))+X
        return X
    
gab_config = {'n_heads':8}