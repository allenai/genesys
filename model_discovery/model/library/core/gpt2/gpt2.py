import torch.nn as nn
try:
    from mamba_ssm.modules.mha import MHA
    from mamba_ssm.modules.mlp import GatedMLP
except:
    MHA=None
    GatedMLP=None   

from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #

from torchtune.modules import RMSNorm

class GAB(GABBase):
    def __init__(self,embed_dim: int, n_heads=8, rotary_pct=0.25, device=None,dtype=None,**kwargs):
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim)
        self.fn = MHA(embed_dim, n_heads, causal=True, rotary_emb_dim=int(rotary_pct*embed_dim//n_heads), **factory_kwargs)
        self.fn2 = GatedMLP(embed_dim, **factory_kwargs)
        self.norm1 = RMSNorm(embed_dim).to(device=device,dtype=dtype)
        self.norm2 = RMSNorm(embed_dim).to(device=device,dtype=dtype)

    def _forward(self,X,**Z): # type hints are optional but recommended
        X = self.fn(self.norm1(X))+X
        X = self.fn2(self.norm2(X))+X
        return X, Z
    
gab_config = {
    'n_heads':8, 
    'rotary_pct':0.25
}

# autoconfig={}

# block_config=gab_config
# block_config.update(autoconfig)

# from .block_registry import BlockRegister

# BlockRegister(
#     name="default",
#     config=block_config
# )(GAB)