import torch.nn as nn
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from model_discovery.model.utils.modules import GABBase
from torchtune.modules import RMSNorm


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc, n_heads, rotary_pct, 
        device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.fn = MHA(embed_dim, n_heads, causal=True, rotary_emb_dim=int(
            rotary_pct * embed_dim // n_heads), **factory_kwargs)
        self.fn2 = GatedMLP(embed_dim, **factory_kwargs)
        self.norm1 = RMSNorm(embed_dim).to(device=device, dtype=dtype)
        self.norm2 = RMSNorm(embed_dim).to(device=device, dtype=dtype)

    def _forward(self, X, **kwargs):
        X = self.fn(self.norm1(X)) + X
        X = self.fn2(self.norm2(X)) + X
        return X


gab_config = {'n_heads': 8, 'rotary_pct': 0.25}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from model_discovery.model.block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)