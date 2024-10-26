# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #

from torchtune.modules import RMSNorm



class RetNet(GAUBase): 
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None, 
            norm_eps: float = 1e-6,**kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

        self.hidden_size = embed_dim
        self.attn_norm = RMSNorm(self.hidden_size, eps=norm_eps).to(device=device, dtype=dtype)
        self.attn = MultiScaleRetention(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)   
        self.mlp_norm = RMSNorm(self.hidden_size, eps=norm_eps).to(device=device, dtype=dtype)
        self.mlp = RetNetMLP(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)   


    def _forward(self,X,**Z): # type hints are optional but recommended
        hidden_states = self.attn_norm(X)
        X = self.attn(hidden_states,**Z)[0]+X
        hidden_states = self.mlp_norm(X)
        X = self.mlp(hidden_states,**Z)[0]+X
        return X,Z


@gau_test
def test_retnet(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    retnet = RetNet(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    Z={}
    y,Z_=retnet(x,**Z)
    assert y.shape==(1,100,128)


CHILDREN_DECLARATIONS = [
    UnitDecl(
        unitname="MultiScaleRetention",
        requirements="",
        inputs=['X'],
        outputs=['Y'],
    ),
    UnitDecl(
        unitname="RetNetMLP",
        requirements="",
        inputs=['X'],
        outputs=['Y'],
    ),
]


SPEC = {
    "unitname": "RetNet",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
RetNet
''',
}
ARGS = {
    "norm_eps":1e-6,
}
CHILDREN = ['MultiScaleRetention','RetNetMLP']
DESC='''
''' 
