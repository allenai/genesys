# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #



class RWKV6(GAUBase):
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, norm_eps: float = 1e-5, device=None,dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        
        # COMPLETING THE CODE HERE #
        self.hidden_size = embed_dim

        self.attn_norm = nn.LayerNorm(self.hidden_size, bias=True, eps=norm_eps, **self.factory_kwargs)
        self.attn=RWKV6Attention(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, bias=True, eps=norm_eps, **self.factory_kwargs)
        self.ffn = RWKV6FeedForward(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)   


    # YOU CAN ADD MORE FUNCTIONS HERE #


    def _forward(self,X,**Z): # type hints are optional but recommended
        # THE CODE HERE MUST BE COMPLETED #
        X1,_=self.attn(self.attn_norm(X),**Z)
        X = X1 +X
        X2,_=self.ffn(self.ffn_norm(X),**Z)
        X = X2 +X
        return X
    

@gau_test
def test_rwkv6(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    rwkv6 = RWKV6(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    Z={}
    y,Z_=rwkv6(x,**Z)
    assert y.shape==(1,100,128)



CHILDREN_DECLARATIONS = [
    UnitDecl(
        unitname="RWKV6Attention",
        requirements="",
        inputs=['X'],
        outputs=['Y'],
    ),
    UnitDecl(
        unitname="RWKV6FeedForward",
        requirements="",
        inputs=['X'],
        outputs=['Y'],
    ),
]

SPEC = {
    "unitname": "RWKV6",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
RWKV6
''',
}
ARGS = {
    'norm_eps': 1e-5,
}
CHILDREN = ['RWKV6Attention','RWKV6FeedForward']
DESC='''
''' 
