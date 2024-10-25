# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #




class RWKV6FeedForward(GAUBase):
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

        self.hidden_size = embed_dim
        hidden_ratio = 3.5
        intermediate_size = int(embed_dim * hidden_ratio)
        intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        kwarg_all['output_dim']=intermediate_size
        self.key = LerpLinear(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.value = nn.Linear(intermediate_size, embed_dim, bias=False, device=device, dtype=dtype)
        kwarg_all['output_dim']=embed_dim
        self.receptance = LerpLinear(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.relu=nn.ReLU()

    def _forward(self,X,**Z):
        shifted = self.time_shift(X)
        delta = shifted - X
        # key = self.act_fn(self.key(x, delta))
        _key=self.key(X,**{'delta':delta})[1]['o']
        r=self.relu(_key)
        key=r*r
        value = self.value(key)
        receptance = self.receptance(X,**{'delta':delta})[1]['o']

        return receptance.sigmoid() * value


@gau_test
def test_rwkv6feedforward(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    rwkv6feedforward = RWKV6FeedForward(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    y=rwkv6feedforward(x)
    assert y.shape==(1,100,128)


CHILDREN_DECLARATIONS = [
    UnitDecl(
        unitname="LerpLinear",
        requirements="",
        inputs=['X','delta'],
        outputs=['Y'],
    ),
]

SPEC = {
    "unitname": "RWKV6FeedForward",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
RWKV6FeedForward
''',
}
ARGS = {}
CHILDREN = ['LerpLinear']
DESC='''
''' 
