# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #

from transformers.activations import ACT2FN


class RetNetMLP(GAUBase): 
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None,
            hidden_size=None,**kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        hidden_size = hidden_size if hidden_size is not None else embed_dim
        self.hidden_size = hidden_size 
        # the final number of params is `hidden_ratio * hidden_size^2`
        hidden_ratio = 2
        intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
        intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = ACT2FN['swish']

    def _forward(self, X,**Z):
        y = self.gate_proj(X)
        gate, y = y.chunk(2, -1)
        z=self.act_fn(gate)*y
        x = self.down_proj(z)
        return x


@gau_test
def test_gatedmlp(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={'hidden_size':128}
    retnetmlp=RetNetMLP(embed_dim,block_loc,kwarg_all,device=device,dtype=dtype,**kwarg_all)
    x=torch.randn(1,100,128).to(device=device,dtype=dtype)
    Z={}
    y,Z_=retnetmlp(x,**Z)
    assert y.shape==(1,100,128)


CHILDREN_DECLARATIONS = []


SPEC ={
    "unitname": "RetNetMLP",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
RetNet MLP
''',
}
ARGS = {
    "hidden_size":None,
}
CHILDREN = []
DESC='''
'''

