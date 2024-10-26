# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #

from typing import Any, Dict, Optional, Tuple, Union

import torch.nn.functional as F

from transformers.utils import logging
from transformers.activations import ACT2FN
# from transformers.utils.import_utils import is_causal_conv1d_available # CAUSE EARLY INIT CUDA


logger = logging.get_logger(__name__)


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #




class SwiGluMLP(GAUBase): 
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None, 
                 intermediate_size=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

        self.hidden_size = embed_dim
        self.intermediate_size = intermediate_size if intermediate_size is not None else int(embed_dim*2.5)                                  
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False,**self.factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False,**self.factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False,**self.factory_kwargs)
        self.act_fn = ACT2FN['silu']

    def _forward(self, X, **Z):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(X)) * self.up_proj(X))
        return down_proj



@gau_test
def test_swiglumlp(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    swiglumlp = SwiGluMLP(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    y = swiglumlp(x)
    assert y.shape==(1,100,128)



CHILDREN_DECLARATIONS = [
]

SPEC = {
    "unitname": "SwiGluMLP",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
SwiGluMLP
''',
}
ARGS = {
    'intermediate_size':None,
}
CHILDREN = []
DESC='''
''' 
