# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #

from typing import Optional


class LoRA(GAUBase):
    def __init__(
        self,
        embed_dim: int,
        block_loc: tuple,
        kwarg_all: dict,
        output_dim: int,
        low_rank_dim: int,
        bias: Optional[bool] = True,
        device=None,
        dtype=None,
        **kwargs
    ):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

        self.input_dim = embed_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        self.lora = nn.Sequential(
            nn.Linear(embed_dim, low_rank_dim, bias=False, device=device, dtype=dtype),
            nn.Tanh(),
            nn.Linear(low_rank_dim, output_dim, bias=bias, device=device, dtype=dtype)
        )

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"input_dim={self.input_dim}, low_rank_dim={self.low_rank_dim}, output_dim={self.output_dim}"
        if not self.bias:
            s += f", bias={self.bias}"
        s += ")"
        return s

    def _forward(self, X, **Z):
        return X, {'o':self.lora(X)}


@gau_test
def test_lora(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    lora = LoRA(embed_dim, block_loc, kwarg_all, output_dim=128, low_rank_dim=32, device=device, dtype=dtype,**kwarg_all)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    y,_=lora(x)
    assert y.shape==(1,100,128)


CHILDREN_DECLARATIONS = []

SPEC = {
    "unitname": "LoRA",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
LoRA
''',
}
ARGS = {}
CHILDREN = []
DESC = '''
''' 
