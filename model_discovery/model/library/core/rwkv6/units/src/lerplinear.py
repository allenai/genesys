# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #

from typing import Optional



class LerpLinear(GAUBase):
    def __init__(
        self,
        embed_dim: int, 
        block_loc: tuple, 
        kwarg_all: dict, 
        output_dim: int,
        low_rank_dim: Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs
    ):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

        self.input_dim = embed_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if self.low_rank_dim is None:
            self.linear = nn.Linear(embed_dim, output_dim, bias=False, device=device, dtype=dtype)
        else:
            kwarg_all['output_dim']=output_dim
            kwarg_all['low_rank_dim']=low_rank_dim
            self.linear = LoRA(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.mu = nn.Parameter(torch.zeros(embed_dim, device=device, dtype=dtype))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}"
        if self.low_rank_dim is not None:
            s += f", low_rank_dim={self.low_rank_dim}"
        s += ")"
        return s

    def _forward(self, X: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if delta is None:
            shifted = self.time_shift(X)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - X
        if self.low_rank_dim is None:
            o = self.linear(X + delta * self.mu)
        else:
            o = self.linear(X + delta * self.mu)[1]['o']
        return X, {'o':o}


@gau_test
def test_lerplinear(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    lerplinear = LerpLinear(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    X = torch.randn(1,100,128).to(device=device,dtype=dtype)
    y=lerplinear(X)
    assert y.shape==(1,100,128)


CHILDREN_DECLARATIONS = [
    UnitDecl(
        unitname="LoRA",
        requirements="",
        inputs=['X'],
        outputs=['Y'],
    ),
]

SPEC = {
    "unitname": "LerpLinear",
    "inputs": ['X','delta'],
    "outputs": ['Y'],
    "document": '''
LerpLinear
''',
}

ARGS = {}
CHILDREN = ['LoRA']
DESC='''
''' 
