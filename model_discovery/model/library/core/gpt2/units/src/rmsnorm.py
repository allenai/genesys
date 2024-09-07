import torch    
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #



class RMSNorm(GAUBase): # DO NOT CHANGE THIS CLASS NAME #
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None,
            eps=1e-5, **kwargs):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.weight = nn.Parameter(torch.ones(embed_dim,**self.factory_kwargs))
        self.variance_epsilon = eps

    def _forward(self, X, **Z):
        input_dtype = X.dtype
        X = X.to(torch.float32)
        variance = X.pow(2).mean(-1, keepdim=True)
        X = X * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * X.to(input_dtype)



@gau_test
def test_rmsnorm(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    rmsnorm = RMSNorm(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    Z={}
    y,Z_=rmsnorm(x,**Z)
    assert y.shape==(1,100,128)


CHILDREN_DECLARATIONS = []

SPEC = {
    "unitname": "RMSNorm",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
RMSNorm
''',
}
ARGS = {
    "eps":1e-5,
}
CHILDREN = []
DESC='''
'''

