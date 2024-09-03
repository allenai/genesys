import torch    
import torch.nn as nn

from model_discovery.model.utils.modules import GAUBase,gau_test # DO NOT CHANGE THIS IMPORT STATEMENT #

import torch.nn.functional as F




class GatedMLP(GAUBase):
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None,
            hidden_features=None,out_features=None,activation=None,bias=False,multiple_of=128,**kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        out_features = out_features if out_features is not None else embed_dim
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * embed_dim / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(embed_dim, 2 * hidden_features, bias=bias, **self.factory_kwargs)
        self.activation = activation if activation is not None else F.silu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **self.factory_kwargs)

    def _forward(self, X, **Z):
        y = self.fc1(X)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


@gau_test
def test_gatedmlp(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,1)
    kwarg_all={'hidden_features':128,'out_features':128,'activation':F.silu,'bias':False,'multiple_of':128}
    gatedmlp=GatedMLP(embed_dim,block_loc,kwarg_all,device=device,dtype=dtype)
    x=torch.randn(1,128)
    y=gatedmlp(x)
    assert y.shape==(1,128)



SPEC ={
    "unitname": "GatedMLP",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
Gated MLP
''',
}
ARGS = {
    "hidden_features":None,
    "out_features":None,
    "activation":None,
    "bias":False,
    "multiple_of":128,
}
CHILDREN = []
DESC='''
'''

