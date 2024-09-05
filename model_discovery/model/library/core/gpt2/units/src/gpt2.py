import torch    
import torch.nn as nn

from model_discovery.model.utils.modules import GAUBase,gau_test # DO NOT CHANGE THIS IMPORT STATEMENT #




class GPT2(GAUBase):
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.mha=MHA(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.mlp=GatedMLP(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.norm1=RMSNorm(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.norm2=RMSNorm(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)   

    def _forward(self,X,**Z): # type hints are optional but recommended
        X1,Z=self.norm1(X,**Z)
        X2,Z=self.mha(X1,**Z)
        X=X+X2
        X3,Z=self.norm2(X,**Z)
        X4,Z=self.mlp(X3,**Z)
        X=X+X4
        return X,Z


@gau_test
def test_gpt2():
    gpt2 = GPT2(embed_dim=128, block_loc=(0,0), kwarg_all={})
    x = torch.randn(1,128)
    y = gpt2(x)
    assert y.shape==(1,128)


SPEC = {
    "unitname": "GPT2",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": '''
GPT2
''',
}
ARGS = {
    "embed_dim":128,
    "block_loc":(0,0),
    "kwarg_all":{},
}
CHILDREN = ['MHA','GatedMLP','RMSNorm']
DESC='''
''' 
