# gab.py

import torch
import torch.nn as nn

from mamba_ssm.modules.mha import MHA
from .utils.modules import MLP


class GAB(nn.Module):
    ''' Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    '''
    def __init__(self,embed_dim: int, n_heads, device=None,dtype=None,**kwargs):
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__()
        self.embed_dim = embed_dim
        # COMPLETING THE CODE HERE #
        self.fn = MHA(embed_dim, n_heads, causal=True, **factory_kwargs)
        self.fn2 = MLP(embed_dim, 4*embed_dim, embed_dim, **factory_kwargs)
        self.norm1 = nn.LayerNorm(embed_dim, **factory_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim, **factory_kwargs)

    def _forward(self,X,**kwargs): # type hints are optional but recommended
        ''' Forward pass of the model '''
        assert X.shape[-1] == self.embed_dim
        # COMPLETING THE CODE HERE #
        X=self.norm1(X)
        X1 = self.fn(X)
        X = X + X1
        X=self.norm2(X)
        X2 = self.fn2(X)
        X = X + X2
        return X

     
    def forward(self,X,**kwargs):
        ''' Forward pass of the model '''
        Y=self._forward(X,**kwargs)
        assert Y.shape[-1] == self.embed_dim
        return Y
    
    
def gab_config()->dict:
    ''' Returns a dictionary of hyperparameters for constructing a GAB layer
        embed_dim, layer_idx, device, dtype should not be included in the dictionary which will be provided by the model
    '''
    # COMPLETING THE CODE HERE #

    return {
        'n_heads':8
    }


# Perform registration after defining gab_config
from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=gab_config()
)(GAB)
