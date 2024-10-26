# GAU_IMPLEMENTATION_FILE   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE TO ALLOW PARSER DETECT THE GAU IMPLEMENTATION FILES #

import torch    
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase,gau_test,UnitDecl # DO NOT CHANGE THIS IMPORT STATEMENT #



class Mamba2(GAUBase): 
    """
    Mamba2: A Generalized Autoregressive Unit (GAU) implementing a double-layer Mamba architecture.

    This class represents a Mamba2 block, which consists of two Mamba layers with normalization.
    It's designed to process sequential data in a causal, differentiable, and parallelizable manner.

    Architecture:
        1. Input Normalization (RMSNorm)
        2. First Mamba Layer
        3. Residual Connection
        4. Second Normalization (RMSNorm)
        5. Second Mamba Layer
        6. Final Residual Connection

    Args:
        embed_dim (int): The dimensionality of the input and output embeddings.
        block_loc (tuple): The location of this block within the larger model architecture.
        kwarg_all (dict): Additional keyword arguments to be passed to child components.
        device (torch.device, optional): The device on which to allocate tensors.
        dtype (torch.dtype, optional): The default dtype for tensors in this module.

    Inputs:
        X (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).
        **Z: Additional keyword arguments for potential future extensions.

    Outputs:
        X (torch.Tensor): Output tensor of shape (batch_size, sequence_length, embed_dim).
        Z (dict): Updated keyword arguments.

    Note:
        This implementation adheres to the GAU (Generalized Autoregressive Unit) interface
        and maintains causal properties for autoregressive processing.
    """
    def __init__(self,embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None,dtype=None, **kwargs): 
        self.factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim, block_loc, kwarg_all) # DO NOT CHANGE THIS LINE #
        self.mamba1=Mamba2Layer(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.mamba2=Mamba2Layer(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.norm1=RMSNorm(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.norm2=RMSNorm(embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)   


    def _forward(self,X,**Z): # type hints are optional but recommended
        X1,Z=self.norm1(X,**Z)
        X2,Z=self.mamba1(X1,**Z)
        X=X+X2
        X3,Z=self.norm2(X,**Z)
        X4,Z=self.mamba2(X3,**Z)
        X=X+X4
        return X,Z
        
    
@gau_test
def test_mamba2(device=None,dtype=None):
    embed_dim=128
    block_loc=(0,6)
    kwarg_all={}
    mamba2 = Mamba2(embed_dim, block_loc, kwarg_all, device=device, dtype=dtype,**kwarg_all)
    x = torch.randn(1,100,128).to(device=device,dtype=dtype)
    Z={}
    y,Z_=mamba2(x,**Z)
    assert y.shape==(1,100,128)


CHILDREN_DECLARATIONS = [
    UnitDecl(
        unitname="Mamba2Layer",
        requirements="",
        inputs=['X'],
        outputs=['Y'],
    ),
    UnitDecl(
        unitname="RMSNorm",
        requirements="",
        inputs=['X'],
        outputs=['Y'],
    ),
]


SPEC = {
    "unitname": "Mamba2",
    "inputs": ['X'],
    "outputs": ['Y'],
    "document": 
    """
    Mamba2: A Generalized Autoregressive Unit (GAU) implementing a double-layer Mamba architecture.

    This class represents a Mamba2 block, which consists of two Mamba layers with normalization.
    It's designed to process sequential data in a causal, differentiable, and parallelizable manner.

    Architecture:
        1. Input Normalization (RMSNorm)
        2. First Mamba Layer
        3. Residual Connection
        4. Second Normalization (RMSNorm)
        5. Second Mamba Layer
        6. Final Residual Connection

    Args:
        embed_dim (int): The dimensionality of the input and output embeddings.
        block_loc (tuple): The location of this block within the larger model architecture.
        kwarg_all (dict): Additional keyword arguments to be passed to child components.
        device (torch.device, optional): The device on which to allocate tensors.
        dtype (torch.dtype, optional): The default dtype for tensors in this module.

    Inputs:
        X (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).
        **Z: Additional keyword arguments for potential future extensions.

    Outputs:
        X (torch.Tensor): Output tensor of shape (batch_size, sequence_length, embed_dim).
        Z (dict): Updated keyword arguments.

    Note:
        This implementation adheres to the GAU (Generalized Autoregressive Unit) interface
        and maintains causal properties for autoregressive processing.
    """,
}
ARGS = {}
CHILDREN = ['Mamba2Layer','RMSNorm']
DESC='''
''' 
