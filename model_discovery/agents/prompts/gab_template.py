# gab.py

import torch
import torch.nn as nn


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #

class GAB(nn.Module):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """
    def __init__(self,embed_dim: int, device=None,dtype=None,**kwargs): # YOU CAN ADD MORE ARGUMENTS #
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__()
        self.embed_dim = embed_dim
        # COMPLETING THE CODE HERE #

        raise NotImplementedError


    # YOU CAN ADD MORE FUNCTIONS HERE #


    def _forward(self,X,**kwargs): # type hints are optional but recommended
        """Forward pass of the model"""
        assert X.shape[-1] == self.embed_dim
        # THE CODE HERE MUST BE COMPLETED #

        raise NotImplementedError
     
    # YOU ARE NOT ALLOWED TO CHANGE THIS FUNCTION #
    def forward(self,X,**kwargs):
        """Forward pass of the model"""
        Y=self._forward(X,**kwargs)
        assert Y.shape[-1] == self.embed_dim
        return Y
    # ------------------------------------------ #
    
    
def gab_config()->dict: # THE ARGUMENTS MUST MATCH THE ADDITIONAL ARGUMENTS YOU DEFINE IN GAB CLASS #
    """Returns a dictionary of hyperparameters for constructing a GAB layer
        embed_dim, device, dtype should not be included in the dictionary which will be provided by the model
    """
    # THE CODE HERE MUST BE COMPLETED #

    raise NotImplementedError
