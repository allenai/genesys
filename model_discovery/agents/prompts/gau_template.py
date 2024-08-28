# gau.py   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE #

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GAUBase # DO NOT CHANGE THIS IMPORT STATEMENT #


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #


class GAU(GAUBase): # DO NOT CHANGE THE NAME OF THIS CLASS
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block within the network, (layer_idx, n_block)
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """
    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None, dtype=None,**kwargs): # YOU CAN ADD MORE ARGUMENTS WITH OPTIONAL DEFAULT VALUES, BUT YOU HAVE TO HAVE embed_dim, device, dtype AS THE ARGUTMENTS #
        self.factory_kwargs = {"device": device, "dtype": dtype} # DO NOT CHANGE THIS LINE, remember to pass it
        super().__init__(embed_dim, block_loc, kwarg_all) # DO NOT CHANGE THIS LINE #
        
        # COMPLETING THE CODE HERE #

        raise NotImplementedError


    # YOU CAN ADD MORE FUNCTIONS HERE #


    def _forward(self, X, **Z): 
        
        # THE CODE HERE MUST BE COMPLETED #

        raise NotImplementedError
    