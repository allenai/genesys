# gau.py   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE #

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GAUBase, gau_test # DO NOT CHANGE THIS IMPORT STATEMENT #


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #


class GAU(GAUBase): # DO NOT CHANGE THE NAME OF THIS CLASS!!! #
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block to be composed within the network, (layer_idx, n_block), e.g. (0, 6) for the first block in a network with 6 blocks in total
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """
    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None, dtype=None,**kwargs): # YOU CAN ADD MORE ARGUMENTS WITH OPTIONAL DEFAULT VALUES, BUT KEEP THE ORIGINAL ONES #
        self.factory_kwargs = {"device": device, "dtype": dtype} # DO NOT CHANGE THIS LINE, REMEMBER TO PASS IT #
        super().__init__(embed_dim, block_loc, kwarg_all) # DO NOT CHANGE THIS LINE #
        
        # COMPLETING THE CODE HERE #

        raise NotImplementedError


    # YOU CAN ADD MORE FUNCTIONS HERE #


    def _forward(self, X, **Z): 
        
        # THE CODE HERE MUST BE COMPLETED #

        raise NotImplementedError


# WRITE YOUR UNIT TEST FUNCTIONS HERE #

@gau_test # DO NOT CHANGE THIS DECORATOR, OTHERWISE IT WON'T BE RECOGNIZED AS A UNIT TEST #
def unittest(device=None, dtype=None)->None: # YOU CAN RENAME THE FUNCTION, BUT DO NOT CHANGE THE ARGUMENTS, IT SHOULD ALSO NOT RETURN ANYTHING #
    # AN AVAILABLE DEVICE AND DTYPE ARE PASSED AS ARGUMENTS, USE THEM TO INITIALIZE YOUR GAU AND MOCK INPUTS #

    # WRITE ASSERTIONS TO PERFORM THE TEST, USE PRINT TO DEBUG #
    
    raise NotImplementedError
