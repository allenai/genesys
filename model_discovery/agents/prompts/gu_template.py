# UNIT_NAME.py
# replace UNIT_NAME with the unit_name of your unit

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GABUnit # DO NOT CHANGE THIS IMPORT STATEMENT #


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #


class UNIT_NAME(GABUnit): # Replace UNIT_NAME with the unit_name of your unit
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all current intermediate variables}
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """
    def __init__(self, embed_dim: int, device=None, dtype=None,**kwargs): # YOU CAN ADD MORE ARGUMENTS, BUT YOU HAVE TO HAVE embed_dim, device, dtype AS THE ARGUTMENTS #
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to all nn layers
        super().__init__(embed_dim) # DO NOT CHANGE THIS LINE #
        
        # COMPLETING THE CODE HERE #

        raise NotImplementedError


    # YOU CAN ADD MORE FUNCTIONS HERE #


    def _forward(self, X, **Z): 
        
        # THE CODE HERE MUST BE COMPLETED #

        raise NotImplementedError
    
    


### Decouple hyperparameters from the unit design
# """ 
# The dictionary of hyperparameters for constructing this unit, the config off all
# units will be merged together and passed to the GAB class
# """
# unit_config = {
#     # THE HYPERPARAMETERS OF ADDITIONAL ARGUMENTS IN GAB CLASS #
# }

