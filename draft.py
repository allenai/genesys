import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}      
        super().__init__(embed_dim, block_loc)
        self.root = DSAB(embed_dim=embed_dim, device=device, dtype=dtype,
            **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase


class DSAB(GAUBase):
    """Dynamic Sparse Attention Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all current intermediate variables}
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z} 
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """

    def __init__(self, embed_dim: int, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}      
        super().__init__(embed_dim)
        self.sparse_attenti