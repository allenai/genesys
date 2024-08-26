# Copyright (c) 2024, Tri Dao, Albert Gu.
from torch import nn
from torch.nn import functional as F
import inspect

from abc import ABC, abstractmethod



# Future TODO: maybe allow the agent to design topology as well, and support complicated structures
# e.g., more dimensions, not only n_block and a linear coordinate, but also in a grid, etc., or even hetereogeneous
# Philosophy of GAB: a super cube allows everything internally, but not the external topology (i.e., for i in range(n_blocks): GAB(i), a linear topology)
# It is still complete but not flexible enough (e.g., agent can design one huge macro block or depth and width related functions to implement strange topology)
class GABBase(nn.Module):
    """ Base class for Generalized Autoregressive Block """
    def __init__(self,embed_dim: int, block_loc: tuple):
        super().__init__()
        self.embed_dim = embed_dim
        self.block_loc = block_loc # location of a block within the network, (layer_idx, n_block)

    def _forward(self, X, **Z): 
        raise NotImplementedError
     
    # YOU ARE NOT ALLOW TO OVERRIDE THIS METHOD #
    def forward(self, X, **Z): # kwargs not parsable by torchscript but more flexible
        """Forward pass of the model"""
        assert len(X.shape) == 3, "Input shape must be (batch, seqlen, embed_dim)"
        assert X.shape[-1] == self.embed_dim
        Y = self._forward(X, **Z)
        if isinstance(Y, tuple):
            Y, Z = Y
        else:
            Z = {}
        assert Y.shape == X.shape, f"GAB Output shape must be the same as input shape, got {Y.shape} instead"
        assert isinstance(Z, dict), "Intermediate variables must be stored in a dict"
        return Y, Z
    

class GAUBase(nn.Module): 
    """ 
    Instead of directly giving the full implementation of a GAB block, the agent need to 
    design a series of nested GAB units and construct the full GAB block as a pipeline of these units.

    GAB is fractal, like GAB itself, each GAB unit accepts X and Z as input and returns Y and Z as output.
    """ 
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def _forward(self, X, **Z):
        raise NotImplementedError
    
    def forward(self, X, **Z):
        assert len(X.shape) == 3, "Input shape must be (batch, seqlen, embed_dim)"
        assert X.shape[-1] == self.embed_dim
        _params = inspect.signature(self._forward).parameters
        _Z = {k: v for k, v in Z.items() if k in _params}
        Y = self._forward(X, **_Z)
        if isinstance(Y, tuple):
            Y, Z_ = Y
        else:
            Z_ = {}
        assert Y.shape == X.shape, f"GAB Unit must has a sequence with the same shape as input in output, got {Y.shape} instead"
        assert isinstance(Z_, dict), "Intermediate variables must be stored in a tuple"
        Z.update(Z_) # the new intermediate variables are updated to the current Z
        return Y, Z




class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y



class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y=self.activation(y)
        y = self.fc2(y)
        return y
    