# Copyright (c) 2024, Tri Dao, Albert Gu.
from torch import nn
from torch.nn import functional as F

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

    def _forward(self, X, **intermediate_vars): 
        raise NotImplementedError
     
    # YOU ARE NOT ALLOW TO OVERRIDE THIS METHOD #
    def forward(self, X, **intermediate_vars):
        """Forward pass of the model"""
        assert len(X.shape) == 3, "Input shape must be (batch, seqlen, embed_dim)"
        assert X.shape[-1] == self.embed_dim
        Y=self._forward(X,**intermediate_vars)
        if isinstance(Y, tuple):
            intermediate_vars = Y[1:]
            Y = Y[0]
        else:
            intermediate_vars = {}
        assert Y.shape == X.shape, f"GAB Output shape must be the same as input shape, got {Y.shape} instead"
        return Y, intermediate_vars


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
    