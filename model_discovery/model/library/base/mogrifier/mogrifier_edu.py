from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module

from einops import repeat, pack, unpack

# constants

Linear = nn.Linear

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# maybe factorized projection

def weight(
    dim_in,
    dim_out,
    k: int | None = None
):
    if not exists(k):
        return Linear(dim_in, dim_out)

    assert k < dim_in and k < dim_out, 'k must be of relative lower rank'

    return nn.Sequential(
        Linear(dim_in, k),
        Linear(k, dim_out)
    )

# main class

class Mogrifier(Module):
    def __init__(
        self,
        dim: int,
        iters = 5,
        factorize_k: int | None = None,
        dim_hidden: int | None = None,
        hidden_factorize_k: int | None = None,
    ):
        super().__init__()
        assert iters > 1

        self.dim = dim

        dim_hidden = default(dim_hidden, dim)
        self.dim_hidden = dim_hidden

        self.iters = iters

        self.Q = nn.Sequential(
            weight(dim_hidden, dim, factorize_k),
            nn.Sigmoid()
        )

        factorize_k = default(hidden_factorize_k, factorize_k)

        self.R = nn.Sequential(
            weight(dim, dim_hidden, factorize_k),
            nn.Sigmoid()
        )

    def forward(
        self,
        inputs: Tensor,
        hiddens: Tensor,
        iters: int | None = None
    ):
        iters = default(iters, self.iters)

        if inputs.ndim == 3 and hiddens.ndim == 2:
            hiddens = repeat(hiddens, 'b d -> b n d', n = inputs.shape[-2])

        assert inputs.shape[-1] == self.dim
        assert hiddens.shape[-1] == self.dim_hidden
        assert inputs.shape[:-2] == hiddens.shape[:-2]

        (inputs, packed_shape), (hiddens, _) = tuple(pack([t], '* d') for t in (inputs, hiddens))

        for ind in range(self.iters):
            is_even = (ind % 2) == 0

            if is_even:
                inputs = 2 * self.Q(hiddens) * inputs
            else:
                hiddens = 2 * self.R(inputs) * hiddens

        inputs, hiddens = tuple(unpack(t, packed_shape, '* d')[0] for t in (inputs, hiddens))
        return inputs, hiddens