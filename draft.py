import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = RootGAU(embed_dim=embed_dim, device=device, dtype=dtype,
            **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase


class RootGAU(GAUBase):
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all current intermediate variables}
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """

    def __init__(self, embed_dim: int, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim)
        self.meta_sorting: GAUBase = MetaSortingGAU(embed_dim=embed_dim,
            device=device, dtype=dtype, **kwargs)
        self.chunked_attention: GAUBase = ChunkedAttentionGAU(embed_dim=
            embed_dim, device=device, dtype=dtype, **kwargs)
        self.hybrid_attention: GAUBase = HybridAttentionGAU(embed_dim=
            embed_dim, device=device, dtype=dtype, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.meta_sorting(X, **Z)
        X, Z = self.chunked_attention(X, **Z)
        Y, Z = self.hybrid_attention(X, **Z)
        return Y, Z


import torch.nn.functional as F


def sinkhorn_balancing(log_alpha, n_iters=5):
    """Applies Sinkhorn balancing to ensure doubly stochastic matrix."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True
            )
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True
            )
    return log_alpha


class MetaSortingGAU(GAUBase):
    """Meta Sorting Generalized Autoregressive Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all current intermediate variables}
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """

    def __init__(self, embed_dim: int, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim)
        self.sorting_network = nn.Linear(embed_dim, embed_dim, **factory_kwargs
            )

    def _forward(self, X, **Z):
        latent_permutations = self.sorting_network(X)
        balanced_permutations = sinkhorn_balancing(latent_permutations)
        balanced_permutations = F.softmax(balanced_permutations, dim=-1)
        sorted_sequence = torch.matmul(balanced_permutations, X)
        truncated_sequence = sorted_sequence
        assert truncated_sequence.shape == X.shape, 'Output shape must match input shape'
        return truncated_sequence, Z


class HybridAttentionGAU(GAUBase):

    def __init__(self, embed_dim: int, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim)

    def _forward(self, X, **Z):
        return X


class ChunkedAttentionGAU(GAUBase):

    def __init__(self, embed_dim: int, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim)

    def _forward(self, X, **Z):
        return X


gab_config = {}