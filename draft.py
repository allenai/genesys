import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = HiPRootGAU(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase, gau_test


class HiPRootGAU(GAUBase):
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block to be composed within the network, (layer_idx, n_block), e.g. (0, 6) for the first block in a network with 6 blocks in total
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.attention_mask_generator = AttentionMaskGenerator(embed_dim=
            self.embed_dim, block_loc=self.block_loc, kwarg_all=self.
            kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.sparse_attention_computation = SparseAttentionComputation(
            embed_dim=self.embed_dim, block_loc=self.block_loc, kwarg_all=
            self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.output_projection = OutputProjection(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        _, Z = self.attention_mask_generator(X, **Z)
        Y, Z = self.sparse_attention_computation(X, **Z)
        Y, Z = self.output_projection(Y, **Z)
        return Y, Z


import torch.nn.functional as F


class SparseAttentionComputation(GAUBase):
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block to be composed within the network, (layer_idx, n_block), e.g. (0, 6) for the first block in a network with 6 blocks in total
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.attention_score_calculator = AttentionScoreCalculator(embed_dim
            =self.embed_dim, block_loc=self.block_loc, kwarg_all=self.
            kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.weighted_sum_calculator = WeightedSumCalculator(embed_dim=self
            .embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        attention_masks = Z.get('attention_masks')
        assert attention_masks is not None, 'Attention masks must be provided in Z'
        Z['attention_masks'] = attention_masks
        attention_scores, Z = self.attention_score_calculator(X, **Z)
        Z['attention_scores'] = attention_scores
        Y, Z = self.weighted_sum_calculator(X, **Z)
        return Y, Z


class WeightedSumCalculator(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        return X


class AttentionScoreCalculator(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        return X


class AttentionMaskGenerator(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        return X


class OutputProjection(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        return X


gab_config = {}





# Dependency analysis

