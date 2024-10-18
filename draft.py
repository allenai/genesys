import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = HybridMamba(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
import torch.nn.functional as F
import inspect


class HybridMamba(GAUBase):
    """
    HybridMamba Block

    This GAU implements the HybridMamba block, which combines Selective State Space Models (S3M) with Adaptive Sliding Window Attention (ASWA), Dynamic Gating Mechanism (DGM), Compressed Global KV-Cache (CGKV), and Adaptive Computation Controller (ACC).

    **Code Example:**

        hybrid_mamba_block = HybridMamba(embed_dim=768, block_loc=(0, 0), kwarg_all={...})
        Y, Z = hybrid_mamba_block(X, **Z)

    **Mathematical Formulation:**

    - **Input Complexity Estimation:**
        \\[
        complexity = ACC.estimate\\_complexity(X)
        \\]

    - **Selective State Space Model Processing:**
        \\[
        S3M\\_output = S3M(X)
        \\]

    - **Adaptive Window Size Determination:**
        \\[
        window\\_size = ACC.determine\\_window\\_size(complexity)
        \\]

    - **Adaptive Sliding Window Attention Processing:**
        \\[
        ASWA\\_output = ASWA(X, window\\_size, CGKV)
        \\]

    - **Dynamic Gating Mechanism:**
        \\[
        output = DGM(S3M\\_output, ASWA\\_output)
        \\]

    - **Compressed Global KV-Cache Update:**
        \\[
        CGKV.update(X, output)
        \\]

    Args:
        embed_dim (int): Dimension of the embeddings.
        block_loc (tuple): Location of the block within the network, (layer_idx, n_block).
        kwarg_all (dict): Dictionary of all keyword arguments for initializing child units.

    Returns:
        torch.Tensor: Output embeddings of shape (B, L, D).
        dict: Updated intermediate variables.

    Note:
        For more details on the components, refer to the HybridMamba proposal.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.acc = AdaptiveComputationController(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.s3m = SelectiveStateSpaceModel(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.aswa = AdaptiveSlidingWindowAttention(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.dgm = DynamicGatingMechanism(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.cgkv = CompressedGlobalKVCache(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        Y_acc, Z_acc = self.acc(X, **Z)
        complexity_score = Z_acc.get('complexity_score', None)
        window_size = Z_acc.get('window_size', None)
        if window_size is None:
            window_size = 512
        Z.update(Z_acc)
        Z['window_size'] = window_size
        Y_s3m, Z_s3m = self.s3m(X, **Z)
        Z.update(Z_s3m)
        Y_aswa, Z_aswa = self.aswa(X, **Z)
        Z.update(Z_aswa)
        Z['Y_aswa'] = Y_aswa
        Y, Z_dgm = self.dgm(Y_s3m, **Z)
        Z.update(Z_dgm)
        Y_cgkv, Z_cgkv = self.cgkv(Y, **Z)
        Z.update(Z_cgkv)
        return Y, Z


class CompressedGlobalKVCache(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        Z_ = {}
        return X, Z_


import torch.nn.functional as F
import inspect


class AdaptiveComputationController(GAUBase):
    """
    AdaptiveComputationController

    This GAU estimates the complexity of the input sequence and determines adaptive parameters
    such as the attention window size based on the input complexity.

    **Description:**

    The AdaptiveComputationController (ACC) is designed to dynamically adjust computational
    resources based on the input complexity. It analyzes the input embeddings and computes
    a complexity score, which is then used to determine adaptive parameters like the attention
    window size for other components in the model.

    **Mathematical Formulation:**

    - **Input Complexity Estimation:**

        The complexity score is computed as:

        \\[
        	ext{complexity\\_score} = \\sigma(W_2 \\cdot 	ext{ReLU}(W_1 \\cdot \\overline{X} + b_1) + b_2)
        \\]

        where:
        - \\( \\overline{X} = rac{1}{L} \\sum_{i=1}^{L} X_i \\) is the average pooling of the input embeddings.
        - \\( W_1, W_2 \\) are learnable weights.
        - \\( b_1, b_2 \\) are learnable biases.
        - \\( \\sigma \\) is the sigmoid activation function.

    - **Adaptive Parameter Determination:**

        The window size is determined based on the complexity score:

        \\[
        	ext{window\\_size} = 	ext{min\\_window\\_size} + 	ext{complexity\\_score} 	imes (	ext{max\\_window\\_size} - 	ext{min\\_window\\_size})
        \\]

    **Code Example:**

        acc = AdaptiveComputationController(embed_dim=768, block_loc=(0, 0), kwarg_all={
            'min_window_size': 128,
            'max_window_size': 512,
            'scale_factor': 1.0
        })
        Y_acc, Z = acc(X, **Z)

    **Args:**

        embed_dim (int): Dimension of the embeddings.
        block_loc (tuple): Location of the block within the network, (layer_idx, n_block).
        kwarg_all (dict): Dictionary of all keyword arguments for initializing child units.

    **Returns:**

        tuple:
            - torch.Tensor: Output embeddings (same as input X).
            - dict: Contains `complexity_score` (torch.Tensor of shape (B,)) and `window_size` (int).

    **Example:**

        >>> acc = AdaptiveComputationController(embed_dim=768, block_loc=(0, 0), kwarg_all={})
        >>> X = torch.randn(2, 1024, 768)
        >>> Y_acc, Z = acc(X)
        >>> print(Z['complexity_score'].shape)
        torch.Size([2])
        >>> print(Z['window_size'])
        256

    **Note:**

        For more info on reStructuredText docstrings, see
        `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__
        and
        `here <https://peps.python.org/pep-0287/>`__.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, min_window_size=128, max_window_size=512,
        scale_factor=1.0, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.complexity_layer = nn.Sequential(nn.Linear(embed_dim,
            embed_dim, **self.factory_kwargs), nn.ReLU(), nn.Linear(
            embed_dim, 1, **self.factory_kwargs))
        assert self.min_window_size > 0, 'min_window_size must be positive'
        assert self.max_window_size >= self.min_window_size, 'max_window_size must be greater or equal to min_window_size'
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.scale_factor = scale_factor

    def _forward(self, X, **Z):
        B, L, D = X.shape
        pooled = torch.mean(X, dim=1)
        complexity_score = torch.sigmoid(self.complexity_layer(pooled)
            ).squeeze(-1)
        window_size = self.min_window_size + complexity_score * (self.
            max_window_size - self.min_window_size)
        window_size = window_size * self.scale_factor
        window_size = window_size.clamp(min=self.min_window_size, max=self.
            max_window_size).int()
        window_size = window_size.max().item()
        Z_update = {'complexity_score': complexity_score, 'window_size':
            window_size}
        return X, Z_update


class DynamicGatingMechanism(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        Z_ = {'Y': None}
        return X, Z_


class AdaptiveSlidingWindowAttention(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        Z_ = {'Y_aswa': None}
        return X, Z_


class SelectiveStateSpaceModel(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        Z_ = {'Y_s3m': None}
        return X, Z_


gab_config = {'min_window_size': 128, 'max_window_size': 512,
    'scale_factor': 1.0}