
import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):
    '''
    GAU Tree Map of Butterfly-AFT Generalized Autoregressive Block (BAGAB):
    ButterflyAFTGAU (Rating: 3.5/5)
        |- AFTMechanismGAU (Rating: 3.5/5)
            |- PositionBiasGAU (Rating: 3.5/5)
            |- ElementWiseOperationGAU (Rating: 3.5/5)
        |- ButterflyTransformGAU (Rating: 3.5/5)
            |- ButterflyLayerGAU (Rating: 3.5/5)
                |- ButterflyStageGAU (Rating: 3.5/5)
                    |- ButterflyParameterGAU (Unimplemented)
            |- ButterflyInitializationGAU (Rating: 3.5/5)
                |- ButterflyStageGAU (Rating: 3.5/5)
                    |- ButterflyParameterGAU (Unimplemented)
                |- ButterflyParameterGAU (Unimplemented)

    Implemented Units: AFTMechanismGAU, ButterflyLayerGAU, PositionBiasGAU, ButterflyTransformGAU, ElementWiseOperationGAU, ButterflyAFTGAU, ButterflyInitializationGAU, ButterflyStageGAU
    Unimplemented Units: AFTMechanismGAU, ButterflyLayerGAU, PositionBiasGAU, ButterflyTransformGAU, ButterflyParameterGAU, ElementWiseOperationGAU, ButterflyInitializationGAU, ButterflyStageGAU
    '''

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = ButterflyAFTGAU(embed_dim=embed_dim, block_loc=
            block_loc, kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


from model_discovery.model.utils.modules import GAUBase


class ButterflyAFTGAU(GAUBase):
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block within the network, (layer_idx, n_block)
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.aft_mechanism = AFTMechanismGAU(embed_dim=embed_dim, block_loc
            =block_loc, kwarg_all=kwarg_all, **self.factory_kwargs, **kwarg_all
            )
        self.butterfly_transform = ButterflyTransformGAU(embed_dim=
            embed_dim, block_loc=block_loc, kwarg_all=kwarg_all, **self.
            factory_kwargs, **kwarg_all)

    def _forward(self, X, **Z):
        aft_output, Z = self.aft_mechanism(X, **Z)
        butterfly_output, Z = self.butterfly_transform(aft_output, **Z)
        return butterfly_output, Z


class ButterflyTransformGAU(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.butterfly_layer = ButterflyLayerGAU(embed_dim=embed_dim,
            block_loc=block_loc, kwarg_all=kwarg_all, **self.factory_kwargs,
            **kwarg_all)
        self.butterfly_init = ButterflyInitializationGAU(embed_dim=
            embed_dim, block_loc=block_loc, kwarg_all=kwarg_all, **self.
            factory_kwargs, **kwarg_all)

    def _forward(self, X, **Z):
        X, Z = self.butterfly_init(X, **Z)
        Y, Z = self.butterfly_layer(X, **Z)
        return Y, Z


class ButterflyLayerGAU(GAUBase):
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block within the network, (layer_idx, n_block)
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.stages = nn.ModuleList([ButterflyStageGAU(embed_dim=embed_dim,
            block_loc=block_loc, kwarg_all=kwarg_all, **self.factory_kwargs,
            **kwarg_all) for _ in range(kwarg_all.get('num_stages', 4))])

    def _forward(self, X, **Z):
        for stage in self.stages:
            X, Z = stage(X, **Z)
        return X, Z

class ButterflyMatrixGAU(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        return X


import torch.nn.functional as F


class ButterflyInitializationGAU(GAUBase):
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block within the network, (layer_idx, n_block)
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.butterfly_size = kwarg_all.get('butterfly_size', embed_dim)
        self.butterfly_matrices = nn.ParameterList([nn.Parameter(torch.
            randn(self.butterfly_size, self.butterfly_size, **self.
            factory_kwargs)) for _ in range(int(torch.log2(torch.tensor(
            self.butterfly_size))))])
        self.butterfly_stage = ButterflyStageGAU(embed_dim=embed_dim,
            block_loc=block_loc, kwarg_all=kwarg_all, **self.factory_kwargs,
            **kwarg_all)
        self.butterfly_parameter = ButterflyParameterGAU(embed_dim=
            embed_dim, block_loc=block_loc, kwarg_all=kwarg_all, **self.
            factory_kwargs, **kwarg_all)

    def _forward(self, X, **Z):
        for matrix in self.butterfly_matrices:
            X = F.linear(X, matrix)
        return X, Z


class ButterflyParameterGAU(GAUBase):
    """
    Generalized Autoregressive Block Unit for learning parameters of Butterfly Factorization.
    Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
    Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
    Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

    embed_dim:    The dimension of the input embeddings
    block_loc:    The location of the block within the network, (layer_idx, n_block)
    kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.param_size = kwarg_all.get('param_size', 128)
        self.butterfly_weights = nn.Parameter(torch.randn(self.param_size,
            embed_dim, **self.factory_kwargs))
        nn.init.xavier_uniform_(self.butterfly_weights)

    def _forward(self, X, **Z):
        transformed_X = torch.matmul(X, self.butterfly_weights)
        Z_ = {'butterfly_weights': self.butterfly_weights}
        return transformed_X, Z_


class ButterflyComputationGAU(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)

    def _forward(self, X, **Z):
        return X


class ButterflyStageGAU(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.butterfly_params = ButterflyParameterGAU(embed_dim=embed_dim,
            block_loc=block_loc, kwarg_all=kwarg_all, **self.factory_kwargs,
            **kwarg_all)

    def _forward(self, X, **Z):
        transformed_X, Z = self.butterfly_params(X, **Z)
        assert transformed_X.shape == X.shape, f'Output shape {transformed_X.shape} does not match input shape {X.shape}'
        return transformed_X, Z

class AFTMechanismGAU(GAUBase):
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block within the network, (layer_idx, n_block)
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.position_bias = PositionBiasGAU(embed_dim=embed_dim, block_loc
            =block_loc, kwarg_all=kwarg_all, **self.factory_kwargs, **kwarg_all
            )
        self.element_wise_operation = ElementWiseOperationGAU(embed_dim=
            embed_dim, block_loc=block_loc, kwarg_all=kwarg_all, **self.
            factory_kwargs, **kwarg_all)

    def _forward(self, X, **Z):
        X, Z = self.position_bias(X, **Z)
        Y, Z = self.element_wise_operation(X, **Z)
        return Y, Z


class ElementWiseOperationGAU(GAUBase):
    """Generalized Autoregressive Block Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block within the network, (layer_idx, n_block)
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.position_bias = nn.Parameter(torch.zeros(1, embed_dim, **self.
            factory_kwargs))

    def _forward(self, X, **Z):
        queries = Z.get('queries', X)
        keys = Z.get('keys', X)
        values = Z.get('values', X)
        elementwise_product = queries * keys
        elementwise_sum = elementwise_product + values + self.position_bias
        return elementwise_sum, {}


class PositionBiasGAU(GAUBase):
    """Position Bias Generalized Autoregressive Unit
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all intermediate variables} 
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable...

        embed_dim:    The dimension of the input embeddings
        block_loc:    The location of the block within the network, (layer_idx, n_block)
        kwarg_all:    A dictionary of all hyperparameters across all units, use it instead of kwargs to initialize the children units
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.position_bias = nn.Parameter(torch.zeros(1, 1, embed_dim, **
            self.factory_kwargs))

    def _forward(self, X, **Z):
        Y = X + self.position_bias
        return Y, {}


gab_config = {}

