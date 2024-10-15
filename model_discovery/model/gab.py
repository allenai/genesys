import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = GPT2(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


import torch.nn.functional as F
from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl


class GPT2(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.mha = ScoringNetworkGAU(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)
        self.mlp = GatedMLP(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm1 = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm2 = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)

    def _forward(self, X, **Z):
        X1, Z = self.norm1(X, **Z)
        X2, Z = self.mha(X1, **Z)
        X = X + X2
        X3, Z = self.norm2(X, **Z)
        X4, Z = self.mlp(X3, **Z)
        X = X + X4
        return X, Z


import torch.nn.functional as F
from torch import Tensor


class RMSNorm(GAUBase):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    This layer applies a variant of layer normalization that uses only the root mean square
    statistics, without centering. It's computationally more efficient than standard
    layer normalization and has been shown to be effective in various NLP tasks.

    Args:
        embed_dim (int): The size of the input feature dimension.
        block_loc (tuple): The location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the parent class.
        device (torch.device, optional): The device on which to allocate the module's parameters.
        dtype (torch.dtype, optional): The dtype of the module's parameters.
        eps (float, optional): A small constant added to the denominator for numerical stability.
            Default: 1e-5.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter of shape (embed_dim,).
        variance_epsilon (float): The epsilon value used in the normalization formula.

    Shape:
        - Input: (*, embed_dim)
        - Output: (*, embed_dim) (same shape as input)

    Examples:
        >>> rmsnorm = RMSNorm(128, (0, 6), {})
        >>> x = torch.randn(1, 100, 128)
        >>> output = rmsnorm(x)
        >>> print(output.shape)
        torch.Size([1, 100, 128])

    References:
        - Paper: "Root Mean Square Layer Normalization" by Biao Zhang and Rico Sennrich
          https://arxiv.org/abs/1910.07467
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, eps=1e-05, **kwargs):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.weight = nn.Parameter(torch.ones(embed_dim, **self.factory_kwargs)
            )
        self.variance_epsilon = eps

    def _forward(self, X, **Z):
        input_dtype = X.dtype
        X = X.to(torch.float32)
        variance = X.pow(2).mean(-1, keepdim=True)
        X = X * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * X.to(input_dtype)


import torch.nn.functional as F


class GatedMLP(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, hidden_features=None, out_features=None,
        activation=None, bias=False, multiple_of=128, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        out_features = out_features if out_features is not None else embed_dim
        hidden_features = (hidden_features if hidden_features is not None else
            int(8 * embed_dim / 3))
        hidden_features = (hidden_features + multiple_of - 1
            ) // multiple_of * multiple_of
        self.fc1 = nn.Linear(embed_dim, 2 * hidden_features, bias=bias, **
            self.factory_kwargs)
        self.activation = activation if activation is not None else F.silu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **
            self.factory_kwargs)

    def _forward(self, X, **Z):
        y = self.fc1(X)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


import torch.nn.functional as F


class ScoringNetworkGAU(GAUBase):
    """
    ScoringNetworkGAU computes relevance scores for each Key-Value (KV) pair based on the current query.
    
    **Code Example:**
    
        .. code-block:: python
    
            # Example of using ScoringNetworkGAU
            scoring_gau = ScoringNetworkGAU(embed_dim=128, block_loc=(0, 1), kwarg_all={})
            x = torch.randn(32, 100, 128)
            y, Z = scoring_gau(x)
            print(y.shape)  # Output: torch.Size([32, 100, 128])
            print(Z['scores'].shape)  # Output: torch.Size([32, 100])
    
    Args:
        embed_dim (int): Dimension of the input embeddings.
        block_loc (tuple): Location of the GAU within the network.
        kwarg_all (dict): Additional keyword arguments.
        device (torch.device, optional): Device to allocate the module.
        dtype (torch.dtype, optional): Data type of the module's parameters.
        scoring_dim (int, optional): Dimension of the scoring network's hidden layer. Default: 64.
    
    Returns:
        Tuple[Tensor, dict]: 
            - Tensor: Output embeddings Y with shape (batch_size, seq_len, embed_dim).
            - dict: Updated intermediate variables containing 'scores' with shape (batch_size, seq_len).
    
    Raises:
        NotImplementedError: If the forward method is not implemented.
    
    Example:
        >>> scoring_gau = ScoringNetworkGAU(embed_dim=128, block_loc=(0, 1), kwarg_all={})
        >>> x = torch.randn(32, 100, 128)
        >>> y, Z = scoring_gau(x)
        >>> print(y.shape)
        torch.Size([32, 100, 128])
        >>> print(Z['scores'].shape)
        torch.Size([32, 100])
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, scoring_dim: int=64, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.scoring_network = nn.Sequential(nn.Linear(embed_dim,
            scoring_dim, **self.factory_kwargs), nn.ReLU(), nn.Linear(
            scoring_dim, 1, **self.factory_kwargs))
        for param in self.scoring_network.parameters():
            param.requires_grad = True

    def _forward(self, X, **Z):
        """
        Computes relevance scores for each KV pair and modifies the input embeddings based on these scores.
        
        Args:
            X (Tensor): Input embeddings of shape (batch_size, seq_len, embed_dim).
            **Z: Intermediate variables.
        
        Returns:
            Tuple[Tensor, dict]: 
                - Y: Tensor of shape (batch_size, seq_len, embed_dim), modified based on relevance scores.
                - Z_: Dictionary containing 'scores' with shape (batch_size, seq_len).
        """
        scores = self.scoring_network(X)
        scores = scores.squeeze(-1)
        scores = torch.sigmoid(scores)
        Y = X * scores.unsqueeze(-1)
        return Y, {'scores': scores}


gab_config = {'eps': 1e-05, 'max_seq_len': 4096, 'rotary_emb_base': 10000,
    'scoring_dim': 64, 'bias': False, 'multiple_of': 128, 'hidden_features':
    None, 'out_features': None, 'activation': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)