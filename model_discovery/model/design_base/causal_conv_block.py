# gab.py

import torch
import torch.nn as nn
from ..block_registry import BlockRegister


__all__ = [
    "GAB",
]

@BlockRegister(
    name="causal_conv_block",
    config={}
)
class GAB(nn.Module):
    ''' Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    '''
    def __init__(self, embed_dim: int, device=None, dtype=None, **kwargs):
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype}  # remember to pass it to nn layers
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_groups = 4  # Grouped convolutions

        # Causal convolution
        self.causal_conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, groups=self.num_groups, **factory_kwargs)
        self.causal_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, groups=self.num_groups, **factory_kwargs)

        self.norm1 = nn.LayerNorm(embed_dim, **factory_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim, **factory_kwargs)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, **factory_kwargs), 
            nn.ReLU(), 
            nn.Linear(4 * embed_dim, embed_dim, **factory_kwargs)
        )
        
    def _forward(self, X, **kwargs):
        ''' Forward pass of the model '''
        assert X.shape[-1] == self.embed_dim

        X = X.transpose(1, 2)  # Transpose to (batch, embed_dim, seqlen) for Conv1d
        
        # Causal convolution with slicing to maintain sequence length
        X = self.causal_conv1(X)[:, :, :-2]
        X = X.transpose(1, 2)  # Transpose back to (batch, seqlen, embed_dim)
        X = self.norm1(X)
        X = torch.relu(X)

        X = X.transpose(1, 2)  # Again transpose for the second convolution
        X = self.causal_conv2(X)[:, :, :-2]
        X = X.transpose(1, 2)
        X = self.norm2(X)
        X = torch.relu(X)
        
        Y = self.ffn(X)
        return Y
    
    def forward(self, X, **kwargs):
        ''' Forward pass of the model '''
        Y = self._forward(X, **kwargs)
        assert Y.shape[-1] == self.embed_dim
        return Y
    

def gab_config() -> dict:
    ''' Returns a dictionary of hyperparameters for constructing a GAB layer
        embed_dim, device, dtype should not be included in the dictionary which will be provided by the model
    '''
    return {
        # No additional hyperparameters for now.
    }


# GAB: {Input -> [C1 -> LN1 -> ReLU1 -> X1]
#            -> [X1 -> C2 -> LN2 -> ReLU2 -> X2]
#            -> [X2 + X1] -> FFN(F1 -> ReLU -> F2) -> Output}
