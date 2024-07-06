# gab.py

import torch
import torch.nn as nn

class GAB(nn.Module):
    ''' Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    '''
    def __init__(self, embed_dim: int, layer_idx: int, device=None, dtype=None, **kwargs):
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx 
        
        # Hyperparameters for the GAB block
        self.inner_dim = kwargs.get("inner_dim", embed_dim * 4)
        self.num_heads = kwargs.get("num_heads", 8)

        # Layers
        self.linear1 = nn.Linear(embed_dim, self.inner_dim, **factory_kwargs)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(self.inner_dim, embed_dim, **factory_kwargs)
        self.attn = nn.MultiheadAttention(embed_dim, self.num_heads, **factory_kwargs)
        
    def _forward(self, X, **kwargs):
        ''' Forward pass of the model '''
        assert X.shape[-1] == self.embed_dim
        
        # Feed forward network part
        Y_ffn = self.linear2(self.act(self.linear1(X)))

        # Self attention part
        Y_attn, _ = self.attn(X, X, X, need_weights=False)

        # Combining both outputs
        Y = Y_ffn + Y_attn

        return Y
     
    def forward(self, X, **kwargs):
        ''' Forward pass of the model '''
        Y = self._forward(X, **kwargs)
        assert Y.shape[-1] == self.embed_dim
        return Y
    
def gab_config() -> dict:
    ''' Returns a dictionary of hyperparameters for constructing a GAB layer
        embed_dim, layer_idx, device, dtype should not be included in the dictionary which will be provided by the model
    '''
    return {
        "inner_dim": 1536,  # 4 * 384
        "num_heads": 8
    }