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
        
        # Define the hyperparameters
        self.hidden_dim = kwargs.get('hidden_dim', 512)
        self.num_heads = kwargs.get('num_heads', 8)
        self.ffn_dim = kwargs.get('ffn_dim', 2048)
        
        # Define the layers
        self.attention = nn.MultiheadAttention(embed_dim, self.num_heads, **factory_kwargs)
        self.linear1 = nn.Linear(embed_dim, self.ffn_dim, **factory_kwargs)
        self.linear2 = nn.Linear(self.ffn_dim, embed_dim, **factory_kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        self.activation = nn.ReLU()

    def _forward(self, X, **kwargs):
        ''' Forward pass of the model '''
        assert X.shape[-1] == self.embed_dim
        
        # Self-attention
        attn_output, _ = self.attention(X, X, X)
        
        # Feed-forward network
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(attn_output))))
        
        return ffn_output
     
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
        'hidden_dim': 512,
        'num_heads': 8,
        'ffn_dim': 2048,
        'dropout': 0.1
    }