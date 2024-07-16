# gab.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# YOU CAN IMPORT MORE MODULES HERE #
from torch.nn import Parameter

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #

class CausalGatedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, **{"device": device, "dtype": dtype})
        self.out_proj = nn.Linear(embed_dim, embed_dim, **{"device": device, "dtype": dtype})
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        attn_output = self.out_proj(attn_output)
        return attn_output


class GAB(nn.Module):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = CausalGatedAttention(embed_dim, num_heads, dropout, **factory_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio), **factory_kwargs),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim, **factory_kwargs),
            nn.Dropout(dropout)
        )

    def _forward(self, X, **kwargs):
        assert X.shape[-1] == self.embed_dim
        B, N, C = X.shape
        mask = torch.tril(torch.ones(N, N, device=X.device, dtype=X.dtype)).unsqueeze(0).unsqueeze(0)
        attn_output = self.attn(X, mask)
        mlp_output = self.mlp(attn_output)
        return mlp_output

    def forward(self, X, **kwargs):
        Y = self._forward(X, **kwargs)
        assert Y.shape[-1] == self.embed_dim
        return Y

def gab_config() -> dict:
    """Returns a dictionary of hyperparameters for constructing a GAB layer
        embed_dim, device, dtype should not be included in the dictionary which will be provided by the model
    """
    return {
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.1
    }



# Perform registration after defining gab_config
from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=gab_config()
)(GAB)