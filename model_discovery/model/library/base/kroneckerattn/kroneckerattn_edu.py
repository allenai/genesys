import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F

class KroneckerSelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = 32):
        super().__init__()
        hidden_dim = heads * dim_heads

        self.heads = heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        h = x.shape[-2]

        x = torch.cat((x.mean(dim=-1), x.mean(dim=-2)), dim=-1)

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv h d) n -> qkv b h d n', h=self.heads, qkv=3)
        
        dots = einsum('bhdi,bhdj->bhij', q, k)
        attn = dots.softmax(dim=-1)
        out = einsum('bhij,bhdj->bhdi', attn, v)
        
        out = rearrange(out, 'b h d n -> b (h d) n')
        out = self.to_out(out)

        # outer sum
        out = rearrange(out[..., :h], 'b c (n 1) -> b c n 1') + rearrange(out[..., h:], 'b c (1 n) -> b c 1 n')
        return out