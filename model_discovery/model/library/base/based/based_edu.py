from einops import rearrange
import torch
from torch import nn
import math

class TaylorExp(nn.Module):
    """
    Feature map to compute 2nd-order Taylor approximation of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim, head_dim_idx, temp=None, eps=1e-12):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx
        self.temp = 1.0 if temp is None else temp
        self.eps = eps
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(self.input_dim)
        self.rrd = math.sqrt(self.rd)

    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        term1 = torch.ones(x[..., :1].shape).to(x.device)
        term2 = x / self.rrd
        term3 = x2 / self.rd
        terms = [term1, term2, term3]
        return torch.cat(terms, dim=self.head_dim_idx)

class TaylorLinAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = 16
        self.num_heads = 16
        self.num_key_value_heads = 16
        self.head_dim = self.d_model // self.num_key_value_heads
        self.eps = 1e-12

        feature_map_kwargs = {
            "input_dim": self.feature_dim,
            "head_dim_idx": -1,
            "eps": 1e-12,
        }
        self.feature_map = TaylorExp(**feature_map_kwargs)
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        b, l, _ = hidden_states.size()
        q = self.proj_q(hidden_states)
        k = self.proj_k(hidden_states)
        v = self.proj_v(hidden_states)
        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        # Compute attention
        num = (q * (k * v).cumsum(dim=2)).sum(dim=-1)
        denom = (q * k.cumsum(dim=2)).sum(dim=-1) + self.eps
        y = num / denom

        y = rearrange(y, "b h l d -> b l (h d)")
        y = self.proj_o(y)
        return y
