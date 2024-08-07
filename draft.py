import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase
from einops import rearrange


class GeometricAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(GeometricAttention, self).__init__()
        self.att = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.copy_gate = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        attn_output, _ = self.att(src, src, src, attn_mask=mask)
        gate = torch.sigmoid(self.copy_gate(src))
        output = gate * src + (1 - gate) * attn_output
        output = self.dropout(output)
        output = self.norm(output + src)
        return output


class SegmentLevelRecurrence(nn.Module):

    def __init__(self, d_model, nhead, mem_len, dropout=0.1):
        super(SegmentLevelRecurrence, self).__init__()
        self.mem_len = mem_len
        self.att = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.memory = None

    def forward(self, src, mask=None):
        if self.memory is None:
            self.memory = src.new_zeros((self.mem_len, src.size(1), src.
                size(2)))
        combined = torch.cat([self.memory, src], dim=0)
        attn_output, _ = self.att(src, combined, combined, attn_mask=mask)
        self.memory = combined[-self.mem_len:].detach()
        output = self.norm(attn_output + src)
        return output


class GAB(GABBase):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, nhead=8, mem_len=128, dropout=0.1):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.geometric_attention = GeometricAttention(embed_dim, nhead, dropout
            )
        self.segment_recurrence = SegmentLevelRecurrence(embed_dim, nhead,
            mem_len, dropout)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.
            ReLU(), nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout
            ), nn.LayerNorm(embed_dim))

    def _forward(self, X, **intermediate_vars):
        mask = None
        X = self.geometric_attention(X, mask)
        X = self.segment_recurrence(X, mask)
        X = self.ffn(X)
        return X


gab_config = {'nhead': 8, 'mem_len': 128, 'dropout': 0.1}
