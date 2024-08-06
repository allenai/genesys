from model_discovery.model.utils.modules import GABBase
from model_discovery.model.utils.modules import GABBase
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionalEmbedding(nn.Module):

    def __init__(self, emb_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, emb_dim))

    def forward(self, x):
        return x + self.pos_emb[:, :x.size(1), :]


class SpectralHyenaLongformer(nn.Module):

    def __init__(self, d_model, seq_len, num_heads, attention_window,
        attention_dilation, filter_order=64):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.pos_emb = PositionalEmbedding(d_model, seq_len)
        self.hyena_filter = nn.Conv1d(d_model, d_model, kernel_size=
            filter_order, padding=filter_order // 2, groups=d_model)
        self.spectral_filter = nn.Conv1d(d_model, d_model, kernel_size=
            filter_order, padding=filter_order // 2, groups=d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x, attention_mask=None):
        bsz, seq_len, _ = x.size()
        x = self.pos_emb(x)
        q = self.query(x).view(bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
        k = self.key(x).view(bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
        v = self.value(x).view(bsz, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self
            .head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.einsum('bhqk,bhvd->bhqd', attn_weights, v
            ).transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        hyena_output = self.hyena_filter(x.transpose(1, 2)).transpose(1, 2)
        spectral_output = self.spectral_filter(x.transpose(1, 2)).transpose(
            1, 2)
        output = attn_output + hyena_output + spectral_output
        output = self.ffn(output)
        return output


gab_config = {}