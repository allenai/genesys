import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase
import torch.nn.functional as F


class ConvCompress(nn.Module):

    def __init__(self, dim, ratio=3, groups=1, **factory_kwargs):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, ratio, stride=ratio, groups=groups,
            **factory_kwargs)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)


class FourierFFTLayer(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states.float(), dim=-1),
            dim=-2).real


class HybridAttention(nn.Module):

    def __init__(self, dim, heads=8, causal=False, compression_factor=3,
        dropout=0.0, **factory_kwargs):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        self.causal = causal
        self.compression_factor = compression_factor
        self.compress_fn = ConvCompress(dim, compression_factor, groups=
            heads, **factory_kwargs)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False, **factory_kwargs)
        self.to_out = nn.Linear(dim, dim, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.null_k = nn.Parameter(torch.zeros(1, 1, dim, **factory_kwargs))
        self.null_v = nn.Parameter(torch.zeros(1, 1, dim, **factory_kwargs))
        self.fft_layer = FourierFFTLayer()

    def forward(self, x, input_mask=None):
        b, t, d, h, cf, device = (*x.shape, self.heads, self.
            compression_factor, x.device)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        padding = cf - t % cf
        if padding < cf:
            k, v = map(lambda t: F.pad(t, (0, 0, padding, 0)), (k, v))
        k, v = map(self.compress_fn, (k, v))
        nk, nv = map(lambda t: t.expand(b, -1, -1), (self.null_k, self.null_v))
        k = torch.cat((nk, k), dim=1)
        v = torch.cat((nv, v), dim=1)
        q, k, v = map(lambda t: t.reshape(*t.shape[:2], h, -1).transpose(1,
            2), (q, k, v))
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * d ** -0.5
        mask_value = -torch.finfo(dots.dtype).max
        if self.causal:
            mask_q = mask_k = torch.arange(t, device=device)
            if padding < cf:
                mask_k = F.pad(mask_k, (padding, 0))
            mask_k, _ = mask_k.reshape(-1, cf).max(dim=-1)
            mask = mask_q[:, None] < mask_k[None, :]
            mask = F.pad(mask, (1, 0), value=False)
            dots.masked_fill_(mask[None, None, ...], mask_value)
            del mask
        if input_mask is not None:
            mask_q = mask_k = input_mask
            if padding < cf:
                mask_k = F.pad(mask_k, (padding, 0), value=True)
            mask_k = mask_k.reshape(b, -1, cf).sum(dim=-1) > 0
            mask = mask_q[:, None, :, None] < mask_k[:, None, None, :]
            mask = F.pad(mask, (1, 0), value=True)
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, d)
        out = self.to_out(out)
        out = self.fft_layer(out)
        return out


class GAB(GABBase):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, heads=8, compression_factor=3, dropout=0.1, causal=False, **
        kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.hybrid_attention = HybridAttention(embed_dim, heads, causal,
            compression_factor, dropout, **factory_kwargs)

    def _forward(self, X, **intermediate_vars):
        return self.hybrid_attention(X)


gab_config = {'heads': 8, 'compression_factor': 3, 'dropout': 0.1, 'causal':
    False}
