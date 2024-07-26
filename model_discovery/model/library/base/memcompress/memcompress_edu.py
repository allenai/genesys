import torch
from torch import nn
import torch.nn.functional as F

# convolutional compression class

class ConvCompress(nn.Module):
    def __init__(self, dim, ratio = 3, groups = 1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, ratio, stride = ratio, groups = groups)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)

# main class

class MemoryCompressedAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        causal = False,
        compression_factor = 3,
        dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'

        self.heads = heads
        self.causal = causal

        self.compression_factor = compression_factor
        self.compress_fn = ConvCompress(dim, compression_factor, groups = heads)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.null_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.null_v = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x, input_mask = None):
        b, t, d, h, cf, device = *x.shape, self.heads, self.compression_factor, x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # make sure keys and values sequence lengths
        # are divisible by the compression factor
        padding = cf - (t % cf)
        if padding < cf:
            k, v = map(lambda t: F.pad(t, (0, 0, padding, 0)), (k, v))

        # compress keys and values
        k, v = map(self.compress_fn, (k, v))

        # attach a null key and value, in the case that the first query has no keys to pay attention to
        nk, nv = map(lambda t: t.expand(b, -1, -1), (self.null_k, self.null_v))
        k = torch.cat((nk, k), dim=1)
        v = torch.cat((nv, v), dim=1)

        # merge heads
        q, k, v = map(lambda t: t.reshape(*t.shape[:2], h, -1).transpose(1, 2), (q, k, v))

        # attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * d ** -0.5

        mask_value = -torch.finfo(dots.dtype).max

        # causal masking, if needed
        if self.causal:
            mask_q = mask_k = torch.arange(t, device=device)

            if padding < cf:
                mask_k = F.pad(mask_k, (padding, 0))

            mask_k, _ = mask_k.reshape(-1, cf).max(dim=-1)
            mask = mask_q[:, None] < mask_k[None, :]
            mask = F.pad(mask, (1, 0), value=False)

            dots.masked_fill_(mask[None, None, ...], mask_value)
            del mask

        # input masking
        if input_mask is not None:
            mask_q = mask_k = input_mask
            if padding < cf:
                mask_k = F.pad(mask_k, (padding, 0), value=True)
            mask_k = mask_k.reshape(b, -1, cf).sum(dim=-1) > 0
            mask = mask_q[:, None, :, None] < mask_k[:, None, None, :]
            mask = F.pad(mask, (1, 0), value=True)

            dots.masked_fill_(~mask, mask_value)
            del mask

        # attention
        attn = dots.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # split heads and combine
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.to_out(out)