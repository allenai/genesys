import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

class RemixerBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        causal = False,
        bias = False
    ):
        super().__init__()
        self.causal = causal
        self.proj_in = nn.Linear(dim, 2 * dim, bias = bias)
        self.mixer = nn.Parameter(torch.randn(seq_len, seq_len))
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.proj_out = nn.Linear(dim, dim, bias = bias)

    def forward(self, x):
        mixer, causal, device = self.mixer, self.causal, x.device
        x, gate = self.proj_in(x).chunk(2, dim = -1)
        x = F.gelu(gate) * x

        if self.causal:
            seq = x.shape[1]
            mask_value = -torch.finfo(x.dtype).max
            mask = torch.ones((seq, seq), device = device, dtype=torch.bool).triu(1)
            mixer = mixer[:seq, :seq]
            mixer = mixer.masked_fill(mask, mask_value)

        mixer = mixer.softmax(dim = -1)
        mixed = einsum('b n d, m n -> b m d', x, mixer)

        alpha = self.alpha.sigmoid()
        out = (x * mixed) * alpha + (x - mixed) * (1 - alpha)

        return self.proj_out(out)