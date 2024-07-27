import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def rearrange_all(tensors, *args, **kwargs):
    return map(lambda t: rearrange(t, *args, **kwargs), tensors)

# feedforward

class GroupLayerNorm(nn.Module):
    def __init__(self, dim, groups = 1, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.groups = groups
        self.g = nn.Parameter(torch.ones(1, groups, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, groups, dim, 1))

    def forward(self, x):
        x = rearrange(x, 'b (g d) n -> b g d n', g = self.groups)
        std = torch.var(x, dim = 2, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 2, keepdim = True)
        out = (x - mean) / (std + self.eps) * self.g + self.b
        return rearrange(out, 'b g d n -> b (g d) n')

class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn,
        groups = 1
    ):
        super().__init__()
        self.norm = GroupLayerNorm(dim, groups = groups)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        groups = 1
    ):
        super().__init__()
        input_dim = dim * groups
        hidden_dim = dim * mult * groups

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, groups = groups),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, 1, groups = groups)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        groups = 1
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.groups = groups
        self.heads = heads
        self.causal = causal
        input_dim = dim * groups
        inner_dim = dim_head * heads * groups

        self.to_q = nn.Conv1d(input_dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv1d(input_dim, inner_dim * 2, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, input_dim, 1)

    def forward(self, x, mask = None, context = None):
        n, device, h, g, causal = x.shape[2], x.device, self.heads, self.groups, self.causal
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = 1))
        q, k, v = rearrange_all((q, k, v), 'b (g h d) n -> (b g h) n d', g = g, h = h)

        q = q * self.scale

        sim = einsum('b i d, b j d -> b i j', q, k)

        if exists(mask):
            mask = repeat(mask, 'b n -> (b g h) n', h = h, g = g)
            mask = rearrange(mask, 'b n -> b n ()') * rearrange(mask, 'b n -> b () n')
            mask_value = max_neg_value(sim)
            sim = sim.masked_fill(~mask, mask_value)

        if causal:
            causal_mask = torch.ones((n, n), device = device).triu(1).bool()
            mask_value = max_neg_value(sim)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b g h) n d -> b (g h d) n', h = h, g = g)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        groups = 1
    ):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, groups = groups), groups = groups)
        self.ff = PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, groups = groups), groups = groups)

    def forward(self, x, mask = None):
        x = self.attn(x, mask = mask) + x
        x = self.ff(x) + x
        return x

# main class

class MultistreamTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens,
        max_seq_len,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_streams = 1
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_streams = num_streams
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        self.pre_transformer_block = TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads)

        for _ in range(depth):
            self.layers.append(TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads, groups = num_streams))

        if num_streams > 1:
            self.query = nn.Parameter(torch.randn(dim))
            self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        self.post_transformer_block = TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads,)

        self.to_logits = nn.Sequential(
            Rearrange('b d n -> b n d'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        b, n, d, device, is_multistream = *x.shape, self.dim, x.device, (self.num_streams > 1)
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')

        x = x + pos_emb
        x = rearrange(x, 'b n d -> b d n')

        x = self.pre_transformer_block(x, mask = mask)
        layers = [x]

        if is_multistream:
            x = repeat(x, 'b d n -> b (s d) n', s = self.num_streams)

        for block in self.layers:
            x = block(x, mask = mask)
            layers.append(x)

        if is_multistream:
            layers = list(map(lambda t: rearrange(t, 'b (s d) n -> (b n) d s', d = d), layers))
            layer_tokens = torch.cat(layers, dim = -1)

            query = repeat(self.query, 'd -> b d ()', b = layer_tokens.shape[0])
            x = self.attn_pool(query, context = layer_tokens)
            x = rearrange(x, '(b n) d () -> b d n', n = n)

        x = self.post_transformer_block(x, mask = mask)
        return self.to_logits(x)