import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# scalenorm

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

# T5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# class

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

# activation functions

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class LaplacianAttnFn(nn.Module):
    """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """

    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

# FLASH

class FLASH(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 2.,
        causal = False,
        dropout = 0.,
        rotary_pos_emb = None,
        norm_klass = nn.LayerNorm,
        shift_tokens = False,
        laplace_attn_fn = False,
        reduce_group_non_causal_attn = True,
        global_ = True,
        local_ = True,
        conv = False,
        kernel = 63,
        square=False,
        castling = False,
        softmax = False,
        quad = False,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()

        # positional embeddings

        self.rotary_pos_emb = rotary_pos_emb
        self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal = causal)

        # norm

        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout)

        # whether to reduce groups in non causal linear attention

        self.reduce_group_non_causal_attn = reduce_group_non_causal_attn

        # projections

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )
        if softmax or square:
            self.qk_offset_scale = OffsetScale(query_key_dim, heads = 2)
        else:
            self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)
        self.to_out = nn.Linear(hidden_dim, dim)

        self.global_ = global_
        self.local_ = local_ 
        self.square = square

        self.conv = conv
        if conv:
            res_kernel_size = kernel
            self.dwconv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=1,
            )
            self.conv_mask = torch.ones((1, 1, res_kernel_size, 1), dtype=torch.float32)
            self.conv_mask[:, :, res_kernel_size // 2 + 1 :, :] = 0.0
        self.castling = castling
        self.softmax = softmax
        self.quad = quad
    def forward(
        self,
        x,
        *,
        mask = None
    ):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # prenorm

        normed_x = self.norm(x)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # initial projections

        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)
        qk = self.to_qk(normed_x)

        # offset and scale
        v0 = v
        if self.square:

            q, k = self.qk_offset_scale(qk)

            if exists(mask):
                l_mask = rearrange(mask, '... -> ... 1')
                k = k.masked_fill(~l_mask, 0.)
            if exists(self.rotary_pos_emb):
                q, k = map(self.rotary_pos_emb.rotate_queries_or_keys, (q, k))

            qsize = q.size(1)
            padding = padding_to_multiple_of(n, qsize)

            if padding > 0:
                q, k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (q, k, v))

                mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
                mask = F.pad(mask, (0, padding), value = False)

            sim = einsum('... i d, ... j d -> ... i j', q, k) / qsize

            sim = sim + self.rel_pos_bias(sim)

            attn = self.attn_fn(sim / qsize)
            attn = self.dropout(attn)
            
            if self.causal:
                causal_mask = torch.ones((qsize, qsize), dtype = torch.bool, device = device).triu(1)
                attn = attn.masked_fill(causal_mask, 0)


            if exists(mask):
                attn = attn.masked_fill(~mask[:,None,:], 0.)

            softmax_out = einsum('... i j, ... j d -> ... i d', attn, v)

            out = gate * (softmax_out)

            # projection out and residual

            return self.to_out(out) + x

        if self.softmax:
            q, k = self.qk_offset_scale(qk)

            if exists(mask):
                l_mask = rearrange(mask, '... -> ... 1')
                k = k.masked_fill(~l_mask, 0.)
            if exists(self.rotary_pos_emb):
                q, k = map(self.rotary_pos_emb.rotate_queries_or_keys, (q, k))

            qsize = q.size(1)
            padding = padding_to_multiple_of(n, qsize)

            if padding > 0:
                q, k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (q, k, v))

                mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
                mask = F.pad(mask, (0, padding), value = False)


            sim = einsum('... i d, ... j d -> ... i j', q, k) / qsize

            sim = sim + self.rel_pos_bias(sim)
            if self.causal:
                causal_mask = torch.ones((qsize, qsize), dtype = torch.bool, device = device).triu(1)
                sim = sim.masked_fill(causal_mask, torch.finfo(sim.dtype).min)

            attn = torch.nn.functional.softmax(sim, dim=-1)

            attn = self.dropout(attn)

            if exists(mask):
                attn = attn.masked_fill(~mask[:,None,:], 0.)



            softmax_out = einsum('... i j, ... j d -> ... i d', attn, v)

            out = gate * (softmax_out)


            return self.to_out(out) + x

        else:
            quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        # mask out linear attention keys

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # padding for groups

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)

        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (n g) d -> b n g d', g = self.group_size), (quad_q, quad_k, lin_q, lin_k, v))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = g)

        # calculate quadratic attention output

        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        sim = sim + self.rel_pos_bias(sim)

        attn = self.attn_fn(sim)
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out = einsum('... i j, ... j d -> ... i d', attn, v)

        # calculate linear attention output

        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g

            # exclusive cumulative sum along group dimension

            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)

            lin_out = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)
        else:
            context_einsum_eq = 'b d e' if self.reduce_group_non_causal_attn else 'b g d e'
            lin_kv = einsum(f'b g n d, b g n e -> {context_einsum_eq}', lin_k, v) / n
            lin_out = einsum(f'b g n d, {context_einsum_eq} -> b g n e', lin_q, lin_kv)

        # fold back groups into full sequence, and excise out padding

        quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out, lin_out))

        # gate
        out = None 
        if self.global_:
            out = lin_attn_out
        if self.local_:
            out = quad_attn_out if out is None else quad_attn_out + lin_attn_out
        
        if self.conv:
            if self.conv_mask.device != out.device:
                self.conv_mask = self.conv_mask.to(out.device)
            self.dwconv.weight.data *= self.conv_mask
            conv_out = self.dwconv(v0.unsqueeze(1)) 
            if self.castling:
                out = 0.5 * v0 + 1.0 / math.pi * out 
                out = out / out.norm(dim=-1, keepdim=True)
                out += conv_out.squeeze(1)
            else:
                out = out + conv_out.squeeze(1)

        out = gate * (out)

        # projection out and residual

        return self.to_out(out) + x
