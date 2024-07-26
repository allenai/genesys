import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from x_transformers import (
    RMSNorm,
    FeedForward
)


import torch
from torch.autograd.function import Function

from einops import einsum, rearrange

def exists(val):
    return val is not None

# custom function

class StopGraddableAttentionFunction(Function):

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        q,
        k,
        v,
        mask,
        attn_mask,
        causal: bool,
        q_stop_grad_mask,
        k_stop_grad_mask,
        v_stop_grad_mask,
    ):
        scale = q.shape[-1] ** -0.5

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * scale

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(col_mask, 'b j -> b 1 1 j')
            sim.masked_fill_(~mask, max_neg_value)

        if exists(attn_mask):
            sim.masked_fill_(~attn_mask, max_neg_value)

        if causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        ctx.args = (
            causal,
            scale,
            mask,
            q_stop_grad_mask,
            k_stop_grad_mask,
            v_stop_grad_mask
        )

        ctx.save_for_backward(
            q, k, v,
            attn,
            out
        )

        return out

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):

        (
            causal,
            scale,
            mask,
            q_stop_grad_mask,
            k_stop_grad_mask,
            v_stop_grad_mask
        ) = ctx.args

        q, k, v, p, o = ctx.saved_tensors

        # stop grad masks are either type bool, with True indicating stop grad, or can be type float, in which case it will scale the gradients

        if q_stop_grad_mask.dtype == torch.bool:
            q_stop_grad_mask = (~q_stop_grad_mask).float()

        if k_stop_grad_mask.dtype == torch.bool:
            k_stop_grad_mask = (~k_stop_grad_mask).float()

        if v_stop_grad_mask.dtype == torch.bool:
            v_stop_grad_mask = (~v_stop_grad_mask).float()

        # softmax D

        D = (do * o).sum(dim = -1, keepdims = True)        

        # stop grad for values

        p_v = p

        if exists(v_stop_grad_mask):
            p_v.mul_(v_stop_grad_mask)

        # dv

        dv = einsum(p_v, do, 'b h i j, b h i d -> b h j d')

        # prep for dq and dk

        dp = einsum(do, v, 'b h i d, b h j d -> b h i j')
        ds = p * scale * (dp - D)

        # handle stop grad masking for queries and keys

        ds_q = ds_k = ds

        if exists(q_stop_grad_mask):
            ds_q.mul_(q_stop_grad_mask)

        if exists(k_stop_grad_mask):            
            ds_k.mul_(k_stop_grad_mask)

        # dq and dk

        dq = einsum(ds_q, k, 'b h i j, b h j d -> b h i d')
        dk = einsum(ds_k, q, 'b h i j, b h i d -> b h j d')

        return dq, dk, dv, None, None, None, None, None, None

# convenience method with defaults

stop_graddable_attn_ = StopGraddableAttentionFunction.apply

def stop_graddable_attn(
    q, k, v,
    mask = None,
    attn_mask = None,
    causal = False,
    q_stop_grad_mask = None,
    k_stop_grad_mask = None,
    v_stop_grad_mask = None
):
    return stop_graddable_attn_(q, k, v, mask, attn_mask, causal, q_stop_grad_mask, k_stop_grad_mask, v_stop_grad_mask)


# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class CausalAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        x,
        attn_mask = None,
        stop_grad_attn_mask = None
    ):
        seq, device = x.shape[-2], x.device

        q, k, v = self.to_qkv(x)

        if exists(stop_grad_attn_mask):
            if not isinstance(stop_grad_attn_mask, tuple):
                stop_grad_attn_mask = (None, stop_grad_attn_mask, stop_grad_attn_mask)

            assert len(stop_grad_attn_mask) == 3, 'stop_grad_attn_mask must be either a stop grad mask (implicit for key / values) or a tuple of 3 Tensor for individual stop grads of queries, keys, values'

            q_stop_grad, k_stop_grad, v_stop_grad = stop_grad_attn_mask

            out = stop_graddable_attn(
                q, k, v,
                attn_mask = attn_mask,
                q_stop_grad_mask = q_stop_grad,
                k_stop_grad_mask = k_stop_grad,
                v_stop_grad_mask = v_stop_grad
            )

        else:
            q = q * self.scale
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            causal_mask = torch.ones((seq, seq), device = device, dtype = torch.bool).triu(1)

            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(causal_mask, mask_value)

            if exists(attn_mask):
                sim = sim.masked_fill(~attn_mask, mask_value)

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # combine heads

        return self.to_out(out)

# transformer

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        max_seq_len = 2048,
        max_reason_seq_len = 4,
        dim_head = 64,
        heads = 8,
        ignore_index = -1,
        stop_grad_next_tokens_to_reason = False
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # embed

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # reasoning tokens

        self.max_reason_seq_len = max_reason_seq_len
        self.reason_tokens = nn.Parameter(torch.randn(max_reason_seq_len, dim))
        nn.init.normal_(self.reason_tokens, std = 0.02)

        # transformer layers

        self.layers = ModuleList([])
        for _ in range(depth):

            attn = CausalAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads
            )

            ff = nn.Sequential(
                RMSNorm(dim),
                FeedForward(dim = dim)
            )

            self.layers.append(ModuleList([attn, ff]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

        # loss related

        self.ignore_index = ignore_index

        # stop gradient settings

        self.stop_grad_next_tokens_to_reason = stop_grad_next_tokens_to_reason

    def forward(
        self,
        x,
        num_reason_tokens = 0,
        num_steps_future_can_use_reason = 2,     # how many positions into the future until a reason token can be attended to
        remove_reason_tokens_at_end = False,
        return_loss = False
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        batch, seq, device = *x.shape, x.device

        assert seq <= self.max_seq_len

        x = self.token_emb(x)

        seq_arange = torch.arange(seq, device = device)
        pos = self.pos_emb(seq_arange)

        attn_kwargs = dict()

        # intersperse reasoning tokens if needed

        has_reason_tokens = num_reason_tokens > 0

        if has_reason_tokens:
            assert num_reason_tokens <= self.max_reason_seq_len

            x = rearrange(x, 'b n d -> b n 1 d')

            reason_tokens = self.reason_tokens[:num_reason_tokens]
            reason_tokens = repeat(reason_tokens, 'r d -> b n r d', b = batch, n = seq)

            x = torch.cat((x, reason_tokens), dim = -2)
            x = rearrange(x, 'b n r d -> b (n r) d')

            # handle absolute positions
            # applied axially to reasoning tokens and main token

            num_tokens_per_timestep = num_reason_tokens + 1
            pos = repeat(pos, 'n d -> (n r) d', r = num_tokens_per_timestep)

            # handle masking for reasoning tokens
            # each reason token can only be attended to by tokens (+ future reasoning tokens) that are {num_steps_future_can_use_reason}

            seq_timesteps = repeat(seq_arange, 'n -> (n r)', r = num_tokens_per_timestep)

            seq_with_reason_range = torch.arange(seq_timesteps.shape[-1], device = device)
            is_reason_token_mask = ~(seq_with_reason_range % num_tokens_per_timestep == 0)

            q_range = rearrange(seq_timesteps, 'n -> n 1')
            k_range = rearrange(seq_timesteps, 'n -> 1 n')

            attn_mask = ~(
                is_reason_token_mask &
                (q_range > k_range) &
                ((q_range - num_steps_future_can_use_reason) <= k_range)
            )

            # whether to fully mask out or stop gradient on attention matrix

            if self.stop_grad_next_tokens_to_reason:
                attn_kwargs = dict(stop_grad_attn_mask = ~attn_mask)
            else:
                attn_kwargs = dict(attn_mask = attn_mask)

        # attention and feedforward, passing in reason tokens mask from above

        x = x + pos

        for attn, ff in self.layers:
            x = attn(x, **attn_kwargs) + x
            x = ff(x) + x

        embed = self.norm(x)

        logits = self.to_logits(embed)

        # whether to remove reason tokens at the very end

        if has_reason_tokens and remove_reason_tokens_at_end:
            logits = rearrange(logits, 'b (n r) c -> b n r c', r = num_tokens_per_timestep)
            logits = logits[..., 0, :]

        if not return_loss:
            return logits

        if has_reason_tokens and not remove_reason_tokens_at_end:
            labels = repeat(labels, 'b n -> b (n r)', r = num_tokens_per_timestep)

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss