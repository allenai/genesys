import math
import torch
from torch import nn
from operator import mul
from math import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce

from local_attention import LocalAttention

# helper functions

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def divisible_by(num, divisor):
    return num % divisor == 0

def all_none(*arr):
    return all(el is None for el in arr)

def rotate_left(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(n, None))
    r = (*pre_slices, slice(0, n))
    return torch.cat((t[l], t[r]), dim=dim)

def rotate_right(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(-n, None))
    r = (*pre_slices, slice(None, -n))
    return torch.cat((t[l], t[r]), dim=dim)

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def merge_heads(h, v):
    b, t, d = v.shape
    return v.view(b, t, h, -1).transpose(1, 2).reshape(b, h, t, -1)

def split_heads(h, v):
    *_, t, d = v.shape
    return v.view(-1, h, t, d).transpose(1, 2).reshape(-1, t, d * h)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [buckets, -1]
    return t.reshape(*shape)

def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)

def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)

def reorder_buckets(t, r):
    return torch.einsum('buv,bvtd->butd', r, t)

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def cumavg(t, dim):
    r = torch.arange(1, t.shape[dim] + 1, device=t.device, dtype=t.dtype)
    expand_slice = [None] * len(t.shape)
    expand_slice[dim] = slice(None, None)
    return t.cumsum(dim=dim) / r[tuple(expand_slice)]

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def expand_batch_and_merge_head(b, t):
    shape = list(t.squeeze(0).shape)
    t = expand_dim(t, 0, b)
    shape[0] = shape[0] * b
    return t.reshape(*shape)

def differentiable_topk(x, k, temperature=1.):
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)


# causal sort net and reordered bucketing attention

def mask_reordering_matrix(R, topk, temperature):
    buckets = R.shape[1]

    mask_value = max_neg_value(R)
    mask = torch.zeros(R.shape, device=R.device).bool()
    i, j = torch.triu_indices(buckets, buckets)
    mask[:, i, j + topk] = True

    R.masked_fill_(mask, mask_value)
    return differentiable_topk(R, topk, temperature)

class CausalSimpleSortNet(nn.Module):
    def __init__(self, heads, bucket_size, max_buckets, n_top_buckets, dim, temperature):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.bucket_size = bucket_size
        self.max_buckets = max_buckets
        self.n_top_buckets = n_top_buckets
        self.temperature = temperature
        self.linear = nn.Parameter(torch.randn(1, heads, dim, max_buckets + n_top_buckets))
        self.act = nn.LeakyReLU()

    def forward(self, q, k, topk=1):
        bh, *_, h, max_buckets = *q.shape, self.heads, self.max_buckets
        b = bh // h
        buckets = k.shape[1] // self.bucket_size

        k_r = torch.cat((cumavg(k, dim=1), k), dim=-1)
        k_r = bucket(buckets, k_r)

        # for causal sort net, take the first token of each bucket to prevent leaking of future to past
        x = k_r[:, :, 0]

        W = expand_batch_and_merge_head(b, self.linear)
        R = self.act(x @ W)
        R = R[:, 0:buckets, 0:(buckets + self.n_top_buckets)]

        return mask_reordering_matrix(R, topk, self.temperature)

class CausalAttentionSortNet(nn.Module):
    def __init__(self, heads, bucket_size, dim, temperature):
        super().__init__()
        self.heads = heads
        self.bucket_size = bucket_size
        self.dim = dim
        self.temperature = temperature

    def forward(self, q, k, topk=1):
        bh, *_, h, dim = *q.shape, self.heads, self.dim

        b = bh // h
        buckets = q.shape[1] // self.bucket_size
        kv_buckets = k.shape[1] // self.bucket_size

        q_r = bucket(buckets, cumavg(q, dim=1))
        k_r = bucket(kv_buckets, cumavg(k, dim=1))

        sq = q_r[:, :, 0]
        sk = k_r.sum(dim=2)
        sk = F.pad(sk, (0, 0, topk, 0))

        R = torch.einsum('bie,bje->bij', sq, sk) * (dim ** -0.5)
        return mask_reordering_matrix(R, topk, self.temperature)

def apply_fn_after_split_ind(dim, ind, fn, t):
    l, r = split_at_index(dim, ind, t)
    return torch.cat((l, fn(r)), dim=dim)

class SinkhornCausalAttention(nn.Module):
    def __init__(self, bucket_size, dim, dim_heads, heads, max_seq_len, dropout = 0., kv_bucket_size = None, use_simple_sort_net = False, n_top_buckets = 2, temperature = 1.):
        super().__init__()
        assert kv_bucket_size is None or bucket_size == kv_bucket_size, 'different bucketing for key/values for causal reordering not supported yet'

        self.dim = dim
        self.heads = heads
        self.bucket_size = bucket_size

        # a learned null key / value for the first bucket (which has nothing in the past to sort to)
        self.null_keys = nn.Parameter(torch.randn(heads, 1, dim_heads))
        self.null_values = nn.Parameter(torch.randn(heads, 1, dim_heads))

        if use_simple_sort_net:
            self.sort_net = CausalSimpleSortNet(heads, bucket_size, max_seq_len // bucket_size, n_top_buckets, dim_heads * 2, temperature)
        else:
            self.sort_net = CausalAttentionSortNet(heads, bucket_size, dim_heads, temperature)

        self.n_top_buckets = n_top_buckets
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        b, h, t, d_h, n_top, d, bsz, device = *q.shape, self.n_top_buckets, self.dim, self.bucket_size, q.device

        bh = b * h
        hh = h // 2
        buckets = t // bsz
        n_top = min(n_top, buckets)

        hh_slice = (slice(None), slice(hh, None))

        rotate_fn = partial(apply_fn_after_split_ind, 1, hh, lambda t: rotate_left(t, bsz-1, dim=2))
        q, k, v = map(rotate_fn, (q, k, v))

        # merge batch and head
        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v))

        # bucket qkv
        b_q, b_k, b_v = map(partial(bucket, buckets), (q, k, v))

        # calculate R
        R = self.sort_net(q, k, topk=n_top)
        R = R.type_as(q).to(q)

        # add null key / values
        b_null_k = self.null_keys[None, :, None, :, :].expand(b, h, n_top, bsz, -1).reshape(bh, n_top, bsz, -1).to(k)
        b_null_v = self.null_values[None, :, None, :, :].expand(b, h, n_top, bsz, -1).reshape(bh, n_top, bsz, -1).to(v)

        b_k_r = torch.cat((b_null_k, b_k), dim=1)
        b_v_r = torch.cat((b_null_v, b_v), dim=1)

        # reorder buckets to buckets of the past
        b_k_r = reorder_buckets(b_k_r, R)
        b_v_r = reorder_buckets(b_v_r, R)

        b_k_r = b_k_r.reshape(bh, buckets, bsz * n_top, -1)
        b_v_r = b_v_r.reshape(bh, buckets, bsz * n_top, -1)

        # and concatenate to original buckets themselves for local attention
        b_k = torch.cat((b_k_r, b_k), dim=2)
        b_v = torch.cat((b_v_r, b_v), dim=2)

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d_h ** -0.5)

        # mask
        mask_value = max_neg_value(q)

        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)

            expand_head = lambda x: x.unsqueeze(1).repeat(1, h, 1)
            q_mask, kv_mask = map(expand_head, (q_mask, kv_mask))

            q_mask[hh_slice] = rotate_left(q_mask[hh_slice], bsz-1, dim=2)
            kv_mask[hh_slice] = rotate_left(kv_mask[hh_slice], bsz-1, dim=2)

            q_mask, kv_mask = map(lambda x: merge_dims(0, 1, x), (q_mask, kv_mask))
            mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask)

            mk_with_null = F.pad(mk, (0, 0, 2, 0), value=True)
            mk_r = batched_index_select(mk_with_null, R.abs().argmax(dim=-1))

            mk_r = mk_r.reshape(bh, buckets, -1)
            mk = torch.cat((mk_r, mk), dim=2)
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # masking for half head rotations
        shift = n_top * bsz
        total_shift = shift + bsz

        mask = torch.ones((b, h, buckets, bsz, total_shift), device=device).bool()
        i, j = torch.triu_indices(bsz, bsz, 1)
        mask[:, :, :, i, j + shift] = False
        mask[:, hh:, -1, 0:shift, 0:shift+1] = False
        mask[:, hh:, -1, 0, 0:shift+1] = True
        mask = mask.reshape(b * h, buckets, bsz, total_shift)

        dots.masked_fill_(~mask, mask_value)
        del mask

        # attention
        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        out = apply_fn_after_split_ind(1, hh, lambda t: rotate_right(t, bsz-1, dim=2), out)
        return out

class SinkhornSelfAttention(nn.Module):
    def __init__(self, dim, bucket_size, max_seq_len, heads = 8, dim_head = None, kv_bucket_size = None, causal = False, non_permutative = True, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, attn_dropout = 0., dropout = 0., context_only = False, use_simple_sort_net = False, n_local_attn_heads = 0, n_top_buckets = 1):
        super().__init__()
        assert dim_head or divisible_by(dim, heads), f'If dim_head is None, dimension {dim} must be divisible by the number of heads {heads}'
        assert not (causal and n_sortcut > 0), 'sortcut can only be used for non causal attention'
        assert not (causal and context_only), 'context only self attention layer cannot be causal'
        assert n_local_attn_heads <= heads, 'number of local attention heads cannot exceed total heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads
        self.dim_head = dim_head

        self.heads = heads
        self.bucket_size = bucket_size
        self.kv_bucket_size = default(kv_bucket_size, bucket_size)

        self.context_only = context_only
        self.to_q = nn.Linear(dim, dim_heads, bias=False)
        self.to_kv = nn.Linear(dim, dim_heads * 2, bias=False) if not context_only else None

        self.to_out = nn.Linear(dim_heads, dim)

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attention = LocalAttention(bucket_size, causal, dropout = attn_dropout, look_forward=(1 if not causal else 0))

        sink_heads = heads - n_local_attn_heads

        attn = SinkhornCausalAttention(bucket_size, dim, dim_head, sink_heads, max_seq_len, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets, temperature = temperature)

        self.sinkhorn_attention = attn

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask = None, context = None, context_mask = None):
        b, t, d, h, dh, l_h = *x.shape, self.heads, self.dim_head, self.n_local_attn_heads
        assert divisible_by(t, self.bucket_size), f'sequence {t} needs to be divisible by bucket size {self.bucket_size}'
        assert not (self.context_only and context is None), 'context key / values must be supplied if context self attention layer'
        assert not (context is not None and (context.shape[0], context.shape[2]) !=  (b, d)), 'contextual key / values must have the same batch and dimensions as the decoder'

        q = self.to_q(x)

        kv = self.to_kv(x).chunk(2, dim=-1) if not self.context_only else (context, context)
        kv_mask = input_mask if not self.context_only else context_mask

        assert divisible_by(kv[0].shape[1], self.kv_bucket_size), 'key/value sequences need to be divisible by key/value bucket size'

        qkv = (q, *kv)
        merge_heads_fn = partial(merge_heads, h)
        q, k, v = map(merge_heads_fn, qkv)

        split_index_fn = partial(split_at_index, 1, l_h)
        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
        has_local, has_sinkhorn = map(lambda x: x.shape[1] > 0, (lq, q))

        out = []

        if has_local > 0:
            out.append(self.local_attention(lq, lk, lv, input_mask = input_mask))

        if has_sinkhorn > 0:
            out.append(self.sinkhorn_attention(q, k, v, q_mask = input_mask, kv_mask = kv_mask))

        out = torch.cat(out, dim=1)
        out = split_heads(h, out)
        out = self.to_out(out)
        out = self.dropout(out)
        return out
