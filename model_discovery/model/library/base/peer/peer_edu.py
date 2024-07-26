import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from functools import partial
from torch.utils.checkpoint import checkpoint

import einx
from einops.layers.torch import Rearrange


from math import sqrt
from einops import einsum, pack, unpack



# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# rmsnorm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main class

class PEER(Module):
    """
    following Algorithm 1 in the paper
    """

    def __init__(
        self,
        dim,
        *,
        heads = 8,                       # tested up to 32 - (hk = heads * num_experts_per_head (16))
        num_experts = 1_000_000,         # he chose 1 million
        num_experts_per_head = 16,       # he settled on 16, but was 32 in PKM paper
        activation = nn.GELU,
        dim_key = None,
        product_key_topk = None,
        separate_embed_per_head = False, # @smerky notes that heads may retrieve same redundant neurons. this setting would allow for separate embeds per head and prevent that
        pre_rmsnorm = False,
        dropout = 0.
    ):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - 2 for product key
        k - number of keys
        """

        super().__init__()

        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        # whether to do separate embedding per head

        num_expert_sets = 1 if not separate_embed_per_head else heads

        self.heads = heads
        self.separate_embed_per_head = separate_embed_per_head
        self.num_experts = num_experts

        # experts that will form the mlp project in / out weights

        self.weight_down_embed = nn.Embedding(num_experts * num_expert_sets, dim)
        self.weight_up_embed = nn.Embedding(num_experts * num_expert_sets, dim)

        # activation function, defaults to gelu

        self.activation = activation()

        # queries and keys for product-key

        assert sqrt(num_experts).is_integer(), '`num_experts` needs to be a square'
        assert (dim % 2) == 0, 'feature dimension should be divisible by 2'

        dim_key = default(dim_key, dim // 2)
        self.num_keys = int(sqrt(num_experts))

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim_key * heads * 2, bias = False),
            Rearrange('b n (p h d) -> p b n h d', p = 2, h = heads)
        )

        self.product_key_topk = default(product_key_topk, num_experts_per_head)
        self.num_experts_per_head = num_experts_per_head

        self.keys = nn.Parameter(torch.zeros(heads, self.num_keys, 2, dim_key))
        nn.init.normal_(self.keys, std = 0.02)

        # dropout

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x
    ):

        x = self.norm(x)

        # queries

        queries = self.to_queries(x)

        # first get similarity with keys

        sim = einsum(queries, self.keys, 'p b n h d, h k p d -> p b n h k')

        # product key logic

        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.product_key_topk, dim = -1)

        all_scores = einx.add('... i, ... j -> ... (i j)', scores_x, scores_y)
        all_indices = einx.add('... i, ... j -> ... (i j)', indices_x * self.num_keys, indices_y)

        scores, pk_indices = all_scores.topk(self.num_experts_per_head, dim = -1)

        indices = all_indices.gather(-1, pk_indices)

        # if separate embeds per head, add appropriate offsets per head

        if self.separate_embed_per_head:
            head_expert_offsets = torch.arange(self.heads, device = x.device) * self.num_experts
            indices = einx.add('b n h k, h -> b n h k', indices, head_expert_offsets)

        # build the weight matrices for projecting in and out
        # basically the experts are the gathered parameters for an MLP

        weights_down = self.weight_down_embed(indices)
        weights_up = self.weight_up_embed(indices)

        # below is basically Algorithm 1 in paper

        x = einsum(x, weights_down, 'b n d, b n h k d -> b n h k')

        x = self.activation(x)
        x = self.dropout(x)

        x = x * scores.softmax(dim = -1)

        x = einsum(x, weights_up, 'b n h k, b n h k d -> b n d')

        return x
    

class ChunkedPEER(Module):
    def __init__(
        self,
        peer: PEER,
        seq_chunk_size: int = 128
    ):
        super().__init__()
        self.peer = peer
        self.seq_chunk_size = seq_chunk_size

    def forward(
        self,
        x
    ):
        peer = self.peer

        if self.training and x.requires_grad:
            peer = partial(checkpoint, peer)            

        out = []
        for chunk in x.split(self.seq_chunk_size, dim = 1):
            chunk_out = peer(chunk)
            out.append(chunk_out)

        return torch.cat(out, dim = 1)



# main class

class PK(Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_key = None,
        num_keys = 1_000,
        product_keys = 2,
        product_key_topk = None,
        final_topk = 16,
        num_experts_per_head = 16
    ):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - product keys
        k - number of keys
        """

        super().__init__()
        assert (dim % 2) == 0
        dim_key = default(dim_key, dim // 2)

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim_key * product_keys * heads, bias = False),
            Rearrange('b n (p h d) -> p b n h d', h = heads, p = product_keys)
        )

        self.num_keys = num_keys
        self.product_keys = product_keys

        self.keys = nn.Parameter(torch.zeros(product_keys, num_keys, heads, dim_key))
        nn.init.normal_(self.keys, std = 0.02)

        product_key_topk = default(product_key_topk, final_topk)
        assert final_topk <= (product_key_topk ** product_keys)

        self.topk = product_key_topk
        self.final_topk = final_topk

        # the maximum index, or the total space being indexed into

        self.max_index = int(num_keys ** product_keys)

    def forward(
        self,
        x,
        softmax_scores = False
    ):

        queries = self.to_queries(x)

        sim = einsum(queries, self.keys, 'p b n h d, p k h d -> p b n h k')

        scores, indices = sim.topk(self.topk, dim = -1)

        # cartesian product indices

        strides = self.num_keys ** torch.arange(self.product_keys, device = x.device)
        indices = einx.multiply('p ..., p -> p ...', indices, strides)

        index, *rest_indices = indices

        for rest_index in rest_indices:
            index = einx.add('... i, ... j -> ... (i j)', index, rest_index)

        # cartesian product score

        score, *rest_scores = scores

        for rest_score in rest_scores:
            score = einx.add('... i, ... j -> ... (i j)', score, rest_score)

        final_scores, final_indices = score, index

        # final topk

        final_scores, pk_indices = final_scores.topk(self.final_topk, dim = -1)

        final_indices = final_indices.gather(-1, pk_indices)

        if softmax_scores:
            final_scores = final_scores.softmax(dim = -1)

        return final_scores, final_indices


# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# rmsnorm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main class

class PKAttention(Module):
    def __init__(
        self,
        dim,
        *,
        causal = True,
        heads = 8,
        num_key_values = 1_000_000,
        key_value_pk_topk = 16,
        dim_key = None,
        product_keys = 2,
        pre_rmsnorm = False,
        dropout = 0.
    ):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - 2 for product key
        k - number of keys
        """

        super().__init__()
        self.causal = causal
        self.heads = heads
        self.num_key_values = num_key_values

        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        # experts that will form the mlp project in / out weights

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim * heads, bias = False),
            Rearrange('b n (h d) -> b n h d', h = heads)
        )

        # keys and values selected using product-key

        self.keys = nn.EmbeddingBag(num_key_values * heads, dim, mode = 'sum')
        self.values = nn.EmbeddingBag(num_key_values * heads, dim, mode = 'sum')

        assert sqrt(num_key_values).is_integer(), '`num_key_values` needs to be a square'
        assert (dim % 2) == 0, 'feature dimension should be divisible by 2'

        self.to_kv_pk_indices = PK(
            dim = dim,
            num_keys = int(sqrt(num_key_values)),
            final_topk = key_value_pk_topk,
            product_keys = product_keys
        )

        # dropout

        self.dropout = nn.Dropout(dropout)

        # output

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim * heads, dim, bias = False)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        device = x.device

        x = self.norm(x)

        # queries

        q = self.to_queries(x)

        q = q * (q.shape[-1] ** -0.5)

        # keys and values

        kv_scores, indices = self.to_kv_pk_indices(x, softmax_scores = True)

        offsets = torch.arange(self.heads, device = device) * self.num_key_values
        indices = einx.add('b n h k, h -> b n h k', indices, offsets)

        indices, packed_shape = pack_one(indices, '* k')
        kv_scores, _ = pack_one(kv_scores, '* k')

        k, v = self.keys(indices, per_sample_weights = kv_scores), self.values(indices, per_sample_weights = kv_scores)

        k = unpack_one(k, packed_shape, '* d')
        v = unpack_one(v, packed_shape, '* d')

        # usual multihead self attention

        sim = einsum(q, k, 'b i h d, b j h d -> b h i j')

        # whether causal or not

        if self.causal:
            assert not exists(mask)
            i, j, device = *sim.shape[-2:], x.device
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        elif exists(mask):
            sim = einx.where('b j, b h i j, -> b h i j', mask, sim, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum(attn, v, 'b h i j, b j h d -> b h i d')

        # combine heads

        return self.to_out(out)

# main

if __name__ == '__main__':
    pk = PK(
        dim = 512,
        num_keys = 100,
        final_topk = 10,
        product_keys = 3
    )

    x = torch.randn(2, 1024, 512)
    score, indices = pk(x)

    assert score.shape == (2, 1024, 8, 10)
    assert indices.shape == (2, 1024, 8, 10)

    peer_attn = PKAttention(
        dim = 256,
        causal = True,
        heads = 8,
        num_key_values = int(1e4),
        pre_rmsnorm = True
    )

    x = torch.randn(2, 512, 256)

    out = peer_attn(x) + x

    assert x.shape == out.shape