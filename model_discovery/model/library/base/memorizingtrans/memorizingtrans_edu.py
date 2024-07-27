import math
from functools import partial
from contextlib import contextmanager
from pathlib import Path
from filelock import FileLock

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops_exts import repeat_many
from einops.layers.torch import Rearrange

import os
import faiss
import numpy as np
from pathlib import Path

from contextlib import ExitStack, contextmanager

from einops import rearrange, pack, unpack

from .modules import KNN, FeedForward, Attention, PreNormResidual, T5RelativePositionBias
from .utils import exists, default, cast_list, unique, all_el_unique, check_shape, l2norm, cpu_count, multi_context, cast_tuple

# multiprocessing

from joblib import Parallel, delayed, cpu_count

# constants

FAISS_INDEX_GPU_ID = int(os.getenv('FAISS_INDEX_GPU_ID', 0))

DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY = './.tmp/knn.memories'

# KNN memory layer, where one can store key / value memories
# can automatically take care of a collection of faiss indices (across batch dimension)

class KNNMemory():
    def __init__(
        self,
        dim,
        max_memories = 16000,
        num_indices = 1,
        memmap_filename = './knn.memory.memmap',
        multiprocessing = True
    ):
        self.dim = dim
        self.num_indices = num_indices
        self.scoped_indices = list(range(num_indices))

        self.max_memories = max_memories
        self.shape = (num_indices, max_memories, 2, dim)
        self.db_offsets = np.zeros(num_indices, dtype = np.int32)

        self.db = np.memmap(memmap_filename, mode = 'w+', dtype = np.float32, shape = self.shape)
        self.knns = [KNN(dim = dim, max_num_entries = max_memories, cap_num_entries = True) for _ in range(num_indices)]
    
        self.n_jobs = cpu_count() if multiprocessing else 1

    def set_scoped_indices(self, indices):
        indices = list(indices)
        assert all_el_unique(indices), f'all scoped batch indices must be unique, received: {indices}'
        assert all([0 <= i < self.num_indices for i in indices]), f'each batch index must be between 0 and less than {self.num_indices}: received {indices}'
        self.scoped_indices = indices

    @contextmanager
    def at_batch_indices(self, indices):
        prev_indices = self.scoped_indices
        self.set_scoped_indices(indices)
        yield self
        self.set_scoped_indices(prev_indices)

    def clear(self, batch_indices = None):
        if not exists(batch_indices):
            batch_indices = list(range(self.num_indices))

        batch_indices = cast_list(batch_indices)

        for index in batch_indices:
            knn = self.knns[index]
            knn.reset()

        self.db_offsets[batch_indices] = 0

    def add(self, memories):
        check_shape(memories, 'b n kv d', d = self.dim, kv = 2, b = len(self.scoped_indices))

        memories = memories.detach().cpu().numpy()
        memories = memories[:, -self.max_memories:]
        num_memories = memories.shape[1]

        knn_insert_ids = np.arange(num_memories)

        keys = np.ascontiguousarray(memories[..., 0, :])
        knns = [self.knns[i] for i in self.scoped_indices]
        db_offsets = [self.db_offsets[i] for i in self.scoped_indices]

        # use joblib to insert new key / value memories into faiss index

        @delayed
        def knn_add(knn, key, db_offset):
            knn.add(key, ids = knn_insert_ids + db_offset)
            return knn

        updated_knns = Parallel(n_jobs = self.n_jobs)(knn_add(*args) for args in zip(knns, keys, db_offsets))
        for knn_idx, scoped_idx in enumerate(self.scoped_indices):
            self.knns[scoped_idx] = updated_knns[knn_idx]

        # add the new memories to the memmap "database"

        add_indices = (rearrange(np.arange(num_memories), 'j -> 1 j') + rearrange(self.db_offsets[list(self.scoped_indices)], 'i -> i 1')) % self.max_memories
        self.db[rearrange(np.array(self.scoped_indices), 'i -> i 1'), add_indices] = memories
        self.db.flush()

        self.db_offsets += num_memories

    def search(
        self,
        queries,
        topk,
        nprobe = 8,
        increment_hits = True,
        increment_age = True
    ):
        check_shape(queries, 'b ... d', d = self.dim, b = len(self.scoped_indices))
        queries, ps = pack([queries], 'b * d')

        device = queries.device
        queries = queries.detach().cpu().numpy()

        all_masks = []
        all_key_values = []

        knns = [self.knns[i] for i in self.scoped_indices]

        # parallelize faiss search

        @delayed
        def knn_search(knn, query):
            return knn.search(query, topk, nprobe, increment_hits = increment_hits, increment_age = increment_age)

        fetched_indices = Parallel(n_jobs = self.n_jobs)(knn_search(*args) for args in zip(knns, queries))

        # get all the memory key / values from memmap 'database'
        # todo - remove for loop below

        for batch_index, indices in zip(self.scoped_indices, fetched_indices):
            mask = indices !=  -1
            db_indices = np.where(mask, indices, 0)

            all_masks.append(torch.from_numpy(mask))

            key_values = self.db[batch_index, db_indices % self.max_memories]
            all_key_values.append(torch.from_numpy(key_values))

        all_masks = torch.stack(all_masks)
        all_key_values = torch.stack(all_key_values)
        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, '... -> ... 1 1'), 0.)

        all_key_values, = unpack(all_key_values, ps, 'b * n kv d')
        all_masks, = unpack(all_masks, ps, 'b * n')

        return all_key_values.to(device), all_masks.to(device)

    def __del__(self):
        if hasattr(self, 'knns'):
            for knn in self.knns:
                del knn
        del self.db

# extends list with some extra methods for collections of KNN memories

class KNNMemoryList(list):
    def cleanup(self):
        for memory in self:
            del memory

    @classmethod
    def create_memories(
        self,
        *,
        batch_size,
        num_memory_layers,
        memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY
    ):
        memories_path = Path(memories_directory)
        memories_path.mkdir(exist_ok = True, parents = True)

        def inner(*args, **kwargs):
            return self([KNNMemory(*args, num_indices = batch_size, memmap_filename = str(memories_path / f'knn.memory.layer.{ind + 1}.memmap'), **kwargs) for ind in range(num_memory_layers)])
        return inner

    @contextmanager
    def at_batch_indices(
        self,
        indices
    ):
        knn_batch_indices_contexts = [memory.at_batch_indices(indices) for memory in self]
        with multi_context(*knn_batch_indices_contexts):
            yield

    def clear_memory(
        self,
        batch_indices = None,
        memory_indices = None
    ):
        memory_indices = default(memory_indices, tuple(range(len(self))))

        for memory_index in memory_indices:
            memory = self[memory_index]
            memory.clear(batch_indices)


# approximate nearest neighbor attention

class KNNAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        num_retrieved_memories = 32,
        xl_max_memories = 0.,
        attn_scale_init = 20,
        gate_output = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(heads, 1, 1) * math.log(attn_scale_init))

        inner_dim = heads * dim_head
        self.xl_max_memories = xl_max_memories

        self.num_retrieved_memories = num_retrieved_memories

        self.dropout = nn.Dropout(dropout)
        self.knn_mem_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.output_gate = nn.Parameter(torch.zeros(1)) if gate_output else None

    def forward(
        self,
        x,
        *,
        knn_memory,
        xl_memory = None,
        add_knn_memory = True,
        rel_pos_bias = None
    ):
        b, n, h, device = *x.shape[:2], self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        # in paper, they showed normalizing of keys led to more stable training
        # we'll just go with full cosine sim attention https://arxiv.org/abs/2010.04245

        q, k = map(l2norm, (q, k))

        # handle xl memory

        if exists(xl_memory):
            k_xl_mem, v_xl_mem = xl_memory.unbind(dim = -2)
            k = torch.cat((k_xl_mem, k), dim = -2)
            v = torch.cat((v_xl_mem, v), dim = -2)

        # calculate local attention

        scale = self.scale.exp()

        sim = einsum('b h i d, b j d -> b h i j', q, k) * scale
        i, j = sim.shape[-2:]

        if exists(rel_pos_bias):
            sim = rel_pos_bias[..., -i:, -j:] + sim

        mask_value = -torch.finfo(sim.dtype).max

        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, mask_value)

        # calculate knn attention over memory, if index is passed in

        mem_kv, mem_mask = knn_memory.search(q, self.num_retrieved_memories)
        mem_k, mem_v = mem_kv.unbind(dim = -2)

        sim_mem = einsum('b h i d, b h i j d -> b h i j', q, mem_k) * scale
        sim_mem = sim_mem.masked_fill(~mem_mask, mask_value)

        # calculate new XL memories, as well as memories to be discarded

        new_kv_memories = torch.stack((k, v), dim = -2).detach()

        if self.xl_max_memories > 0:
            new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories[:, :-self.xl_max_memories], new_kv_memories[:, -self.xl_max_memories:]
        else:
            new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories, None

        # add memories to be discarded into KNN memory

        if add_knn_memory and new_kv_memories_discarded.numel() > 0:
            knn_memory.add(new_kv_memories_discarded)

        # attention (combining local and distant)

        sim = torch.cat((sim_mem, sim), dim = -1)
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        local_attn, mem_attn = attn[..., self.num_retrieved_memories:], attn[..., :self.num_retrieved_memories]
        local_out = einsum('b h i j, b j d -> b h i d', local_attn, v)
        mem_out = einsum('b h i j, b h i j d -> b h i d', mem_attn, mem_v)

        out = local_out + mem_out

        # combine heads and project out

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # use flamingo styled gating of output, so that memorizing transformers can be gated into an existing LLM
        # preparation to add this to block-recurrent-transformer-pytorch, for the pinnacle of long context attention network

        if exists(self.output_gate):
            out = out * self.output_gate.tanh()

        return out, new_xl_kv_memories

# main class

class MemorizingTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        knn_attn_heads = None,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        memorizing_layers = None,
        max_knn_memories = 250000,
        num_retrieved_memories = 32,
        clear_memories_on_sos_token_id = None,
        clear_memories_on_eos_token_id = None,
        knn_memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
        shift_knn_memories_down = 0.,
        pad_id = 0,
        xl_max_memories = 0,
        xl_memory_layers = None,
        shift_xl_memories_down = 0.,
        knn_memory_multiprocessing = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pad_id = pad_id

        block_wrapper = partial(PreNormResidual, dim)
        valid_layers = set(range(1, depth + 1))

        memorizing_layers = default(memorizing_layers, (depth // 2,)) # default KNN attention layer to midpoint of transformer
        memorizing_layers = cast_tuple(memorizing_layers)
        memorizing_layers = tuple(filter(lambda i: i in valid_layers, memorizing_layers))

        self.dim_head = dim_head

        knn_attn_heads = default(knn_attn_heads, heads)

        # xl memory hyperparameter

        if xl_max_memories > 0:
            xl_memory_layers = default(xl_memory_layers, tuple(range(1, depth + 1)))
            xl_memory_layers = unique(xl_memory_layers)
            self.xl_memory_layers = tuple(filter(lambda i: i in valid_layers, xl_memory_layers))            
            self.num_xl_memory_layers = len(self.xl_memory_layers)
        else:
            self.xl_memory_layers = tuple()
            self.num_xl_memory_layers = 0

        # knn memory hyperparameters

        self.max_knn_memories = max_knn_memories
        self.knn_memories_directory = knn_memories_directory
        self.memorizing_layers = unique(memorizing_layers)
        self.num_memory_layers = len(memorizing_layers)

        self.clear_memories_on_sos_token_id = clear_memories_on_sos_token_id
        self.clear_memories_on_eos_token_id = clear_memories_on_eos_token_id

        # relative positional bias

        self.rel_pos_bias = T5RelativePositionBias(scale = dim_head ** 0.5, heads = heads)
        self.knn_rel_pos_bias = T5RelativePositionBias(scale = dim_head ** 0.5, heads = heads)

        # layers

        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer_num = idx + 1

            use_xl_memories = layer_num in self.xl_memory_layers
            use_knn_attention = layer_num in memorizing_layers
            xl_max_memories_layer = 0 if not use_xl_memories else xl_max_memories

            if use_knn_attention:
                attn = KNNAttention(dim = dim, dim_head = dim_head, heads = knn_attn_heads, dropout = attn_dropout, num_retrieved_memories = num_retrieved_memories, xl_max_memories = xl_max_memories_layer)
            else:
                attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, xl_max_memories = xl_max_memories_layer)

            self.layers.append(nn.ModuleList([
                block_wrapper(attn),
                block_wrapper(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        # memory layer shifting
        # from a little known paper https://arxiv.org/abs/2012.15688

        self.shift_knn_memories_down = shift_knn_memories_down
        self.shift_xl_memories_down = shift_xl_memories_down

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

        # knn memories init

        self.knn_mem_kwargs = dict(
            dim = self.dim_head,
            max_memories = self.max_knn_memories,
            multiprocessing = knn_memory_multiprocessing
        )

    def create_knn_memories(
        self,
        *,
        batch_size
    ):
        return KNNMemoryList.create_memories(
            batch_size = batch_size,
            num_memory_layers = self.num_memory_layers,
            memories_directory = self.knn_memories_directory,
        )(**self.knn_mem_kwargs)

    @contextmanager
    def knn_memories_context(
        self,
        **kwargs
    ):
        knn_dir = Path(self.knn_memories_directory)
        knn_dir.mkdir(exist_ok = True, parents = True)
        lock = FileLock(str(knn_dir / 'mutex'))

        with lock:
            knn_memories = self.create_knn_memories(**kwargs)
            yield knn_memories
            knn_memories.cleanup()

    def clear_memory(self, x, token_id):
        """ clears the KNN memories based on if the batch row contains the specified token id """
        """ for auto-clearing KNN memories based on start and end of strings """

        clear_memory = (x == token_id).any(dim = -1)
        batch_indices, _ = clear_memory.nonzero(as_tuple = True)
        batch_indices_to_clear = batch_indices.tolist()

        if len(batch_indices_to_clear) == 0:
            return

        knn_memories.clear_memory(batch_indices_to_clear)

    def forward(
        self,
        x,
        knn_memories,
        xl_memories = None,
        labels = None,
        add_knn_memory = True
    ):
        batch_size, seq_len, *_, device = *x.shape, x.device
        x = self.token_emb(x)

        # validate KNN memories to have enough indices for batch size

        assert all([memory.num_indices == batch_size for memory in knn_memories]), f'you passed in an input with batch size {batch_size} but your memories were not instantiated with that number of KNN indices'

        # if KNN memories are passed in, and researcher wants memories auto-cleared on <sos> token detection
        # do the appropriate logic

        if exists(self.clear_memories_on_sos_token_id):
            self.clear_memory(x, self.clear_memories_on_sos_token_id)

        # handle XL memories

        xl_memories = default(xl_memories, (None,) * self.num_xl_memory_layers)
        assert len(xl_memories) == self.num_xl_memory_layers
        has_xl_memories = len(xl_memories) > 0

        # shifting memories a number of layers down, little known technique shown to enhance memories from Ernie-Doc paper

        if len(knn_memories) > 0 and self.shift_knn_memories_down > 0:
            knn_memories = [*knn_memories[self.shift_knn_memories_down:], *knn_memories[:self.shift_knn_memories_down]]

        if len(xl_memories) > 0 and self.shift_xl_memories_down > 0:
            xl_memories = [*xl_memories[self.shift_xl_memories_down:], *xl_memories[:self.shift_xl_memories_down]]

        # iterate through the memories in order of the ascending layers that contain KNNAttention

        xl_memories_iter = iter(xl_memories)
        knn_memories_iter = iter(knn_memories)

        # positional bias

        max_context_len = max([seq_len, *map(lambda t: (t.shape[-3] if exists(t) else 0) + seq_len, xl_memories)])

        rel_pos_bias = self.rel_pos_bias(seq_len, max_context_len, device = device)
        knn_rel_pos_bias = self.knn_rel_pos_bias(seq_len, max_context_len, device = device)

        # keep track of new xl memories

        new_xl_memories = [] if has_xl_memories else None

        # go through all layers

        for ind, (attn, ff) in enumerate(self.layers):
            layer_num = ind + 1

            is_memorizing_layer = layer_num in self.memorizing_layers
            is_xl_memory_layer = layer_num in self.xl_memory_layers

            attn_kwargs = dict(rel_pos_bias = rel_pos_bias if not is_memorizing_layer else knn_rel_pos_bias)

            if is_memorizing_layer:
                attn_kwargs = {**attn_kwargs, 'knn_memory': next(knn_memories_iter), 'add_knn_memory': add_knn_memory}

            if is_xl_memory_layer:
                attn_kwargs = {**attn_kwargs, 'xl_memory': next(xl_memories_iter)}

            # attention

            x, xl_mem = attn(x, **attn_kwargs)

            # add new XL memories if needed

            if exists(xl_mem):
                new_xl_memories.append(xl_mem)

            # feedforward

            x = ff(x)

        # to logits

        logits = self.to_logits(x)

        # auto-clear KNN memories on end of string token

        if exists(self.clear_memories_on_eos_token_id):
            self.clear_memory(x, self.clear_memories_on_eos_token_id)

        # for training

        if not exists(labels):
            if exists(new_xl_memories):
                return logits, new_xl_memories

            return logits

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = self.pad_id)

        if exists(new_xl_memories):
            return loss, new_xl_memories

        return loss