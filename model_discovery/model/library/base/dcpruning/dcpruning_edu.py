"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from utils.root_finding import entmax_bisect


class DynamicTensorFast:
    def __init__(
        self,
        batch_size,
        embedding_dims,
        capacity=64,
        resizable=True,
        reduce_fragmentation=False,
        compact=True,
        dtype=torch.float32,
        device="cpu",
        debug=False,
    ):
        self.batch_size = batch_size
        self.capacity = capacity
        self.embedding_dims = embedding_dims
        self.resizable = resizable
        self.reduce_fragmentation = reduce_fragmentation
        self.debug = debug
        self.compact = compact

        self.tensors = []
        for i, embedding_dim in enumerate(embedding_dims):
            if isinstance(embedding_dim, (list, tuple)):
                # Number of heads is specified
                assert len(embedding_dim) == 2
                self.tensors.append(
                    torch.zeros(
                        (batch_size, embedding_dim[0], capacity, embedding_dim[1]),
                        dtype=dtype,
                        device=device,
                    )
                )  # !!!
            else:
                self.tensors.append(
                    torch.zeros(
                        (batch_size, capacity, embedding_dim),
                        dtype=dtype,
                        device=device,
                    )
                )

        self.mask = torch.zeros((batch_size, capacity), dtype=torch.bool, device=device)
        self.max_padded_length = 0

        if self.debug:
            self.token_ids = torch.full(
                (batch_size, capacity), dtype=torch.long, device=device, fill_value=-1
            )
            self.next_token_id = 0

    def to(self, device=None, dtype=None):
        for i in range(len(self.tensors)):
            self.tensors[i] = self.tensors[i].to(device=device, dtype=dtype)
        self.mask = self.mask.to(device=device)
        if self.debug:
            self.token_ids = self.token_ids.to(device=device)

    def _effective_size(self):
        if self.reduce_fragmentation and self.max_padded_length > 0:
            return 2 ** int(math.ceil(math.log2(self.max_padded_length)))
        else:
            return self.max_padded_length

    def _resize(self, new_capacity):
        for i, old_tensor in enumerate(self.tensors):
            if len(old_tensor.shape) == 4:
                new_tensor = torch.zeros(
                    (
                        old_tensor.shape[0],
                        old_tensor.shape[1],
                        new_capacity,
                        old_tensor.shape[3],
                    ),
                    dtype=old_tensor.dtype,
                    device=old_tensor.device,
                )
            else:
                new_tensor = torch.zeros(
                    (old_tensor.shape[0], new_capacity, old_tensor.shape[2]),
                    dtype=old_tensor.dtype,
                    device=old_tensor.device,
                )
            new_tensor[..., : self.capacity, :] = old_tensor[..., : self.capacity, :]
            self.tensors[i] = new_tensor

        new_mask = torch.zeros(
            (self.mask.shape[0], new_capacity),
            dtype=self.mask.dtype,
            device=self.mask.device,
        )
        new_mask[:, : self.capacity] = self.mask[:, : self.capacity]
        self.mask = new_mask

        if self.debug:
            new_token_ids = torch.full(
                (self.token_ids.shape[0], new_capacity),
                dtype=self.token_ids.dtype,
                device=self.token_ids.device,
                fill_value=-1,
            )
            new_token_ids[:, : self.capacity] = self.token_ids[:, : self.capacity]
            self.token_ids = new_token_ids

        self.capacity = new_capacity

    def append(self, tensors) -> torch.LongTensor:
        # Sanity check
        assert len(tensors) == len(self.embedding_dims)
        for tensor, embedding_dim in zip(tensors, self.embedding_dims):
            if isinstance(embedding_dim, (tuple, list)):
                # Number of heads is specified
                assert len(tensor.shape) == 3
                assert tensor.shape[0] == self.batch_size
                assert tensor.shape[1] == embedding_dim[0]
                assert tensor.shape[2] == embedding_dim[1]
            else:
                assert len(tensor.shape) == 2
                assert tensor.shape[0] == self.batch_size
                assert tensor.shape[1] == embedding_dim

        # Find insertion point
        effective_size = self._effective_size()
        if effective_size == 0:
            max_length = 0
            self.max_padded_length = 1
            insertion_point = torch.zeros(
                (self.batch_size,), device=self.mask.device, dtype=torch.long
            )
        else:
            mask = self.mask[:, :effective_size]
            result = mask.min(dim=1)
            insertion_point = (
                result.indices * (~result.values) + mask.shape[1] * result.values
            )
            max_length = insertion_point.max().item()
            self.max_padded_length = max(self.max_padded_length, max_length + 1)

        if max_length == self.capacity:
            # Needs resizing
            if not self.resizable:
                raise RuntimeError(
                    "The pre-allocated buffer has been exhausted. "
                    "Increase the capacity or set resizable=True."
                )
            new_capacity = (self.capacity * 2) if self.capacity > 0 else 1
            self._resize(new_capacity)

        for i, tensor in enumerate(tensors):
            if len(tensor.shape) == 3:
                self.tensors[i].scatter_(
                    2,
                    insertion_point[:, None, None, None].expand(
                        -1, tensor.shape[1], -1, tensor.shape[-1]
                    ),
                    tensor[:, :, None],
                )
            else:
                self.tensors[i].scatter_(
                    1,
                    insertion_point[:, None, None].expand(-1, -1, tensor.shape[-1]),
                    tensor[:, None],
                )

        self.mask.scatter_(1, insertion_point[:, None], True)

        if self.debug:
            self.token_ids.scatter_(1, insertion_point[:, None], self.next_token_id)
            self.next_token_id += 1

        return insertion_point

    def remove(self, mask: torch.BoolTensor):
        expected_size = self._effective_size()
        assert mask.shape[0] == self.batch_size
        assert mask.shape[1] == expected_size
        assert len(mask.shape) == 2
        inv_mask = ~mask
        self.mask[:, :expected_size] &= inv_mask
        if self.debug:
            self.token_ids[:, :expected_size] *= inv_mask
            self.token_ids[:, :expected_size] += mask * (-1)

        if self.compact:
            # Compute load factor
            mask = self.mask[:, : self.max_padded_length]
            ratio = mask.sum(dim=1).max().item() / mask.shape[1]

        if self.compact and ratio < 0.9:
            # Find offset
            mask = self.mask[:, :expected_size]
            result = mask.min(dim=1)
            insertion_point = (
                result.indices * (~result.values) + mask.shape[1] * result.values
            )
            offset = insertion_point.min().item()
            if self.reduce_fragmentation and offset > 0:
                offset = 2 ** int(math.floor(math.log2(offset)))

            # Compact data structure
            indices = torch.argsort(~self.mask[:, offset:expected_size].long()) + offset
            self.mask[:, offset:expected_size] = self.mask.gather(1, indices)
            if self.debug:
                self.token_ids[:, offset:expected_size] = self.token_ids.gather(
                    1, indices
                )
            for i, (tensor, emb_dim) in enumerate(
                zip(self.tensors, self.embedding_dims)
            ):
                if isinstance(emb_dim, (tuple, list)):
                    indices_ = indices[:, None, :, None].expand(
                        -1, emb_dim[0], -1, emb_dim[1]
                    )
                    self.tensors[i][:, :, offset:expected_size] = tensor.gather(
                        2, indices_
                    )
                else:
                    indices_ = indices[:, :, None].expand(-1, -1, emb_dim)
                    self.tensors[i][:, offset:expected_size] = tensor.gather(
                        1, indices_
                    )

            # Find new max padded length
            mask_sum = torch.flip(self.mask[:, offset:expected_size].any(dim=0), (0,))
            result = mask_sum.max(dim=0)
            last_value = result.values.item()
            padded_length = mask_sum.shape[0] - result.indices.item() + offset
            if last_value:
                self.max_padded_length = padded_length
            else:
                self.max_padded_length = 0

    def values(self, tensor_ids=None):
        padded_length = self._effective_size()
        tensors = []
        for i, (tensor, emb_dim) in enumerate(zip(self.tensors, self.embedding_dims)):
            if tensor_ids is None or i in tensor_ids:
                tensors.append(tensor[..., :padded_length, :])
        return tensors, self.mask[:, :padded_length]

    def get_token_ids(self, compact=False):
        assert self.debug
        assert (self.token_ids[:, self.max_padded_length :] == -1).all()
        if compact:
            result = []
            for row in self.token_ids[:, : self._effective_size()]:
                ids = row[torch.where(row != -1)].sort().values
                result.append(ids)
            return result
        else:
            return self.token_ids[:, : self._effective_size()]

    def get_dense_mask(self) -> torch.BoolTensor:
        assert self.debug
        token_ids = self.token_ids[:, : self.max_padded_length]
        mask = torch.zeros(
            (self.batch_size, self.next_token_id + 1),
            dtype=torch.bool,
            device=token_ids.device,
        )

        # Index 0 is a dummy index to deal with gaps (token_id = -1)
        mask.scatter_(1, token_ids + 1, True)
        return mask[:, 1:]

    def get_dense_values(self):
        assert self.debug
        result = []
        for row_idx, row in enumerate(self.token_ids[:, : self._effective_size()]):
            ids = row.argsort()[(row == -1).sum() :]
            sub_result = []
            for tensor, emb_dim in zip(self.tensors, self.embedding_dims):
                if isinstance(emb_dim, (tuple, list)):
                    # Restore correct shape (number of heads)
                    tensor = tensor.view(self.batch_size, emb_dim[0], -1, emb_dim[1])
                sub_result.append(tensor[row_idx, ..., ids, :])
            result.append(sub_result)
        return result


class CausalSelfAttention(nn.Module):
    def __init__(self, block_num, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        self.sparse_attention = config.sparse_attention
        self.block_num = block_num
        self.config = config

        if self.sparse_attention:
            # initialize alpha to 1, this will be overwritten
            # setting alpa to 1 is unstable, so we set it to 1 + eps
            self.sparsity_alpha = 1.000001

            self.int_n_embd = config.int_n_embd if config.int_n_embd else config.n_embd

            self.q_int = nn.Linear(config.n_embd, self.int_n_embd, bias=False)
            self.k_int = nn.Linear(config.n_embd, self.int_n_embd, bias=False)

            self.int_bias = nn.Parameter(
                torch.ones(
                    1,
                )
                * config.sparse_attention_int_bias,
            )

            torch.nn.init.normal_(
                self.q_int.weight, mean=0.0, std=1 / math.sqrt(config.n_embd)
            )
            torch.nn.init.normal_(
                self.k_int.weight, mean=0.0, std=1 / math.sqrt(config.n_embd)
            )

            # bias for the dropping probabilities. Here we assume a token does not drop itself.
            self.register_buffer(
                "bias_int",
                torch.tril(
                    torch.ones(config.block_size, config.block_size), diagonal=-1
                ).view(1, 1, config.block_size, config.block_size),
            )

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # We use torch 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        assert self.flash

        # bias for the attention mask for casual decoding
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(
        self,
        x,
        prev_attn_mask=None,
        mask=None,
        store=None,
        validity_map=None,
        first_generation=False,
    ):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        device = x.device

        q, k, v = self.c_attn.forward(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.sparse_attention:
            q_int = self.q_int(x)
            k_int = self.k_int(x)

        store_mask = None
        insertion_indices = None

        if store is not None:
            # we are using caching while generating
            if first_generation:
                # if this is the first generation step, we need to insert everything into the store
                for i in range(T):
                    # add everything to the cache
                    if not self.sparse_attention:
                        store.append([k[:, :, i, :], v[:, :, i, :]])
                    else:
                        store.append([k[:, :, i, :], v[:, :, i, :], k_int[:, i, :]])

                # get the correct last insertion indices, due to padding within the prefixes
                insertion_indices = torch.sum(validity_map, dim=-1) - 1

                # After inseting in the store, remove based on the padding mask
                _, store_mask = store.values()
                store_mask = store_mask.clone()
                store_mask[:, : validity_map.shape[-1]] = validity_map

                store.remove(torch.logical_not(store_mask))
            else:
                # add new elements to the store
                if not self.sparse_attention:
                    store.append([k[:, :, 0, :], v[:, :, 0, :]])

                    (k, v), store_mask = store.values()
                else:
                    insertion_indices = store.append(
                        [k[:, :, 0, :], v[:, :, 0, :], k_int[:, 0, :]]
                    )

                    (k, v, k_int), store_mask = store.values()

                validity_map = store_mask

        context_T = k.shape[2]

        if not self.sparse_attention:
            # regular causal attention
            attn_mask = torch.zeros(B, 1, T, context_T, device=x.device, dtype=x.dtype)

            if validity_map is not None:
                # filter out the attention mask to only include the tokens that are not yet processed
                attn_mask = attn_mask.masked_fill(
                    validity_map[:, None, None, :] == 0,
                    float("-inf"),
                )

            cumprobs = 0  # for compatibility
        else:
            p_int_raw = (
                (
                    torch.matmul(q_int, k_int.transpose(-1, -2))
                    / math.sqrt(self.int_n_embd)
                    + self.int_bias
                )
                .unsqueeze(1)
                .unsqueeze(-1)
            )

            if self.sparsity_alpha == "inf":
                # in eval mode we replace the alpha-sigmoid with the step function
                p_int = (p_int_raw > 0)[..., 0]
            else:
                # Compare the raw drop scores with the values 0 to get the drop probabilities.
                p_int_raw = torch.cat([p_int_raw, torch.zeros_like(p_int_raw)], dim=-1)

                # Take only the first value of the entmax_bisect output, which is the probability of dropping.
                p_int = entmax_bisect(p_int_raw.to(torch.float32), self.sparsity_alpha)[
                    ..., 0
                ]

            if store is not None:
                # here we need to drop from the store
                if first_generation:
                    p_int = p_int.float()
                    p_int = p_int.masked_fill(self.bias_int[:, :, :T, :T] == 0, 1)

                    p_int = p_int.masked_fill(validity_map[:, None, None, :] == 0, 0)

                    # Multiply together probs from the previous tokens.
                    cumprobs = torch.cumprod(p_int, dim=-2)

                    attn_mask = torch.log(cumprobs)

                    if prev_attn_mask is not None:
                        attn_mask = attn_mask + prev_attn_mask

                    store_mask[:, : validity_map.shape[-1]] = cumprobs[
                        torch.arange(B, device=device), 0, insertion_indices, :
                    ].bool()

                    store.remove(torch.logical_not(store_mask))
                else:
                    # specify that we cannot drop ourselves
                    p_int[
                        torch.arange(B, device=device), 0, 0, insertion_indices
                    ] = True

                    p_int = torch.logical_and(p_int, validity_map[:, None, None, :])

                    attn_mask = p_int  # scaled_dot_product_attention can also handle boolean masks
                    cumprobs = None

                    store.remove(torch.logical_not(p_int[:, 0, 0, :]))
            else:
                # training phase
                p_int = p_int.masked_fill(self.bias_int[:, :, :T, :T] == 0, 1)

                if validity_map is not None:
                    p_int = p_int.masked_fill(validity_map[:, None, None, :] == 0, 0)

                # Multiply together probs from the previous tokens.
                cumprobs = torch.cumprod(p_int, dim=-2)

                # Just for stability reasons add an epsilon ...
                attn_mask = torch.log(cumprobs + (1e-40 if self.training else 0)).to(
                    p_int_raw.dtype
                )

                if prev_attn_mask is not None:
                    attn_mask = attn_mask + prev_attn_mask

        if T == context_T:
            # Add casual masking, only during training
            attn_mask = attn_mask.masked_fill(
                self.bias[:, :, :T, :T] == 0, float("-inf")
            )

        if mask is not None:  # masking of tokens during training
            attn_mask = attn_mask.masked_fill(
                mask[:, None, None, :] == 0, float("-inf")
            )

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, cumprobs, attn_mask


