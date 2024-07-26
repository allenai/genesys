# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn
import math

import torch.nn.functional as F
from apex.normalization import FusedLayerNorm as LayerNorm

import copy


def MultiwayWrapper(args, module, dim=0):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=0):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


class XPos(nn.Module):
    def __init__(
        self, head_dim, scale_base = 512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, len):
        scale = self.scale ** (torch.arange(0, len, 1) - len // 2).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)
        return (sin, cos, scale)
    
    

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.block_size = args.block_size
        self.half_block_size = self.block_size // 2

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(
            args, nn.Linear(embed_dim, embed_dim, bias=True)
        )
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.embed_dim))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout, inplace=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
    ):
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        src_len, key_bsz, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query) # tgt_len, bsz, dim
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling
        if self.block_size > 0 and tgt_len > self.block_size: # divide block
            assert tgt_len % self.half_block_size == 0
            if incremental_state is not None:
                incremental_state["prev_key"] = k.view(
                    bsz, self.num_heads, -1, self.head_dim
                )
                incremental_state["prev_value"] = v.view(
                    bsz, self.num_heads, -1, self.head_dim
                )

            q = q.view(-1, self.half_block_size, bsz * self.num_heads, self.head_dim).transpose(1, 2).reshape(-1, self.half_block_size, self.head_dim)
            k = F.pad(k, (0, 0, 0, 0, self.half_block_size, 0)).unfold(0, self.block_size, self.half_block_size).reshape(-1, self.head_dim, self.block_size).transpose(1, 2)
            v = F.pad(v, (0, 0, 0, 0, self.half_block_size, 0)).unfold(0, self.block_size, self.half_block_size).reshape(-1, self.head_dim, self.block_size).transpose(1, 2)
            bsz *= tgt_len // self.half_block_size
            tgt_len = self.half_block_size
            src_len = self.block_size
            
        else:
            q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            if incremental_state is not None:
                if "prev_key" in incremental_state:
                    prev_key = incremental_state["prev_key"].view(
                        bsz * self.num_heads, -1, self.head_dim
                    )
                    prev_value = incremental_state["prev_value"].view(
                        bsz * self.num_heads, -1, self.head_dim
                    )
                    k = torch.cat([prev_key, k], dim=1)
                    v = torch.cat([prev_value, v], dim=1)
                incremental_state["prev_key"] = k.view(
                    bsz, self.num_heads, -1, self.head_dim
                )
                incremental_state["prev_value"] = v.view(
                    bsz, self.num_heads, -1, self.head_dim
                )
                src_len = k.size(1)

        if isinstance(rel_pos, tuple): # XPos implementation
            sin, cos, scale = rel_pos
            if self.self_attention:
                k = apply_rotary_pos_emb(k, sin, cos, scale = 1 / scale)
                q = apply_rotary_pos_emb(q, sin[-q.shape[1]:], cos[-q.shape[1]:], scale = scale[-q.shape[1]:])
            else:
                k = apply_rotary_pos_emb(k, sin[:k.shape[1]], cos[:k.shape[1]], scale = 1 / scale[:k.shape[1]])
                q = apply_rotary_pos_emb(q, sin[k.shape[1]:], cos[k.shape[1]:], scale = scale[k.shape[1]:])

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if isinstance(rel_pos, torch.Tensor):
            rel_pos = rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + rel_pos

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        attn_probs = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs, v)
        if bsz > key_bsz: # merge block
            attn = attn.view(-1, key_bsz * self.num_heads, self.half_block_size, self.head_dim).transpose(1, 2).reshape(-1, key_bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).reshape(-1, bsz, embed_dim)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len
        ).transpose(1, 0)
        return attn, attn_weights