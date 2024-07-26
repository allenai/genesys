# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_seq import FeedForwardLayer, TransformerOutput
from models.utils import pos_emb, skew, unskew





class AdaptiveMask(nn.Module):
    def __init__(
        self,
        size,
        ramp_size,
        init_ratio=0,
        shape=(1,),
    ):
        super(AdaptiveMask, self).__init__()
        self.size = size
        self.ramp_size = ramp_size
        self.size_ratio = nn.Parameter(torch.zeros(*shape) + init_ratio)
        mask_template = torch.linspace(1 - size, 0, steps=size)
        self.register_buffer("mask_template", mask_template)

    def prepare_mask(self, span):
        mask = self.mask_template + self.size_ratio * self.size
        mask = mask / self.ramp_size + 1
        mask = mask.clamp(0, 1)
        if span < self.size:
            # the input could have been trimmed beforehand to save computation
            mask = mask.narrow(-1, self.size - span, span)
        self.mask_prepared = mask

    def forward(self, x):
        if hasattr(self, "mask_prepared"):
            return x * self.mask_prepared

        mask = self.mask_template + self.size_ratio * self.size
        mask = mask / self.ramp_size + 1
        mask = mask.clamp(0, 1)
        if x.size(-1) < self.size:
            # the input could have been trimmed beforehand to save computation
            mask = mask.narrow(-1, self.size - x.size(-1), x.size(-1))
        x = x * mask
        return x

    def get_max_size(self, include_ramp=True):
        max_size = self.size_ratio.max().item()
        max_size = max_size * self.size
        if include_ramp:
            max_size += self.ramp_size
        max_size = max(0, min(self.size, math.ceil(max_size)))
        return max_size

    def get_avg_size(self, include_ramp=True):
        avg_size = self.size_ratio.mean().item()
        avg_size = avg_size * self.size
        if include_ramp:
            avg_size += self.ramp_size
        avg_size = max(0, min(self.size, math.ceil(avg_size)))
        return avg_size

    def param_clamp(self):
        self.size_ratio.data.clamp_(0, 1)


class AdaptiveSpan(nn.Module):
    def __init__(self, args, size, loss_coeff, ramp_size, init_ratio):
        super(AdaptiveSpan, self).__init__()
        self.size = size
        self.loss_coeff = loss_coeff
        self.args = args
        if self.args.adapt_span_layer:
            self.mask = AdaptiveMask(self.size, ramp_size, init_ratio=init_ratio)
        else:
            self.mask = AdaptiveMask(
                self.size, ramp_size, init_ratio=init_ratio, shape=(args.nheads, 1, 1),
            )

    def forward(self, attn):
        if self.args.adapt_span_layer:
            attn = self.mask(attn)
        elif self.args.feedback:
            B = attn.size(0)
            attn = attn.reshape(B // self.args.nheads, self.args.nheads, 1, -1)
            attn = self.mask(attn)
            attn = attn.view(B, -1)
        else:
            B, M = attn.size(0), attn.size(1)
            attn = attn.reshape(B // self.args.nheads, self.args.nheads, M, -1)
            attn = self.mask(attn)
            attn = attn.view(B, M, -1)
        return attn

    # how many steps can be skipped
    def get_trim_len(self):
        L = self.size
        trim_len = min(L - 1, L - self.mask.get_max_size())
        trim_len = (
            math.floor(trim_len / self.args.adapt_span_trim_step)
            * self.args.adapt_span_trim_step
        )  # for better memory caching
        return trim_len

    # determine how long the cache should be
    def get_cache_size(self):
        trim_len = self.get_trim_len()
        # give a buffer of 64 steps as spans can increase during training
        return min(self.size, self.size - trim_len + 64)

    # trim out unnecessary memory computation
    def trim_memory(self, key, value, key_pe, val_pe):
        trim_len = self.get_trim_len()
        if key is not None:
            if self.args.feedback:
                cache_size = key.size(1)
            else:
                cache_size = key.size(1) - self.args.mem_sz
            trim_len_cache = trim_len - (self.size - cache_size)
            if self.args.feedback:
                # keys and values must have cut to the right sizes beforehand.
                # Also adapt_span_cache=False, so cache can't be shorter.
                assert trim_len_cache == 0
            if trim_len_cache > 0:
                key = key[:, trim_len_cache:, :]
                value = value[:, trim_len_cache:, :]
            elif trim_len_cache < 0:
                print(
                    "warning: cache is too short. cache_size={} trim_len={}".format(
                        cache_size, trim_len
                    )
                )
                key = F.pad(key, [0, 0, -trim_len_cache, 0])
                value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
            if val_pe is not None:
                val_pe = val_pe[:, trim_len:, :]
        return key, value, key_pe, val_pe

    # compute the loss
    def get_loss(self):
        return self.mask.size_ratio.mean() * self.loss_coeff * self.size

    def param_clamp(self):
        self.mask.param_clamp()

class SeqAttention(nn.Module):
    """
    Sequential self-attention layer.

    Each position only attends to its previous L positions (doesn't include the current
    position) using relative position embeddings.
    """

    def __init__(self, args):
        super(SeqAttention, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        self.key_pe, self.val_pe = None, None
        self.key_pe = pos_emb(args, (1, args.head_dim, args.attn_lim))

        if self.args.adapt_span:
            self.adaptive_span = AdaptiveSpan(
                args,
                args.attn_lim,
                args.adapt_span_loss,
                args.adapt_span_len,
                args.adapt_span_init,
            )

    def forward(self, query, key, value):
        # query = B x M x H
        # key, value = B x (M+L) x H
        aux_loss = 0

        key_pe, val_pe = self.key_pe, self.val_pe
        if self.args.adapt_span:
            key, value, key_pe, val_pe = self.adaptive_span.trim_memory(
                key, value, key_pe, val_pe
            )

        attn = 0

        # compute attention from context
        attn = torch.matmul(
            query, key.transpose(-1, -2)
        )  # B x M (dest) x (M+L) (src)
        attn = unskew(attn)  # B x M x L

        # compute the effect of position embedding
        attn = attn + torch.matmul(query, key_pe)  # B x M x L

        attn = attn / math.sqrt(self.args.head_dim)  # B x M X L
        attn = F.softmax(attn, dim=-1)
        if self.args.adapt_span:
            attn = self.adaptive_span(attn)
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        attn = self.dropout(attn)  # B x M X L

        out = 0
        attn_cont = skew(attn, 0)  # B x M X (L+M)
        out = out + torch.matmul(attn_cont, value)  # B x M x H

        return out, aux_loss


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadSeqAttention, self).__init__()
        self.args = args
        self.attn = SeqAttention(args)

        self.proj_query = nn.Linear(
            args.hid_sz, args.head_dim * args.nheads, bias=False
        )
        self.proj_out = nn.Linear(args.head_dim * args.nheads, args.hid_sz, bias=False)
        if self.args.pre_norm:
            self.proj_out.weight.data.div_(math.sqrt(self.args.nlayers * 2))
        self.proj_val = nn.Linear(
            args.hid_sz, args.head_dim * args.nheads, bias=False
        )
        self.proj_key = nn.Linear(
            args.hid_sz, args.head_dim * args.nheads, bias=False
        )

    def head_reshape(self, x):
        K = self.args.nheads
        D = self.args.head_dim
        sz = x.size()
        sz = sz[:-1] + (K, D)  # B x (M+L) x K x D
        x = x.view(sz)  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value):
        B = query.size(0)
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out, aux_loss = self.attn(query, key, value)  # B_K x M x D
        out = out.view(B, self.args.nheads, M, self.args.head_dim)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)  # B x M x H
        return out, aux_loss



def add_args(parser):
    parser.add_argument(
        "--staircase-size",
        type=int,
        default=64,
        help="number of tokens in each transformer forward",
    )
    parser.add_argument(
        "--max-staircase-size-forward",
        type=int,
        default=63,
        help="max number of fresh tokens considered",
    )
    parser.add_argument(
        "--fix-staircase-size-forward",
        type=int,
        default=-1,
        help="max number of fresh tokens considered",
    )
    parser.add_argument(
        "--validation-staircase-size-forward",
        type=int,
        default=32,
        help="max number of fresh tokens considered during validation",
    )
    parser.add_argument(
        "--staircase-module-fixed-length",
        action="store_true",
        default=False,
        help="init h_prev with 0s to ensure the transformer module has fixed forward length",
    )
    parser.add_argument(
        "--out-drop",
        type=float,
        default=0,
        help="insert a dropout before the last linear layer",
    )
    parser.add_argument(
        "--emb-drop", type=float, default=0, help="dropout on input embedding"
    )


class StaircaseSeqAttention(nn.Module):
    """
    Sequential self-attention layer.

    Each position only attends to its previous L positions (doesn't include the current
    position) using relative position embeddings.
    """

    def __init__(self, args):
        super(StaircaseSeqAttention, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        self.key_pe, self.val_pe = None, None
        self.key_pe = pos_emb(args, (1, args.head_dim, args.attn_lim))

    def forward(self, query, key, value):
        # query = B x M x H
        # key, value = B x (M+L) x H
        # mask_causal M * L
        mask_causal = query.new_zeros(
            key.size(1), key.size(1)).fill_(float("-inf"))
        mask_causal = mask_causal.triu(diagonal=1)
        mask_causal = mask_causal[-query.size(1):, ]
        aux_loss = 0

        key_pe, val_pe = self.key_pe, self.val_pe

        attn = 0

        attn = torch.matmul(query, key.transpose(-1, -2))

        L_size = attn.size(-1)
        attn_pos = torch.matmul(query, key_pe[:, :, -L_size:])  # B x M x L
        attn_pos = skew(attn_pos, 0)  # B x M x (N + L)
        attn_pos = attn_pos[:, :, -L_size - 1: -1]  # B x M x L
        attn = attn + attn_pos
        attn = attn + mask_causal

        attn = attn / math.sqrt(self.args.head_dim)  # B x M X L
        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)  # B x M X L

        out = 0

        out = out + torch.matmul(attn, value)  # B x S x H

        return out, aux_loss


class StaircaseMultiHeadSeqAttention(MultiHeadSeqAttention):
    def __init__(self, args):
        super(StaircaseMultiHeadSeqAttention, self).__init__(args)
        self.args = args
        self.attn = StaircaseSeqAttention(args)


class TransformerModLayer(nn.Module):
    def __init__(self, args, layer_ind):
        super(TransformerModLayer, self).__init__()
        self.args = args
        self.attn = StaircaseMultiHeadSeqAttention(args)
        self.ff = FeedForwardLayer(args)
        self.norm1 = nn.LayerNorm(args.hid_sz)
        self.norm2 = nn.LayerNorm(args.hid_sz)

    def attention(self, query, key, value):
        return self.attn(query, key, value)[0]

    def forward(self, h, context, **kargs):
        # h = B x S x H
        if self.args.pre_norm:
            # add layer norm on context as well
            context = self.norm1(context)
            attn_out = self.attention(self.norm1(h), context, context)
        else:
            attn_out = self.attention(h, context, context)

        # FF
        if self.args.pre_norm:
            h2 = h + attn_out  # B x S x H
            ff_out = self.ff(self.norm2(h2))
            out = h2 + ff_out  # B x S x H
        else:
            h2 = self.norm1(h + attn_out)  # B x S x H
            ff_out = self.ff(h2)
            out = self.norm2(h2 + ff_out)  # B x S x H

        return out

    def get_cache_size(self):
        return 0


class TransformerMod(nn.Module):
    def __init__(self, args):
        super(TransformerMod, self).__init__()
        self.args = args
        self.build_layers()
        for l in range(1, len(self.layers)):
            self.layers[l].attn.attn.key_pe = self.layers[0].attn.attn.key_pe
            self.layers[l].attn.attn.val_pe = self.layers[0].attn.attn.val_pe

    def build_layers(self):
        self.layers = nn.ModuleList()
        for l in range(self.args.nlayers):
            self.layers.append(TransformerModLayer(self.args, l))

    def get_layer(self, layer_ind):
        return self.layers[layer_ind]

    def forward(self, h, context):
        # h : B x S x H
        for l in range(self.args.nlayers):
            # only forward size get updated from the context as well
            h = self.get_layer(l)(h, context)  # B x S x H
            forward_size = h.size(1)
            if h.size(1) == context.size(1):
                # self-attention
                context = h
            else:
                context = torch.cat([context[:, :-forward_size, :], h], dim=1)
        return h


class StaircaseModel(nn.Module):
    def __init__(self, args):
        super(StaircaseModel, self).__init__()
        self.args = args
        self.transformer = TransformerMod(args)
        self.fix_staircase_size_forward = self.args.fix_staircase_size_forward
        self.staircase_size = self.args.staircase_size
        self.mem_size = self.args.mem_sz
        self.hidden_size = self.args.hid_sz

        assert self.mem_size % self.fix_staircase_size_forward == 0
        assert self.staircase_size % self.fix_staircase_size_forward == 0
        self.validation_staircase_size_forward = (
            self.args.validation_staircase_size_forward
        )

        self.in_emb = nn.Embedding(args.vocab_sz, args.hid_sz)

        self.out = TransformerOutput(args)
        if args.emb_drop > 0:
            self.emb_dropout = nn.Dropout(args.emb_drop)
        if args.out_drop > 0:
            self.out_dropout = nn.Dropout(args.out_drop)
        if self.args.pre_norm:
            self.out_norm = nn.LayerNorm(args.hid_sz)

    def init_hid_cache(self, batch_sz):
        # creates a cache of # steps
        # 256 / 64 = 4
        # cache size
        # 192
        # 128
        # 64
        # 0
        steps = self.staircase_size // self.fix_staircase_size_forward
        hid = []
        for i in range(steps):
            cache_size = self.staircase_size - \
                (i + 1) * self.fix_staircase_size_forward
            if cache_size > 0:
                hid.append(
                    [
                        torch.zeros((batch_sz, cache_size, self.hidden_size)).to(
                            self.args.device
                        )
                        for i in range(self.args.nlayers)
                    ]
                )
        return hid

    def get_cache(self, h_prev, idx):
        if idx >= len(h_prev):
            return [None]
        return h_prev[idx]

    def assemble_context(self, cache, prev_outputs, new_tokens):
        context = [cache, prev_outputs, new_tokens]
        context = [i for i in context if i is not None]
        context = torch.cat(context, dim=1)
        return context

    def assemble_query(self, prev_outputs, new_tokens):
        query = [prev_outputs, new_tokens]
        query = [i for i in query if i is not None]
        query = torch.cat(query, dim=1)
        return query

    def get_new_tokens(self, h, start_idx, end_idx):
        if end_idx > h.size(1):
            return None
        return h[:, start_idx:end_idx, :]

    def forward(self, x, h_prev, target=None, **kargs):
        # input h B x M
        # assume h_prev [B, staircase_size], and will init with 0s
        # create output placeholder
        # output [B, mem_size, hidden_size]
        hid_after_embed = self.in_emb(x)  # B x M x H
        if self.args.emb_drop > 0:
            hid_after_embed = self.emb_dropout(hid_after_embed)
        # no cache between forwards

        output = hid_after_embed.new_zeros(
            (hid_after_embed.size(0), self.mem_size, self.hidden_size)
        )
        start_idx = 0
        # generate scheduling for staircases, randomness for training purpose
        total_steps = (
            self.mem_size + self.staircase_size
        ) // self.fix_staircase_size_forward - 1
        prev_output = None
        for step_idx in range(total_steps):
            end_idx = start_idx + self.fix_staircase_size_forward
            cache = self.get_cache(h_prev, step_idx)
            new_tokens = self.get_new_tokens(
                hid_after_embed, start_idx, end_idx)
            # should put into cache:
            cache_id = step_idx - (
                total_steps
                - (self.staircase_size // self.fix_staircase_size_forward)
                + 1
            )
            # consumed all forward steps
            # when should we move prev_output forwards
            if step_idx >= self.staircase_size // self.fix_staircase_size_forward:
                prev_output = prev_output[:,
                                          self.fix_staircase_size_forward:, :]
            # input to the first layer of the model
            context = self.assemble_context(
                cache[0], prev_output, new_tokens)
            h = self.assemble_query(prev_output, new_tokens)
            assert context.size(1) <= self.staircase_size
            # forward into layers
            cache_for_next = []
            for layer in range(self.args.nlayers):
                # only forward size get updated from the context as well
                h = self.transformer.get_layer(layer)(h, context)
                if h.size(1) == context.size(1):
                    # self-attention
                    context = h
                elif layer + 1 < self.args.nlayers:
                    context = torch.cat([cache[layer + 1], h], dim=1)
                # put into cache
                if cache_id >= 0:
                    cache_for_next.append(h)
            # the output from the last layer
            prev_output = h
            # put into cache
            if len(cache_for_next) > 0:
                h_prev[cache_id] = cache_for_next

            start_idx = end_idx
            # put into output
            if step_idx - self.staircase_size // self.fix_staircase_size_forward >= -1:
                offset = (
                    step_idx
                    - self.staircase_size // self.fix_staircase_size_forward
                    + 1
                )
                output[
                    :,
                    offset
                    * self.fix_staircase_size_forward: offset
                    * self.fix_staircase_size_forward
                    + prev_output.size(1),
                    :,
                ] = prev_output
        out = output
        if self.args.pre_norm:
            out = self.out_norm(out)
        if self.args.out_drop > 0:
            out = self.out_dropout(out)
        out = self.out(out, target)
        # feeding to the next transformer step.
        return out, h_prev, 0.0