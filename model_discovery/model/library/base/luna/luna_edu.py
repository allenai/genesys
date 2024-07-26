# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils
from fairseq.modules import LayerNorm, LunarMultiheadAttention
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout

import math
from typing import Dict, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor, nn

from fairseq.incremental_decoding_utils import with_incremental_state

@with_incremental_state
class LunarCausalAttention(nn.Module):
    """Lunar Causal attention.
    See "Linformer: Self-Attention with Linear Complexity" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        tie_kv=True,
        q_noise=0.0,
        qn_block_size=8,
        parallel=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.pq_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.pc_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if tie_kv:
            self.c_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
            self.k_proj = self.v_proj = None
        else:
            self.k_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
            self.v_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
            self.c_proj = None

        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

    def _compute_pattention(self, pq, context, key_padding_mask):
        # N x B x D
        len, bsz, dim = context.size()
        # N x B x D
        k = self.pc_proj(context)
        # N x B*H x K -> B*H x N x K
        k = k.view(len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # B x H x L x K -> B*H x L x K -> B*H x K x L
        pq = pq.view(bsz * self.num_heads, -1, self.head_dim).transpose(1, 2)
        # B*H x N x L
        pattn = k.bmm(pq)
        pattn = F.softplus(pattn, beta=math.log(2.0))
        return pattn

    def forward(
        self,
        query,
        pquery,
        key_padding_mask: Optional[Tensor] = None,
        pkey_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            pkey_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, proj_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        pq = None
        num_steps = None
        saved_state = None
        key_accum_mat = None
        value_accum_mat = None

        if pq is None:
            plen = pquery.size(0)
            # L x B x D -> L x B x H x K
            pq = self.pq_proj(pquery).view(plen, bsz, self.num_heads, self.head_dim)
            # L x B x H x K -> B x H x L x K
            pq = pq.permute(1, 2, 0, 3) * self.scaling

        plen = pq.size(2)
        # B*H x N x L
        pattn_weights = self._compute_pattention(pq, query, key_padding_mask)
        pattn_weights = self.dropout_module(pattn_weights)

        # N x B x D -> B*H x N x K
        q = self.q_proj(query) * self.scaling
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # N x B x D -> B*H x N x K
        if self.c_proj is not None:
            k = v = self.c_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        else:
            k = self.k_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = self.v_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        efficient_causal_attention = efficient_causal_attention_parallel if self.parallel else efficient_causal_attention_seq

        if saved_state is not None:
            # key accumulative matrix are store with shape (bsz, num_heads, head_dim, plen)
            if "prev_key_accum_mat" in saved_state:
                _prev_key_accum_mat = saved_state["prev_key_accum_mat"]
                key_accum_mat = _prev_key_accum_mat.view(bsz * self.num_heads, self.head_dim, plen)
            # value accumulative matrix are store with shape (bsz, num_heads, plen, head_dim)
            if "prev_value_accum_mat" in saved_state:
                _prev_value_accum_mat = saved_state["prev_value_accum_mat"]
                value_accum_mat = _prev_value_accum_mat.view(bsz * self.num_heads, plen, self.head_dim)
            if "prev_num_steps" in saved_state:
                _prev_num_steps = saved_state["prev_num_steps"]
                num_steps = _prev_num_steps.view(bsz * self.num_heads) + 1.0

        if num_steps is None:
            num_steps = query.new_ones(bsz * self.num_heads)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if pkey_padding_mask is not None and pkey_padding_mask.dim() == 0:
            pkey_padding_mask = None

        attn_weights = efficient_causal_attention(q, k, pattn_weights)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, plen]

        if pkey_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, plen)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(pkey_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(pkey_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, plen)

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = efficient_causal_attention(attn_probs, pattn_weights, v)

        if saved_state is not None:
            saved_state["prev_pquery"] = pq
            saved_state["prev_key_accum_mat"] = key_accum_mat.view(bsz, self.num_heads, self.head_dim, plen)
            saved_state["prev_value_accum_mat"] = value_accum_mat.view(bsz, self.num_heads, plen, self.head_dim)
            saved_state["prev_num_steps"] = num_steps.view(bsz, self.num_heads)
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, plen).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


def efficient_causal_attention_seq(x, y, z):
    """
    efficient causal attention operation
    Args:
        x (Tensor): Tensor with shape `(batch, n, d1)`
        y (Tensor): Tensor with shape `(batch, n, d1)`
        z (Tensor): Tensor with shape '(batch, n, d2)`

    return:
    """
    n = x.size(1)
    rets = []
    accum_mat = 0
    for i in range(n):
        xx = x[:, i:i + 1] # B x 1 x d1
        yy = y[:, i:i + 1] # B x 1 x d1
        zz = z[:, i:i + 1] # B x 1 x d2

        # B x d1 x d2
        accum_mat = accum_mat + torch.bmm(yy.transpose(1, 2), zz)
        # B x 1 x d2
        rets.append(torch.bmm(xx, accum_mat).div(i + 1.))
    # B x N x d2
    return torch.cat(rets, dim=1)


def efficient_causal_attention_parallel(x, y, z):
    """
    efficient causal attention operation
    Args:
        x (Tensor): Tensor with shape `(batch, n, d1)`
        y (Tensor): Tensor with shape `(batch, n, d1)`
        z (Tensor): Tensor with shape '(batch, n, d2)`
    return:
    """
    bsz, n, d1 = x.size()
    # (bsz, n, d1, 1) x (bsz, n, 1, d2) -> (bsz, n, d1, d2)
    sum_mat = torch.matmul(y.unsqueeze(3), z.unsqueeze(2))
    accum_mat = torch.cumsum(sum_mat, dim=1)
    # (bsz, n, 1, d1) x (bsz, n, d1, d2) -> (bsz, n, 1, d2) -> (bsz, n, d2)
    res = torch.matmul(x.unsqueeze(2), accum_mat).squeeze(2)
    # (1, n, 1)
    length_div = torch.arange(1, n + 1, device=x.device).unsqueeze(0).unsqueeze(2)
    res = res / length_div
    return res


class LunaDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, index):
        super().__init__()
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.index = index
        self.normalize_before = args.decoder_normalize_before
        self.embed_dim = args.decoder_embed_dim

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)

        self.self_attn = self.build_self_attention(self.embed_dim, args)

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.encoder_atten_proj_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.activation_fn = utils.get_activation_fn(activation=getattr(args, "activation_fn", "relu"))
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(float(activation_dropout_p), module_name=self.__class__.__name__)

        self.fc1 = self.build_fc1(self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.fc2 = self.build_fc2(args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True
        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return LunarCausalAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            tie_kv=not args.untie_luna_kv,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return LunarMultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            args.decoder_projected_attention_heads,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            tie_kv=not args.untie_luna_kv,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        px,
        encoder_out,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        encoder_projected_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            px (Tensor): projected input to the layer of shape `(proj_len, batch, embed_dim)`
            encoder_out (Tensor): output from encoder of shape `(encoder_seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            encoder_projected_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, proj_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
            projected output of shape `(proj_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        static_px = px is None

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn = self.self_attn(query=x, pquery=px,
                                 key_padding_mask=self_attn_padding_mask,
                                 pkey_padding_mask=encoder_projected_padding_mask,
                                 incremental_state=incremental_state,
                                 need_weights=False)

        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        presidual = px
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
            px = self.encoder_atten_proj_layer_norm(px) if not static_px else None

        x, px, attn = self.encoder_attn(query=x, pquery=px, context=encoder_out,
                                        context_padding_mask=encoder_padding_mask,
                                        pcontext_padding_mask=encoder_projected_padding_mask,
                                        incremental_state=incremental_state,
                                        static_context=True,
                                        need_weights=need_attn or (not self.training and self.need_attn),
                                        need_head_weights=need_head_weights)
        # apply dropout
        x = self.dropout_module(x)
        px = self.dropout_module(px) if not static_px else None

        x = residual + x
        px = presidual + px if not static_px else None
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
            px = self.encoder_atten_proj_layer_norm(px) if not static_px else None

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, px, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn