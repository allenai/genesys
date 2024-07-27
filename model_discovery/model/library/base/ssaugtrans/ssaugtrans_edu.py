import json
import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, List, Optional

from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from .local_attention import LocalAttention
from .s4 import S4Module


class SpadeDecoderLayerBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        layer_idx,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
        has_relative_attention_bias=False,
    ):
        super().__init__()
        assert not cfg.cross_self_attention and no_encoder_attn

        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            has_relative_attention_bias=has_relative_attention_bias,
        )
        self.attn_ln = (
            LayerNorm(self.embed_dim)
            if utils.safe_getattr(cfg, "scale_attn", False)
            else None
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim
        scale_heads = utils.safe_getattr(cfg, "scale_heads", False)
        self.c_attn = (
            nn.Parameter(torch.ones((self.nh,)), requires_grad=True)
            if scale_heads
            else None
        )

        if layer_idx % cfg.s4_every_n_layers == 0:
            assert self.c_attn is None  # handled via self.norm_local
            if isinstance(cfg.s4_lr, str):
                cfg.s4_lr = json.loads(cfg.s4_lr)
            self.s4_module = S4Module(cfg, is_decoder=True)
            self.s4_local_combine = cfg.s4_local_combine
            if self.s4_local_combine == "concat":
                self.aggregate = quant_noise(
                    nn.Linear(2 * self.embed_dim, self.embed_dim),
                    self.quant_noise, self.quant_noise_block_size
                )
                self.norm_global = LayerNorm(self.embed_dim, export=cfg.export)
                self.norm_local = LayerNorm(self.embed_dim, export=cfg.export)
            elif self.s4_local_combine == "add":
                self.s4_weight = cfg.s4_weight
                self.norm_global = LayerNorm(self.embed_dim, export=cfg.export)
                self.norm_local = LayerNorm(self.embed_dim, export=cfg.export)
            elif self.s4_local_combine == "stack":
                pass
            else:
                raise KeyError("concat/add/stack S4 and local attention")
        else:
            self.s4_module = None

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.decoder.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg, has_relative_attention_bias)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.ffn_layernorm = (
            LayerNorm(cfg.decoder.ffn_embed_dim)
            if utils.safe_getattr(cfg, "scale_fc", False)
            else None
        )
        self.w_resid = (
            nn.Parameter(
                torch.ones(
                    self.embed_dim,
                ),
                requires_grad=True,
            )
            if utils.safe_getattr(cfg, "scale_resids", False)
            else None
        )

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False, has_relative_attention_bias=False,
    ):
        kwargs = {
            "local_radius": cfg.local_radius,
            "has_relative_attention_bias": has_relative_attention_bias,
            "relative_attention_num_buckets": cfg.relative_attention_num_buckets,
            "relative_attention_max_distance": cfg.relative_attention_max_distance,
        }
        return LocalAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            is_decoder=True,
            dropout=cfg.attention_dropout,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            kwargs=kwargs,
        )

    def build_encoder_attention(self, embed_dim, cfg, has_relative_attention_bias=False):
        kwargs = {
            "local_radius": cfg.local_radius,
            "has_relative_attention_bias": has_relative_attention_bias,
            "relative_attention_num_buckets": cfg.relative_attention_num_buckets,
            "relative_attention_max_distance": cfg.relative_attention_max_distance,
        }
        return LocalAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            is_decoder=True,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            kwargs=kwargs,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        local_attn_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        assert encoder_out is None and encoder_padding_mask is None \
               and incremental_state is None and prev_self_attn_state is None \
               and prev_attn_state is None, "local attention only supports language modeling"

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        """compute global information using S4"""
        s4_x = None
        if self.s4_module is not None:
            s4_x = self.s4_module(x, encoder_padding_mask)
            if self.s4_local_combine == "stack":
                x = s4_x  # directly feed s4_x to local attention

        """compute local attention"""
        local_x, _, position_bias = self.self_attn(
            hidden_states=x,
            local_mask=local_attn_mask,
            position_bias=position_bias,
        )
        if self.c_attn is not None:
            tgt_len, bsz = local_x.size(0), local_x.size(1)
            local_x = local_x.view(tgt_len, bsz, self.nh, self.head_dim)
            local_x = torch.einsum("tbhd,h->tbhd", local_x, self.c_attn)
            local_x = local_x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            local_x = self.attn_ln(local_x)

        """aggregate local and global information"""
        if self.s4_module is not None:
            if self.s4_local_combine == "stack":
                x = local_x
            else:
                local_x = self.norm_local(local_x)
                s4_x = self.norm_global(s4_x)
                if self.s4_local_combine == "concat":
                    x = torch.concat((s4_x, local_x), dim=-1)
                    x = self.aggregate(x)
                elif self.s4_local_combine == "add":
                    x = self.s4_weight * s4_x + (1.0 - self.s4_weight) * local_x
        else:
            x = local_x

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, None, None, position_bias