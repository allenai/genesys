import logging
import math
import warnings
from functools import partialmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.models.llama import modeling_llama

import utils
from args.model_args import FreezeType, ModelArguments, SoftMaxScaleType

logger = logging.getLogger(__name__)

origin_apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
OriginLlamaAttention = modeling_llama.LlamaAttention
OriginLlamaFlashAttention2 = modeling_llama.LlamaFlashAttention2


def monkey_patch_before(args: ModelArguments):
    utils.log_once(logger, f"monkey patch before {args}")
    if args.nope:
        modeling_llama.apply_rotary_pos_emb = nope_monkey_patch
    else:
        # logger.warning("Skipping monkey patch for RoPE model")
        # return
        pass
    if args.use_flash_attention:
        modeling_llama.LlamaAttention.forward = utils.forbidden_func  # must use flash attention
        LlamaFlashAttention2MonkeyPatch.__init__ = partialmethod(
            LlamaFlashAttention2MonkeyPatch.__init__,
            softmax_scale=args.softmax_scale,
            softmax_scale_type=args.softmax_scale_type,
        )  # type: ignore
        modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2MonkeyPatch
    else:  # output_attentions=True and entropy=True
        modeling_llama.LlamaFlashAttention2.forward = utils.forbidden_func
        LlamaAttentionMonkeyPatch.__init__ = partialmethod(
            LlamaAttentionMonkeyPatch.__init__,
            softmax_scale=args.softmax_scale,
            softmax_scale_type=args.softmax_scale_type,
        )  # type: ignore
        modeling_llama.LlamaAttention = LlamaAttentionMonkeyPatch


def monkey_patch_after(model: PreTrainedModel, args: ModelArguments):
    if args.yarn is not None:
        assert isinstance(model, modeling_llama.LlamaForCausalLM)
        from .LlamaYaRNScaledRotaryEmbedding import (
            patch_llama_for_yarn_scaled_rotary_embeddings,
        )

        patch_llama_for_yarn_scaled_rotary_embeddings(model, args.yarn, 2048)


def prepare_for_training(model: PreTrainedModel, args: ModelArguments):
    assert isinstance(model, modeling_llama.LlamaForCausalLM)
    assert isinstance(model.config, modeling_llama.LlamaConfig)

    if not args.yarn:
        # yarn patches rope, cannot assert yarn
        _assert_rope_type(type(model.model.layers[0].self_attn.rotary_emb), args.scale_type)  # type: ignore
    if args.freeze_type == FreezeType.BASE:
        if hasattr(model.model.layers[0].self_attn, "scale_param"):
            # freeze model except scale_param
            model.requires_grad_(False)  # freeze everything
            for i in range(model.config.num_hidden_layers):
                model.model.layers[i].self_attn.scale_param.requires_grad_(True)  # type: ignore


def _assert_rope_type(rope_type: type, scale_type: Optional[str]):
    logger.info(f"RoPE module: {rope_type}")  # check rotary type
    if scale_type is None:
        assert rope_type is modeling_llama.LlamaRotaryEmbedding
    elif scale_type == "linear":
        assert rope_type is modeling_llama.LlamaLinearScalingRotaryEmbedding
    elif scale_type == "dynamic":
        assert rope_type is modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding
    else:
        assert False, f"unknown scale type {scale_type}"


def nope_monkey_patch(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    utils.log_once(logger, f"monkey patching rope to nope")
    return q, k


class ScaleMixin(OriginLlamaAttention):
    def __init__(
        self,
        config,
        softmax_scale: float,
        softmax_scale_b: float,
        softmax_scale_type: SoftMaxScaleType,
        QNA: bool,
        KNA: bool,
        window_attn: Optional[int],
    ):
        super().__init__(config)
        self.softmax_scale = softmax_scale
        self.softmax_scale_type = softmax_scale_type
        self.QNA = QNA
        self.KNA = KNA
        self.window_attn = window_attn
        if self.softmax_scale_type == SoftMaxScaleType.HS:
            init_value = softmax_scale
            self.scale_param = torch.nn.Parameter(torch.full((self.num_heads,), init_value, dtype=torch.bfloat16))

    def query_scale(
        self,
        query_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ):
        utils.log_once(
            logger,
            f"monkey patching llama attention: {self.softmax_scale=} {self.softmax_scale_type=}",
        )
        bsz, num_heads, q_len, head_dim = query_states.shape
        assert num_heads == self.num_heads and head_dim == self.head_dim
        assert position_ids is not None
        assert position_ids.shape == (bsz, q_len) or position_ids.shape == (1, q_len), position_ids.shape
        # scale_factor = scale / math.sqrt(head_dim)
        # return query_states * scale_factor.unsqueeze(-1).to(query_states.dtype).to(query_states.device)
        if self.softmax_scale_type == SoftMaxScaleType.CONST:
            scale_factor = self.softmax_scale / math.sqrt(head_dim)
        elif self.softmax_scale_type == SoftMaxScaleType.HS:
            assert head_dim == 64
            origin_scale = 1.0 / math.sqrt(head_dim)
            scale_factor = self.scale_param * origin_scale
            scale_factor = scale_factor.unsqueeze(-1).unsqueeze(-1).to(query_states.dtype)
        else:
            assert False, f"unknown softmax scale type {self.softmax_scale_type}"
        query_states = query_states * scale_factor
        assert query_states.shape == (bsz, num_heads, q_len, head_dim), query_states.shape
        return query_states


class LlamaAttentionMonkeyPatch(ScaleMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        utils.log_once(logger, "Using normal attention")
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)  # type: ignore

        past_key_value = (key_states, value_states) if use_cache else None  # type: ignore

        ##########
        query_states = self.query_scale(query_states, position_ids)
        ##########

        key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
        value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))  # scaled in query scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        ##########
        attn_weights = _calc_entropy(attn_weights)  # overwrite to entropy
        ##########

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value  # type: ignore


class LlamaFlashAttention2MonkeyPatch(OriginLlamaFlashAttention2, ScaleMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        utils.log_once(logger, "Using flash attention")
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)  # type: ignore

        past_key_value = (key_states, value_states) if use_cache else None  # type: ignore

        ##########
        query_states = self.query_scale(query_states, position_ids)
        ##########

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(  # type: ignore
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            ##########
            softmax_scale=1,  # scaled in query scale
            ##########
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


@torch.no_grad()
def _calc_entropy(attn: torch.Tensor):
    # tensor: [bs, n_heads, seq_len, seq_len]
    # assert torch.sum(attn, dim=-1, keepdim=True).allclose(torch.ones_like(attn, dtype=torch.float32), atol=5e-4)
    info = attn * torch.log2(attn)
    info = torch.where(torch.isnan(info), torch.zeros_like(info), info)  # nan comes from 0 * log(0), which should be 0
    entropy = -torch.sum(info, dim=-1)
    return entropy  # [bs, n_heads, seq_len]