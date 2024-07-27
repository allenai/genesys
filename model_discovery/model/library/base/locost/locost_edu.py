import copy
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.utils import DUMMY_INPUTS, DUMMY_MASK, is_torch_fx_proxy, logging

from longt5_models import (
    LongT5LayerCrossAttention,
    LongT5LayerFF,
    LongT5LayerNorm,
    LongT5LayerSelfAttention,
)
from models_config import LOCOSTConfig
from s4d_models import S4D

logger = logging.get_logger(__name__)


def clamp_inf(hidden_states):
    if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)


class LOCOSTLayerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model

        self.ssm_layer = S4D(config)

        self.layer_prenorm = LongT5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

        self.config = config

    def forward(
        self,
        hidden_states,
        original_attention_mask=None,
        output_attentions=False,
        **kwargs: Any,  # to accept past_key_value and use_cache kwargs
    ):
        normed_hidden_states = self.layer_prenorm(hidden_states)

        ssm_outputs = self.ssm_layer(
            normed_hidden_states,
            attention_mask=original_attention_mask,
            output_attentions=output_attentions,
        )
        ssm_hidden_states = ssm_outputs[0]
        kernel_outputs = ssm_outputs[1:]

        hidden_states = hidden_states + F.dropout(
            ssm_hidden_states, p=self.config.dropout_rate, training=self.training
        )

        outputs = (hidden_states, None, None, None) + kernel_outputs
        return outputs  # hidden_states, present_key_value, (self-attention position bias), (self-attention weights), kernel


class LOCOSTBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder

        self.layer = nn.ModuleList()
        if self.is_decoder:
            self_attn = LongT5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        else:

            self_attn = LOCOSTLayerSelfAttention(config)

        self.layer.append(self_attn)
        if self.is_decoder:
            cross_attn_layer = LongT5LayerCrossAttention(config)
            self.layer.append(cross_attn_layer)
        ff_layer = LongT5LayerFF(config)
        self.layer.append(ff_layer)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        original_attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        use_cache=False,
        output_attentions=False,
        past_key_value=None,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning(
                    "`past_key_values` is passed to the encoder. Please make sure this is intended."
                )
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attn_output = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            original_attention_mask=original_attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states, present_key_value_state = self_attn_output[:2]
        attention_outputs = self_attn_output[2:]
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None

        clamp_inf(hidden_states)
        if do_cross_attention:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]
            clamp_inf(hidden_states)
            if present_key_value_state is not None:
                present_key_value_state = (
                    present_key_value_state + cross_attention_outputs[1]
                )

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # FF
        hidden_states = self.layer[-1](hidden_states)

        clamp_inf(hidden_states)
        outputs = (hidden_states, present_key_value_state) + attention_outputs
        # hidde_states, present_key_value, (self-attention position bias), (self-attention weights), kernel, (cross-attention position bias), (cross-attention weights)
        return outputs

