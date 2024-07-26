# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Pytorch DenseGAU RetNet model."""
from typing import List, Optional, Tuple, Union
import math
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers.models.llama.configuration_llama import LlamaConfig

from llama.utils import _make_causal_mask, LlamaRMSNorm, LLAMA_START_DOCSTRING, LLAMA_INPUTS_DOCSTRING
from modules import LlamaPreTrainedModel, LlamaDecoderLayer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

# added for retention
# Copied from https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiscale_retention.py
def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\
def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


#Parameter efficient HiddenProjection
class HiddenProjection(nn.Module):
    def __init__(self, input_dim, mid_reduction_ratio=16, final_reduction_ratio=4):
        super(HiddenProjection, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // mid_reduction_ratio, bias=False)
        self.fc2 = nn.Linear(input_dim // mid_reduction_ratio, int(input_dim // final_reduction_ratio), bias=False)

    def forward(self, x):
        fc1_output = F.silu(self.fc1(x))
        fc2_output = self.fc2(fc1_output)
        return fc2_output

# Copied and modified from transformers.models.bart.modeling_bart._expand_mask

class MultiScaleGauRetention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.query_key_dim = config.query_key_dim
        self.num_heads = config.num_attention_heads
        self.factor = config.v_factor
        self.head_dim = config.hidden_size * self.factor // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.q_proj = nn.Linear(self.hidden_size, self.query_key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.query_key_dim, bias=False)
        self.key_dim = self.query_key_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.expansion_dim = int(config.hidden_size * self.factor)
        self.group_norm = LlamaRMSNorm(self.expansion_dim // config.num_attention_heads, eps=config.rms_norm_eps)
        self.to_hidden = nn.Sequential(
            nn.Linear(config.hidden_size, self.expansion_dim * 2, bias=False),
            nn.SiLU()
        )
        self.to_out = nn.Sequential(
            nn.Linear(self.expansion_dim, config.hidden_size, bias=False),
            nn.Dropout(0)
        )
        self.config = config
        self.k_select = HiddenProjection(self.hidden_size, 32, 2)
        self.v_select = HiddenProjection(self.hidden_size, 32, 0.5)
        self.k_norm = LlamaRMSNorm(self.query_key_dim, eps=config.rms_norm_eps)
        self.v_norm = LlamaRMSNorm(self.expansion_dim, eps=config.rms_norm_eps)
        if config.deepnorm:
            self.alpha = math.pow(2.0 * config.num_hidden_layers, 0.25)
        else:
            self.alpha = 1.0
        self.dropout_module = torch.nn.Dropout(config.dropout)
        self.reset_parameters()

    #
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_select.fc1.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_select.fc2.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_select.fc1.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_select.fc2.weight, gain=2 ** -2.5)
        for module in self.to_out.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=2 ** -1)
        for module in self.to_hidden.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)

    def forward(
            self,
            forward_impl: 'parallel',
            hidden_states: torch.Tensor,
            rel_pos,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            k_features=None,  # dense
            v_features=None,  # dense
            dense=False,
            dense_layers=0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, tgt_len, _ = hidden_states.size()
        (sin, cos), inner_mask = rel_pos
        x = hidden_states
        q = F.silu(self.q_proj(x))
        k = F.silu(self.k_proj(x))
        v, gate = self.to_hidden(x).chunk(2, dim=-1)

        k *= self.scaling
        k_curr = k
        v_curr = v

        if dense:
            k_gate = self.k_select(hidden_states.clone())
            for i, k_past in enumerate(k_features):
                k = k.clone() + F.silu(k_gate) * k_past
            k = self.k_norm(k)

            v_gate = self.v_select(hidden_states.clone())
            for i, v_past in enumerate(v_features):
                v = v.clone() + F.silu(v_gate) * v_past
            v = self.v_norm(v)

        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if forward_impl == 'parallel':
            output = self.parallel_forward(qr, kr, v, inner_mask)
        elif forward_impl == 'recurrent':
            output, past_key_value = self.recurrent_forward(qr, kr, v, inner_mask, past_key_value=past_key_value)

        output = self.group_norm(output)
        output = output.reshape(bsz, tgt_len, self.expansion_dim) * gate  # gate
        output = self.to_out(output)
        output = self.dropout_module(output)

        return output, past_key_value, k_curr, v_curr

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # retntion parallel forward
    def recurrent_forward(
            self,
            qr, kr, v,
            decay,
            past_key_value,
    ):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in past_key_value:
            prev_kv = past_key_value["prev_key_value"]
            prev_scale = past_key_value["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(self.num_heads, 1,
                                                                           1) + kv / scale.sqrt().view(self.num_heads,
                                                                                                       1, 1)
        else:
            scale = torch.ones_like(decay)

        past_key_value["prev_key_value"] = kv
        past_key_value["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output, past_key_value

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.expansion_dim // self.num_heads).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2)  # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)

        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output


class RetNetRelPos(nn.Module):
    def __init__(self, decoder_embed_dim, decoder_retention_heads, query_key_dim):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, (query_key_dim // decoder_retention_heads) // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(decoder_retention_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def forward(self, slen, activate_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class DenseGAURetNetModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.retnet_rel_pos = RetNetRelPos(config.hidden_size, config.num_attention_heads,
                                           config.query_key_dim)

        if config.deepnorm:
            init_scale = math.pow(8.0 * config.num_hidden_layers, 0.25)
            for name, p in self.named_parameters():

                if (
                        "fc1" in name
                        or "fc2" in name
                        or "gate_proj" in name
                        or "down_proj" in name
                        or "up_proj" in name
                        or "out_proj" in name
                        or "v_proj" in name
                        or "to_hidden" in name
                        or "to_output" in name

                ):
                    p.data.div_(init_scale)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            forward_impl: Optional[str] = 'parallel',
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            sequence_offset=0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layer
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = [] if use_cache else None

        #
        k_features = []
        v_features = []
        dense_layers = 0
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None and len(
                past_key_values) != 0 else {}

            slen = input_ids.size(1)
            if forward_impl == 'recurrent':
                slen = sequence_offset
            rel_pos = self.retnet_rel_pos(slen, forward_impl == 'recurrent',)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                hidden_states = layer_outputslayer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                dense = False
                if idx >= 1:
                    dense = True

                layer_outputs, past_key_value, k_curr, v_curr = decoder_layer(
                    hidden_states,
                    rel_pos,
                    forward_impl=forward_impl,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    k_features=k_features,
                    v_features=v_features,
                    dense=dense,
                    dense_layers=dense_layers,
                )
                dense_layers += 1
                k_features.append(k_curr)
                v_features.append(v_curr)
                if len(k_features) > self.config.dense_block_layers:
                    k_features.pop(0)
                if len(v_features) > self.config.dense_block_layers:
                    v_features.pop(0)

            hidden_states = layer_outputs  # used to be 3 ele,tmp  1


            if use_cache:
                next_decoder_cache.append(past_key_value)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



