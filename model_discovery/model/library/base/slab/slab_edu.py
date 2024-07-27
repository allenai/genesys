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
""" PyTorch LLaMA model."""
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig
from functools import partial
from llama import LlamaAttention, LlamaMLP, LlamaRMSNorm


logger = logging.get_logger(__name__)



class RepBN(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm1d(channels, eps=eps, momentum=0.1)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.ones_(self.weight)
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        # construct the mask_input, size to be (BxL) x C: L is the real length here
        if pad_mask is None or not self.training:
            mask_input = x.contiguous().view(-1, C)
        else:
            bn_mask = (pad_mask == 1)
            mask_input = x[bn_mask, :]
            mask_input = mask_input.contiguous().view(-1, C)

        o_bn = self.bn(mask_input)

        if pad_mask is None or not self.training:
            output = o_bn.view(B, T, C)
        else:
            output = x.clone()
            output[bn_mask, :] = o_bn

        x = output + self.alpha * x
        return x


class LinearNorm(nn.Module):
    def __init__(self, dim, norm1, norm2, eps=1e-5, warm=10000, step=18000, r0=1.0):
        super(LinearNorm, self).__init__()
        self.register_buffer('num_step', torch.tensor(0))
        self.register_buffer('warm', torch.tensor(warm))
        self.register_buffer('iter', torch.tensor(step))
        self.register_buffer('total_step', torch.tensor(step))
        self.r0 = r0
        self.norm1 = norm1(dim, eps)
        self.norm2 = norm2(dim, eps)

    def forward(self, x, pad_mask=None):
        if self.training:
            if self.warm > 0:
                if self.num_step % 16 == 0:
                    self.warm.copy_(self.warm - 1)
                x = self.norm1(x)
            else:
                lamda = self.r0 * self.iter / self.total_step
                if self.iter > 0:
                    if self.num_step % 16 == 0:
                        self.iter.copy_(self.iter - 1)
                x1 = self.norm1(x)
                x2 = self.norm2(x, pad_mask)
                x = lamda * x1 + (1 - lamda) * x2
            self.num_step.copy_((self.num_step + 1) % 16)
        else:
            x = self.norm2(x, pad_mask)
        return x


linearnorm = partial(LinearNorm, norm1=LlamaRMSNorm, norm2=RepBN)
# linearnorm = LlamaRMSNorm
# linearnorm = RepBN

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size=config.intermediate_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = linearnorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = linearnorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.input_layernorm = nn.Identity()
        # self.post_attention_layernorm = nn.Identity()

    def merge_bn(self):
        # Attention
        miu = self.input_layernorm.norm2.bn.running_mean
        sigma2 = self.input_layernorm.norm2.bn.running_var
        gamma = self.input_layernorm.norm2.bn.weight
        beta = self.input_layernorm.norm2.bn.bias
        eps = self.input_layernorm.norm2.bn.eps
        alpha = self.input_layernorm.norm2.alpha

        w_q = self.self_attn.q_proj.weight.data.transpose(0, 1)
        w_k = self.self_attn.k_proj.weight.data.transpose(0, 1)
        w_v = self.self_attn.v_proj.weight.data.transpose(0, 1)

        self.self_attn.q_proj = nn.Linear(self.self_attn.hidden_size, self.self_attn.num_heads * self.self_attn.head_dim, bias=True).to(w_q.device)
        self.self_attn.k_proj = nn.Linear(self.self_attn.hidden_size, self.self_attn.num_heads * self.self_attn.head_dim, bias=True).to(w_q.device)
        self.self_attn.v_proj = nn.Linear(self.self_attn.hidden_size, self.self_attn.num_heads * self.self_attn.head_dim, bias=True).to(w_q.device)

        a = gamma / torch.sqrt(sigma2 + eps) + alpha
        b = beta - gamma * miu / torch.sqrt(sigma2 + eps)
        a = torch.diag(a)

        w_q_n = (a @ w_q).transpose(0, 1)
        b_q_n = (b.unsqueeze(0) @ w_q).squeeze(0)
        self.self_attn.q_proj.weight.data.copy_(w_q_n)
        self.self_attn.q_proj.bias.data.copy_(b_q_n)
        w_k_n = (a @ w_k).transpose(0, 1)
        b_k_n = (b.unsqueeze(0) @ w_k).squeeze(0)
        self.self_attn.k_proj.weight.data.copy_(w_k_n)
        self.self_attn.k_proj.bias.data.copy_(b_k_n)
        w_v_n = (a @ w_v).transpose(0, 1)
        b_v_n = (b.unsqueeze(0) @ w_v).squeeze(0)
        self.self_attn.v_proj.weight.data.copy_(w_v_n)
        self.self_attn.v_proj.bias.data.copy_(b_v_n)
        self.input_layernorm = nn.Identity()

        # mlp
        miu = self.post_attention_layernorm.norm2.bn.running_mean
        sigma2 = self.post_attention_layernorm.norm2.bn.running_var
        gamma = self.post_attention_layernorm.norm2.bn.weight
        beta = self.post_attention_layernorm.norm2.bn.bias
        eps = self.post_attention_layernorm.norm2.bn.eps
        alpha = self.post_attention_layernorm.norm2.alpha

        w_g = self.mlp.gate_proj.weight.data.transpose(0, 1)
        w_u = self.mlp.up_proj.weight.data.transpose(0, 1)

        self.mlp.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.mlp.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)

        a = gamma / torch.sqrt(sigma2 + eps) + alpha
        b = beta - gamma * miu / torch.sqrt(sigma2 + eps)
        a = torch.diag(a)

        w_g_n = (a @ w_g).transpose(0, 1)
        b_g_n = (b.unsqueeze(0) @ w_g).squeeze(0)
        self.mlp.gate_proj.weight.data.copy_(w_g_n)
        self.mlp.gate_proj.bias.data.copy_(b_g_n)
        w_u_n = (a @ w_u).transpose(0, 1)
        b_u_n = (b.unsqueeze(0) @ w_u).squeeze(0)
        self.mlp.up_proj.weight.data.copy_(w_u_n)
        self.mlp.up_proj.bias.data.copy_(b_u_n)
        self.post_attention_layernorm = nn.Identity()
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # hidden_states = self.input_layernorm(hidden_states, pad_mask)
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states, pad_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        # breakpoint()
        return outputs

