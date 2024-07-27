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
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from .configuration_llama import LlamaConfig
from .modules import LlamaRotaryEmbedding, apply_rotary_pos_emb


logger = logging.get_logger(__name__)



"""
Assume that we have batch inputs that have the dimension [bsz, seq_len], what we want to do is:
1. We do not change the attention weights of the first and the last two layers, because:
    The first layer's attention is irregular
    The last two layers' attention variance is much bigger than the previous ones
2. Decide a chunk of tokens for each layer that apply the attention transition
    Sequentially eliminate and add attention weights on different regions
    Within one layer, we just focus on a specific region and ignore other parts
    With layer count increasing, the attention should be dispensed to positions more closer to the end
3. Decide a number for the attention lowest boundary to eliminate
    Do not change the original distribution so much
    Do not affect the attention weights that are big enough
    We want to eliminate 10%-20% of the whole attention, which will not change the original distribution so much
"""
def attention_handler(seq_len, one_layer_inter_use=2):
    """
    Args:
    seq_len: the input length
    one_layer_inter_use: dicide how many intervals will be strengthened within one layer
    return:
        layer_indices: the sequence of layers to implement attention transition
        start_end_tuple_lists: a list of start_end_tuple lists
    """
    if seq_len < 400:
        return 


    # We divide our seq_len into 29+2+1+one_layer_inter_use-1 intervals and only strengthen the middle intervals, except for 0,1,-1
    total_interval = 31 + one_layer_inter_use
    interval = int(seq_len / total_interval) - int(seq_len / total_interval) % 5
    layer_indices = []
    start_end_tuple_lists = []
    # We use total 29 layers, except for 0,-2,-1

    for i in range(29):

        layer_indices.append(i+1)
        start_end_tuple_list = []
        for j in range(one_layer_inter_use):
            start = (i+j+2) * interval
            end = (i+j+3) * interval
            tup = (start, end)
            start_end_tuple_list.append(tup)
        start_end_tuple_lists.append(start_end_tuple_list)

    return layer_indices, start_end_tuple_lists, interval

"""
Attention Dispension is aimed to eliminate attention weights lower than some number and add them on a specific region's attention
Things to emphasis:
1. Attention eliminated shold not affect the original layer output so much
2. Attntion dispensation is functioned only on a specific region, while elimination functions on all attention exists
3. When dispense attention weights, we just dispense attention weights on previous region
"""

def attention_dispense(attn_weights, start_end_tuple, alpha=0.5, beta=0.1, strategy='adding'):
    """
    args:
    attn_weights: attention weights to be implemented, which has the size (bsz, num_heads, seq_len, seq_len)
        Notice that for attention weights, attentions of which positions further than the current one are 0
    start_end_tuple: A tuple consists start position and end postion to implement
    """
    start, end = start_end_tuple
    start = int(start)
    end = int(end)
    interval = end - start
    # Use attn_mask to eliminate attentions
    lower_boundary = alpha / end

    with torch.no_grad():
        # Attention Elimination functions on all attentions
        attn_rebuild_mask = torch.where(attn_weights[:,:,start:end,:end]>=lower_boundary, 1, 0)
        attn_weights[:,:,start:end,:end] = attn_weights[:,:,start:end,:end] * attn_rebuild_mask
        attn_weights[:,:,start:end,0] = attn_weights[:,:,start:end,0] / 2
        
        # bsz * head_dim * seq_len
        weights_sum = torch.sum(attn_weights[:,:,start:end,:end], dim=-1, dtype=torch.float32)
        attn_weights_eliminaed = torch.ones_like(weights_sum) - weights_sum

        #torch.save(attn_weights_eliminaed, '/root/autodl-fs/eliminated_weights')
            
        # Distribute the attention weights by adding or by multiplying a coefficient.
        
        # Distribute by adding
        if strategy == 'adding':
        # Attention dispension is sent to the previous attention interval, not all attentions
            if start != interval:
                # 0.3:0.7
                # Because some inervals have no weight bigger than lower_boundary, we use 0.001 to avoid 0
                prev_mask = torch.where(attn_weights[:,:, start:end, start-2*interval:end-interval]>=lower_boundary, 1, 0.01)
                #print('1',prev_mask.size())
                nonzero_num = torch.sum(prev_mask, dim=-1, dtype=torch.float32)
                average_weights = torch.div(attn_weights_eliminaed,nonzero_num) * beta
                add_matrix = average_weights[...,None]
                # 0.3:0.7
                attn_weights[:,:,start:end,start-interval:end-interval] = attn_weights[:,:,start:end,start-interval:end-interval] + 7 * add_matrix * prev_mask[:,:,:,-interval:]
                attn_weights[:,:,start:end,start-2*interval:end-2*interval] = attn_weights[:,:,start:end,start-2*interval:end-2*interval] + 3 * add_matrix * prev_mask[:,:,:,:interval]
            else:
                prev_mask = torch.where(attn_weights[:,:, start:end, :end-interval]>=lower_boundary, 1, 0.001)
                #print('2',prev_mask.size())
                nonzero_num = torch.sum(prev_mask, dim=-1, dtype=torch.float32)
                average_weights = torch.div(attn_weights_eliminaed,nonzero_num) * beta
                #average_weights = torch.div(attn_weights_eliminaed,nonzero_num)
                add_matrix = average_weights[...,None]
                attn_weights[:,:,start:end,start-interval:end-interval] = attn_weights[:,:,start:end,start-interval:end-interval] + 10 * add_matrix * prev_mask[:,:,:,-interval:]
        if strategy == 'multi':
            # Distribute by multiplying
            prev_attn_sum = torch.sum(attn_weights[:,:, start:end, :end], dim=-1)
            multi_coeff = attn_weights_eliminaed / prev_attn_sum
            attn_weights[:,:,start:end, :end] = attn_weights[:,:,start:end,:end] * multi_coeff[...,None]
            
    return attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_end_tuple_list: Optional[Tuple] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_transition: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        #hidden_states.to(self.q_proj.weight.device)
        #print(self.q_proj.weight.device, hidden_states.device)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids.to(query_states.device))
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

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
            attn_weights = attn_weights + attention_mask.to(attn_weights.device)
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        
        #########################################################################################
        # Use attention transition in attention layer
        #########################################################################################
        
        if attn_transition and start_end_tuple_list is not None:
            for start_end_tuple in start_end_tuple_list:
                attn_weights = attention_dispense(attn_weights=attn_weights, start_end_tuple=start_end_tuple)
        
        attn_output = torch.matmul(attn_weights.to(query_states.dtype), value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

