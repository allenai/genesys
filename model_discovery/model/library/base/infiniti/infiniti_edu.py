# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
""" PyTorch Llama model, with Infini-Attention."""

import os
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    logging,
)
from transformers.utils.import_utils import is_torch_fx_available

from transformers import LlamaConfig

from .rotary_embedding import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

DEBUG = os.environ.get("DEBUG", False)


def debug_print(*args):
    if DEBUG:
        print(*args)

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)



# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        raise NotImplementedError("forward method must be implemented in derived classes")
    


class LlamaInfiniAttention(LlamaAttention):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__(config, layer_idx)

        # Each head has its own gate
        # init with -100 to make it close to 0 effect at the beginning
        self.gate = nn.Parameter(torch.full((1, self.num_heads, 1, 1), 0.0))
        # self.segment_size = config.segment_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        memory: Optional[dict] = None,
        norm_term: Optional[dict] = None,
        no_memory_update: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        segment = hidden_states  # no need to split in TYPE-2 implementation

        # Pre-allocate tensor for all outputs
        bsz, _, hidden_dim = hidden_states.size()

        query_states = self.q_proj(segment)
        key_states = self.k_proj(segment)
        value_states = self.v_proj(segment)

        # Assuming the presence of batch size and dimension handling as before
        bsz, q_len, _ = segment.size()  # q_len == self.segment_size
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        debug_print("Query States Shape:", query_states.shape)
        debug_print("Key States Shape:", key_states.shape)
        debug_print("Value States Shape:", value_states.shape)

        # memory and norm_term should use layer_idx to store the memory and norm_term
        if no_memory_update:
            memory = {}
            norm_term = {}
            memory_output = None
        else:
            # Infini Attention memory does not use PE
            # Memory retrieval and attention calculation per segment
            memory_output = self._retrieve_from_memory(
                query_states,
                memory.get(self.layer_idx, None) if memory is not None else None,
                norm_term.get(self.layer_idx, None) if norm_term is not None else None,
            )
            debug_print("Memory Output Shape:", memory_output.shape)

        # Update memory with current segment's key and value states
        if no_memory_update:
            # do not update memory
            pass
        else:
            updated_memory, updated_norm_term = self._update_memory(
                key_states,
                value_states,
                memory.get(self.layer_idx, None) if memory is not None else None,
                norm_term.get(self.layer_idx, None) if norm_term is not None else None,
            )
            debug_print("Memory Output Shape:", updated_memory.shape)
            debug_print("Updated Memory Shape:", updated_norm_term.shape)
            if memory is None and norm_term is None:
                memory = {}
                norm_term = {}
            memory[self.layer_idx] = updated_memory.detach()
            norm_term[self.layer_idx] = updated_norm_term.detach()

        # Rotary embeddings, set seq_len to q_len as we are processing a segment
        cos, sin = self.rotary_emb(value_states, position_ids)

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,  # cos[:, : min(self.segment_size, q_len), :],
            sin,  # sin[:, : min(self.segment_size, q_len), :],
            None,
        )

        # Basic cache
        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            # causal_mask = causal_mask[
            #     :, :, : min(self.segment_size, q_len), : key_states.shape[-2]
            # ]  # FIXME: This is wrong, should be [:, :, :, :self.segment_size]
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        debug_print("causal_mask.shape", causal_mask.shape)
        debug_print("query_states.shape", query_states.shape)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        if memory_output is None:
            combined_output = attn_output
        else:
            combined_output = (
                F.sigmoid(self.gate) * memory_output
                + (1 - F.sigmoid(self.gate)) * attn_output
            )

        # Prepare output for this segment
        combined_output = combined_output.transpose(1, 2).contiguous()
        combined_output = combined_output.view(bsz, q_len, self.hidden_size)

        final_output = self.o_proj(combined_output)

        if no_memory_update:
            memory = None
            norm_term = None

        return (
            final_output,
            None,
            None,
            memory,
            norm_term,
        )

    def _retrieve_from_memory(self, query_states, memory, norm_term):
        # query_states: [batch_size, num_heads, seq_len, head_dim]

        # Check if memory is initialized
        if memory is None or norm_term is None:
            debug_print("[Retrieve] No memory or norm term found")
            return torch.zeros_like(query_states)

        debug_print("[Retrieve] query_states.shape", query_states.shape)
        debug_print("[Retrieve] self.memory.shape", memory.shape)

        # Apply ELU activation
        query_states = F.elu(query_states) + 1  # ELU activation + 1 for stability
        memory_output = torch.matmul(
            # GQA
            query_states,
            memory.repeat(1, self.num_key_value_groups, 1, 1),
        )

        debug_print("[Retrieve] memory_output.shape", memory_output.shape)
        debug_print("[Retrieve] self.norm_term.shape", norm_term.shape)

        # Broadcast norm_term to the shape of query_states, then sum across head_dim for normalization
        norm_term_broadcastable = torch.matmul(
            query_states,
            # GQA
            norm_term.transpose(-2, -1).repeat(1, self.num_key_value_groups, 1, 1),
        )
        debug_print(
            "[Broadcast] norm_term_broadcastable.shape", norm_term_broadcastable.shape
        )

        # Perform division
        memory_output = memory_output / norm_term_broadcastable
        return memory_output

    def _update_memory(self, key_states, value_states, memory, norm_term):
        # key_states: [batch_size, num_heads, seq_len, head_dim]
        # value_states: [batch_size, num_heads, seq_len, value_dim]

        key_states = F.elu(key_states) + 1  # Apply ELU activation

        if memory is not None:
            memory = memory + torch.matmul(key_states.transpose(-2, -1), value_states)
        else:
            memory = torch.matmul(key_states.transpose(-2, -1), value_states)

        if norm_term is not None:
            norm_term = norm_term + key_states.sum(
                dim=2, keepdim=True
            )  # Update normalization term
        else:
            norm_term = key_states.sum(
                dim=2, keepdim=True
            )  # Initialize normalization term

        debug_print("[Update] self.memory.shape", memory.shape)
        debug_print("[Update] self.norm_term.shape", norm_term.shape)

        return memory, norm_term

