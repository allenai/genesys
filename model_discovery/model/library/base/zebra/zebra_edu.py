import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


from transformers.utils import (
    logging,
)

from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    repeat_kv,
    apply_rotary_pos_emb,
)


logger = logging.get_logger(__name__)


def _pad_to_multiple(x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0) -> torch.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    if x.shape[dim] % block_len == 0:
        return x
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = torch.nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x


def _pad_to_multiple_2d(x: torch.Tensor, block_len: int, dim_1: int, dim_2: int, pad_value: int = 0) -> torch.Tensor:
    pad_len_1 = -x.shape[dim_1] % block_len
    pad_len_2 = -x.shape[dim_2] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim_1] += pad_len_1
        new_shape[dim_2] += pad_len_2
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim_1] = (0, pad_len_1)
    pad[dim_2] = (0, pad_len_2)
    pad = sum(pad[::-1], ())
    x = torch.nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x


def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    assert x.shape[
               dim] % block_len == 0, f"sequence length({x.shape[dim]}) should be multiple of block length({block_len})"
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1):]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)


def _concatenate_2_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.
    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]

    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 0)
    pad = sum(pad[::-1], ())
    # [batch_size, num_blocks, block_len] -> [batch_size, 1 + num_blocks , block_len]
    x = torch.nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list = []
    for i in range(2):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # [batch_size, num_blocks, 2 * block_len, ...]
    return torch.cat(blocks_list, dim=sequence_dim)


def _get_local_casual_attention_mask(block_len: int, device=None) -> torch.Tensor:
    m = torch.cat([torch.zeros((block_len, block_len + 1)), torch.ones((block_len, block_len))], dim=-1).to(device)
    m = m.reshape(-1)[: block_len * block_len * 2]
    return m.reshape(block_len, block_len * 2).unsqueeze(0).unsqueeze(0) > 0.5


def _get_local_attention_mask(m: torch.Tensor, block_len: int) -> torch.Tensor:
    """ Construct the local attention mask from the original attention mask.
        The Input shape is: [batch_size, 1, seq_len, seq_len]
        The Output shape is: [batch_size * num_blocks, 1, block_len, 2 * block_len]
    """
    # First Padding to Multiple of block_len
    if m.shape[-2] % block_len != 0 or m.shape[-1] % block_len != 0:
        m = _pad_to_multiple_2d(m, block_len, dim_1=-2, dim_2=-1, pad_value=1)

    # Reshape to [batch_size, 1, num_blocks, block_len, num_blocks, block_len]
    num_blocks = m.shape[-2] // block_len
    output_shape = m.shape[:-2] + (num_blocks, block_len) + (num_blocks, block_len)
    blocked_m = m.reshape(output_shape)

    # Padding One Block at dim -2
    pad = [(0, 0)] * blocked_m.ndim
    pad[-2] = (1, 0)
    pad = sum(pad[::-1], ())
    # [batch_size, 1, num_blocks, block_len, 1 + num_blocks, block_len]
    padded_m = torch.nn.functional.pad(blocked_m, pad=pad, mode="constant", value=1)
    mask_block_list = []
    for i in range(2):
        indices = [slice(0, None)] * padded_m.ndim
        indices[-2] = slice(i, i + num_blocks)
        indices = tuple(indices)
        mask_block_list.append(padded_m[indices])
    # shape of [batch_size, 1, num_blocks, block_len, num_block, 2 * block_len]
    cat_m = torch.cat(mask_block_list, dim=-1)
    # shape of [num_blocks, batch_size, 1, block_len, 2 * block_len]
    ret_m = cat_m[:, :, torch.arange(num_blocks), :, torch.arange(num_blocks), :].transpose(0, 1).transpose(1, 2)
    return ret_m


def attention_mask_func(attn_score, attn_mask):
    dtype = attn_score.dtype
    attn_score = attn_score.mask_fill(attn_mask, torch.finfo(dtype).min)
    # attn_score = attn_score.mask_fill(attn_mask, -10000.0)
    return attn_score


class MaskedSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_func = attention_mask_func

    def forward(self, input, mask):
        dtype = input.dtype
        input = input.to(dtype=torch.float32)
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output).to(dtype)
        return probs


class ZebraMixAttention(nn.Module):
    """Sparse attention implementation by Kaiqiang"""

    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings
        # Added for Mix Attention by Kaiqiang
        self.self_attn_type = config.self_attn_type
        self.block_len = config.window_size
        self.layer_group_size = config.layer_group_size
        self.softmax_func = MaskedSoftmax()

        # Addef for Mix Attention by Kaiqiang
        if self.self_attn_type == "mix":
            if self.layer_id % self.layer_group_size == 0:
                self.self_attn_type = "full"
            else:
                self.self_attn_type = "sparse"

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            rope_theta = self.config.rope_theta
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=rope_theta,
                    scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=rope_theta,
                    scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def full_attention(self, query_states, key_states, value_states, attention_mask, help_args):
        bsz = help_args["bsz"]
        q_len = help_args["q_len"]
        kv_seq_len = help_args["kv_seq_len"]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if self.self_attn_type == "full" and attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Full Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if self.self_attn_type == "full" and attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Full Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
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

        return attn_output, attn_weights

    def sparse_attention(self, query_states, key_states, value_states, attention_mask, help_args):
        """ states: bsz, self.num_heads, q/k/v_len, self.head_dim
        """
        bsz = help_args["bsz"]
        q_len = help_args["q_len"]
        kv_seq_len = help_args["kv_seq_len"]

        assert q_len == kv_seq_len, \
            f"sparse attention only used for training when q_len({q_len} == kv_seq_len({kv_seq_len}))"

        # Convert Attention to 0 (valid) and 1 (invalid)
        if attention_mask is not None:
            attention_mask = (attention_mask < 0)

        # Transpose to shape: bsz, seq_len, num_heads, head_dim
        query_layer = query_states.transpose(1, 2)
        key_layer = key_states.transpose(1, 2)
        value_layer = value_states.transpose(1, 2)

        # Padded to multiple
        query_layer = _pad_to_multiple(query_layer, self.block_len, dim=1, pad_value=0)
        key_layer = _pad_to_multiple(key_layer, self.block_len, dim=1, pad_value=0)
        value_layer = _pad_to_multiple(value_layer, self.block_len, dim=1, pad_value=0)

        padded_seq_len = query_layer.shape[1]
        num_blocks = padded_seq_len // self.block_len

        ###############################################
        # Processing Q,K,V for local attention
        ###############################################

        # split into blocks -> (batch_size, num_blocks, block_len, num_heads_per_partition, dim_per_head)
        query_layer_local = _split_into_blocks(query_layer, self.block_len, dim=1)
        key_layer_local = _split_into_blocks(key_layer, self.block_len, dim=1)
        value_layer_local = _split_into_blocks(value_layer, self.block_len, dim=1)

        # Concatenate 2 blocks for keys and values
        # -> (batch_size, num_blocks, 2 * block_len, num_heads_per_partition, dim_per_head)
        key_layer_local = _concatenate_2_blocks(key_layer_local, block_dim=1, sequence_dim=2)
        value_layer_local = _concatenate_2_blocks(value_layer_local, block_dim=1, sequence_dim=2)

        ###############################################
        # Calculate Local Attention Score
        ###############################################

        # Compute Local Attention Scores
        # -> (batch_size, num_heads_per_partition, num_blocks, block_len, 2 * block_len)
        attn_score_local = torch.einsum(
            "...qhd,...khd->...hqk", query_layer_local, key_layer_local
        ).transpose(1, 2)

        alpha = 1.0 / self.norm_factor
        attn_score_local = alpha * attn_score_local

        # Convert Shape to [b, np, sq, sk] Style
        # -> (batch_size, num_heads_per_partition, padded_seq_len, 2 * block_len)
        new_shape = (bsz, self.num_heads, padded_seq_len, 2 * self.block_len)
        attn_score_local = attn_score_local.reshape(new_shape)

        ###############################################
        # Building Local Attention Masks
        ###############################################

        # Get local attention mask
        # -> (batch_size * num_blocks, 1, block_len, 2 * block_len)
        attn_mask_local = _get_local_attention_mask(attention_mask, self.block_len)
        attn_mask_local_ = _get_local_casual_attention_mask(self.block_len, device=attn_mask_local.device)
        attn_mask_local = torch.logical_or(attn_mask_local, attn_mask_local_)

        # Convert Shape to [b, np, sq, sk] Style
        # -> (batch_size, 1, padded_seq_len, 2 * block_len)
        new_shape = (bsz, 1, padded_seq_len, 2 * self.block_len)
        attn_mask_local = attn_mask_local.reshape(new_shape)

        ###############################################
        # Calculating attention probabilities
        ###############################################

        # using softmax to calculate the attention probabilities
        attn_probs = self.softmax_func(attn_score_local, attn_mask_local)

        # Convert attn_probs
        # -> (batch_size, num_heads_per_partition, num_blocks, block_len, 2 * block_len)
        shape = (bsz, self.num_heads, num_blocks, self.block_len, 2 * self.block_len)
        attn_probs = attn_probs.reshape(shape)
        # Convert attn_probs
        # -> (batch_size, num_blocks, num_heads_per_partition, block_len, 2 * block_len)
        attn_probs = attn_probs.transpose(1, 2)

        # shape: (batch_Size, num_blocks, block_len, n_head, dim_per_head)
        attn_outputs = torch.einsum(
            "...hqk,...khd->...qhd", attn_probs, value_layer_local
        )

        # convert attn_output
        # -> (batch_size, num_blocks * block_len, n_head * dim_per_head)
        attn_outputs = attn_outputs.reshape(
            bsz,
            padded_seq_len,
            self.num_heads * self.head_dim
        )

        # Removing the padded length and transpose
        # -> (batch_size, seq_len, dim_per_partition)
        attn_outputs = attn_outputs.narrow(1, 0, q_len)

        if attn_outputs.size() != (bsz, q_len, self.num_heads * self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_heads * self.head_dim)}, but is"
                f" {attn_outputs.size()}"
            )

        return attn_outputs, attn_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if position_ids is not None:
            position_ids = position_ids.to(hidden_states.device)
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = query_states.to(hidden_states.device)
        key_states = key_states.to(hidden_states.device)
        value_states = value_states.to(hidden_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # make on same device
        cos = cos.to(hidden_states.device)
        sin = sin.to(hidden_states.device)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        help_args = {
            "bsz": bsz,
            "q_len": q_len,
            "kv_seq_len": kv_seq_len
        }

        attention_mask = attention_mask.to(hidden_states.device)

        if self.self_attn_type == "full":
            # use fully attention for full attention
            attn_output, attn_weights = self.full_attention(
                query_states, key_states, value_states, attention_mask, help_args
            )
        elif use_cache:
            # use full attention with truncated key/value cache for sparse attention
            truncate_attention_mask = attention_mask
            if attention_mask.shape[3] > self.block_len + 1:
                truncate_attention_mask[:, :, :, :-(self.block_len + 1)].fill_(torch.finfo(attention_mask.dtype).min)
            attn_output, attn_weights = self.full_attention(
                query_states, key_states, value_states, truncate_attention_mask, help_args
            )
        else:
            # use sparse attention only for training
            # print("In cache", position_ids, self.layer_id, "use sparse attn")
            attn_output, attn_weights = self.sparse_attention(
                query_states, key_states, value_states, attention_mask, help_args
            )

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

