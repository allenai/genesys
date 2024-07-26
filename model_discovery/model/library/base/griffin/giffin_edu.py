# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Griffin model."""
from typing import Literal, overload
import math

from recurrentgemma import common
from recurrentgemma.torch import array_typing as at
from recurrentgemma.torch import layers
from recurrentgemma.torch.modules import *
import torch
from torch import nn
from torch.utils import checkpoint
import einops



class LocalAttentionBlock(nn.Module):
  """Local Multi-Head Attention (MHA) block."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      window_size: int,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the local attention block.

    Args:
      width: The width of the block.
      num_heads: The number of heads for the attention mechanism.
      window_size: The local attention window size.
      final_w_init_variance_scale: The scale for the initialization of the last
        layer of the block.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialization.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.window_size = window_size
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Layers.
    self.proj_q = nn.Linear(
        in_features=self.width,
        out_features=self.width,
        bias=False,
        device=device,
        dtype=dtype,
    )
    self.proj_k = nn.Linear(
        in_features=self.width,
        out_features=self.head_dim,
        bias=False,
        device=device,
        dtype=dtype,
    )
    self.proj_v = nn.Linear(
        in_features=self.width,
        out_features=self.head_dim,
        bias=False,
        device=device,
        dtype=dtype,
    )
    self.proj_final = nn.Linear(
        in_features=self.width,
        out_features=self.width,
        bias=True,
        device=device,
        dtype=dtype,
    )


  @property
  def head_dim(self) -> int:
    return self.width // self.num_heads

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the queries, keys and values projections."""
    torch.nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0 / self.width))

  def out_w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the final projection."""
    std = math.sqrt(self.final_w_init_variance_scale / self.width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(
      self,
      x: at.Activations,
      segment_pos: at.SegmentPos,
      cache: AttentionBlockCache | None = None,
      return_cache: bool = True,
  ) -> tuple[at.Activations, AttentionBlockCache | None]:
    """Calls the local attention block.

    Args:
      x: Sequence of input activations.
      segment_pos: Positions of each token in the sequence.
      cache: Optiona KV-cache for the block, of previous keys and values.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the block together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    b, t, _ = x.shape
    assert segment_pos.shape == (b, t), segment_pos.shape

    # Generate keys, values and queries.
    queries = self.proj_q(x)
    keys = self.proj_k(x)
    values = self.proj_v(x)
    queries = einops.rearrange(
        queries, "... (n h) -> ... n h", n=self.num_heads
    )
    keys = einops.rearrange(keys, "... (n h) -> ... n h", n=1)
    values = einops.rearrange(values, "... (n h) -> ... n h", n=1)

    # Apply rotary embeddings.
    queries = _apply_rope(queries, segment_pos)
    keys = _apply_rope(keys, segment_pos)

    if cache is not None:
      no_cache_keys, no_cache_values = keys, values

      keys = torch.concatenate([cache.keys, no_cache_keys], dim=-3)
      values = torch.concatenate([cache.values, no_cache_values], dim=-3)
      attn_mask = _compute_cache_mask(t, cache.num_tokens, self.window_size)

      if return_cache:
        new_cache = _update_attention_cache(
            no_cache_keys, no_cache_values, segment_pos, cache
        )
      else:
        new_cache = None

    else:
      attn_mask = _compute_forward_pass_mask(segment_pos, self.window_size)

      if return_cache:
        new_cache = _attention_cache_from_prompt(
            keys, values, segment_pos, self.window_size
        )
      else:
        new_cache = None

    # Compute attention.
    logits = einops.einsum(queries, keys, "b t n h, b s n h -> b n t s")
    logits = logits * (self.head_dim**-0.5)
    # Expand for heads axis.
    attn_mask = torch.unsqueeze(attn_mask, dim=1)

    masked_logits = torch.where(attn_mask, logits, _MIN_LOGITS_VALUE)
    masked_logits = masked_logits.type(torch.float32)

    probs = nn.functional.softmax(masked_logits, dim=-1).type_as(x)
    encoded = einops.einsum(probs, values, "b n t s, b s n h -> b t n h")
    encoded = einops.rearrange(
        encoded, "... n h -> ... (n h)", n=self.num_heads
    )
    attn_output = self.proj_final(encoded)

    return attn_output, new_cache



class RecurrentBlock(nn.Module):
  """Griffin and Hawk's recurrent block."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the recurrent block.

    Args:
      width: The width of the block.
      num_heads: The number of RG-LRU heads/blocks to use.
      lru_width: Internal dimension to be projected into for RG-LRU to operate
        on.
      conv1d_temporal_width: The temporal width of the 1d convolution.
      final_w_init_variance_scale: The scale for the initialization of the last
        layer of the block.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialization.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.lru_width = lru_width or width
    self.conv1d_temporal_width = conv1d_temporal_width
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Layers.
    self.linear_y = nn.Linear(
        in_features=self.width,
        out_features=self.lru_width,
        device=device,
        dtype=dtype,
    )
    self.linear_x = nn.Linear(
        in_features=self.width,
        out_features=self.lru_width,
        device=device,
        dtype=dtype,
    )
    self.linear_out = nn.Linear(
        in_features=self.lru_width,
        out_features=self.width,
        device=device,
        dtype=dtype,
    )
    self.conv_1d = layers.Conv1D(
        width=self.lru_width,
        temporal_width=self.conv1d_temporal_width,
        device=device,
        dtype=dtype,
    )
    self.rg_lru = layers.RGLRU(
        width=self.lru_width,
        num_heads=self.num_heads,
        device=device,
        dtype=dtype,
    )


  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the linear x and y layers of the block."""
    torch.nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0 / self.width))

  def out_w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weights of the last layer of the block."""
    std = math.sqrt(self.final_w_init_variance_scale / self.lru_width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(
      self,
      x: at.Activations,
      segment_pos: at.SegmentPos,
      cache: RecurrentBlockCache | None = None,
      return_cache: bool = True,
  ) -> tuple[at.Activations, RecurrentBlockCache | None]:
    """Calls the recurrent block.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      cache: Optional cache with the previous state of the RG-LRU and Conv1D.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the block together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    # y branch.
    y = self.linear_y(x)
    y = gelu(y)

    # x branch.
    x = self.linear_x(x)
    x, conv1d_state = self.conv_1d(
        x=x,
        segment_pos=segment_pos,
        cache=None if cache is None else cache.conv1d_state,
        return_cache=return_cache,
    )
    x, rg_lru_state = self.rg_lru(
        x=x,
        segment_pos=segment_pos,
        cache=None if cache is None else cache.rg_lru_state,
        return_cache=return_cache,
    )

    # Join branches.
    x = x * y
    x = self.linear_out(x)

    if not return_cache:
      return x, None

    return x, RecurrentBlockCache(
        conv1d_state=conv1d_state,
        rg_lru_state=rg_lru_state,
    )


class ResidualBlock(nn.Module):
  """Griffin and Hawk's residual block."""

  def __init__(
      self,
      width: int,
      mlp_expanded_width: int,
      num_heads: int,
      attention_window_size: int,
      temporal_block_type: common.TemporalBlockType,
      lru_width: int | None = None,
      conv1d_temporal_width: int = 4,
      final_w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the residual block.

    Args:
      width: The width of the block.
      mlp_expanded_width: The width of the expansion inside the MLP block.
      num_heads: The number of heads for the Attention or the RG-LRU.
      attention_window_size: The window size for the local attention block.
      temporal_block_type: Either "RECURRENT" or "ATTENTION", specifying the
        type of recurrent block to use.
      lru_width: The width of the RG-LRU if different from `width`.
      conv1d_temporal_width: The width of the temporal convolution.
      final_w_init_variance_scale: The scale for the variance of the
        initializations of the sub blocks.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialization.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.mlp_expanded_width = mlp_expanded_width
    self.num_heads = num_heads
    self.attention_window_size = attention_window_size
    self.temporal_block_type = temporal_block_type
    self.lru_width = lru_width
    self.conv1d_temporal_width = conv1d_temporal_width
    self.final_w_init_variance_scale = final_w_init_variance_scale

    # Sub-blocks and layers.
    self.temporal_pre_norm = layers.RMSNorm(
        width=self.width, device=device, dtype=dtype
    )

    match self.temporal_block_type:
      case common.TemporalBlockType.RECURRENT:
        self.recurrent_block = RecurrentBlock(
            width=self.width,
            num_heads=self.num_heads,
            lru_width=self.lru_width,
            conv1d_temporal_width=self.conv1d_temporal_width,
            final_w_init_variance_scale=self.final_w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

      case common.TemporalBlockType.ATTENTION:
        self.attention_block = LocalAttentionBlock(
            width=self.width,
            num_heads=self.num_heads,
            window_size=self.attention_window_size,
            final_w_init_variance_scale=self.final_w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

    self.channel_pre_norm = layers.RMSNorm(
        width=width, device=device, dtype=dtype,
    )
    self.mlp_block = MLPBlock(
        width=self.width,
        expanded_width=self.mlp_expanded_width,
        final_w_init_variance_scale=self.final_w_init_variance_scale,
        device=device,
        dtype=dtype,
    )


  @property
  def temporal_block(self) -> nn.Module:
    """Alias for the temporal block.

    This creates a common interface while making the layer / parameter types
    easily identifiable by name in a state dictionary.
    """
    match self.temporal_block_type:
      case common.TemporalBlockType.RECURRENT:
        return self.recurrent_block
      case common.TemporalBlockType.ATTENTION:
        return self.attention_block

  @at.typed
  def forward(
      self,
      x: at.Activations,
      segment_pos: at.SegmentPos,
      cache: ResidualBlockCache | None = None,
      return_cache: bool = True,
  ) -> tuple[at.Activations, ResidualBlockCache | None]:
    """Calls the residual block.

    Args:
      x: Sequence of input activations.
      segment_pos: Positions of each token in the sequence.
      cache: Optional cache for the block.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the block together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    raw_x = x

    inputs_normalized = self.temporal_pre_norm(raw_x)
    x, cache = self.temporal_block(
        inputs_normalized, segment_pos, cache, return_cache=return_cache
    )

    residual = x + raw_x

    x = self.channel_pre_norm(residual)
    x = self.mlp_block(x)

    x = x + residual

    return x, cache



class Griffin(nn.Module):
  """Griffin model - https://arxiv.org/abs/2402.19427."""

  def __init__(
      self,
      config: common.GriffinConfig,
      gradient_checkpointing: bool = True,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the Griffin model.

    Args:
      config: The Griffin config.
      gradient_checkpointing: Whether to apply gradient checkpointing on every
        residual block.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.config = config
    self.gradient_checkpointing = gradient_checkpointing

    self.embedder = Embedder(
        vocab_size=self.config.vocab_size,
        embed_dim=self.config.width,
        scale_by_sqrt_dim=self.config.embeddings_scale_by_sqrt_dim,
        device=device,
        dtype=dtype,
    )

    self.blocks = nn.ModuleList([
        ResidualBlock(
            width=self.config.width,
            mlp_expanded_width=self.config.mlp_expanded_width,
            num_heads=self.config.num_heads,
            attention_window_size=self.config.attention_window_size,
            temporal_block_type=block_type,
            lru_width=self.config.lru_width,
            final_w_init_variance_scale=2.0 / self.config.num_layers,
            device=device,
            dtype=dtype,
        )
        for block_type in self.config.block_types
    ])
    self.final_norm = layers.RMSNorm(
        width=self.config.width, device=device, dtype=dtype
    )


  @at.typed
  def forward(
      self,
      tokens: at.Tokens,
      segment_pos: at.SegmentPos,
      cache: Cache | None = None,
      return_logits: bool = True,
      return_cache: bool = True,
  ) -> tuple[at.TokenLogits | None, Cache | None]:
    """Calls Griffin.

    Args:
      tokens: Sequence of input tokens.
      segment_pos: Positions of each token in the sequence.
      cache: Optiona for the model.
      return_logits: Whether to compute and return the logits.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the model together with the updated cache. If `cache` is None
      than the returned updated cache is empty initialized and filled in from
      the input sequence.
    """
    if not return_logits and not return_cache:
      return None, None

    input_emb = self.embedder.encode(tokens)
    x = input_emb

    new_cache = {}
    for i, block in enumerate(self.blocks):
      block_name = f"blocks.{i}"
      block_cache = None if cache is None else cache[block_name]
      if self.gradient_checkpointing:
        x, new_cache[block_name] = checkpoint.checkpoint(
            block,
            x,
            segment_pos,
            block_cache,
            return_cache,
            use_reentrant=False,
            determinism_check="none",
        )
      else:
        x, new_cache[block_name] = block(x, segment_pos, block_cache)

    if not return_logits:
      return None, new_cache

    x = self.final_norm(x)
    logits = self.embedder.decode(x)

    c = self.config.logits_soft_cap
    if c:
      logits = nn.functional.tanh(logits / c) * c

    if not return_cache:
      return logits, None

    return logits, new_cache
