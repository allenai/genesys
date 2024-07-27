from __future__ import annotations

import functools

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from neurocache.utils import COMMON_MODULE_NAMES, NeurocacheType

from .cache import BatchedCache, OnDeviceCache, OnDeviceCacheConfig
from .config import NeurocacheConfig


Array = torch.Tensor


def get_attribute(object, attr_name):
    for option in COMMON_MODULE_NAMES[attr_name]:
        if hasattr(object, option):
            return getattr(object, option)
    raise AttributeError("Cannot infer attribute. Options: {}".format(attr_name))


def repeat_kv(kv, n_groups):
    bsz, seq_len, num_kv_heads, n_neighbors, head_dim = kv.shape
    if n_groups == 1:
        return kv
    kv = kv[..., None, :, :].expand(bsz, seq_len, num_kv_heads, n_groups, n_neighbors, head_dim)
    return kv.reshape(bsz, seq_len, -1, n_neighbors, head_dim)


class CacheAttention(nn.Module):
    """
    Wrapper that adds cache attention to a layer.

    Decoder Layer: (
        ...
        self_attn
        cache_attn
        layer_norm
        ...
    )

    Args:
        config: Configuration object containing the parameters of model.
        base_layer: Layer to be wrapped.
    """

    def __init__(
        self,
        base_decoder_layer: nn.Module,
        config: NeurocacheConfig,
        base_config: PretrainedConfig,
    ):
        super().__init__()
        self.config = config

        try:
            # check if the model supports grouped query attention
            self.num_kv_heads = get_attribute(base_config, "num_kv_heads")
        except AttributeError:
            self.num_kv_heads = get_attribute(base_config, "num_heads")

        self.num_heads = get_attribute(base_config, "num_heads")
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.hidden_size = get_attribute(base_config, "hidden_size")

        self.head_dim = self.hidden_size // self.num_heads
        self.scaler = self.head_dim**-0.5
        self.r_hidden_size = self.hidden_size // self.config.compression_factor

        self.k_proj = nn.Linear(self.r_hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.r_hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.apply(self._weight_init)

    @torch.no_grad()
    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=module.in_features**-0.5)

    @torch.no_grad()
    def _init_proj_weights(self, self_attn_module):
        """Initialize projection weights with self attention weights."""
        self.q_proj.weight.data = get_attribute(self_attn_module, "q_proj").weight.data.clone()
        self.o_proj.weight.data = get_attribute(self_attn_module, "o_proj").weight.data.clone()

        if self.config.compression_factor == 1:
            # copy the weights
            self.k_proj.weight.data = get_attribute(self_attn_module, "k_proj").weight.data.clone()
            self.v_proj.weight.data = get_attribute(self_attn_module, "v_proj").weight.data.clone()
        else:
            # slice the weights
            self.k_proj.weight.data = (
                get_attribute(self_attn_module, "k_proj").weight[:, : self.r_hidden_size].clone()
            )
            self.v_proj.weight.data = (
                get_attribute(self_attn_module, "v_proj").weight[:, : self.r_hidden_size].clone()
            )

    def ext_attention(
        self,
        external_keys: Array,
        external_values: Array,
        queries: Array,
    ) -> Array:
        """Attention over (keys, values) retrieved from cache.

        Args:
            external_keys: per-query keys from cache, of shape
                [batch_size, num_queries, num_heads, num_neighbors, head_size]
            external_values: per-query values from cache, of shape
                [batch_size, num_queries, num_heads, num_neighbors, head_size]
            queries: current queries, of shape:
                [batch_size, num_queries, num_heads, head_size]

        Returns:
            Attention outputs of shape [batch_size, num_queries, num_heads, head_size]
        """
        assert external_keys.ndim == 5
        assert queries.ndim == 4
        assert external_values.shape == external_keys.shape

        # Compute attention weights.
        ext_attn = torch.einsum("...qhd,...qhid->...hqi", queries, external_keys)
        if ext_attn.dtype == torch.float16:
            ext_attn = nn.functional.softmax(ext_attn, dim=-1, dtype=torch.float32).to(
                torch.float16
            )
        else:
            ext_attn = nn.functional.softmax(ext_attn, dim=-1)

        # Compute weighted sum of values.
        attn_output = torch.einsum("...hqi,...qhid->...qhd", ext_attn, external_values)
        return attn_output

    def forward(self, hidden_states, retrieved_hidden_states, cached_key_values=None):
        """Attention over retrieved cached states.
            1. Project hidden states into queries.
            2. Project cached states into keys and values.
            3. Repeat neighbors for each query if context size > 1.
            4. Compute attention over keys and values.
            5. Project attention outputs back to hidden states.

        Args:
            hidden_states: output of the previous layer.
                [batch_size, seq_len, hidden_size]
            retrieved_hidden_states: hidden states retrieved from cache.
                [batch_size, seq_len, num_neighbors, hidden_size]
            cached_key_values: cached kv from previous passes for generation.
                [batch_size, context_size - 1, num_kv_heads, num_neighbors, head_dim]
            use_cache: whether to use key values for generation.
        """
        assert retrieved_hidden_states.shape[-1] == self.r_hidden_size
        assert hidden_states.shape[-1] == self.hidden_size
        batch_size, seq_len, hidden_size = hidden_states.shape
        _, _, num_neighbors, r_hidden_size = retrieved_hidden_states.shape

        # Project hidden states into queries.
        queries = self.q_proj(hidden_states) * self.scaler
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Project external states into keys and values.
        keys = self.k_proj(retrieved_hidden_states)
        keys = keys.view(batch_size, seq_len, self.num_kv_heads, num_neighbors, self.head_dim)
        values = self.v_proj(retrieved_hidden_states)
        values = values.view(batch_size, seq_len, self.num_kv_heads, num_neighbors, self.head_dim)

        def repeat_neighbors(neighbors: Array, ctx_size: int):
            assert neighbors.ndim == 5
            # Allows access to previous tokens neighbors
            # i-th token will have the neighbors of all tokens from (i - c) to i.
            # We clip so the first tokens do not have negative indices
            num_kv = neighbors.shape[1]
            n_heads = neighbors.shape[2]
            context = (
                torch.arange(num_kv)[:, None] + torch.arange(-ctx_size + 1, 1)[None, :]
            ).clamp_(0, num_kv - 1)

            neighbors = neighbors.moveaxis(1, 2)
            neighbors = neighbors[:, :, context]
            neighbors = neighbors.reshape(batch_size, n_heads, num_kv, -1, self.head_dim)
            neighbors = neighbors.moveaxis(1, 2)
            return neighbors

        # Repeat neighbors for each query.
        if self.config.context_size > 1:
            if cached_key_values is not None:
                keys = torch.cat([cached_key_values[2], keys], dim=1)
                values = torch.cat([cached_key_values[3], values], dim=1)

            cached_key_values = (
                keys[:, -self.config.context_size + 1 :],
                values[:, -self.config.context_size + 1 :],
            )
            keys = repeat_neighbors(keys, self.config.context_size)
            values = repeat_neighbors(values, self.config.context_size)

            if keys.shape[1] > seq_len:
                keys = keys[:, -seq_len:]
                values = values[:, -seq_len:]
        else:
            cached_key_values = (None, None)

        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        # Compute attention over keys and values.
        attn_output = self.ext_attention(keys, values, queries)

        # Project attention outputs back to hidden states.
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, cached_key_values


class Neurocache(nn.Module):
    """
    Wrapper that handles cache retrieval, attention, and cache updates.

    Args:
        base_model: model to be wrapped with cache.
        config: Configuration object containing the parameters of model.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        config: OnDeviceCacheConfig,
    ):
        super().__init__()
        self.config = config
        self.base_model_config = base_model.config
        self.hidden_size = get_attribute(base_model.config, "hidden_size")

        if self.config.compression_factor > 1:
            # project hidden states to lower dimension and normalize
            def _weight_init(m):
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=m.in_features**-0.5)

            self.cache_dim = self.hidden_size // self.config.compression_factor
            self.h_proj = nn.Linear(self.hidden_size, self.cache_dim, bias=False)
            self.h_proj.apply(_weight_init)
            self.h_norm = nn.LayerNorm(self.cache_dim)
        else:
            # no-op
            self.h_proj = nn.Identity()
            self.h_norm = nn.Identity()
            self.cache_dim = self.hidden_size

        try:
            self.cache_dtype = getattr(torch, config.cache_dtype)
        except AttributeError:
            raise ValueError(
                """Invalid cache dtype: {}, available options are float16, bfloat16, and float32""".format(
                    config.cache_dtype
                )
            )

        self.retrieval_state = {}
        self.cache_attns = nn.ModuleList([])
        self.caches = nn.ModuleList([])
        self.hooks = self._register_hooks(base_model)
        self.enabled = True
        self.update_cache = True

    def enable(self):
        """Enable cache."""
        self.enabled = True

    def disable(self):
        """Disable cache."""
        self.enabled = False

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.caches = nn.ModuleList([])
        self.cache_attns = nn.ModuleList([])
        self.hooks = []

    def should_reencode(self):
        """Check if retrieved cache should be re-encoded."""
        return (
            not self.config.inference_mode
            and self.training
            and self.config.neurocache_type == NeurocacheType.ONDEVICE
            and self.config.compression_factor > 1
        )

    def disable_update(self):
        """Disable cache update."""
        self.update_cache = False

    def enable_update(self):
        """Enable cache update."""
        self.update_cache = True

    def reinitialize_cache(self):
        """Reinitialize cache.
        This can be used to change the batch size of the cache.
        """
        self.retrieval_state = {}
        for cache in self.caches:
            cache.wrapped = None

    def infer_layers(self, base_model: PreTrainedModel):
        """Infer the layers list of the base model."""
        if hasattr(base_model, "base_model"):
            base_model = base_model.model

        if hasattr(base_model, "model"):
            base_model = base_model.model

        if hasattr(base_model, "decoder"):
            base_model = base_model.decoder

        try:
            layers = get_attribute(base_model, "layers")
            num_layers = len(layers)
        except AttributeError:
            raise AttributeError("Cannot infer decoder layers.")

        return layers, num_layers

    def get_layer_indices(self, num_layers: int):
        """Get the cache, retrieval, and attention layers indices."""

        if self.config.retrieval_map is None or len(self.config.retrieval_map) == 0:
            cache_idx = num_layers * 3 // 4
            retrieval_map = {cache_idx: cache_idx}
        else:
            retrieval_map = self.config.retrieval_map

        if self.config.attention_layers is None or len(self.config.attention_layers) == 0:
            attention_layers = list(range(min(retrieval_map.keys()), num_layers))
            self.config.attention_layers = attention_layers
        else:
            attention_layers = self.config.attention_layers

        # Check that cache layers are before attention layers
        assert min(attention_layers) >= min(
            retrieval_map.keys()
        ), "Attention layers must be greater than or equal to retrieval layers"
        return retrieval_map, sorted(set(attention_layers))

    def _register_hooks(self, base_model: PreTrainedModel):
        """Register hooks for cache retrieval, attention, and cache updates."""
        layers, num_layers = self.infer_layers(base_model)
        retrieval_map, attention_layers = self.get_layer_indices(num_layers)

        # Initialize cache attention layers
        for idx in attention_layers:
            self.cache_attns.append(
                CacheAttention(layers[idx], self.config, self.base_model_config)
            )

        layer_idx2cache_idx = {v: k for k, v in enumerate(sorted(set(retrieval_map.values())))}

        # Register retrieval hooks
        hooks = []
        for retrieval_idx, cachelayer_idx in retrieval_map.items():
            hook = get_attribute(layers[retrieval_idx], "self_attn").register_forward_hook(
                functools.partial(self.retrieve_hook, idx=layer_idx2cache_idx[cachelayer_idx]),
                prepend=True,
                with_kwargs=True,
            )
            hooks.append(hook)

        # Register cache updates hooks
        split_dims = () if self.config.global_cache else (0,)
        for layer_idx, cache_idx in layer_idx2cache_idx.items():
            # Create batched caches as many as the number of unique cache indices
            # We initialize the cache in the first forward pass to infer batch size
            # and start retrieval in the next passes.
            self.caches.append(BatchedCache(None, split_dims))

            hook = get_attribute(layers[layer_idx], "self_attn").register_forward_hook(
                functools.partial(self.update_hook, idx=cache_idx), with_kwargs=True
            )
            hooks.append(hook)

        # Register cache attention hooks
        for i, idx in enumerate(attention_layers):
            hook = get_attribute(layers[idx], "self_attn").register_forward_hook(
                functools.partial(self.attention_hook, idx=i), with_kwargs=True
            )
            hooks.append(hook)
        return hooks

    def reset_cache(self, start_of_sequence: Array):
        """Reset cache at the beginning of a sequence."""
        if not self.config.global_cache:
            for cache in self.caches:
                if cache.wrapped is not None:
                    cache.reset(start_of_sequence)

    @torch.no_grad()
    def retrieve_topk(self, phs: Array, input_mask: Array, idx: int):
        """Retrieve topk neighbors from cache."""
        return self.caches[idx].topk_retrieval(phs, input_mask, self.config.topk)

    def attention_hook(self, module, args, kwargs, outputs, idx):
        """Compute attention over retrieved cache, and combine with self attention."""
        if not self.enabled:
            return

        cache_attn_output, cached_key_values = self.cache_attns[idx](
            kwargs["hidden_states"],
            self.retrieval_state["retrieved_hidden_states"],
            cached_key_values=kwargs["past_key_value"],
        )

        # Residual connection with self attention output
        residual = outputs[0] + cache_attn_output
        past_key_value = outputs[-1] + cached_key_values if outputs[-1] is not None else None
        outputs = (residual,) + outputs[1:-1] + (past_key_value,)
        return outputs

    def retrieve_hook(self, module, args, kwargs, outputs, idx):
        """Retrieve topk neighbors from cache using the hidden states"""
        if not self.enabled:
            return

        hs = kwargs["hidden_states"].detach()
        batch_size, seq_len, hidden_size = hs.shape

        # Infer padding mask from attention mask
        if kwargs.get("attention_mask") is not None and False:
            input_mask = kwargs["attention_mask"].detach().to(torch.bool)
            input_mask = input_mask.all(axis=-1).reshape(batch_size, seq_len).logical_not()
        else:
            input_mask = None

        # Project hidden states to lower dimension and normalize these
        # will be used as queries for the cache and as values to be
        # stored in the cache. We detach since we do not backpropagate
        # through the cache. No gradients are computed during retrieval.
        with torch.no_grad():
            projected_hs = self.h_norm(self.h_proj(hs)).to(self.cache_dtype)

        # Since we need to infer the batch size before initializing
        # the cache we initialize the cache in the first forward pass
        # and start retrieval in the next passes.
        if (
            self.caches[idx].wrapped is None
            and self.config.neurocache_type == NeurocacheType.ONDEVICE
        ):
            value_dim = hidden_size if self.should_reencode() else 0

            # TODO: Just pass OnDeviceCacheConfig instead of individual args
            self.caches[idx].wrapped = OnDeviceCache(
                batch_size,
                self.config.cache_size,
                self.cache_dim,
                value_dim,
                self.config.neighborhood_size,
                self.config.similarity_fn,
                dtype=self.cache_dtype,
                ordering=self.config.cache_type,
            ).to(projected_hs.device)

            retrieved_hs_dim = value_dim if self.should_reencode() else self.cache_dim
            retrieved_hs = torch.zeros(
                batch_size,
                seq_len,
                self.config.topk * self.config.neighborhood_size,
                retrieved_hs_dim,
                dtype=hs.dtype,
            ).to(projected_hs.device)

        # Else if the cache is already initialized retrieve from cache
        else:
            # retrieve topk neighbors from cache
            assert self.caches[idx].wrapped is not None
            assert self.caches[idx].wrapped.num_caches == batch_size or self.config.global_cache, (
                "Cache batch size does not match hidden states batch size. "
                "Please re-initalize Neurocache."
            )
            keys, values = self.retrieve_topk(projected_hs, input_mask, idx)
            retrieved_hs = values if self.should_reencode() else keys
            assert retrieved_hs is not None, "Retrieved cache is None"
            # This is no-op if dtype is the same
            retrieved_hs = retrieved_hs.to(hs.dtype)

        if self.should_reencode():
            # re-project retrieved values (full size hidden states)
            # to train h_proj and h_norm layers
            retrieved_hs = self.h_norm(self.h_proj(retrieved_hs))

        # Store retrieved states for attention in following layers
        self.retrieval_state["retrieved_hidden_states"] = retrieved_hs

    def update_hook(self, module, args, kwargs, outputs, idx):
        """Update the cache with the new hidden states."""
        if not self.enabled or not self.update_cache:
            return

        hs = kwargs["hidden_states"].detach()
        batch_size, seq_len, _ = hs.shape

        # Infer padding mask from attention mask
        if kwargs.get("attention_mask") is not None and False:
            input_mask = kwargs["attention_mask"].detach().to(torch.bool)
            input_mask = input_mask.all(axis=-1).reshape(batch_size, seq_len).logical_not()
        else:
            input_mask = None

        # Project hidden states to lower dimension and normalize these
        # will be used as queries for the cache and as values to be
        # stored in the cache. This is always done without gradients,
        # even during training since we do not backpropagate through the cache.
        with torch.no_grad():
            # TODO: if the same layer is used for retrieval and update, we can
            # avoid encoding the hidden states twice.
            projected_hs = self.h_norm(self.h_proj(hs)).to(self.cache_dtype)

        if self.should_reencode():
            # Store both projected and original hidden states
            self.caches[idx].update(projected_hs, hs.to(self.cache_dtype), input_mask)
        else:
            # Store only projected hidden states
            self.caches[idx].update(projected_hs, None, input_mask)