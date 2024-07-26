"""Model file that houses the architecture of the model."""

from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PatchEmbed(layers.Layer):
    """Image to Patch Embedding.

    Args:
        image_size (`Tuple[int]`): Size of the input image.
        patch_size (`Tuple[int]`): Size of the patch.
        embed_dim (`int`): Dimension of the embedding.
        chunk_size (`int`): Number of patches to be chunked.
    """

    def __init__(
        self,
        image_size: Tuple[int],
        patch_size: Tuple[int],
        embed_dim: int,
        chunk_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Compute the patch resolution.
        patch_resolution = [
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        ]

        # Store the parameters.
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_resolution = patch_resolution
        self.num_patches = patch_resolution[0] * patch_resolution[1]

        # Define the positions of the patches.
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

        # Create the layers.
        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            name="projection",
        )
        self.flatten = layers.Reshape(
            target_shape=(-1, embed_dim),
            name="flatten",
        )
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=embed_dim,
            name="position_embedding",
        )
        self.layernorm = keras.layers.LayerNormalization(
            epsilon=1e-5,
            name="layernorm",
        )
        self.chunking_layer = layers.Reshape(
            target_shape=(self.num_patches // chunk_size, chunk_size, embed_dim),
            name="chunking_layer",
        )

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, int, int, int]:
        """Call function.

        Args:
            inputs (`tf.Tensor`): Input tensor.

        Returns:
            `Tuple[tf.Tensor, int, int, int]`: Tuple of the projected input, number of patches,
                patch resolution, and embedding dimension.
        """
        # Project the inputs to the embedding dimension.
        x = self.projection(inputs)

        # Flatten the pathces and add position embedding.
        x = self.flatten(x)
        x = x + self.position_embedding(self.positions)

        # Normalize the embeddings.
        x = self.layernorm(x)

        # Chunk the tokens.
        x = self.chunking_layer(x)

        return x


class FeedForwardNetwork(layers.Layer):
    """Feed Forward Network.

    Args:
        dims (`int`): Number of units in FFN.
        dropout (`float`): Dropout probability for FFN.
    """

    def __init__(self, dims: int, dropout: float, **kwargs):
        super().__init__(**kwargs)

        # Create the layers.
        self.ffn = keras.Sequential(
            [
                layers.Dense(units=4 * dims, activation=tf.nn.gelu),
                layers.Dense(units=dims),
                layers.Dropout(rate=dropout),
            ],
            name="ffn",
        )
        self.add = layers.Add(
            name="add",
        )
        self.layernorm = layers.LayerNormalization(
            epsilon=1e-5,
            name="layernorm",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Call function.

        Args:
            inputs (`tf.Tensor`): Input tensor.

        Returns:
            `tf.Tensor`: Output tensor."""
        # Apply the FFN.
        x = self.layernorm(inputs)
        x = self.add([inputs, self.ffn(x)])
        return x


class BaseAttention(layers.Layer):
    """Base Attention Module.

    Args:
        num_heads (`int`): Number of attention heads.
        key_dim (`int`): Size of each attention head for key.
        dropout (`float`): Dropout probability for attention module.
    """

    def __init__(self, num_heads: int, key_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            name="mha",
        )
        self.q_layernorm = layers.LayerNormalization(
            epsilon=1e-5,
            name="q_layernorm",
        )
        self.k_layernorm = layers.LayerNormalization(
            epsilon=1e-5,
            name="k_layernorm",
        )
        self.v_layernorm = layers.LayerNormalization(
            epsilon=1e-5,
            name="v_layernorm",
        )
        self.add = layers.Add(
            name="add",
        )

        self.attn_scores = None

    def call(
        self, input_query: tf.Tensor, key: tf.Tensor, value: tf.Tensor
    ) -> tf.Tensor:
        """Call function.

        Args:
            input_query (`tf.Tensor`): Input query tensor.
            key (`tf.Tensor`): Key tensor.
            value (`tf.Tensor`): Value tensor.

        Returns:
            `tf.Tensor`: Output tensor."""
        # Apply the attention module.
        query = self.q_layernorm(input_query)
        key = self.k_layernorm(key)
        value = self.v_layernorm(value)
        (attn_outs, attn_scores) = self.mha(
            query=query,
            key=key,
            value=value,
            return_attention_scores=True,
        )

        # Save the attention scores for later visualization.
        self.attn_scores = attn_scores

        # Add the input to the attention output.
        x = self.add([input_query, attn_outs])
        return x


class AttentionWithFFN(layers.Layer):
    """Attention with Feed Forward Network.

    Args:
        ffn_dims (`int`): Number of units in FFN.
        ffn_dropout (`float`): Dropout probability for FFN.
        num_heads (`int`): Number of attention heads.
        key_dim (`int`): Size of each attention head for key.
        attn_dropout (`float`): Dropout probability for attention module.
    """

    def __init__(
        self,
        ffn_dims: int,
        ffn_dropout: float,
        num_heads: int,
        key_dim: int,
        attn_dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Create the layers.
        self.attention = BaseAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=attn_dropout,
            name="base_attn",
        )
        self.ffn = FeedForwardNetwork(
            dims=ffn_dims,
            dropout=ffn_dropout,
            name="ffn",
        )

        self.attn_scores = None

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
        """Call function.

        Args:
            query (`tf.Tensor`): Input query tensor.
            key (`tf.Tensor`): Key tensor.
            value (`tf.Tensor`): Value tensor.

        Returns:
            `tf.Tensor`: Output tensor.
        """
        # Apply the attention module.
        x = self.attention(query, key, value)

        # Save the attention scores for later visualization.
        self.attn_scores = self.attention.attn_scores

        # Apply the FFN.
        x = self.ffn(x)
        return x


class CustomRecurrentCell(layers.Layer):
    """Custom Recurrent Cell.

    Args:
        chunk_size (`int`): Number of tokens in a chunk.
        r (`int`): One Cross Attention per **r** Self Attention.
        num_layers (`int`): Number of layers.
        ffn_dims (`int`): Number of units in FFN.
        ffn_dropout (`float`): Dropout probability for FFN.
        num_heads (`int`): Number of attention heads.
        key_dim (`int`): Size of each attention head for key.
        attn_dropout (`float`): Dropout probability for attention module.
    """

    def __init__(
        self,
        chunk_size: int,
        r: int,
        num_layers: int,
        ffn_dims: int,
        ffn_dropout: float,
        num_heads: int,
        key_dim: int,
        attn_dropout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Save the arguments.
        self.chunk_size = chunk_size
        self.r = r
        self.num_layers = num_layers
        self.ffn_dims = ffn_dims
        self.ffn_droput = ffn_dropout
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn_dropout = attn_dropout

        # Create the state_size and output_size. This is important for
        # custom recurrence logic.
        self.state_size = tf.TensorShape([chunk_size, ffn_dims])
        self.output_size = tf.TensorShape([chunk_size, ffn_dims])

        self.get_attn_scores = False
        self.attn_scores = []

        ########################################################################
        # Perceptual Module
        ########################################################################
        perceptual_module = list()
        for layer_idx in range(num_layers):
            perceptual_module.append(
                AttentionWithFFN(
                    ffn_dims=ffn_dims,
                    ffn_dropout=ffn_dropout,
                    num_heads=num_heads,
                    key_dim=key_dim,
                    attn_dropout=attn_dropout,
                    name=f"pm_self_attn_{layer_idx}",
                )
            )
            if layer_idx % r == 0:
                perceptual_module.append(
                    AttentionWithFFN(
                        ffn_dims=ffn_dims,
                        ffn_dropout=ffn_dropout,
                        num_heads=num_heads,
                        key_dim=key_dim,
                        attn_dropout=attn_dropout,
                        name=f"pm_cross_attn_ffn_{layer_idx}",
                    )
                )
        self.perceptual_module = perceptual_module

        ########################################################################
        # Temporal Latent Bottleneck Module
        ########################################################################
        self.tlb_module = AttentionWithFFN(
            ffn_dims=ffn_dims,
            ffn_dropout=ffn_dropout,
            num_heads=num_heads,
            key_dim=key_dim,
            attn_dropout=attn_dropout,
            name=f"tlb_cross_attn_ffn",
        )

    def call(self, inputs, states) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Call function.

        Args:
            inputs (`tf.Tensor`): Input tensor.
            states (`List[tf.Tensor]`): List of state tensors.

        Returns:
            `Tuple[tf.Tensor, List[tf.Tensor]]`: Tuple of output tensor and list
                of state tensors.
        """
        # inputs => (batch, chunk_size, dims)
        # states => [(batch, chunk_size, units)]
        slow_stream = states[0]
        fast_stream = inputs

        for layer_idx, layer in enumerate(self.perceptual_module):
            fast_stream = layer(query=fast_stream, key=fast_stream, value=fast_stream)

            if layer_idx % self.r == 0:
                fast_stream = layer(
                    query=fast_stream, key=slow_stream, value=slow_stream
                )

        slow_stream = self.tlb_module(
            query=slow_stream, key=fast_stream, value=fast_stream
        )

        # Save the attention scores for later visualization.
        if self.get_attn_scores:
            self.attn_scores.append(self.tlb_module.attn_scores)

        return fast_stream, [slow_stream]
