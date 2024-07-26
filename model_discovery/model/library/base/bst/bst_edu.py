"""Unstructured filters and convolutions."""
import jax
from jax import numpy as jnp
from einops import rearrange

from .modules import dense, get_bias, get_init_context, positional_emb, BRecT
from .config import alpha

win_length = 512  # (w)
seq_length = 4096  # (l)

def get_filters_unstruct(channels):
    """Returns trainable filters and biases.

    Args:
        channels: number of filters.

    Returns:
        h: filter of shape (seq_length, channels, dim)
        b: bias of shape (channels, dim)
    """
    t = jnp.linspace(0.0, 1.0, seq_length)
    h = jnp.exp(-alpha * t) * dense(positional_emb(t))
    b = get_bias()
    return h, b

def multichannel_convolution(u, h, b):
    """Multichannel convolution function.

    Args:
        u: input of shape (seq_length, dim)
        h: filters of shape (seq_length, channels, dim)
        b: bias of shape (channels, dim)
    """
    h = rearrange(h, "l c d -> c d l")
    fft_size = seq_length * 2
    u_f = jnp.fft.rfft(u, n=fft_size)
    h_f = jnp.fft.rfft(h, n=fft_size)
    y = jnp.fft.irfft(h_f * u_f, n=fft_size, norm="forward")[..., :seq_length]  # (c, d, l)
    y = y + u * b[..., None]  # (c, d, l)
    y = rearrange(y, "c d l -> l d c")
    return y

"""Context state collection for BST-SH variant."""
num_heads = 8  # (h)
num_states = 32  # (s)

def SH_context_states(u):
    """Single-Head Context Collection."""
    h, b = get_filters_unstruct(channels=1)
    y_1 = multichannel_convolution(u, h, b)  # y_1: (l, d, 1)
    y_h = dense(y_1)  # lift to multiple heads, y_h: (l, d, h)
    context_states = jnp.split(y_h, seq_length // win_length, axis=0)
    return context_states  # (l/w, w, d, h)

"""Context state collection for BST-MH variant."""

def MH_context_states(u):
    """Multi-Head Context Collection."""
    h, b = get_filters_unstruct(channels=num_heads)
    y_h = multichannel_convolution(u, h, b)  # y_h: (l, d, h)
    context_states = jnp.split(y_h, seq_length // win_length, axis=0)
    return context_states  # (l/w, w, d, h)

"""Context state collection for BST-MF variant."""

def MF_context_states(u):
    """Multi-Filter Context Collection."""
    h, b = get_filters_unstruct(channels=num_states)
    y_s = multichannel_convolution(u, h, b)  # y_s: (l, d, s)
    context_states = jnp.split(y_s, seq_length // win_length, axis=0)  # (l/w, w, d, s)
    context_states = context_states[:, -1, ...]  # collect the last context states, (l/w, d, s)
    context_states = rearrange(context_states, "lw d s -> lw s d")  # shift context states
    context_states = jnp.roll(context_states, 1, axis=1)  # corresponding to windows
    init_context = get_init_context(num_states)  # replace the initial window with trainable weights, (d, s)
    context_states[0] = init_context
    context_states = dense(context_states)  # lift to multiple heads
    return context_states  # (l/w, s, d, h)

"""Block-State Transformer Layer."""

block_transformer = jax.vmap(BRecT.nonrecurrent_cell)

def BST(u):
    """Block-State Transformer Layer."""
    global MF  # True if Multi-Filter, False otherwise (SH/MH)
    u = jnp.split(u, seq_length // win_length, axis=0)  # split inputs into windows (l/w, w, d)
    context_states = SH_context_states(u) if not MF else MH_context_states(u) if num_heads > 1 else MF_context_states(u)  # collect context states from SSM outputs
    y = block_transformer(  # pass the contexts in place of recurrent states
        token_embeddings=u,
        recurrent_state=context_states,
        use_cross_attn_causal_mask=not MF,
        use_cross_positional_emb=MF,  # context IDs
    )
    return rearrange(y, "lw w d -> (lw w) d")  # (l, d)
