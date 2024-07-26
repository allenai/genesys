import torch
import torch.nn as nn
import torch.nn.functional as F

# from long_net.attend import FlashAttention
from zeta.nn.attention.flash_attention import FlashAttention
from long_net.utils import XPOS, RelativePositionBias, LayerNorm, FeedForward, RMSNorm, RotaryEmbedding, SwiGLU


# add alibi, qk layer norm, one write head, multihway,
class DilatedAttention(nn.Module):
    """
    Dilated Attention Module.

    Arguments:
        dim: The dimension of the attention layers.
        heads: The number of attention heads.
        dilation_rate: The dilation rate for dilated attention.
        segment_size: The segment size for dilated attention.
        dropout (optional): The dropout probability. Default: 0.0
        causal (optional): If set to True, the attention mechanism is causal. Default: False
        use_xpos (optional): If set to True, xpos is used for positional encoding. Default: False
        use_rel_pos_bias (optional): If set to True, relative position bias is used in the attention mechanism. Default: False

    Usage:
        The `DilatedAttention` class can be used as a module for neural networks and is especially suited for transformer architectures.

        Example:
            attention = DilatedAttention(dim=512, heads=8, dilation_rate=2, segment_size=64, use_xpos=True, use_rel_pos_bias=True)
            output = attention(input_tensor)

        This will return the output tensor after applying dilated attention. The `use_xpos` and `use_rel_pos_bias` parameters allow for switching on positional encoding and relative positional bias respectively.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dilation_rate: int,
        segment_size: int,
        dropout: float = 0.0,
        causal: bool = False,
        use_xpos: bool = False,
        use_rel_pos_bias: bool = False,
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda:0",
    ) -> None:
        super(DilatedAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dilation_rate = dilation_rate
        self.segment_size = segment_size
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias
        self.qk_norm = qk_norm
        self.dtype = dtype
        self.device = device

        self.attention = FlashAttention(causal=self.causal, dropout=dropout).to(
            device
        )

        if use_xpos:
            self.xpos = XPOS(head_dim=dim // heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(
                num_buckets=32, max_distance=128, n_heads=heads
            )

        self.norm = nn.LayerNorm(dim)

        # head offsets
        self.head_offsets = nn.Parameter(torch.randn(heads, dim))

        # Linear Projections
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def get_mask(self, i, j):
        """i = row, j=column"""
        return torch.ones((i, j), device=self.device, dtype=torch.bool).triu(
            j - i + 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DilatedAttention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        batch_size, seq_len, _ = x.shape
        padding_len = -seq_len % self.segment_size
        x = F.pad(x, (0, 0, 0, padding_len))
        seq_len = seq_len + padding_len

        if self.use_xpos:
            x = self.xpos(x)

        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.dim)
        x = x[:, :, :: self.dilation_rate, :]

        # qk_norm
        if self.qk_norm:
            q, k, v = map(
                self.norm, (self.proj_q(x), self.proj_k(x), self.proj_v(x))
            )
        else:
            q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        # Perform attention
        attn_output = self.attention(q, k, v)

        # if use rel pos => apply relative positioning bias
        if self.use_rel_pos_bias:
            attn_output += self.relative_bias(
                batch_size, attn_output.size(1), attn_output.size(1)
            )

        # if causal create a mask and apply to the output
        if self.causal:
            mask = self.get_mask(attn_output.size(1), attn_output.size(1))

            attn_output = attn_output.masked_fill(mask, float("-inf"))

        # apply dropout
        attn_output = self.dropout(attn_output)
        # Scatter and concatenate
        attn_output = attn_output.reshape(batch_size, -1, self.dim)
        return attn_output
    


# helpers
def exists(val):
    return val is not None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# top k filtering


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

# Assuming necessary imports like RotaryEmbedding, SwiGLU, etc. are present



class ParallelTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        dilation_rate: int = 2,
        segment_size: int = 64,
        heads=8,
        ff_mult=4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (
            attn_inner_dim,
            dim_head,
            dim_head,
            (ff_inner_dim * 2),
        )

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(
            dim, sum(self.fused_dims), bias=False
        )
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.attn = DilatedAttention(
            dim,
            heads,
            dilation_rate,
            segment_size,
            qk_norm=True,
            *args,
            **kwargs,
        )

        self.ff_out = nn.Sequential(
            SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # attention

        attn = self.attn(x)

        # # aggregate values

        # out = einsum("b h i j, b j d -> b h i d", attn, v)

        # # merge heads

        # out = rearrange(out, "b h n d -> b n (h d)")
        return attn


# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        ff_mult=4,
        dilation_rate: int = 2,
        segment_size: int = 64,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.feedforward = (FeedForward(dim, dim, dropout=0.1),)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ParallelTransformerBlock(
                            dim,
                            dim_head,
                            dilation_rate,
                            segment_size,
                            heads,
                            ff_mult,
                        ),
                        FeedForward(dim, dim, dropout=0.1),
                    ]
                )
            )

    def forward(self, x):
        for block, ff in self.layers:
            x = block(x) + x
            x = ff(x) + x
        return x


# classes


class LongNetTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens,
        dim_head=64,
        heads=8,
        ff_mult=4,
        dilation_rate: int = 2,
        segment_size: int = 64,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, ff_mult, dilation_rate, segment_size
        )

        self.to_logits = nn.Sequential(RMSNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        return self.to_logits(x)

