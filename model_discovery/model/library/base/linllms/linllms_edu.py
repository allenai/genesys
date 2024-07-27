import math
from typing import Callable, Tuple
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from lightning_attn import lightning_attn_func as lightning_attn_ops
from .utils import get_pos_embed


############### Linear attention addition ################


def no_slope_tensor(n_attention_heads: int, device: torch.device, dtype: torch.dtype):
    """
    This function returns a tensor of zeros, which is equivalent to not using any decay.
    n_attention_heads: number of attention heads
    device: device to use
    dtype: data type to use
    """
    slopes = torch.zeros(n_attention_heads, 1, 1, device=device, dtype=dtype)

    return slopes


def get_slopes_power_of_2(n, start):
    """
    This function returns a list of slopes for the linear attention function given a power of 3 number of heads.
    It is taken from the lightning attention code.
    n: number of attention heads
    start: (optional) start value for the slope tensor
    """
    ratio = 2 ** (-(2 ** -(math.log2(n) - 3)))
    if start is None:
        start = ratio
    return [start * ratio ** i for i in range(n)]


def get_slopes(n, start):
    """
    This function returns a list of slopes for the linear attention function.
    It is taken from the lightning attention code.
    n: number of attention heads
    start: (optional) start value for the slope tensor
    """
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n, start
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            get_slopes_power_of_2(closest_power_of_2, start)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def get_slope_tensor(
    n_attention_heads: int,
    start: float = None,
    use_retnet_slopes: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    """
    This function returns a tensor of slopes for the linear attention function. This determines the decay of the attention function.
    n_attention_heads: number of attention heads
    start: (optional) start value for the slope tensor
    use_retnet_slopes: (optional) use the RetNet slopes instead of the default lightning attention ones
    device: (optional) device to use
    dtype: (optional) data type to use
    """
    if use_retnet_slopes:
        head_count = torch.arange(n_attention_heads, device=device, dtype=dtype)
        gamma = 1 - torch.exp2(-5 - head_count.float())
        slopes = -torch.log(gamma.unsqueeze(-1))
    else:
        # h, 1, 1
        slopes = torch.tensor(get_slopes(n_attention_heads, start), dtype=dtype, device=device).reshape(
            n_attention_heads,
            1,
        )
    return slopes


def recurrent_forward(
    queries, keys, vals, s, qk_scale=1, start=None, use_decay=False, use_retnet_slopes=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes the output of the linear attention function in a recurrent manner.
    Its result is equivalent to the parallel computation of the linear attention function.
    queries: queries, shape (batch_size, num_heads, seq_len, dim_qk)
    keys: keys, shape (batch_size, num_heads, seq_len, dim_qk)
    vals: values, shape (batch_size, num_heads, seq_len, dim_v)
    s: current state of the RNN, shape (batch_size, num_heads, dim_qk, dim_v)
    use_decay: (optional) use the decaying factor on the distance between queries and keys
    use_retnet_slopes: (optional) use the RetNet slopes instead of the default lightning attention ones
    qk_scale: scale factor for queries and keys
    start: (optional) start value for the slope tensor in case decay is used
    """
    if use_decay:
        slope = get_slope_tensor(queries.shape[1], start, use_retnet_slopes, queries.device, queries.dtype)
        gamma = torch.exp(-slope).reshape(1, queries.shape[1], 1, 1)
    else:
        gamma = 1.0
    s_n = s + (keys.transpose(-1, -2) * qk_scale) @ vals
    output = queries @ s_n
    return output, gamma * s_n


def lightning_attn_func(
    q, k, v, qk_scale: float, start: float = None, use_decay: bool = True, use_retnet_slopes=False
) -> torch.Tensor:
    """
    This is the lightning attention function, which is a kernel linear approximation of the softmax function
    Almost the same as linear_attn_func but using the triton kernel from lightning_attn and a decaying factor (from RetNet paper https://arxiv.org/pdf/2307.08621.pdf)
    as defined by the depth_slope_tensor function (using no_slope_tensor is equivalent to linear_attn_func).
    Args:
        q: queries, shape (batch_size, num_heads, seq_len, dim_qk)
        k: keys, shape (batch_size, num_heads, seq_len, dim_qk)
        v: values, shape (batch_size, num_heads, seq_len, dim_v)
        qk_scale: scale factor for queries and keys
        start: (optional) start value for the slope tensor in case decay is used
        use_decay: (optional) use the decaying factor on the distance between queries and keys
        use_retnet_slopes: (optional) use the RetNet slopes instead of the default lightning attention ones
    """
    h = q.shape[1]
    if use_decay:
        s = get_slope_tensor(h, start, use_retnet_slopes, q.device, torch.float32)
    else:
        s = no_slope_tensor(h, q.device, q.dtype)
    output = lightning_attn_ops(q, k * qk_scale, v, s)

    return output


class LinearAttn(nn.Module):
    """
    This class implements the linear attention layer.
    It can be used as a drop-in replacement for the CustomAttn class.
    The forward method can be run in parallel or recurrent mode depending on the use_cache parameter,
    which folows the same logic as the CustomAttn class with qk_cache or without it.
    """

    def __init__(self, layer_id, args):
        super().__init__()
        self.params = args
        self.n_heads = args.n_heads
        self.n_heads_kv = args.n_heads_kv
        self.qk_head_dim = args.qk_head_dim
        self.qk_in_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim

        self.qk_in_head_dim = args.dim // args.n_heads
        self.in_proj = nn.Linear(
            args.dim,
            (
                args.n_heads * self.qk_in_head_dim
                + self.n_heads_kv * self.qk_in_head_dim
                + self.n_heads_kv * self.v_head_dim
            ),
            bias=False,
        )

        self.out_proj = nn.Linear(args.n_heads * self.v_head_dim, args.dim, bias=False)

        self.pos_embed = get_pos_embed(args)

        self.apply_qk_norm = args.apply_qk_norm

        self._totrain_gn = nn.GroupNorm(
            num_groups=args.n_heads, num_channels=args.n_heads * self.v_head_dim, affine=False
        )

        self._totrain_embed = nn.Linear(
            args.n_heads * self.qk_in_head_dim,
            args.n_heads * self.qk_head_dim,
        )

        if args.n_heads != args.n_heads_kv:
            self._totrain_embed_kv = nn.Linear(
                args.n_heads_kv * self.qk_in_head_dim,
                args.n_heads_kv * self.qk_head_dim,
            )

        self.linear_attn_fn = partial(
            lightning_attn_func,
            use_decay=args.use_decay,
            use_retnet_slopes=args.use_retnet_slopes,
            start=args.decay_start,
        )
        self.recurrent_forward_fn = partial(
            recurrent_forward,
            use_decay=args.use_decay,
            use_retnet_slopes=args.use_retnet_slopes,
            start=args.decay_start,
        )

        self.mask = None

        # initialize weights by trunc_normal(1/sqrt(fan_in))
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

        norm_dim = self.n_heads * self.qk_head_dim

        # initialize norm layers for queries and keys if needed
        NormClass = args.norm_type
        self._totrain_q_norm = (
            NormClass(
                norm_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self._totrain_k_norm = (
            NormClass(
                norm_dim,
                eps=args.norm_eps,
            )
            if self.apply_qk_norm
            else nn.Identity()
        )
        self.qk_scale = 1.0 / math.sqrt(self.qk_head_dim)

    def repeat_kv(self, hidden_states, n_rep):
        """
        This function repeats the key and value tensors to match the number of queries.
        This is needed when the number of key-value heads is different from the number of query heads (GQA or MQA).
        """
        if n_rep == 1:
            return hidden_states
        hidden_states2 = hidden_states.transpose(1, 2)
        batch, num_key_value_heads, slen, head_dim = hidden_states2.shape
        hidden_states2 = hidden_states2[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states2.reshape(batch, num_key_value_heads * n_rep, slen, head_dim).transpose(1, 2)

    def _set_mask(self, seqlen: int, device):
        if self.mask is None or self.mask.shape[-1] < seqlen:
            self.mask = torch.tril(torch.ones(1, 1, seqlen, seqlen, requires_grad=False), diagonal=0).to(device)

    def _get_qkv(self, x: torch.Tensor, offset=0):
        """
        This function computes the queries, keys, and values for the linear attention function.
        It re-uses the projection layer from a usual transformer model and applies the kernels to the queries and keys (one layer + relu).
        """
        batchsize, seqlen, _ = x.shape
        queries, keys, vals = self.in_proj(x).split(
            [
                self.n_heads * self.qk_in_head_dim,
                self.n_heads_kv * self.qk_in_head_dim,
                self.n_heads_kv * self.v_head_dim,
            ],
            dim=-1,
        )
        vals = vals.view(batchsize, seqlen, self.n_heads_kv, self.v_head_dim)

        queries = F.relu(self._totrain_embed(queries.view(batchsize, seqlen, self.n_heads * self.qk_in_head_dim))).view(
            batchsize, seqlen, self.n_heads, self.qk_head_dim
        )
        if self.n_heads != self.n_heads_kv:
            keys = F.relu(
                self._totrain_embed_kv(keys.view(batchsize, seqlen, self.n_heads_kv * self.qk_in_head_dim))
            ).view(batchsize, seqlen, self.n_heads_kv, self.qk_head_dim)
        else:
            keys = F.relu(
                self._totrain_embed(keys.view(batchsize, seqlen, self.n_heads_kv * self.qk_in_head_dim))
            ).view(batchsize, seqlen, self.n_heads_kv, self.qk_head_dim)

        keys = self.repeat_kv(keys, n_rep=self.n_heads // self.n_heads_kv)
        vals = self.repeat_kv(vals, n_rep=self.n_heads // self.n_heads_kv)

        queries = self._totrain_q_norm(queries.view(batchsize, seqlen, self.n_heads * self.qk_head_dim)).view(
            batchsize, seqlen, self.n_heads, self.qk_head_dim
        )
        keys = self._totrain_k_norm(keys.reshape(batchsize, seqlen, self.n_heads * self.qk_head_dim)).view(
            batchsize, seqlen, self.n_heads, self.qk_head_dim
        )

        queries, keys, vals = self.pos_embed(
            queries,
            keys,
            vals,
            offset=offset,
        )

        queries = queries.transpose(1, 2).contiguous()
        keys = keys.transpose(1, 2).contiguous()
        vals = vals.transpose(1, 2).contiguous()
        return queries, keys, vals

    def _output(self, output: torch.Tensor):
        """
        This function computes the output of the linear attention function.
        It applies the group normalization and the output projection layer.
        """
        output = output.transpose(1, 2).contiguous()
        batchsize, seqlen = output.shape[:2]

        output = self._totrain_gn(output.reshape(batchsize * seqlen, self.v_head_dim * self.n_heads))

        output = output.view(batchsize, seqlen, self.v_head_dim * self.n_heads)

        output = self.out_proj(output)

        return output

    def forward(
        self, x: torch.Tensor, is_causal: bool = True, past_key_value=None, use_cache=False, attention_mask=None
    ):
        """
        Run the linear attention function either in parallel (use_cache=False) or recurrent mode (use_cache=True).
        x: [B, T, D]
        is_causal: bool must be True
        past_key_value: None or tuple of (state, offset), this is a hack to repurpose the key_value cache for recurrent inference
        use_cache: bool if set to true, run the model in recurrent mode, else run parallel mode
        attention_mask: None,
        """
        assert is_causal, "LinearAttn class only supports causal mode"
        if attention_mask is not None and attention_mask.all():
            attention_mask = None
        increment = x.shape[1] if attention_mask is None else attention_mask.sum(dim=1)

        if not use_cache:
            output = self.forward_parallel(x, is_causal, attention_mask=attention_mask)
        else:
            if past_key_value is None:
                past_key_value = (None, 0)
            output, s = self.forward_recurrent(x, past_key_value[0], past_key_value[1])
            past_key_value = (s, past_key_value[1] + increment)

        return output, past_key_value

    def forward_parallel(self, x: torch.Tensor, causal, attention_mask=None):
        """
        Use the linear attention function to compute the output in parallel.
        x: [B, T, D]
        causal: bool must be True
        attention_mask: None, not supported for linear attention in parallel mode
        """
        assert attention_mask is None, "Attention mask not supported for linear attention"
        queries, keys, vals = self._get_qkv(x)
        output = self.linear_attn_fn(queries, keys, vals, self.qk_scale)
        return self._output(output)

    def forward_recurrent(
        self,
        x: torch.Tensor,
        s: torch.Tensor = None,
        offset=0,
    ):
        """
        Use the linear attention function to compute the output in recurrent mode.
        Loops over the sequence length and computes the output and the state update at each step.
        x: [B, sequence_length, D] input features
        s: [B, head, h_dim, h_dim] (optional) input recurrent state
        offset: int or [B,] (optional) sequence offset for positional embedding, encodes the current position of x in the sequence.
        """
        if s is None:
            s = torch.zeros(
                x.shape[0],
                self.n_heads,
                self.qk_head_dim,
                self.v_head_dim,
                device=x.device,
                dtype=x.dtype,
            )
        queries, keys, vals = self._get_qkv(x, offset)

        out = []
        for i in range(x.shape[1]):
            output, s = self.recurrent_forward_fn(
                queries[:, :, i : i + 1], keys[:, :, i : i + 1], vals[:, :, i : i + 1], s, qk_scale=self.qk_scale
            )
            out.append(output)

        output = torch.cat(out, dim=2)
        return self._output(output), s

