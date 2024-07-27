import torch
import torch.nn as nn

from math import sqrt
from torch import Tensor
from typing import Optional, Tuple, Union

from torch.nn import Linear, Module, Parameter

try:
    from torch.nn.modules.linear import _LinearWithBias
except ImportError:
    _LinearWithBias = None


def hopfield_core_forward(query,                           # type: Tensor
                          key,                             # type: Tensor
                          value,                           # type: Tensor
                          embed_dim_to_check,              # type: int
                          num_heads,                       # type: int
                          in_proj_weight,                  # type: Optional[Tensor]
                          in_proj_bias,                    # type: Optional[Tensor]
                          bias_k,                          # type: Optional[Tensor]
                          bias_v,                          # type: Optional[Tensor]
                          add_zero_attn,                   # type: bool
                          dropout_p,                       # type: float
                          out_proj_weight,                 # type: Tensor
                          out_proj_bias,                   # type: Tensor
                          training=True,                   # type: bool
                          key_padding_mask=None,           # type: Optional[Tensor]
                          need_weights=True,               # type: bool
                          attn_mask=None,                  # type: Optional[Tensor]
                          use_separate_proj_weight=False,  # type: bool
                          q_proj_weight=None,              # type: Optional[Tensor]
                          k_proj_weight=None,              # type: Optional[Tensor]
                          v_proj_weight=None,              # type: Optional[Tensor]
                          static_k=None,                   # type: Optional[Tensor]
                          static_v=None,                   # type: Optional[Tensor]

                          key_as_static=False,             # type: bool
                          query_as_static=False,           # type: bool
                          value_as_static=False,           # type: bool
                          value_as_connected=False,        # type: bool
                          normalize_pattern=False,         # type: bool
                          normalize_pattern_eps=1e-5,      # type: float
                          p_norm_weight=None,              # type: Optional[Tensor]
                          p_norm_bias=None,                # type: Optional[Tensor]
                          head_dim=None,                   # type: Optional[int]
                          pattern_dim=None,                # type: Optional[int]
                          scaling=None,                    # type: Optional[Union[float, Tensor]]
                          update_steps_max=0,              # type: Optional[Union[int, Tensor]]
                          update_steps_eps=1e-4,           # type: Union[float, Tensor]
                          return_raw_associations=False,   # type: bool
                          return_projected_patterns=False  # type: bool
                          ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
            See "Hopfield Networks is All You Need" for more details in the setting of Hopfield networks.
        embed_dim_to_check: total dimension of the model (in case of default head dimension).
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.

        key_as_static: interpret specified key as being static.
        query_as_static: interpret specified key as being static.
        value_as_static: interpret specified key as being static.
        value_as_connected: connect value projection with key projection.
        normalize_pattern: enable normalization of patterns.
        normalize_pattern_eps: offset of the denominator for numerical stability.
        p_norm_weight, p_norm_bias: pattern normalization weight and bias.
        head_dim: dimensionality of each head.
        pattern_dim: dimensionality of each projected value input.
        scaling: scaling of association heads, often represented as beta (one entry per head).
        update_steps_max: maximum count of association update steps (None equals to infinity).
        update_steps_eps: minimum difference threshold between two consecutive association update steps.
        return_raw_associations: return raw association (softmax) values, unmodified.
        return_projected_patterns: return pattern projection values, unmodified.

    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, head_dim)`, where S is the source sequence length, N is the batch size.
        - static_v: :math:`(N*num_heads, S, head_dim)`, where S is the source sequence length, N is the batch size.

        - scaling: :math:`(num_heads,)`, where num_heads is the amount of heads.

        Outputs:
        - attn_output: :math:`(L, N, E)`, where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)`, where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        - attn_raw: :math:``(N, num_heads, L, S)`, where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and nn.functional.has_torch_function(tens_ops):
            return nn.functional.handle_torch_function(
                hopfield_core_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v,
                key_as_static=key_as_static, query_as_static=query_as_static,
                value_as_static=value_as_static, value_as_connected=value_as_connected,
                normalize_pattern=normalize_pattern, normalize_pattern_eps=normalize_pattern_eps,
                p_norm_weight=p_norm_weight, p_norm_bias=p_norm_bias,
                head_dim=head_dim, pattern_dim=pattern_dim, scaling=scaling, update_steps_max=update_steps_max,
                update_steps_eps=update_steps_eps, return_raw_associations=return_raw_associations)
    tgt_len, bsz, embed_dim = query.shape[0], value.shape[1], query.shape[2]
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    assert (scaling is None) or (type(scaling) in (float, torch.Tensor))
    if type(scaling) == torch.Tensor:
        assert scaling.ndimension() == 1 and scaling.shape[0] == num_heads, "only one entry per head."

    assert (update_steps_max is None) or (type(update_steps_max) in (int, torch.Tensor))
    if type(update_steps_max) == torch.Tensor:
        assert update_steps_max.ndimension() == 1 and update_steps_max.shape[0] == num_heads, "only one entry per head."
    elif type(update_steps_max) == int:
        update_steps_max = torch.tensor([update_steps_max] * num_heads, dtype=torch.int32, device=query.device)
    elif update_steps_max is None:
        update_steps_max = -torch.ones(size=(num_heads,), dtype=torch.int32, device=query.device)

    assert type(update_steps_eps) in (float, torch.Tensor)
    if type(update_steps_eps) == torch.Tensor:
        assert update_steps_eps.ndimension() == 1 and update_steps_eps.shape[0] == num_heads, "only one entry per head."
        assert (update_steps_eps <= 0.0).sum() == 0, "only positive thresholds allowed."
        update_steps_eps = update_steps_eps.to(device=query.device)
    elif type(update_steps_eps) == float:
        assert update_steps_eps > 0, "only positive thresholds allowed."
        update_steps_eps = torch.tensor([update_steps_eps] * num_heads, dtype=query.dtype, device=query.device)

    # Adapt dimensionality of each each.
    if head_dim is None:
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, r'embed_dim must be divisible by num_heads.'
    hopfield_dim = num_heads * head_dim

    # Adapt dimensionality of each value projection.
    if pattern_dim is None:
        pattern_dim = head_dim
    assert (not value_as_connected) or (pattern_dim == head_dim)

    q, k, v, xi, src_len = None, None, None, None, 0
    update_step, xi_old, xi_difference_norm = 0, None, float(r'+inf')
    update_active_heads = torch.tensor([[[True]]] * num_heads * bsz, device=query.device)
    assert update_active_heads.any(), "at least one head needs to be active."

    ####################################################################################################################
    #                                         BEGIN HOPFIELD UPDATE ITERATION                                          #
    ####################################################################################################################

    while update_active_heads.any():

        # The query is already projected into the "Hopfield" space at "update_step" equals 0.
        # No more projection necessary if "update_step" greater than 0.
        if update_step == 0:
            if not use_separate_proj_weight:

                if torch.equal(query, key) and torch.equal(key, value) and not (
                        key_as_static or query_as_static or value_as_static):
                    # self-attention
                    q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

                elif torch.equal(key, value) and not (key_as_static or value_as_static):
                    # encoder-decoder attention
                    _start, _end = 0, hopfield_dim
                    if query_as_static:
                        q = query.repeat(1, num_heads, 1)
                    else:
                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        q = nn.functional.linear(query, _w, _b)
                        _start = hopfield_dim
                    _end = None

                    if key is None:
                        assert value is None
                        k = None
                        v = None
                    else:

                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

                else:
                    _start, _end = 0, hopfield_dim
                    if query_as_static:
                        q = query.repeat(1, num_heads, 1)
                    else:
                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        q = nn.functional.linear(query, _w, _b)
                        _start += hopfield_dim
                        _end += hopfield_dim

                    if key_as_static:
                        k = key.repeat(1, num_heads, 1)
                    else:
                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        k = nn.functional.linear(key, _w, _b)
                        _start += hopfield_dim
                        _end += hopfield_dim

                    if value_as_static:
                        v = value.repeat(1, num_heads, 1)
                    else:
                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _w = in_proj_weight[_start:_end, :]
                        if _b is not None:
                            _b = _b[_start:_end]
                        v = nn.functional.linear(value, _w, _b)
            else:
                _start, _end = 0, hopfield_dim
                if query_as_static:
                    q = query.repeat(1, num_heads, 1)
                else:
                    q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
                    len1, len2 = q_proj_weight_non_opt.size()
                    assert len1 == hopfield_dim and len2 == query.size(-1)
                    if in_proj_bias is not None:
                        q = nn.functional.linear(query, q_proj_weight_non_opt, in_proj_bias[_start:_end])
                        _start += hopfield_dim
                        _end += hopfield_dim
                    else:
                        q = nn.functional.linear(query, q_proj_weight_non_opt, in_proj_bias)

                v = value
                if key_as_static:
                    k = key.repeat(1, num_heads, 1)
                else:
                    k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
                    len1, len2 = k_proj_weight_non_opt.size()
                    assert len1 == hopfield_dim and len2 == key.size(-1)

                    _bias = None if in_proj_bias is None else in_proj_bias[_start:_end]
                    k = nn.functional.linear(key, k_proj_weight_non_opt, _bias)
                    if value_as_connected:
                        v = nn.functional.linear(v, k_proj_weight_non_opt, _bias)
                    _start += hopfield_dim
                    _end += num_heads * pattern_dim

                if value_as_static:
                    if not (value_as_connected or key_as_static):
                        v = v.repeat(1, num_heads, 1)
                else:
                    v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
                    len1, len2 = v_proj_weight_non_opt.size()
                    assert len1 == (num_heads * pattern_dim) and len2 == v.size(-1)
                    if in_proj_bias is not None:
                        v = nn.functional.linear(v, v_proj_weight_non_opt, in_proj_bias[_start:])
                    else:
                        v = nn.functional.linear(v, v_proj_weight_non_opt, in_proj_bias)

            if attn_mask is not None:
                assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                       attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or \
                       attn_mask.dtype == torch.bool, \
                       'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
                if attn_mask.dtype == torch.uint8:
                    warnings.warn(
                        "Byte tensor for attn_mask in nn.HopfieldCore is deprecated. Use bool tensor instead.")
                    attn_mask = attn_mask.to(torch.bool)

                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                    if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                        raise RuntimeError('The size of the 2D attn_mask is not correct.')
                elif attn_mask.dim() == 3:
                    if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                        raise RuntimeError('The size of the 3D attn_mask is not correct.')
                else:
                    raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
                # attn_mask's dim is 3 now.

            # Optionally normalize patterns.
            if normalize_pattern:
                q = torch.nn.functional.layer_norm(
                    input=q.reshape(shape=(-1, head_dim)), normalized_shape=(head_dim,),
                    weight=p_norm_weight, bias=p_norm_bias, eps=normalize_pattern_eps).reshape(shape=q.shape)
                k = torch.nn.functional.layer_norm(
                    input=k.reshape(shape=(-1, head_dim)), normalized_shape=(head_dim,),
                    weight=p_norm_weight, bias=p_norm_bias, eps=normalize_pattern_eps).reshape(shape=k.shape)

        else:
            active_xi = xi.masked_select(mask=update_active_heads).view(size=(-1, *xi.shape[1:]))
            active_k = k.masked_select(mask=update_active_heads).view(size=(-1, *k.shape[1:]))
            q = torch.masked_scatter(input=q, mask=update_active_heads, source=torch.bmm(active_xi, active_k))

        # Optionally scale association heads (each head separately).
        if type(scaling) == float:
            q = q * scaling
        elif type(scaling) == torch.Tensor:
            q = q * scaling.view(1, 1, -1).repeat(repeats=(1, 1, q.shape[2] // scaling.shape[0]))

        if update_step == 0:
            # convert ByteTensor key_padding_mask to bool
            if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for key_padding_mask in nn.HopfieldCore is deprecated. Use bool tensor instead.")
                key_padding_mask = key_padding_mask.to(torch.bool)

            if bias_k is not None and bias_v is not None:
                if static_k is None and static_v is None and key_as_static is None and value_as_static is None:
                    k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                    v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                    if attn_mask is not None:
                        attn_mask = nn.functional.pad(attn_mask, [0, 1])
                    if key_padding_mask is not None:
                        key_padding_mask = nn.functional.pad(key_padding_mask, [0, 1])
                else:
                    assert static_k is None, "bias cannot be added to static key."
                    assert static_v is None, "bias cannot be added to static value."
                    assert not key_as_static, "bias cannot be added to static key."
                    assert not value_as_static, "bias cannot be added to static value."
            else:
                assert bias_k is None
                assert bias_v is None

            q = q.contiguous().view(tgt_len, -1, head_dim).transpose(0, 1)
            if k is not None:
                k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            if v is not None:
                v = v.contiguous().view(v.shape[0], bsz * num_heads, -1).transpose(0, 1)

            if static_k is not None:
                assert static_k.size(0) == bsz * num_heads
                assert static_k.size(2) == head_dim
                k = static_k

            if static_v is not None:
                assert static_v.size(0) == bsz * num_heads
                assert static_v.size(2) == pattern_dim
                v = static_v

            src_len = k.size(1)

            if key_padding_mask is not None:
                assert key_padding_mask.size(0) == bsz
                assert key_padding_mask.size(1) == src_len

            if add_zero_attn:
                src_len += 1
                k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
                v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
                if attn_mask is not None:
                    attn_mask = nn.functional.pad(attn_mask, [0, 1])
                if key_padding_mask is not None:
                    key_padding_mask = nn.functional.pad(key_padding_mask, [0, 1])

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        # Compute new xi for Hopfield retrieve iterations.
        if xi is None:
            xi = nn.functional.softmax(attn_output_weights, dim=-1)
        else:
            xi = torch.masked_scatter(input=xi, mask=update_active_heads, source=nn.functional.softmax(
                attn_output_weights.masked_select(mask=update_active_heads).view(size=(-1, *xi.shape[1:])), dim=-1))

        # Compute threshold-based stopping criterion for Hopfield retrieve iterations.
        with torch.no_grad():
            xi_active = xi.view(size=(bsz, num_heads, tgt_len, src_len))
            update_active_heads = (update_step < update_steps_max) | (update_steps_max < 0)
            if xi_old is not None:
                update_active_heads &= ((xi_old - xi_active).norm(p=2, dim=(2, 3)).max(axis=0)[0]) > update_steps_eps
            update_active_heads = update_active_heads.unsqueeze(dim=1).unsqueeze(dim=2).repeat(repeats=(bsz, 1, 1))
            xi_old = xi_active
        update_step += 1

    ####################################################################################################################
    #                                          END HOPFIELD UPDATE ITERATION                                           #
    ####################################################################################################################

    attn_output_weights = nn.functional.dropout(xi, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.shape[:2]) == [bsz * num_heads, tgt_len]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
    if out_proj_weight is not None:
        assert attn_output.shape[2] == num_heads * pattern_dim
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

    xi = xi.view(bsz, num_heads, tgt_len, src_len) if return_raw_associations else None
    v = v.view(bsz, num_heads, src_len, -1) if return_projected_patterns else None
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads, xi, v
    else:
        return attn_output, None, xi, v
    
class HopfieldCore(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See references: "Hopfield Networks is All You Need" and
                    "Attention Is All You Need" (on which this implementation is partly based on).

    .. math::
        \text{HopfieldHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> hopfield_attn = HopfieldCore(embed_dim, num_heads)
        >>> attn_output, attn_output_weights, attn_matrix = hopfield_attn(query, key, value)
    """
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }

    def __init__(self,
                 embed_dim=None,                  # type: Optional[int]
                 num_heads=1,                     # type: int
                 dropout=0.0,                     # type: float
                 bias=True,                       # type: bool
                 add_bias_kv=False,               # type: bool
                 add_zero_attn=False,             # type: bool
                 kdim=None,                       # type: Optional[int]
                 vdim=None,                       # type: Optional[int]

                 head_dim=None,                   # type: Optional[int]
                 pattern_dim=None,                # type: Optional[int]
                 out_dim=None,                    # type: Optional[int]
                 disable_out_projection=False,    # type: bool
                 key_as_static=False,             # type: bool
                 query_as_static=False,           # type: bool
                 value_as_static=False,           # type: bool
                 value_as_connected=False,        # type: bool
                 normalize_pattern=False,         # type: bool
                 normalize_pattern_affine=False,  # type: bool
                 normalize_pattern_eps=1e-5       # type: float
                 ):
        super(HopfieldCore, self).__init__()

        assert (type(key_as_static) == bool) and (type(query_as_static) == bool) and (type(value_as_static) == bool)
        self.key_as_static, self.query_as_static, self.value_as_static = key_as_static, query_as_static, value_as_static
        num_non_static = 3 - (self.key_as_static + self.query_as_static + self.value_as_static)
        assert 0 <= num_non_static < 4

        self.value_as_connected = value_as_connected
        self.normalize_pattern, self.normalize_pattern_affine = normalize_pattern, normalize_pattern_affine
        self.normalize_pattern_eps = normalize_pattern_eps
        self.disable_out_projection = disable_out_projection

        # In case of a static-only executions, check corresponding projections and normalizations.
        self.static_execution = self._check_execution_mode()
        if self.static_execution:
            embed_dim, kdim, vdim = None, None, None
        if embed_dim is None:
            assert self.static_execution, r'static-only execution requires all projections to be deactivated.'

        # Check and set all other properties, conditioned on <static_execution>.
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = all((
            self.kdim == embed_dim, self.vdim == embed_dim, pattern_dim is None, not self.value_as_connected))
        assert (not self.value_as_connected) or (self.kdim == self.vdim), r'key and value need to be of same dimension.'

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = None
        self.pattern_dim = pattern_dim
        self.virtual_hopfield_dim = None
        self.virtual_pattern_dim = None
        if not self.static_execution:
            if head_dim is None:
                self.head_dim = embed_dim // num_heads
                assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads."
            else:
                assert head_dim > 0, "dimension of the association space has to be positive."
                self.head_dim = head_dim
            if self.pattern_dim is None:
                self.pattern_dim = self.head_dim
            self.virtual_hopfield_dim = self.num_heads * self.head_dim
            self.virtual_pattern_dim = self.num_heads * self.pattern_dim

        self.out_dim = embed_dim if out_dim is None else out_dim
        assert disable_out_projection or (self.out_dim > 0), "output projection dimension has to be positive."

        if normalize_pattern_affine:
            assert normalize_pattern, "affine pattern normalization without pattern normalization has no effect."
            self.p_norm_weight = Parameter(torch.Tensor(head_dim))
            self.p_norm_bias = Parameter(torch.Tensor(head_dim))
        else:
            self.register_parameter('p_norm_weight', None)
            self.register_parameter('p_norm_bias', None)

        if self._qkv_same_embed_dim is False:
            if query_as_static:
                self.register_parameter('q_proj_weight', None)
            else:
                self.q_proj_weight = Parameter(torch.Tensor(self.virtual_hopfield_dim, embed_dim))
            if key_as_static:
                self.register_parameter('k_proj_weight', None)
            else:
                self.k_proj_weight = Parameter(torch.Tensor(self.virtual_hopfield_dim, self.kdim))
            if value_as_static:
                self.register_parameter('v_proj_weight', None)
            else:
                self.v_proj_weight = Parameter(torch.Tensor(
                    self.virtual_pattern_dim,
                    self.virtual_hopfield_dim if (value_as_connected and not key_as_static) else self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            if num_non_static > 0:
                self.in_proj_weight = Parameter(torch.empty(
                    (not query_as_static) * self.virtual_hopfield_dim +
                    (not key_as_static) * self.virtual_hopfield_dim +
                    (not value_as_static) * self.virtual_pattern_dim, embed_dim))
            else:
                self.register_parameter('in_proj_weight', None)
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias and (num_non_static > 0):
            self.in_proj_bias = Parameter(torch.empty(
                (not query_as_static) * self.virtual_hopfield_dim +
                (not key_as_static) * self.virtual_hopfield_dim + self.virtual_pattern_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        if disable_out_projection:
            self.register_parameter('out_proj', None)
        else:
            if bias and _LinearWithBias is not None:
                self.out_proj = _LinearWithBias(self.virtual_pattern_dim, self.out_dim)
            else:
                self.out_proj = Linear(self.virtual_pattern_dim, self.out_dim, bias=bias)

        self.bias_k, self.bias_v = None, None
        if add_bias_kv:
            if not key_as_static:
                self.bias_k = Parameter(torch.empty(1, 1, self.virtual_hopfield_dim))
            if not value_as_static:
                self.bias_v = Parameter(torch.empty(1, 1, self.virtual_hopfield_dim))
            assert not (self.bias_k is None and self.bias_v is None), r'cannot set key/value bias if both are static.'

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def _check_execution_mode(self) -> bool:
        return all((
            self.key_as_static, self.query_as_static, self.value_as_static, not self.value_as_connected,
            not self.normalize_pattern, not self.normalize_pattern_affine, self.disable_out_projection
        ))

    def reset_parameters(self):
        if self.p_norm_weight is not None:
            nn.init.ones_(self.p_norm_weight)
            nn.init.zeros_(self.p_norm_bias)

        if self._qkv_same_embed_dim and (self.in_proj_weight is not None):
            nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
        else:
            if self.q_proj_weight is not None:
                nn.init.normal_(self.q_proj_weight, mean=0.0, std=0.02)
            if self.k_proj_weight is not None:
                nn.init.normal_(self.k_proj_weight, mean=0.0, std=0.02)
            if self.v_proj_weight is not None:
                nn.init.normal_(self.v_proj_weight, mean=0.0, std=0.02)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
        if not self.disable_out_projection:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.normal_(self.bias_k, mean=0.0, std=0.02)
        if self.bias_v is not None:
            nn.init.normal_(self.bias_v, mean=0.0, std=0.02)

    def __setstate__(self, state):
        super(HopfieldCore, self).__setstate__(state)

    def forward(self,
                query,                            # type: Tensor
                key,                              # type: Tensor
                value,                            # type: Tensor
                key_padding_mask=None,            # type: Optional[Tensor]
                need_weights=True,                # type: bool
                attn_mask=None,                   # type: Optional[Tensor]

                scaling=None,                     # type: Optional[Tensor]
                update_steps_max=0,               # type: Optional[int]
                update_steps_eps=1e-4,            # type: float
                return_raw_associations=False,    # type: bool
                return_pattern_projections=False  # type: bool
                ):
        # type: (...) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
                See "Hopfield Networks is All You Need" for more details in the setting of Hopfield networks.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

            scaling: scaling of association heads, often represented as beta (one entry per head).
            update_steps_max: maximum count of association update steps (None equals to infinity).
            update_steps_eps: minimum difference threshold between two consecutive association update steps.
            return_raw_associations: return raw association (softmax) values, unmodified.
            return_pattern_projections: return pattern projection values, unmodified.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.

            - scaling: :math:`(num_heads,)`, where num_heads is the amount of heads.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
            - attn_raw: :math:``(N, num_heads, L, S)`, where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        if self.query_as_static and self.key_as_static:
            assert query.shape[2] == key.shape[2], \
                f'query shape[2] of {query.shape[2]} and key shape[2] of {key.shape[2]} need to be equal'
            head_dim, embed_dim_to_check = query.shape[2], query.shape[2]
        else:
            assert self.query_as_static or (query.shape[2] == self.embed_dim), \
                f'query shape[2] of {query.shape[2]} invalid, needs to be {self.embed_dim}.'
            assert (not self.query_as_static) or (self.query_as_static and query.shape[2] == self.head_dim), \
                f'query shape[2] of {query.shape[2]} invalid, needs to be {self.head_dim}'

            assert self.key_as_static or (key.shape[2] == self.kdim), \
                f'key shape[2] of {key.shape[2]} invalid, needs to be {self.kdim}.'
            assert (not self.key_as_static) or (self.key_as_static and key.shape[2] == self.head_dim), \
                f'key shape[2] of {key.shape[2]} invalid, needs to be {self.head_dim}'
            head_dim, embed_dim_to_check = self.head_dim, self.head_dim if self.query_as_static else self.embed_dim

        assert self.value_as_static or (value.shape[2] == self.vdim), \
            f'value shape[2] of {value.shape[2]} invalid, needs to be {self.vdim}.'
        assert any((
            not self.value_as_static, self.value_as_static and value.shape[2] == self.pattern_dim,
            self.disable_out_projection)
        ), f'value shape[2] of {value.shape[2]} invalid, needs to be {self.pattern_dim}'

        out_weights, out_bias = None, None
        if not self.disable_out_projection:
            out_weights, out_bias = self.out_proj.weight, self.out_proj.bias

        if not self._qkv_same_embed_dim:
            return hopfield_core_forward(
                query=query, key=key, value=value, embed_dim_to_check=embed_dim_to_check, num_heads=self.num_heads,
                in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias, bias_k=self.bias_k,
                bias_v=self.bias_v, add_zero_attn=self.add_zero_attn, dropout_p=self.dropout,
                out_proj_weight=out_weights, out_proj_bias=out_bias, training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,

                key_as_static=self.key_as_static, query_as_static=self.query_as_static,
                value_as_static=self.value_as_static, value_as_connected=self.value_as_connected,
                normalize_pattern=self.normalize_pattern, normalize_pattern_eps=self.normalize_pattern_eps,
                p_norm_weight=self.p_norm_weight, p_norm_bias=self.p_norm_bias,
                head_dim=head_dim, pattern_dim=self.pattern_dim, scaling=scaling,
                update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
                return_raw_associations=return_raw_associations, return_projected_patterns=return_pattern_projections)
        else:
            return hopfield_core_forward(
                query=query, key=key, value=value, embed_dim_to_check=embed_dim_to_check, num_heads=self.num_heads,
                in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias, bias_k=self.bias_k,
                bias_v=self.bias_v, add_zero_attn=self.add_zero_attn, dropout_p=self.dropout,
                out_proj_weight=out_weights, out_proj_bias=out_bias, training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask,

                key_as_static=self.key_as_static, query_as_static=self.query_as_static,
                value_as_static=self.value_as_static, value_as_connected=self.value_as_connected,
                normalize_pattern=self.normalize_pattern, normalize_pattern_eps=self.normalize_pattern_eps,
                p_norm_weight=self.p_norm_weight, p_norm_bias=self.p_norm_bias,
                head_dim=head_dim, pattern_dim=self.pattern_dim, scaling=scaling,
                update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
                return_raw_associations=return_raw_associations, return_projected_patterns=return_pattern_projections)
        

class Hopfield(Module):
    """
    Module with underlying Hopfield association.
    """

    def __init__(self,
                 input_size: Optional[int] = None,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 pattern_size: Optional[int] = None,
                 num_heads: int = 1,
                 scaling: Optional[Union[float, Tensor]] = None,
                 update_steps_max: Optional[Union[int, Tensor]] = 0,
                 update_steps_eps: Union[float, Tensor] = 1e-4,

                 normalize_stored_pattern: bool = True,
                 normalize_stored_pattern_affine: bool = True,
                 normalize_stored_pattern_eps: float = 1e-5,
                 normalize_state_pattern: bool = True,
                 normalize_state_pattern_affine: bool = True,
                 normalize_state_pattern_eps: float = 1e-5,
                 normalize_pattern_projection: bool = True,
                 normalize_pattern_projection_affine: bool = True,
                 normalize_pattern_projection_eps: float = 1e-5,
                 normalize_hopfield_space: bool = False,
                 normalize_hopfield_space_affine: bool = False,
                 normalize_hopfield_space_eps: float = 1e-5,
                 stored_pattern_as_static: bool = False,
                 state_pattern_as_static: bool = False,
                 pattern_projection_as_static: bool = False,
                 pattern_projection_as_connected: bool = False,
                 stored_pattern_size: Optional[int] = None,
                 pattern_projection_size: Optional[int] = None,

                 batch_first: bool = True,
                 association_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 input_bias: bool = True,
                 concat_bias_pattern: bool = False,
                 add_zero_association: bool = False,
                 disable_out_projection: bool = False
                 ):
        """
        Initialise new instance of a Hopfield module.

        :param input_size: depth of the input (state pattern)
        :param hidden_size: depth of the association space
        :param output_size: depth of the output projection
        :param pattern_size: depth of patterns to be selected
        :param num_heads: amount of parallel association heads
        :param scaling: scaling of association heads, often represented as beta (one entry per head)
        :param update_steps_max: maximum count of association update steps (None equals to infinity)
        :param update_steps_eps: minimum difference threshold between two consecutive association update steps
        :param normalize_stored_pattern: apply normalization on stored patterns
        :param normalize_stored_pattern_affine: additionally enable affine normalization of stored patterns
        :param normalize_stored_pattern_eps: offset of the denominator for numerical stability
        :param normalize_state_pattern: apply normalization on state patterns
        :param normalize_state_pattern_affine: additionally enable affine normalization of state patterns
        :param normalize_state_pattern_eps: offset of the denominator for numerical stability
        :param normalize_pattern_projection: apply normalization on the pattern projection
        :param normalize_pattern_projection_affine: additionally enable affine normalization of pattern projection
        :param normalize_pattern_projection_eps: offset of the denominator for numerical stability
        :param normalize_hopfield_space: enable normalization of patterns in the Hopfield space
        :param normalize_hopfield_space_affine: additionally enable affine normalization of patterns in Hopfield space
        :param normalize_hopfield_space_eps: offset of the denominator for numerical stability
        :param stored_pattern_as_static: interpret specified stored patterns as being static
        :param state_pattern_as_static: interpret specified state patterns as being static
        :param pattern_projection_as_static: interpret specified pattern projections as being static
        :param pattern_projection_as_connected: connect pattern projection with stored pattern
        :param stored_pattern_size: depth of input (stored pattern)
        :param pattern_projection_size: depth of input (pattern projection)
        :param batch_first: flag for specifying if the first dimension of data fed to "forward" reflects the batch size
        :param association_activation: additional activation to be applied on the result of the Hopfield association
        :param dropout: dropout probability applied on the association matrix
        :param input_bias: bias to be added to input (state and stored pattern as well as pattern projection)
        :param concat_bias_pattern: bias to be concatenated to stored pattern as well as pattern projection
        :param add_zero_association: add a new batch of zeros to stored pattern as well as pattern projection
        :param disable_out_projection: disable output projection
        """
        super(Hopfield, self).__init__()
        assert type(batch_first) == bool, f'"batch_first" needs to be a boolean, not {type(batch_first)}.'
        assert (association_activation is None) or (type(association_activation) == str)

        # Initialise Hopfield association module.
        self.association_core = HopfieldCore(
            embed_dim=input_size, num_heads=num_heads, dropout=dropout, bias=input_bias,
            add_bias_kv=concat_bias_pattern, add_zero_attn=add_zero_association, kdim=stored_pattern_size,
            vdim=pattern_projection_size, head_dim=hidden_size, pattern_dim=pattern_size, out_dim=output_size,
            disable_out_projection=disable_out_projection, key_as_static=stored_pattern_as_static,
            query_as_static=state_pattern_as_static, value_as_static=pattern_projection_as_static,
            value_as_connected=pattern_projection_as_connected, normalize_pattern=normalize_hopfield_space,
            normalize_pattern_affine=normalize_hopfield_space_affine,
            normalize_pattern_eps=normalize_hopfield_space_eps)
        self.association_activation = None
        if association_activation is not None:
            self.association_activation = getattr(torch, association_activation, None)

        # Initialise stored pattern normalization.
        self.norm_stored_pattern = None
        if normalize_stored_pattern_affine:
            assert normalize_stored_pattern, "affine normalization without normalization has no effect."
        if normalize_stored_pattern:
            normalized_shape = input_size if stored_pattern_size is None else stored_pattern_size
            assert normalized_shape is not None, "stored pattern size required for setting up normalisation"
            self.norm_stored_pattern = nn.LayerNorm(
                normalized_shape=normalized_shape, elementwise_affine=normalize_stored_pattern_affine,
                eps=normalize_stored_pattern_eps)

        # Initialise state pattern normalization.
        self.norm_state_pattern = None
        if normalize_state_pattern_affine:
            assert normalize_state_pattern, "affine normalization without normalization has no effect."
        if normalize_state_pattern:
            assert input_size is not None, "input size required for setting up normalisation"
            self.norm_state_pattern = nn.LayerNorm(
                normalized_shape=input_size, elementwise_affine=normalize_state_pattern_affine,
                eps=normalize_state_pattern_eps)

        # Initialise pattern projection normalization.
        self.norm_pattern_projection = None
        if normalize_pattern_projection_affine:
            assert normalize_pattern_projection, "affine normalization without normalization has no effect."
        if normalize_pattern_projection:
            normalized_shape = input_size if pattern_projection_size is None else pattern_projection_size
            assert normalized_shape is not None, "pattern projection size required for setting up normalisation"
            self.norm_pattern_projection = nn.LayerNorm(
                normalized_shape=normalized_shape, elementwise_affine=normalize_pattern_projection_affine,
                eps=normalize_pattern_projection_eps)

        # Initialise remaining auxiliary properties.
        if self.association_core.static_execution:
            self.__scaling = 1.0 if scaling is None else scaling
        else:
            assert self.association_core.head_dim > 0, f'invalid hidden dimension encountered.'
            self.__scaling = (1.0 / sqrt(self.association_core.head_dim)) if scaling is None else scaling
        self.__batch_first = batch_first
        self.__update_steps_max = update_steps_max
        self.__update_steps_eps = update_steps_eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset Hopfield association.

        :return: None
        """
        for module in (self.association_core, self.norm_stored_pattern,
                       self.norm_state_pattern, self.norm_pattern_projection):
            if hasattr(module, r'reset_parameters'):
                module.reset_parameters()

    def _maybe_transpose(self, *args: Tuple[Tensor, ...]) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Eventually transpose specified data.

        :param args: tensors to eventually transpose (dependent on the state of "batch_first")
        :return: eventually transposed tensors
        """
        transposed_result = tuple(_.transpose(0, 1) for _ in args) if self.__batch_first else args
        return transposed_result[0] if len(transposed_result) == 1 else transposed_result

    def _associate(self, data: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
                   return_raw_associations: bool = False, return_projected_patterns: bool = False,
                   stored_pattern_padding_mask: Optional[Tensor] = None,
                   association_mask: Optional[Tensor] = None) -> Tuple[Optional[Tensor], ...]:
        """
        Apply Hopfield association module on specified data.

        :param data: data to be processed by Hopfield core module
        :param return_raw_associations: return raw association (softmax) values, unmodified
        :param return_projected_patterns: return pattern projection values, unmodified
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: Hopfield-processed input data
        """
        assert (type(data) == Tensor) or ((type(data) == tuple) and (len(data) == 3)), \
            r'either one tensor to be used as "stored pattern", "state pattern" and' \
            r' "pattern_projection" must be provided, or three separate ones.'
        if type(data) == Tensor:
            stored_pattern, state_pattern, pattern_projection = data, data, data
        else:
            stored_pattern, state_pattern, pattern_projection = data

        # Optionally transpose data.
        stored_pattern, state_pattern, pattern_projection = self._maybe_transpose(
            stored_pattern, state_pattern, pattern_projection)

        # Optionally apply stored pattern normalization.
        if self.norm_stored_pattern is not None:
            stored_pattern = self.norm_stored_pattern(input=stored_pattern.reshape(
                shape=(-1, stored_pattern.shape[2]))).reshape(shape=stored_pattern.shape)

        # Optionally apply state pattern normalization.
        if self.norm_state_pattern is not None:
            state_pattern = self.norm_state_pattern(input=state_pattern.reshape(
                shape=(-1, state_pattern.shape[2]))).reshape(shape=state_pattern.shape)

        # Optionally apply pattern projection normalization.
        if self.norm_pattern_projection is not None:
            pattern_projection = self.norm_pattern_projection(input=pattern_projection.reshape(
                shape=(-1, pattern_projection.shape[2]))).reshape(shape=pattern_projection.shape)

        # Apply Hopfield association and optional activation function.
        return self.association_core(
            query=state_pattern, key=stored_pattern, value=pattern_projection,
            key_padding_mask=stored_pattern_padding_mask, need_weights=False, attn_mask=association_mask,
            scaling=self.__scaling, update_steps_max=self.__update_steps_max, update_steps_eps=self.__update_steps_eps,
            return_raw_associations=return_raw_associations, return_pattern_projections=return_projected_patterns)

    def forward(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
                stored_pattern_padding_mask: Optional[Tensor] = None,
                association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Hopfield association on specified data.

        :param input: data to be processed by Hopfield association module
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: Hopfield-processed input data
        """
        association_output = self._maybe_transpose(self._associate(
            data=input, return_raw_associations=False,
            stored_pattern_padding_mask=stored_pattern_padding_mask,
            association_mask=association_mask)[0])
        if self.association_activation is not None:
            association_output = self.association_activation(association_output)
        return association_output


class HopfieldPooling(Module):
    """
    Wrapper class encapsulating a trainable but fixed state pattern and "Hopfield" in
    one combined module to be used as a Hopfield-based pooling layer.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 pattern_size: Optional[int] = None,
                 num_heads: int = 1,
                 scaling: Optional[Union[float, Tensor]] = None,
                 update_steps_max: Optional[Union[int, Tensor]] = 0,
                 update_steps_eps: Union[float, Tensor] = 1e-4,

                 normalize_stored_pattern: bool = True,
                 normalize_stored_pattern_affine: bool = True,
                 normalize_state_pattern: bool = True,
                 normalize_state_pattern_affine: bool = True,
                 normalize_pattern_projection: bool = True,
                 normalize_pattern_projection_affine: bool = True,
                 normalize_hopfield_space: bool = False,
                 normalize_hopfield_space_affine: bool = False,
                 stored_pattern_as_static: bool = False,
                 state_pattern_as_static: bool = False,
                 pattern_projection_as_static: bool = False,
                 pattern_projection_as_connected: bool = False,
                 stored_pattern_size: Optional[int] = None,
                 pattern_projection_size: Optional[int] = None,

                 batch_first: bool = True,
                 association_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 input_bias: bool = True,
                 concat_bias_pattern: bool = False,
                 add_zero_association: bool = False,
                 disable_out_projection: bool = False,
                 quantity: int = 1,
                 trainable: bool = True
                 ):
        """
        Initialise a new instance of a Hopfield-based pooling layer.

        :param input_size: depth of the input (state pattern)
        :param hidden_size: depth of the association space
        :param output_size: depth of the output projection
        :param pattern_size: depth of patterns to be selected
        :param num_heads: amount of parallel association heads
        :param scaling: scaling of association heads, often represented as beta (one entry per head)
        :param update_steps_max: maximum count of association update steps (None equals to infinity)
        :param update_steps_eps: minimum difference threshold between two consecutive association update steps
        :param normalize_stored_pattern: apply normalization on stored patterns
        :param normalize_stored_pattern_affine: additionally enable affine normalization of stored patterns
        :param normalize_state_pattern: apply normalization on state patterns
        :param normalize_state_pattern_affine: additionally enable affine normalization of state patterns
        :param normalize_pattern_projection: apply normalization on the pattern projection
        :param normalize_pattern_projection_affine: additionally enable affine normalization of pattern projection
        :param normalize_hopfield_space: enable normalization of patterns in the Hopfield space
        :param normalize_hopfield_space_affine: additionally enable affine normalization of patterns in Hopfield space
        :param stored_pattern_as_static: interpret specified stored patterns as being static
        :param state_pattern_as_static: interpret specified state patterns as being static
        :param pattern_projection_as_static: interpret specified pattern projections as being static
        :param pattern_projection_as_connected: connect pattern projection with stored pattern
        :param stored_pattern_size: depth of input (stored pattern)
        :param pattern_projection_size: depth of input (pattern projection)
        :param batch_first: flag for specifying if the first dimension of data fed to "forward" reflects the batch size
        :param association_activation: additional activation to be applied on the result of the Hopfield association
        :param dropout: dropout probability applied on the association matrix
        :param input_bias: bias to be added to input (state and stored pattern as well as pattern projection)
        :param concat_bias_pattern: bias to be concatenated to stored pattern as well as pattern projection
        :param add_zero_association: add a new batch of zeros to stored pattern as well as pattern projection
        :param disable_out_projection: disable output projection
        :param quantity: amount of state patterns
        :param trainable: state pattern used for pooling is trainable
        """
        super(HopfieldPooling, self).__init__()
        self.hopfield = Hopfield(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size, pattern_size=pattern_size,
            num_heads=num_heads, scaling=scaling, update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
            normalize_stored_pattern=normalize_stored_pattern,
            normalize_stored_pattern_affine=normalize_stored_pattern_affine,
            normalize_state_pattern=normalize_state_pattern,
            normalize_state_pattern_affine=normalize_state_pattern_affine,
            normalize_pattern_projection=normalize_pattern_projection,
            normalize_pattern_projection_affine=normalize_pattern_projection_affine,
            normalize_hopfield_space=normalize_hopfield_space,
            normalize_hopfield_space_affine=normalize_hopfield_space_affine,
            stored_pattern_as_static=stored_pattern_as_static, state_pattern_as_static=state_pattern_as_static,
            pattern_projection_as_static=pattern_projection_as_static,
            pattern_projection_as_connected=pattern_projection_as_connected, stored_pattern_size=stored_pattern_size,
            pattern_projection_size=pattern_projection_size, batch_first=batch_first,
            association_activation=association_activation, dropout=dropout, input_bias=input_bias,
            concat_bias_pattern=concat_bias_pattern, add_zero_association=add_zero_association,
            disable_out_projection=disable_out_projection)
        self._quantity = quantity
        pooling_weight_size = self.hopfield.hidden_size if state_pattern_as_static else self.hopfield.input_size
        self.pooling_weights = nn.Parameter(torch.empty(size=(*(
            (1, quantity) if batch_first else (quantity, 1)
        ), input_size if pooling_weight_size is None else pooling_weight_size)), requires_grad=trainable)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset pooling weights and underlying Hopfield association.

        :return: None
        """
        if hasattr(self.hopfield, r'reset_parameters'):
            self.hopfield.reset_parameters()

        # Explicitly initialise pooling weights.
        nn.init.normal_(self.pooling_weights, mean=0.0, std=0.02)

    def _prepare_input(self, input: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Prepare input for Hopfield association.

        :param input: data to be prepared
        :return: stored pattern, expanded state pattern as well as pattern projection
        """
        assert (type(input) == Tensor) or ((type(input) == tuple) and (len(input) == 2)), \
            r'either one tensor to be used as "stored pattern" and' \
            r' "pattern_projection" must be provided, or two separate ones.'
        if type(input) == Tensor:
            stored_pattern, pattern_projection = input, input
        else:
            stored_pattern, pattern_projection = input

        batch_size = stored_pattern.shape[0 if self.batch_first else 1]
        return stored_pattern, self.pooling_weights.expand(size=(*(
            (batch_size, self.quantity) if self.batch_first else (self.quantity, batch_size)
        ), self.pooling_weights.shape[2])), pattern_projection

    def forward(self, input: Union[Tensor, Tuple[Tensor, Tensor]], stored_pattern_padding_mask: Optional[Tensor] = None,
                association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute Hopfield-based pooling on specified data.

        :param input: data to be pooled
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: Hopfield-pooled input data
        """
        return self.hopfield(
            input=self._prepare_input(input=input),
            stored_pattern_padding_mask=stored_pattern_padding_mask,
            association_mask=association_mask).flatten(start_dim=1)



class HopfieldLayer(Module):
    """
    Wrapper class encapsulating a trainable but fixed stored pattern, pattern projection and "Hopfield" in
    one combined module to be used as a Hopfield-based pooling layer.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 pattern_size: Optional[int] = None,
                 num_heads: int = 1,
                 scaling: Optional[Union[float, Tensor]] = None,
                 update_steps_max: Optional[Union[int, Tensor]] = 0,
                 update_steps_eps: Union[float, Tensor] = 1e-4,
                 lookup_weights_as_separated: bool = False,
                 lookup_targets_as_trainable: bool = True,

                 normalize_stored_pattern: bool = True,
                 normalize_stored_pattern_affine: bool = True,
                 normalize_state_pattern: bool = True,
                 normalize_state_pattern_affine: bool = True,
                 normalize_pattern_projection: bool = True,
                 normalize_pattern_projection_affine: bool = True,
                 normalize_hopfield_space: bool = False,
                 normalize_hopfield_space_affine: bool = False,
                 stored_pattern_as_static: bool = False,
                 state_pattern_as_static: bool = False,
                 pattern_projection_as_static: bool = False,
                 pattern_projection_as_connected: bool = False,
                 stored_pattern_size: Optional[int] = None,
                 pattern_projection_size: Optional[int] = None,

                 batch_first: bool = True,
                 association_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 input_bias: bool = True,
                 concat_bias_pattern: bool = False,
                 add_zero_association: bool = False,
                 disable_out_projection: bool = False,
                 quantity: int = 1,
                 trainable: bool = True
                 ):
        """
        Initialise a new instance of a Hopfield-based lookup layer.

        :param input_size: depth of the input (state pattern)
        :param hidden_size: depth of the association space
        :param output_size: depth of the output projection
        :param pattern_size: depth of patterns to be selected
        :param num_heads: amount of parallel association heads
        :param scaling: scaling of association heads, often represented as beta (one entry per head)
        :param update_steps_max: maximum count of association update steps (None equals to infinity)
        :param update_steps_eps: minimum difference threshold between two consecutive association update steps
        :param lookup_weights_as_separated: separate lookup weights from lookup target weights
        :param lookup_targets_as_trainable: employ trainable lookup target weights (used as pattern projection input)
        :param normalize_stored_pattern: apply normalization on stored patterns
        :param normalize_stored_pattern_affine: additionally enable affine normalization of stored patterns
        :param normalize_state_pattern: apply normalization on state patterns
        :param normalize_state_pattern_affine: additionally enable affine normalization of state patterns
        :param normalize_pattern_projection: apply normalization on the pattern projection
        :param normalize_pattern_projection_affine: additionally enable affine normalization of pattern projection
        :param normalize_hopfield_space: enable normalization of patterns in the Hopfield space
        :param normalize_hopfield_space_affine: additionally enable affine normalization of patterns in Hopfield space
        :param stored_pattern_as_static: interpret specified stored patterns as being static
        :param state_pattern_as_static: interpret specified state patterns as being static
        :param pattern_projection_as_static: interpret specified pattern projections as being static
        :param pattern_projection_as_connected: connect pattern projection with stored pattern
        :param stored_pattern_size: depth of input (stored pattern)
        :param pattern_projection_size: depth of input (pattern projection)
        :param batch_first: flag for specifying if the first dimension of data fed to "forward" reflects the batch size
        :param association_activation: additional activation to be applied on the result of the Hopfield association
        :param dropout: dropout probability applied on the association matrix
        :param input_bias: bias to be added to input (state and stored pattern as well as pattern projection)
        :param concat_bias_pattern: bias to be concatenated to stored pattern as well as pattern projection
        :param add_zero_association: add a new batch of zeros to stored pattern as well as pattern projection
        :param disable_out_projection: disable output projection
        :param quantity: amount of stored patterns
        :param trainable: stored pattern used for lookup is trainable
        """
        super(HopfieldLayer, self).__init__()
        self.hopfield = Hopfield(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size, pattern_size=pattern_size,
            num_heads=num_heads, scaling=scaling, update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
            normalize_stored_pattern=normalize_stored_pattern,
            normalize_stored_pattern_affine=normalize_stored_pattern_affine,
            normalize_state_pattern=normalize_state_pattern,
            normalize_state_pattern_affine=normalize_state_pattern_affine,
            normalize_pattern_projection=normalize_pattern_projection,
            normalize_pattern_projection_affine=normalize_pattern_projection_affine,
            normalize_hopfield_space=normalize_hopfield_space,
            normalize_hopfield_space_affine=normalize_hopfield_space_affine,
            stored_pattern_as_static=stored_pattern_as_static, state_pattern_as_static=state_pattern_as_static,
            pattern_projection_as_static=pattern_projection_as_static,
            pattern_projection_as_connected=pattern_projection_as_connected, stored_pattern_size=stored_pattern_size,
            pattern_projection_size=pattern_projection_size, batch_first=batch_first,
            association_activation=association_activation, dropout=dropout, input_bias=input_bias,
            concat_bias_pattern=concat_bias_pattern, add_zero_association=add_zero_association,
            disable_out_projection=disable_out_projection)
        self._quantity = quantity
        lookup_weight_size = self.hopfield.hidden_size if stored_pattern_as_static else self.hopfield.stored_pattern_dim
        self.lookup_weights = nn.Parameter(torch.empty(size=(*(
            (1, quantity) if batch_first else (quantity, 1)
        ), input_size if lookup_weight_size is None else lookup_weight_size)), requires_grad=trainable)

        if lookup_weights_as_separated:
            target_weight_size = self.lookup_weights.shape[
                2] if pattern_projection_size is None else pattern_projection_size
            self.target_weights = nn.Parameter(torch.empty(size=(*(
                (1, quantity) if batch_first else (quantity, 1)
            ), target_weight_size)), requires_grad=lookup_targets_as_trainable)
        else:
            self.register_parameter(name=r'target_weights', param=None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset lookup and lookup target weights, including underlying Hopfield association.

        :return: None
        """
        if hasattr(self.hopfield, r'reset_parameters'):
            self.hopfield.reset_parameters()

        # Explicitly initialise lookup and target weights.
        nn.init.normal_(self.lookup_weights, mean=0.0, std=0.02)
        if self.target_weights is not None:
            nn.init.normal_(self.target_weights, mean=0.0, std=0.02)

    def _prepare_input(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Prepare input for Hopfield association.

        :param input: data to be prepared
        :return: stored pattern, expanded state pattern as well as pattern projection
        """
        batch_size = input.shape[0 if self.batch_first else 1]
        stored_pattern = self.lookup_weights.expand(size=(*(
            (batch_size, self.quantity) if self.batch_first else (self.quantity, batch_size)
        ), self.lookup_weights.shape[2]))
        if self.target_weights is None:
            pattern_projection = stored_pattern
        else:
            pattern_projection = self.target_weights.expand(size=(*(
                (batch_size, self.quantity) if self.batch_first else (self.quantity, batch_size)
            ), self.target_weights.shape[2]))

        return stored_pattern, input, pattern_projection

    def forward(self, input: Tensor, stored_pattern_padding_mask: Optional[Tensor] = None,
                association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute Hopfield-based lookup on specified data.

        :param input: data to used in lookup
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: result of Hopfield-based lookup on input data
        """
        return self.hopfield(
            input=self._prepare_input(input=input),
            stored_pattern_padding_mask=stored_pattern_padding_mask,
            association_mask=association_mask)

