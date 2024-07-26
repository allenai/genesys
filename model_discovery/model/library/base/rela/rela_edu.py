# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

import func
from models import model
from utils import util, dtype


# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import func
from utils import util, dtype


def dot_attention(query, memory, mem_mask, hidden_size,
                  ln=False, num_heads=1, cache=None, dropout=None,
                  out_map=True, scope=None):
    """
    dotted attention model
    :param query: [batch_size, qey_len, dim]
    :param memory: [batch_size, seq_len, mem_dim] or None
    :param mem_mask: [batch_size, seq_len]
    :param hidden_size: attention space dimension
    :param ln: whether use layer normalization
    :param num_heads: attention head number
    :param dropout: attention dropout, default disable
    :param out_map: output additional mapping
    :param cache: cache-based decoding
    :param scope:
    :return: a value matrix, [batch_size, qey_len, mem_dim]
    """
    with tf.variable_scope(scope or "dot_attention", reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx())):
        if memory is None:
            # suppose self-attention from queries alone
            h = func.linear(query, hidden_size * 3, ln=ln, scope="qkv_map")
            q, k, v = tf.split(h, 3, -1)

            if cache is not None:
                k = tf.concat([cache['k'], k], axis=1)
                v = tf.concat([cache['v'], v], axis=1)
                cache = {
                    'k': k,
                    'v': v,
                }
        else:
            q = func.linear(query, hidden_size, ln=ln, scope="q_map")
            if cache is not None and ('mk' in cache and 'mv' in cache):
                k, v = cache['mk'], cache['mv']
            else:
                k = func.linear(memory, hidden_size, ln=ln, scope="k_map")
                v = func.linear(memory, hidden_size, ln=ln, scope="v_map")

            if cache is not None:
                cache['mk'] = k
                cache['mv'] = v

        q = func.split_heads(q, num_heads)
        k = func.split_heads(k, num_heads)
        v = func.split_heads(v, num_heads)

        q *= (hidden_size // num_heads) ** (-0.5)

        # q * k => attention weights
        logits = tf.matmul(q, k, transpose_b=True)

        # convert the mask to 0-1 form and multiply to logits
        if mem_mask is not None:
            zero_one_mask = tf.to_float(tf.equal(mem_mask, 0.0))
            logits *= zero_one_mask

        # replace softmax with relu
        # weights = tf.nn.softmax(logits)
        weights = tf.nn.relu(logits)

        dweights = util.valid_apply_dropout(weights, dropout)

        # weights * v => attention vectors
        o = tf.matmul(dweights, v)
        o = func.combine_heads(o)

        # perform RMSNorm to stabilize running
        o = gated_rms_norm(o, scope="post")

        if out_map:
            o = func.linear(o, hidden_size, ln=ln, scope="o_map")

        results = {
            'weights': weights,
            'output': o,
            'cache': cache
        }

        return results


def gated_rms_norm(x, eps=None, scope=None):
    """RMS-based Layer normalization layer"""
    if eps is None:
        eps = dtype.epsilon()
    with tf.variable_scope(scope or "rms_norm",
                           dtype=tf.as_dtype(dtype.floatx())):
        layer_size = util.shape_list(x)[-1]

        scale = tf.get_variable("scale", [layer_size], initializer=tf.ones_initializer())
        gate = tf.get_variable("gate", [layer_size], initializer=None)

        ms = tf.reduce_mean(x ** 2, -1, keep_dims=True)

        # adding gating here which slightly improves quality
        return scale * x * tf.rsqrt(ms + eps) * tf.nn.sigmoid(gate * x)
    


def decoder(target, state, params):
    mask = dtype.tf_to_float(tf.cast(target, tf.bool))
    hidden_size = params.hidden_size
    initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

    is_training = ('decoder' not in state)

    if is_training:
        target, mask = util.remove_invalid_seq(target, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "tgt_embedding"
    tgt_emb = tf.get_variable(embed_name,
                              [params.tgt_vocab.size(), params.embed_size],
                              initializer=initializer)
    tgt_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(tgt_emb, target) * (hidden_size ** 0.5)
    inputs = tf.nn.bias_add(inputs, tgt_bias)

    # shift
    if is_training:
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])
        inputs = inputs[:, :-1, :]
        inputs = func.add_timing_signal(inputs)
    else:
        inputs = tf.cond(tf.reduce_all(tf.equal(target, params.tgt_vocab.pad())),
                         lambda: tf.zeros_like(inputs),
                         lambda: inputs)
        mask = tf.ones_like(mask)
        inputs = func.add_timing_signal(inputs, time=dtype.tf_to_float(state['time']))

    inputs = util.valid_apply_dropout(inputs, params.dropout)

    with tf.variable_scope("decoder"):
        x = inputs
        for layer in range(params.num_decoder_layer):
            if params.deep_transformer_init:
                layer_initializer = tf.variance_scaling_initializer(
                    params.initializer_gain * (layer + 1) ** -0.5,
                    mode="fan_avg",
                    distribution="uniform")
            else:
                layer_initializer = None
            with tf.variable_scope("layer_{}".format(layer), initializer=layer_initializer):
                with tf.variable_scope("self_attention"):
                    y = dot_attention(
                        x,
                        None,
                        func.attention_bias(tf.shape(mask)[1], "causal"),
                        hidden_size,
                        num_heads=params.num_heads,
                        dropout=params.attention_dropout,
                        cache=None if is_training else
                        state['decoder']['state']['layer_{}'.format(layer)]
                    )
                    if not is_training:
                        # k, v
                        state['decoder']['state']['layer_{}'.format(layer)] \
                            .update(y['cache'])

                    y = y['output']
                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)

                with tf.variable_scope("feed_forward"):
                    y = func.ffn_layer(
                        x,
                        params.filter_size,
                        hidden_size,
                        dropout=params.relu_dropout,
                    )

                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)
    feature = x
    if 'dev_decode' in state:
        feature = x[:, -1, :]

    embed_name = "tgt_embedding" if params.shared_target_softmax_embedding \
        else "softmax_embedding"
    embed_name = "embedding" if params.shared_source_target_embedding \
        else embed_name
    softmax_emb = tf.get_variable(embed_name,
                                  [params.tgt_vocab.size(), params.embed_size],
                                  initializer=initializer)
    feature = tf.reshape(feature, [-1, params.embed_size])
    logits = tf.matmul(feature, softmax_emb, False, True)

    logits = tf.cast(logits, tf.float32)

    soft_label, normalizer = util.label_smooth(
        target,
        util.shape_list(logits)[-1],
        factor=params.label_smooth)
    centropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=soft_label
    )
    centropy -= normalizer
    centropy = tf.reshape(centropy, tf.shape(target))

    mask = tf.cast(mask, tf.float32)
    per_sample_loss = tf.reduce_sum(centropy * mask, -1) / tf.reduce_sum(mask, -1)
    loss = tf.reduce_mean(per_sample_loss)

    # these mask tricks mainly used to deal with zero shapes, such as [0, 1]
    loss = tf.cond(tf.equal(tf.shape(target)[0], 0),
                   lambda: tf.constant(0, tf.float32),
                   lambda: loss)

    return loss, logits, state, per_sample_loss


