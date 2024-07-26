import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    MultiheadAttention,
)

import talkconv_cuda

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class TaLKConvolutionEncoderFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input_x, offset_left, offset_right, max_left, max_right):
    output = talkconv_cuda.talk_convolution_encoder_forward(input_x, offset_left, offset_right, max_left, max_right)

    ctx.save_for_backward(input_x, offset_left, offset_right)
    ctx.max_left = max_left
    ctx.max_right = max_right

    return output

  @staticmethod
  @amp.float_function
  def backward(ctx, grad_output):
    input_x, offset_left, offset_right = ctx.saved_tensors
    max_left = ctx.max_left
    max_right = ctx.max_right

    retval = talkconv_cuda.talk_convolution_encoder_backward(input_x, offset_left, offset_right, max_left, max_right, grad_output.contiguous())

    return tuple([retval[0], retval[1], retval[2], None, None])


class TaLKConvolutionDecoderFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input_x, offset_left, max_left):
    output = talkconv_cuda.talk_convolution_decoder_forward(input_x, offset_left, max_left)

    ctx.save_for_backward(input_x, offset_left)
    ctx.max_left = max_left

    return output

  @staticmethod
  @amp.float_function
  def backward(ctx, grad_output):
    input_x, offset_left = ctx.saved_tensors
    max_left = ctx.max_left

    retval = talkconv_cuda.talk_convolution_decoder_backward(input_x, offset_left, max_left, grad_output.contiguous())

    return tuple([retval[0], retval[1], None])
  

class TaLKConv(nn.Module):
  def __init__(self, hid_dim, offsets_dropout=0.0, decode=False, num_heads=1, min_len_left=1, min_len_right=1):
    super().__init__()

    self.hid_dim = hid_dim
    self.decode = decode

    self.num_heads = num_heads
    self.R = self.hid_dim // self.num_heads

    self.min_len_left = min_len_left
    self.min_len_right = min_len_right

    if not self.decode:
        self.offsets = nn.Linear(self.hid_dim, self.num_heads * 2, bias=True)
    else:
        self.offsets = nn.Linear(self.hid_dim, self.num_heads, bias=True)

    self.do = nn.Dropout(offsets_dropout)


  def forward(self, x, incremental_state=None, mask=None):

    _, B, C = x.size()
    H = self.num_heads
    R = C // H
    K = self.min_len_left + self.min_len_right + 1


    if incremental_state is not None:
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is None:
            input_buffer = x * (1/K)
        else:
            input_buffer = torch.cat([input_buffer, (x * (1/K)) + input_buffer[-1:]], dim=0)

        self._set_input_buffer(incremental_state, input_buffer)
        x_sum = input_buffer.view(-1, B*H, R)

        T = x.shape[0]
    else:
        T = x.shape[0]

        x_sum = torch.cumsum(x.view(T, B*H, R)*(1/K), 0)


    x_offsets = torch.sigmoid(self.offsets(x))
    x_offsets = self.do(x_offsets)

    if not self.decode:
        x_offset_left, x_offset_right = x_offsets[:,:,:H].contiguous().view(T, B*H), x_offsets[:,:,H:].contiguous().view(T, B*H)
    else:
        x_offset_left = x_offsets.view(T, B*H)


    if incremental_state is not None:
        x_output = talkconv_cuda.talk_convolution_decoder_inference_forward(x_sum, x_offset_left.squeeze(-1), self.min_len_left)
    else:
        if not self.decode:
            x_output = TaLKConvolutionEncoderFunction.apply(x_sum, x_offset_left.squeeze(-1), x_offset_right.squeeze(-1), self.min_len_left, self.min_len_right)
        else:
            x_output = TaLKConvolutionDecoderFunction.apply(x_sum, x_offset_left.squeeze(-1), self.min_len_left)


    x_output = x_output.view(T, B, C)

    return x_output

  def _get_input_buffer(self, incremental_state):
    return utils.get_incremental_state(self, incremental_state, 'input_buffer')

  def _set_input_buffer(self, incremental_state, new_buffer):
    return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)

  def reorder_incremental_state(self, incremental_state, new_order):
    input_buffer = self._get_input_buffer(incremental_state)
    if input_buffer is not None:
      input_buffer = input_buffer.index_select(1, new_order)
      self._set_input_buffer(incremental_state, input_buffer)


class TaLKConvDecoder(FairseqIncrementalDecoder):
    """
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TaLKConvDecoderLayer(args, no_encoder_attn, kernel_size=args.decoder_kernel_size_list[i])
            for i in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

        self.acts_reg = []

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class TaLKConvDecoderLayer(nn.Module):
    """Decoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        kernel_size: kernel size of the convolution
    """

    def __init__(self, args, no_encoder_attn=False, kernel_size=1):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.conv_dim = args.decoder_conv_dim
        if args.decoder_glu:
            self.linear1 = Linear(self.embed_dim, 2*self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None

        self.conv = TaLKConv(self.conv_dim, offsets_dropout=args.weight_dropout, decode=True, num_heads=args.decoder_attention_heads, min_len_left=kernel_size, min_len_right=0)

        self.linear2 = Linear(self.conv_dim, self.embed_dim)

        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.input_dropout = args.input_dropout
        self.normalize_before = args.decoder_normalize_before

        self.conv_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout, encoder_decoder_attention=True
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state,
                prev_conv_state=None, prev_attn_state=None, conv_mask=None,
                conv_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.conv_layer_norm, x, before=True)
        if prev_conv_state is not None:
            if incremental_state is None:
                incremental_state = {}
            self.conv._set_input_buffer(incremental_state, prev_conv_state)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)

        x = self.conv(x, incremental_state=incremental_state)

        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.conv_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = swish(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return 'dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}'.format(
            self.dropout, self.relu_dropout, self.input_dropout, self.normalize_before)



def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

def swish(x, beta=1):
    return x * torch.sigmoid(beta * x)
