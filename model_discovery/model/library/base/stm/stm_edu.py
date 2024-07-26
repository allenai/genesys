import abc
import inspect
import warnings

import torch
from makefun import create_function
from torch import nn
from transformers import PreTrainedModel, AutoModel, WEIGHTS_NAME
from transformers.utils import logging

from .modules import MockTensorForConditionalGeneration
from .utils import _extract_keyword_args, _fix_args, _fix_kwargs, _stack_args, _stack_kwargs

from .configuration_sled import SledConfig

logger = logging.get_logger(__name__)


PREFIX_KEY = 'prefix_length'


def _unstack_encoder_outputs(stacked_output, n, bs):
    if isinstance(stacked_output, tuple):
        return [tuple(v if not isinstance(v, torch.Tensor) else v[i * bs:(i + 1) * bs] for v in stacked_output)
                for i in range(n)]
    # works for dict as well as structured outputs
    return [type(stacked_output)(**{k: v if not isinstance(v, torch.Tensor) else v[i*bs:(i+1)*bs]
                                    for k, v in stacked_output.items()})
            for i in range(n)]


def _slice_tensor(v, start, end, prefix_length=None):
    prefix_length = prefix_length or 0
    return v[:, start+prefix_length:end+prefix_length]


def _merge_encoder_outputs(encoder_outputs_list):
    # a list of 4-tuples, first value is the returned value from the encoder, then the start and end indices inside the
    # tensors that we should take, and finally prefix_length (None if was not used)

    # presumed order of returned tuple from encoders: last_hidden_state, hidden_states, attentions

    # the first output, as returned by the underlying model on the first window
    resulting_output = encoder_outputs_list[0][0]
    if isinstance(resulting_output, tuple):  # not in return dict mode:
        resulting_list = []
        for i in range(len(resulting_output)):
            if resulting_output[i] is None:
                resulting_list.append(None)
            elif (
                    isinstance(resulting_output[i], (int, float, tuple, MockTensorForConditionalGeneration))
                    or resulting_output[i].dim() != 3
            ):
                continue
            else:
                assert isinstance(resulting_output[i], torch.Tensor)
                # tensors are of of size (N, w, d), N the batch size, w the current window size and d the hidden
                # state size/logits dimension these are the only parts in the encoder output that we need to merge
                # between windows
                resulting_list.append(
                    torch.cat(tuple(_slice_tensor(out[i], start, end, prefix_length)
                                    for out, start, end, prefix_length in encoder_outputs_list), dim=1)
                )  # requires extra GPU memory because it doesn't dump the old copy of the tensors yet
        resulting_output = tuple(resulting_list)
    else:
        for key in resulting_output.keys():
            if resulting_output[key] is None:
                continue
            if isinstance(resulting_output[key], tuple):
                resulting_output[key] = None  # encoder outputs are not tuples, only the decoders
            else:
                assert isinstance(resulting_output[key], torch.Tensor)
                if resulting_output[key].dim() != 3:
                    continue  # decoder outputs may be 4d tensors
                # tensors are of of size (N, w, d), N the batch size, w the current window size and d the hidden
                # state size/logits dimension
                resulting_output[key] = torch.cat(
                    tuple(_slice_tensor(out[key], start, end, prefix_length)
                          for out, start, end, prefix_length in encoder_outputs_list), dim=1
                )

    return resulting_output


class MockDecoder(nn.Module):
    def forward(self, *_, **__):
        return tuple()

    def to(self, *_, **__):
        return self


class SledPretrainedModel(PreTrainedModel, metaclass=abc.ABCMeta):
    config_class = SledConfig
    auto_model_loader = AutoModel
    IGNORE_CONFIG_KEYS = {'model_type', '_name_or_path'}  # config keys we allow to be mismatched between the
    # SledConfig and the underlying model's config

    def __init__(self, underlying_model: PreTrainedModel, config: SledConfig):
        """

        :param underlying_model: The backbone model to use.
                                Warning - once given, it should not be used directly, as it may cause unexpected behaviours
        :param config:
        """
        super(SledPretrainedModel, self).__init__(config)
        self._underlying_model = (
            underlying_model  # crucial this will be before any calls to members that is in the base model
        )

        self._window_fraction = config.window_fraction
        self._context_size = config.context_size
        self._window_margin = int(config.context_size * (1 - config.window_fraction) / 2)
        self._sliding_method = config.sliding_method or 'dynamic'
        assert self._sliding_method in {'loop', 'stacked', 'dynamic', 'decoderonly'}

        for override_member in ['is_parallelizable', 'supports_gradient_checkpointing']:
            setattr(self, override_member, getattr(underlying_model, override_member))

        # setting the base_model_prefix to return the correct underlying model and link to some methods
        # implemented in the base
        self.base_model_prefix = "sled_base_model_prefix"
        self.sled_base_model_prefix = self._underlying_model.base_model

        # override generation preparation functions that may be overridden by underlying models but will be
        # found in our wrapper. We wished we could do it a follows:
        # for method_name, _ in inspect.getmembers(PreTrainedModel, predicate=inspect.isfunction):
        #     if method_name not in {"_replicate_for_data_parallel", 'modules'}:
        #         setattr(self, method_name, getattr(underlying_model, method_name))
        # However, the above is too broad and dangerous, so we will do it directly
        for method_name in {"_init_weights", "prepare_inputs_for_generation"}:
            if hasattr(underlying_model, method_name):
                setattr(self, method_name, getattr(underlying_model, method_name))

        # set the resize_token_embeddings
        vocab_size = underlying_model.get_input_embeddings().weight.size(0)
        assert hasattr(self.config, 'vocab_size'), 'Underlying models must have a vocab_size config'
        assert underlying_model.config.vocab_size == vocab_size
        self.resize_token_embeddings(vocab_size)  # the underlying model may have a different vocab size compared to its base config

        self._verified_config_consistency = False
        self._verify_config_consistency()
        self._verified_config_consistency = False  # We would like to do it later again (before the first forward)

        self._prepend_prefix = config.prepend_prefix
        self._encode_prefix = config.encode_prefix

        # now, let's create the forward function
        self._create_forward_function()

    @property
    def underlying_model(self):
        return self._underlying_model

    def resize_token_embeddings(self, new_num_tokens=None):
        res = self.underlying_model.resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = self.underlying_model.vocab_size  # sync them
        return res

    @property
    def _ignore_keys(self):
        return self.IGNORE_CONFIG_KEYS

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica.forward = create_function(self._signature, replica._forward)
        return replica

    def _create_forward_function(self):
        # https://stackoverflow.com/a/15638720
        self._underlying_model_signature = inspect.signature(self._underlying_model.forward)
        self._forward_kwargs_names = [param.name for param in self._underlying_model_signature.parameters.values()]
        assert PREFIX_KEY not in self._forward_kwargs_names
        # if we want to prepend questions in every window, we need to set the forward signature to expect the
        # input_prefix (e.g. question) as a separate input sequence

        # we want to remove any typing information as it may cause issues in the custom function build do to
        # non-imported modules. It is ugly and shouldn't be done like that, but it works..
        params = [self._underlying_model_signature.parameters[p].replace(annotation=inspect.Parameter.empty)
                  for p in self._underlying_model_signature.parameters]
        params.append(inspect.Parameter(name=PREFIX_KEY, default=None, kind=params[-1].kind))
        self._signature = str(self._underlying_model_signature.replace(parameters=params,
                                                                       return_annotation=inspect.Signature.empty))

        # HF trainer uses the signature to choose which parts to take from a dataset, so we need to make sure our
        # wrapped forward function has the correct signature (dynamically creating it here)
        self.forward = create_function(self._signature, self._forward)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            try:
                return self._underlying_model.__getattribute__(item)
            except AttributeError:
                return self._underlying_model.__getattr__(item)

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        # the actual forward implementation of the model.
        raise NotImplementedError

    def _set_underlying_model_attr(self, attr_name, new_val):
        if hasattr(self._underlying_model, attr_name):
            setattr(self._underlying_model, attr_name, new_val)
        elif hasattr(self._underlying_model, "model") and hasattr(self._underlying_model.model, attr_name):
            setattr(self._underlying_model.model, attr_name, new_val)
        else:
            raise ValueError(f"Cannot use this model as we cannot set its {attr_name}")

    def _run_sliding_window_forward(self, args_tensor_inds, kwargs_tensor_keys, s, *args,
                                    prefix_length=None, **kwargs):
        sm = self._sliding_method if self._sliding_method != 'dynamic' else \
            ('loop' if not self.training else 'stacked')
        try:
            if sm == 'decoderonly':
                return self._skip_forward_for_decoder_only(args_tensor_inds, kwargs_tensor_keys, s, *args,
                                                           prefix_length=prefix_length, **kwargs)
            if sm == 'loop':
                return self._run_sliding_window_forward_loop(args_tensor_inds, kwargs_tensor_keys, s, *args,
                                        prefix_length=prefix_length, **kwargs)
            return self._run_sliding_window_forward_stacked(args_tensor_inds, kwargs_tensor_keys, s, *args,
                                                         prefix_length=prefix_length, **kwargs)
        finally:
            # so that if the model crashes halfway through it will be restored to working order
            pass

    def _skip_forward_for_decoder_only(self, args_tensor_inds, kwargs_tensor_keys, s, *args,
                                       prefix_length=None, **kwargs):
        # NOTE - this will work probably only with BART.
        embeder = self if hasattr(self, 'embed_tokens') else self.get_encoder() # account for sled encoder
        return (embeder.embed_tokens(kwargs['input_ids']), )


    def _run_sliding_window_forward_loop(self, args_tensor_inds, kwargs_tensor_keys, s, *args,
                                    prefix_length=None, **kwargs):
        forward_kwargs = _extract_keyword_args(kwargs, self._forward_kwargs_names, None)
        encoder_outputs_list = []
        if prefix_length is not None and self._prepend_prefix:
            # we were given prefixes in the input, and we are expected to treat them
            prefix_length, s = self._handle_prefix(prefix_length, s)

            if self._encode_prefix:
                # encode the question as well, if needed
                context_start_ind, context_end_ind, update_start_ind, update_end_ind = 0, prefix_length, 0, prefix_length

                encoder_outputs = self._underlying_model.forward(
                    *_fix_args(args, args_tensor_inds, context_start_ind, context_end_ind, None),
                    **_fix_kwargs(forward_kwargs, kwargs_tensor_keys, context_start_ind, context_end_ind, None),
                )
                encoder_outputs_list.append((encoder_outputs, update_start_ind, update_end_ind, None))
                # we will need to make sure all input tensors will also drop everything with the prefix
        else:
            prefix_length = None  # we need to ignore the prefix and treat the entire input as one long document

        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in self._window_indices(s):
            encoder_outputs = self._underlying_model.forward(
                *_fix_args(args, args_tensor_inds, context_start_ind, context_end_ind, prefix_length),
                **_fix_kwargs(forward_kwargs, kwargs_tensor_keys, context_start_ind, context_end_ind, prefix_length),
            )
            encoder_outputs_list.append((encoder_outputs, update_start_ind, update_end_ind, prefix_length))

        return _merge_encoder_outputs(encoder_outputs_list)

    def _handle_prefix(self, prefix_length, s):
        prefix_length_ = prefix_length[0].detach().cpu().item()
        assert torch.all(prefix_length == prefix_length_).item(), \
            'Using different length prefixes in the same batch is not supported. Either group your batch by ' \
            'prefix length, or pad the prefixes to match in length (and do not forget to set the attention ' \
            'mask to 0 where appropriate)'
        if hasattr(self.underlying_model.config, 'max_position_embeddings'):
            assert self._context_size + prefix_length_ <= self.underlying_model.config.max_position_embeddings, \
                f'The prefix length + SLEDs chunk size must be at most the max length that the backbone model can handle'
        return prefix_length_, s-prefix_length_

    def _run_sliding_window_forward_stacked(self, args_tensor_inds, kwargs_tensor_keys, s, *args,
                                    prefix_length=None, **kwargs):
        forward_kwargs = _extract_keyword_args(kwargs, self._forward_kwargs_names, None)
        stacks_args = []
        stacks_kwargs = []
        stacks_info = []

        if prefix_length is not None and self._prepend_prefix:
            # we were given prefixes in the input, and we are expected to treat them
            prefix_length, s = self._handle_prefix(prefix_length, s)

            if self._encode_prefix:
                # encode the question as well, if needed
                context_start_ind, context_end_ind, update_start_ind, update_end_ind = 0, prefix_length, 0, prefix_length
                # need to pad it to match the seq len of the rest
                # we may have too short samples as well so don't want to pad too much
                pad = min(s, self._context_size)
                assert pad >= 0, f'We have a weird situation. pad={pad}, s={s}, ' \
                                 f'prefix_length={prefix_length} and self._context_size={self._context_size}'
                stacks_args.append(_fix_args(args, args_tensor_inds, context_start_ind, context_end_ind, None, pad))
                stacks_kwargs.append(_fix_kwargs(forward_kwargs, kwargs_tensor_keys, context_start_ind,
                                                 context_end_ind, None, pad))
                stacks_info.append([None, update_start_ind, update_end_ind, None])
        else:
            prefix_length = None  # we need to ignore the prefix and treat the entire input as one long document

        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in self._window_indices(s):
            stacks_args.append(_fix_args(args, args_tensor_inds, context_start_ind, context_end_ind, prefix_length))
            stacks_kwargs.append(_fix_kwargs(forward_kwargs, kwargs_tensor_keys, context_start_ind,
                                             context_end_ind, prefix_length))
            stacks_info.append([None, update_start_ind, update_end_ind, prefix_length])

        encoder_outputs2 = self._underlying_model.forward(
            *_stack_args(stacks_args, args_tensor_inds),
            **_stack_kwargs(stacks_kwargs, kwargs_tensor_keys))
        bs = forward_kwargs[kwargs_tensor_keys[0]].size()[0] if len(kwargs_tensor_keys) > 0 else \
            args[args_tensor_inds[0]].size()[0]
        for si, eo in zip(stacks_info, _unstack_encoder_outputs(encoder_outputs2, len(stacks_info), bs)):
            si[0] = eo
        res = _merge_encoder_outputs(stacks_info)

        return res

    def _window_indices(self, total_seq_len):
        """
        when total_seq_len is smaller than our desired context length, we do not do sliding window at all.
        However, if it is longer, then we ALWAYS require the context length to be maximal, even if some windows have
        a lot of overlap.
        Also, first window will always update from the start, and last window will always update until the end.
        when applied, returns a generator that in each iteration produces for numbers:
        context_start_ind, context_end_ind, update_start_ind, update_end_ind

        context_start_ind, context_end_ind are indices in [0, total_seq_len],
        where context_end_ind > context_start_ind and when
        total_seq_len <= context_length then always context_end_ind = context_start_ind+context_length.
        The sequence of context_start_ind is strictly monotonic and same for context_end_ind.
        context_start_ind always start in 0 and
        context_end_ind will always end in total_seq_len.
        Gives us what token indices to take from the long input.

        update_start_ind, update_end_ind are indices in [0, min(total_seq_len, context_length)],
        where update_end_ind > update_start_ind
        and for all windows that are not in the edges (i.e. first/last window) we have
        update_end_ind-update_start_ind=context_length*window_fraction.
        For first window update_start_ind is always 0, and for last window,
        update_end_ind is always min(total_seq_len, context_length).
        They represents the start and end indices from the selected window of
        which tokens should be taken out for the final encoding

        When doing a full itartion, accounting for the fact that
        update_start_ind, update_end_ind are shifted by context_start_ind, we hould get that all indices in
        [0, total_seq_len] were covered exactly once

        Examples
        >>> from transformers import T5Tokenizer, T5Model
        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model_ = T5Model.from_pretrained('t5-small')
        >>> model = SledModel(model_, 512)  # testing with padding of 50% and context of 512

        >>> list(model._window_indices(256))  # List of: (context_start, context_end, update_start, update_end). short sequence
        [(0, 256, 0, 256)]
        >>> list(model._window_indices(510))  # another short sequence
        [(0, 510, 0, 510)]
        >>> list(model._window_indices(512))  # sequence of exactly the context size
        [(0, 512, 0, 512)]
        >>> list(model._window_indices(514))  # sequence of slightly more than the context size
        [(0, 512, 0, 384), (2, 514, 382, 512)]
        >>> list(model._window_indices(766))  # long sequence that does not require a full stride (update in the last chunk is smaller than what is possible)
        [(0, 512, 0, 384), (254, 766, 130, 512)]
        >>> list(model._window_indices(768))  # long sequence for exactly two perfect chunks
        [(0, 512, 0, 384), (256, 768, 128, 512)]
        >>> list(model._window_indices(780))  # very long sequence that does not require a full stride (update in the last chunk is smaller than what is possible)
        [(0, 512, 0, 384), (256, 768, 128, 384), (268, 780, 372, 512)]
        >>> windows = list(model._window_indices(1050))
        >>> windows
        [(0, 512, 0, 384), (256, 768, 128, 384), (512, 1024, 128, 384), (538, 1050, 358, 512)]
        >>> windows = sum([list(range(us+cs, ue+cs)) for cs, _, us, ue in windows], [])  # verify it covers exactly all the indices, each once
        >>> windows[:10]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> windows[500:510]
        [500, 501, 502, 503, 504, 505, 506, 507, 508, 509]
        >>> len(windows)
        1050
        >>> len(set(windows))
        1050
        >>> model = SledModel(model_, 256, window_fraction=0.75)  # now testing with padding of 25% and context of 256

        >>> list(model._window_indices(128))  # List of: (context_start, context_end, update_start, update_end). short sequence
        [(0, 128, 0, 128)]
        >>> list(model._window_indices(254))  # another short sequence
        [(0, 254, 0, 254)]
        >>> list(model._window_indices(256))  # sequence of exactly the context size
        [(0, 256, 0, 256)]
        >>> list(model._window_indices(258))  # sequence of slightly more than the context size. margin is 256/8 -> 32
        [(0, 256, 0, 224), (2, 258, 222, 256)]
        >>> list(model._window_indices(446))  # long sequence that does not require a full stride (update in the last chunk is smaller than what is possible). stride should be 256-64=192
        [(0, 256, 0, 224), (190, 446, 34, 256)]
        >>> list(model._window_indices(448))  # long sequence for exactly two perfect chunks
        [(0, 256, 0, 224), (192, 448, 32, 256)]
        >>> list(model._window_indices(500))  # very long sequence that does not require a full stride (update in the last chunk is smaller than what is possible)
        [(0, 256, 0, 224), (192, 448, 32, 224), (244, 500, 172, 256)]
        >>> windows = list(model._window_indices(1050))
        >>> windows
        [(0, 256, 0, 224), (192, 448, 32, 224), (384, 640, 32, 224), (576, 832, 32, 224), (768, 1024, 32, 224), (794, 1050, 198, 256)]
        >>> windows = sum([list(range(us+cs, ue+cs)) for cs, _, us, ue in windows], [])  # verify it covers exactly all the indices, each once
        >>> windows[:10]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> windows[500:510]
        [500, 501, 502, 503, 504, 505, 506, 507, 508, 509]
        >>> len(windows)
        1050
        >>> len(set(windows))
        1050
        """
        if total_seq_len <= self._context_size:
            yield 0, total_seq_len, 0, total_seq_len
        else:
            stride = self._context_size - 2 * self._window_margin
            context_start = update_start_ind = 0
            context_end = self._context_size
            update_end_ind = context_end - self._window_margin
            yield context_start, context_end, update_start_ind, update_end_ind  # first window always should update from the beginning
            while context_end < total_seq_len:
                context_end = min(total_seq_len, context_end + stride)
                context_start = (
                    context_start + stride if context_end < total_seq_len else total_seq_len - self._context_size
                )
                update_start_ind = max(update_start_ind + stride, update_end_ind)
                # last window always should update until the end
                update_end_ind = (
                    min(total_seq_len, update_end_ind + stride) if context_end < total_seq_len else total_seq_len
                )

                cs, ce, us, ue = context_start, context_end, update_start_ind - context_start, \
                                 update_end_ind - context_start

                yield cs, ce, us, ue

    def _fill_prefix_inputs(self, kwargs, kwargs_tensor_keys):
        prefix_inputs = {}
        k = PREFIX_KEY
        if PREFIX_KEY in kwargs:
            if self._prepend_prefix:
                if k not in kwargs_tensor_keys:
                    warnings.warn(f'{k} is missing from kwargs_tensor_keys (though expected for SLED prefix prepending)')
                else:
                    kwargs_tensor_keys.remove(k)
                    prefix_inputs[k] = kwargs.pop(k)
            elif k in kwargs_tensor_keys:
                warnings.warn(f'{k} is given in kwargs_tensor_keys even though sled should not prepend prefix, '
                              f'that would mean the prefix would be ignored and the entire input will be treated '
                              f'as a single long document, which is probably not what you meant')
        return prefix_inputs

    @staticmethod
    def _prep_attention_mask_for_cross_attention(encode_prefix, attention_mask, prefix_length=None):
        # if we need to drop the prefix encodings, we also need to adjust the attention mask before decoding
        if not encode_prefix and prefix_length is not None:
            prefix_length = int(prefix_length[0])
            return attention_mask[..., prefix_length:]
        return attention_mask


class SledModel(SledPretrainedModel):
    """
    >>> from transformers import T5Tokenizer, T5Model, BartModel, BartTokenizer

    >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
    >>> model_ = T5Model.from_pretrained('t5-small')
    >>> model = SledModel(model_, 4)

    >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
    >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
    >>> outputs = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone())
    >>> outputs = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=True)
    >>> outputs = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=False)
    """

    def __init__(self, underlying_model: PreTrainedModel, config: SledConfig):
        super(SledModel, self).__init__(underlying_model, config)
        # validate the model can be used
        self._decoder_attr_name = getattr(underlying_model, "get_decoder_attr_name", lambda: "decoder")()
        self._encoder_attr_name = getattr(underlying_model, "get_encoder_attr_name", lambda: "encoder")()
        self._set_underlying_model_attr(self._decoder_attr_name, self.get_decoder())
        self._mock_decoder = MockDecoder()
        assert "return_dict" in self._forward_kwargs_names
        assert "encoder_outputs" in self._forward_kwargs_names

    def _forward(self, *args, **kwargs):
        self._verify_config_consistency()
        kwargs, args = _fill_kwargs_with_args(self._forward_kwargs_names, *args, **kwargs)
        kwargs.setdefault("encoder_outputs", None)
        return_dict = kwargs.setdefault("return_dict", None)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        labels = kwargs.get("labels", None)
        kwargs["labels"] = None
        kwargs["return_dict"] = False
        kwargs.setdefault("labels", None)
        args_tensor_inds, kwargs_tensor_keys, s = _find_tensor_inds_and_size(*args, **kwargs)
        prefix_inputs = self._fill_prefix_inputs(kwargs, kwargs_tensor_keys)

        forward_kwargs = _extract_keyword_args(kwargs, self._forward_kwargs_names, None)
        if forward_kwargs["encoder_outputs"] is None:
            # encode, but first let's set decoder to be a mock, no reason to apply it over partial windows
            self._prep_for_encoding()  # todo - add try catch every time we 'prep' something to rever the state on fail?

            forward_kwargs["encoder_outputs"] = self._run_sliding_window_forward(
                args_tensor_inds, kwargs_tensor_keys, s, *args, **prefix_inputs, **forward_kwargs
            )
            forward_kwargs['attention_mask'] = self._prep_attention_mask_for_cross_attention(self._encode_prefix,
                forward_kwargs['attention_mask'], prefix_inputs.get('prefix_length', None))

        # now, let's decode
        forward_kwargs["return_dict"] = return_dict
        forward_kwargs["labels"] = labels
        self._fix_post_encoding()
        if 'decoder_input_ids' in self._forward_kwargs_names and \
            forward_kwargs.get('decoder_input_ids', None) is None and \
                hasattr(self, 'prepare_decoder_input_ids_from_labels') :
            logger.warning('Passing a batch through the model without the decoder_input_ids is likely to cause issues. '
                           'If you encounter cuda errors, make sure you use the prepare_decoder_input_ids_from_labels '
                           'function of the model correctly before passing the input. '
                           'If you are only performing prediction without training, you can safely ignore this message')
        res = self._underlying_model.forward(
            *args, **_extract_keyword_args(forward_kwargs, self._forward_kwargs_names)
        )

        return res

    def _prep_for_encoding(self):
        if not getattr(self, '_preped_for_encoding', False):
            self._preped_for_encoding = True
            self._decoder = self.get_decoder()
            self._mock_decoder.first_device = getattr(self._decoder, "first_device", None)
            self._set_underlying_model_attr(self._decoder_attr_name, self._mock_decoder)

    def _fix_post_encoding(self):
        assert self._preped_for_encoding
        self._preped_for_encoding = False
        self._set_underlying_model_attr(self._decoder_attr_name, self._decoder)
