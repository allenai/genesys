# coding=utf-8
# Copyright 2022. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BiGS model. """
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.utils import logging
from .configuration_bigs import BiGSConfig

from einops import repeat
from torch.linalg import eigh

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigs"
_CONFIG_FOR_DOC = "BiGSConfig"
_TOKENIZER_FOR_DOC = "BiGSTokenizer"

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

BIGS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "JunxiongWang/BiGS_128",
    "JunxiongWang/BiGS_512",
    "JunxiongWang/BiGS_1024",
    "JunxiongWang/BiGS_4096",
    # See all BiGS models at https://huggingface.co/models?filter=BiGS
]


def log_step_initializer(H=1024, dt_min=0.01, dt_max=1):
    # Generate dt
    log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
    ) + math.log(dt_min)
    return log_dt

try:
    import pykeops
    from pykeops.torch import Genred
    has_pykeops = True
    print("Pykeops installation found.")

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
        return tensors

    def cauchy_keops(v, z, w):
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'

        cauchy_mult = Genred(
            f'ComplexDivide({expr_num}, {expr_denom})',
            [
                'v = Vj(2)',
                'z = Vi(2)',
                'w = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2*cauchy_mult(v, z, w, backend='GPU')
        return _r2c(r)

    def log_vandermonde_keops(v, x, L):
        expr = 'ComplexMult(v, ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'v = Vj(2)',
                'x = Vj(2)',
                'l = Vi(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        v, x, l = _broadcast_dims(v, x, l)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(v, x, l, backend='GPU')
        return _r2c(r).real

    def log_vandermonde_transpose_keops(u, v, x, L):
        """
        u: ... H L
        v: ... H N
        x: ... H N
        Returns: ... H N

        V = Vandermonde(a, L) : (H N L)
        contract_L(V * u * v)
        """
        expr = 'ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'u = Vj(2)',
                'v = Vi(2)',
                'x = Vi(2)',
                'l = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        u, v, x, l = _broadcast_dims(u, v, x, l)
        u = _c2r(u)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(u, v, x, l, backend='GPU')
        return _r2c(r)

except ImportError:
    has_pykeops = False
    print("Switch to torch vandermonde kernel.")

class BiGSEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# S4D Kernel module
class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
        dt_min, dt_max ==> initialize log steps
    """

    def __init__(self, N=64, use_pykeops_kernel=False, dt_min=0.01, dt_max=1):
        super().__init__()

        log_step = log_step_initializer(1, dt_min, dt_max)
        self.C = nn.Parameter(torch.normal(0, 0.5 ** 0.5, (N, 2)))

        A_re = -0.5 * torch.ones(N)
        A_im = math.pi * torch.arange(N)

        self.register_parameter("log_step", nn.Parameter(log_step))
        self.register_parameter("A_re", nn.Parameter(A_re))
        self.register_parameter("A_im", nn.Parameter(A_im))
        self.use_pykeops_kernel = use_pykeops_kernel

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """
        # Materialize parameters
        dt = torch.exp(self.log_step)  # (H)
        A = torch.clamp(self.A_re, None, -1e-4) + 1j * self.A_im
        C = (self.C[..., 0] + 1j * self.C[..., 1]).unsqueeze(0)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1) # (H N)
        C = C * (torch.exp(dtA)-1.) / A

        if has_pykeops and self.use_pykeops_kernel:
            K = log_vandermonde_keops(C, dtA, L)
        else:
            K = dtA.unsqueeze(-1) * torch.arange(L, device=dtA.device)  # (H N L)
            K = torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        
        return K


## S4D module
class S4dLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # the simplest ssm
        self.N = config.num_ssm
        self.D = nn.Parameter(torch.randn(1))
        self.kernel = S4DKernel(N=self.N, use_pykeops_kernel=config.use_pykeops_kernel)

    def forward(self, u):
        """ Input shape (B, L, H) """
        """ Output shape (B, L, H) """
        # convert into (B H L)
        u = u.transpose(-1, -2)
        L = u.size(-1)  # u is the input
        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)
        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)
        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D
        # convert back to B, H, L
        y = y.transpose(-1, -2)
        return y


class BiGSLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_ssm = config.num_ssm
        self.max_seq_length = config.max_position_embeddings
        self.pre_norm = config.pre_norm
        self.decode = config.decode
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # ssm layers
        self.fs4 = S4dLayer(config)
        self.bs4 = S4dLayer(config)
        # dense layers
        self.dv = nn.Linear(config.hidden_size, config.intermediate_size)
        self.du_forward = nn.Linear(config.hidden_size, config.hidden_size)
        self.du_backward = nn.Linear(config.hidden_size, config.hidden_size)
        self.duc_forward = nn.Linear(config.hidden_size, config.hidden_size)
        self.duc_backward =  nn.Linear(config.hidden_size, config.hidden_size)
        self.dol = nn.Linear(config.hidden_size, config.intermediate_size)
        self.do = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(
        self,
        hidden_states
    ):
        hidden_residual = hidden_states
        hidden_states =  self.LayerNorm(hidden_states)
        # gating
        v = nn.functional.gelu(self.dv(hidden_states))
        u_forward = nn.functional.gelu(self.du_forward(hidden_states))
        u_backward = nn.functional.gelu(self.du_backward(torch.flip(hidden_states, dims=[1])))
        # s4 layers
        fs4_output = self.fs4(u_forward)
        bs4_output = self.bs4(u_backward)
        # instead of sum, we use multiplication
        uc_forward = self.duc_forward(fs4_output)
        uc_backward = torch.flip(self.duc_backward(bs4_output), dims=[1])
        hidden_states = self.do(nn.functional.gelu(self.dol(uc_forward * uc_backward)) * v)
        hidden_states = hidden_residual + hidden_states
        return hidden_states


class BiGSEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BiGSLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                hidden_states = layer_module(
                    hidden_states,
                )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states
                ]
                if v is not None
            )
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
