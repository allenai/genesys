# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size




class ParallelKerpleLog(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
        self,
        neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        
        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size
        
        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

        self.cached_matrix = None
        self.cached_seq_len = None
    
    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd
    
    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p*torch.log(1+self.bias_a*diff) # log kernel
        
        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])

        return x + bias


class ParallelKerplePower(torch.nn.Module):
    """Kernelized Alibi Relative Position Bias parallelized in the heads dimension"""

    def __init__(
        self,
        neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        
        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size
        
        # Allocate weights and initialize.
        # bias_kernel = -bias_a*|m-n|^bias_p
        # weight_kernel = exp(-wei_a*|m-n|^wei_p)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        self.bias_a, self.bias_p, self.wei_a, self.wei_p = None, None, None, None
        
        if self.pos_emb.endswith('original'):
            slopes = torch.Tensor(self._get_slopes(self.heads))[
                self.model_parallel_rank * self.num_heads_per_partition : (self.model_parallel_rank + 1) * self.num_heads_per_partition
            ][:,None,None]
            slopes = slopes.to(torch.cuda.current_device()).to(neox_args.params_dtype)
            self.bias_a = Parameter(slopes, requires_grad=False)
        else:
            bias_arg, wei_arg = self.pos_emb.split('_')[-2:]
            self.bias_p = get_parameter(2, 'uniform') if 'p' in bias_arg else None
            self.bias_a = get_parameter(1, 'uniform') if 'a' in bias_arg else None
            self.wei_p = get_parameter(2, 'uniform') if 'p' in wei_arg else None
            self.wei_a = get_parameter(1, 'uniform') if 'a' in wei_arg else None

        self.cached_matrix = None
        self.cached_seq_len = None
    
    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            dd.update(get_stats('bias_a', self.bias_a))
        if self.bias_p is not None:
            self.bias_p.data = self.bias_p.data.clamp(min=self.eps, max=2)
            dd.update(get_stats('bias_p', self.bias_p))
        if self.wei_a is not None:
            self.wei_a.data = self.wei_a.data.clamp(min=self.eps)
            dd.update(get_stats('wei_a', self.wei_a))
        if self.wei_p is not None:
            self.wei_p.data = self.wei_p.data.clamp(min=self.eps, max=2)
            dd.update(get_stats('wei_p', self.wei_p))
        return dd
    
    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        
        # get bias matrix
        if self.bias_p is None and self.bias_a is None:
            bias = 0.0
        else:
            if self.bias_p is not None:
                self.bias_p.data = self.bias_p.data.clamp(min=self.eps, max=2)
                bias = diff.pow(self.bias_p)
            else:
                bias = diff
            if self.bias_a is not None:
                self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
                bias = -bias*self.bias_a
            else:
                bias = -bias

        # get weight matrix
        if self.wei_p is None and self.wei_a is None:
            wei = 1.0
        else:
            if self.wei_p is not None:
                self.wei_p.data = self.wei_p.data.clamp(min=self.eps, max=2)
                wei = diff.pow(self.wei_p)
            else:
                wei = diff
            if self.wei_a is not None:
                self.wei_a.data = self.wei_a.data.clamp(min=self.eps)
                wei = (-wei*self.wei_a).exp()
            else:
                wei = (-wei).exp()
        
        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])
            if type(wei) != float:
                wei = wei[:, seq_len_k - 1, :].view(wei.shape[0], 1, wei.shape[2])

        return x*wei + bias

