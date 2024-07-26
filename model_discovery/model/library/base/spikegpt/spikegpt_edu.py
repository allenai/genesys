from src.spikingjelly.clock_driven.surrogate import ATan as atan
from src.spikingjelly.clock_driven import functional, surrogate, lava_exchange
from src.spikingjelly.clock_driven.neuron import LIFNode, check_backend

from typing import Callable

try:
    from src.spikingjelly.clock_driven import neuron_kernel
except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    neuron_kernel = None


# from src.spikingjelly_n.activation_based import neuron,functional,surrogate
import math, os
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
#import pdb

from src.rwkv import RWKV_Init, RWKV_ChannelMix, RWKV_TimeMix

try:
    from deepspeed.ops.adam import FusedAdam
except:
    pass  # some poor windows users cant install deepspeed

logger = logging.getLogger(__name__)

RWKV_HEAD_QK_DIM = 0
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)




class MultiStepLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6):
        """
        * :ref:`API in English <MultiStepLIFNode.__init__-en>`

        .. _MultiStepLIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param decay_input: 输入是否会衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param backend: 使用哪种计算后端，可以为 ``'torch'`` 或 ``'cupy'``。``'cupy'`` 速度更快，但仅支持GPU。
        :type backend: str

        多步版本的 :class:`spikingjelly.clock_driven.neuron.LIFNode`。

        .. tip::

            对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
            使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepLIFNode.__init__-cn>`

        .. _MultiStepLIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        The multi-step version of :class:`spikingjelly.clock_driven.neuron.LIFNode`.

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.

        """
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)

        check_backend(backend)

        self.backend = backend

        self.lava_s_cale = lava_s_cale

        if backend == 'lava':
            self.lava_neuron = self.to_lava()
        else:
            self.lava_neuron = None

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input, self.tau, self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)

            return spike

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

    def to_lava(self):
        return lava_exchange.to_lava_neuron(self)

    def reset(self):
        super().reset()
        if self.lava_neuron is not None:
            self.lava_neuron.current_state.zero_()
            self.lava_neuron.voltage_state.zero_()


########################################################################################################
# The GPT Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.lif1 = MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0), backend='cupy',
                                            v_threshold=1.)
        self.lif2 = MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0), backend='cupy',
                                            v_threshold=1.)
        self.dropout = nn.Dropout(0.03)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, 0)
        else:
            self.att = RWKV_TimeMix(config, layer_id)

        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            x = x + self.lif1(self.ffnPre(self.ln1(x)).permute(1, 0, 2)).permute(1, 0, 2)  # better in some cases
        else:
            x = x + self.lif1(self.att(self.ln1(x)).permute(1, 0, 2)).permute(1, 0, 2)
        x = x + self.lif2(self.ffn(self.ln2(x)).permute(1, 0, 2)).permute(1, 0, 2)
        x = self.dropout(x)

        return x

