import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = GPT2(embed_dim=embed_dim, block_loc=block_loc,
            kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


import torch.nn.functional as F
from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl


class GPT2(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.mha = SelectiveGatedMHA(embed_dim=self.embed_dim, block_loc=
            self.block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs,
            **self.kwarg_all)
        self.mlp = GatedMLP(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm1 = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.norm2 = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)

    def _forward(self, X, **Z):
        X1, Z = self.norm1(X, **Z)
        X2, Z = self.mha(X1, **Z)
        X = X + X2
        X3, Z = self.norm2(X, **Z)
        X4, Z = self.mlp(X3, **Z)
        X = X + X4
        return X, Z


import torch.nn.functional as F


class GatedMLP(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, hidden_features=None, out_features=None,
        activation=None, bias=False, multiple_of=128, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        out_features = out_features if out_features is not None else embed_dim
        hidden_features = (hidden_features if hidden_features is not None else
            int(8 * embed_dim / 3))
        hidden_features = (hidden_features + multiple_of - 1
            ) // multiple_of * multiple_of
        self.fc1 = nn.Linear(embed_dim, 2 * hidden_features, bias=bias, **
            self.factory_kwargs)
        self.activation = activation if activation is not None else F.silu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **
            self.factory_kwargs)

    def _forward(self, X, **Z):
        y = self.fc1(X)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


import torch.nn.functional as F
from torch import Tensor


class RMSNorm(GAUBase):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    This layer applies a variant of layer normalization that uses only the root mean square
    statistics, without centering. It's computationally more efficient than standard
    layer normalization and has been shown to be effective in various NLP tasks.

    Args:
        embed_dim (int): The size of the input feature dimension.
        block_loc (tuple): The location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the parent class.
        device (torch.device, optional): The device on which to allocate the module's parameters.
        dtype (torch.dtype, optional): The dtype of the module's parameters.
        eps (float, optional): A small constant added to the denominator for numerical stability.
            Default: 1e-5.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter of shape (embed_dim,).
        variance_epsilon (float): The epsilon value used in the normalization formula.

    Shape:
        - Input: (*, embed_dim)
        - Output: (*, embed_dim) (same shape as input)

    Examples:
        >>> rmsnorm = RMSNorm(128, (0, 6), {})
        >>> x = torch.randn(1, 100, 128)
        >>> output = rmsnorm(x)
        >>> print(output.shape)
        torch.Size([1, 100, 128])

    References:
        - Paper: "Root Mean Square Layer Normalization" by Biao Zhang and Rico Sennrich
          https://arxiv.org/abs/1910.07467
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, eps=1e-05, **kwargs):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.weight = nn.Parameter(torch.ones(embed_dim, **self.factory_kwargs)
            )
        self.variance_epsilon = eps

    def _forward(self, X, **Z):
        input_dtype = X.dtype
        X = X.to(torch.float32)
        variance = X.pow(2).mean(-1, keepdim=True)
        X = X * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * X.to(input_dtype)


import torch.nn.functional as F
import math


class SelectiveGatedMHA(GAUBase):
    """
    SelectiveGatedMHA: Hierarchical Selective Attention with Dynamic Parameter Generation

    This module implements a Multi-Head Attention mechanism with selective gating and dynamic parameter generation.
    It introduces content-dependent gating to selectively focus computation on important inputs and dynamically generates
    parameters based on the input content. It also incorporates hierarchical memory management for efficient processing
    of long sequences.

    **Key Components:**
    - **SelectiveGate**: Computes importance scores and generates binary gates to select important inputs.
    - **DynamicParamGen**: Generates dynamic parameters conditioned on the input content.
    - **HierMemManager**: Manages memory efficiently by processing inputs in blocks.

    **Args:**
        embed_dim (int): The embedding dimension of the input.
        block_loc (tuple): The location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the module.
        device (torch.device, optional): The device to allocate parameters to.
        dtype (torch.dtype, optional): The data type of parameters.
        num_heads (int, optional): Number of attention heads. Default: 8.
        head_dim (int, optional): Dimension of each attention head. If None, calculated as embed_dim // num_heads.
        block_size (int, optional): Size of blocks for hierarchical memory management. Default: 64.

    **Inputs:**
        X (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

    **Outputs:**
        Y (Tensor): Output tensor of shape (batch_size, seq_length, embed_dim).
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_heads: int=8, head_dim: int=None,
        block_size: int=64, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.num_heads = num_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.block_size = block_size
        self.selective_gate = SelectiveGate(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.param_gen = DynamicParamGen(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.mem_manager = HierMemManager(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self.to_qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, **self.
            factory_kwargs)
        self.to_out = nn.Linear(self.embed_dim, self.embed_dim, **self.
            factory_kwargs)

    def _forward(self, X, **Z):
        B, L, D = X.shape
        input_dtype = X.dtype
        X = X.to(**self.factory_kwargs)
        _, Z_ = self.selective_gate(X, **Z)
        Z.update(Z_)
        gates = Z_['gates'].to(**self.factory_kwargs)
        _, Z_ = self.param_gen(X, **Z)
        Z.update(Z_)
        params = Z_['params'].to(**self.factory_kwargs)
        _, Z_ = self.mem_manager(X, **Z)
        Z.update(Z_)
        X_blocks = Z_['X_blocks']
        L_orig = Z_['L_orig']
        block_size = Z_['block_size']
        num_blocks = X_blocks.size(1)
        outputs = []
        past_k = []
        past_v = []
        cumulative_len = 0
        for block_idx in range(num_blocks):
            X_block = X_blocks[:, block_idx, :, :]
            block_seq_len = X_block.size(1)
            block_start = block_idx * block_size
            block_end = min(block_start + block_seq_len, L_orig)
            seq_in_block = block_end - block_start
            qkv = self.to_qkv(X_block[:, :seq_in_block])
            qkv = qkv.chunk(3, dim=-1)
            q, k, v = [t.view(B, seq_in_block, self.num_heads, self.
                head_dim) for t in qkv]
            block_gates = gates[:, block_start:block_end, :, :]
            block_params = params[:, block_start:block_end, :, :]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            block_gates = block_gates.transpose(1, 2)
            block_params = block_params.transpose(1, 2)
            k = k * block_gates
            v = v * block_params * block_gates
            all_k = torch.cat(past_k + [k], dim=2)
            all_v = torch.cat(past_v + [v], dim=2)
            attn_scores = torch.matmul(q, all_k.transpose(-2, -1)) / math.sqrt(
                self.head_dim)
            total_len = cumulative_len + seq_in_block
            causal_mask = torch.tril(torch.ones(seq_in_block, total_len,
                device=X.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0)
                .unsqueeze(0), float('-inf'))
            attn = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn, all_v)
            out = out.transpose(1, 2).contiguous().view(B, seq_in_block,
                self.embed_dim)
            outputs.append(out)
            past_k.append(k)
            past_v.append(v)
            cumulative_len += seq_in_block
        Y = torch.cat(outputs, dim=1)[:, :L_orig, :]
        Y = self.to_out(Y)
        Y = Y.to(dtype=input_dtype)
        return Y, Z


import torch.nn.functional as F


class HierMemManager(GAUBase):
    """
    HierMemManager Module

    Processes input X in blocks for efficient memory management, particularly useful for handling long sequences.

    **Args:**
        embed_dim (int): The embedding dimension of the input.
        block_loc (tuple): The location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the module.
        device (torch.device, optional): The device to allocate parameters to.
        dtype (torch.dtype, optional): The data type of parameters.
        block_size (int, optional): Size of blocks for hierarchical memory management.

    **Inputs:**
        X (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

    **Outputs:**
        X_blocks (Tensor): Blocked input tensor of shape (batch_size, num_blocks, block_size, embed_dim).
        L_orig (int): Original sequence length before padding.
        block_size (int): Block size used for blocking.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, block_size: int=64, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.block_size = block_size

    def _forward(self, X, **Z):
        B, L, D = X.shape
        X = X.to(**self.factory_kwargs)
        pad_len = (self.block_size - L % self.block_size) % self.block_size
        if pad_len > 0:
            X_padded = F.pad(X, (0, 0, 0, pad_len))
        else:
            X_padded = X
        L_padded = X_padded.size(1)
        num_blocks = L_padded // self.block_size
        X_blocks = X_padded.view(B, num_blocks, self.block_size, D)
        return X, {'X_blocks': X_blocks, 'L_orig': L, 'block_size': self.
            block_size}


class DynamicParamGen(GAUBase):
    """
    DynamicParamGen Module

    Generates dynamic parameters based on input X, enabling content-dependent parameter generation for the attention mechanism.

    **Args:**
        embed_dim (int): The embedding dimension of the input.
        block_loc (tuple): The location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the module.
        device (torch.device, optional): The device to allocate parameters to.
        dtype (torch.dtype, optional): The data type of parameters.
        num_heads (int, optional): Number of attention heads. Default: 8.
        head_dim (int, optional): Dimension of each attention head.

    **Inputs:**
        X (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

    **Outputs:**
        params (Tensor): Dynamic parameters tensor of shape (batch_size, seq_length, num_heads, head_dim).
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_heads: int=8, head_dim: int=None, **kwargs
        ):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.num_heads = num_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.param_proj = nn.Linear(embed_dim, self.num_heads * self.
            head_dim, **self.factory_kwargs)

    def _forward(self, X, **Z):
        X = X.to(**self.factory_kwargs)
        params = self.param_proj(X)
        params = params.view(X.size(0), X.size(1), self.num_heads, self.
            head_dim)
        return X, {'params': params.to(**self.factory_kwargs)}


import torch.nn.functional as F


class SelectiveGate(GAUBase):
    """
    SelectiveGate Module

    Computes importance scores and generates binary gates based on input X, allowing the model to selectively focus
    computation on important inputs.

    **Args:**
        embed_dim (int): The embedding dimension of the input.
        block_loc (tuple): The location of this block in the model architecture.
        kwarg_all (dict): Additional keyword arguments passed to the module.
        device (torch.device, optional): The device to allocate parameters to.
        dtype (torch.dtype, optional): The data type of parameters.
        num_heads (int, optional): Number of attention heads. Default: 8.

    **Inputs:**
        X (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

    **Outputs:**
        gates (Tensor): Binary gate tensor of shape (batch_size, seq_length, num_heads, 1).
        scores (Tensor): Importance scores tensor of shape (batch_size, seq_length, num_heads).
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_heads: int=8, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.num_heads = num_heads
        self.gate_proj = nn.Linear(embed_dim, num_heads, **self.factory_kwargs)
        self.threshold = nn.Parameter(torch.zeros(1, **self.factory_kwargs))

    def _forward(self, X, **Z):
        B, L, _ = X.shape
        X = X.to(**self.factory_kwargs)
        scores = torch.sigmoid(self.gate_proj(X))
        temperature = 1.1
        hard_gates = (scores > self.threshold).float()
        soft_gates = torch.sigmoid((scores - self.threshold) / temperature)
        gates = hard_gates.detach() + soft_gates - soft_gates.detach()
        gates = gates.view(B, L, self.num_heads, 1)
        return X, {'gates': gates.to(**self.factory_kwargs), 'scores':
            scores.to(**self.factory_kwargs)}


gab_config = {'num_heads': 8, 'softmax_scale': None, 'out_proj_bias': True,
    'n_heads': 8, 'num_heads_kv': None, 'd_conv': 0, 'mlp_dim': 0,
    'head_dim': None, 'causal': True, 'qkv_proj_bias': True,
    'rotary_emb_base': 10000, 'max_seq_len': 4096, 'block_size': 64, 'eps':
    1e-05, 'bias': False, 'multiple_of': 128, 'hidden_features': None,
    'out_features': None, 'activation': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)