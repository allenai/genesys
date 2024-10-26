import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GABBase


class GAB(GABBase):

    def __init__(self, embed_dim: int, block_loc: tuple, device=None, dtype
        =None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc)
        self.root = TTT(embed_dim=embed_dim, block_loc=block_loc, kwarg_all
            =kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z):
        X, Z = self.root(X, **Z)
        return X, Z


import torch.nn.functional as F
from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.utils import logging


class TTT(GAUBase):
    """
    Problem Statement
This paper addresses the challenge of long context in recurrent neural networks (RNNs). While RNNs offer linear computational complexity, their performance suffers in long sequences due to the limited expressive power of their fixed-size hidden states. This limitation contrasts with Transformers, which excel in long-context scenarios but have quadratic complexity.

Main Claims
The paper proposes a new class of sequence modeling layers called Test-Time Training (TTT) layers that offer both linear complexity and expressive hidden states.
The key idea is to make the hidden state a machine learning model itself, where the update rule is a step of self-supervised learning. This allows for continuous training of the hidden state even on test sequences.
The paper introduces two instantiations of TTT layers: TTT-Linear, with a linear model as the hidden state, and TTT-MLP, with a two-layer multi-layer perceptron (MLP) as the hidden state.
Both TTT-Linear and TTT-MLP demonstrate competitive performance compared to strong Transformer and Mamba (a modern RNN) baselines across various model sizes.
Unlike Mamba, both TTT layers show a continuous decrease in perplexity as they condition on more tokens in long sequences.
TTT-Linear, with preliminary systems optimization, is faster than Transformers at 8k context and matches Mamba in wall-clock time.
Methodology
The paper introduces TTT layers, which use a self-supervised learning approach to update the hidden state. The update rule is effectively a gradient step on a self-supervised loss function, allowing for "training" of the hidden state at test time. Two implementations are explored: TTT-Linear, where the hidden state is a linear model, and TTT-MLP, where the hidden state is a two-layer MLP. The paper also proposes mini-batch TTT and a dual form to improve hardware efficiency and speed up computations.

Key Results
In short-context (2k and 8k tokens) experiments on the Pile dataset, both TTT-Linear and TTT-MLP demonstrate performance comparable to or exceeding Mamba and Transformer baselines.
In long-context (1k to 32k tokens) experiments on the Books3 subset of the Pile, both TTT-Linear and TTT-MLP outperform Mamba, especially at longer context lengths.
TTT-Linear with the Mamba backbone outperforms both Mamba and Transformers with the Transformer backbone across various model sizes.
With preliminary systems optimization, TTT-Linear is already faster than Transformers at 8k context and matches Mamba in wall-clock time.
TTT-MLP shows potential for even better performance in long-context scenarios but currently faces challenges in memory I/O.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.hidden_size = embed_dim
        kwarg_all['num_attention_heads'] = max(4, embed_dim // 64)
        self.seq_modeling_block = TTTLinear(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        kwarg_all['intermediate_size'] = int(embed_dim * 2.5)
        self.mlp = SwiGluMLP(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.conv = Conv(embed_dim=self.embed_dim, block_loc=self.block_loc,
            kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.seq_norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.ffn_norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)

    def _forward(self, X, **Z):
        hidden_states = X
        position_ids = torch.arange(0, X.shape[1], dtype=torch.long, device
            =X.device).unsqueeze(0)
        residual = hidden_states
        hidden_states = self.conv(hidden_states, **Z)[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.seq_norm(hidden_states, **Z)[0]
        Z['position_ids'] = position_ids
        hidden_states = self.seq_modeling_block(hidden_states, **Z)[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states, **Z)[0]
        hidden_states = self.mlp(hidden_states, **Z)[0]
        hidden_states = residual + hidden_states
        return hidden_states


import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.utils import logging
from transformers.activations import ACT2FN


class SwiGluMLP(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, intermediate_size=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.hidden_size = embed_dim
        self.intermediate_size = (intermediate_size if intermediate_size is not
            None else int(embed_dim * 2.5))
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,
            bias=False, **self.factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,
            bias=False, **self.factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,
            bias=False, **self.factory_kwargs)
        self.act_fn = ACT2FN['silu']

    def _forward(self, X, **Z):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(X)) * self.
            up_proj(X))
        return down_proj


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
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils._pytree import tree_map
from transformers.utils import logging


class TTTLinear(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, num_attention_heads=4,
        scan_checkpoint_group_size=4, conv_kernel=4, mini_batch_size=16,
        rope_theta=10000.0, ttt_base_lr=1.0, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.num_heads = num_attention_heads
        self.width = embed_dim
        self.hidden_size = embed_dim
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = mini_batch_size
        self.rope_theta = rope_theta
        self.ttt_base_lr = ttt_base_lr
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1, **self.
            factory_kwargs)
        self.register_buffer('token_idx', token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.
            mini_batch_size,), **self.factory_kwargs))
        self.conv_kernel = conv_kernel
        self._init_qkvo_proj()
        self.rope_theta = self.rope_theta
        kwargs['dim'] = self.head_dim
        kwargs['max_position_embeddings'] = self.mini_batch_size
        kwargs['base'] = self.rope_theta
        self.rotary_emb = RotaryEmbedding(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
        self._init_ttt_lr_gate()
        self._init_ttt_ln()
        self.post_norm = nn.LayerNorm(self.width, eps=1e-06, **self.
            factory_kwargs)
        self.scan_checkpoint_group_size = scan_checkpoint_group_size
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads,
            self.head_dim, self.head_dim), **self.factory_kwargs))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim,
            **self.factory_kwargs))

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def permute_qk(self, q, k):
        bsz, num_head, seq_len, head_dim = q.shape
        q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4
            ).reshape(bsz, num_head, seq_len, head_dim)
        k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4
            ).reshape(bsz, num_head, seq_len, head_dim)
        return q, k

    def undo_permute_qk(self, q, k):
        bsz, num_head, seq_len, head_dim = q.shape
        q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4
            ).reshape(bsz, num_head, seq_len, head_dim)
        k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4
            ).reshape(bsz, num_head, seq_len, head_dim)
        return q, k

    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim,
            bias=False, **self.factory_kwargs)
        self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim,
            bias=False, **self.factory_kwargs)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim,
            bias=False, **self.factory_kwargs)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim,
            bias=False, **self.factory_kwargs)

    def _init_ttt_lr_gate(self):
        linear_weight_data = nn.Linear(self.width, 1, bias=True, **self.
            factory_kwargs).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(torch.stack([torch.
            normal(0, 0.02, size=linear_weight_data.shape, **self.
            factory_kwargs) for _ in range(self.num_heads)], dim=0))
        linear_bias_data = nn.Linear(self.width, 1, bias=True, **self.
            factory_kwargs).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(torch.stack([torch.
            zeros_like(linear_bias_data, **self.factory_kwargs) for _ in
            range(self.num_heads)], dim=0))

    def ln_fused_l2_bwd(self, x, l2_target, gamma, beta, eps=1e-06):
        """Batch backward for LayerNorm fused with L2 loss."""
        D = x.shape[-1]
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)
        x_hat = (x - mu) / std
        y = gamma * x_hat + beta
        grad_output = y - l2_target
        grad_x_hat = grad_output * gamma
        z = 1.0 / D * (D * grad_x_hat - grad_x_hat.sum(dim=-1, keepdim=True
            ) - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)) / std
        return z

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim, **self.factory_kwargs
            ).weight.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.
            unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim, **self.factory_kwargs
            ).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze
            (0), (self.num_heads, 1)))

    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = self.q_proj(hidden_states), self.k_proj(hidden_states
            ), self.v_proj(hidden_states)
        return XQ, XK, XV

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.
            num_heads, self.head_dim))

    def get_eta(self, X, mini_batch_size):
        ttt_lr = torch.einsum('bnkc,hdc->bhnkd', X, self.
            learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(1,
            -1, 1, 1, 1)
        ttt_lr = F.sigmoid(ttt_lr)
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.ttt_base_lr * ttt_lr / self.head_dim
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[0:mini_batch_size]
        token_idx = torch.clamp_min(token_idx, 0.0)
        token_eta = torch.broadcast_to(token_idx.reshape(1, 1, 1,
            mini_batch_size, 1), (X.shape[0], self.num_heads, X.shape[1],
            mini_batch_size, 1))
        return token_eta, ttt_lr_eta

    def get_ttt_inputs(self, inputs, mini_batch_size):
        XQ = inputs['XQ']
        XK = inputs['XK']
        XV = inputs['XV']
        X = inputs['X']
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.width)
        XQ = XQ.reshape(B, self.num_heads, L // mini_batch_size,
            mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, L // mini_batch_size,
            mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, L // mini_batch_size,
            mini_batch_size, self.head_dim)
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_size)
        eta = token_eta * ttt_lr_eta
        inputs = {'XQ': XQ, 'XK': XK, 'XV': XV, 'eta': eta, 'token_eta':
            token_eta, 'ttt_lr_eta': ttt_lr_eta}
        return inputs

    def ln_fwd(self, x, gamma, beta, eps=1e-06):
        """Batch forward for LayerNorm."""
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)
        x_hat = (x - mu) / std
        y = gamma * x_hat + beta
        return y

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = q * cos + self.rotate_half(q) * sin
        k_embed = k * cos + self.rotate_half(k) * sin
        return q_embed, k_embed

    def scan(self, f, init, xs, out, checkpoint_group=0):
        """Minic jax.lax.scan function."""
        carry = init
        if isinstance(xs, dict):
            num_items = len(next(iter(xs.values())))
        else:
            num_items = len(xs[0])

        def scan_fn(carry, i_start, i_end):
            for i in range(i_start, i_end):
                if isinstance(xs, dict):
                    x = {key: tensor[i] for key, tensor in xs.items()}
                else:
                    x = [x[i] for x in xs]
                carry, y = f(carry, x)
                out[i] = y
            return carry
        if checkpoint_group > 0:
            ckpt_every_n = num_items // checkpoint_group
            for k in range(0, num_items, ckpt_every_n):
                carry = torch.utils.checkpoint.checkpoint(scan_fn, carry, k,
                    min(k + ckpt_every_n, num_items), use_reentrant=False)
        else:
            carry = scan_fn(carry, 0, num_items)
        return carry, out

    def ttt(self, inputs, mini_batch_size, last_mini_batch_params_dict):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size
        B = inputs['XV'].shape[0]
        num_mini_batch = inputs['XV'].shape[2]
        L = inputs['XV'].shape[2] * inputs['XV'].shape[3]
        device = inputs['XV'].device
        dtype = inputs['XV'].dtype
        use_dual_form = True

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict['W1_states']
            b1_init = params_dict['b1_states']
            XQ_mini_batch = inputs['XQ']
            XV_mini_batch = inputs['XV']
            XK_mini_batch = inputs['XK']
            eta_mini_batch = inputs['eta']
            token_eta_mini_batch = inputs['token_eta']
            ttt_lr_eta_mini_batch = inputs['ttt_lr_eta']
            X1 = XK_mini_batch
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch
            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1,
                self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.
                head_dim)
            grad_l_wrt_Z1 = self.ln_fused_l2_bwd(Z1, reconstruction_target,
                ln_weight, ln_bias)
            if use_dual_form:
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                Z1_bar = (XQ_mini_batch @ W1_init - eta_mini_batch * Attn1 @
                    grad_l_wrt_Z1 + b1_bar)
                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2
                    ) @ grad_l_wrt_Z1
                b1_last = b1_init - torch.sum(last_eta_mini_batch *
                    grad_l_wrt_Z1, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch, (*ttt_lr_eta_mini_batch.shape[:2
                    ], mini_batch_size, mini_batch_size))
                grad_W1 = torch.einsum('bhki,bhkj->bhkij', X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum('bhnk,bhkij->bhnij', torch.tril(
                    ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict['W1_grad'].unsqueeze(2)
                grad_b1 = torch.einsum('bhnk,bhki->bhni', torch.tril(
                    ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict['b1_grad']
                W1_bar = W1_init.unsqueeze(2
                    ) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3
                    ) + b1_bar
                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]
            Z1_bar = self.ln_fwd(Z1_bar, ln_weight, ln_bias)
            XQW_mini_batch = XQ_mini_batch + Z1_bar
            last_param_dict = {'W1_states': W1_last, 'b1_states': b1_last,
                'W1_grad': grad_W1_last, 'b1_grad': grad_b1_last}
            return last_param_dict, XQW_mini_batch
        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {'W1_states': torch.tile(self.W1.unsqueeze(0
                ), dims=(B, 1, 1, 1)), 'b1_states': torch.tile(self.b1.
                unsqueeze(0), dims=(B, 1, 1, 1))}
            init_params_dict.update(W1_grad=torch.zeros_like(
                init_params_dict['W1_states']))
            init_params_dict.update(b1_grad=torch.zeros_like(
                init_params_dict['b1_states']))
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
        XQW_batch = torch.empty((num_mini_batch, B, self.num_heads,
            mini_batch_size, self.head_dim), device=device, dtype=dtype)
        batch_params_dict, XQW_batch = self.scan(compute_mini_batch,
            init_params_dict, inputs, XQW_batch, self.
            scan_checkpoint_group_size if self.training else 0)
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict

    def _forward(self, X, position_ids, **Z):
        hidden_states = X
        B, L = hidden_states.shape[:2]
        reminder_len = L % self.mini_batch_size
        num_mini_batch = L // self.mini_batch_size
        last_mini_batch_params_dict = None
        XQ, XK, XV = self.get_qkv_projections(hidden_states)
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        Z['position_ids'] = position_ids % self.mini_batch_size
        Z['input'] = XV
        _, Z = self.rotary_emb(X, **Z)
        cos = Z['cos']
        sin = Z['sin']
        XQ, XK = self.permute_qk(XQ, XK)
        XQ, XK = self.apply_rotary_pos_emb(XQ, XK, cos, sin)
        XQ, XK = self.undo_permute_qk(XQ, XK)
        output_hidden_states = []
        if num_mini_batch > 0:
            inputs = {'XQ': XQ[:, :, :num_mini_batch * self.mini_batch_size
                ], 'XK': XK[:, :, :num_mini_batch * self.mini_batch_size],
                'XV': XV[:, :, :num_mini_batch * self.mini_batch_size], 'X':
                hidden_states[:, :num_mini_batch * self.mini_batch_size]}
            output_mod, last_mini_batch_params_dict = self.ttt(self.
                get_ttt_inputs(inputs, self.mini_batch_size),
                mini_batch_size=self.mini_batch_size,
                last_mini_batch_params_dict=last_mini_batch_params_dict)
            output_hidden_states.append(output_mod)
        if reminder_len > 0:
            inputs = {'XQ': XQ[:, :, -reminder_len:], 'XK': XK[:, :, -
                reminder_len:], 'XV': XV[:, :, -reminder_len:], 'X':
                hidden_states[:, -reminder_len:]}
            output_reminder, _ = self.ttt(self.get_ttt_inputs(inputs,
                reminder_len), mini_batch_size=reminder_len,
                last_mini_batch_params_dict=last_mini_batch_params_dict)
            output_hidden_states.append(output_reminder)
        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)
        return output_hidden_states


import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.utils import logging


class RotaryEmbedding(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, dim=None, max_position_embeddings=16, base
        =10000, scaling_factor=1.0, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.scaling_factor = scaling_factor
        self.dim = dim if dim is not None else embed_dim // 4
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2, dtype=
            torch.int64).float().to(device) / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @torch.no_grad()
    def _forward(self, X, input, position_ids, **Z):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = input.device.type
        device_type = device_type if isinstance(device_type, str
            ) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        Z['cos'] = cos.to(**self.factory_kwargs)
        Z['sin'] = sin.to(**self.factory_kwargs)
        return X, Z


import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils._pytree import tree_map
from transformers.utils import logging
from transformers.activations import ACT2FN
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except:
    causal_conv1d_update, causal_conv1d_fn = None, None


class Conv(GAUBase):

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, conv_kernel=4, rms_norm_eps=1e-06, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        kwarg_all['eps'] = rms_norm_eps
        self.norm = RMSNorm(embed_dim=self.embed_dim, block_loc=self.
            block_loc, kwarg_all=self.kwarg_all, **self.factory_kwargs, **
            self.kwarg_all)
        self.conv = nn.Conv1d(embed_dim, embed_dim, bias=True, kernel_size=
            conv_kernel, groups=embed_dim, padding=conv_kernel - 1, **self.
            factory_kwargs)

    def __call__(self, X, **Z):
        hidden_states = X
        seq_len = hidden_states.shape[1]
        hidden_states = self.norm(hidden_states, **Z)[0]
        hidden_states = hidden_states.transpose(1, 2)
        if causal_conv1d_fn is None:
            hidden_states = self.conv(hidden_states)[..., :seq_len]
        else:
            conv_weights = self.conv.weight.view(self.conv.weight.size(0),
                self.conv.weight.size(2))
            hidden_states = causal_conv1d_fn(hidden_states, conv_weights,
                self.conv.bias, activation=None)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


gab_config = {'conv_kernel': 4, 'rms_norm_eps': 1e-06,
    'num_attention_heads': 4, 'scan_checkpoint_group_size': 4,
    'mini_batch_size': 16, 'rope_theta': 10000.0, 'ttt_base_lr': 1.0,
    'intermediate_size': None, 'eps': 1e-05, 'dim': None,
    'max_position_embeddings': 16, 'base': 10000, 'scaling_factor': 1.0}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)