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
        self.mha = MHA(embed_dim=self.embed_dim, block_loc=self.block_loc,
            kwarg_all=self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
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
import math
from einops import rearrange, repeat


class MHA(GAUBase):
    """Multi-head self-attention and cross-attention"""

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        n_heads: int=8, causal: bool=True, num_heads_kv: int=None, head_dim:
        int=None, mlp_dim: int=0, qkv_proj_bias: bool=True, out_proj_bias:
        bool=True, softmax_scale: float=None, rotary_emb_base=10000.0,
        d_conv: int=0, device=None, dtype=None, **kwargs) ->None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.d_conv = d_conv
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.num_heads = n_heads
        self.num_heads_kv = (num_heads_kv if num_heads_kv is not None else
            n_heads)
        assert self.num_heads % self.num_heads_kv == 0, 'num_heads must be divisible by num_heads_kv'
        if head_dim is None:
            assert self.embed_dim % n_heads == 0, 'embed_dim must be divisible by num_heads'
        self.head_dim = (head_dim if head_dim is not None else self.
            embed_dim // n_heads)
        self.mlp_dim = math.ceil(mlp_dim / 256) * 256
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads
        kwarg_all['rotary_emb_dim'] = self.head_dim
        self.rotary_emb = RotaryPositionalEmbeddings(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)
        self.in_proj = nn.Linear(embed_dim, qkv_dim + self.mlp_dim, bias=
            qkv_proj_bias, **self.factory_kwargs)
        if self.d_conv > 0:
            self.conv1d = nn.Conv1d(qkv_dim, qkv_dim, kernel_size=self.
                d_conv, padding=self.d_conv - 1, groups=qkv_dim, **self.
                factory_kwargs)
        self.out_proj = nn.Linear(out_dim + self.mlp_dim // 2, embed_dim,
            bias=out_proj_bias, **self.factory_kwargs)

    def _forward(self, X, **Z):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        qkv = self.in_proj(X)
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.
                mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)
        if self.d_conv > 0:
            qkv = rearrange(self.conv1d(rearrange(qkv, 'b s d -> b d s'))[
                ..., :-(self.d_conv - 1)], 'b d s -> b s d').contiguous()
        q, k, v = qkv.split([self.num_heads * self.head_dim] * 3, dim=-1)
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)
        Z['input_emb'] = q
        _, Z = self.rotary_emb(X, **Z)
        q = Z['output_emb']
        Z['input_emb'] = k
        _, Z = self.rotary_emb(X, **Z)
        k = Z['output_emb']
        k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads //
            self.num_heads_kv)
        v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads //
            self.num_heads_kv)
        context = F.scaled_dot_product_attention(q.transpose(1, 2), k.
            transpose(1, 2), v.transpose(1, 2), is_causal=self.causal,
            scale=self.softmax_scale).transpose(1, 2)
        context = rearrange(context, '... h d -> ... (h d)')
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        return out


import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class RotaryPositionalEmbeddings(GAUBase):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, rotary_emb_base: int=10000, rotary_emb_dim:
        int=None, max_seq_len: int=4096, **kwargs) ->None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.dim = rotary_emb_dim
        self.base = rotary_emb_base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / self.base ** (torch.arange(0, self.dim, 2, **self.
            factory_kwargs)[:self.dim // 2].float() / self.dim)
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int=4096) ->None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=
            self.theta.device)
        idx_theta = torch.einsum('i, j -> ij', seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)],
            dim=-1)
        self.register_buffer('cache', cache, persistent=False)

    def _forward(self, X: Tensor, input_emb: Tensor, input_pos: Optional[
        Tensor]=None) ->Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        seq_len = input_emb.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[
            input_pos]
        xshaped = input_emb.float().reshape(*input_emb.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2
            )
        x_out = torch.stack([xshaped[..., 0] * rope_cache[..., 0] - xshaped
            [..., 1] * rope_cache[..., 1], xshaped[..., 1] * rope_cache[...,
            0] + xshaped[..., 0] * rope_cache[..., 1]], -1)
        x_out = x_out.flatten(3)
        output_emb = x_out.type_as(input_emb)
        return X, {'output_emb': output_emb}


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


class GatedMLP(GAUBase):
    """
    Gated Multi-Layer Perceptron (GatedMLP)

    This GAU introduces a gating mechanism within an MLP to control the flow of information,
    enhancing the model's ability to focus on relevant features and suppress irrelevant ones.

    **Mathematical Formulation:**

    Given an input tensor \\( X \\in \\mathbb{R}^{B 	imes L 	imes D} \\):

    \\[
    Y = 	ext{GatedMLP}(X) = 	ext{FC}_2(	ext{Activation}(	ext{FC}_1(X)) \\odot 	ext{Gate}(X))
    \\]

    Where:
    - \\( 	ext{FC}_1 \\) is a linear transformation expanding the dimensionality.
    - \\( 	ext{Activation} \\) applies a non-linear function (e.g., SiLU).
    - \\( 	ext{Gate} \\) is a linear transformation followed by a sigmoid to gate the activations.
    - \\( \\odot \\) denotes element-wise multiplication.
    - \\( 	ext{FC}_2 \\) projects back to the original embedding dimension.

    **Attributes:**
        - `fc1`: Linear layer projecting input to twice the hidden features for gating.
        - `activation`: Activation function applied to the gated output.
        - `fc2`: Linear layer projecting back to the output features.

    **Args:**
        embed_dim (int): Embedding dimension of the input and output.
        block_loc (tuple): Location of the block within the network, e.g., (layer_idx, n_block).
        kwarg_all (dict): Dictionary of all keyword arguments for initializing child GAUs.
        device (torch.device, optional): Device to allocate layers.
        dtype (torch.dtype, optional): Data type of the layers.
        hidden_features (int, optional): Number of hidden units in the first linear layer. Defaults to `int(8 * embed_dim / 3)`.
        out_features (int, optional): Number of output units in the second linear layer. Defaults to `embed_dim`.
        activation (callable, optional): Activation function to use (e.g., `F.silu`).
        bias (bool, optional): Whether to include bias terms in linear layers. Defaults to `False`.
        multiple_of (int, optional): Ensures hidden features are a multiple of this value. Defaults to `128`.
        **kwargs: Additional keyword arguments.

    **Shape:**
        - Input: (B, L, D)
        - Output: (B, L, D)

    **Example:**
        gated_mlp = GatedMLP(
            embed_dim=64,
            block_loc=(0, 0),
            kwarg_all={},
            hidden_features=128,
            out_features=64,
            activation=F.silu,
            bias=True,
            multiple_of=128,
            device='cuda',
            dtype=torch.float32
        )
        X = torch.randn(2, 10, 64)
        Y, Z = gated_mlp(X)

    **References:**
        - Liu, H., Dai, Z., So, D. R., & Le, Q. V. (2021). Pay Attention to MLPs. Neural Information Processing Systems, 34, 9204-9215.
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, hidden_features: int=None, out_features:
        int=None, activation: callable=None, bias: bool=False, multiple_of:
        int=128, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        """
        Initializes the GatedMLP GAU.

        Args:
            embed_dim (int): Embedding dimension of the input and output.
            block_loc (tuple): Location of the block within the network, e.g., (layer_idx, n_block).
            kwarg_all (dict): Dictionary of all keyword arguments for initializing child GAUs.
            device (torch.device, optional): Device to allocate layers.
            dtype (torch.dtype, optional): Data type of the layers.
            hidden_features (int, optional): Number of hidden units in the first linear layer.
                Defaults to `int(8 * embed_dim / 3)`.
            out_features (int, optional): Number of output units in the second linear layer.
                Defaults to `embed_dim`.
            activation (callable, optional): Activation function to use. Defaults to `F.silu`.
            bias (bool, optional): Whether to include bias terms in linear layers. Defaults to `False`.
            multiple_of (int, optional): Ensures hidden features are a multiple of this value. Defaults to `128`.
            **kwargs: Additional keyword arguments.
        """
        self.out_features = (out_features if out_features is not None else
            embed_dim)
        self.hidden_features = (hidden_features if hidden_features is not
            None else int(8 * embed_dim / 3))
        self.hidden_features = (self.hidden_features + multiple_of - 1
            ) // multiple_of * multiple_of
        self.fc1 = nn.Linear(embed_dim, 2 * self.hidden_features, bias=bias,
            **self.factory_kwargs)
        self.activation = activation if activation is not None else F.silu
        self.fc2 = nn.Linear(self.hidden_features, self.out_features, bias=
            bias, **self.factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the linear layers with Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def _forward(self, X, **Z):
        """
        Forward pass of the GatedMLP.

        Args:
            X (Tensor): Input tensor of shape (B, L, D).
            **Z: Intermediate variables.

        Returns:
            Tuple[Tensor, dict]: Output tensor and updated intermediate variables.
        """
        assert X.dim() == 3 and X.size(-1
            ) == self.embed_dim, f'Expected input shape (*, embed_dim), got {X.shape}'
        y = self.fc1(X)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y, {}


gab_config = {'n_heads': 8, 'causal': True, 'num_heads_kv': None,
    'head_dim': None, 'mlp_dim': 0, 'qkv_proj_bias': True, 'out_proj_bias':
    True, 'softmax_scale': None, 'rotary_emb_base': 10000, 'd_conv': 0,
    'max_seq_len': 4096, 'eps': 1e-05, 'hidden_features': None,
    'out_features': None, 'activation': None, 'bias': False, 'multiple_of': 128
    }



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)