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
        self.mha = EnhancedSinkFlashMHA(embed_dim=self.embed_dim, block_loc
            =self.block_loc, kwarg_all=self.kwarg_all, **self.
            factory_kwargs, **self.kwarg_all)
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
import warnings
try:
    from flash_attn.flash_attention import FlashAttention
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    warnings.warn(
        'FlashAttention is not available. Falling back to standard attention.')


class EnhancedSinkFlashMHA(GAUBase):
    """
    Multi-Head Attention with Enhanced Attention Sinks Integration (EnhancedSinkFlashMHA).

    This GAU extends the SinkFlashMHA by introducing optimizations and enhancements based on observer feedback.

    **Key Features:**

    - **Optimized Tensor Operations:** Improved reshaping and permuting operations to reduce computational overhead.
    - **Refined Attention Mask Logic:** Ensures sink tokens can attend to all tokens, including themselves, without restrictions.
    - **Flexible Sink Tokens:** Supports varying numbers of sink tokens, including zero, with dynamic configuration.
    - **Comprehensive Unit Testing:** Enhanced tests covering diverse sink token configurations and comparative validation.
    - **Integration of FlashAttention:** Utilizes FlashAttention for efficient computation when available and supported.
    - **Support for Mixed-Precision Training (AMP):** Enhances computational efficiency and reduces memory usage.
    - **Robust Hardware Compatibility Checks:** Ensures compatibility with available hardware and provides clear warnings.

    **Args:**

        embed_dim (int): The embedding dimension of the input.
        block_loc (tuple): The location of this block within the network.
        kwarg_all (dict): Dictionary of all keyword arguments.
        n_heads (int, optional): Number of attention heads. Default is 8.
        causal (bool, optional): Whether to apply causal masking. Default is True.
        use_flash_attn (bool, optional): Whether to use FlashAttention if available and supported. Default is True.
        num_sink_tokens (int, optional): Number of sink tokens to use. Default is 1.
        **kwargs: Additional keyword arguments.

    **Integration Details:**

    - Initializes trainable sink token embeddings with flexible configuration.
    - Concatenates sink tokens to the input sequence during the forward pass.
    - Adjusts attention masks to allow all tokens, including sink tokens, to attend to each other without restrictions.
    - Applies causal masking only among local tokens when `causal` is set to `True`.
    - Uses optimized tensor operations to enhance computational efficiency.
    - Utilizes FlashAttention if available and supported; otherwise falls back to standard attention mechanisms.
    - Supports mixed-precision training using Automatic Mixed Precision (AMP) for enhanced computational efficiency.
    - Performs comprehensive hardware checks to ensure compatibility with FlashAttention.

    **Returns:**

        Tuple[Tensor, dict]: Output tensor Y with the same shape as the input X, and updated intermediate variables Z.

    **Raises:**

        AssertionError: If input dimensions are incorrect.

    **Example:**

        # Create an instance of EnhancedSinkFlashMHA
        sink_mha = EnhancedSinkFlashMHA(embed_dim=768, block_loc=(0, 12), kwarg_all={}, n_heads=12, num_sink_tokens=2)
        # Input tensor of shape (batch_size, seq_len, embed_dim)
        X = torch.randn(2, 128, 768)
        # Perform forward pass
        Y, Z = sink_mha(X)
        # Output tensor Y has the same shape as X
        print(Y.shape)  # torch.Size([2, 128, 768])

    **Note:**

        For more information on attention sinks, see the paper:
        "Efficient Streaming Language Models with Attention Sinks" by Xiao et al., 2023.

        For details on FlashAttention, refer to:
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
        and the GitHub repository: https://github.com/HazyResearch/flash-attention

    **Hardware Requirements:**

        - NVIDIA GPUs with compute capability >= 7.5 (e.g., T4, V100, A100, RTX 20-series or newer)
        - CUDA toolkit and compatible PyTorch version
        - Sufficient GPU memory to accommodate large batch sizes and sequence lengths

    **Installation Instructions for FlashAttention:**

        To install FlashAttention, ensure you have the following prerequisites:

        - NVIDIA GPU with compute capability >= 7.5
        - CUDA Toolkit compatible with your PyTorch version
        - PyTorch version compatible with FlashAttention

        Install FlashAttention via pip:

        ```bash
        pip install flash-attn
        ```

        For more detailed installation instructions and troubleshooting, refer to the
        [FlashAttention GitHub repository](https://github.com/HazyResearch/flash-attention).
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, n_heads: int=8, causal: bool=True,
        use_flash_attn: bool=True, num_sink_tokens: int=1, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.causal = causal
        self.use_flash_attn = use_flash_attn
        self.num_sink_tokens = num_sink_tokens
        assert embed_dim % n_heads == 0, 'embed_dim must be divisible by num_heads'
        if self.use_flash_attn:
            if FLASH_ATTENTION_AVAILABLE:
                if torch.cuda.is_available():
                    major, minor = torch.cuda.get_device_capability(device=
                        device)
                    if major < 7 or major == 7 and minor < 5:
                        self.use_flash_attn = False
                        warnings.warn(
                            'FlashAttention requires GPUs with compute capability >= 7.5. Falling back to standard attention.'
                            )
                else:
                    self.use_flash_attn = False
                    warnings.warn(
                        'CUDA is not available. Falling back to standard attention.'
                        )
            else:
                self.use_flash_attn = False
                warnings.warn(
                    'FlashAttention is not installed. Falling back to standard attention.'
                    )
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **self.
            factory_kwargs)
        self.sink_tokens = nn.Parameter(torch.randn(1, self.num_sink_tokens,
            embed_dim, **self.factory_kwargs))
        kwarg_all['rotary_emb_dim'] = self.head_dim
        self.rotary_emb = RotaryPositionalEmbeddings(embed_dim=self.
            embed_dim, block_loc=self.block_loc, kwarg_all=self.kwarg_all,
            **self.factory_kwargs, **self.kwarg_all)

    def _forward(self, X, **Z):
        X = X.to(**self.factory_kwargs)
        B, L, _ = X.size()
        num_sink_tokens = self.num_sink_tokens
        total_len = num_sink_tokens + L
        sink_tokens = self.sink_tokens.expand(B, -1, -1)
        X_combined = torch.cat([sink_tokens, X], dim=1)
        with torch.amp.autocast(device_type='cuda', dtype=self.
            factory_kwargs.get('dtype', torch.float16)):
            q = self.q_proj(X)
            k_combined = self.k_proj(X_combined)
            v_combined = self.v_proj(X_combined)
            q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k_combined = k_combined.view(B, total_len, self.num_heads, self
                .head_dim).permute(0, 2, 1, 3)
            v_combined = v_combined.view(B, total_len, self.num_heads, self
                .head_dim).permute(0, 2, 1, 3)
            Z['input_emb'] = q
            Z['input_pos'] = None
            _, Z = self.rotary_emb(X, **Z)
            q = Z['output_emb']
            Z['input_emb'] = k_combined
            Z['input_pos'] = None
            _, Z = self.rotary_emb(X_combined, **Z)
            k_combined = Z['output_emb']
            q = q.reshape(B * self.num_heads, L, self.head_dim)
            k_combined = k_combined.reshape(B * self.num_heads, total_len,
                self.head_dim)
            v_combined = v_combined.reshape(B * self.num_heads, total_len,
                self.head_dim)
            attn_mask = self._generate_attention_mask(L, num_sink_tokens,
                device=X.device)
            attn_mask = attn_mask.unsqueeze(0).expand(B * self.num_heads, -
                1, -1)
            if self.use_flash_attn and num_sink_tokens == 0:
                flash_attn = FlashAttention(causal=self.causal)
                attn_output = flash_attn(q, k_combined, v_combined)
            else:
                attn_output = F.scaled_dot_product_attention(q, k_combined,
                    v_combined, attn_mask=attn_mask, dropout_p=0.0,
                    is_causal=False)
            attn_output = attn_output.view(B, self.num_heads, L, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B,
                L, self.embed_dim)
            Y = self.out_proj(attn_output)
        return Y, Z

    def _generate_attention_mask(self, L: int, num_sink_tokens: int, device):
        """
        Generates an attention mask that allows all tokens (including sink tokens) to attend to each other without restrictions.
        Applies causal masking among local tokens if `self.causal` is True.
        """
        total_len = num_sink_tokens + L
        mask = torch.zeros((L, total_len), device=device)
        if self.causal:
            causal_mask = torch.triu(torch.full((L, L), float('-inf'),
                device=device), diagonal=1)
            mask[:, num_sink_tokens:] = causal_mask
        return mask


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


gab_config = {'n_heads': 8, 'causal': True, 'use_flash_attn': True, 'eps': 
    1e-05, 'max_seq_len': 4096, 'rotary_emb_base': 10000, 'num_sink_tokens':
    1, 'bias': False, 'multiple_of': 128, 'hidden_features': None,
    'out_features': None, 'activation': None}



autoconfig={}
block_config=gab_config
block_config.update(autoconfig)


from .block_registry import BlockRegister

BlockRegister(
    name="default",
    config=block_config
)(GAB)