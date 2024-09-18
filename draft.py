import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
import torch.nn.functional as F
import math
from typing import Optional


class DynamicHybridLayer(GAUBase):
    """
    Dynamic Hybrid Layer (DHL)

    This GAU integrates a Selective State Space Model (SSM) block and an Attention block,
    combining their outputs using a dynamic gating mechanism informed by a context vector
    extracted from the input. The DHL adaptively balances the contributions of SSM and
    attention to efficiently handle long sequences while maintaining the ability to capture
    complex dependencies.

    **Enhancements Implemented:**
    - **Hierarchical Chunking:** Processes very long sequences efficiently by splitting them into manageable chunks.
    - **Adaptive Gating Mechanism:** Adjusts the balance between SSM and Attention outputs based on sequence length.
    - **Optimized Computations:** Reduces computational overhead by projecting gating inputs to a lower dimension.
    - **Error Handling and Input Validation:** Ensures robustness by checking input types and dimensions.

    **Inputs:**
        - **X:** Input sequence tensor of shape (B, L, D), where B is batch size, L is sequence length, and D is embedding dimension.

    **Outputs:**
        - **Y:** Output sequence tensor of the same shape as X.

    **Intermediate Variables in Z:**
        - **'X_norm':** Normalized input tensor.
        - **'S':** Output from the SelectiveSSMUnit.
        - **'A':** Output from the AttentionUnit.
        - **'C':** Context vector extracted from the input.

    **Child GAUs:**
        - **SelectiveSSMUnit:** Processes the normalized input through a selective SSM.
        - **AttentionUnit:** Applies an efficient attention mechanism to the normalized input.
        - **ContextVectorExtractor:** Extracts a context vector from the normalized input.

    **Usage Example:**

        layer = DynamicHybridLayer(embed_dim=64, block_loc=(0,1), kwarg_all={})
        Y, Z = layer(X)
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, ssm_dim: int=None, num_heads: int=8,
        max_chunk_length: int=1024, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        ssm_dim = ssm_dim if ssm_dim is not None else embed_dim
        self.max_chunk_length = max_chunk_length
        self.norm = nn.LayerNorm(embed_dim, **self.factory_kwargs)
        kwarg_all['ssm_dim'] = ssm_dim
        kwarg_all['num_heads'] = num_heads
        self.ssm_unit = SelectiveSSMUnit(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        self.attention_unit = AttentionUnit(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        self.context_unit = ContextVectorExtractor(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        gating_input_dim = self.embed_dim // 4
        self.gate_proj = nn.Linear(self.embed_dim, gating_input_dim, **self
            .factory_kwargs)
        self.gate = nn.Linear(gating_input_dim, 1, **self.factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **self.factory_kwargs)

    def _forward(self, X, **Z):
        if not isinstance(X, torch.Tensor):
            raise TypeError('Input X must be a torch.Tensor')
        if X.dim() != 3:
            raise ValueError(
                f'Input X must be a 3D tensor, got {X.dim()}D tensor instead')
        if X.shape[-1] != self.embed_dim:
            raise ValueError(
                f'Expected last dimension of X to be {self.embed_dim}, got {X.shape[-1]}'
                )
        B, L, D = X.shape
        if L > self.max_chunk_length:
            chunks = X.split(self.max_chunk_length, dim=1)
            outputs = []
            for chunk in chunks:
                Y_chunk, Z = self._process_chunk(chunk, **Z)
                outputs.append(Y_chunk)
            Y = torch.cat(outputs, dim=1)
        else:
            Y, Z = self._process_chunk(X, **Z)
        return Y, Z

    def _process_chunk(self, X_chunk, **Z):
        X_norm = self.norm(X_chunk)
        Z['X_norm'] = X_norm
        S, Z = self.ssm_unit(X_norm, **Z)
        A, Z = self.attention_unit(X_norm, **Z)
        C, Z = self.context_unit(X_norm, **Z)
        if S is None:
            S = X_norm
        if A is None:
            A = X_norm
        if C is None:
            C = torch.mean(X_norm, dim=1, keepdim=True)
        seq_len_factor = min(1.0, X_chunk.shape[1] / self.max_chunk_length)
        gating_input = (S + A + seq_len_factor * C.expand_as(S)) / 3
        gating_hidden = self.gate_proj(gating_input)
        G = torch.sigmoid(self.gate(gating_hidden))
        Y = G * S + (1 - G) * A
        Y = self.out_proj(Y)
        Y = Y + X_chunk
        return Y, Z
    

'''
GPT2
       |- MHA
           |- RotaryPositionalEmbeddings
       |- DynamicHybridLayer (Rating: 4.6/5)
           |- SelectiveSSMUnit (Rating: 4.5/5)
               |- SSMUnit (Rating: 4.5/5)
           |- AttentionUnit
           |- ContextVectorExtractor (Rating: 4.0/5)
       |- RMSNorm

Implemented Units: SelectiveSSMUnit, RotaryPositionalEmbeddings, RMSNorm, SSMUnit, AttentionUnit, MHA, ContextVectorExtractor, GPT2, DynamicHybridLayer
All units are implemented.

'''