import torch
import torch.nn as nn
from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl
import torch.nn.functional as F
import logging
from typing import Dict, Tuple


class DynamicHybridLayer(GAUBase):
    """
    Dynamic Hybrid Layer (DHL)

    This GAU integrates a Selective State Space Model (SSM) block, an Attention block,
    and combines their outputs using a dynamic gating mechanism informed by a context vector
    extracted from the input. The DHL adaptively balances the contributions of SSM and
    attention to efficiently handle long sequences while maintaining the ability to capture
    complex dependencies.

    Args:
        embed_dim (int): Embedding dimension.
        block_loc (Tuple[int, int]): Location of the block within the network.
        kwarg_all (dict): Dictionary of all keyword arguments.

    Inputs:
        - X: Input sequence tensor of shape (B, L, D).

    Outputs:
        - Y: Output sequence tensor of the same shape as X.

    Child GAUs:
        - SelectiveSSMUnit
        - AttentionUnit
        - AdvancedContextVectorExtractor
        - HierarchicalChunkProcessor
        - MemoryCompressor
    """

    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict,
        device=None, dtype=None, sequence_threshold=1024, ssm_dim_factor=
        1.0, chunk_size=1024, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(embed_dim, block_loc, kwarg_all)
        self.embed_dim = embed_dim
        self.block_loc = block_loc
        self.kwarg_all = kwarg_all
        self.ssm_dim = int(self.embed_dim * ssm_dim_factor)
        self.norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
        self.s_proj = nn.Linear(self.embed_dim, gate_proj_dim, **self.
            factory_kwargs)
        self.a_proj = nn.Linear(self.embed_dim, gate_proj_dim, **self.
            factory_kwargs)
        self.c_proj = nn.Linear(self.embed_dim, gate_proj_dim, **self.
            factory_kwargs)
        self.ssm_unit = SelectiveSSMUnit(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        self.attention_unit = AttentionUnit(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        self.context_unit = AdvancedContextVectorExtractor(embed_dim=
            self.embed_dim, block_loc=self.block_loc, kwarg_all=
            self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.gate = nn.Linear(gate_proj_dim * 3, self.embed_dim, **self.
            factory_kwargs)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, **self.
            factory_kwargs)
        self.hierarchical_processor = HierarchicalChunkProcessor(embed_dim=
            self.embed_dim, block_loc=self.block_loc, kwarg_all=
            self.kwarg_all, **self.factory_kwargs, **self.kwarg_all)
        self.memory_compressor = MemoryCompressor(embed_dim=self.embed_dim,
            block_loc=self.block_loc, kwarg_all=self.kwarg_all, **
            self.factory_kwargs, **self.kwarg_all)
        self.sequence_threshold = sequence_threshold
        ssm_dim_factor = ssm_dim_factor
        gate_proj_dim = gate_proj_dim
        self.chunk_size = chunk_size

    def _forward(self, X, **Z):
        if not isinstance(X, torch.Tensor):
            raise TypeError('Input X must be a torch.Tensor')
        if X.shape[-1] != self.embed_dim:
            raise ValueError(
                f'Expected input of shape (B, L, {self.embed_dim}), got {X.shape}'
                )
        B, L, D = X.shape
        X_norm = self.norm(X)
        Z['X_norm'] = X_norm
        if L > self.sequence_threshold:
            Z['chunk_size'] = self.chunk_size
            Y_hp, Z_ = self.hierarchical_processor(X_norm, **Z)
            X_norm = Y_hp
            Z.update(Z_)
            Y_mc, Z_ = self.memory_compressor(X_norm, **Z)
            X_norm = Y_mc
            Z.update(Z_)
        else:
            pass
        Y_s, Z_ = self.ssm_unit(X_norm, **Z)
        S = Y_s
        Z.update(Z_)
        Y_a, Z_ = self.attention_unit(X_norm, **Z)
        A = Y_a
        Z.update(Z_)
        Y_c, Z_ = self.context_unit(X_norm, **Z)
        C = Y_c
        Z.update(Z_)
        if S is None:
            S = X_norm
        if A is None:
            A = X_norm
        if C is None:
            C = torch.mean(X_norm, dim=1, keepdim=True).expand(B, L, D)
        S_proj = self.s_proj(S)
        A_proj = self.a_proj(A)
        C_proj = self.c_proj(C)
        concatenated = torch.cat([S_proj, A_proj, C_proj], dim=-1)
        G = torch.sigmoid(self.gate(concatenated))
        seq_factor = min(L / self.sequence_threshold, 1.0)
        G = G * seq_factor + (1 - seq_factor) * (1 - G)
        Y = G * S + (1 - G) * A
        Y = self.out_proj(Y)
        Y = Y + X
        if self.training and L < 5000:
            logging.debug(
                f'Gating values mean: {G.mean().item()} at block {self.block_loc}'
                )
        return Y, Z
