from typing import Optional
from dataclasses import dataclass

from transformers.modeling_outputs import CausalLMOutput
from transformers import PreTrainedModel,PretrainedConfig

import torch
from torch import nn, Tensor
from model_discovery.model.utils.modules import MLP

from gab import GAB, gab_config


@dataclass
class GAMConfig(PretrainedConfig):
    '''Configurations for Generalized Autoregressive Models.'''

    d_model: int
    n_block: int
    batch_tokens: int 
    vocab_size: int = None


class Block(nn.Module):
    def __init__(
        self, d_model, block_loc, block_config, norm_epsilon=1e-5, device=None, dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.block_loc=block_loc
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon,**factory_kwargs)
        self.gab = GAB(embed_dim=d_model, device=device, dtype=dtype, **block_config)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_epsilon,**factory_kwargs)
        self.mlp = MLP(in_features=d_model, out_features=d_model, hidden_features=d_model * 4, **factory_kwargs)


    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, **intermediate_vars
    ):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states, intermediate_vars = self.gab(hidden_states, **intermediate_vars)
        residual = hidden_states + residual
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, intermediate_vars


class GAM(nn.Module):
    ''' Generalized Autoregressive Models
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
    '''
    def __init__(
        self,
        d_model: int,
        n_block: int,
        vocab_size: int = 50277,
        norm_epsilon: float = 1e-5,
        device = None,
        dtype = None,
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, **self.factory_kwargs)

        block_config = gab_config()
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    block_loc=(layer_idx,n_block),
                    block_config=block_config,
                    norm_epsilon=norm_epsilon,
                    **self.factory_kwargs,
                )
                for layer_idx in range(n_block)
            ]
        )

        self.norm_out = nn.LayerNorm(
            d_model, eps=norm_epsilon, **self.factory_kwargs
        )

    def forward(self, input_ids):
        hidden_states = self.embedding(input_ids)
        residual = None
        intermediate_vars = {}
        for block in self.blocks:
            hidden_states, residual, intermediate_vars = block(
                hidden_states, residual, **intermediate_vars
            )
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_out(residual)
        return hidden_states


class ModisLMHeadModel(PreTrainedModel):
    ''' Generalized Autoregressive Models with LM Head '''
    config_class = GAMConfig

    def __init__(
        self,
        config: GAMConfig,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(config)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = GAM(
            d_model=config.d_model,
            n_block=config.n_block,
            vocab_size=config.vocab_size,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, **factory_kwargs)

    def forward(self, input_ids, **gab_kwargs):
        hidden_states = self.backbone(input_ids, **gab_kwargs)
        lm_logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=lm_logits)
