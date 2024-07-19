from transformers.modeling_outputs import CausalLMOutput
from transformers import PreTrainedModel, PretrainedConfig
from dataclasses import dataclass

import torch
from torch import nn

from gab import GAB, gab_config


@dataclass
class GAMConfig(PretrainedConfig):
    '''Configurations for Generalized Autoregressive Models.'''

    d_model: int
    n_block: int
    batch_tokens: int 
    vocab_size: int = None


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
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, **self.factory_kwargs)

        block_config = gab_config()
        self.blocks = nn.ModuleList(
            [
                GAB(
                    embed_dim=d_model, 
                    device=device, 
                    dtype=dtype, 
                    **block_config
                )
                for _ in range(n_block)
            ]
        )
        self.norm_out = nn.LayerNorm(
            d_model, eps=norm_epsilon, **self.factory_kwargs
        )

    def forward(self, input_ids, **gab_kwargs):
        hidden_states = self.embedding(input_ids)
        for block in self.blocks:
            hidden_states= block(
                hidden_states,
                **gab_kwargs
            )
        hidden_states = self.norm_out(hidden_states)
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
