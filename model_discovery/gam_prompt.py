from typing import Optional

from transformers.modeling_outputs import CausalLMOutput
from transformers import PreTrainedModel

import torch
from torch import nn, Tensor

from .model.configs.gam_config import GAMConfig
from .model.gab import GAB, gab_config


class Block(nn.Module):
    def __init__(
        self, d_model, block_config, norm_epsilon=1e-5, layer_idx=None, device=None, dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon,**factory_kwargs)
        self.gab = GAB(embed_dim=d_model, layer_idx=layer_idx, device=device, dtype=dtype, **block_config)
        self.layer_idx = layer_idx

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **gab_kwargs
    ):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.gab(hidden_states, inference_params=inference_params, **gab_kwargs)
        return hidden_states, residual


class GAM(nn.Module):
    ''' Generalized Autoregressive Models
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
    '''
    def __init__(
        self,
        d_model: int,
        n_layer: int,
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
        self.n_layer = n_layer
        self.layers = nn.ModuleList(
            [
                Block(
                    d_model,
                    block_config,
                    norm_epsilon=norm_epsilon,
                    layer_idx=i,
                    **self.factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(
            d_model, eps=norm_epsilon, **self.factory_kwargs
        )

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
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
        self.config = config
        self.d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__(config)
        self.backbone = GAM(
            d_model=self.d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False, **factory_kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=lm_logits)
