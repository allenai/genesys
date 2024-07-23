import math
from functools import partial
import json
from typing import Optional
import os
import copy

from collections import namedtuple

from transformers.modeling_outputs import CausalLMOutput
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoTokenizer
)

import torch
from torch import nn, Tensor
from model_discovery.configs.gam_config import GAMConfig
from model_discovery.model.utils.generation import decode
from model_discovery.model.utils.hf import load_config_hf, load_state_dict_hf
# from model_discovery.model.utils.generation import GenerationMixin
from .utils.modules import GatedMLP, MLP

from model_discovery.model.block_registry import BlockRegister

try: 
    from model_discovery.model.ops.triton.layer_norm import (
        RMSNorm,
        layer_norm_fn,
        rms_norm_fn
    )
except:
    RMSNorm       = None
    layer_norm_fn = None
    rms_norm_fn   = None 

from model_discovery import utils as U


class Block(nn.Module):
    def __init__(
        self, dim, gab, mlp_cls=None, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a gab constructor with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> GAB -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> GAB, returning both
        the hidden_states (output of the gab) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.gab = gab()
        self.use_template = mlp_cls is not None
        if self.use_template: 
            self.fused_add_norm = fused_add_norm
            self.residual_in_fp32 = residual_in_fp32
            self.norm = norm_cls(dim)
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
            if self.fused_add_norm:
                assert RMSNorm is not None, "RMSNorm import fails"
                assert isinstance(
                    self.norm, (nn.LayerNorm, RMSNorm)
                ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, **gab_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = GAB(LN(residual))
        """

        # Template includes layer norms, residual connection, and MLP
        if self.use_template:
            if not self.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                    is_rms_norm=isinstance(self.norm, RMSNorm)
                )
        else:
            residual = hidden_states 

        # If no template, then just GAB, all norms and res should be taken care by GAB

        hidden_states = self.gab(hidden_states, **gab_kwargs)
        
        if self.use_template:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.gab.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    block_implementation,
    d_model,
    block_config,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False, # This is for performance reason: we can fuse add + layer_norm
    use_template=False,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    constructor = partial(
        block_implementation,
        embed_dim=d_model,
        device=device,
        dtype=dtype,
        **block_config
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    mlp_cls = partial(
        MLP, hidden_features=d_model*4, out_features=d_model, **factory_kwargs
    ) if use_template else None # or use regular MLP
    block = Block(
        d_model,
        constructor,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_block,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual blocks at initialization by a factor of 1/√N where N is the # of residual blocks.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_block)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_block)


class GAM(nn.Module):
    ''' Generalized Autoregressive Models
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
    '''
    def __init__(
        self,
        d_model: int,
        n_block: int,
        block_implementation,
        vocab_size: int = 50277,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg = None,
        fused_add_norm = False,
        residual_in_fp32 = False,
        use_template = False,
        device = None,
        dtype = None,
        block_config: dict = {},
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, **self.factory_kwargs)
        self.use_template = use_template
        self.n_block=n_block

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm if RMSNorm else None 
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.blocks = nn.ModuleList(
            [
                create_block(
                    block_implementation,
                    d_model,
                    block_config,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    use_template=use_template,
                    **self.factory_kwargs,
                )
                for _ in range(n_block)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm or not RMSNorm else RMSNorm)(
            d_model, eps=norm_epsilon, **self.factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_block=n_block,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, block in enumerate(self.blocks)
        }

    def forward(self, input_ids, inference_params=None, **gab_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for block in self.blocks:
            hidden_states, residual = block(
                hidden_states, residual, **gab_kwargs
            )
        if self.use_template: # in template, all blocks are pre-res (none for L1), so need a final res here
            if not self.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = layer_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm_f, RMSNorm)
                )
        else:
            hidden_states = self.norm_f(hidden_states) # in non-template, res should be handled by the block
        return hidden_states

    def print_size(self):
        size=(
            f' - GAM params: {U.strmodelsize(self)}\n'
            f'   - Embedding: {U.strmodelsize(self.embedding)}\n'
            f'   - Non-embedding: {U.strmodelsize(self.blocks)}\n'
            f'     - Block: {U.strmodelsize(self.blocks[0])} x {len(self.blocks)}\n'
            f'       - GAB: {U.strmodelsize(self.blocks[0].gab)}\n'
        )
        if self.blocks[0].use_template:
            size+=f'       - MLP: {U.strmodelsize(self.blocks[0].mlp)}\n'
        return size




class ModisLMHeadModel(PreTrainedModel):
    ''' Generalized Autoregressive Models with LM Head '''
    config_class = GAMConfig

    def __init__(
        self,
        config: GAMConfig,
        block_implementation,
        initializer_cfg=None,
        device=None,
        dtype=None,
        block_config = {},
    ) -> None:
        """Initializes LM model 

        :param config: 
            The global GAM model configuration. 
        :param block_implementation: 
            The specific GAB model to use. 
        :param initializer_cfg: 
        """
        self.config = config
        if 'd_model' in block_config: # override d_model if it is in block_config for auto-tune
            self.d_model = block_config['d_model']
            block_config.pop('d_model')
        else:
            self.d_model = config.d_model
        if 'n_block' in block_config: # override n_block if it is in block_config for auto-tune
            self.n_block = block_config['n_block']
            block_config.pop('n_block')
        else:
            self.n_block = config.n_block
        vocab_size = config.vocab_size
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__(config)
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = GAM(
            d_model=self.d_model,
            n_block=self.n_block,
            block_implementation=block_implementation,
            block_config=block_config,
            vocab_size=vocab_size,
            rms_norm=config.rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm = config.fused_add_norm,
            residual_in_fp32=config.residual_in_fp32,
            use_template=config.use_template,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_block=self.n_block,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **gab_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, **gab_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(
            cls,
            config: GAMConfig,
            pretrained_model_name,
            device=None,
            dtype=None,
            **kwargs
        ):
        config_data = load_config_hf(pretrained_model_name)
        config = config().update_from_dict(config_data)
        name = kwargs["gab_name"]
        gab,gab_config = BlockRegister.load_block(name)
        del kwargs["gab_name"]
        #kwargs["block_config"] = gab_config
        
        model = cls(
            config,
            block_implementation=gab,
            block_config=gab_config,
            device=device,
            dtype=dtype,
            **kwargs
        )
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory, **kwargs):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)

    def generate( # seems no need, PTM already has this
        self,
        input_ids,
        max_length,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        output = decode(
            input_ids,
            self,
            max_length,
            top_k=top_k,
            top_p=top_p,
            min_p = min_p,
            temperature=temperature,
            **kwargs
        )
        if not output_scores:
            output.scores = None

        return output if return_dict_in_generate else output.sequences

    @classmethod
    def from_config(cls,config,**kwargs):
        """Loads the model from configuration 

        :param config: 
            The global configuration. 
        """
        name = kwargs["gab_name"]
        gab,gab_config = BlockRegister.load_block(name)
        kwargs["block_implementation"] = gab 
        del kwargs["gab_name"]
        kwargs["block_config"] = gab_config
        
        return cls(config,**kwargs)

    def print_size(self,printout=True):
        size=''
        size+='|------Model size------|\n'
        tied='tied' if self.config.tie_embeddings else 'untied'
        size+=f' Total params: {U.strmodelsize(self)} ({tied})\n'
        size+=self.backbone.print_size()
        size+=f' - LM Head params: {U.strmodelsize(self.lm_head)}\n'
        size+='|----------------------|\n'
        if printout:
            print(size)
        return size