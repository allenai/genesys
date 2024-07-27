"""
Sparse Token Transformer with Attetion Back Tracking
***
2022
"""

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.utils import logging

from .modules import BertAttention, BertIntermediate, BertOutput

logger = logging.get_logger(__name__)

EPS = 1e-7
USE_LTP_ON_CONCRETE = False



class LTPPruneToken(nn.Module):
    def __init__(self):
        super().__init__()

        self.soft_pruning = True
        self.threshold = None # nn.Parameter(torch.randn((1,), dtype=torch.float32))
        self.last_mask = None
        self.new_attention_mask = None
        self.temperature = 5e-4
    
    def init_threshold(self, l, L):
        self.threshold = nn.Parameter(torch.tensor([0.01 * l / L], dtype=torch.float32))

    def forward(self, x, attention_score, attention_mask):
        # x: (N, T, H)
        # attention_score: (N, HEAD, T, T)
        N, T0, H = x.shape
        _N, HEAD, T1, T2 = attention_score.shape
        assert T1 == T2
        assert T0 == T1
        T = T1
        assert N == _N

        if self.soft_pruning:
            #score (N, T)
            score = torch.mean(torch.mean(attention_score, dim=1), dim=1)
            self.last_mask = torch.sigmoid((score - self.threshold) / self.temperature)
        else:
            score = torch.mean(torch.mean(attention_score, dim=1), dim=1)
            self.last_mask = (score > self.threshold) * 1.0
            # this is replace the attention mask for next layer. so equivalent to drop the token.
            # have to update attention mask when hard pruning, according to LTP implementation.
            new_attention_mask = (1-self.last_mask) * (-10000)
            attention_mask = new_attention_mask.view(*attention_mask.shape)
        self.last_mask = self.last_mask.unsqueeze(-1) # masking layer output
        self.new_attention_mask = attention_mask
        
        return x * self.last_mask


class BertLayer(nn.Module):
    def __init__(self, config, arch='bert'):
        super().__init__()
        self.arch = arch
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config, arch=arch)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute", arch=arch)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config, arch=arch)
        
        if arch == 'vit':
            self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        #ltp
        self.ltp_prune_token = False
        self.ltp_prune_token_module = LTPPruneToken()

        #concrete dropout
        self.concrete_weight_regularizer = 1e-6
        self.concrete_dropout_regularizer = 1e-5
        self.concrete_calc_loss = False
        if USE_LTP_ON_CONCRETE:
            self.concrete_init_min = 0.001
            self.concrete_init_max = 0.1
        else:
            self.concrete_init_min = 0.0
            self.concrete_init_max = self.concrete_init_min
        self.concrete_prop_p_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(self.concrete_init_min, self.concrete_init_max))
        self.temperature = 0.1
        self.input_dimensionality = 0

        self.concrete_loss_factor = 1e-3

    def init_p_logits(self):
        torch.nn.init.uniform_(self.p_logit, self.concrete_init_min, self.concrete_init_max)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self.input_dimensionality = hidden_states[0].numel() # Number of elements of first item in batch

        if self.arch == 'bert':
            # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
            self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
            )
            self.self_attention_outputs = self_attention_outputs
            attention_output = self_attention_outputs[0]

            # if decoder, the last output is tuple of self-attn cache
            if self.is_decoder: raise Exception()
            else: outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            cross_attn_present_key_value = None
            if self.is_decoder and encoder_hidden_states is not None: raise Exception()

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
            if self.ltp_prune_token:
                layer_output = self.ltp_prune_token_module(layer_output, self_attention_outputs[-1], attention_mask)
            self.layer_output = layer_output
            outputs = (layer_output,) + outputs

            # if decoder, return the attn key/values as the last output
            if self.is_decoder: raise Exception()

            return outputs
        elif self.arch == 'vit':
            self_attention_outputs = self.attention(
                hidden_states=self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            # first residual connection
            hidden_states = attention_output + hidden_states

            # in ViT, layernorm is also applied after self-attention
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)

            # second residual connection is done here
            layer_output = self.output(layer_output, hidden_states)

            outputs = (layer_output,) + outputs

            return outputs
        else:
            raise Exception()

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    def loss_concrete(self, input_dict):
        if self.concrete_calc_loss:
            if USE_LTP_ON_CONCRETE:
                loss = torch.mean(torch.mean(self.output.dense.concrete_mask.squeeze(-1), dim=-1) / torch.mean(input_dict['attention_mask'].squeeze(-1) * 1.0, dim = -1)) * 1e-1
            else:
                # p = torch.sigmoid(self.p_logit)

                # sum_of_square = 0
                # for param in self.parameters():
                #     sum_of_square += torch.sum(torch.pow(param, 2))
                
                # weights_regularizer = self.concrete_weight_regularizer * sum_of_square / (1 - p + EPS)
                
                # dropout_regularizer = p * torch.log(p + EPS) + (1. - p) * torch.log(1. - p + EPS)
                # dropout_regularizer *= self.concrete_dropout_regularizer * self.input_dimensionality
                
                # loss = weights_regularizer + dropout_regularizer
                loss = ((self.p_logit - self.concrete_init_min) ** 2) * self.concrete_loss_factor
                #loss = (torch.sigmoid(self.p_logit) ** 2) * 1e-6
                #raise_if_nan(loss)
                #loss = 0
            return loss
        else:
            return 0
        

