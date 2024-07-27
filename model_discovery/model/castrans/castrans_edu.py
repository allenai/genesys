import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from modeling_bart_ours import BartSimCAS
from transformers import BartConfig


class SimCAS(nn.Module):
    def __init__(self, mname, pad_token_id, args=None):
        super(SimCAS, self).__init__()
        self.model = BartSimCAS.from_pretrained(mname, cache_dir="./local_cache")
        self.pad_token_id = pad_token_id

    def forward(self, text_id, candidate_id):
        
        batch_size = text_id.size(0)
        input_mask = text_id != self.pad_token_id
        cand_mask = candidate_id != self.pad_token_id
   
        cand_mask[:, 0] = 1
   
        output = self.model(
            input_ids=text_id, 
            attention_mask=input_mask,
            decoder_input_ids=candidate_id, 
            decoder_attention_mask=cand_mask,
            output_hidden_states=True,
            output_attentions=True,
            )
        
        probs = output[0]  # [bz, seq_len, word_dim]
        
        if self.training:
            with torch.no_grad():
                candidate_id = candidate_id[:, 1:]
                all_reward = torch.gather(F.log_softmax(probs[:, :-1].clone(), dim=-1), 2, candidate_id.unsqueeze(-1)).squeeze(-1)
                all_reward = torch.mean(all_reward, dim=-1)
                all_reward = torch.exp(all_reward) * 10000
                num_cls = self.model.agent_dict['chunk_num']
                self.model.agent_dict['select_rewards'] = all_reward * self.model.agent_dict['attention'][num_cls:] / (
                    1 - torch.sum(self.model.agent_dict['attention'][:num_cls]))
                
                if self.model.agent_dict['real_len'] > 2048:
                    self.model.agent_dict['skip_rewards'] = all_reward / (self.model.agent_dict['real_len'])
                else:
                    self.model.agent_dict['skip_rewards'] = all_reward / max(self.model.agent_dict['all_len'], 2048)
        
        output = {"probs": probs}
        return output
    
    def initialize(self):
        try:
            self.model.model.encoder.initialize()
        except:
            self.model.led.encoder.initialize()

    def scoring_mode(self):
        self.model.model.scoring_mode()

    def generation_mode(self):
        self.model.model.generation_mode()

    def generate(
        self,
        **model_kwargs,
    ):
        return self.model.generate(
            **model_kwargs)