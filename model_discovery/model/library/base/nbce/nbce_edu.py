import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from utils import n_tokens_in_prompt
from transformers import TopPLogitsWarper, LogitsProcessorList
from numpy import typing as npt
from transformers import LogitsProcessor

LOGIT_BIAS = 100


class RestrictiveTokensLogitsProcessor(LogitsProcessor):
    """ Restrictive decoding is done by adding logits_bias to the relevant tokens. Based on:
    https://help.openai.com/en/articles/5247780-using-logit-bias-to-define-token-probability
    """

    def __init__(self,
                 restrictive_token_ids: npt.NDArray[int],
                 eos_token_id: int,
                 prompt_length_to_skip: int = 0,
                 logits_bias: int = LOGIT_BIAS):
        self.restrictive_token_ids = restrictive_token_ids
        self.eos_token_id = eos_token_id
        self.logits_bias = logits_bias
        self.prompt_length_to_skip = prompt_length_to_skip
        self.mask = np.ones(restrictive_token_ids.shape[0], dtype=bool)

        self._preprocess_restrictive_array()

    def _preprocess_restrictive_array(self):
        # extend restrictive_token_ids to include eos as last token for each sequence
        if not (self.restrictive_token_ids[:, -1] == self.eos_token_id).all():
            self.restrictive_token_ids = np.column_stack(
                (self.restrictive_token_ids, np.ones(self.restrictive_token_ids.shape[0]) * self.eos_token_id)). \
                astype(int)

    def update_new_prompt_length_to_skip(self, prompt_length_to_skip: int):
        self.prompt_length_to_skip = prompt_length_to_skip
        self.mask = np.ones(self.restrictive_token_ids.shape[0], dtype=bool)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # assert input_ids.shape[0] == 1, "This implementation doesn't support batching"
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        if new_tokens_length > 0:
            self.mask = self.mask & (self.restrictive_token_ids[:, new_tokens_length - 1] == input_ids[
                0, -1].item())


        # scores[:, self.restrictive_token_ids[self.mask, new_tokens_length]] += self.logits_bias
        scores[:,:, self.restrictive_token_ids[self.mask, new_tokens_length]] += self.logits_bias
        # print(scores.shape, self.restrictive_token_ids[self.mask, new_tokens_length].shape)
        # scores[:,self.restrictive_token_ids[self.mask, new_tokens_length]:,:] += self.logits_bias

        return scores


processors = LogitsProcessorList()
processors.append(TopPLogitsWarper(0.95))



def combine_past_key_values(past_key_value):
    present = ()
    for layer_past in zip(*past_key_value):
        key, value = tuple(zip(*layer_past))
        key = torch.cat(key, dim=0)
        value = torch.cat(value, dim=0)
        present += ((key, value), )
    return present

def generate_NBCE_position_ids(attention_mask: torch.Tensor, max_window_size: int,
                              past_key_values: Tuple[Tuple[torch.Tensor]],
                              sum_windows_size: int, windows_key_values: Tuple[Tuple[torch.Tensor]]) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    n_task_tokens = position_ids.shape[1] - sum_windows_size
    position_ids[0, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)
    position_ids.masked_fill_(attention_mask == 0, 1)
    if past_key_values:  # i.e., first token is already generated
        position_ids = position_ids[:, -1].unsqueeze(-1)
    elif windows_key_values:  # i.e., we are in the first token generation
        position_ids = position_ids[:, sum_windows_size:]
    return position_ids


class NBCEModelWrapper:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 task: str,
                 device: str,
                 context_window_size: int,
                 right_indentation: bool = False,
                 beta: float = None
                 ):
        self.model = model
        self.task = task
        self.beta = beta
        self.tokenizer = tokenizer
        self.context_window_size = context_window_size
        self.device = device
        self.right_indentation = right_indentation
    def get_chunks(self, texts, max_length):

        demo_encoding = []

        max_length = self.context_window_size -(max_length*2)
        demo_encoding_batch = [[self.tokenizer.pad_token_id] * max_length]
        attention_mask_batch = [[0] * max_length]
        for text in texts:
            demo_input_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            # if len(demo_input_ids) < 30: continue
            if len(demo_encoding) + len(demo_input_ids) <= max_length:
                demo_encoding += demo_input_ids
            else:
                demo_encoding_batch.append((demo_encoding + demo_input_ids)[-max_length:])
                attention_mask_batch.append([1] * max_length)
                demo_encoding = []
            if len(demo_encoding_batch) >= len(texts)-1:break


        if len(demo_encoding_batch) == 0:  # doesn't need chunk!
            demo_encoding_batch.append(demo_encoding)
            attention_mask_batch.append([1] * len(demo_encoding))

        return demo_encoding_batch, attention_mask_batch


    def _get_windows(self, texts: List[str], max_length:int) -> List[Dict]:
        windows = []
        if self.right_indentation:
            max_window_size = max(n_tokens_in_prompt(self.tokenizer, t, add_special_tokens=True) for t in texts)


        demo_encoding_batch , attention_mask_batch = self.get_chunks(texts, max_length=max_length)

        demo_encoding_batch = torch.LongTensor(demo_encoding_batch).to(self.device)
        attention_mask_batch = torch.LongTensor(attention_mask_batch).to(self.device)
        for demo_encoding, attention_mask in zip(demo_encoding_batch, attention_mask_batch):
            window_size = len(demo_encoding)
            with torch.no_grad():

                output = self.model(
                    input_ids = demo_encoding.unsqueeze(0),
                    attention_mask = attention_mask.unsqueeze(0),
                    use_cache = True
                )

            windows.append({'text': self.tokenizer.decode(demo_encoding, skip_special_tokens=True),
                            'encoded_input': demo_encoding.unsqueeze(0),
                            'attention_mask': attention_mask.unsqueeze(0),
                            'window_size': window_size,
                            'output': output,
                            'past': output['past_key_values']})
        return windows

    def get_contexts_cache(self, contexts: List[str], max_length:int) -> Dict:
        windows = self._get_windows(contexts, max_length=max_length)
        windows_sizes = [window['window_size'] for window in windows]
        j = np.argmax(windows_sizes)
       
        past_attention_mask=torch.cat([window['attention_mask'] for window in windows],  dim=0)

        return {'past_key_values': combine_past_key_values([window['past'] for window in windows]),
                'max_window_size': max(windows_sizes),
                'past_attention_mask':past_attention_mask,

                'sum_windows_size': sum(windows_sizes) - (len(windows) - 1)}


    def nbce_generate(self,
                     contexts: Optional[List[str]] = None,
                     task_text: Optional[str] = None,
                     answer: Optional[str] = None,
                     contexts_cache: Optional[Dict] = None,
                     restrictive_logit_preprocessor: Optional[RestrictiveTokensLogitsProcessor] = None,
                     **kwargs
                     ) -> str:
        assert (contexts is None) != (contexts_cache is None), "nbce_generate should work with contexts or cache, not with both!"
        cache = contexts_cache or self.get_contexts_cache(contexts)
        encoded_task_text = self.tokenizer(task_text, add_special_tokens=False, return_tensors='pt').to(self.device)
        if restrictive_logit_preprocessor:
            restrictive_logit_preprocessor.update_new_prompt_length_to_skip(encoded_task_text['input_ids'].shape[1])
            kwargs['logits_processor'] = [restrictive_logit_preprocessor]
        n=cache['past_attention_mask'].shape[0]
        logits_processor = LogitsProcessorList([
            restrictive_logit_preprocessor,
        ])

        attention_mask = torch.cat([cache['past_attention_mask'], encoded_task_text['attention_mask'].tile(n, 1)],
                                   dim=-1).to(self.device)

        input_ids = encoded_task_text['input_ids'].tile(n, 1)
        past_key_values=cache['past_key_values']
       
        ids=input_ids[0].unsqueeze(0)
        if self.task in ['banking77', 'clinic150', 'nlu', 'nluscenario', 'trecfine']:
            res=''
            for j in range(kwargs['max_new_tokens']) :
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                    )
                    past_key_values= outputs.past_key_values
                    logits = logits_processor(ids, outputs.logits)[:, -1]


                    beta, eta = self.beta, 0.1
                    logits = logits - logits.logsumexp(dim=-1, keepdims=True)
                   
                    entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
                    k = entropy[1:].argmin() + 1
                    logits_max = logits[k]
                    # logits_max = logits[1:].mean(0).clip(-100, 0) #mean Eq.7

                    logits_uncond = logits[0]
                    logits_merged = (1 + beta) * logits_max - beta * logits_uncond
                    logits = torch.where(logits_uncond > -100, logits_merged, logits_max)
                    
                    tau = 0.01
                    probas = torch.nn.functional.softmax(logits[None] / tau, dim=-1)
                    next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)

                    if next_tokens[0]== self.tokenizer.eos_token_id:
                        break

                    ret = self.tokenizer.batch_decode(next_tokens)
                    
                    input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
                    ids = torch.cat([ids, input_ids[0].unsqueeze(0)], dim=-1)
                    attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=self.device)],
                                               dim=-1)
                    res += '' + ret[0]
        elif self.task in ['piqa', 'hellaswag' ,'obqa', 'copa', 'arce']:

            answer_encoding = self.tokenizer(
                answer,
                padding=True,
                return_tensors='pt',
                add_special_tokens=False
            ).to(self.device)
            res = torch.empty(0).to(self.device)
            for candidate_encoding, candidate_mask in zip(answer_encoding.input_ids,
                                                          answer_encoding.attention_mask):
                candidate_encoding = candidate_encoding[torch.where(candidate_mask)].unsqueeze(0)
                multi_encoding = torch.cat((input_ids, candidate_encoding.tile(n, 1)), dim=-1)
                multi_attention_mask = torch.cat((cache['past_attention_mask'], torch.ones(multi_encoding.shape, device=self.device)),
                                                 dim=-1)
              
                with torch.no_grad():
                        outputs = self.model(
                            input_ids=multi_encoding,
                            attention_mask=multi_attention_mask,
                            past_key_values=past_key_values
                    ).logits


                beta, eta = self.beta, 0.1
                logits = outputs[:, (input_ids.shape[1] - 1): -1]
                logits = logits - logits.logsumexp(dim=-1, keepdims=True)
                entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
                k = entropy[:, 0].argmin()
                logits_max = logits[k]
                # logits_max = logits[1:].mean(0).clip(-100, 0) #mean Eq.7
                logits_uncond = logits[0]
                logits_merged = (1 + beta) * logits_max - beta * logits_uncond
                logits = torch.where(logits_uncond > -100, logits_merged, logits_max)

                logits = logits[torch.arange(logits.shape[0]).to(self.device), candidate_encoding.flatten()].mean()
                res = torch.cat((res, logits.unsqueeze(0)), dim=0)
            res= res.argmax(dim=-1).item()
        else:
            attention_mask = torch.cat([cache['past_attention_mask'], encoded_task_text['attention_mask'].tile(n, 1)],
                                       dim=-1).to(self.device)

        

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values
                ).logits

            beta, eta = self.beta, 0.1
            
            logits = logits_processor(ids, outputs)[:, -1]

            logits = logits - logits.logsumexp(dim=-1, keepdims=True)
            entropy = -(logits.exp() * logits.clip(-100, 0)).sum(dim=-1)
           
            k = entropy[1:].argmin() + 1
            logits_max = logits[k]
            # logits_max = logits[1:].mean(0).clip(-100, 0) #mean Eq.7

            logits_uncond = logits[0]
            logits_merged = (1 + beta) * logits_max - beta * logits_uncond

            logits = torch.where(logits_uncond > -100, logits_merged, logits_max)
           
            tau = 0.01
            probas = torch.nn.functional.softmax(logits[None] / tau, dim=-1)

            next_tokens = torch.multinomial(probas, num_samples=1).squeeze(1)
            res = self.tokenizer.batch_decode(next_tokens)[0]

        return res