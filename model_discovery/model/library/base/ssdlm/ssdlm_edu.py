#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import logging
import math

import torch
from tqdm.auto import tqdm

from transformers import (
    MODEL_MAPPING,
    AutoModelForSequenceClassification,
)
from transformers.utils.versions import require_version

from termcolor import colored


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_time_variables(t, total_t, device): # according to https://arxiv.org/pdf/2102.09672.pdf

    def ft(small_t, big_t, s=1e-4):
        return torch.cos((small_t / big_t + s) / (1 + s) * math.pi / 2) ** 2

    alpha_t_bar = ft(t, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    alpha_t_minus_bar = ft(t-1, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    beta_t = 1 - (alpha_t_bar / alpha_t_minus_bar)
    beta_t_til = (1 - alpha_t_minus_bar) / (1 - alpha_t_bar) * beta_t
    alpha_t = 1 - beta_t
    return alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t


def apply_controlling_drift(args, perturbed_inputs_diralpha):
    if args.decode_ctr_lr <= 0:
        args.ctr_loss = -1
        return perturbed_inputs_diralpha

    if args.ctr_model is None:
        ctr_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        args.ctr_model = AutoModelForSequenceClassification.from_pretrained(ctr_model_name).to(args.accelerator.device)
    optimizing_label_index = 2

    for ctr_i in range(1):
        with torch.enable_grad():
            perturbed_inputs_diralpha_4ctr = perturbed_inputs_diralpha.clone()
            perturbed_inputs_diralpha_4ctr.requires_grad_()
            perturbed_inputs_simplex_4ctr = torch.nn.functional.softmax(perturbed_inputs_diralpha_4ctr, dim=-1)
            perturbed_inputs_embeds_4ctr = torch.nn.functional.linear(perturbed_inputs_simplex_4ctr, args.ctr_model.get_input_embeddings().weight.t())
            ctr_loss = -torch.nn.functional.log_softmax(args.ctr_model(inputs_embeds=perturbed_inputs_embeds_4ctr).logits, dim=-1)[:,optimizing_label_index].mean()
            args.ctr_loss = ctr_loss
            ctr_delta = -torch.autograd.grad(ctr_loss, perturbed_inputs_diralpha_4ctr)[0] # indexing 0 because the return is a tuple

        perturbed_inputs_diralpha = perturbed_inputs_diralpha + args.decode_ctr_lr * ctr_delta
    
    return perturbed_inputs_diralpha


def logits_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3
    very_low_value = -10000

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    logits = logits.masked_fill(valid_indices == 0, very_low_value - one_hot_value)
    return torch.clamp(logits, max=very_low_value + one_hot_value) - very_low_value


def decode(args, batch_input_ids, dec_depth, total_t, model_embedding_lut, embedding_sum_layer, timestep_layer, model, tokenizer):
    batch_size = args.per_device_eval_batch_size
    if args.decode_truncate_len > 0:
        diffusion_input_ids = batch_input_ids[:, args.context_size:-args.decode_truncate_len]
    else:
        diffusion_input_ids = batch_input_ids[:, args.context_size:]
    
    # for each decode step
    assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
    else:
        unit_context_input_ids = None
    history_decode_ids = None

    for i in range(dec_depth):
        unit_noise = args.noise_manual_scale * args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(args.accelerator.device)
        xt = unit_noise

        if unit_context_input_ids is not None:
            context_inputs_embeds = model_embedding_lut(unit_context_input_ids)
        else:
            context_inputs_embeds = None

        t_range = list(range(1, args.sigma_num_steps+1))
        t_range.reverse()
        progress_bar = tqdm(range(len(t_range)), disable=not args.accelerator.is_local_main_process)
        
        for t in t_range:
            selected_t = torch.FloatTensor([t]).repeat(batch_size).to(args.accelerator.device)
            alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t = get_time_variables(selected_t, total_t, args.accelerator.device)
            beta_t_til = beta_t_til.view(batch_size, 1, 1)
            zt = args.noise_manual_scale * args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(args.accelerator.device)
            
            perturbed_inputs_diralpha = xt
            
            mean_or_protect_for_nan = True # (HACK: for the nan issue)
            if mean_or_protect_for_nan:
                perturbed_inputs_simplex = torch.nn.functional.softmax(perturbed_inputs_diralpha, dim=-1)
            else:
                perturbed_inputs_diralpha = torch.exp(perturbed_inputs_diralpha)
                dir_model = torch.distributions.dirichlet.Dirichlet(perturbed_inputs_diralpha)
                perturbed_inputs_simplex = dir_model.sample()

            # pass to the model, conditioned on the timestep as well
            perturbed_inputs_embeds = embedding_sum_layer(perturbed_inputs_simplex)
            t_progress = selected_t / total_t
            timestep_embeds = timestep_layer(t_progress.view(batch_size,1,1).repeat(1,unit_seq_len,1))

            diffusion_embeds = perturbed_inputs_embeds + timestep_embeds
            if context_inputs_embeds is not None:
                diffusion_embeds = torch.cat((context_inputs_embeds, diffusion_embeds), dim=1)
            outputs = model(inputs_embeds=diffusion_embeds, output_hidden_states=False)
            equivalent_score = outputs.logits
            if unit_context_input_ids is not None:
                equivalent_score = equivalent_score[:, unit_context_input_ids.size(1):].contiguous()

            equivalent_score = apply_controlling_drift(args, equivalent_score)
            
            if t > 1:
                sigma_t = torch.sqrt(beta_t_til)
            else:
                sigma_t = 0
            if args.loss_mode == "l2_on_z":
                raise NotImplementedError("l2_on_z samping is not implemented yet")
            else:
                projected_logits = logits_projection(equivalent_score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
                xt = torch.sqrt(alpha_t_minus_bar).view(-1, 1, 1) * projected_logits
                xt = xt + torch.sqrt(1 - alpha_t_minus_bar).view(-1, 1, 1) * zt

            progress_bar.update(1)

            if t % 200 == 0 or t == 1:
                simplex = torch.nn.functional.softmax(xt, dim=-1)
                logger.info(f"sigma_t={sigma_t}, training_coef_at_t={torch.sqrt(1 - alpha_t_bar)}")
                logger.info(f"predicted simplex's entropy={torch.distributions.categorical.Categorical(logits=equivalent_score).entropy()}, logit_max,min,mean={torch.max(equivalent_score)},{torch.min(equivalent_score)},{torch.mean(equivalent_score)}")

                if unit_context_input_ids is not None:
                    context_sequences = tokenizer.batch_decode(unit_context_input_ids.detach().to('cpu'))
                    logger.info(f"context: {context_sequences}")
                
                real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'))
                logger.info(f"t={t}: {colored(str(sampled_sequences), 'red')}")

                simplex = equivalent_score
                real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'))
                logger.info(f"t={t} (before +z): {colored(str(sampled_sequences), 'green')}")

                alt_i = 1 # look at the second best candidate
                alt_real_token_ids_list = torch.topk(simplex, alt_i+1, dim=-1).indices[:, :, alt_i].view(batch_size, unit_seq_len)
                alt_sampled_sequences = tokenizer.batch_decode(alt_real_token_ids_list.clone().detach().to('cpu'))
                logger.info(f"t={t} (alt{alt_i+1}): {colored(str(alt_sampled_sequences), 'blue')}")

                logger.info(f"ctr loss: {args.ctr_loss}")
                logger.info(f"non-zero vocab: {torch.count_nonzero(projected_logits > -args.one_hot_value+0.0001) / simplex.size(0) / simplex.size(1)} out of {torch.numel(projected_logits) / simplex.size(0) / simplex.size(1)}")
        
        unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)
        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

    if args.context_size > 0:
        init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))
    else:
        init_context_input_ids = None
        context_sequences = None
    gold_sequences = tokenizer.batch_decode(diffusion_input_ids.clone().detach().to('cpu'))
    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'))
    logger.info(f"context: {context_sequences}")
    logger.info(f"gold: {colored(str(gold_sequences), 'yellow')}")
    logger.info(f"t={t}: {colored(str(sampled_sequences), 'red')}")

    return history_decode_ids, init_context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences

