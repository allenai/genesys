import torch
import random
import os
import numpy as np
import argparse
import wandb
import logging

import transformers
from huggingface_hub import login

from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from accelerate import notebook_launcher
import functools as ft
from argparse import Namespace

from .trainer.data_loader import load_datasets
from .trainer.modis_trainer import ModisTrainer
from .model.configs.gam_config import (
    GAMConfig,
    GAMConfig_10M,
    GAMConfig_debug
)
from .model.gam import ModisLMHeadModel
from .evals.evaluator import run_eval
from . import utils as U

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

util_logger = logging.getLogger('model_discovery.train')


def setup_environ(args):
    if not os.environ.get("HF_KEY"):
        raise ValueError('Must set KH_KEY!')
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError('Must set WANDB_API_KEY')
    if not os.environ.get("DATA_DIR"):
        raise ValueError("Must set data_dir")
    if not args.ckpt_dir:
        raise ValueError('Must specify the checkpoitn directory via `--ckpt_dir`')

    ### make checkpoint dir 
    U.mkdir(args.ckpt_dir)
    util_logger.info(f'Creating checkpoint directory: {args.ckpt_dir}')
    
    ### log into the hf hub 
    login(os.environ.get("HF_KEY",None))

    ## initialize wandb
    util_logger.info(f'Setting up wandb...')
    if args.resume and os.path.exists(f"{args.ckpt_dir}/{args.config}/{args.modelname}/wandb_id.txt"):
        wandb_id = open(f"{args.ckpt_dir}/{args.config}/{args.modelname}/wandb_id.txt").read()
        wandb.init(
            resume="must",
            project=args.wandb_project,
            id=wandb_id
        )
    else: 
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.modelname}_{args.config}"
        )
        
    #### seed run
    util_logger.info(f'Setting seed: seed={args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        

def get_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [U.pjoin(output_dir, d) for d in os.listdir(output_dir) 
                   if U.pexists(U.pjoin(output_dir, d, "pytorch_model.bin")) and d.startswith("checkpoint")]
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1] # dir of the last checkpoint

def run_train(args):
    # two default dirs: ckpts and data
    if isinstance(args, dict):
        args = Namespace(**args)

    setup_environ(args) 
    if args.resume and U.pexists(f"{args.ckpt_dir}/{args.config}/{args.modelname}/pretrained"):
        print(f"Model {args.modelname} is already pretrained")
        return
    
    config: GAMConfig =eval(f"{args.config}()")
    model = ModisLMHeadModel(config, dtype=torch.bfloat16, device="cuda") # seems should not be bf16 for tf32 mode
    model.backbone.print_size()
    
    tokenized_datasets, tokenizer = load_datasets(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_tokens=config.param_magnitude*config.training_token_multiplier # suggested by Chinchilla
    num_steps = int(training_tokens / (config.per_device_train_batch_size * args.n_gpus * args.n_nodes * config.context_length))+1

    training_args=TrainingArguments(
        learning_rate=config.learning_rate,
        max_steps=num_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size = config.per_device_train_batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        output_dir=f"{args.ckpt_dir}/{args.config}/{args.modelname}",
        logging_steps=25,
        save_steps=args.save_steps,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        tf32=True,
        ddp_find_unused_parameters=False,  # Set this to False
        # torch_compile=True, # TODO: debug this
        save_total_limit=5,
        report_to="wandb",
    )

    trainer = ModisTrainer(
        model=model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"], 
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
    )
    
    open(f"{args.ckpt_dir}/{args.config}/{args.modelname}/wandb_id.txt", "w").write(wandb.run.id)

    # Automatically resume from the latest checkpoint if it exists
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if args.resume and last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found, starting training from scratch")
        trainer.train()
    trainer.save_model(training_args.output_dir+'/pretrained')
    print(f"Model saved at {training_args.output_dir}/pretrained")

    wandb.finish()

def main(argv):
    """Main run entry point 

    :param argv: 
        The CLI arguments. 
    """
    run_train(argv)
    run_eval(argv)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="test1") # should be named after the agent
    parser.add_argument("--config", type=str, default="GAMConfig_debug")
    parser.add_argument("--resume", type=bool, default=True) # whether resume from the latest checkpoint if there is one, or fully retrain
    parser.add_argument("--n_gpus", type=int, default=6)
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_hf") # adamw_apex_fused is faster but BUGGY
    parser.add_argument("--wandb_project", type=str, default='model_discovery')
    parser.add_argument("--wandb_entity", type=str, default='aristo')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default='')

    args = parser.parse_args()

    main(args)
