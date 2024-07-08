import torch
import os
import argparse
import wandb

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
    GAMConfig_10M
)
from .model.gam import ModisLMHeadModel

from . import utils as U

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint")]
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1]

def run(args) -> None:
    """Runs the trainer 

    :param args: 
        The global configuration 
    """
    ### log into huggingface
    login(os.environ.get("HF_KEY",None))

    ### set up wandb
    if not os.environ["DATA_DIR"]:
        raise ValueError(
            f'Must specify data directory'
        )
    
    if isinstance(args, dict):
        args = Namespace(**args)
    config: GAMConfig = eval(f"{args.config}()")

    # seems should not be bf16 for tf32 mode
    model = ModisLMHeadModel(
        config,
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    model.backbone.print_size()

    # # Iterate over the model's parameters and print their types
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Type: {param.dtype}")
        
    tokenized_datasets, tokenizer = load_datasets(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    #exit('exiting')
    
    training_tokens=config.param_magnitude*config.training_token_multiplier # suggested by Chinchilla
    num_steps = int(training_tokens / (config.per_device_train_batch_size * args.n_gpus * args.n_nodes * config.context_length))+1

    training_args=TrainingArguments(
        learning_rate=config.learning_rate,
        max_steps=num_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        output_dir=f"ckpts/{args.config}/{args.modelname}",
        logging_steps=50,
        save_steps=500,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        tf32=True,
        ddp_find_unused_parameters=False,  # Set this to False
        # torch_compile=True, # TODO: debug this
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

    # Automatically resume from the latest checkpoint if it exists
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if args.resume and last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found, starting training from scratch")
        trainer.train()
    trainer.save_model(training_args.output_dir+'/pretrained',False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="GPT-2") # should be named after the agent
    parser.add_argument("--config", type=str, default="GAMConfig_10M")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--resume", type=bool, default=False) # whether resume from the latest checkpoint if there is one
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_hf") # adamw_apex_fused 
    args = parser.parse_args()

    run(vars(args))

    #wandb.init(project="modis", name=f"{args.modelname}_{args.config}")
    #notebook_launcher(run, args=(vars(args),), num_processes=args.n_gpus)
