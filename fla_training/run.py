# -*- coding: utf-8 -*-
import os
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, Trainer
import numpy as np
import torch

# import fla  # noqa # cause trouble when using with hf hub
import logging
from flame.data import DataCollatorForLanguageModeling
from flame.logging import LogCallback, get_logger
from flame.parser import get_train_args

logger = get_logger(__name__)

from model_discovery.ve.data_loader import load_datasets

from model_discovery.configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)
# from fla_training.hf_configs import hf_config_from_args
import model_discovery.utils as U



util_logger = logging.getLogger('model_discovery.run')
util_logger.setLevel(logging.INFO)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

TRAINING_TOKEN_MULTIPLIER = 100



def setup(args,log_fn=None) -> None:
    """Sets up the run environment 

    :param args: 
        The global run configuration
    :raises: ValueError 
    """
    log_fn = log_fn if log_fn else lambda x,y='RUNNINNG': print(f'[{y}] {x}')

    log_fn('Setting up the run environment...')

    args.evoname = 'HF_BASELINES'
    args.data_dir=os.environ.get("DATA_DIR")
    args.ckpt_dir=os.environ.get("CKPT_DIR")
    if not os.environ.get("HF_KEY"):
        raise ValueError('Must set HF_KEY')
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError('Must set WANDB_API_KEY')

    ### make checkpoint dir
    U.mkdir(U.pjoin(args.ckpt_dir,args.evoname,'ve'))
    util_logger.info(f'Creating checkpoint directory: {args.ckpt_dir}/{args.evoname}/ve')

    args.design_id=args.model_name_or_path
    return args


def load_model(model_type,scale):
    if model_type == 'mamba2':
        if scale == '350M':
            config = AutoConfig.from_pretrained('state-spaces/mamba2-370m')
            print(config)
            return AutoModelForCausalLM.from_config(config)
    else:
        raise ValueError(f'Unknown model type: {model_type}')


def load_cfg(scale):
    if scale == '1300M':
        return GAMConfig_1300M()
    elif scale == '350M':
        return GAMConfig_350M()
    else:
        raise ValueError(f'Unknown scale: {scale}')


def main():
    args = get_train_args()
    logger.info(args)

    args = setup(args)

    model_type,scale = args.model_name_or_path.split('_')
    cfg = load_cfg(scale)

    dataset, tokenizer = load_datasets(cfg)

    if args.from_config:
        logger.info("All model params are randomly initialized for from-scratch training.")
        # model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model_name_or_path))
        # model = AutoModelForCausalLM.from_config(model_config)
        model = load_model(model_type,scale)
    else:
        logger.info(f"Loading pretrained checkpoint {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.train() 

    trainable_params, all_param = model.num_parameters(only_trainable=True), model.num_parameters()
    logger.info(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")
    logger.info(f"{tokenizer}\n{model}\n{model.config}")

    # logger.info(f"Loading the `{args.split}` split directly from the cache {args.cache_dir}...")
    # dataset = load_from_disk(args.cache_dir)
    logger.info(f"{dataset}")
    logger.info(f"Shuffling the dataset with seed {args.seed}")
    dataset = dataset.shuffle(seed=args.seed)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    if args.lr_scheduler_type == 'warmup_stable_decay':
        args.lr_scheduler_kwargs = {
            'num_stable_steps': args.max_steps * 0.9 - args.warmup_steps,
            'num_decay_steps': args.max_steps * 0.1
        }
    
    n_gpus = torch.cuda.device_count()
    training_tokens=trainable_params*TRAINING_TOKEN_MULTIPLIER
    num_steps = int(np.ceil(training_tokens / (cfg.batch_tokens)))
    per_device_batch_size=(cfg.batch_tokens // cfg.context_length)//n_gpus//cfg.gradient_accumulation_steps

    print(f"Training tokens: {U.strscale(training_tokens)}, num steps: {num_steps}, per device batch size: {per_device_batch_size}, num gpus: {n_gpus}")

    args.max_steps = num_steps
    args.per_device_train_batch_size = per_device_batch_size
    args.gradient_accumulation_steps = cfg.gradient_accumulation_steps
    args.learning_rate=cfg.learning_rate
    args.output_dir=f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}"
    args.logging_steps=5
    args.save_steps=50
    args.save_total_limit=5
    args.dataloader_num_workers=16
    args.dataloader_pin_memory=True
    args.tf32=True
    # args.ddp_find_unused_parameters=args.ddp_find_unused_parameters  # Set this to False

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        # eval_dataset=tokenized_datasets["eval"], 
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        callbacks=[LogCallback()],
    )

    args.output_dir = f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}"
    U.mkdir(args.output_dir)

    results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)
    trainer.state.save_to_json(f"{trainer.args.output_dir}/trainer_state.json")

    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
