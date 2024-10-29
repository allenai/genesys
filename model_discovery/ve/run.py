''' Run Verification Engine '''

import json
import sys
import torch
import random
import os
import numpy as np
import argparse
import wandb
import time
import logging
from datetime import datetime
import uuid
import shlex
# import transformers
from huggingface_hub import login

from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from accelerate import notebook_launcher
import functools as ft
from argparse import Namespace
import subprocess

from .data_loader import load_datasets
from .modis_trainer import ModisTrainer
from ..configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)
from ..configs.const import *
from ..model.gam import ModisLMHeadModel
from .evaluator import cli_evaluate
from .. import utils as U

from ..model.block_registry import BlockRegister

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True

util_logger = logging.getLogger('model_discovery.run')
util_logger.setLevel(logging.INFO)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import socket

def find_free_port(start_port=25986, max_port=65535):
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))  # Try to bind to the port
                return port  # If successful, return this port
            except OSError:
                continue  # If the port is already in use, try the next one
    raise RuntimeError("No free ports available in the specified range.")



parser = argparse.ArgumentParser()
parser.add_argument("--evoname", type=str, default="evolution_test") # the name of the whole evolution
parser.add_argument("--design_id", type=str, default="test") # evosytem will assign acronym_scale as id
parser.add_argument("--resume", action='store_true', help="Whether to resume from the latest checkpoint if there is one, or fully retrain")
parser.add_argument("--scale", type=str, default='debug') 
parser.add_argument("--n_gpus", type=int, default=torch.cuda.device_count())
parser.add_argument("--n_nodes", type=int, default=1)
parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS)
parser.add_argument("--training_token_multiplier", type=int, default=20) # by default equals to 20 suggested by Chinchilla
parser.add_argument("--optim", type=str, default=DEFAULT_OPTIM) # adamw_apex_fused is faster but BUGGY
parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
parser.add_argument("--wandb_entity", type=str, default=DEFAULT_WANDB_ENTITY)
parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
parser.add_argument("--ckpt_dir", type=str, default=None)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--download_data_only", action='store_true')
parser.add_argument("--logging_steps", type=int, default=DEFAULT_LOG_STEPS)
parser.add_argument("--gab_name", type=str, default='default') ## name of gab block to use 
parser.add_argument("--PERF_PROF_MODE", action='store_true') # Performance profiler mode, used when optimizing training efficiency, will not resume from checkpoint
parser.add_argument("--gradient_accumulation_steps", type=int, default=1) # auto find batch size
# parser.add_argument("--tune_lr_in_auto_bs", type=bool, default=False) # tune lr or tune grad accumulation steps, do not use it as it may change the behavior of training
parser.add_argument("--auto_find_batch_size_hf", type=bool, default=False) # whether use hf auto_find_batch_size (fast but not stable) or custom one
parser.add_argument("--ddp_find_unused_parameters",  type=bool, default=True) # whether use ddp find unused parameters feature in HF Trainer for safer but slower training
parser.add_argument("--RANDOM_TESTING", action='store_true') # whether use random testing

parser.add_argument("--eval_tasks", type=str, default='None')
parser.add_argument("--training_data", type=str, default='None')
parser.add_argument("--tokenizer", type=str, default='None')
parser.add_argument("--context_length", type=str, default='None') # need convert to int

# PATCH for the evolution
parser.add_argument("--mode", type=str, default='test') # Performance profiler mode, used when optimizing training efficiency, will not resume from checkpoint
parser.add_argument("--params", type=str, default='') 
parser.add_argument("--sess_id", type=str, default='') 
parser.add_argument("--cpu_only", action='store_true') 
parser.add_argument("--silent", action='store_true')






########################################### Tools ###########################################


TIME_LOWER={
    '14M':174,
    '31M':437,
    '70M':22.3*60,
    '125M':146.3*60,
    '350M':17*3600,
    '760M':54.3*3600,
    '1300M':137.5*3600,
}


def _explore_setup(args,slow_threshold=5):
    setup(args)
    gab,gab_config = BlockRegister.load_block(args.gab_name)
    free_port = find_free_port()
    util_logger.info(f"Using port for training: {free_port}")
    num_steps=10 # a small number for testing OOM

    time_start = time.perf_counter()
    notebook_launcher(
        run_train, 
        args=(vars(args),gab,gab_config,num_steps), 
        num_processes=args.n_gpus, 
        use_port=free_port,
    )
    time_elapsed = time.perf_counter() - time_start
    scale = args.scale
    n_gpus = args.n_gpus

    config = eval(f"GAMConfig_{args.scale}()")
    time_lower = TIME_LOWER[scale] * 8/n_gpus
    training_tokens = config.reference_size * 20
    num_steps = int(np.ceil(training_tokens / (config.batch_tokens)))
    time_lower = time_lower * 10 / num_steps 

    if time_elapsed > time_lower*slow_threshold: # X times slower than the lower bound
        util_logger.warning(f"Training time is too long: {time_elapsed:.1f} s, expected: {time_lower:.1f} s")
        local_doc = U.read_local_doc()
        if 'too_slow' not in local_doc:
            local_doc['too_slow'] = {}
        local_doc['too_slow'][f'{args.design_id}'] = (time_elapsed,time_lower)
        U.write_local_doc(local_doc)
    else:
        local_record = U.read_local_doc('.record')
        if 'speed_record' not in local_record:
            local_record['speed_record'] = {}
        local_record['speed_record'][f'{args.design_id}'] = (time_elapsed,time_lower)
        U.write_local_doc(local_record,'.record')


# stable but slow
def _auto_tune_setup(args,log_fn=None): # Need to be called before training after models are prepared
    log_fn = log_fn if log_fn else lambda x,y=None: None
    config = eval(f"GAMConfig_{args.scale}()")
    config.training_data = ['cosmopedia-v2']
    args.mode='_explore_setup'
    args_dict = vars(args)
    gradient_accumulation_steps=config.gradient_accumulation_steps
    
    while True:
        log_fn(f"Exploring with gradient_accumulation_steps: {gradient_accumulation_steps}")
        util_logger.info(f"Exploring with gradient_accumulation_steps: {gradient_accumulation_steps}")
        args.gradient_accumulation_steps=gradient_accumulation_steps
        cmd_args = [f"--{key}={value}" if value is not True else f"--{key}" for key, value in args_dict.items() if value is not False and value is not None and value != '']
        cmd = f"python -m model_discovery.ve.run {' '.join(map(shlex.quote, cmd_args))}"

        print(f'Running: {cmd}')

        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        ddp_find_unused_parameters = args.ddp_find_unused_parameters
        if 'Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass.' in process.stderr:
            ddp_find_unused_parameters = False

        if "CUDA out of memory" in process.stderr:
            log_fn(f"CUDA out of memory error occurred with gradient_accumulation_steps: {gradient_accumulation_steps}")
            util_logger.error(f"CUDA out of memory error occurred with gradient_accumulation_steps: {gradient_accumulation_steps}")
            gradient_accumulation_steps *= 2
        elif process.returncode != 0:  # Any other error occurred
            error_msg = process.stderr.strip()
            log_fn(f"Error occurred during setup with gradient_accumulation_steps={gradient_accumulation_steps}:\n{error_msg}")
            util_logger.error(f"Error during setup with gradient_accumulation_steps={gradient_accumulation_steps}:\n{error_msg}")
            raise RuntimeError(f"Setup failed with error:\n{error_msg}")
        else:
            log_fn(f"Test training completed successfully with gradient_accumulation_steps: {gradient_accumulation_steps}")
            util_logger.info(f"Test training completed successfully with gradient_accumulation_steps: {gradient_accumulation_steps}")
            break
    return gradient_accumulation_steps,ddp_find_unused_parameters



########################################### Major Functions ###########################################



def setup(args,log_fn=None) -> None:
    """Sets up the run environment 

    :param args: 
        The global run configuration
    :raises: ValueError 
    """
    log_fn = log_fn if log_fn else lambda x,y=None: None

    log_fn('Setting up the run environment...')

    if not args.data_dir: # use the data dir from the environment by default
        args.data_dir=os.environ.get("DATA_DIR")
    if not args.ckpt_dir:
        args.ckpt_dir=os.environ.get("CKPT_DIR")
    if not os.environ.get("HF_KEY"):
        raise ValueError('Must set HF_KEY')
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError('Must set WANDB_API_KEY')
    if not args.data_dir:
        raise ValueError('Must specify the data directory via `--data_dir`')
    if not args.ckpt_dir:
        raise ValueError('Must specify the checkpoint directory via `--ckpt_dir`')

    # if not os.environ.get("DATA_DIR") or args.data_dir:
    #     util_logger.info(
    #         f'Manually changing the data directory based on input: {args.data_dir}'
    #     )
    #     os.environ["DATA_DIR"] = args.data_dir 
        
    ### make checkpoint dir
    U.mkdir(U.pjoin(args.ckpt_dir,args.evoname,'ve'))
    util_logger.info(f'Creating checkpoint directory: {args.ckpt_dir}/{args.evoname}/ve')

    #### seed run
    util_logger.info(f'Setting seed: seed={args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # ### log into the hf hub 
    # login(os.environ.get("HF_KEY",None))


def before_train(args,log_fn):
    start = time.perf_counter()
    log_fn('Preparing the model...')
    gab,gab_config = BlockRegister.load_block(args.gab_name)
    if args.PERF_PROF_MODE or args.RANDOM_TESTING: # skip the following if in performance profiling mode
        return args,gab,gab_config
        
    ## initialize wandb
    util_logger.info(f'Setting up wandb...')
    global wandb_ids
    wandb_ids=U.load_json(
        f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}/wandb_ids.json"
    )
    if args.resume and 'pretrain' in wandb_ids:
        wandb.init(
            resume="must", 
            project=wandb_ids['project'], 
            entity=wandb_ids['entity'],
            id=wandb_ids['pretrain']['id'],
            name=wandb_ids['pretrain']['name']
        )
    else: 
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.evoname}_{args.design_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
    util_logger.info(f'Time elapsed for setting up wandb: {(time.perf_counter() - start):.1f} s')
    
    # if not args.auto_find_batch_size_hf:    
    log_fn('Auto tuning the gradient accumulation steps...')
    try:
        args.gradient_accumulation_steps,args.ddp_find_unused_parameters = _auto_tune_setup(args,log_fn) # always use it for safety
    except Exception as e:
        util_logger.error(f"Error during auto tuning the gradient accumulation steps: {e}")
        scale=args.design_id.split('_')[-1]
        design=args.design_id[:-len(scale)-1]
        U.log_error_model(design,scale)
        log_fn(f'Evaluation failed with error...','ERROR')
        sys.exit()
    log_fn('Auto tuning the gradient accumulation steps done.')
    return args,gab,gab_config


class LogFnCallback(TrainerCallback):
    def __init__(self, log_fn):
        self.log_fn = log_fn

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get('loss','N/A')
        step = logs.get('step','N/A')
        epoch = logs.get('epoch','N/A')
        lr = logs.get('lr','N/A')
        self.log_fn(f"Training in progress: loss={loss}, step={step}, epoch={epoch}, lr={lr}",'TRAINING')

def run_train(args,gab,gab_config,num_steps=None,log_fn=None) -> None: 
    """Runs the full training pipeline 

    :param args: 
        The global configuration for training.
    """
    log_fn = log_fn if log_fn else lambda x,y=None: None
    with U.CodeTimer("setup model"):
        log_fn('Setting up the model...')
        start=time.perf_counter()
        if isinstance(args, dict):
            args = Namespace(**args)
        config = eval(f"GAMConfig_{args.scale}()")
        model = ModisLMHeadModel(
            config, gab, dtype=torch.bfloat16, device="cuda",
            block_config=gab_config,
            RANDOM_TESTING=args.RANDOM_TESTING
        ) # seems should not be bf16 for tf32 mode
        model.print_size()
        log_fn('Setting up the model done.')

    with U.CodeTimer("loading dataset"):
        log_fn('Loading the dataset...')
        if args.training_data != 'None':
            config.training_data=args.training_data.split(',')
        if args.context_length != 'None':
            config.context_length=int(args.context_length)
        if args.tokenizer != 'None':
            config.tokenizer=args.tokenizer
        tokenized_datasets, tokenizer = load_datasets(config,log_fn)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        if args.download_data_only:
            util_logger.info('Donwloaded data, now stopping...')
            exit('exiting after data download')
        log_fn('Loading the dataset done.')

    log_fn('Setting up the training arguments...')
    num_params = sum(p.numel() for p in model.parameters())
    training_tokens=num_params*args.training_token_multiplier 
    
    if config.per_device_batch_size:
        if num_steps is None:
            num_steps = int(training_tokens / (config.per_device_batch_size * args.n_gpus * args.n_nodes * config.context_length))+1
        per_device_batch_size=config.per_device_batch_size
    else: # auto find bs based on training tokens, can preset gradient_accumulation_steps to avoid too many auto tune steps
        if num_steps is None:
            num_steps = int(np.ceil(training_tokens / (config.batch_tokens)))
        per_device_batch_size=(config.batch_tokens // config.context_length)//args.n_gpus//args.gradient_accumulation_steps

    print(f"Training tokens: {U.strscale(training_tokens)}, num steps: {num_steps}, per device batch size: {per_device_batch_size}, num gpus: {args.n_gpus}")

    training_args=TrainingArguments(
        learning_rate=config.learning_rate,
        max_steps=num_steps,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size = per_device_batch_size * 2,
        # auto_find_batch_size=args.auto_find_batch_size_hf, # unstable! Manually implement it
        gradient_accumulation_steps=args.gradient_accumulation_steps, # use args one, not config one, to use with prep_setup
        optim=args.optim,
        output_dir=f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        tf32=True,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,  # Set this to False
        # torch_compile=True, # TODO: debug this
        save_total_limit=5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={
            "min_lr_rate": 0.1, 
        },
        warmup_ratio=0.02,
        report_to="wandb" if not args.PERF_PROF_MODE else None,
    )
    U.mkdir(training_args.output_dir)
    
    with U.CodeTimer("setting up trainer"):
        trainer = ModisTrainer(
            model=model,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["eval"], 
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            callbacks=[LogFnCallback(log_fn)],
            # tune_lr_in_auto_bs=args.tune_lr_in_auto_bs, # tune lr or tune grad accumulation steps
        )
    print(f'Time elapsed for setting up trainer: {(time.perf_counter() - start):.1f} s')
    
    # class PrintBatchSizeCallback(TrainerCallback):
    #     def on_train_begin(self, args, state, control, **kwargs):
    #         args.max_steps = int(np.ceil(num_steps * per_device_batch_size / state.train_batch_size))
    #         print(f"Auto-determined batch size: {state.train_batch_size}, max steps adjusted: {args.max_steps}")

    # if training_args.auto_find_batch_size:
    #     trainer.add_callback(PrintBatchSizeCallback())

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.PERF_PROF_MODE:
        exec_profiler(trainer)
    else:
        exec_train(args,training_args, trainer,log_fn)


def exec_train(args,training_args, trainer,log_fn):
    # if wandb is defined
    
    if 'wandb_ids' in globals():
        global wandb_ids
        wandb_ids['pretrain']={}
        wandb_ids['pretrain']['id']=wandb.run.id
        wandb_ids['project']=args.wandb_project
        wandb_ids['entity']=args.wandb_entity
        wandb_ids['pretrain']['name'] = wandb.run.name
        U.save_json(wandb_ids,f"{training_args.output_dir}/wandb_ids.json")
    else:
        print('No wandb id found, skip recording wandb metrics.')
    
    log_fn('Starting training...')
    # Automatically resume from the latest checkpoint if it exists
    if args.training_token_multiplier > 0:
        last_checkpoint = U.get_last_checkpoint(training_args.output_dir)
        if args.resume and last_checkpoint:
            util_logger.info(
                f"Resuming training from checkpoint: {last_checkpoint}"
            )
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            util_logger.info(
                "No checkpoint found, starting training from scratch"
            )
            trainer.train()
    else:
        util_logger.info(
            "Training token multiplier is set to 0, skipping training."
        )
    log_fn('Training done.')

    if args.mode=='_explore_setup':
        return
    
    trainer.save_model(training_args.output_dir+'/pretrained')
    util_logger.info(
        f"Model saved at {training_args.output_dir}/pretrained"
    )
    trainer.state.save_to_json(f"{training_args.output_dir}/trainer_state.json")
    log_fn('Saving the model and trainer state done.')

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

def trace_handler(p,export=True):
    print('Profiler results (by CUDA time):')
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print('Profiler results (by CPU time):')
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    if export:
        print('Exporting profiler results')
        output_dir=f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}"
        U.mkdir(f"{output_dir}/profiler")
        p.export_chrome_trace(f"{output_dir}/profiler/trainer_trace_" + str(p.step_num) + ".json") # check chrome://tracing

def exec_profiler(trainer):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank != -1: # CHANGE IT TO 0 TO ENABLE PROFILING or -1 to SKIP PROFILING
        trainer.train()
    else:
        start = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], 
                                    schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=50, repeat=1),
                                    on_trace_ready=trace_handler,
                                    profile_memory=True,
                                    with_stack=True,
                                    use_cuda=True,
                                    with_flops=True,
                                    record_shapes=True) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train()
        print(f'Profiling time: {(time.perf_counter() - start):.1f} s')
        # trace_handler(prof,False) # uncomment if not using on_trace_ready

def after_train(args,log_fn):
    if args.PERF_PROF_MODE or args.RANDOM_TESTING: return
    log_fn('Finishing training and saving results...')
    start=time.perf_counter()
    output_dir=f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}"
    history,system_metrics=get_history(wandb.run.id)
    history.to_csv(f"{output_dir}/train_logs.csv")
    system_metrics.to_csv(f"{output_dir}/system_metrics.csv")
    util_logger.info(f"Training logs saved at {output_dir}")
    log_fn('Saving results done.')
    wandb.finish()
    util_logger.info(f"Time elapsed for finishing training: {(time.perf_counter() - start):.1f} s")

def train(args,log_fn=None):
    log_fn = log_fn if log_fn else lambda x,y=None: None
    if (not args.PERF_PROF_MODE) and args.resume and U.pexists(f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}/pretrained"):
        util_logger.info(f"Model {args.design_id} is already pretrained")
        return
    start = time.perf_counter()
    args,gab,gab_config=before_train(args,log_fn)
    # check_problem(args.design_id,log_fn) # check after testing in before_train
    free_port = find_free_port()
    util_logger.info(f"Using port for training: {free_port}")
    print('Running with args:',args)
    notebook_launcher(
        run_train, 
        args=(vars(args),gab,gab_config,None,log_fn), 
        num_processes=args.n_gpus, 
        use_port=free_port,
    )
    after_train(args,log_fn)
    util_logger.info(f'Training time: {(time.perf_counter() - start):.1f} s')
    log_fn(f'Training done. Total time: {(time.perf_counter() - start):.1f} s')


def get_eval_results(output_dir):
    try:
        dir1=U.pjoin(output_dir, "eval_results")
        dir2=U.pjoin(dir1, os.listdir(dir1)[0])
        dir3=U.pjoin(dir2, os.listdir(dir2)[0])
        result=U.load_json(dir3)
        return result
    except:
        return None

def run_eval(args,log_fn):
    ### NOTE: Remember to uncomment the following 3 lines
    # if args.resume and get_eval_results(f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}"):
    #     print(f"Model {args.design_id} is already evaluated")
    #     return
    print("Evaluation Start")
    log_fn('Setting up the evaluation arguments...')
    cfg=eval(f"GAMConfig_{args.scale}()")
    if not args.RANDOM_TESTING:
        wandb_ids=U.load_json(f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}/wandb_ids.json")
        wandb_ids['evaluate']={}
        wandb_name=f"{args.evoname}_{args.design_id}_eval_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        wandb_ids['evaluate']['name'] = wandb_name
        U.save_json(wandb_ids,f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}/wandb_ids.json")
    
    if args.eval_tasks == 'None':
        eval_tasks = ",".join(cfg.eval_tasks)
    else:
        eval_tasks = args.eval_tasks

    sys.argv = [
        "",
        "--model", "modis",
        "--model_args", f"pretrained={args.evoname}/{args.scale}/{args.design_id},ckpt_dir={args.ckpt_dir},gab_name={args.gab_name}",
        "--tasks", eval_tasks, 
        # "--device", "cuda",
        "--batch_size", f"auto",
        "--max_batch_size", f"{cfg.eval_batch_size}",
        "--output_path", f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}/eval_results",
        "--cache_requests", "true", # refresh for debugging, true for normal 
    ]
    if not args.RANDOM_TESTING:
        sys.argv += [
            "--wandb_args", f"project={args.wandb_project},entity={args.wandb_entity},name={wandb_name}"
        ]
    log_fn('Setting up the evaluation arguments done. Preparing the model...')
    gab,gab_config=BlockRegister.load_block(args.gab_name)
    free_port = find_free_port()
    util_logger.info(f"Using port for evaluation: {free_port}")
    log_fn('Preparation done. launching evaluation...')
    notebook_launcher(cli_evaluate, args=(None,gab,gab_config,log_fn), num_processes=args.n_gpus, use_port=free_port)
    
def evalu(args,log_fn=None):
    log_fn = log_fn if log_fn else lambda x,y=None: None
    if args.PERF_PROF_MODE: return
    # check_problem(args.design_id,log_fn)
    start = time.perf_counter()
    log_fn('Evaluating the model...')
    try:
        run_eval(args,log_fn)
    except Exception as e:
        util_logger.error(f"Error during evaluation: {e}")
        scale=args.design_id.split('_')[-1]
        design=args.design_id[:-len(scale)-1]
        U.log_error_model(design,scale)
        log_fn(f'Evaluation failed with error...','ERROR')
        sys.exit()
    util_logger.info(f"Evaluation time: {(time.perf_counter() - start):.1f} s")
    log_fn(f'Evaluation done. Total time: {(time.perf_counter() - start):.1f} s')

def get_history(run_id, project_path = "aristo/model_discovery"):
    api = wandb.Api()
    run = api.run(f"{project_path}/{run_id}")
    history = run.history()
    system_metrics = run.history(stream='systemMetrics')
    for i in ['temp','powerWatts','_timestamp','cpu_percent','powerPercent','network','memoryAllocatedBytes','_wandb','disk','memory.rssMB','memory.availableMB']:
        system_metrics=system_metrics[[col for col in system_metrics.columns if i not in col]]
    system_note={
        "system.memory": "System Memory Utilization (%)",
        "system.proc.memory.percent": "Process Memory In Use (non-swap) (%)",
        "system.cpu": "Process CPU Utilization (%)",
        "system.proc.cpu.threads": "Process CPU Threads In Use",
    }
    gpu_ids=[]
    for i in system_metrics.columns:
        if 'gpu' in i:
            gpu_id=i.split('.')[2]
            if gpu_id not in gpu_ids:
                gpu_ids.append(gpu_id)
    for i in gpu_ids:
        system_metrics.rename(columns={f"system.gpu.{i}.gpu": f"GPU {i} Utilization (%)",
                                       f"system.gpu.{i}.memoryAllocated": f"GPU {i} Memory Allocated (%)",
                                       f"system.gpu.{i}.memory": f"GPU {i} Time Spent Accessing Memory (%)"},inplace=True)
    system_metrics.rename(columns=system_note,inplace=True)
    return history,system_metrics


def get_history_report(wandb_ids):
    report={}
    try:
        run_id=wandb_ids['pretrain']['id']
        wandb_entity=wandb_ids['entity']
        wandb_project=wandb_ids['project']
        history,system_metrics=get_history(
            run_id,
            project_path=f"{wandb_entity}/{wandb_project}"
        )
        report["training_record.csv"]=str(history.to_csv(index=False))
        report["system_metrics.csv"]=str(system_metrics.to_csv(index=False))
    except Exception as e:
        util_logger.error(f"Error getting history report: {e}")
        pass
    return report


def report(args,log_fn=None) -> dict:
    """Returns the training report 

    :param args: 
        The global training configuration. 
    """
    log_fn = log_fn if log_fn else lambda x,y=None: None
    if args.PERF_PROF_MODE: return
    # check_problem(args.design_id,log_fn)
    outdir=f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}"
    if args.resume and U.pexists(f"{outdir}/report.json"):
        util_logger.info(f"Report already exists at {outdir}/report.json")
        log_fn(f'Report already exists at {outdir}/report.json')
        return
    log_fn('Generating reports...')
    report={}
    wandb_ids=U.load_json(f"{outdir}/wandb_ids.json")
    report["wandb_ids.json"]=wandb_ids
    report.update(get_history_report(wandb_ids))

    trainer_state=U.load_json(f"{outdir}/trainer_state.json")
    eval_results=get_eval_results(outdir)

    # trainer_state.pop("log_history")
    # only keep the last log_history, detail can be found in wandb
    if 'log_history' in trainer_state:
        trainer_state['log_history'] = [trainer_state['log_history'][-1]]
    if 'stateful_callbacks' in trainer_state:
        trainer_state.pop("stateful_callbacks")
    for i in ['upper_git_hash','transformers_version','pretty_env_info','git_hash']:
        if i in eval_results:
            eval_results.pop(i)
    
    report["trainer_state.json"]=trainer_state
    report["eval_results.json"]=eval_results
    
    with open(f"{outdir}/report.json", 'w') as report_out:
        report_out.write(json.dumps(report,indent=4))
        
    #json.dump(report, open(f"{outdir}/report.json", 'w'), indent=4)
    util_logger.info(f"Report saved at {outdir}/report.json")
    log_fn(f'Report generated.')
    return report


def check_problem(design_id,log_fn):
    local_doc = U.read_local_doc()
    if f'{design_id}' in local_doc.get('too_slow',{}):
        time_elapsed,time_lower = local_doc['too_slow'][f'{design_id}']
        log_fn(f'{design_id} is too slow in this machine: {time_elapsed:.1f} s, lower bound: {time_lower:.1f} s x 5, skipping...','EXIT')
        sys.exit()
    scale=design_id.split('_')[-1]
    design=design_id[:-len(scale)-1]
    if design in local_doc.get('error_models',{}):
        log_fn(f'{design_id} is too slow in this machine: {local_doc["error_models"][design]} x 5, skipping...','EXIT')
        sys.exit()

def main(args,log_fn=None):
    """Main run entry point 

    :param args: 
        The CLI arguments. 
    """
    log_fn = log_fn if log_fn else lambda x,y=None: None

    check_problem(args.design_id,log_fn) # check before starting
    
    start = time.perf_counter()
    print(f"Starting run with args: {args}")
    log_fn('Starting verification...','BEGIN')
    if args.RANDOM_TESTING:
        args.evoname = "random"
        args.design_id = "random"
        args.scale = "14M"
        args.gab_name = "random"
        args.training_token_multiplier = 0
        args.resume = False
        print("Running random testing...")
        

    setup(args,log_fn)
    train(args,log_fn)
    evalu(args,log_fn)
    report(args,log_fn)
    log_fn(f'Done. Total time: {(time.perf_counter() - start):.1f} s','EXIT')
    # util_logger.info(f"Total time: {(time.perf_counter() - start):.1f} s")
    print(f"Total time: {(time.perf_counter() - start):.1f} s")
    

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.mode=='test':
        args.evoname = "ve_test"
        args.design_id = "test"
        args.resume = True
        # args.n_gpus = 1
        # args.PERF_PROF_MODE = True
        main(args)
    elif args.mode=='_explore_setup':
        _explore_setup(args)


