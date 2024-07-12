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

from .data_loader import load_datasets
from .modis_trainer import ModisTrainer
from ..configs.gam_config import (
    GAMConfig,
    GAMConfig_10M,
    GAMConfig_debug
)
from ..model.gam import ModisLMHeadModel
from .evaluator import cli_evaluate
from .. import utils as U

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

util_logger = logging.getLogger('model_discovery.run')


def setup(args) -> None:
    """Sets up the run environment 

    :param args: 
        The global run configuration
    :raises: ValueError 
    """
    if not args.data_dir: # use the data dir from the environment by default
        args.data_dir=os.environ.get("DATA_DIR")
    if not args.ckpt_dir:
        args.ckpt_dir=os.environ.get("CKPT_DIR")
    if not os.environ.get("HF_KEY"):
        raise ValueError('Must set KH_KEY!')
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError('Must set WANDB_API_KEY')
    if not args.data_dir:
        raise ValueError("Must set data_dir")
    if not args.data_dir:
        raise ValueError('Must specify the checkpoint directory via `--ckpt_dir`')

    if not os.environ.get("DATA_DIR") or args.data_dir:
        util_logger.info(
            f'Manually changing the data directory based on input: {args.data_dir}'
        )
        os.environ["DATA_DIR"] = args.data_dir 
        
    ### make checkpoint dir
    U.mkdir(args.ckpt_dir)
    util_logger.info(f'Creating checkpoint directory: {args.ckpt_dir}')

    #### seed run
    util_logger.info(f'Setting seed: seed={args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    ### log into the hf hub 
    login(os.environ.get("HF_KEY",None))


def before_train(args):
    if args.PERF_PROF_MODE: return # skip the following if in performance profiling mode

    ## initialize wandb
    util_logger.info(f'Setting up wandb...')
    global wandb_ids
    wandb_ids=U.load_json(
        f"{args.ckpt_dir}/{args.config}/{args.modelname}/wandb_ids.json"
    )
    if args.resume and 'pretrain' in wandb_ids:
        wandb.init(resume="must", project=args.wandb_project, id=wandb_ids['pretrain'])
    else: 
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.modelname}_{args.config}"
        )
    

def run_train(args) -> None:
    """Runs the full training pipeline 

    :param args: 
        The global configuration for training.
    """
    if isinstance(args, dict):
        args = Namespace(**args)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config: GAMConfig = eval(f"{args.config}()")
    model = ModisLMHeadModel.from_config(
        config,
        dtype=torch.bfloat16, # TODO: allow for other dtypes
        device="cuda" if torch.cuda.is_available() else "cpu",
        gab_name=args.gab_name
    )
    model.backbone.print_size()
    
    tokenized_datasets, tokenizer = load_datasets(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    if args.download_data_only:
        util_logger.info('Donwloaded data, now stopping...')
        exit('exiting after data download')
        
    num_params = sum(p.numel() for p in model.parameters())
    training_tokens=num_params*config.training_token_multiplier # suggested by Chinchilla
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
        report_to="wandb" if not args.PERF_PROF_MODE else None,
    )

    trainer = ModisTrainer(
        model=model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"], 
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
    )

    if args.PERF_PROF_MODE:
        exec_profiler(trainer)
    else:
        exec_train(training_args, trainer)


def exec_train(training_args, trainer):
    global wandb_ids
    wandb_ids['pretrain']=wandb.run.id
    U.save_json(wandb_ids,f"{training_args.output_dir}/wandb_ids.json")
    
    # Automatically resume from the latest checkpoint if it exists
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

    trainer.save_model(training_args.output_dir+'/pretrained')
    util_logger.info(
        f"Model saved at {training_args.output_dir}/pretrained"
    )
    trainer.state.save_to_json(f"{training_args.output_dir}/trainer_state.json")

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

def trace_handler(p):
    print('Profiler results (by CUDA time):')
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print('Profiler results (by CPU time):')
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print('Exporting profiler results')
    output_dir=f"{args.ckpt_dir}/{args.config}/{args.modelname}"
    p.export_chrome_trace(f"{output_dir}/profiler/trainer_trace_" + str(p.step_num) + ".json") # check chrome://tracing

def exec_profiler(trainer):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank != -1: # CHANGE IT TO 0 TO ENABLE PROFILING or -1 to just test running
        trainer.train()
    else:
        start = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], 
                                    # schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=30, repeat=1),
                                    # on_trace_ready=trace_handler,
                                    profile_memory=True,
                                    with_stack=True,
                                    with_flops=True,
                                    record_shapes=True) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train()
        print(f'Profiling time: {(time.perf_counter() - start):.1f} s')
        trace_handler(prof) # uncomment if not using on_trace_ready

def after_train(args):
    if args.PERF_PROF_MODE: return
    output_dir=f"{args.ckpt_dir}/{args.config}/{args.modelname}"
    history,system_metrics=get_history(wandb.run.id)
    history.to_csv(f"{output_dir}/train_logs.csv")
    system_metrics.to_csv(f"{output_dir}/system_metrics.csv")
    util_logger.info(f"Training logs saved at {output_dir}")
    wandb.finish()

def train(args):
    if (not args.PERF_PROF_MODE) and args.resume and U.pexists(f"{args.ckpt_dir}/{args.config}/{args.modelname}/pretrained"):
        util_logger.info(f"Model {args.config}/{args.modelname} is already pretrained")
        return
    start = time.perf_counter()
    before_train(args)
    notebook_launcher(run_train, args=(vars(args),), num_processes=args.n_gpus)
    after_train(args)
    util_logger.info(f'Training time: {(time.perf_counter() - start):.1f} s')


def get_eval_results(output_dir):
    try:
        dir1=U.pjoin(output_dir, "eval_results")
        dir2=U.pjoin(dir1, os.listdir(dir1)[0])
        dir3=U.pjoin(dir2, os.listdir(dir2)[0])
        result=U.load_json(dir3)
        return result
    except:
        return None

def run_eval(args):
    if args.resume and get_eval_results(f"{args.ckpt_dir}/{args.config}/{args.modelname}"):
        print(f"Model {args.config}/{args.modelname} is already evaluated")
        return
    print("Evaluation Start")
    cfg=eval(f"{args.config}()")
    sys.argv = [
        "",
        "--model", "modis",
        "--model_args", f"pretrained={args.config}/{args.modelname},ckpt_dir={args.ckpt_dir},gab_name={args.gab_name}",
        "--tasks", ",".join(cfg.eval_tasks), 
        # "--device", "cuda",
        "--batch_size", f"{cfg.eval_batch_size}",
        "--output_path", f"{args.ckpt_dir}/{args.config}/{args.modelname}/eval_results",
        "--cache_requests", "true",
        # "--wandb_args", "project=modis",
    ]
    notebook_launcher(cli_evaluate, num_processes=args.n_gpus)
    
def evalu(args):
    start = time.perf_counter()
    run_eval(args)
    util_logger.info(f"Evaluation time: {(time.perf_counter() - start):.1f} s")

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

def report(args) -> dict:
    """Returns the training report 

    :param args: 
        The global training configuration. 
    """
    outdir=f"{args.ckpt_dir}/{args.config}/{args.modelname}"
    if args.resume and U.pexists(f"{outdir}/report.json"):
        util_logger.info(f"Report already exists at {outdir}/report.json")
        return
    run_id=U.load_json(f"{outdir}/wandb_ids.json")['pretrain']
    history,system_metrics=get_history(
        run_id,
        project_path=f"{args.wandb_entity}/{args.wandb_project}"
    )
    trainer_state=U.load_json(f"{outdir}/trainer_state.json")
    eval_results=get_eval_results(outdir)

    trainer_state.pop("log_history")
    trainer_state.pop("stateful_callbacks")
    for i in ['upper_git_hash','transformers_version','pretty_env_info','git_hash']:
        eval_results.pop(i)
    
    report={
        "training_record.csv":str(history.to_csv(index=False)),
        "system_metrics.csv":str(system_metrics.to_csv(index=False)),
        "trainer_state.json": trainer_state,
        "eval_results.json": eval_results,
    }
    with open(f"{outdir}/report.json", 'w') as report_out:
        report_out.write(json.dumps(report,indent=4))
    with open(f'{args.ckpt_dir}/metrics.json','w') as json_out: # Q: why do twice? 
        json_out.write(json.dumps(report,indent=4))
        
    #json.dump(report, open(f"{outdir}/report.json", 'w'), indent=4)
    util_logger.info(f"Report saved at {outdir}/report.json")
    
    return report

def main(args):
    """Main run entry point 

    :param args: 
        The CLI arguments. 
    """
    start = time.perf_counter()
    setup(args)
    train(args)
    evalu(args)
    report(args)
    util_logger.info(f"Total time: {(time.perf_counter() - start):.1f} s")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="test") # should be named after the agent
    parser.add_argument("--config", type=str, default="GAMConfig_debug")
    parser.add_argument("--resume", type=bool, default=True) # whether resume from the latest checkpoint if there is one, or fully retrain
    parser.add_argument("--n_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_hf") # adamw_apex_fused is faster but BUGGY
    parser.add_argument("--wandb_project", type=str, default='model_discovery')
    parser.add_argument("--wandb_entity", type=str, default='aristo')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default='')
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--download_data_only", action='store_true')
    parser.add_argument("--gab_name", type=str, default='default') ## name of gab block to use 
    parser.add_argument("--PERF_PROF_MODE", type=bool, default=False) # Performance profiler mode, used when optimizing training efficiency, will not resume from checkpoint
    
    args = parser.parse_args()
    
    main(args)

