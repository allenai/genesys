
import sys
import time
from datetime import datetime

import model_discovery.utils as U


from accelerate import notebook_launcher





def run_eval(args,log_fn):
    ### NOTE: Remember to uncomment the following 3 lines
    # if args.resume and get_eval_results(f"{args.ckpt_dir}/{args.evoname}/ve/{args.design_id}"):
    #     print(f"Model {args.design_id} is already evaluated")
    #     return
    print("Evaluation Start")
    log_fn('Setting up the evaluation arguments...')
    HF_MODE = args.hf_config != 'none' # TODO: check if it matters in eval
    if HF_MODE:
        raise NotImplementedError('HF mode is not implemented yet')
        _,cfg=hf_config_from_args(args.hf_config)
    else:
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
    log_fn = log_fn if log_fn else lambda x,y='RUNNING': print(f'[{y}] {x}')
    start = time.perf_counter()
    log_fn('Evaluating the model...')
    run_eval(args,log_fn)
    print(f"Evaluation time: {(time.perf_counter() - start):.1f} s")
    log_fn(f'Evaluation done. Total time: {(time.perf_counter() - start):.1f} s')



