import os,sys
import json

from model_discovery.agents.roles.checker import *

from model_discovery.model.library import MODEL2CODE

from model_discovery.configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

from exec_utils import (
    BuildTool
)


from model_discovery.ve.run import main as ve_main
from model_discovery.ve.run import parser as ve_parser
from model_discovery import utils as U


LIBRARY_PATH = U.pjoin(os.path.dirname(os.path.abspath(__file__)),'base')
print(LIBRARY_PATH)
ckpt_dir=os.environ.get("CKPT_DIR")


def check_tune(scale, model_name, path=None):
    checker = BuildTool(
        tool_type="checker",
    )
    cfg = eval(f"GAMConfig_{scale}()")

    if path is None:
        assert model_name in MODEL2CODE, "Model name not found in MODEL2CODE, path not provided as well"
        code=MODEL2CODE[model_name]
        path = U.pjoin(LIBRARY_PATH, model_name)
    else:
        code=U.read_file(U.pjoin(path, f'{model_name}.py')) # assert model_name is a path

    checkpass,report,code,results = checker.check(cfg,code,model_name,True)
    if not checkpass:
        print(report)
        raise Exception('Model does not pass the checker')
    autocfg = checker.tune(cfg,code,model_name)
    print('Tuning complete, saving the code with autocfg.')
    code=code+f'\n\n\n{autocfg}\nblock_config=gab_config\nblock_config.update(autoconfig)'
    code+='\n\n\nfrom .block_registry import BlockRegister\n\nBlockRegister(\n    name="default",\n    config=block_config\n)(GAB)'
    with open(U.pjoin(path,'gab.py'),'w') as f:
        f.write(code)
    savedir=U.pjoin(path, 'reports')
    U.mkdir(savedir,exist_ok=True)
    with open(U.pjoin(savedir,f'check_{scale}.json'),'w') as f:
        json.dump(results,f,indent=4)


def run(scale,model_name,args,training_token_multiplier=20,path=None): # do a single verify
    if path is None:
        assert model_name in MODEL2CODE, "Model name not found in MODEL2CODE, path not provided as well"
        path=U.pjoin(LIBRARY_PATH, model_name)
    with open(U.pjoin(path,'gab.py'),'r') as f:
        code=f.read()
    with open('/home/junyanc/model_discovery/model_discovery/model/gab.py','w') as f:
        f.write(code)
    args.evoname='LIBRARY_HOLD'
    args.design_id=model_name+'_'+scale
    assert training_token_multiplier>0
    if training_token_multiplier!=20:
        args.design_id+=f'-{training_token_multiplier}x'
    args.scale=scale
    args.ckpt_dir=ckpt_dir
    args.data_dir=os.environ.get("DATA_DIR")
    args.resume=False
    args.training_token_multiplier=training_token_multiplier
    args.logging_steps=10
    args.port="25869"
    args.tune_lr_in_auto_bs=False

    # args.n_gpus = 1 # use it for the first time setup and data loading

    reportdir=f"{ckpt_dir}/{args.evoname}/ve/{args.design_id}/report.json"
    if not os.path.exists(reportdir):
        ve_main(args)
    report=U.load_json(reportdir)
    savedir=f"{LIBRARY_PATH}/{model_name}/reports"
    U.mkdir(savedir,exist_ok=True)
    report_name=f"report_{scale}.json"
    if training_token_multiplier!=20:
        tail=f"{training_token_multiplier}x"
        report_name=f"report_{scale}-{tail}.json"
    U.save_json(report,U.pjoin(savedir,report_name))


if __name__ == "__main__":
    model_name = 'gpt2' 
    path = '/home/junyanc/model_discovery/model_discovery/model/library/core/gpt2/gau'
    scale = '14M' 
    args = ve_parser.parse_args()

    if args.mode=='check':
        check_tune(scale, model_name, path)
    else:
        run(scale, model_name,args, path=path) # Then run this

