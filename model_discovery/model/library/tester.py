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
from model_discovery.model.composer import GAUTree



LIBRARY_PATH = U.pjoin(os.path.dirname(os.path.abspath(__file__)),'core')
print(LIBRARY_PATH)
ckpt_dir=os.environ.get("CKPT_DIR")


def check_tune(scale, model_name, path=None, code=None, check_only=False, cpu_only=False, reformat_only=False,skip_tune=False):
    checker = BuildTool(
        tool_type="checker",
    )
    cfg = eval(f"GAMConfig_{scale}()")

    gabpath = os.path.join(os.environ.get('CKPT_DIR'))

    if code is None:
        if path is None:
            assert model_name in MODEL2CODE, "Model name not found in MODEL2CODE, path not provided as well"
            code=MODEL2CODE[model_name]
            path = U.pjoin(LIBRARY_PATH, model_name)
        else:
            _path = U.pjoin(path, f'{model_name}.py')
            code=U.read_file(_path) # assert model_name is a path

    checkpass,report,code,results = checker.check(code,model_name,True, cpu_only=cpu_only, reformat_only=reformat_only)
    if skip_tune:
        return code
    if not checkpass:
        print(report)
        raise Exception('Model does not pass the checker')
    print('Starting tuning...')
    autocfg = checker.tune(cfg,code,model_name, cpu_only=cpu_only) # cause segment error when using with evo
    if autocfg is None:
        return None
    # autocfg = "autoconfig = { }"
    print('Tuning complete, saving the code with autocfg.')
    code=code+f'\n\n\n{autocfg}\nblock_config=gab_config\nblock_config.update(autoconfig)'
    code+='\n\n\nfrom model_discovery.model.block_registry import BlockRegister\n\nBlockRegister(\n    name="default",\n    config=block_config\n)(GAB)'
    if check_only:
        return code
    with open(U.pjoin(gabpath,'gab.py'),'w') as f:
        f.write(code)
    savedir=U.pjoin(path, 'reports')
    U.mkdir(savedir,exist_ok=True)
    with open(U.pjoin(savedir,f'check_{scale}.json'),'w') as f:
        json.dump(results,f,indent=4)


def run(scale,model_name,args,training_token_multiplier=20,path=None): # do a single verify
    if path is None:
        assert model_name in MODEL2CODE, "Model name not found in MODEL2CODE, path not provided as well"
        path=U.pjoin(LIBRARY_PATH, model_name)
    # with open(U.pjoin(path,'gab.py'),'r') as f:
    #     code=f.read()
    # ckpt_dir = os.environ.get('CKPT_DIR')
    # with open(U.pjoin(ckpt_dir,'gab.py'),'w') as f:
    #     f.write(code)
    args.evoname='LIBRARY_HOLD'
    args.design_id=model_name+'_'+scale
    assert training_token_multiplier>0
    if training_token_multiplier!=20:
        args.design_id+=f'-{training_token_multiplier}x'
    args.scale=scale
    args.ckpt_dir=ckpt_dir
    args.data_dir=os.environ.get("DATA_DIR")
    args.resume=True
    args.training_token_multiplier=training_token_multiplier
    args.logging_steps=10
    # args.port="25869"
    args.tune_lr_in_auto_bs=False
    args.lmeval_batch_size='64'

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
    # spectraladaptivegpt 
    model_name = 'vqhpmemory' 
    # path = None
    tree_dir = None
    # tree_dir = f'/home/junyanc/model_discovery/model_discovery/model/library/core/{model_name}/units'
    # path = f'/home/junyanc/model_discovery/model_discovery/model/library/core/{model_name}/gau'
    
    ckpt_dir = os.environ.get('CKPT_DIR')
    path = U.pjoin(ckpt_dir,'HOLD')

    
    scale = '350M' 
    args = ve_parser.parse_args()



    training_token_multiplier = 20

    if args.mode=='check':
        if tree_dir is not None and U.pexists(tree_dir):
            tree=GAUTree.load_from_base(tree_dir)
            U.mkdir(path)
            with open(U.pjoin(path,model_name+'.py'),'w') as f:
                f.write(tree.compose())
            check_tune(scale, model_name, path)
        else:
            check_tune(scale, model_name, path)
    else:
        run(scale, model_name,args, path=path,training_token_multiplier=training_token_multiplier) # Then run this

