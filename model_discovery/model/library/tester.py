import os,sys

from model_discovery.agents.checker import *

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


def check_tune(scale, model_name):
    checker = BuildTool(
        tool_type="checker",
    )
    cfg = eval(f"GAMConfig_{scale}()")
    code=MODEL2CODE[model_name]
    checkpass,report,code = checker.check(cfg,code,model_name)
    if not checkpass:
        print(report)
        raise Exception('Model does not pass the checker')
    autocfg = checker.tune(cfg,code,model_name)
    # U.save_json(autocfg,f"{LIBRARY_PATH}/{model_name}/autocfg.json")
    code=code+f'\n\n\n{autocfg}\nblock_config=gab_config\nblock_config.update(autoconfig)'
    code+='\n\n\nfrom .block_registry import BlockRegister\n\nBlockRegister(\n    name="default",\n    config=block_config\n)(GAB)'
    with open(U.pjoin(LIBRARY_PATH,model_name,'gab.py'),'w') as f:
        f.write(code)

def run(scale,model_name,args): # do a single verify
    with open(U.pjoin(LIBRARY_PATH,model_name,'gab.py'),'r') as f:
        code=f.read()
    with open('/home/junyanc/model_discovery/model_discovery/model/gab.py','w') as f:
        f.write(code)
    args.evoname='LIBRARY_HOLD'
    args.design_id=model_name+'_'+scale
    args.config=f'GAMConfig_{scale}'
    args.ckpt_dir=ckpt_dir
    args.data_dir=os.environ.get("DATA_DIR")
    args.resume=True
    args.training_token_multiplier=20
    args.logging_steps=10
    # args.n_gpus=4
    args.port="25869"
    reportdir=f"{ckpt_dir}/{args.evoname}/ve/{args.design_id}/report.json"
    if not os.path.exists(reportdir):
        ve_main(args)
    report=U.load_json(reportdir)
    savedir=f"{LIBRARY_PATH}/{model_name}/reports"
    U.mkdir(savedir)
    U.save_json(report,U.pjoin(savedir,f"report_{scale}.json"))


if __name__ == "__main__":

    scale = '14M' #sys.argv[1]
    model_name = 'retnet' # sys.argv[2]
    args = ve_parser.parse_args()

    if args.mode=='check':
        check_tune(scale, model_name)
    else:
        run(scale, model_name,args) # Then run this

