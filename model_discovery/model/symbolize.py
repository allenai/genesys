''' Symbolic Representation and Operations of GAB '''


from .library import MODEL2CODE

from torch.fx import symbolic_trace
from exec_utils import (
    BuildTool
)
from ..configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)
from .loader import reload_gam



def load_gab(model_name: str,scale='14M'):
    code=MODEL2CODE[model_name]
    checker = BuildTool(
        tool_type="checker",
    )
    try:
        checkpass,gab_code = checker._check_format_and_reformat(gab_code)
        assert checkpass
    except AssertionError as e:
        print('Model does not pass the format checker')
        raise e
    exec(code,globals())
    cfg = eval(f"GAMConfig_{scale}()")
    glm,_ = reload_gam(cfg,gab_code,model_name)
    gam = glm.backbone
    gab = gam.blocks[0].gab
    return gab



if __name__ == '__main__':
    sym=load_gab('rwkv6')
    print(sym)

    