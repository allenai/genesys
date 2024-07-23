import sys
import inspect

from model_discovery.agents.checker import *

from model_discovery.model.library import MODEL2CLASS,MODEL2CODE

from model_discovery.configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

from exec_utils import (
    BuildTool
)



class ModelTester:
    def __init__(self):
        self.checker = BuildTool(
            tool_type="checker",
        )

    def check(self, cfg, code, design_name):
        checkpass,check_report,code = self.checker.check(cfg,code,design_name)




if __name__ == "__main__":
    tester = ModelTester()

    scale = sys.argv[1]
    cfg = eval(f"GAMConfig_{scale}()")
    model_name = sys.argv[2]
    model_code = MODEL2CODE[model_name]
    tester.check(cfg, model_code, model_name)

