import sys

from exec_utils import build_config
from exec_utils.register import Registry
from exec_utils.param import (
    ModuleParams,
    RootParams,
    ParamField,
)

@Registry(
    resource_type="config",
    name="pipeline_runner"
)
class RunnerParams(ModuleParams):
    """Utilities for running pipelines 
    
    :param prompt_type: 
        The type of pipeline to run. 

    """
    pipeline_type: str = ParamField(
        default='',
        metadata={"help" : 'The type of pipeline to run'}
    )
    

def main(argv):
    """Runs pipelines 

    :param argv:
        Incoming parameters to run a pipeline. 
    """
    config = build_config(argv)
    pipeline = Registry.build_model("pipeline_type",config)
    
    try: 
        pipeline()
    except Exception as e:
        raise e

if __name__ == "__main__":
    main(sys.argv[1:])    
