from __future__ import annotations 

import time
import random
import os
import json
import logging

from copy import deepcopy
from contextlib import contextmanager
import signal
import time
from typing import (
    List,
    Any,
    Dict,
    Type
)
from tqdm import tqdm

from ..aliases import (
    PathOrStr,
    ConfigType
)
from ..register import Registry
from ..base import ConfigurablePipelineUtil
from ..factory import BuildTool
from ..param import (
    ModuleParams,
    RootParams,
    ParamField,
)

util_logger = logging.getLogger(
    'exec_utils.pipelines.python_execute_python'
)

def read_data(data_path: str) -> List[Dict[str,str]]:
    """Reads the code data 

    :param data_path: 
        The path of the code data. 
    """
    if not data_path or not os.path.isfile(data_path):
        raise ValueError(f'Unknown code data: {data_path}')

    data = []
    skip = 0
    
    with open(data_path) as my_data:
        for line in my_data:
            json_line = json.loads(line.strip())
            if "code" not in json_line:
                util_logger.warning(f'Line found without code attribute, skipping')
                skip += 1
                continue 
            
            data.append(json_line)

    util_logger.info(f'Parsed data with {len(data)} instances, {skip} items skipped...')
    return data


def execute_code(code_executor,data: Dict[str,str]) -> None:
    """Executes code based on the provided data 

    :param code_executor: 
        The code execution agent. 
    :param data: 
        The data with the provided code. 
    """
    pass

@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f'block timedout after {duration} seconds')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


@Registry(
    resource_type="pipeline_type",
    name="python_execution_pipeline",
    cache=None,
)
class PythonExecution(ConfigurablePipelineUtil):

    def __init__(self,code_agent,config: ConfigType) -> None:
        self.code_agent = code_agent
        self.config = config
        
    @classmethod
    def from_config(cls,config: ConfigType):
        """Builds the prompt pipeline from configuration 

        :param config: 
            The global configuration for prompting. 
        """
        code_agent = BuildTool(config)
        return cls(code_agent,config) 

    def run_pipeline(self,*args,**kwargs) -> None:
        """Runs the code execution pipeline"""
        config = self.config

        data = read_data(config.code_execution_data)
        errors = 0
        total = 0
        timeout_error = 0.
        
        for k,instance in enumerate(data):
            code = instance["code"]
            question = instance.get("question","no question")

            ### temporary blocks
            if k == 5485 or k == 5641 or k == 6286 or k == 6734 or k == 6508: continue 
            print(f"question {k}: {question}")

            if k > 5416:
                code = f"#{k}\n{code}"

            s = time.time()
            e_time = None
            with timeout(5):
                try: 
                    code_response = self.code_agent(
                        query=code,
                        action_type="python"
                    )
                except Exception as e:
                    
                    print("timeout!!!!")
                    print(e)
                    print("\n\n")
                    timeout_error += 1
                    continue 

            e_time = time.time() - s
            
            # if k % 1000 == 0:
            #     ### clear hugginface cache
            #     self.code_agent(
            #         query="rm -rf  ../../root/.cache/huggingface",
            #         action_type="execute",
            #     )

            #print("\n")
            if "Traceback (most recent call last)" in code_response  or not code_response.strip():
                errors += 1
                #print(code_response)
                continue

            total += 1
            instance["code_response"] = code_response
            instance["time"] = e_time
            print(code_response)
            print("===================")
            #if k >= 800: break
            
        ###
        util_logger.info(f'Finished, {errors} errors found (or blank outputs, conservative estimate), total={total},timeout={timeout_error}')
        post_func = Registry.find_function(config.execute_postprocess_func)
        if post_func: 
            post_func(self.config,data)
        

@Registry(
    resource_type="config",
    name="python_code_config"
)
class PromptParams(ModuleParams):
    """Utilities for running pipelines 
    
    :param prompt_data: 
        Pointer to the prompt dataset containing lines 
        of json items. 

    """
    code_execution_data: str = ParamField(
        default='',
        metadata={
            "help" : 'The dataset to use for code testing',
            "exclude_hash" : True,
        }
    )
    execute_postprocess_func: str = ParamField(
        default='',
        metadata={
            "help" : "A function to use for pre-processing a piece of code",
            "exclude_hash" : True,
        }
    )
    execution_timeout: int = ParamField(
        default=3,
        metadata={
            "help" : "The execution timeout limit",
            "exclude_hash" : True,
        }
    )
