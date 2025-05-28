from __future__ import annotations 

import random
import os
import json
import logging

from copy import deepcopy
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
from ..factory import BuildAgent
from ..param import (
    ModuleParams,
    RootParams,
    ParamField,
)

__all__ = [
    "run_prompt"
]

util_logger = logging.getLogger(
    'exec_utils.pipelines.prompt_pipeline'
)

_PromptInstance = Dict[str,str]

def run_prompt(
        llm_agent: Type[SimpleLMAgent],
        config: ConfigType,
        dataset: List[_PromptInstance],
    ) -> None:
    """Run the prompt loop

    :param llm_agent: 
        The LLM agent to prompt.
    :param config: 
        The global configuration. 
    :param dataset: 
        The target dataset with instances for 
        prompting. 

    """
    prompt   = config.prompt
    max_data = config.max_prompt_data

    costs = 0.0
    model_output = []
    token_counts = {"input_tokens" : 0.0, "output_tokens" : 0.0}
    
    for k,instance in enumerate(tqdm(dataset[:max_data])):
        
        query_template = prompt or instance.get("prompt",instance.get("query"))
        to_print = deepcopy(instance)

        query = query_template.format(
            **instance 
        )
        
        try: 
            model_out = llm_agent(query)

            #print(model_out)
            #print(model_out["comment"])
            #print("=======================\n\n")

            # try:
            #     print("QUESTION=============")
            #     print(instance["question"])
            #     print("NEW QUESTION==========")

            #     if 'accepted' in model_out: 
            #         print(f"accepted: {model_out['accepted']}")
            #         print(model_out["comment"])
            #     else:
            #         print("dictionary")
            #         print(f"accepted: {model_out['feedback']['accepted']}")
            #         print(json.dumps(model_out,indent=4))
            #     # print("CHANGES===========")
            #     # print(instance["past_questions"])
            #     # print("\n\n\n")
            #     # print(model_out["changes"])
            #     # print("==============")
            #     # print("OLD CODE===============")
            #     # print(instance["code"])
            #     # print("NEW CODE===============")
            #     # print(model_out["code"])
            #     print("\n\n")
            # except:
            #     pass
            #     #print(model_out['feedback'])
            #     #continue 
                
            costs += model_out["_details"]["cost"]
            token_counts["input_tokens"] += model_out["_details"]["input_tokens"]
            token_counts["output_tokens"] += model_out["_details"]["output_tokens"]

            to_print["prompt_out"] = model_out
            model_output.append(to_print)

        except KeyboardInterrupt:
            util_logger.info('Interrupted, ending...')
            break 
        except Exception as e:
            util_logger.warning("error encountered",exc_info=True)
            break 
        
    
    return (model_output,costs,token_counts) 
                
def read_dataset(dataset_path: PathOrStr,prompt: str) -> List[
        _PromptInstance
    ]:
    """Reads the prompting dataset. 

    :param dataset_path: 
        The path of the target dataset. 
    :raises: 
        ValueError 

    """
    prompt_items = []
    if not os.path.isfile(dataset_path):
        raise ValueError(
            f"Unknown dataset path: {dataset_path}"
        )
    
    with open(dataset_path) as prompt_dataset:

        for line in tqdm(prompt_dataset):
            json_line = json.loads(line.strip())
            prompt_items.append(json_line)

            if not prompt and not json_line.get("prompt",json_line.get("query")):
                raise ValueError(
                    f"No query specified, must specify --prompt or put in data"
                )


    util_logger.info(
        f"Finished reading dataset with {len(prompt_items)} instances"
    )
    
    return prompt_items

@Registry(
    resource_type="pipeline_type",
    name="prompting_pipeline",
    cache=None,
)
class PromptPipeline(ConfigurablePipelineUtil):
    """Runs a prompting pipeline

    attributes 
    --------------
    :param llm_agent: 
        The underlying LLM being used for prompting 
    :param config: 
        The global configuration, with details about where
        the prompting data is, where to put output, etc. 

    methods 
    --------------
    :method run_pipeline: 
        The main core of the pipeline where the prompting 
        pipeline gets implemented. 

    """

    def __init__(self,llm_agent,config: ConfigType) -> None:
        self.llm_agent = llm_agent
        self.config = config
        
    def run_pipeline(self,*args,**kwargs) -> None:
        """Runs the prompting pipeline."""
        config = self.config 
        dataset = read_dataset(config.prompt_data,config.prompt)

        ### shuffle dataset using seed 
        random.Random(config.seed).shuffle(dataset)

        prompt_func = Registry.find_function(config.prompt_function) or run_prompt
        
        model_output,costs,token_count = prompt_func(
            self.llm_agent,
            self.config,
            dataset
        )
        util_logger.info(f'Final costs: {costs}')
        util_logger.info(f'Token counts: {token_count}')

        print_func = Registry.find_function(config.printing_function)
        if print_func is not None:
            print_func(
                config=config,
                model_output=model_output
            )

    @classmethod
    def from_config(cls,config: ConfigType):
        """Builds the prompt pipeline from configuration 

        :param config: 
            The global configuration for prompting. 
        """
        llm_agent = BuildAgent(config)
        return cls(llm_agent,config) 
    
@Registry(
    resource_type="config",
    name="prompt_config"
)
class PromptParams(ModuleParams):
    """Utilities for running pipelines 
    
    :param prompt_data: 
        Pointer to the prompt dataset containing lines 
        of json items. 

    """
    prompt: str = ParamField(
        default='',
        metadata={"help" : 'The target prompt'}
    )
    prompt_data: str = ParamField(
        default='',
        metadata={
            "help" : 'The dataset to use for prompting',
            "exclude_hash" : True,
        }
    )
    output_data: str = ParamField(
        default='',
        metadata={
            "help" : 'The output dataset to print results to',
            "exclude_hash" : True,
        }
    )
    shuffle_prompt_data: bool = ParamField(
        default=False,
        metadata={
            "help" : 'Shuffle the data',
            "exclude_hash" : True,
        }
    )
    max_prompt_data: int = ParamField(
        default=10000000,
        metadata={
            "help"         : 'The maximum allowed prompt data',
            "exclude_hash" : True,
        }
    )
    printing_function: str = ParamField(
        default='',
        metadata={
            "help"         : 'The printing function to use to process data in the end',
            "exclude_hash" : True,
        }
    )
    prompt_function: str = ParamField(
        default='',
        metadata={
            "help" : 'Prompt function',
            "exclude_hash" : True,
        }
    )
