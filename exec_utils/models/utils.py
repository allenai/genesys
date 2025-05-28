import os
import ast
import openai
import json
import logging
import together 

from ..aliases import (
    PathOrStr,
    ConfigType
)
from typing import (
    Tuple,
    Callable,
    Dict,
    Any,
    Union,
    List
)
from langchain_core.language_models import BaseChatModel
# from langchain_google_vertexai import (
#     VertexAI,
#     ChatVertexAI
# )
from langchain_openai import ChatOpenAI

__all__ = [
    "setup_langchain",
    "parse_dict_output",
    "setup_openai",
    "openai_costs",
    "check_model_type",
]

util_logger = logging.getLogger('exec_utils.models.utils')
    
def setup_langchain(config: ConfigType) -> Tuple[
        BaseChatModel,
        Callable[int,Dict[str,int]]
    ]:
    """Sets up a langchain model model and returns the type of function needed 
    for calling it. 

    :raises: 
        ValueError 
    :returns: 
        The langchain model and the function for computing max output tokens
    """

    model_name = config.model_name
    
    if "gpt-" in model_name:

        key = os.environ.get("MY_OPENAI_KEY",'') if not config.openai_api_key \
          else config.openai_api_key

        model = ChatOpenAI(
            model=model_name,
            temperature=config.temperature,
            max_retries=config.max_retries,
            api_key=key
        )

        # if "gpt-4-0125-preview" in model_name:
        #     max_val = 100000
        if "gpt-3.5-turbo" in model_name and "16k" not in model_name:
            max_val = 3900
        elif "32k" in model_name:
            max_val = 32000
        elif "16k" in model_name:
            max_val = 16000
        elif "0125-preview" in model_name:
            max_val = 4000
        else: 
            max_val = 8000
            
        f = lambda n: {"max_tokens" : max(max_val - n,192)}
        
    elif "gemini-pro" in model_name:
        raise NotImplemented('gemini not implemented in this release')
        
        # # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm
        # model = ChatVertexAI(
        #     model_name=model_name,
        #     max_retries=config.max_retries,
        #     temperature=config.temperature,
        #     convert_system_message_to_human=True
        # )

        # f = lambda n : {"max_output_tokens" : n}

    ### add more model interfaces here 
    else:
        raise ValueError(
            f"Uknown langchain model name: {model_name}"
        )
    
    return (model,f) 

def parse_dict_output(output: str,details: bool =True) -> Dict[str,str]:
    """Parses raw model output and tries to put it into a dictionary format. 
    Attempts first to parse it as json, then as a native python dictionary, 
    then gives up and passed the raw string in a dictionary, with details 
    about the final format. 
    
    :param output: 
        The raw output of a model. 
    
    """
    parsed = "raw_string"
    
    try:
        ### try json 
        try:
            feedback = json.loads(output)
            parsed = "json"
            
            if not isinstance(feedback,dict):
                raise json.JSONDecodeError
                        
        except json.JSONDecodeError:

            ## try raw dict
            try:
                feedback = ast.literal_eval(output)
                #if "feedback" not in feedback:
                #    feedback = {"feedback" : feedback}
                parsed = "dict"
                    
            except SyntaxError:
                raise 
                
    except:
        feedback = {"feedback": output}
        parsed = "raw_string"

    if details:
        feedback["_details"] = {}
        feedback["_details"]["parsed"] = parsed
        feedback["_details"]["raw_output"] = output
        
    return feedback


def setup_openai(config: ConfigType) -> Tuple[openai.OpenAI,Union[List[str],None]]:
    """Returns back an openai client after setting up API key 

    :param config: 
        The global configuration containing details about setup. 

    """
    stop = None 
    ### check if it involves together ai models
    if "gpt" in config.model_name: 
        key = os.environ.get("MY_OPENAI_KEY",'') if not config.openai_api_key \
          else config.openai_api_key

        client = openai.OpenAI(api_key=key)

    else:
        key = os.environ.get("TOGETHER_API_KEY",'') if not config.together_api_key \
          else config.together_api_key
        
        client = openai.OpenAI(
            api_key=key,
            base_url="https://api.together.xyz/v1",
        )

        available_stops = {
            i["name"] : i["config"]["stop"] for i in together.Models().list() \
            if i and "config" in i and i["config"] and "stop" in i["config"]
        }
        #available_stops = {}
        if config.model_name in available_stops:
            stop = available_stops[config.model_name]
        else:
            util_logger.warning(
                f'No stops found for this model: {config.model_name}'
            )
    return (client,stop) 

_OPENAI_USAGE = {
    "gpt-4o-2024-05-13"      : (0.005/1000,0.015/1000),
    "gpt-4o"                 : (0.005/1000,0.015/1000),
    "gpt-4-turbo"            : (0.01/1000,0.03/1000),
    "gpt-4-turbo-2024-04-09" : (0.01/1000,0.03/1000),
    "gpt-4-0125-preview"     : (0.01/1000,0.03/1000),
    "gpt-4-1106-preview"     : (0.01/1000,0.03/1000),
    "gpt-3.5-turbo-0125"     : (0.0005/1000,0.0015/1000),
    "gpt-3.5-turbo-instruct" : (0.0015/1000,0.0020/1000),
    "gpt-4"                  : (0.03/1000,0.06/1000),
    "gpt-4-32k"              : (0.06/1000,0.12/1000),
    "gpt-4-32k-0314"         : (0.06/1000,0.12/1000),
    "gpt-3.5-turbo-1106"     : (0.0010/1000,0.0020/1000),
    "gpt-3.5-turbo-0613"     : (0.0015/1000,0.0020/1000),
    "gpt-3.5-turbo-16k-0613" : (0.0030/1000,0.0040/1000),
    "gpt-3.5-turbo-16k"      : (0.0030/1000,0.0040/1000),
    "gpt-3.5-turbo-0301"     : (0.0015/1000,0.0020/1000),
    "gpt-3.5-turbo"          : (0.0030/1000,0.0060/1000),
    "gpt-3.5-turbo-instruct" : (0.0015/1000,0.0020/1000),
}
    
def openai_costs(model_name: str,usage) -> float:
    """Computes the cost of an openai model run 

    :param model_name: 
        The name of the model being used. 
    :param usage: 
        Information about usage from the openai API. 
    """
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    if model_name not in _OPENAI_USAGE:
        if "gpt" in model_name: 
            util_logger.warning(
                f"Unknown model, cannot compute usage: {model_name}"
            )
        return 0.0 

    input_costs,output_costs = _OPENAI_USAGE[model_name]
    return prompt_tokens*input_costs + output_costs*completion_tokens
    
def check_model_type(agent_config: ConfigType) -> None:
    """Manual check on the types of models and APIs being used. 

    :param agent_config: 
        The configuration for the agent 
    """
    if agent_config.agent_type != "agent_model_type":
        return
    mname = agent_config.model_name
    mtype = agent_config.model_type
    
    if "gpt" in mname and "-preview" in mname and mtype != "openai":
        util_logger.warning(
            f"Switching model=`{mtype}` type to `openai` for model={mname}"
        )
        agent_config.model_type = "openai"
