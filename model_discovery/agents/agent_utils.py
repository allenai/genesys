import time
import os
import json
from typing import List,Any,Optional,Dict,Union
from pydantic import BaseModel
import inspect

from exec_utils.models.model import ModelState,OpenAIModel,ModelRuntimeError,ModelOutput
from exec_utils.models.utils import openai_costs


'''
#################################################################################################################################
## Patch for the OpenAIModel structured outputs: https://platform.openai.com/docs/guides/structured-outputs/introduction
#################################################################################################################################
'''


def structured__call__(
        model: OpenAIModel,
        prompt: str,
        response_format: Union[Dict[str,Any],BaseModel]=None,
        instruction: Optional[str]="",
        examples: Optional[List[Any]] = [],
        history: Optional[List[Any]] = [],
        model_state: Optional[ModelState] = None,
        logprobs=False,
        **kwargs
    ):
    """Makes a call the underlying model 

    :param prompt: 
        The prompt to the underlying model. 
    :param instruction: 
        The optional instruction to the model 
    :param examples: 
        The optional set of in-context examples 
    :param history: 
        The optional history items for model 
    :param model_state: 
        The optional model state at the point of querying 
    
    """
    if model_state is None:
        return _prompt_model_structured(
            model,
            model.create_message(
                query=prompt,
                instruction=instruction,
                examples=examples,
                history=history,
            ),
            response_format,
            logprobs=logprobs,
            **kwargs
        )
    
    message = model_state.create_message(
        query=prompt,
        manual_history=history
    )
    return _prompt_model_structured(model,message,response_format,logprobs=logprobs,**kwargs)



def _prompt_model_structured(model,message,response_format,logprobs=False,**kwargs) -> str:
    """Main method for calling the underlying LM. 
    
    :see: https://github.com/jiangjiechen/auction-arena/blob/main/src/bidder_base.py#L167
    :param prompt: 
        The input prompt object to the model. 
    
    """
    for i in range(model._config.num_calls):
        # try:
        return call_model_structured(model,message,response_format,logprobs=logprobs)
        # except Exception as e:
        #     model.logging.warning(
        #         f'Issue encountered while running running, msg={e}, retrying',
        #         exc_info=True
        #     )
            
        #     time.sleep(2**(i+1))

            # raise ModelRuntimeError(
            #     f'Error encountered when running model, msg={e}'
            # )


def call_model_structured(model,message,response_format, logprobs=False) -> ModelOutput:
    """Calls the underlying model 
    
    :param message: 
        The formatted message to pass to move.  
    
    :param response_format:
        see https://openai.com/index/introducing-structured-outputs-in-the-api/ for more details.

    example for json_schema, it also supports pydantic models, see url above:
    
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "math_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                        "type": "string"
                        },
                        "output": {
                        "type": "string"
                        }
                    },
                    "required": ["explanation", "output"],
                    "additionalProperties": False
                    }
                },
                "final_answer": {
                    "type": "string"
                }
                },
                "required": ["steps", "final_answer"],
                "additionalProperties": False
            }
        }
    } 
    or pydantic BaseModel
    """
    
    assert 'gpt-4o' in model._config.model_name
    
    if (response_format is not None and inspect.isclass(response_format)
        and issubclass(response_format,BaseModel)):
        model_fn=model.model_obj.beta.chat.completions.parse
    else:
        model_fn=model.model_obj.chat.completions.create
    completions = model_fn(
        model=model._config.model_name,
        messages=message,
        max_tokens=model._config.max_output_tokens,
        stop=model._stop,
        temperature=model._config.temperature,
        response_format=response_format,
        logprobs=logprobs
    )
    msg=completions.choices[0].message
    if (response_format is not None and inspect.isclass(response_format)
        and issubclass(response_format,BaseModel)):
        if msg.refusal:
            raise ModelRuntimeError(msg.refusal)
        else:
            output = str(msg.parsed.json())
    else:
        output = msg.content

    cost = openai_costs(
        model._config.model_name,
        completions.usage
    )
    model._model_cost += cost
    token_probs = completions.choices[0].token_probs.content if logprobs else []
    return ModelOutput(
        text=output,
        cost=cost,
        token_probs=token_probs,
        input_tokens=completions.usage.prompt_tokens,
        output_tokens=completions.usage.completion_tokens
    )


'''
#################################################################################################################################
## Patch for Claude Agent and possibly structured outputs with LangChain
## https://api.python.langchain.com/en/latest/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html
#################################################################################################################################
'''











