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
        try:
            return call_model_structured(model,message,response_format,logprobs=logprobs)
        except Exception as e:
            model.logging.warning(
                f'Issue encountered while running running, msg={e}, retrying',
                exc_info=True
            )
            
            time.sleep(2**(i+1))

    raise ModelRuntimeError(
        f'Error encountered when running model, msg={e}'
    )


def call_model_structured(model,message,response_format, logprobs=False) -> ModelOutput:
    """Calls the underlying model 
    
    :param message: 
        The formatted message to pass to move.  
    
    :param response_format:
        see https://openai.com/index/introducing-structured-outputs-in-the-api/ for more details.

    example for json_schema, it also supports pydantic models, see url above:
    pydantic BaseModel # Recommended!
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
## https://python.langchain.com/v0.1/docs/modules/model_io/chat/structured_output/#anthropic 
#################################################################################################################################
'''

import anthropic
from langchain_anthropic import ChatAnthropic


def claude_create_message(query,instruction=[],examples=[],history=[]):
    messages = []
    for content,role in history:
        role = 'user' if role!='assistant' else 'assistant'
        messages.append({"content": content, "role": role})
    messages.append({"content": query, "role": "user"})
    return messages


def claude__call__(
        model: OpenAIModel, # model in config is ignored
        prompt: str,
        response_format: Union[Dict[str,Any],BaseModel]=None, # not supported for claude
        instruction: Optional[str]="",
        examples: Optional[List[Any]] = [],
        history: Optional[List[Any]] = [],
        model_state: Optional[ModelState] = None,
        logprobs=False, # not supported for claude
        system: Optional[str] = None,
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
            claude_create_message(
                query=prompt,
                instruction=instruction,
                examples=examples,
                history=history,
            ),
            response_format,
            logprobs=logprobs,
            system=system,
            **kwargs
        )
    
    message = claude_create_message(
        query=prompt,
        instruction=instruction,
        examples=examples,
        history=history
    )
    return _prompt_model_claude(model,message,system,response_format,logprobs,**kwargs)



def _prompt_model_claude(model,message,system,response_format,logprobs=False,**kwargs) -> str:
    """Main method for calling the underlying LM. 
    
    :see: https://github.com/jiangjiechen/auction-arena/blob/main/src/bidder_base.py#L167
    :param prompt: 
        The input prompt object to the model. 
    
    """
    e='Unknown error'
    for i in range(model._config.num_calls):
        try:
            return call_model_claude(model,message,system,response_format,logprobs)
        except Exception as e:
            model.logging.warning(
                f'Issue encountered while running running, msg={e}, retrying',
                exc_info=True
            )
            
            time.sleep(2**(i+1))

    raise ModelRuntimeError(
        f'Error encountered when running model, msg={e}'
    )


def to_langchain_message(message,system):
    messages = [("system",system)]
    for msg in message:
        messages.append((msg['role'],msg['content']))
    return messages

def call_model_claude(model,message,system,response_format, logprobs=False) -> ModelOutput:
    """Calls the claude model 
    
    https://docs.anthropic.com/en/api/messages

    logprobs is not supported for claude
    response_format is not stably supported for claude
    """
    
    if response_format is not None and inspect.isclass(response_format) and issubclass(response_format,BaseModel):
        model = ChatAnthropic(
            model="claude-3-5-sonnet-20240620", 
            temperature=model._config.temperature,
            max_tokens=model._config.max_output_tokens,
        )
        structured_llm = model.with_structured_output(response_format)

        RET=structured_llm.invoke(to_langchain_message(message,system))
        return ModelOutput(
            text=str(RET.json()),
            cost=0,
            token_probs=[],
            input_tokens=0,
            output_tokens=0,
        )
    else:
        RET=anthropic.Anthropic().messages.create(
            model="claude-3-5-sonnet-20240620", # model in config is ignored
            max_tokens=model._config.max_output_token,
            messages=message, 
            temperature=model._config.temperature,
            system=system, # claude does not has system role, system prompt must be passed separately
        )
        if RET['type']=='error':
            raise Exception(RET['error'])
        else:
            return ModelOutput(
                text=RET['content'][0]['text'],
                cost=0,
                token_probs=[],
                input_tokens=RET['usage']['input_tokens'],
                output_tokens=RET['usage']['output_tokens']
            )






