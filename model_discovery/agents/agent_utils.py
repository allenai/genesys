import time
import os
import json
from typing import List,Any,Optional,Dict,Union
from pydantic import BaseModel
import inspect

from exec_utils.models.model import ModelState,OpenAIModel,ModelRuntimeError,UtilityModel
from exec_utils.models.utils import openai_costs




class ModelOutputPlus(UtilityModel):
    """Helper class for showing model output 

    :param text: 
        The text produced by the model 
    :param cost: 
        The cost of running inference on the input 
        producing the text 
    :param log_probs: 
        Details about model logprobs 

    """
    text: str
    token_probs: List = []
    cost: float = 0.0
    input_tokens: int = 0.
    output_tokens: int = 0.
    usage: Dict = {}
    
    def __repr__(self):
        return self.text


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


def call_model_structured(model,message,response_format, logprobs=False) -> ModelOutputPlus:
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
    return ModelOutputPlus(
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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class ConversationHistory:
    def __init__(self,history,query):
        # Initialize an empty list to store conversation turns
        self.turns = []
        for content,role in history:
            if isinstance(content,str):
                if role=='assistant':
                    self.add_turn_assistant(content)
                else:
                    self.add_turn_user(content)
            else:
                self.add_turn_raw(content,'assistant') # raw content from agent
        self.add_turn_user(query)

    def add_turn_raw(self,content,role):
        self.turns.append({
            "role": role,
            "content": content
        })

    def add_turn_assistant(self, content):
        # Add an assistant's turn to the conversation history
        self.turns.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })

    def add_turn_user(self, content):
        # Add a user's turn to the conversation history
        self.turns.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })

    def get_turns(self,use_cache=True):
        # Retrieve conversation turns with specific formatting
        result = []
        user_turns_processed = 0
        # Iterate through turns in reverse order
        for turn in reversed(self.turns):
            if turn["role"] == "user" and user_turns_processed < 2 and use_cache:
                # Add the last two user turns with ephemeral cache control
                turn['content'][0]['cache_control'] = {"type": "ephemeral"}
                user_turns_processed += 1
            result.append(turn)
        # Return the turns in the original order
        return list(reversed(result))


def claude__call__(
        model: OpenAIModel, # model in config is ignored
        prompt: str,
        response_format: Union[Dict[str,Any],BaseModel]=None, # not supported for claude
        instruction: Optional[str]="",
        examples: Optional[List[Any]] = [],
        history: Optional[List[Any]] = [],
        model_state: Optional[ModelState] = None,
        logprobs=False, # not supported for claude
        use_cache: bool = True, # XXX: not working with structured outputs now!
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
    messages=ConversationHistory(history,prompt).get_turns(use_cache)
    if use_cache:
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system=[{"type": "text", "text": system}]
    if model_state is None:
        return _prompt_model_structured(
            model,
            messages,
            response_format,
            logprobs=logprobs,
            system=system,
            **kwargs
        )
    
    return _prompt_model_claude(model,messages,system,response_format,logprobs,use_cache,**kwargs)



def _prompt_model_claude(model,message,system,response_format,logprobs=False,use_cache=True,**kwargs) -> str:
    """Main method for calling the underlying LM. 
    
    :see: https://github.com/jiangjiechen/auction-arena/blob/main/src/bidder_base.py#L167
    :param prompt: 
        The input prompt object to the model. 
    
    """
    ERROR=[]
    for i in range(model._config.num_calls):
        try:
            return call_model_claude(model,message,system,response_format,logprobs,use_cache)
        except Exception as e:
            model.logging.warning(
                f'Issue encountered while running running, msg={e}, retrying',
                exc_info=True
            )
            
            time.sleep(2**(i+1))
            ERROR.append(f'Attempt {i+1} error: {e}')

    if len(ERROR)>0:
        ERROR='\n'.join(ERROR)
    else:
        ERROR='Unknown error'
    raise ModelRuntimeError(
        f'Error encountered when running model, msg={ERROR}'
    )


# def to_langchain_message(message,system):
#     messages = [SystemMessage(system)]
#     for msg in message:
#         msg_type=HumanMessage if msg['role']=='user' else AIMessage
#         messages.append(msg_type(msg['content']))
#     return messages

def call_model_claude(model,message,system,response_format, logprobs=False,use_cache=True) -> ModelOutputPlus:
    """Calls the claude model 
    
    https://docs.anthropic.com/en/api/messages

    logprobs is not supported for claude
    response_format is not stably supported for claude, but not a big deal, just use tool using feature to implement

    Prompt caching is powerful:
    Cookbook: https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb
    Use with LangChain: https://github.com/langchain-ai/langchain/pull/25644

    Suppose 5 rounds, each round generates 1K tokens, 1K input sys/round (2K INCRE/round), 2K system, input costs:
    No cache: 2K+4K+6K+8K+10K = 30K * 1 = SR+I(R-1)R/2
    Cache: 2K*1.25 + (2*0.1)+2*1.25 + (4*0.1)+2*1.25 + (6*0.1)+2*1.25 + (8*0.1)+2*1.25 
         = Write: 1.25(S+(R-1)I) + Read: 0.1(S(R-1)+I(R-1)(R-2)/2)  # R>=2
    R: num of rounds
    I: incremental input tokens per round
    S: system tokens
    
    R=5, I=4K, S=3K
    No cache: SR+I(R-1)R/2 = 3K*5+(5-1)*5/2*4K = 15+40 = 55K
    Cache: 1.25(S+(R-1)I) + 0.1(S(R-1)+I(R-1)(R-2)/2) = 19*1.25 + 0.1*(12+24)=3.6+23.75=27.35K

    """
    if use_cache: # XXX: not working with structured outputs now!
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    else:
        extra_headers={}
    
    if response_format is not None and inspect.isclass(response_format) and issubclass(response_format,BaseModel):
        lc_model = ChatAnthropic(
            model="claude-3-5-sonnet-20240620", 
            temperature=model._config.temperature,
            max_tokens=model._config.max_output_tokens,
            extra_headers=extra_headers
        )
        structured_llm = lc_model.with_structured_output(response_format,include_raw=True)
        tools_args=structured_llm.dict()['first']['steps__']['raw']['kwargs']
        # tools[0]['cache_control']={"type": "ephemeral"}
        
        RET=anthropic.Anthropic().messages.create(
            model="claude-3-5-sonnet-20240620", # model in config is ignored
            max_tokens=model._config.max_output_tokens,
            messages=message, 
            temperature=model._config.temperature,
            system=system, # claude does not has system role, system prompt must be passed separately
            extra_headers=extra_headers,
            **tools_args
        )

        try:
            assert RET.content[0].type=='tool_use'
            parsed=response_format.model_validate(RET.content[0].input) 
        except Exception as e:
            raise e

        # message=to_langchain_message(message,system)
        # RET=structured_llm.invoke(message)
        
        RET=RET.dict()
        if RET['type']=='error':
            raise Exception(RET['error'])
        usage=RET['usage']
        cost=usage['input_tokens']*3/1e6 + usage['output_tokens']*15/1e6
        cost+=usage['cache_creation_input_tokens']*3.75/1e6
        cost+=usage['cache_read_input_tokens']*0.3/1e6
        usage['cost']=cost
        return ModelOutputPlus(
            text=str(parsed.json()),
            cost=cost,
            token_probs=[],
            input_tokens=usage['input_tokens'],
            output_tokens=usage['output_tokens'],
            usage=usage,
        )
    else:
        RET=anthropic.Anthropic().messages.create(
            model="claude-3-5-sonnet-20240620", # model in config is ignored
            max_tokens=model._config.max_output_tokens,
            messages=message, 
            temperature=model._config.temperature,
            system=system, # claude does not has system role, system prompt must be passed separately
            extra_headers=extra_headers
        )
        if RET['type']=='error':
            raise Exception(RET['error'])
        else:
            cost=RET['usage']['input_tokens']*3/1e6 + RET['usage']['output_tokens']*15/1e6
            RET['usage']['cost']=cost
            return ModelOutputPlus(
                text=RET['content'][0]['text'],
                cost=cost,
                token_probs=[],
                input_tokens=RET['usage']['input_tokens'],
                output_tokens=RET['usage']['output_tokens'],
                usage=RET['usage']
            )






