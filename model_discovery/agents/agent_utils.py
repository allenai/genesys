import time
import os
import json

from typing import List,Any,Optional,Dict,Union
from pydantic import BaseModel
import inspect
import tiktoken 
import copy

from exec_utils.models.model import ModelState,OpenAIModel,ModelRuntimeError,UtilityModel
from exec_utils.models.utils import openai_costs


## YOU SHOULD ALWAYS THE CALLS HERE INSTEAD OF THE ORIGINAL CALLS IN EXEC_UTILS


OPENAI_COSTS_DICT={
    "gpt-4o-2024-08-06":{
        'input':2.5/1e6,
        'output':10/1e6,
    },
    "gpt-4o-mini":{
        'input':0.15/1e6,
        'output':0.6/1e6,
    },
    "o1-preview":{
        'input':15/1e6,
        'output':60/1e6,
    },
    "o1-mini":{
        'input':3/1e6,
        'output':12/1e6,
    },
}

def openai_costs(usage,model_name):
    costs=OPENAI_COSTS_DICT
    usage['cost']=usage['input_tokens']*costs[model_name]['input'] + usage['output_tokens']*costs[model_name]['output']
    usage['model_name']=model_name
    return usage

ANTHROPIC_COSTS_DICT={
    "claude-3-5-sonnet-20240620":{
        'input':3/1e6,
        'output':15/1e6,
        'cache_creation':3.75/1e6,
        'cache_read':0.3/1e6,
    }
}


def anthropic_costs(usage,model_name='claude-3-5-sonnet-20240620'):
    costs=ANTHROPIC_COSTS_DICT
    cost=usage['input_tokens']*costs[model_name]['input'] + usage['output_tokens']*costs[model_name]['output']
    if 'cache_creation_input_tokens' in usage:
        cost+=usage['cache_creation_input_tokens']*costs[model_name]['cache_creation']
    if 'cache_read_input_tokens' in usage:
        cost+=usage['cache_read_input_tokens']*costs[model_name]['cache_read']
    usage['cost']=cost
    usage['model_name']=model_name
    return usage

OPENAI_TOKEN_LIMITS={
    "gpt-4o-2024-08-06":128000,
    "gpt-4o-mini":128000,
    "o1-preview":128000,
    "o1-mini":128000,
}

OPENAI_OUTPUT_BUFFER={
    "gpt-4o-2024-08-06":16384,
    "gpt-4o-mini":16384,
    "o1-preview":32768,
    "o1-mini":32768,
}


ANTHROPIC_TOKEN_LIMITS={
    "claude-3-5-sonnet-20240620":200000,
}

ANTHROPIC_OUTPUT_BUFFER={
    "claude-3-5-sonnet-20240620":8192,
}



def get_token_limit(model_name):
    if 'gpt' in model_name or 'o1' in model_name:
        return OPENAI_TOKEN_LIMITS[model_name] - OPENAI_OUTPUT_BUFFER[model_name]
    elif 'claude' in model_name:
        return ANTHROPIC_TOKEN_LIMITS[model_name] - ANTHROPIC_OUTPUT_BUFFER[model_name]
    else:
        raise ValueError(f'Unsupported model: {model_name}')

def _encode_text(text,model_name,truncate=None):
    if 'gpt' in model_name or 'o1' in model_name:
        enc=tiktoken.encoding_for_model(model_name)
        tokens = enc.encode(text)
    elif 'claude' in model_name:
        client = anthropic.Client()
        tokenizer =  client.get_tokenizer()
        tokens = tokenizer.encode(text).ids
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    if truncate is not None:
        tokens = tokens[:truncate]
    return tokens

def decode_text(tokens,model_name):
    if 'gpt' in model_name or 'o1' in model_name:
        enc=tiktoken.encoding_for_model(model_name)
        return enc.decode(tokens)
    elif 'claude' in model_name:
        client = anthropic.Client()
        tokenizer =  client.get_tokenizer()
        return tokenizer.decode(tokens)
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    
def count_tokens(text,model_name):
    return len(_encode_text(text,model_name))

def truncate_text(text,token_limit,model_name,buffer=128):
    tokens=_encode_text(text,model_name,truncate=token_limit-buffer)
    text=decode_text(tokens,model_name)+'\n\n... (truncated)'
    return text

def truncate_history(history,token_limit,model_name,buffer=128):
    truncated_history=[]
    for content,role in history[::-1]: # add latest messages first
        num_tokens=count_tokens(content,model_name)
        if num_tokens > token_limit:
            content = truncate_text(content,token_limit,model_name,buffer)
            truncated_history.append((content,role))
            break
        truncated_history.append((content,role))
        token_limit-=num_tokens
    truncated_history=truncated_history[::-1] # revert to original order
    return truncated_history

def context_safe_guard(history,model_name,prompt=None,system=None,buffer=128):
    history = copy.deepcopy(history)
    _token_limit=get_token_limit(model_name)
    token_limit=_token_limit
    if system is not None:
        token_limit-=count_tokens(system,model_name)
    if token_limit<0:
        try_truncate = _token_limit-count_tokens(system,model_name)
        raise ValueError(f'Token limit exceeded by system prompt: {token_limit}')
    if prompt is not None:
        token_limit-=count_tokens(prompt,model_name)
    if token_limit<0:
        try_truncate = _token_limit-count_tokens(system,model_name)
        prompt = truncate_text(prompt,try_truncate,model_name,buffer)
        history = []
    else:
        history=truncate_history(history,token_limit,model_name,buffer)
    return history,prompt


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
    history,prompt=context_safe_guard(history,model._config.model_name,prompt,system)
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


def o1_beta_message_patch(message):
    msgs=[]
    for msg in message:
        msgs.append({
            'role':msg['role'] if msg['role']!='system' else 'user',
            'content':msg['content']
        })
    return msgs

def call_model_structured(model,message,response_format, logprobs=False) -> ModelOutputPlus:
    """Calls the underlying model 
    
    :param message: 
        The formatted message to pass to move.  
    
    :param response_format:
        see https://openai.com/index/introducing-structured-outputs-in-the-api/ for more details.

    example for json_schema, it also supports pydantic models, see url above:
    pydantic BaseModel # Recommended!
    """
    
    # assert 'gpt-4o' in model._config.model_name
    
    if (response_format is not None and inspect.isclass(response_format)
        and issubclass(response_format,BaseModel)):
        model_fn=model.model_obj.beta.chat.completions.parse
    else:
        model_fn=model.model_obj.chat.completions.create
    
    # XXX: patches for o1 beta stage
    fn_kwargs={}
    if model._config.model_name in ['o1-mini','o1-preview']: 
        message=o1_beta_message_patch(message)
        logprobs=False 
        fn_kwargs['max_completion_tokens']=model._config.max_output_tokens*4-1 # leave spaces for raesoning tokens
        model_fn=model.model_obj.chat.completions.create # parse is not supported for o1 
        response_format=None
    else:
        fn_kwargs['max_completion_tokens']=model._config.max_output_tokens
        fn_kwargs['temperature']=model._config.temperature


    completions = model_fn(
        model=model._config.model_name,
        messages=message,
        # stop=model._stop,
        response_format=response_format,
        logprobs=logprobs,
        **fn_kwargs
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

    usage={
        'input_tokens':completions.usage.prompt_tokens,
        'output_tokens':completions.usage.completion_tokens,    
    }
    if 'o1' in model._config.model_name:
        usage['reasoning_tokens']=completions.usage.completion_tokens_details
    usage=openai_costs(usage,model._config.model_name)
    cost=usage['cost']
    model._model_cost += cost
    token_probs = completions.choices[0].token_probs.content if logprobs else []
    return ModelOutputPlus(
        text=output,
        cost=cost,
        token_probs=token_probs,
        input_tokens=completions.usage.prompt_tokens,
        output_tokens=completions.usage.completion_tokens,
        usage=usage
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
        if history:
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
        use_cache: bool = False, # XXX: not working with self-managed contexts as its changing, hard to track
        system: Optional[str] = None,
        model_name: Optional[str] = 'claude-3-5-sonnet-20240620',
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
    history=context_safe_guard(history,model._config.model_name,prompt,system)
    messages=ConversationHistory(history,prompt).get_turns(use_cache)
    model._config.model_name=model_name
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
    No cache: 2K+4K+6K+8K+10K = 30K * 1 = SR+I(R-1)R/2 = ( SR + 0.5IR^2 - 0.5IR ) * C 
    Cache: 2K*1.25 + (2*0.1)+2*1.25 + (4*0.1)+2*1.25 + (6*0.1)+2*1.25 + (8*0.1)+2*1.25 
         = Write: 1.25(S+(R-1)I) + Read: 0.1(S(R-1)+I(R-1)(R-2)/2)  # R>=2
         = ( 1.15S + 1.1RI - 1.15I + 0.1SR + 0.05IR^2 ) * C
    R: num of rounds
    I: incremental input tokens per round
    S: system tokens
    C: input cost per token

    R=5, I=4K, S=3K
    No cache: SR+I(R-1)R/2 = 3K*5+(5-1)*5/2*4K = 15+40 = 55K
    Cache: 1.25(S+(R-1)I) + 0.1(S(R-1)+I(R-1)(R-2)/2) = 19*1.25 + 0.1*(12+24)=3.6+23.75=27.35K
    """

    if use_cache: # XXX: not working with structured outputs now!
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    else:
        extra_headers={}
    
    if response_format is not None and inspect.isclass(response_format) and issubclass(response_format,BaseModel):
        lc_model = ChatAnthropic(model=model._config.model_name)
        structured_llm = lc_model.with_structured_output(response_format,include_raw=True)
        tools_args=structured_llm.dict()['first']['steps__']['raw']['kwargs']
        # tools[0]['cache_control']={"type": "ephemeral"}
    else:
        tools_args={}
        
    RET=anthropic.Anthropic().messages.create(
        model=model._config.model_name, # model in config is ignored
        max_tokens=model._config.max_output_tokens,
        messages=message, 
        temperature=model._config.temperature,
        system=system, # claude does not has system role, system prompt must be passed separately
        extra_headers=extra_headers,
        **tools_args
    )
    RET=RET.dict()
    if RET['type']=='error':
        raise Exception(RET['error'])

    if tools_args!={}:
        try:
            assert RET['content'][0]['type']=='tool_use'
            parsed=response_format.model_validate(RET['content'][0]['input']) 
        except Exception as e:
            raise e
        text=parsed.model_dump_json()
    else:
        text=RET['content'][0]['text']

    usage=anthropic_costs(RET['usage'],model._config.model_name)

    return ModelOutputPlus(
        text=text,
        cost=usage['cost'],
        token_probs=[],
        input_tokens=usage['input_tokens'],
        output_tokens=usage['output_tokens'],
        usage=usage,
    )






