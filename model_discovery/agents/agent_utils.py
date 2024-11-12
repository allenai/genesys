import time
import os
import json
import re

from typing import List,Any,Optional,Dict,Union
from pydantic import BaseModel
import inspect
import tiktoken 
import copy

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


from exec_utils.models.model import ModelState,OpenAIModel,ModelRuntimeError,UtilityModel
from exec_utils.models.utils import openai_costs


## YOU SHOULD ALWAYS THE CALLS HERE INSTEAD OF THE ORIGINAL CALLS IN EXEC_UTILS





def find_level1_blocks(text):
    # Regular expressions for opening and closing patterns
    opening_pattern = r'```[^\s]+'  # Matches any pattern like ```xxx followed by non-whitespace characters
    closing_pattern = r'```(?=\s|$)'  # Matches standalone closing patterns, followed by space, newline, or end of string

    # Finding all opening and closing positions
    open_positions = [(m.start(), m.group()) for m in re.finditer(opening_pattern, text)]
    close_positions = [m.start() for m in re.finditer(closing_pattern, text)]

    matches = []
    open_stack = []
    nesting_level = 0

    i, j = 0, 0
    last_match_end = -1

    while i < len(open_positions) or j < len(close_positions):
        if i < len(open_positions) and (j >= len(close_positions) or open_positions[i][0] < close_positions[j]):
            # Handle an opening pattern
            open_stack.append(open_positions[i])
            nesting_level += 1
            i += 1
        else:
            # Handle a closing pattern
            if open_stack:
                start_pos, start_tag = open_stack.pop()
                nesting_level -= 1
                # If we're back to level 0, it's a level 1 match
                if nesting_level == 0:
                    match_start = start_pos
                    match_end = close_positions[j] + len('```')
                    if match_start > last_match_end:
                        matches.append((match_start, match_end))
                        last_match_end = match_end
            j += 1

    # Extract the substrings corresponding to the level 1 matches
    result = [text[start:end] for start, end in matches]
    return result


def block_finder(raw_text:str,block_tag:str):
   blocks = find_level1_blocks(raw_text)
   matches = [block[len(f'```{block_tag}'):-3].strip() for block in blocks if block.startswith(f'```{block_tag}')]
   return matches


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
    "claude-3-5-sonnet-20241022":{
        'input':3/1e6,
        'output':15/1e6,
        'cache_creation':3.75/1e6,
        'cache_read':0.3/1e6,
    }
}


def anthropic_costs(usage,model_name='claude-3-5-sonnet-20241022'):
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
    "claude-3-5-sonnet-20241022":199999,
}

ANTHROPIC_OUTPUT_BUFFER={
    "claude-3-5-sonnet-20241022":8192,
}

SAFE_BUFFER=12000 # there might be api-side overhead

try:
    ANTHROPIC_CLIENT = anthropic.Client()
    ANTHROPIC_TOKENIZER =  ANTHROPIC_CLIENT.get_tokenizer()
except:
    ANTHROPIC_TOKENIZER = None


def get_token_limit(model_name):
    if model_name in OPENAI_TOKEN_LIMITS:
        return OPENAI_TOKEN_LIMITS[model_name] - OPENAI_OUTPUT_BUFFER[model_name] - SAFE_BUFFER
    elif model_name in ANTHROPIC_TOKEN_LIMITS:
        return ANTHROPIC_TOKEN_LIMITS[model_name] - ANTHROPIC_OUTPUT_BUFFER[model_name] - SAFE_BUFFER
    else:
        raise ValueError(f'Unsupported model: {model_name}')

def _encode_text(text,model_name,truncate=None):
    if 'gpt' in model_name or 'o1' in model_name:
        enc=tiktoken.encoding_for_model(model_name)
        tokens = enc.encode(text)
    elif 'claude' in model_name:
        if ANTHROPIC_TOKENIZER is None:
            enc = tiktoken.get_encoding("o200k_base")
            tokens = enc.encode(text)
        else:
            tokens = ANTHROPIC_TOKENIZER.encode(text).ids
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
        if ANTHROPIC_TOKENIZER is None:
            enc = tiktoken.get_encoding("o200k_base")
            return enc.decode(tokens)
        else:
            return ANTHROPIC_TOKENIZER.decode(tokens)
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    
def count_tokens(text,model_name):
    if text is None or text=='':
        return 0
    return len(_encode_text(text,model_name))

def count_msg(msg,model_name):
    if 'claude' in model_name:
        return count_tokens(msg['content'][0]['text'],model_name)
    else:
        return count_tokens(msg['content'],model_name)

def truncate_text(text,token_limit,model_name):
    tokens=_encode_text(text,model_name,truncate=token_limit)
    text=decode_text(tokens,model_name)+'\n\n... (truncated)'
    return text
    
def truncate_msg(msg,token_limit,model_name):
    if 'claude' in model_name:
        msg['content'][0]['text']=truncate_text(msg['content'][0]['text'],token_limit,model_name)
    else:
        msg['content']=truncate_text(msg['content'],token_limit,model_name)
    return msg


def compose_message(system=None,prompt=None,history=None):
    message=[]
    if system is not None:
        message.append(system)
    if history is not None:
        message+=history
    if prompt is not None:
        message.append(prompt)
    return message


CKPT_DIR = os.environ['CKPT_DIR']
CTX_ERROR_LOG_DIR = os.path.join(CKPT_DIR,'ctx_error_logs')
os.makedirs(CTX_ERROR_LOG_DIR,exist_ok=True)

# XXX: seems still not guaranteed to be safe, why??
def context_safe_guard(message,model_name,system_tokens=None): # message: list of dicts [{ 'role':str , 'content':str }]
    # Get system and prompt messages without modifying original list
    effective_limit = get_token_limit(model_name)
    format_buffer = 100  # Adjust based on model's message formatting overhead

    if len(message)==1:
        tokens = count_msg(message[0],model_name)
        if tokens > effective_limit:
            with open(os.path.join(CTX_ERROR_LOG_DIR,f'prompt_{time.time()}.json'),'w') as f:
                json.dump(message,f)
            raise ValueError(f'Context Error: Prompt message is too long: {tokens} tokens')
        return message

    prompt = message[-1]
    is_claude=False
    if system_tokens is None:
        system=message[0]
        system_tokens = count_msg(system, model_name)
        history = message[1:-1] if len(message)>2 else []  # Get remaining messages if any
    else:
        is_claude=True
        history = message[:-1] if len(message)>1 else []

    history = copy.deepcopy(list(history))

    if system_tokens > effective_limit: # should not happen at all! Not make sense to query anymore
        with open(os.path.join(CTX_ERROR_LOG_DIR,f'system_{time.time()}.json'),'w') as f:
            json.dump(message,f)
        raise ValueError(f'Context Error: System message is too long: {system_tokens} tokens')
    prompt_tokens = count_msg(prompt, model_name)
    if prompt_tokens+system_tokens > effective_limit: # should not happen at all! Not make sense to query anymore
        with open(os.path.join(CTX_ERROR_LOG_DIR,f'prompt_{time.time()}.json'),'w') as f:
            json.dump(message,f)
        raise ValueError(f'Context Error: Prompt message is too long: {prompt_tokens} tokens')
    history_tokens = [count_msg(msg, model_name) for msg in history]

    while True:
        total_tokens = system_tokens + prompt_tokens + sum(history_tokens)
        if total_tokens < effective_limit:
            break
        non_last_tokens = sum(history_tokens[1::]) if len(history_tokens) > 1 else 0
        if non_last_tokens + system_tokens + prompt_tokens <= effective_limit: # can be solved by truncating the first message
            last_limit = effective_limit - system_tokens - prompt_tokens - format_buffer - non_last_tokens
            history[0] = truncate_msg(history[0], last_limit, model_name)  
            history_tokens[0] = count_msg(history[0], model_name)
        else:
            history.pop(0)
            history_tokens.pop(0)
    if is_claude:
        return compose_message(history=history,prompt=prompt)
    else:
        return compose_message(history=history,prompt=prompt,system=system)



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
    message = context_safe_guard(message,model._config.model_name)
    for i in range(model._config.num_calls):
        try:
            return call_model_structured(model,message,response_format,logprobs=logprobs)
        except Exception as e:
            model.logging.warning(
                f'Issue encountered while running running, msg={e}, retrying',
                exc_info=True
            )
            if 'timeout' in str(e) or 'timed out' in str(e) or 'invalid content' in str(e) or 'Internal server error' in str(e):
                time.sleep(2*(i+1))
            else:
                if 'context_length_exceeded' in str(e):
                    with open(os.path.join(CTX_ERROR_LOG_DIR,f'message_{time.time()}.json'),'w') as f:
                        json.dump(message,f)
                raise e

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
        use_cache: bool = False, # XXX: not working with self-managed contexts as its changing, hard to track
        system: Optional[str] = None,
        model_name: Optional[str] = 'claude-3-5-sonnet-20241022',
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
    model._config.model_name=model_name
    if use_cache:
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system=[{"type": "text", "text": system}]
    # if model_state is None:
    #     return _prompt_model_structured(
    #         model,
    #         messages,
    #         response_format,
    #         logprobs=logprobs,
    #         system=system,
    #         **kwargs
    #     )
    
    return _prompt_model_claude(model,messages,system,response_format,logprobs,use_cache,**kwargs)



def _prompt_model_claude(model,message,system,response_format,logprobs=False,use_cache=True,**kwargs) -> str:
    """Main method for calling the underlying LM. 
    
    :see: https://github.com/jiangjiechen/auction-arena/blob/main/src/bidder_base.py#L167
    :param prompt: 
        The input prompt object to the model. 
    
    """
    ERROR=[]
    system_tokens=count_tokens(system[0]['text'],model._config.model_name)
    message=context_safe_guard(message,model._config.model_name,system_tokens)
    for i in range(model._config.num_calls):
        try:
            return call_model_claude(model,message,system,response_format,logprobs,use_cache)
        except Exception as e:
            model.logging.warning(
                f'Issue encountered while running running, msg={e}, retrying',
                exc_info=True
            )
            if 'timeout' in str(e) or 'timed out' in str(e) or 'invalid content' in str(e) or 'Internal server error' in str(e):
                time.sleep(2**(i+1))
            else:
                raise e
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
        
    # TODO: guard here 
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






