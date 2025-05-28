import time
import openai

from abc import abstractmethod
from dataclasses import (
    dataclass,
    field
)
from typing import (
    Union,
    TypeVar,
    List,
    Dict,
    Tuple,
    Callable,
    Type,
    Any,
    Optional,
    Generic
)
from ..aliases import ConfigType
from ..base import (
    UtilResource,
    UtilBase,
    ProtectedNameSpace,
    UtilityModel
)
from ..param import (
    ModuleParams,
    ParamField,
)
from .prompt import (
    SimplePrompt,
    format_prompt
)
from .utils import *
from ..utils import *
from ..cache import setup_cache
from ..register import Registry

from langchain_community.callbacks import get_openai_callback
from langchain_core.exceptions import LangChainException
from langchain_core.language_models import BaseLanguageModel as LangchainModel
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage as LangchainMessage
)
# from vertexai.preview.generative_models import (
#     Content,
#     FunctionDeclaration,
#     GenerativeModel,
#     Part,
#     Tool
# )

__all__ = [
    "ModelBase",
    "LangchainModel",
    "OpenAIModel",
    "ModelState",
]

C = TypeVar("C",bound="ModelBase")
T = TypeVar("T")

PromptType = Union[SimplePrompt,str]


class ModelRuntimeError(Exception):
    pass


class ModelBase(UtilResource):
    """Base class building models
    
    Attributes
    --------
    :param cost: 
        The running cost of the model (set to 0. by default).
    :param utility_type: 
        The type of utility. 
    """

    def prompt(self,prompt: PromptType,**kwargs) -> Dict[str,Any]:
        """High-level method to call the  model. 

        :param prompt: 
            The target prompt, either in string form or formatted. 
        """
        return self(prompt)

    query = prompt
    
    def __call__(self,prompt: PromptType,**kwargs):
        raise NotImplementedError
    
    @property
    def cost(self) -> float:
        """Returns the cost associated with a model (by default 0.)

        :returns: 
            The dollar amount associated with this model. (0. by defailt) 
        """
        self.logging.warning(
            'Explicit cost function not implemented!'
        )
        return 0.

    @property
    def utility_type(self) -> str:
        """The type of utility of this item
        """
        return "model"

class ModelOutput(UtilityModel):
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
    
    def __repr__(self):
        return self.text

class ModelState(Generic[T]):
    """Utility class for tracking a model's state. 
    """

    def __init__(
            self,
            fn: Callable[Any,List[T]], 
            instruction: Optional[str]="",
            examples: Optional[List[Any]]=[],
            output_format: Optional[str] = "",
            use_history: Optional[bool] = False,
    ) -> None:
        """Creates a model state instance. 

        :param fn: 
            The function used for creating model message. 
        :param static_message: 
            The static message for the model. 
        :param output_format: 
            The model's outptu format (optional) 

        """
        self.fn = fn
        self.static_message: List[T] = self.fn(
             instruction=instruction,
             examples=examples
        )

        self.output_format = output_format
        
        ## private variables 
        self._history: List[T] = []
        self._query_state = "user"
        self._intermediate = False
        self.use_history = use_history 

    @property
    def history(self) -> List[T]:
        """Returns the history 
        """
        return self._history
    
    @property
    def query_state(self) -> str:
        """Returns the query state
        """
        return self._query_state

    @query_state.setter
    def query_state(self,state: str) -> None:
        """Changes the query state 
        """
        if state not in set(["assistant","user","system"]):
            raise ValueError(
                f"Unknown message type: {state}"
            )
        self._query_state = state
        
    
    def reset_state(self) -> None:
        """Resets the model state by clearning the history, 
        the intermediate history, and intermediate status.
        """
        self._history = []
        self._query_state = "user"
        self._intermediate = False


    def add_to_history(self,msg: str,msg_type: str) -> None:
        """Add items to the intermediate history for laer 

        :param msg: 
            The message to add to the history. 
        :param msg_type: 
            The type of message to add (either, "user" or "assistant") 
        :raises: 
            ValueError 
        """
        if self.use_history is False:
            return 
        
        if msg_type not in set(["assistant","user","system"]):
            raise ValueError(f"Unknown message type: {msg_type}")

        for m in self.fn(history=[(msg,msg_type)]): 
            self._history.append(m)

    def create_message(
        self,
        query: str,
        manual_history: Optional[List] = None,
    ) -> List[T]:
        """Creates a message object

        :param query: 
            The input string query to the model.  
        :param fn: 
            Model specific functin for formatting data
        """
        full_history = []

        if self.output_format:
            query = f"{query}\n{self.output_format}"
        
        query_m = self.fn(
            query=query,
            query_state=self._query_state
        )
        
        for m in self.static_message:
            full_history.append(m)

        ### history 
        if self.use_history and not manual_history:
            for m in self._history: 
                full_history.append(m)
        elif manual_history:
            history = self.fn(history=manual_history)
            for m in history:
                full_history.append(m)
                
        for m in query_m:
            full_history.append(m)
            
        return full_history         
    

class PromptableModel(ModelBase):
    """Base class building models
    
    Attributes
    --------
    :param cost: 
        The running cost of the model (set to 0. by default).
    :param utility_type: 
        The type of utility. 
    """
    PromptableState = ModelState[Any]
    
    def __init__(
            self,
            model_obj: Any,
            config: ConfigType,
            max_tokens: Union[
                Callable[int,Dict[str,int]],
                None
            ] = None,
            stop: Optional[List[str]] = None
    ) -> None:
        """Creates a promptable moel intance that includes a `model_obj`, 
        a configuration and (optionally) a function for computing maximum 
        tokens. 


        :param model_obj: 
            The underlying model object or instance (will differ
            depending on the API) 
        :param config: 
            The global configuration for the model. 
        :param max_tokens: 
            An optional function for computing model maximum length. 
        :param stop: 
            A list of stop symbols for the model if needed 

        """
        self.model_obj = model_obj
        self._config = config
        self._max_tokens = max_tokens 
        self._model_cost = 0.
        self._stop = stop
        
    @property
    def config(self) -> ConfigType:
        """Returns the model's configuration"""
        
        return self._config

    @property
    def cost(self) -> float:
        """Returns the cost associated with a model (by default 0.)

        :returns: 
            The dollar amount associated with this model. (0. by defailt) 
        """
        return self._model_cost

    @classmethod
    def create_message(
            cls,
            query: Optional[str] = "",
            instruction: Optional[str]="",
            examples: Optional[List[Any]]=[],
            history: Optional[List[Tuple[str,str]]] = [],
            query_state: Optional[str] = "user",
        ) -> List[
            Any
        ]:
        """Main method for formatting data for the promptable model.

        :param instruction: 
            The (optional) model instruction 
        :param examples: 
            The (optional) set of model examples marked with the 
            the type associated with each example  
        :param history: 
            The history of the model up to that point, which includes 
            additional examples. 
        :param query_state: 
            The source of the current query. 

        """
        messages = []
        if instruction:
            messages.append(
                cls.format_message(instruction,"system")
            )
        if examples:
            messages += [
                cls.format_message(m,t) for (m,t) in examples
            ]
        if history:
            messages += [
                cls.format_message(m,t) for (m,t) in history
            ]
        if query:
            messages.append(
                cls.format_message(query,query_state)
            )

        return messages

    @classmethod
    def format_message(cls,msg,msg_type):
        raise NotImplementedError
    
    def call_model(self,message):
        raise NotImplementedError
    
    def _prompt_model(self,message,**kwargs) -> str:
        """Main method for calling the underlying LM. 
        
        :see: https://github.com/jiangjiechen/auction-arena/blob/main/src/bidder_base.py#L167
        :param prompt: 
            The input prompt object to the model. 
        
        """
        for i in range(self._config.num_calls):
            try:
                return self.call_model(message)
            except Exception as e:
                self.logging.warning(
                    f'Issue encountered while running running, msg={e}, retrying',
                    exc_info=True
                )
                
                time.sleep(2**(i+1))

        raise ModelRuntimeError(
            f'Error encountered when running model, msg={e}'
        )

                
    def prompt(self,prompt: PromptType,**kwargs) -> Dict[str,Any]:
        """High-level method to call the  model. 

        :param prompt: 
            The target prompt, either in string form or formatted. 
        """
        return self(prompt,**kwargs)
    
    def __call__(
            self,
            prompt: str,
            instruction: Optional[str]="",
            examples: Optional[List[Any]] = [],
            history: Optional[List[Any]] = [],
            model_state: Optional[ModelState] = None,
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
            return self._prompt_model(
                self.create_message(
                    query=prompt,
                    instruction=instruction,
                    examples=examples,
                    history=history,
                ),
                **kwargs
            )
        
        message = model_state.create_message(
            query=prompt,
            manual_history=history
        )
        return self._prompt_model(message,**kwargs)
    
    @classmethod
    def create_model_state(
            cls,
            instruction: Optional[str]="",
            examples: Optional[List[Any]]=[],
            output_format: Optional[str]="",
            use_history: Optional[bool]=False,
        ) -> ModelState:
        """Create an initial model state based on provided instructions 
        and set of in-context examples.

        :param instruction: 
            The static instruction for this model. 
        :param examples: 
            The set of static in-context examples for this 
            model. 
        :param output_format: 
            An optional string specification of the output 
            format. 

        """
        
        return cls.PromptableState(
            fn=cls.create_message,
            instruction=instruction,
            examples=examples,
            output_format=output_format,
            use_history=use_history
        )
    
@Registry(
    resource_type="model_type",
    name="openai",
    cache=None,
)
class OpenAIModel(PromptableModel):
    """Wrapper around OpenAI API for running different GPT models 
    
    Example
    ----------

    Below is an example for how to load a generic `OpenAIModel` using 
    the `BuildModel` factory 

    >>> from exec_utils import BuildModel 
    >>> model = BuildModel(model_type='openai')
    >>> model("what is your name") 

    """
    PromptableState = ModelState[Dict[str,str]]

    @classmethod 
    def format_message(cls,msg: str,msg_type: str) -> Dict[str,str]:
        return {"role" : msg_type, "content": msg}
        
    def call_model(self,message) -> ModelOutput:
        """Calls the underlying model 
        
        :param message: 
            The formatted message to pass to move.  

        """
        response_format = None
        json_set = {
            "gpt-3.5-turbo-0125"                      : "openai",
            "gpt-4-0125-preview"                      : "openai",
            #"mistralai/Mixtral-8x7B-Instruct-v0.1"    : "together",
            "mistralai/Mistral-7B-Instruct-v0.1"      : "together",
            "togethercomputer/CodeLlama-34b-Instruct" : "together",
        }
            
        if self._config.model_name in json_set:
            
            response_format={ "type": "json_object" }
            json_schema = None 
            if hasattr(self._config,"agent_details"): 
                json_schema = self._config.agent_details.get("json_schema")
                
            # https://www.together.ai/blog/function-calling-json-mode
            if json_schema and json_set[self._config.model_name] == "together":
                response_format["schema"] = json_schema

        completions = self.model_obj.chat.completions.create(
            model=self._config.model_name,
            messages=message,
            max_tokens=self._config.max_output_tokens,
            stop=self._stop,
            temperature=self._config.temperature,
            response_format=response_format,
        )

        output = completions.choices[0].message.content

        cost = openai_costs(
            self._config.model_name,
            completions.usage
        )
        self._model_cost += cost
        return ModelOutput(
            text=output,
            cost=cost,
            input_tokens=completions.usage.prompt_tokens,
            output_tokens=completions.usage.completion_tokens
        )
    
    @classmethod
    def from_config(cls: Type[C],config: ConfigType,**kwargs) -> C:
        """Build a model instance from configuration 

        :param config: 
            The global configuration used to create an instance. 
        """
        openai_client,stop = setup_openai(config) 
        return cls(openai_client,config,stop=stop)


@Registry(
    resource_type="model_type",
    name="lang_chain",
    cache=None
)
class LangchainModel(PromptableModel):
    """Wrapper around langchain API. Is currently used to run OpenAI 
    and Google/VertexAI models.
    
    Attributes
    ----------
    :param model: 
       The underlying langchain model 
    :param max_tokens:  
       A function specific to the type of the langchain model for computing token output 
    :param cost: 
       A running total of the model costs 
    :param config:  
       The global configuration . 

    Methods 
    ----------
    prompt(prompt): 
        The main method for processing queries with the model 

    
    Example
    ----------

    Below is an example for how to load a generic `LangchainModel` using 
    the `BuildModel` factory 

    >>> from exec_utils import BuildModel 
    >>> model = BuildModel(model_type='lang_chain')
    >>> model("what is your name") 

    """
    PromptableState = ModelState[LangchainMessage]
    
    @classmethod
    def format_message(cls,msg: str,msg_type: str) -> LangchainMessage:
        if msg_type == "assistant":
            return AIMessage(content=msg)
        elif msg_type == "user":
            return HumanMessage(content=msg)
        elif msg_type == "system":
            return SystemMessage(content=msg)
        
        raise ValueError(
            f"Unknown msg_type={msg_type}"
        )

    def call_model(self,message) -> ModelOutput:
        """Calls the underlying model 
        
        :param message: 
            The formatted message to pass to move.  

        """
        with get_openai_callback() as cb:
            input_num = self.model_obj.get_num_tokens_from_messages(message)
            kwrd = self._max_tokens(input_num)
            result = self.model_obj.invoke(message,**kwrd)

        self._model_cost += cb.total_cost
        return ModelOutput(
            text=result.content
        )
    

    @classmethod
    def from_config(cls: Type[C],config: ConfigType,**kwargs) -> C:
        """Build a model instance from configuration 

        :param config: 
            The global configuration used to create an instance. 
        """
        model,max_length = setup_langchain(config)
                
        return cls(
            model_obj=model,
            max_tokens=max_length,
            config=config
        )

# @Registry(
#     resource_type="model_type",
#     name="vertex",
#     cache=None
# )
# class VertexModel(PromptableModel):
#     """Class for working with vertexAI models and APIs.

#     Example
#     ----------

#     Below is an example for how to load a generic `LangchainModel` using 
#     the `BuildModel` factory 

#     >>> from exec_utils import BuildModel 
#     >>> model = BuildModel(model_type='vertex',model_name="gemini-pro")
#     >>> model("what is your name") 

#     """

#     PromptableState = ModelState[Content]

#     @classmethod
#     def format_message(cls,msg: str,msg_type: str) -> Content:
#         """Formats a mesaage for the vertexai model 

#         :param msg: 
#             The message to pass in string form. 
#         :param msg_type: 
#             The source of the message (e.g., ai, human, etc..x)

#         """
#         role_name = ''
#         if msg_type == "assistant":
#             role_name = "model"
#         elif msg_type == "user" or msg_type == "system":
#             role_name = "user"
#         else:
#             raise ValueError(
#                 f"Unknown role type: {msg_type}"
#             )
#         return Content(
#             role=role_name,
#             parts=[Part.from_text(msg)]
#         )

#     def call_model(self,message) -> ModelOutput:
#         """Calls the underlying model 
        
#         :param message: 
#             The formatted message to pass to move.  

#         """
#         model_out = self.model_obj.generate_content(message)
        
#         return ModelOutput(
#             text=model_out.text
#         )

#     @classmethod
#     def from_config(cls: Type[C],config: ConfigType,**kwargs) -> C:
#         """Build a model instance from configuration 

#         :param config: 
#             The global configuration used to create an instance. 
#         """
#         model = GenerativeModel(config.model_name)
        
#         #vertexai.init(project=PROJECT_ID, location=LOCATION)
        
#         return cls(
#             model_obj=model,
#             config=config
#         )
        

    
@Registry(
    resource_type="config",
    name="lm_exec_utils.agents.model"
)
class Params(ModuleParams):
    """Parameters for model classes
    
    :param model_type: 
        The type of model implementation class. 
    :param model_name: 
        The name of the particular model type (e.g., gpt4).
    :param temperature: 
        The particular temperature parameter to used for decoding 
    :param max_tries: 
        The maximum number of retries to call the model after failures, langchain 
        openai setting
    :param num_tries: 
        The maximum number of manual tries in this code. 

    """
    
    model_type: str = ParamField(
        default="openai",
        metadata={"help" : 'The type of agent model to use' }
    )
    model_name: str = ParamField(
        default='gpt-3.5-turbo',
        metadata={"help" : 'The particular type of base model to use'}
    )
    temperature: float = ParamField(
        default=0.0,
        metadata={"help" : 'Decoding temperature'}
    )
    max_retries: int = ParamField(
        default=30,
        metadata={
            "help" : 'The number of retries for the model',
            "exclude_hash" : True,
        }
    )
    num_calls: int = ParamField(
        default=5,
        metadata={
            "help" : 'The number of calls internally to try',
            "exclude_hash" : True,
        }
    )
    openai_api_key: str = ParamField(
        default='',
        metadata={
            "help" : 'Openai api key (if not specified in environment)',
            "exclude_hash" : True,
        }
    )
    together_api_key: str = ParamField(
        default='',
        metadata={
            "help" : 'Together.ai API key',
            "exclude_hash" : True,
        }
    )
    max_output_tokens: int = ParamField(
        default=500,
        metadata={"help" : 'The maximum number of output tokens'}
    )
