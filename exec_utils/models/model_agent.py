from __future__ import annotations

import os
import json

from typing import (
    Union,
    TypeVar,
    List,
    Dict,
    Tuple,
    Callable,
    Type,
    Any
)
from ..cache import setup_cache
from ..register import Registry
from ..base import (
    UtilResource,
    ProtectedNameSpace,
    UtilityModel
)
from ..aliases import ConfigType
from ..param import (
    ModuleParams,
    ParamField
)
from .utils import parse_dict_output
from ..utils import create_agent_config

__all__ = [
    "ModelAgentBase",
    "SimpleLMAgent",
]


class ModelAgentBase(UtilResource):
    """Base class for agent models

    Attributes
    ----------
    :param model: 
        The underlying LLM used for this agent. 
    :param model_state: 
        Tracks the current state of the model in use. 
    :param config
        The configuration for this agent with details about 
        the agent type, name, (optionally) type of model being
        used. 
    :param cost: 
        The running cost of the model. 

    Methods 
    ----------
    query(query: str) -> dict
        The main method for handling input queries to the agent. 

    """
    
    def query(
            self,
            query: str,
            source: Optional[str]="user",
            **kwargs
        ):
        """Main method for querying the underlying model 
        
        :param query: 
            The target query to the agent model 
        :param source: 
            The source of the query (optional) that 
            indicates where the query is coming from 
            (e.g., human, assistant, etc)

        """
        raise NotImplementedError

    def parse_output(self,raw_output: str) -> Any:
        """Execution the output of the agent  

        :param raw_output: 
        """
        return raw_output 
    
    def __call__(self,query: str,**kwargs):
        return self.query(query,**kwargs)

    @property
    def details(self) -> Dict[str,Any]:
        """Return details of the underlying agent 
        """
        return self._config.agent_details

    @property
    def name(self) -> str:
        """Returns the name of the agent"""
        return self._name

DICT_FORMAT="""
Please format your answer using a python dictionary or json representation as follows
{{
    "feedback" : '''<your feedback in triple quotes>'''
}}
""".strip()

@Registry(
    resource_type="agent_model_type",
    name="simple",
    cache="query"
)
class SimpleLMAgent(ModelAgentBase):
    """A simple base agent model that takes an `instruction`, `prompt` and produces output in some (optional)
    `output_format` 
    
    Attributes
    ----------
    :param model: 
        The underlying LLM. 
    :param config:
        The configuration for this agent. 
    :param agent_details: 
        A dictionary description of the agent. 
    :param cost: 
        The running cost of the model. 
    :param model_state: 
        The state of the model at the moment of use. 

    Methods 
    ----------
    query(query: str) -> dict
        The main method for handling input queries to the agent. 
    parse_output(raw_output: str) -> dict
        Main method for formatting the output of the agent after 
        running a query. 

    """
    _AGENT_INSTRUCTION: str = "You are a helpful agent who is meant to answer questions"
    _FORMAT: str = DICT_FORMAT
    _EXAMPLES: Optional[List[str]] = []
    USE_HISTORY: bool = False
    
    def __init__(
            self,
            name: str,
            instruction: str,
            output_format: str,
            model: Type[PromptableModel],
            examples: Optional[List[str]]=[],
            config: Optional[ConfigType]=None, 
            **kwargs
    ) -> None:
        """Initializes agent model

        :param name: 
            The name of the agent. 
        :param instruction: 
            An optional persistent instruction string 
            for the agent. 
        :param output_format: 
            An optional specification of the output format 
            for this agent. 
        :param model: 
            The underlying LLM model for this agent. 
        :param examples: 
            The optional in-context examples for this agent. 
        :param config: 
            The configuration for this agent, with more details
            about the agent, model settings, etc. 

        """

        self.model = model

        self.model_state = self.model.create_model_state(
            instruction=instruction,
            examples=examples,
            output_format=output_format,
            use_history = self.USE_HISTORY or config.use_history
        )
        self._config = config
        self._name = name

        custom_details = self._config.agent_details.get("_params",{}) 
        self.logging.info(
            f"Agent name=`{name}`, model_details={json.dumps(custom_details,indent=4)}"
        )
        
        
    def parse_output(self,raw_output: ModelOutput) -> Dict[Any,Any]:
        """Execution the output of the agent.   

        :param raw_output: 
        """
        parsed_out = parse_dict_output(raw_output.text)
        try: 
            parsed_out["_details"]["agent_profile"] = {
                "name" : self._config.agent_details.get("name","unknown"),
                "instruction" : self._config.agent_details.get("instruction","unknown"),
                "agent_type"  : self._config.agent_details["agent_type"],
            }
        except:
            pass
        
        
        if self.model_state.use_history:
            self.model_state.add_to_history(
                parsed_out["_details"]["raw_output"],
                "assistant"
            )
            
        parsed_out["_details"]["running_cost"] = self.cost
        parsed_out["_details"]["cost"] = raw_output.cost
        parsed_out["_details"]["input_tokens"] = raw_output.input_tokens
        parsed_out["_details"]["output_tokens"] = raw_output.output_tokens
        
        return parsed_out

    def clear_history(self) -> None:
        """Called when starting a new interaction or 
        dialogue with the agent. 

        """
        self.logging.debug("Clearning the history") 
        self.model_state.reset_state()
        
    def query(
            self,
            query: str,
            source: Optional[str] = "user",
            manual_history: Optional[Tuple[str,str]] = [], 
            **kwargs
        ) -> Dict[str,Any]:
        """Main method for querying the underlying model from a single input
        
        :param query: 
            The target query to pass to the agent. 
        :param source: 
            The source of the query (optional) that indicates where 
            the query is coming from (e.g., human, assistant, etc)
        :param intermediate: 
            Indicates whether the query is part of a series of queries
            to the agent (which requires using the history)  

        """
        self.model_state.query_state = source

        
        raw_response = self.model(
            prompt=query,
            model_state=self.model_state,
            history=manual_history
        )
        ### add to history
        if self.model_state.use_history and not manual_history:
            self.model_state.add_to_history(query,source)
            
        ### parse output and add to history
        response = self.parse_output(raw_response)
            
        return response

    @classmethod
    def from_config(cls,config: ConfigType,**kwargs) -> AgentType:
        """Load an agent from configuration.  

        :param config: 
            The agent configuration to create an object.  
        :raises: 
            ValueError
        """
        if not hasattr(config,"agent_details"):
            config = create_agent_config(config)
        
        model = Registry.build_model("model_type",config,**kwargs)
        agent_details = config.agent_details
        name = agent_details.get("name","nameless_agent")
        
        ## default to class instructions and format if not specified
        instruction = agent_details.get("instruction",cls._AGENT_INSTRUCTION)
        output_format = agent_details.get("format",cls._FORMAT)
        examples = agent_details.get("examples",cls._EXAMPLES)

        
        ## build the agent 
        return cls(
            name,
            model=model,
            instruction=instruction,
            output_format=output_format,
            examples=examples,
            config=config
        )

    @property
    def cost(self) -> float:
        """Estimate the total cost of this agent during session
      
        :returns: simply returns the underlying model costs 
        """
        return self.model.cost

    @property
    def config(self):
        return self._config

    
@Registry(
    resource_type="config",
    name="llm_sim.models.model_agent"
)
class Params(ModuleParams):
    """Parameters for agent class
    
    :param use_history: 
        Binary switch that specifies that past history when 
        be used when building prompt. 
    :param agent_model_type: 
        Specifies the particular agent model implementation 
        that should be used (e.g., openaiAI API or langchain). 

    """
    model_config = ProtectedNameSpace

    use_history: bool = ParamField(
        default=False,
        metadata={"help" : 'Use history when running model'}
    )
    agent_model_type: str = ParamField(
        default="simple",
        metadata={"help" : 'The type of agent model to use'}
    )
