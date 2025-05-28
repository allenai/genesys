from copy import deepcopy 
from typing import Union, Type,Callable, Dict

from .base import UtilResource
from .register import Registry
from .aliases import ConfigType
from .utils import *
from .param import (
    ModuleParams,
    ParamField
)
from .models.utils import check_model_type

__all__ = [
    "Agent"
]

@Registry(
    resource_type="config",
    name="exec_utils.agent",
)
class Params(ModuleParams):
    """Parameters for agent class
    
    :param agent_file: 
        The file specifying the agent properties (e.g., instruction, 
        personality type, model parameters, ...)
    :param agent_type:
        The type of underlying agent being employed or built. 
    :param agent: 
        The type of `Agent` class to instantiate. 
            
    """
    
    agent_file: str = ParamField(
        default='',
        metadata={"help" : 'Pointer to the agent speification file'}
    )
    agent_type: str = ParamField(
        default='',
        metadata={"help" : 'The type of agent to use'}
    )
    agent: str = ParamField(
        default='simple_agent',
        metadata={"help" : 'The agent implementation'}
    )

@Registry(
    resource_type="agent",
    name="simple_agent"
)
class Agent(UtilResource):

    _ALIASES = {
        "model"       : "agent_model_type",
        "agent_model" : "agent_model_type",
        "model_agent" : "agent_model_type",
        "tool"        : "tool_type",
    }

    @classmethod
    def from_config(cls,config,**kwargs):
        """Factory method for loading a particular type of agent

        :param config: 
            The global configuration used to build a type of agent 
            (either a tool or a model agent). 

        """
        agent_config = create_agent_config(config,cls._ALIASES,**kwargs)
        alias = cls._ALIASES.get(agent_config.agent_type,None)

        ### manual check of model settings
        check_model_type(agent_config)
    
        return Registry.build_model(
            agent_config.agent_type,
            agent_config,
            **kwargs
        ) 
        
    
