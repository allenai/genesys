from copy import deepcopy
from abc import abstractmethod
from typing import (
    Union,
    Type,
    Callable,
    Dict,
    TypeVar,
    List,
    Any
)
from .base import (
    UtilResource,
    UtilityModel
)
from .register import Registry
from .aliases import ConfigType
from .utils import *
from .group import GroupBase 
from .param import (
    ModuleParams,
    ParamField
)
from .frontend import SystemOut

__all__ = [
    "System",
    "SystemOut",
]

C = TypeVar("C",bound="System")
    
@Registry("config","llm_sim.system")
class Params(ModuleParams):
    """Parameters for agent class

    """
    system_type: str = ParamField(
        default='simple',
        metadata={"help" : 'The type of agent to use'}
    )

class SystemBase(UtilResource):
    """Base class for system implementations 
    
    Methods 
    ----------
    query_system(query: str)
        The main method for processing queries with the full system (alias for __call__)  

    """
    def __call__(self,query: str,**kwargs):
        return self.query_system(query,**kwargs)

    def for_frontend(self,feedback) -> SystemOut:
        """Special method for formatting this according to the frontend

        """
        raise NotImplementedError
        
    @abstractmethod
    def query_system(self,query: str,**kwargs):
        """Main function for implementing system calls

        :param query: the query to the system 
        """
        raise NotImplementedError
    

SystemOutput = Union[Dict[str,Any],SystemOut]

    
@Registry(
    resource_type="system_type",
    name="simple",
    cache="query_system",
) 
class System(SystemBase):
    """Base system that simply executes 
    
    Attributes
    ----------
    :param group: 
        The initialized agent group
    
    Methods 
    ----------
    query_system(query: str)
        The main method for processing queries with the full system and returning output 
    for_frontend(query): 
       (optional) Formats the system output to a compatible format for our frontend code 
    
    """

    
    def __init__(self,agent_group: List[Type[GroupBase]]) -> None:
        self.group = agent_group


    def for_frontend(self,feedback) -> SystemOut:
        """Special method for formatting this according to the frontend

        :param feedback: 
            The raw system feedback. 
        :returns: 
            A formatted version of that feedback compatible with frontend code. 
        """
        print_output = []
        for agent,feedback_dict in feedback.items():
            print_output += [
                f"`{agent}` says\n\n",
                f'{feedback_dict["feedback"]}\n'
            ]
                
        return SystemOut(
            print_output=print_output,
            raw_data=feedback
        )
    
    def query_system(self,query: str,**kwargs) -> SystemOutput:
        """Main function for implementing system calls. 
        This one just returns the output of the agents

        :param query: 
            The query to the overall system. 
        """
        frontend = kwargs.get("frontend",False)
        feedback =  self.group(query)

        if frontend:
            return self.for_frontend(feedback)
        return feedback 
        
    @classmethod
    def from_config(cls: Type[C],config: ConfigType,**kwargs) -> C:
        """Build a system from configuration 

        :param config: 
            The global configuration used to build the instance. 
        
        """
        agent_group = Registry.build_model("group_type",config)
        return cls(agent_group)
