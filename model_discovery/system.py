from types import ModuleType
from typing import (
    Type,
    List,
    Dict,
    Any,
    TypeVar,
    Tuple,
    Optional,
    Union
)
import exec_utils
from exec_utils.aliases import ConfigType
from exec_utils import BuildAgent
from .agents import *

C = TypeVar("C",bound="DiscoverySystem")

__all__ = [
    "DiscoverySystem",
    "BuildSystem",
]

@exec_utils.Registry("config","discovery_system")
class CustomParams(exec_utils.ModuleParams):
    """Parameters for working with this discovery system 
    
    :param max_design_attempts: 
        The number of attempts that the designer agent ca 
        make. 
    :param max_design_refines: 
        The number of times the designer can refine a design 
    :param reviwer_threshold: 
        The threshold for accepting a design for the reviewer. 
    :param designer_spec: 
        Pointer to the designer specification 
    :param reviewer_spec: 
       Pointer to the reviewer specification. 

    """
    max_design_attempts: int = exec_utils.ParamField(
        default=10,
        metadata={
            "help"         : 'The maximum number of designer queries',
            "exclude_hash" : True,
        }
    )
    max_design_refines: int = exec_utils.ParamField(
        default=10,
        metadata={
            "help"         : 'The maximum number of designer refinements',
            "exclude_hash" : True,
        }
    )
    reviewer_threshold: int = exec_utils.ParamField(
        default=5,
        metadata={
            "help"         : 'The threshold for accepting a design',
            "exclude_hash" : True,
        }
    )

    ### agent profiles
    designer_spec: str = exec_utils.ParamField(
        default='etc/agent_spec/designer.json',
        metadata={
            "help"         : 'Specification of design agent',
            "exclude_hash" : True,
        }
    )
    reviewer_spec: str = exec_utils.ParamField(
        default='etc/agent_spec/reviewer.json',
        metadata={
            "help"         : 'Specification of reviewer agent',
            "exclude_hash" : True,
        }
    )

@exec_utils.Registry(
    resource_type="system_type",
    name="discovery_system",
    #cache="query_system",
)
class DiscoverySystem(exec_utils.System):
    """Overall system for discovery

    """

    def __init__(
        self,
        designer : Type[exec_utils.SimpleLMAgent],
        reviewer : Type[exec_utils.SimpleLMAgent],
        config: ConfigType 
    ) -> None:
        """Create a `DiscoverySystem` instance 

        
        :param designer: 
            The designer agent. 
        :param reviewer: 
           The reviewer agent. 
        :param config: 
           System global configuration. 
        """
        self.designer = designer
        self.reviewer = reviewer
        self._config = config
        
    def query_system(
        self,
        query: str,
        stream: Optional[ModuleType] = None,
        frontend: Optional[bool] = False,
        status: Optional[bool] = True,
        model_name: Optional[str]='',
        **kwargs
    ) -> list:
        """Main function for implementing system calls.

        :param query: 
            The query to the overall system. 
        :param stream: 
            The streamlit module for writing to frontend 
        :param frontend: 
            Switch indiciating whether system is being used 
            with a frontend. 
        :param model_name: 
            The name of the model to query (if specified and 
            multiple models are provided). 
        

        """
        print("hello world")
    
    @classmethod
    def from_config(cls: Type[C],config: ConfigType,**kwargs) -> C:
        """The main method for instantiating system instances from configuration. 

        :param config: 
            The global configuration used to create instance. 
        :returns: 
            A `DiscoverySystem` instance from configuration.  
        """
        ### creates designer and reviewer agents
        
        designer = BuildAgent(
            config,
            agent_file=config.designer_spec,
            agent_model_type="designer_agent"
        )
        reviewer = BuildAgent(
            config,
            agent_file=config.reviewer_spec,
            agent_model_type="reviewer_agent"
        )

        ## can add more components as needed 

        return cls(
            designer,
            reviewer,
            config
        )
    
def BuildSystem(
        config: Optional[ConfigType] = None,
        **kwargs
    ):
    """Factory for building an overall system

    :param config: 
        The optional configuration object. 
    """
    from exec_utils import BuildSystem
    kwargs["system_type"] = "discovery_system"
    return BuildSystem(config,**kwargs)
