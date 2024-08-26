''' System of Review & Design Agents for Model Discovery '''

import exec_utils
import pathlib
import os

import numpy as np

from typing import (
    Type,
    List,
    Dict,
    Any,
    TypeVar,
    Tuple,
    Optional,
    Union,
    NamedTuple,
)
from types import ModuleType
from dataclasses import dataclass

from exec_utils.aliases import ConfigType
from exec_utils import (
    BuildAgent,
    BuildTool
)
from .agents.roles import *
from .agents.flow.alang import AgentDialogManager,AgentDialogFlowNaive,ALangCompiler,SYSTEM_CALLER,FAILED,ROLE
from .agents.flow.naive_flows import design_flow_definition,review_naive,design_naive,naive_design_review
from .agents.flow.gau_flows import gu_design_scratch

import model_discovery.utils as U

C = TypeVar("C",bound="ModelDiscoverySystem")

from .configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

# import multiprocessing as mp

__all__ = [
    "ModelDiscoverySystem",
    "BuildSystem",
]

PROJ_SRC = os.path.abspath(os.path.dirname(__file__))
SYSTEM_OUT = os.path.abspath(f"{PROJ_SRC}/../_runs")



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
    :param block_template: 
       Points to the file for the GAB block for prompting. 
    :param gam_template: 
       Points to the file for the GAM template used in the prompt 
    :param debug_step: 
       Print the system steps when running. 
    :param gam_config: 
       The target configuration for the GAM model being explored.
    :param run_name: 
       The name identifier of the model search session. 

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
        default=3,
        metadata={
            "help"         : 'The threshold for accepting a design',
            "exclude_hash" : True,
        }
    )
    ### agent profiles
    designer_spec: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/designer.json'
        ),
        metadata={
            "help"         : 'Specification of design agent',
            "exclude_hash" : True,
        }
    )
    reviewer_spec_creative: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/reviewer_creative.json'
        ),
        metadata={
            "help"         : 'Specification of creative reviewer agent',
            "exclude_hash" : True,
        }
    )
    reviewer_spec_balance: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/reviewer_balance.json'
        ),
        metadata={
            "help"         : 'Specification of balance reviewer agent',
            "exclude_hash" : True,
        }
    )
    reviewer_spec_rigorous: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/reviewer_rigorous.json'
        ),
        metadata={
            "help"         : 'Specification of rigorous reviewer agent',
            "exclude_hash" : True,
        }
    )
    debugger_spec: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/debugger.json'
        ),
        metadata={
            "help"         : 'Specification of debugger agent',
            "exclude_hash" : True,
        }
    )
    ### code information
    block_template: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/agents/prompts/gab_template.py'
        ),
        metadata={
            "help"         : 'Location of block for prompting ',
        }
    )
    gam_template: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/agents/prompts/gam_prompt.py'
        ),
        metadata={
            "help" : 'Location of code prompting ',
        }
    )
    gam_implementation: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/model/gam.py'
        ),
        metadata={
            "help" : 'Location of code prompting ',
        }
    )
    lib_dir: str = exec_utils.ParamField(
        default=None,
        metadata={
            "help" : 'GAU Library directory',
        }
    )
    ### debugging
    debug_steps: bool = exec_utils.ParamField(
        default=True,
        metadata={
            "help"         : 'Debug the steps of the system',
            "exclude_hash" : True,
        }
    )
    jupyter: bool = exec_utils.ParamField(
        default=False,
        metadata={
            "help"         : 'Inside of jupyter',
            "exclude_hash" : True,
        }
    )
    gam_config: str = exec_utils.ParamField(
        default='GAMConfig_14M',
        metadata={
            "help"         : 'Debug the steps of the system',
            "exclude_hash" : True,
        }
    )
    #### name of the system run 
    run_name: str = exec_utils.ParamField(
        default='demo',
        metadata={
            "help"         : 'The name of the run',
            "exclude_hash" : True,
        }
    )
    from_json: str = exec_utils.ParamField(
        default='',
        metadata={
            "help"         : 'The location of json',
            "exclude_hash" : True,
        }
    )


def get_context_info(config,templated=False) -> Tuple[str,str]:
    """Grabs the block and model implementation details for the prompt 

    :param config: 
        The global configuration 
    :raises: ValueError 

    """
    if not os.path.isfile(config.block_template):
        raise ValueError(f'Cannot find the block template: {config.block_template}')
    if templated:
        config.gam_template = config.gam_template.replace('.py','_templated.py')
    if not os.path.isfile(config.gam_template):
        raise ValueError(f'Cannot find the code context')
    block = open(config.block_template).read()
    code = open(config.gam_template).read()
    
    return (block,code) 

class NaiveHandler: 
    def __init__(self,message,*args,**kwargs):
        self.message = message
        
    def __enter__(self):
        print(f'\n[START: {self.message}]\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'\n[FINISH: {self.message}]\n')

class PrintSystem:
    def __init__(self,config):
        self.jupyter = config.jupyter   
        self.status = NaiveHandler     
        self._isprintsystem = True
    
    def write(self,msg,**kwargs):
        print(msg)
    def markdown(self,msg,**kwargs):
        print(msg)


@exec_utils.Registry(
    resource_type="system_type",
    name="model_discovery_system",
    #cache="query_system",
)
class ModelDiscoverySystem(exec_utils.System):
    """Overall system for discovery
    
    Attributes
    --------
    :param designer: 
        The designer LLM agent. 
    :param reviewers: 
        The reviewer LLM agents. 
    :param checker: 
        The checker tool agent. 
    :param debugger:
        The debugger LLM agent.
    :param block_template: 
        The templated code to be filled in during 
        the block search. 
    :param gam_config: 
        the global model configuration 
    :param config: 
        The system level configuration. 
        
    Methods 
    --------
    query_sysetem(query, stream, frontend, status) 
        The main method for querying the system with the 
        different agents. 

    Example 
    --------

    >>> from model_search import BuildSystem  
    >>> system = BuildSystem()

    """

    def __init__(
        self,
        designer : Type[exec_utils.SimpleLMAgent],
        reviewers : Dict[str,Type[exec_utils.SimpleLMAgent]],
        checker  : Type[exec_utils.BaseTool], 
        debugger : Type[exec_utils.SimpleLMAgent],
        lib_dir: str,
        *,
        block_template: str,
        gam_template: str,
        gam_config: Type[GAMConfig],
        config: ConfigType 
    ) -> None:
        """Create a `DiscoverySystem` instance 

        
        :param designer: 
            The designer agent. 
        :param reviewers: 
           The reviewer agents. 
        :param block_template: 
           The autoregressive block that the model has to fill in.
        :param model_implementation: 
           The full model implementation for context 
        :param config: 
           System global configuration. 

        """
        ### modules 
        self.designer = designer
        self.reviewers = reviewers
        self.checker = checker
        self.debugger = debugger
        
        self.gam_py = gam_template
        self.gab_py = block_template
        self._config = config
        self._cfg = gam_config
        self.lib_dir = lib_dir


        # Load flows

        self.review_flow=AgentDialogFlowNaive('Model Review Flow',review_naive)
        self.DESIGN_ALANG =design_flow_definition()
        self.design_flow,self.DESIGN_ALANG_reformatted=ALangCompiler().compile(self.DESIGN_ALANG,design_flow_definition,reformat=True)

        self.design_flow_naive=AgentDialogFlowNaive('Model Design Flow',design_naive)

        # self.design_fn=naive_design_review
        self.design_fn_scratch=gu_design_scratch

    def get_system_info(self):
        system_info = {}
        system_info['agents']={
            'designer':self.designer.config,
            'reviewers':{style:agent.config for style,agent in self.reviewers.items()},
            'debugger':self.debugger.config
        }

    def new_session(self,log_dir=None,stream=None):
        U.mkdir(log_dir)
        self.log_dir = log_dir
        self.sess_state = {} 
        self.dialog = AgentDialogManager(log_dir,self.get_system_info(),stream)

    def set_config(self,cfg:GAMConfig): # reset the gam config of the system
        self._cfg = cfg

    def query_system(
        self,
        query: Optional[str] = '',
        log_dir: Optional[str] = None,
        stream: Optional[ModuleType] = None,
        frontend: Optional[bool] = False,
        status: Optional[bool] = True,
        metadata = None, # such as seeds, non str instructions
        **kwargs
    ) -> list:
        """Main function for implementing system calls.

        :param query: 
            The query to the overall system. It should be an instruct from the upper evo system
        :param stream: 
            The (optional) streamlit module for writing to frontend 
        :param frontend: 
            Switch indiciating whether system is being used 
            with a frontend.
        :param metadata:
            Additional information about the query. Mainly about the seeds.
        """

        status_handler = stream.status if stream and status else NaiveHandler
        if stream is None:# and self._config.debug_steps:
            stream = PrintSystem(self._config)
        self.new_session(log_dir,stream)
        if metadata:
            seeds = metadata['seeds'] # a list of NodeObjects, see definition in evolution.py 
            seed_types = [seed.type for seed in seeds] # the types of the seeds, can select the flow based on the types later
        # now its from scratch, could be decided by the type of the provided seed
        title,code,explain,summary,autocfg,reviews,ratings,check_results = self.design_fn_scratch(self,query,stream,status_handler)
        return title,code,explain,summary,autocfg,reviews,ratings,check_results

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
        checker = BuildTool(
            tool_type="checker",
        )
        reviewers = {
            'balance': BuildAgent(
                config,
                agent_file=config.reviewer_spec_balance,
                agent_model_type="reviewer_agent"
            ),
            'creative': BuildAgent(
                config,
                agent_file=config.reviewer_spec_creative,
                agent_model_type="reviewer_agent"
            ),
            'rigorous': BuildAgent(
                config,
                agent_file=config.reviewer_spec_rigorous,
                agent_model_type="reviewer_agent"
            )
        }
        debugger = BuildAgent(
            config,
            agent_file=config.debugger_spec,
            agent_model_type="designer_agent"
        )
        
        ### get the model information for context
        cfg = eval(f"{config.gam_config}()")
        block, code = get_context_info(config,templated=cfg.use_template)
        
        return cls(
            designer,
            reviewers,
            checker,
            debugger,
            lib_dir=config.lib_dir,
            block_template=block,
            gam_template=code,
            config=config,
            gam_config=cfg
        )

def BuildSystem(
        config: Optional[ConfigType] = None,
        **kwargs
    ) -> ModelDiscoverySystem:
    """Factory for building an overall system

    :param config: 
        The optional configuration object. 

        
    >>> from model_discovery import BuildSystem 
    >>> system = BuildSystem() 

    """
    from exec_utils import BuildSystem
    kwargs["system_type"] = "model_discovery_system"
    
    # if config and not config.wdir:
    #     wdir = f"{SYSTEM_OUT}/{time.strftime('%Y%m%d_%H%M%S')}"
    #     kwargs["wdir"] = wdir
    # elif "wdir" not in kwargs:
    #     wdir = f"{SYSTEM_OUT}/{time.strftime('%Y%m%d_%H%M%S')}"
    #     kwargs["wdir"] = wdir
        
    return BuildSystem(config,**kwargs)



