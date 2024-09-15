''' System of Review & Design Agents for Model Discovery '''

import exec_utils
import pathlib
import os

import numpy as np
import datetime

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
from .agents.flow.gau_flows import gu_design_mutation,DesignModes,RunningModes
from .agents.search_utils import SuperScholarSearcher

# from .evolution import NodeObject

import model_discovery.utils as U

C = TypeVar("C",bound="ModelDiscoverySystem")

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
    # reviewer_spec_creative: str = exec_utils.ParamField(
    #     default=os.path.abspath(
    #         f'{PROJ_SRC}/../etc/agent_spec/reviewer_creative.json'
    #     ),
    #     metadata={
    #         "help"         : 'Specification of creative reviewer agent',
    #         "exclude_hash" : True,
    #     }
    # )
    # reviewer_spec_balance: str = exec_utils.ParamField(
    #     default=os.path.abspath(
    #         f'{PROJ_SRC}/../etc/agent_spec/reviewer_balance.json'
    #     ),
    #     metadata={
    #         "help"         : 'Specification of balance reviewer agent',
    #         "exclude_hash" : True,
    #     }
    # )
    # reviewer_spec_rigorous: str = exec_utils.ParamField(
    #     default=os.path.abspath(
    #         f'{PROJ_SRC}/../etc/agent_spec/reviewer_rigorous.json'
    #     ),
    #     metadata={
    #         "help"         : 'Specification of rigorous reviewer agent',
    #         "exclude_hash" : True,
    #     }
    # )
    debugger_spec: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/debugger.json'
        ),
        metadata={
            "help"         : 'Specification of debugger agent',
            "exclude_hash" : True,
        }
    )
    claude_spec: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/claude.json'
        ),
        metadata={
            "help"         : 'Specification of claude agent',
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


class NaiveHandler: 
    def __init__(self,message,*args,**kwargs):
        self.message = message

    def __enter__(self):
        print(f'\n[START: {self.message}]\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'\n[FINISH: {self.message}]\n')

class NaiveSpinner:
    def __init__(self,message,*args,**kwargs):
        self.message = message

    def __enter__(self):
        print(f'\n[START: {self.message}]\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'\n[FINISH: {self.message}]\n')

class StatusHandlerWrapper:
    def __init__(self, handler_class, log_function):
        self.handler_class = handler_class
        self.log_function = log_function

    def __call__(self, message, *args, **kwargs):
        class WrappedHandler:
            def __init__(cls, message, log_function, *args, **kwargs):
                cls.message = message
                cls.log_function = log_function
                cls.original_handler = self.handler_class(message, *args, **kwargs)

            def __enter__(cls):
                cls.log_function(cls.message, 'enter')
                return cls.original_handler.__enter__()

            def __exit__(cls, exc_type, exc_val, exc_tb):
                cls.log_function(cls.message, 'exit')
                return cls.original_handler.__exit__(exc_type, exc_val, exc_tb)

        return WrappedHandler(message, self.log_function, *args, **kwargs)
    
class SpinnerWrapper:
    def __init__(self, spinner_class, log_function):
        self.spinner_class = spinner_class
        self.log_function = log_function

    def __call__(self, message, *args, **kwargs):
        class WrappedSpinner:
            def __init__(cls, message, log_function, *args, **kwargs):
                cls.message = message
                cls.log_function = log_function
                cls.original_spinner = self.spinner_class(message, *args, **kwargs)

            def __enter__(cls):
                cls.log_function(cls.message, 'enter')
                return cls.original_spinner.__enter__()

            def __exit__(cls, exc_type, exc_val, exc_tb):
                cls.log_function(cls.message, 'exit')
                return cls.original_spinner.__exit__(exc_type, exc_val, exc_tb)

        return WrappedSpinner(message, self.log_function, *args, **kwargs)
    

class PrintSystem:
    def __init__(self,config):
        self.jupyter = config.jupyter   
        self._isprintsystem = True
        self.status = NaiveHandler
        self.spinner = NaiveSpinner
    
    def write(self,msg,**kwargs):
        print(msg)

    def markdown(self,msg,**kwargs):
        print(msg)


class StreamWrapper:
    def __init__(self,stream,log_file):
        self.stream=stream
        self.log_file=log_file
        self._log=[]
        if U.pexists(self.log_file):
            self._log=eval(U.read_file(self.log_file))
        self.status = StatusHandlerWrapper(stream.status, self.log)
        self.spinner = SpinnerWrapper(stream.spinner, self.log)
    
    def log(self,msg,type):
        self._log.append((datetime.datetime.now(),msg,type))
        U.write_file(self.log_file,str(self._log))
    
    def write(self,msg,**kwargs):
        self.stream.write(msg,**kwargs)
        self.log(msg,'write')

    def markdown(self,msg,**kwargs):
        self.stream.markdown(msg,**kwargs)
        self.log(msg,'markdown')
        

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
    # :param reviewers: 
    #     The reviewer LLM agents. 
    :param checker: 
        The checker tool agent. 
    :param debugger:
        The debugger LLM agent.
    :param block_template: 
        The templated code to be filled in during 
        the block search. 
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
        # reviewers : Dict[str,Type[exec_utils.SimpleLMAgent]],
        checker  : Type[exec_utils.BaseTool], 
        debugger : Type[exec_utils.SimpleLMAgent],
        claude : Type[exec_utils.SimpleLMAgent],
        *,
        config: ConfigType 
    ) -> None:
        """Create a `DiscoverySystem` instance 

        
        :param designer: 
            The designer agent. 
        # :param reviewers: 
        #    The reviewer agents. 
        :param block_template: 
           The autoregressive block that the model has to fill in.
        :param model_implementation: 
           The full model implementation for context 
        :param config: 
           System global configuration. 

        """
        ### modules 
        self.designer = designer
        # self.reviewers = reviewers
        self.checker = checker
        self.debugger = debugger

        self.claude = claude
        
        # self.gam_py = gam_template
        # self.gab_py = block_template
        self._config = config
        
        # to be set later
        self.ptree = None
        self.sss = None

        # Load flows
        self.design_fn_mutation=gu_design_mutation

    def bind_ptree(self,ptree,stream): # need to bind a tree before start working, should be done immediately
        self.ptree = ptree
        self.sss = SuperScholarSearcher(ptree,stream)

    def get_system_info(self):
        system_info = {}
        system_info['agents']={
            'gpt4o':self.designer.config,
            # 'reviewers':{style:agent.config for style,agent in self.reviewers.items()},
            'gpt4o-mini':self.debugger.config,
            'claude3.5_sonnet':self.claude.config
        }

    def new_session(self,design_id,stream):
        self.log_dir = U.pjoin(self.ptree.session_dir(design_id), 'log')
        log_file = U.pjoin(self.log_dir,'stream.log')
        self.sess_state = {} 
        self.dialog = AgentDialogManager(self.log_dir,self.get_system_info(),stream)
        design_stream = StreamWrapper(stream,log_file)
        return design_stream

    def query_system(
        self,
        query='', # user query
        instruct=None,
        seed=None,
        refs=None,
        design_id=None,
        stream: Optional[ModuleType] = None,
        design_cfg = {},
        search_cfg = {},
        mode=DesignModes.MUTATION,
        proposal=None, # implementation only mode, directly implement a proposal, experimental
        **kwargs
    ) -> list:
        """Main function for implementing system calls.

        Proposer Dual + Implementer Trio

        :param stream: 
            The (optional) streamlit module for writing to frontend 
        :param frontend: 
            Switch indiciating whether system is being used 
            with a frontend.
        :param metadata:
            Additional information about the query. Mainly about the seeds.
        """

        assert self.ptree is not None, 'Phylogenetic tree is not initialized, please bind a tree first'
        
        user_input = query

        if stream is None: # and self._config.debug_steps:
            stream = PrintSystem(self._config)

        DEFAULT_AGENTS={
            'DESIGN_PROPOSER':'claude3.5_sonnet',
            'PROPOSAL_REVIEWER':'claude3.5_sonnet',
            'IMPLEMENTATION_PLANNER':'claude3.5_sonnet',
            'IMPLEMENTATION_CODER':'o1_mini',
            'IMPLEMENTATION_OBSERVER':'None',
            'SEARCH_ASSISTANT':'None', # None means no separate search assistant
        }
        DEFAULT_MAX_ATTEMPTS={
            'design_proposal':10,
            'implementation_debug':7,
            'post_refinement':0,
            'max_search_rounds':3,
        }
        DEFAULT_TERMINATION={
            'max_failed_rounds':3,
            'max_total_budget':0, # 0 means no limit
            'max_debug_budget':2,
        }
        DEFAULT_THRESHOLD={
            'proposal_rating':4,
            'implementation_rating':3,
        }
        DEFAULT_SEARCH_SETTINGS={
            'proposal_search':True,
            'proposal_review_search':True,
            'search_for_papers_num':10,
        }
        DEFAULT_MODE=RunningModes.BOTH
        DEFAULT_NUM_SAMPLES={
            'proposal':1,
            'implementation':1,
        }
        
        design_cfg['max_attemps']=U.safe_get_cfg_dict(design_cfg,'max_attemps',DEFAULT_MAX_ATTEMPTS)
        design_cfg['agent_types']=U.safe_get_cfg_dict(design_cfg,'agent_types',DEFAULT_AGENTS)
        design_cfg['termination']=U.safe_get_cfg_dict(design_cfg,'termination',DEFAULT_TERMINATION)
        design_cfg['threshold']=U.safe_get_cfg_dict(design_cfg,'threshold',DEFAULT_THRESHOLD)
        design_cfg['search_settings']=U.safe_get_cfg_dict(design_cfg,'search_settings',DEFAULT_SEARCH_SETTINGS)
        design_cfg['running_mode']=design_cfg.get('running_mode',DEFAULT_MODE)
        design_cfg['num_samples']=U.safe_get_cfg_dict(design_cfg,'num_samples',DEFAULT_NUM_SAMPLES)

        self.sss.reconfig(search_cfg,stream)
        
        try:
            cols=stream.columns(2)
            with cols[0]:
                with stream.expander('Check design configurations'):
                    stream.write(design_cfg)
            with cols[1]:
                with stream.expander('Check search configurations'):
                    search_cfg=self.sss.cfg
                    stream.write(search_cfg)
        except:
            pass

        # 1. create or retrieve a new session
        if design_id is None: # if provided, then its resuming a session
            assert seed, "Must provide seed to create a new design"
            seed_ids = [seed.acronym for seed in seed]
            ref_ids = [ref.acronym for ref in refs] if refs else []
            design_id=self.ptree.new_design(seed_ids, ref_ids, instruct, design_cfg['num_samples'], mode)
        else: # resuming a session
            stream.write(f"Restoring design session: {design_id}")
            mode=self.ptree.session_get(design_id,'mode')
        
        design_stream=self.new_session(design_id,stream)
        if mode==DesignModes.MUTATION:
            self.design_fn_mutation(self,design_stream,design_id,design_cfg,user_input,proposal)
        elif mode==DesignModes.SCRATCH:
            raise NotImplementedError('Scratch mode is unstable, do not use it')
        elif mode==DesignModes.CROSSOVER:
            raise NotImplementedError('Crossover mode is not implemented')
        else:
            raise ValueError(f'Invalid design mode: {mode}')


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
        # reviewers = {
        #     'balance': BuildAgent(
        #         config,
        #         agent_file=config.reviewer_spec_balance,
        #         agent_model_type="reviewer_agent"
        #     ),
        #     'creative': BuildAgent(
        #         config,
        #         agent_file=config.reviewer_spec_creative,
        #         agent_model_type="reviewer_agent"
        #     ),
        #     'rigorous': BuildAgent(
        #         config,
        #         agent_file=config.reviewer_spec_rigorous,
        #         agent_model_type="reviewer_agent"
        #     )
        # }
        debugger = BuildAgent(
            config,
            agent_file=config.debugger_spec,
            agent_model_type="designer_agent"
        )
        claude = BuildAgent(
            config,
            agent_file=config.claude_spec,
            agent_model_type="claude_agent"
        )
        
        return cls(
            designer,
            # reviewers,
            checker,
            debugger,
            claude,
            config=config,
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



