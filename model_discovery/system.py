''' System of Review & Design Agents for Model Discovery '''

import pathlib
import os

import numpy as np
import datetime
import traceback

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
import uuid
import time

from exec_utils.aliases import ConfigType
from exec_utils.factory import BuildAgent
from exec_utils.models.model_agent import SimpleLMAgent
from exec_utils.tools.tool import BaseTool
from exec_utils.system import System as ExecSystem
from exec_utils.register import Registry as ExecRegistry
from exec_utils.param import ModuleParams as ExecModuleParams
from exec_utils.param import ParamField as ExecParamField
from .agents.roles import *
from .agents.flow.alang import AgentDialogManager

from .agents.flow.gau_flows import gu_design,DesignModes,RunningModes,\
    AGENT_TYPES,AGENT_OPTIONS,DEFAULT_AGENT_WEIGHTS,DESIGN_TERMINAL_STATES,\
        DESIGN_ACTIVE_STATES,DESIGN_ZOMBIE_THRESHOLD
from .agents.search_utils import SuperScholarSearcher

import model_discovery.utils as U
from exec_utils.config import build_config
from exec_utils.utils import create_agent_config


C = TypeVar("C",bound="ModelDiscoverySystem")

__all__ = [
    "ModelDiscoverySystem",
    "BuildSystem",
]

PROJ_SRC = os.path.abspath(os.path.dirname(__file__))
SYSTEM_OUT = os.path.abspath(f"{PROJ_SRC}/../_runs")





@ExecRegistry("config","discovery_system")
class CustomParams(ExecModuleParams):
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
    max_design_attempts: int = ExecParamField(
        default=10,
        metadata={
            "help"         : 'The maximum number of designer queries',
            "exclude_hash" : True,
        }
    )
    max_design_refines: int = ExecParamField(
        default=10,
        metadata={
            "help"         : 'The maximum number of designer refinements',
            "exclude_hash" : True,
        }
    )
    reviewer_threshold: int = ExecParamField(
        default=3,
        metadata={
            "help"         : 'The threshold for accepting a design',
            "exclude_hash" : True,
        }
    )
    ### agent profiles
    designer_spec: str = ExecParamField(
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
    debugger_spec: str = ExecParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/debugger.json'
        ),
        metadata={
            "help"         : 'Specification of debugger agent',
            "exclude_hash" : True,
        }
    )
    claude_spec: str = ExecParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/claude.json'
        ),
        metadata={
            "help"         : 'Specification of claude agent',
            "exclude_hash" : True,
        }
    )
    ### code information
    block_template: str = ExecParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/agents/prompts/gab_template.py'
        ),
        metadata={
            "help"         : 'Location of block for prompting ',
        }
    )
    gam_template: str = ExecParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/agents/prompts/gam_prompt.py'
        ),
        metadata={
            "help" : 'Location of code prompting ',
        }
    )
    gam_implementation: str = ExecParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/model/gam.py'
        ),
        metadata={
            "help" : 'Location of code prompting ',
        }
    )
    ### debugging
    debug_steps: bool = ExecParamField(
        default=True,
        metadata={
            "help"         : 'Debug the steps of the system',
            "exclude_hash" : True,
        }
    )
    jupyter: bool = ExecParamField(
        default=False,
        metadata={
            "help"         : 'Inside of jupyter',
            "exclude_hash" : True,
        }
    )
    #### name of the system run 
    run_name: str = ExecParamField(
        default='demo',
        metadata={
            "help"         : 'The name of the run',
            "exclude_hash" : True,
        }
    )
    from_json: str = ExecParamField(
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

class SilentSpinner(NaiveSpinner):
    def __init__(self,message,*args,**kwargs):
        super().__init__(message,*args,**kwargs)
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class SilentHandler(NaiveHandler):
    def __init__(self,message,*args,**kwargs):
        super().__init__(message,*args,**kwargs)
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

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
                cls.log_function(cls.message, 'enterspinner')
                return cls.original_spinner.__enter__()

            def __exit__(cls, exc_type, exc_val, exc_tb):
                cls.log_function(cls.message, 'exit')
                return cls.original_spinner.__exit__(exc_type, exc_val, exc_tb)

        return WrappedSpinner(message, self.log_function, *args, **kwargs)
    

class PrintSystem:
    def __init__(self,config,silent=False):
        self.jupyter = config.jupyter   
        self._isprintsystem = True
        self.silent=silent
        self.status = NaiveHandler if not silent else SilentHandler
        self.spinner = NaiveSpinner if not silent else SilentSpinner
    
    def write(self,msg,**kwargs):
        if not self.silent:
            print(msg)

    def markdown(self,msg,**kwargs):
        if not self.silent:
            print(msg)
    
    def spinner(self,msg,**kwargs):
        if not self.silent:
            print(msg)

    def code(self,code,**kwargs):
        if not self.silent:
            print(code)

    def balloons(self,**kwargs):
        if not self.silent:
            print('🎈🎈🎈🎈🎈')

    def snow(self,**kwargs):
        if not self.silent:
            print('❄️❄️❄️❄️❄️')

def safe_backup(file):
    count=1
    content=U.read_file(file)
    while U.pexists(file+f'.backup{count}'):
        count+=1
    U.write_file(file+f'.backup{count}',content)

class StreamWrapper:
    def __init__(self,stream,log_file):
        self.stream=stream
        self.log_file=log_file
        self._log=[]
        self.status = StatusHandlerWrapper(stream.status, self.log)
        self.spinner = SpinnerWrapper(stream.spinner, self.log)
    
    def log(self,msg,type):
        _msg=str(msg).replace('\n','/NEWLINE/').replace('\t','/TAB/')
        self._log.append((datetime.datetime.now(),_msg,type))
        line=str(self._log[-1])+'\n'
        U.append_file(self.log_file,line)
    
    def write(self,msg,**kwargs):
        self.stream.write(msg,**kwargs)
        self.log(msg,'write')

    def markdown(self,msg,**kwargs):
        self.stream.markdown(msg,**kwargs)
        self.log(msg,'markdown')

    def balloons(self,**kwargs):
        self.stream.balloons(**kwargs)
        self.log('balloons','balloons')

    def snow(self,**kwargs):
        self.stream.snow(**kwargs)
        self.log('snow','snow')
    
    def code(self,code,**kwargs):
        self.stream.code(code,**kwargs)
        self.log(code,'code')

        

DEFAULT_AGENTS={
    'DESIGN_PROPOSER':'o1_mini',
    'PROPOSAL_REVIEWER':'o1_mini',
    'IMPLEMENTATION_PLANNER':'o1_mini',
    'IMPLEMENTATION_CODER':'o1_mini',
    'IMPLEMENTATION_OBSERVER':'o1_mini',
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
DEFAULT_MODE=RunningModes.BOTH.value
DEFAULT_NUM_SAMPLES={
    'proposal':1,
    'implementation':1,
    'rerank_method':'rating',
}
DEFAULT_UNITTEST_PASS_REQUIRED=False
DEFAULT_CROSSOVER_NO_REF=True
DEFAULT_MUTATION_NO_TREE=True
DEFAULT_SCRATCH_NO_TREE=False
DEFAULT_USE_UNLIMITED_PROMPT=False

@ExecRegistry(
    resource_type="system_type",
    name="model_discovery_system",
    #cache="query_system",
)
class ModelDiscoverySystem(ExecSystem):
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
        designer : Type[SimpleLMAgent],
        checker  : Type[BaseTool], 
        debugger : Type[SimpleLMAgent],
        claude : Type[SimpleLMAgent],
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
        self.design_fn=gu_design

    def bind_ptree(self,ptree,stream): # need to bind a tree before start working, should be done immediately
        self.ptree = ptree
        self.sss = SuperScholarSearcher(ptree,stream)

    def get_system_info(self):
        system_info = {}
        system_info['agents']={
            'gpt4o':self.designer.config,
            'gpt4o-mini':self.debugger.config,
            'claude3.5_sonnet':self.claude.config
        }

    def new_session(self,sess_id,stream,log_collection=None):
        self.log_dir = U.pjoin(self.ptree.session_dir(sess_id), 'log')
        U.mkdir(self.log_dir)
        log_file = U.pjoin(self.log_dir,f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        self.sess_state = {} 
        self.dialog = AgentDialogManager(self.log_dir,self.get_system_info(),stream)
        design_stream = StreamWrapper(stream,log_file)
        log_fn = None
        if log_collection:
            latest_log = str(time.time())
            log_ref = log_collection.document(sess_id).collection('logs').document(latest_log)
            # index_ref = log_collection.document('index')
            index_ref,_ = self.ptree.CM.get_design_sessions_index()
            def log_fn(msg,status='RUNNING'):
                timestamp = str(time.time())
                try:
                    log_ref.set({
                        timestamp:{
                            'status':status,
                            'message':msg
                        }
                    },merge=True)
                except Exception as e:
                    log = log_ref.get().to_dict()
                    ind = 1
                    while True:
                        backup_ref = log_collection.document(sess_id).collection('logs').document(f'{latest_log}_{ind}')
                        if not backup_ref.get().exists:
                            break
                        ind += 1
                    while True:
                        try:
                            backup_ref.set(log)
                            break
                        except Exception as e:
                            log.pop(list(log.keys())[0])
                    log_ref.set({
                        timestamp:{
                            'status':status,
                            'message':msg
                        }
                    }) # restart the log

                # if status in DESIGN_TERMINAL_STATES+['BEGIN']: # only update the index at begining and ends
                index_ref.set({
                    sess_id:{
                        'timestamp':timestamp,
                        'status':status,
                        'latest_log':latest_log
                    }
                },merge=True)
        return design_stream,log_fn

    def query_system(
        self,
        query='', # user query
        instruct=None,
        seeds=None,
        refs=None,
        sess_id=None,
        stream: Optional[ModuleType] = None,
        design_cfg = {},
        search_cfg = {},
        proposal=None, # implementation only mode, directly implement a proposal, experimental
        silent=False,
        cpu_only=False,
        log_collection=None,
        demo_mode=False,
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
            stream = PrintSystem(self._config,silent=silent)
        
        self.checker.silent=silent
        
        design_cfg['max_attemps']=U.safe_get_cfg_dict(design_cfg,'max_attemps',DEFAULT_MAX_ATTEMPTS)
        design_cfg['agent_types']=U.safe_get_cfg_dict(design_cfg,'agent_types',DEFAULT_AGENTS)
        design_cfg['termination']=U.safe_get_cfg_dict(design_cfg,'termination',DEFAULT_TERMINATION)
        design_cfg['threshold']=U.safe_get_cfg_dict(design_cfg,'threshold',DEFAULT_THRESHOLD)
        design_cfg['search_settings']=U.safe_get_cfg_dict(design_cfg,'search_settings',DEFAULT_SEARCH_SETTINGS)
        design_cfg['running_mode']=RunningModes(design_cfg.get('running_mode',DEFAULT_MODE))
        design_cfg['num_samples']=U.safe_get_cfg_dict(design_cfg,'num_samples',DEFAULT_NUM_SAMPLES)
        design_cfg['unittest_pass_required']=design_cfg.get('unittest_pass_required',DEFAULT_UNITTEST_PASS_REQUIRED)
        design_cfg['agent_weights']=U.safe_get_cfg_dict(design_cfg,'agent_weights',DEFAULT_AGENT_WEIGHTS)
        design_cfg['crossover_no_ref'] = design_cfg.get('crossover_no_ref',DEFAULT_CROSSOVER_NO_REF)
        design_cfg['mutation_no_tree'] = design_cfg.get('mutation_no_tree',DEFAULT_MUTATION_NO_TREE)
        design_cfg['scratch_no_tree'] = design_cfg.get('scratch_no_tree',DEFAULT_SCRATCH_NO_TREE)
        design_cfg['use_unlimited_prompt'] = design_cfg.get('use_unlimited_prompt',DEFAULT_USE_UNLIMITED_PROMPT)

        self.sss.reconfig(search_cfg,stream)
        self.sss._refresh_db()
        
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
        if sess_id is None: # if provided, then its resuming a session
            assert seeds is not None, "Must provide seed to create a new design, empty for scratch mode"
            seed_ids = [seed.acronym for seed in seeds]
            if design_cfg['crossover_no_ref'] and len(seeds)>1 and refs:
                stream.write('Crossover no ref is set to True, ignore all references')
                refs=None
            ref_ids = [ref.acronym for ref in refs] if refs else []
            sess_id=self.ptree.new_design(seed_ids, ref_ids, instruct, design_cfg['num_samples'], demo_mode=demo_mode)
            stream.write(f"Starting new design session: {sess_id}")
        else: # resuming a session or creating a new session with a given id
            sessdata = self.ptree.get_design_session(sess_id)
            if sessdata is None:
                seed_ids = [seed.acronym for seed in seeds]
                if design_cfg['crossover_no_ref'] and len(seeds)>1 and refs:
                    stream.write('Crossover no ref is set to True, ignore all references')
                    refs=None
                ref_ids = [ref.acronym for ref in refs] if refs else []
                self.ptree.new_design(seed_ids, ref_ids, instruct, design_cfg['num_samples'], sess_id, demo_mode=demo_mode)
                stream.write(f"Starting new design session: {sess_id}")
            else:
                stream.write(f"Restoring design session: {sess_id}")
        
        design_stream,log_fn=self.new_session(sess_id,stream,log_collection)
        
        try:
            self.design_fn(self,design_stream,sess_id,design_cfg,user_input,proposal,cpu_only=cpu_only,log_fn=log_fn)
        except Exception as e:
            trace = traceback.format_exc()
            error_msg = f'Error in design session {sess_id}:\n\n{e}\n\n{trace}'
            self.log_error(error_msg)
            print(error_msg)


    def log_error(self,error_msg):
        error_log_dir = U.pjoin(self.ptree.db_dir,'error_logs')
        U.mkdir(error_log_dir)
        error_log_file = U.pjoin(error_log_dir,f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        U.write_file(error_log_file,error_msg)
        

    @classmethod
    def from_config(cls: Type[C],config: ConfigType,demo_mode: bool = False,**kwargs) -> C:
        """The main method for instantiating system instances from configuration. 

        :param config: 
            The global configuration used to create instance. 
        :returns: 
            A `DiscoverySystem` instance from configuration. 

        """
        ### creates designer and reviewer agents

        if demo_mode:
            print(f'Building agents in demo mode')

        print(f'Building designer agent')
        designer = BuildAgent(
            config,
            agent_file=config.designer_spec,
            agent_model_type="designer_agent"
        )
        print(f'Building checker')
        checker = Checker(demo=demo_mode)
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
        print(f'Building debugger agent')
        debugger = BuildAgent(
            config,
            agent_file=config.debugger_spec,
            agent_model_type="designer_agent"
        )
        print(f'Building claude agent')
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
        demo_mode: bool = False,
        **kwargs
    ) -> ModelDiscoverySystem:
    """Factory for building an overall system

    :param config: 
        The optional configuration object. 

        
    >>> from model_discovery import BuildSystem 
    >>> system = BuildSystem() 

    """
    # from exec_utils import BuildSystem
    kwargs["system_type"] = "model_discovery_system"
    
    # if config and not config.wdir:
    #     wdir = f"{SYSTEM_OUT}/{time.strftime('%Y%m%d_%H%M%S')}"
    #     kwargs["wdir"] = wdir
    # elif "wdir" not in kwargs:
    #     wdir = f"{SYSTEM_OUT}/{time.strftime('%Y%m%d_%H%M%S')}"
    #     kwargs["wdir"] = wdir
    if config is None:
        config = build_config(**kwargs)
    print(f'Starting to build agent system')
    agent_system = ModelDiscoverySystem.from_config(config,demo_mode=demo_mode,**kwargs)
    return agent_system
    # return BuildSystem(config,**kwargs)



