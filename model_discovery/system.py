''' System of Review & Design Agents for Model Discovery '''

import exec_utils
import pathlib
import os
import json
import time
import tempfile

#from IPython.display import display, Markdown, Latex
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

from exec_utils.aliases import ConfigType
from exec_utils import (
    BuildAgent,
    BuildTool
)
from .agents import *
from .agents.prompts.prompts import (
    DESIGNER_PROMPT,
    REVIEWER_PROMPT,
    GAB_ERROR
)

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
        default=5,
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
    reviewer_spec: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/reviewer.json'
        ),
        metadata={
            "help"         : 'Specification of reviewer agent',
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
        default=False,
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
        default='GAMConfig_10M',
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


def get_context_info(config) -> Tuple[str,str]:
    """Grabs the block and model implementation details for the prompt 

    :param config: 
        The global configuration 
    :raises: ValueError 

    """
    if not os.path.isfile(config.block_template):
        raise ValueError(f'Cannot find the block template: {config.block_template}')
    if not os.path.isfile(config.gam_template):
        raise ValueError(f'Cannot find the code context')
    block = open(config.block_template).read()
    code = open(config.gam_template).read()
    
    return (block,code) 

class EmptyHandler: 
    def __init__(self,*args,**kwargs):
        pass 
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class PrintSystem:
    def __init__(self,config):
        self.jupyter = config.jupyter        
    
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
    :param reviewer: 
        The reviewer LLM agent. 
    :param checker: 
        The checker tool agent. 
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
        reviewer : Type[exec_utils.SimpleLMAgent],
        checker  : Type[exec_utils.BaseTool], 
        *,
        block_template: str,
        gam_template: str,
        gam_config: Type[GAMConfig],
        config: ConfigType 
    ) -> None:
        """Create a `DiscoverySystem` instance 

        
        :param designer: 
            The designer agent. 
        :param reviewer: 
           The reviewer agent. 
        :param block_template: 
           The autoregressive block that the model has to fill in.
        :param model_implementation: 
           The full model implementation for context 
        :param config: 
           System global configuration. 

        """
        ### modules 
        self.designer = designer
        self.reviewer = reviewer
        self.checker = checker
        
        self.gam_py = gam_template
        self.gab_py = block_template
        self._config = config
        self._cfg = gam_config

        ###
        self._queries = [] 

    def set_config(self,cfg:GAMConfig): # reset the gam config of the system
        self._cfg = cfg

    def query_system(
        self,
        query: Optional[str] = '',
        stream: Optional[ModuleType] = None,
        frontend: Optional[bool] = False,
        status: Optional[bool] = True,
        **kwargs
    ) -> list:
        """Main function for implementing system calls.

        :param query: 
            The query to the overall system. 
        :param stream: 
            The (optional) streamlit module for writing to frontend 
        :param frontend: 
            Switch indiciating whether system is being used 
            with a frontend.
        
        """
        status_handler = stream.status if stream and status else EmptyHandler
        if stream is None and self._config.debug_steps:
            stream = PrintSystem(self._config)
            
        problem_history = []

        query = f"{query}\nPlease only write raw Python code and nothing more, no special formatting or extra text."
        query = DESIGNER_PROMPT.format(
            gam_py=self.gam_py,
            gab_py=self.gab_py,
            config=self._cfg.print_config(), #<--- need to parameterize 
            instruct=query,
        )
        source = 'user'
        found_design = False
        self._queries.append(query)
        
        for attempt in range(self._config.max_design_attempts):
            self.logging.info(f'Attempting design, attempt={attempt}')
            
            design_name = f"{self._config.run_name}_{attempt}_{len(self._queries)}"

            with status_handler(f"Attempt {attempt+1}"): 
            
                designer_out = self.designer(
                    query,
                    source=source,
                    manual_history=problem_history, #<--- dialogue history and state
                )
                problem_history.append((query,source))
            
                try:
                    code = designer_out.get("code",None)
                    problem_history.append((str(code),"assistant"))
                
                    if code and stream:
                        stream.write('Model authored code block...')
                        stream.markdown(f'```python\n{code}```')
                        
                    #if "# gab.py" not in code: raise
                    assert "# gab.py" in code 

                # except Exception as e: # <-- should be checker's job?
                #     query = f"An error was encountered when running the code, error={e}. Please try again."
                #     source = 'user'
                #     continue
                except AssertionError:
                    source = "user"
                    query  = GAB_ERROR
                    continue 
                
                checkpass,check_report = self.checker.check(self._cfg,code,design_name)
                # mp.set_start_method('spawn')
                # queue = mp.Queue()
                # testing_process = mp.Process(target=self.run_check, args=(self, code, design_name, queue))
                # testing_process.start()
                # testing_process.join()
                # checkpass, check_report = queue.get()
                            
                if stream:
                    stream.write(
                        f"""<details><summary>code check</summary>{check_report}</details>""",
                        unsafe_allow_html=True
                    )
                
                ### FOR DEBUGGING, REMEMBER TO UNCOMMENT!
                if not checkpass:
                    query = f"The designed model didn't pass, you need to try again. Here is the report:\n{check_report}. Please fix"
                    source = 'user'
                    self.checker.reset()
                    continue 

                problem_history.append(("The designed model passed, now scoring","user"))

                found_design = True
                break

        #### now have the agent defend the design
        if found_design: 
            report_query = (
                "The designed model passed the tests, now please generate a text report explaining and justifying your design."
                " Generate a creative name of your design as the title of your report in the first line of your response."
                " Do not include abbreviations or acronyms of your design in the title. You can use them in the body of the report."
            )
            with status_handler(f"Querying agent for report..."):
                self.logging.info('Now trying to compile self report...')
                self_report = self.designer(
                    report_query,
                    source='user',
                    manual_history=problem_history, 
                )
                if stream:
                    stream.markdown(self_report["code"]) #<-- change

            ### TODO: query the review agent
            
        #### now use the 


        ### TODO: return the design artifacts: name, code, report, explanation, etc.
        if not found_design:
            return None,None,None
        title=self_report['code'].split('\n')[0].replace('#','').strip()
        return title,code,self_report['code']

    def run_check(self,code,design_name,queue):
        checkpass,check_report = self.checker.check(self._cfg,code,design_name)
        queue.put((checkpass, check_report))

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
        reviewer = BuildAgent(
            config,
            agent_file=config.reviewer_spec,
            agent_model_type="reviewer_agent"
        )
        
        
        ### get the model information for context
        block, code = get_context_info(config)
        cfg = eval(f"{config.gam_config}()")
        
        return cls(
            designer,
            reviewer,
            checker,
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

