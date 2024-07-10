import exec_utils
import pathlib
import os
import time
import tempfile

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
from .prompts import (
    DESIGNER_PROMPT,
    REVIEWER_PROMPT,
    GAB_ERROR
)

C = TypeVar("C",bound="ModelDiscoverySystem")

from .model.configs.gam_config import (
    GAMConfig_10M,
    GAMConfig
)

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
            f'{PROJ_SRC}/gab_template.py'
        ),
        #default=GB.__file__,
        metadata={
            "help"         : 'Location of block for prompting ',
        }
    )
    gam_template: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/gam_prompt.py'
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
    gam_config: str = exec_utils.ParamField(
        default='GAMConfig_10M',
        metadata={
            "help"         : 'Debug the steps of the system',
            "exclude_hash" : True,
        }
    )
    
def get_context_info(config) -> Tuple[str,str]:
    """Grabs the block and model implementation details for the prompt 

    :param config: 
        The global configuration 
    :raises ValueError : 
    """
    if not os.path.isfile(config.block_template):
        raise ValueError(f'Cannot find the block template: {config.block_template}')
    if not os.path.isfile(config.gam_template):
        raise ValueError(f'Cannot find the code context')
    if not os.path.isfile(config.gam_implementation):
        raise ValueError(f'Cannot find the gam implementation')
    block = open(config.block_template).read()
    code = open(config.gam_template).read()
    implementation = open(config.gam_implementation).read()
    
    return (block,code,implementation)

class EmptyHandler: 
    def __init__(self,*args,**kwargs):
        pass 
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    
@exec_utils.Registry(
    resource_type="system_type",
    name="model_discovery_system",
    #cache="query_system",
)
class ModelDiscoverySystem(exec_utils.System):
    """Overall system for discovery

    """

    def __init__(
        self,
        designer : Type[exec_utils.SimpleLMAgent],
        reviewer : Type[exec_utils.SimpleLMAgent],
        checker  : Type[exec_utils.BaseTool], 
        *,
        block_template: str,
        model_implementation: str,
        gam_template: str,
        gam_config,
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
        self.gam_implementation = model_implementation
        self.gab_py = block_template
        self._config = config
        self._cfg = gam_config

        ###
        self._queries = [] 

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

        
        problem_history = []
        query = DESIGNER_PROMPT.format(
            gam_py=self.gam_py,
            gab_py=self.gab_py,
            config=self._cfg.print_config(), #<--- need to parameterize 
            instruct=query,
        )
        source = 'user'
        self._queries.append(query)
        
        for attempt in range(self._config.max_design_attempts):

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
                
                    if self._config.debug_steps:
                        print(f"DESIGNER CODE PROPOSED #={attempt}:\n===================\n {designer_out['code']}")
                    if code and stream:
                        stream.write('Model authored code block...')
                        stream.markdown(f'```python\n{code}```')
                        
                    if "# gab.py" not in code: raise
                except Exception as e:
                    query = GAB_ERROR
                    source = 'user'
                    continue

                checkpass,check_report = self.checker.check(self._cfg,code)
                if self._config.debug_steps:
                    print(f"CODE CHECKER.....\n-----------------\npass={checkpass}\n{check_report}")
                if stream:
                    stream.write(
                        f"""<details><summary>code check output</summary>{check_report}</details>""",
                        unsafe_allow_html=True
                    )
                    
                if not checkpass:
                    query = f"The designed model didn't pass, you need to try again. Here is the report:\n{check_report}"
                    source = 'user'
                    self.checker.reset()
                    continue 
            
                break 

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
        block, code, implementation = get_context_info(config)
        cfg = eval(f"{config.gam_config}()")
        
        return cls(
            designer,
            reviewer,
            checker,
            block_template=block,
            model_implementation=implementation,
            gam_template=code,
            config=config,
            gam_config=cfg
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
    kwargs["system_type"] = "model_discovery_system"

    if config and not config.wdir:
        wdir = f"{SYSTEM_OUT}/{time.strftime('%Y%m%d_%H%M%S')}"
        kwargs["wdir"] = wdir
    elif "wdir" not in kwargs:
        wdir = f"{SYSTEM_OUT}/{time.strftime('%Y%m%d_%H%M%S')}"
        kwargs["wdir"] = wdir
        
    return BuildSystem(config,**kwargs)
