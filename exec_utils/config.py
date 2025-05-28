import os
import logging 

from typing import (
    Optional,
    List,
)
from .aliases import ConfigType
from .param import (
    ModuleParams,
    RootParams,
    ParamField,
)

from .utils import (
    register_proj,
    make_wdir
)
from .register import Registry

__all__ = [
    "build_config",
]

_LOGGING_LEVELS = {
    "info"    : logging.INFO,
    "debug"   : logging.DEBUG,
    "warning" : logging.WARNING,
    "error"   : logging.ERROR,
    "quiet"   : logging.ERROR,
}

@Registry("config","exec_util")
class GlobalConfig(ModuleParams):
    """Global configuration settings

    :param seed: 
        The global seed for the library.  
    :param logging: 
        The global logging level for the library. 
    :param external_proj: 
        The location of external project with custom classes. 
    :param wdir: 
        The working directory.  
        
    """
    
    seed: int = ParamField(
        default=42,
        metadata={"help" : 'Global seed'}
    )
    logging: str = ParamField(
        default='info',
        metadata={
            "help"    : 'The logging level',
            "choices" : list(_LOGGING_LEVELS.keys()),
            "exclude_hash" : True,
        }
    )
    external_proj: str = ParamField(
        default='',
        metadata={
            "help" : 'Path point to external custom modules.',
            "exclude_hash" : True,
        }
    )
    wdir: str = ParamField(
        default='',
        metadata={
            "help" : 'The optional working directory. ',
            "exclude_hash" : True,
        }
    )
    override: bool = ParamField(
        default=False,
        metadata={
            "help" : 'Remove existing directories when they exist',
            "exclude_hash" : True,
        }
    )
    log_to_file: bool = ParamField(
        default=False,
        metadata={
            "help" : 'Print log to external file',
            "exclude_hash" : True,
        }
    )
    
def _set_logging(level: str, config: Optional[ConfigType] = None) -> None:
    """Sets the logging level 

    :param level: 
        A text description of the target logging level.  
    :param config: 
        An (optional) global configuration with additional details 
        about logging. 
    """
    level = _LOGGING_LEVELS.get(
        level,
        logging.INFO
    )
    log_file = None 
    if config and config.wdir and config.log_to_file:
        log_file = os.path.join(
            config.wdir,
            "pipeline.log"
        )
        
    logging.basicConfig(
        filename=log_file,
        level=level,
    )

def build_config(
        argv: Optional[List[str]]=[],
        **kwargs
    ) -> ConfigType:
    """Builds a library level configuration object. 

    :param argv: 
        Input to the configuration parser (optional).  
    
    """
    external_proj = kwargs.get("external_proj","")
    if "--external_proj" in argv and not external_proj:
        loc = argv.index("--external_proj")+1
        external_proj = argv[loc]

    if external_proj: 
        register_proj(external_proj)

        
    config = Registry.build_config(argv,**kwargs)
    if config.wdir:
        make_wdir(config.wdir,config.override)

    ### set the logging 
    _set_logging(config.logging,config)
    
    return config

