from typing import (
    Optional,
    Type,
    Union,
    Tuple,
    Any,
    Dict
)

from .aliases import ConfigType
from .register import Registry
from .base import UtilBase
from .config import build_config

__all__ = [
    "BuildAgent",
    "BuildModel",
    "BuildGroup",
    "BuildSystem",
    "BuildTool",
]

_Component = Type[UtilBase]
  
def _update_caching(config: ConfigType,kwargs) -> None:
  """Updates the configuration to take into account global caching 
  preferences. 
  
  :param config:  
      The optional configuration provided to the factory method.
  
  """
  if config.cache_type == "diskcache" and not config.no_caching \
    and "do_caching" not in kwargs:
    kwargs["do_caching"] = True
    

def _check_config(config: Union[ConfigType,None],kwargs) -> Tuple[ConfigType,Dict[str,Any]]:
    """Checks where the config exists and it not build its from kwargs

    :param config:  
        The optional configuration provided to the factory method. 
    :param kwargs: 
        The additional keyword arguments provided to the factory 
        (becomes essential when the configuration is empty)

    """
    updated_kwargs = kwargs
    if config is None:
        config = build_config(**kwargs)
        updated_kwargs = {}
        if "do_caching" in kwargs:
            updated_kwargs["do_caching"] = kwargs["do_caching"]
            
    _update_caching(config,updated_kwargs)
    
    return (config,updated_kwargs)
        
    
def BuildAgent(
        config: Optional[ConfigType] = None,
        **kwargs
    ) -> _Component:
    """Factory method for building groups. It will build
    the configuration on the fly if not provided.

    :param config: 
        The global configuration used to build the object.
    :returns: 
        A agent object build from configuration. 
    
    """
    config,updated_kwargs = _check_config(config,kwargs)    
    return Registry.build_model("agent",config,**updated_kwargs)

def BuildModel(
        config: Optional[ConfigType] = None,
        **kwargs
    ) -> _Component:
    """Factory method for building groups. It will build
    the configuration on the fly if not provided.

    :param config: 
        The global configuration used to build the object.
    :returns: 
        A modle object build from configuration.

    """
    config,updated_kwargs = _check_config(config,kwargs) 
    return Registry.build_model("model_type",config,**updated_kwargs)

def BuildGroup(
        config: Optional[ConfigType] = None,
        **kwargs
    ) -> _Component:
    """Factory method for building groups. It will build
    the configuration on the fly if not provided. 

    :param config: 
        The global configuration used to build the object.
    :returns: 
        A group object build from configuration. 
 
    """
    config,updated_kwargs = _check_config(config,kwargs)
    return Registry.build_model("group_type",config,**updated_kwargs)

def BuildSystem(
        config: Optional[ConfigType] = None,
        **kwargs
    ) -> _Component:
    """Factory method for building groups. It will build
    the configuration on the fly if not provided.

    :param config: 
        The global configuration used to build the object.
    :returns: 
        A system object build from configuration. 

    """
    config,updated_kwargs = _check_config(config,kwargs) 
    return Registry.build_model("system_type",config,**updated_kwargs)

def BuildTool(
        config: Optional[ConfigType] = None,
        **kwargs
    ) -> _Component:
    """Factory method for building groups. It will build
    the configuration on the fly if not provided.

    :param config: 
        The global configuration used to build the object.
    :returns: 
        A tool object build from configuration

    """
    config,updated_kwargs = _check_config(config,kwargs) 
    return Registry.build_model("tool_type",config,**updated_kwargs)
