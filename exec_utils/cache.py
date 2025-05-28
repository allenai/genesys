import os

from functools import lru_cache
from typing import (
    Any,
    Optional,
    Type
)
from .register import Registry
from .aliases import ConfigType
from .param import (
    ModuleParams,
    RootParams,
    ParamField,
)

__all__ = [
    "setup_cache"
]

def setup_cache(
        obj: Type['exec_utils.base.ConfigurableUtil'],
        cached_method: str,
        config: Optional[ConfigType],
    ) -> None:
    """Sets up the caching for a utility or arbitrary object. 
    
    :param obj: 
        The object on which caching will be applied. 
    :param config: 
        The configuration with settings for caching. 
    :param cached_method: 
        A string representation of the name of the 
        method in the object that you want to cache. 

    """
    no_caching = False
    max_caching = 1000
    
    if config and hasattr(config,"no_caching"):
        no_caching = config.no_caching
    if config and hasattr(config,"max_lru_cache"):
        max_caching = config.max_lru_cache

    if not no_caching: 
        setattr(
            obj,
            cached_method,
            lru_cache(maxsize=max_caching)(getattr(obj,cached_method))
        )

        obj.logging.debug(
            f'Set up logging with max_cache={max_caching}'
        )
                
    
@Registry("config","exec_util.cache")
class Param(ModuleParams):
    """Global configuration settings for caching. 

    :param no_caching: 
        Turns off the caching globally. 
    :param cache_type: 
        The type of caching that you want to use. 
        Implements LRU caching and diskcache. 
    :param cache_dir: 
        The location of the caching directory. 
    :param cache_id: 
        The name of the cache file. 

    """
    
    no_caching: bool = ParamField(
        default=False,
        metadata={"help": 'Removing caching from all agents'}
    )
    max_lru_cache: int = ParamField(
        default=1000,
        metadata={
            "help" : 'The maximum size of the LRU cache',
            "exclude_hash" : True,
        }
    )
    cache_type: str = ParamField(
        default='lru',
        metadata={"help": 'The type of caching to use'}
    )
    cache_dir: str = ParamField(
        default=os.path.expanduser('~/.cache'),
        metadata={"help": 'The location of the caching'}
    )
    cache_loc: str = ParamField(
        default='',
        metadata={"help": 'The location of a  particular cache'}
    )
    cache_id: str = ParamField(
        default='42',
        metadata={"help": 'The identifier for caching'}
    )
