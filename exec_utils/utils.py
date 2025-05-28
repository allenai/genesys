import os
import json
import ast
import logging
import importlib as imp

from diskcache import Cache
from shutil import rmtree
from functools import lru_cache
from pathlib import Path
from typing import (
    Dict,
    Any,
    List,
    Type,
    Optional
)
from copy import deepcopy
from diskcache import Cache

from .aliases import PathOrStr,ConfigType
from .param import ModuleParams

__all__ = [
    "create_agent_config",
    "get_group_specs",
    "register_proj",
    "create_cache",
    "make_wdir",
]

util_logger = logging.getLogger('exec_utils.utils')

def create_instance_hash(obj: Any):
    """Creates an unique hash code for this obj (used for 
    caching, serialization, plus other things) 

    :param obj: 
        The target object.
    """
    pass

def create_agent_config(config: ConfigType,aliases: Dict[str,str],**kwargs) -> ConfigType:
    """Reads a agent specification file from a config (if not specified) and 
    creates new configuration. 
    
    :param config: 
        The configuration from which to build the agent config.  
    :raises: 
        ValueError
    """
    agent_config = deepcopy(config)
    if "agent_model_type" in kwargs:
        agent_config.agent_model_type = kwargs["agent_model_type"]
    
    path = kwargs.get("spec","")
    if not path:
        path = agent_config.agent_file
        if not path:
            path = kwargs.get("agent_file",'')
            

    agent_spec = vars(agent_config).get("agent_details")
    agent_type = vars(agent_config).get("agent_type")
    
    ### read from file 
    if not agent_spec:

        if not os.path.isfile(path):
            raise ValueError(f"Cannot find agent spec: {path}")

        
        with open(path) as spec:
            spec_s = spec.read() 

            try:
                agent_spec = json.loads(spec_s)
            except json.JSONDecodeError:
                agent_spec = ast.literal_eval(spec_s)

            if not agent_type: 
                agent_type = agent_spec.get("agent_type")

            agent_config.agent_details = agent_spec 
            
    agent_config.__dict__.update(
        agent_spec.get("_params",{})
    )
    if not agent_type or agent_type not in aliases: 
        raise ValueError(
            f"Invalid agent type: {agent_type}"
        )

    agent_config.agent_type = aliases[agent_type]
    
    return agent_config

def get_group_specs(path: PathOrStr) -> List[str]:
    """Returns the list of group files 

    :raises: ValueError 
    """
    if not os.path.isdir(path):
        raise ValueError(
            f"Cannot find group spec path: {path}"
        )

    group_list = [
        os.path.join(path,a) for a in os.listdir(path) if \
        Path(a).suffix == ".json"
    ]

    return group_list

def register_proj(path: PathOrStr) -> None:
    """Loads a set of python files in `path` that contained 
    new registered classes. 
    
    :param path: 
        The path to either a single custom module or a directory 
        containing multiple files with custom components
    :raises: 
        Exception

    """
    if os.path.isdir(path):
        mod_list = [
            os.path.join(path,a) for a in os.listdir(path) if \
            Path(a).suffix == ".py"
        ]
    elif Path(path).suffix == ".py":
        mod_list =  [path]
    else:
        raise ValueError(
            f'Please specify valid register path: {path}'
        )

    for module in mod_list:
        modified_path = module.replace('/','.').replace('.py','').strip()
        try: 
            example_package = imp.import_module(modified_path)
        except Exception as e:
            raise
        
def create_cache(
        obj: Type['exec_utils.base.ConfigurableUtil'],
        cached_method: str,
        config: Optional[ConfigType],
        *,
        manual_cache: Optional[bool] = False,
    ) -> None:
    """Sets up a caching mechanism for a particular utility. 
    Implements both LRU caching and diskcaching. 
    
    :param obj: 
        The object on which caching should be done.  
    :param cached_method: 
        The model to cache or memoize.  
    :param config: 
        The global configuration, with details of the 
        type caching that should be used. 
    :param manual_cache: 
        Switch to indicate whether the caching is set manually
    :raises: 
        ValueError

    """
    cache_type = config.cache_type
    max_caching = config.max_lru_cache

    if cache_type == "lru":
        setattr(
            obj,
            cached_method,
            lru_cache(maxsize=max_caching)(getattr(obj,cached_method))
        )
        obj.logging.info(f'Set up LRU caching with max_cache={max_caching}')
        return 
        
    elif cache_type == "diskcache":
        ### diskcache must be manually specific in the constructor
        
        if manual_cache:
            cname = obj.identifier

            if not config.cache_loc: 
                config.cache_loc = f"{config.cache_dir}/{config.cache_id}_{cname}"

            cache = Cache(config.cache_loc)
            setattr(
                obj,
                cached_method,
                cache.memoize(ignore=("stream",))(getattr(obj,cached_method))
            )
            obj.logging.info(
                f'Set up disk caching, loc={config.cache_loc}'
            )
            return config.cache_loc
            
        return 
        
    raise ValueError(f'Unknown caching type: {cache_type}')

def make_wdir(
        wdir: PathOrStr,
        override: Optional[bool]=False
    ) -> str:
    """Makes a working directory 

    :param wdir: 
        The working directory 
    :param config: 
        The global config (if available)
    :raises: 
        ValueError 
    """
    
    if os.path.isdir(wdir) and not override:
        raise ValueError(
            f"Directory already exists, use --override"
        )
    elif os.path.isdir(wdir):
        rmtree(wdir)

    os.makedirs(wdir)
    util_logger.info(
        f"Created working directory at path={wdir}"
    )
    return wdir

