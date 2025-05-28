from abc import abstractmethod
from collections import defaultdict

import functools

from .param import (
    RootParams,
    ModuleParams
)
from typing import (
    Type,
    Dict,
    Type,
    List,
    Optional,
    Any,
    Tuple,
    DefaultDict,
    Callable,
    Union,
)
from .base import UtilBase
from .aliases import ConfigType
from .utils import create_cache

__all__ = [
    "Registry",
]

class Register(UtilBase):
    """General class decorator for registering new class implementations 
    
    Methods
    --------
    factory_update_method(class_to_register): 
        Method for added a class to the registry. 

    """
    
    @abstractmethod
    def factory_update_method(self,class_to_register: Type[UtilBase]) -> None:
        """Method for adding class to registry
        
        :param class_to_register: 
            The newly created class to add.
        """
        raise NotImplementedError
    
    def __call__(self,*args,**kwargs):
        class_to_add = args[0]
        self.factory_update_method(class_to_add)
        return class_to_add

    @classmethod
    def build_model(cls, config: ConfigType,**kwargs) -> Type[UtilBase]:
        """Builds a class within `cls.IMPLEMENTATIONS` from configuration 

        :param config: 
            The global config used for building an object. 
        :raises: 
            ValueError 
        """
        raise NotImplementedError

    @classmethod
    def factory(cls,config: ConfigType,**kwargs):
        """Alias for `build_model`.

        :param config: 
            The global config used for building an object. 
        :raises: 
            ValueError
        """
        return cls.build_model(config,**kwargs)
    

_ClassRegister = Dict[str,UtilBase]
_ResourceName = Tuple[str,str]

class Registry(Register):
    """A registry of classes registered under certain utility types and names.
    
    Examples
    --------
    
    Below would be an example class register into `Registry` (in a notebook):
    
    Starting with 

    >>> import exec_utils
    >>> from exec_utils import Registry 

    Then implement the following class: 
        
    ```python 
        
        @Registry(
            resource_type="my_new_resource",
            name="my_new_class_a",
            cache="my_method",
        )
        class A(exec_utils.UtilResource):
            def my_method(self,method_input):
                return "my output"
            @classmethod 
            def from_config(cls,config,**kwargs): 
                return cls()
    ```

    where we have

        :param resource_type:
            The type of resource in `Registry`. 
        :param name: 
            The internal name of the implementation.  
        :param cache: 
            The name of the method in the class that 
            should be cached. 

    Here's how such a class could then be called with a 
    configuration: 

    >>> from exec_utils import build_config
    >>> config = build_config(ignore_attr=True) 
    >>> config.my_new_resource = "my_new_class_a"
    >>> new_obj = Registry.build_model("my_new_resource",config)  
    
    Where the following method is called:

         :method build_model(resource_type,config): 
              Factory method for building `resource_type` from `config`
              (where `config.resource_type` will specify the particular 
              utility to find in the registry) 
   
    Inside `build_model` calls `from_config` once the registred class has
    been found.

    
    """
    
    ### utility store 
    IMPLEMENTATIONS: DefaultDict[str,_ClassRegister] = defaultdict(dict)
    FUNCTIONS: DefaultDict[str,Callable] = defaultdict()
    
    ## config and argument parser info 
    PARAMS: List[ModuleParams] = []
    PROG = "exec_util"
    DESCRIPTION = "Utilities for agent modeling and execution"

    ## Caching
    CACHE_METHOD: DefaultDict[_ResourceName,str] = defaultdict()
    CACHE_DIR: DefaultDict[str,str] = defaultdict()
        
    def __init__(
            self,
            resource_type: str,
            name: str,
            cache: Optional[str]=None
    ) -> None:
        """Creates `Register` instance. 

        :param resource_type: 
            The type of utility being registered
        :param name: 
            The unique identifier of the resource being registered. 
        :param cache: 
            The name of the method to be cached (if it exists and 
            caching is used)
        """
        functools.update_wrapper(self, resource_type,name)
        self.name = name
        self.resource_type = resource_type
        self.cache = cache 
        
    def factory_update_method(self,class_to_register: Union[Type[UtilBase],Callable]) -> None:
        """Method for adding class to the registry. 
        
        :param class_to_register: 
            The newly created class to add to the registry
        """
        resource_type = self.resource_type 
        identifier = self.name

        if resource_type == "config":
            class_to_register.GROUP = identifier
            self.PARAMS.append(class_to_register)

            return

        elif resource_type == "function" or resource_type == "function_type":
            self.FUNCTIONS[identifier] = class_to_register
            return 
        
        if identifier in self.IMPLEMENTATIONS[resource_type]:

            self.logging.warning(
                f'Resource exists, type={resource_type}, name={identifier}, overriding (also cache setting).'
            )
            
        self.IMPLEMENTATIONS[resource_type][identifier] = class_to_register
        self.CACHE_METHOD[(resource_type,identifier)] = self.cache
        
    @classmethod 
    def build_model(cls,resource_type: str, config: ConfigType,**kwargs) -> Type[UtilBase]:
        """Factory method for building registered obects from configuration. 

        :param resource_type: 
            The type of resource you want to build. 
        :param config: 
            The global configuration. 
        :raises: 
            ValueError

        """
        store = cls.IMPLEMENTATIONS.get(resource_type,{})
        config_key = vars(config).get(resource_type,None)
        resource = store.get(config_key,None)
        
        if not store or config_key is None or resource is None:
            raise ValueError(
                f"Bad resource spec, key or config: type={resource_type},key={config_key}"
            )
        
        ### caching
        cache_method = cls.CACHE_METHOD.get((resource_type,config_key),None)
        do_cache = kwargs.get("do_caching",False)
        if do_cache:
            config.no_caching = False
        
        if do_cache and cache_method:
            kwargs.pop("do_caching",None)
            
        model = resource.from_config(config,**kwargs)

        if not config.no_caching and cache_method is not None:
            
            cache_info = create_cache(
                obj=model,
                cached_method=cache_method,
                config=config,
                manual_cache=do_cache,
            )
        
            ## store info about diskcache
            if cache_info:
                cls.CACHE_DIR[model.identifier] = cache_info
                
        return model

    @classmethod
    def find_function(cls,function_name):
        return cls.FUNCTIONS.get(function_name,None)
        
    @classmethod
    def get_keys(cls) -> Dict[str,List[str]]:
        """Returns the a list of the names of the different utilties and 
        the available registered implementations.


        """
        return {k: list(v.keys()) for k,v in cls.IMPLEMENTATIONS.items()}

    @classmethod
    def build_config(cls,argv: Optional[List[str]],**kwargs) -> ConfigType:
        """Builds a config parser from a registry of configuration groups. 
        
        :param argv: 
            Input values to the configuration parser and object. 
        :returns: 
            A `ConfigType` object with global configuration values
        :raises: 
            ValueError
        """
        ignore_attr = kwargs.get("ignore_attr",False)
        
        params = RootParams(
            root=cls.PARAMS
        )
        registered_keys = cls.get_keys()
                
        config_parser = params.build_parser(
            prog=cls.PROG,
            description=cls.DESCRIPTION,
            global_keys=registered_keys
        )

        config = config_parser.parse_args(argv)
        
        ### update certain manual settings
        for key,value in kwargs.items():
            setattr(config,key,value)

        ### check that all register items have values in config
        for key in registered_keys:
            if not hasattr(config,key) and not ignore_attr and key != "pipeline_type":
                raise ValueError(
                    f"Config missing key, cannot instantiate some utilities: {key}"
                )
            
        return config

