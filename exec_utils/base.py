import logging
import hashlib
import json

from abc import ABC
from collections import OrderedDict
from typing import Mapping, Any
from pydantic import (
    BaseModel,
    field_validator,
    Field,
    ConfigDict,
)

__all__ = [
    "ConfigurableUtil",
    "UtilResource",
]

class UtilBase(ABC):
    """Baseclass for utility classes"""

    @property
    def logging(self) -> logging.RootLogger:
        """Returns an instance logger 
        
        :rtype: logging.RootLogger
        """
        return logging.getLogger(
            f"{__name__}.{type(self).__name__}"
        )

    @property 
    def identifier(self) -> str:
        """Produces a unique identifier utility classes based 
        on the object's underlying configuration. 

        Used for caching and other 
        purposes. 
        
        """
        from exec_utils.param import ModuleParams

        if hasattr(self,"config"):
            config = self.config
        elif hasattr(self,"_config"):
            config = self._config
        else:
            return ''

        config_rep = json.dumps(
            {
                k: v for k,v in config.__dict__.items() \
                if v and k not in ModuleParams.EXCLUDED_KEYS
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(',',':')
            )

        rep = hashlib.md5(config_rep.encode('utf-8')).hexdigest()[:8]
        rep = f"{self.__class__.__name__}_{rep}"

        for attr,value in sorted(self.__dict__.items()):
            if isinstance(value,UtilBase) and value.identifier:
                rep = f"{rep}_{value.identifier}"
            if isinstance(value,dict):
                for dattr,dvalue in sorted(value.items()):
                    if isinstance(dvalue,UtilBase) and dvalue.identifier:
                         rep = f"{rep}_{dvalue.identifier}"
        return rep
        
class ConfigurableUtil(UtilBase):
    """Configurable class"""

    @classmethod
    def from_config(cls,config: Mapping[Any,Any],**kwargs):
        """Build an instance from configuration 

        :param config: the configuration to build an object 
        """
        raise NotImplementedError

class ConfigurablePipelineUtil(ConfigurableUtil):
    """Class for running a. 


    """

    def __call__(self,*args,**kwargs):
        return self.run_pipeline()
        
    def run_pipeline(self,*args,**kwargs):
        raise NotImplementedError
        
    def __init__(self,config):
        self.config = config
        
    @classmethod
    def from_config(cls,config):
        return cls(config)


class UtilResource(ConfigurableUtil):
    """Base class for utilities that have resources or costs associated with them 
    """
    
    @property
    def cost(self) -> float:
        """The cost associated with resource, defaults to 0.

        :returns: a numeric cost associated with item 
        """
        raise NotImplementedError

    @property 
    def utility_type(self) -> str:
        """The type of utility of this item"""
        raise NotImplementedError

    def query(self,query: str,**kwargs):
        raise NotImplementedError
    

class UtilityModel(BaseModel):
    """Base model for pydanic style classes

    """
    model_config = ConfigDict(protected_namespaces=())

    
Validator = field_validator
ModelField = Field
ProtectedNameSpace = ConfigDict(protected_namespaces=())
