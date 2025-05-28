import time

from sympy import sympify
from abc import abstractmethod
from typing import (
    Union,
    TypeVar,
    List,
    Dict,
    Tuple,
    Callable,
    Type,
    Any
)
from ..aliases import ConfigType
from ..base import UtilResource
from ..param import (
    ModuleParams,
    ParamField,
)
from ..cache import setup_cache
from ..register import Registry

__all__ = [
    "BaseTool",
    "Calculator",
]
    
class BaseTool(UtilResource):
    """Base class for tools"""

    @property
    def cost(self):
        """No cost associated with calcuator"""
        return 0.0
        
    @property
    def name(self):
        """No cost associated with calcuator"""
        raise NotImplementedError
        
    def __call__(self,query: str,**kwargs):
        return self.query(query,**kwargs)
    
    @classmethod
    def from_config(cls,config,**kwargs):
        return cls()


@Registry("tool_type","calculator")
class Calculator(BaseTool):
    
    """A simple calculator tool that can evaluate expressions"""

    def _query_tool(self,query: str,**kwargs) -> Dict[str,Any]:
        ### uses `eval`, be carefule
        ### https://docs.sympy.org/latest/modules/core.html#module-sympy.core.sympify
        parsed = True
        out = {}
        try:
            v = sympify(query)
        except Exception as e:
            v = e
            parse = False
        out["feedback"] = v
        out["_details"] = {}
        out["_details"]["parsed"] = parsed

        return out

    def query(self,query: str,**kwargs):
        return self._query_tool(query,**kwargs)
    
    @property
    def cost(self):
        """No cost associated with calcuator"""
        return 0.0

    @property
    def name(self):
        """The name"""
        return "calculator"

    @classmethod
    def from_config(cls,config,**kwargs):
        return cls()

@Registry("config","lm_exec_utils.tools.tool")
class Params(ModuleParams):
    """Parameters for model classes
    
    :param tool_type: 
        The type of tool to use when building systems. 

    """
    
    tool_type: str = ParamField(
        default='',
        metadata={"help" : 'The particular type of tool to use'}
    )
