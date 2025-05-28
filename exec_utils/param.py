import collections
import inspect

from typing import (
    Optional,
    ClassVar,
    TypeVar,
    Type,
    Union,
    Dict,
    Any,
    Set
)
from argparse import (
    ArgumentParser,
    Namespace,
    BooleanOptionalAction,
    _ArgumentGroup as ArgumentGroup,
)
from pydantic import (
    RootModel,
    Field
)
from .base import UtilityModel

__all__ = [
    "ParamField",
    "ModuleParams",
    "RootParams",
]

ParamField = Field
C = TypeVar("C",bound="ArgumentParser")
D = TypeVar("D",bound="RootParams")

class ModuleParams(UtilityModel):
    """Utility for building parsers from configuration 
    
    :param GROUP: the name of the parameter group (optional)
    """
    GROUP: ClassVar[str] = ""
    EXCLUDED_KEYS: ClassVar[Set[str]] = set()
    

    @classmethod
    def load(cls, parser: Optional[C] = None,**kwargs) -> C:
        """Build or modifies an argument parser with the class attributes 

        :param parser: 
            An existing parser or None (will instantiate a parser if empty)
        :returns: 
            The underlying argument parser. 
        """
        parser = ArgumentParser() if parser is None else parser
        group = None if not cls.GROUP else parser.add_argument_group(cls.GROUP)

        return cls._populate_parser(parser,group=group,**kwargs) 

    @classmethod
    def _populate_parser(
            cls,
            parser: C,
            group: Optional[ArgumentGroup] = None,
            **kwargs
    ) -> C:
        """Adds parameters to an argument parser. 

        :params parser: 
            The input argument parser 
        :param groups: 
            The particular group (if specified) in that parser 
        :returns: 
            An updated or fresh argument parser. 
 
        """
        global_keys = kwargs.get("global_keys",{})

        
        for (field_name,field) in cls.model_fields.items():
            
            extra = {}
            choices = None
            
            if hasattr(field,"json_schema_extra") and field.json_schema_extra:
                extra = field.json_schema_extra.get("metadata",{})

            choices = extra.get("choices",None)
            if global_keys and field_name in global_keys:
                if isinstance(choices,list): 
                    choices += global_keys[field_name]
                else:
                    choices = global_keys[field_name]

            type_annot = None 
            if extra.get("type",None):
                type_annot = extra["type"]
            elif inspect.isclass(field.annotation):
                type_annot = field.annotation

            exclude_hash = None
            
            if extra.get("exclude_hash"):
                cls.EXCLUDED_KEYS.add(field_name)
                
            args = {
                "help"    : extra.get("help",None),
                "choices" : choices,
                "type"    : type_annot,
                "required": field.is_required(),
                "default" : field.default,
            }
                
            if isinstance(field.default,bool):
                args["type"] = bool
                args["required"] = None
                args["action"] = BooleanOptionalAction
                
            arg_obj = group if group else parser
            arg_obj.add_argument(f"--{field_name}",**args)

        return parser

class RootParams(RootModel):
    """A special model class for building a parameter class 
    from multiple individual classes. 
    
    """
    def build_parser(self,parser: Optional[C] = None,**kwargs) -> C:
        """Build a config parser from the underlying paramers items included
        
        :returns: argument parser 
        """
        prog = kwargs.get("prog","")
        descr = kwargs.get("description","")
        choices = kwargs.get("choices",{})
        
        parser = ArgumentParser(prog=prog,description=descr) if parser is None else parser
        
        for param_class in self.root:
            parser = param_class.load(parser,**kwargs)
            
        return parser 

    def __add__(self,other_root: C) -> C:
        if other_root.root == self.root:
           return self 
        return RootParams(root=self.root+other_root.root)


