from collections import defaultdict

import functools

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

class BlockRegister:
    """General class decorator for registering new class implementations 
    
    Methods
    --------
    factory_update_method(class_to_register): 
        Method for added a class to the registry. 

    """
    IMPLEMENTATIONS: DefaultDict[str,Callable] = defaultdict(dict)
    
    def __init__(
            self,
            name: str,
    ) -> None:
        """Creates `Register` instance. 

        :param name: 
            The unique identifier of the resource being registered. 
        """
        functools.update_wrapper(self, name)
        self.name = name

    def factory_update_method(self,class_to_register: Callable) -> None:
        """Method for adding class to the registry. 
        
        :param class_to_register: 
            The newly created class to add to the registry
        """
        self.IMPLEMENTATIONS[self.name] = class_to_register
        
    def __call__(self,*args,**kwargs):
        class_to_add = args[0]
        self.factory_update_method(class_to_add)
        return class_to_add

    def add_block(self,name,block_implementation):
        self.IMPLEMENTATIONS[name] = block_implementation

    def load_block(self,name):
        if name not in self.IMPLEMENTATIONS:
            raise ValueError(
                f'Block not found by name: {name}'
            )
        return self.IMPLEMENTATIONS[name]
