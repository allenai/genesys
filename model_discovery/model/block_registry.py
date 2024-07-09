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
            config: Optional[dict] = {},
    ) -> None:
        """Creates `Register` instance. 

        :param name: 
            The unique identifier of the resource being registered. 
        :param config: 
            The configuration associated with the block implementation 
        """
        functools.update_wrapper(self, name)
        self.name = name
        self.config = config 

    def factory_update_method(self,class_to_register: Callable) -> None:
        """Method for adding class to the registry. 
        
        :param class_to_register: 
            The newly created class to add to the registry
        """
        self.IMPLEMENTATIONS[self.name] = (class_to_register,self.config)
        
    def __call__(self,*args,**kwargs):
        class_to_add = args[0]
        self.factory_update_method(class_to_add)
        return class_to_add
    
    @classmethod 
    def add_block(cls,name,block_implementation, config: Optional[dict] = {}) -> None:
        cls.IMPLEMENTATIONS[name] = (block_implementation,config)

    @classmethod
    def load_block(cls,name):
        if name not in cls.IMPLEMENTATIONS:
            raise ValueError(
                f'Block not found by name: {name}'
            )
        return cls.IMPLEMENTATIONS[name]
