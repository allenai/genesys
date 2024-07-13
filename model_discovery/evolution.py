from __future__ import annotations

# ''' System of R & D Agents and Selector for Scale Climbing Evolution '''

# class ScaleClimbingEvolution:

#     def __init__(self) -> None:
#         pass


# def evolve():
#     # Evolve new individuals in a population
#     pass

# def select():
#     # Select the best individuals in a population
#     pass
    
# def scale_climbing():
#     # Scale climbing evolutionary algorithm for model hyperparameters
#     pass


import exec_utils
import pathlib
import os
import json
import time
import tempfile

from types import ModuleType
from typing import (
    Type,
    List,
    Dict,
    Any,
    TypeVar,
    Tuple,
    Optional,
    Union
)
from .system import BuildSystem
from exec_utils.factory import _check_config
from exec_utils import BuildSystem as NativeBuild


__all__ = [
    "EvolutionSystem",
    "BuildEvolution",
]

@exec_utils.Registry("config","evolution")
class CustomParams(exec_utils.ModuleParams):
    scales: str = exec_utils.ParamField(
        default='',
        metadata={
            "help"         : 'The different scales to ',
            "exclude_hash" : True,
        }
    )

@exec_utils.Registry(
    resource_type="system_type",
    name="evolution",
    #cache="query_system",
)
class EvolutionSystem(exec_utils.System):
    def __init__(self,agent_system,config,**kwargs):
        self.agents = agent_system
        self._config = config
                    
    def query_system(self,
        query: Optional[str] = '',
        stream: Optional[ModuleType] = None,
        frontend: Optional[bool] = False,
        status: Optional[bool] = True,
        **kwargs
    ) -> list:
        for scale in self._config.scales.split(','):
            print(scale)

    @classmethod
    def from_config(cls,config,**kwargs):
        """Loads all the evolution components from configuration 

        :param config:
            The global configuration spec. 

        """
        config.system_type = "model_discovery_system"
        agent = BuildSystem(
            config,
            **kwargs
        )
        return cls(agent,config) 

def BuildEvolution(
        config: Optional[ConfigType] = None,
        **kwargs
    ):
    """Factory for loading evolution system 

    :param config: 
        Configuration object (optional) 

    """
    kwargs["system_type"] = "evolution"
    evolution = NativeBuild(config,**kwargs)
    return evolution
