''' System of R & D Agents and Selector for Scale Climbing Evolution '''

from __future__ import annotations

import exec_utils
import pathlib
import os
import json
import time
import tempfile
from dataclasses import dataclass, field, asdict
import networkx as nx

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
from exec_utils.aliases import ConfigType

from model_discovery import utils as U
from .configs.gam_config import (
    GAMConfig, GAMConfig_10M, GAMConfig_35M, GAMConfig_70M, GAMConfig_130M,
)


__all__ = [
    "EvolutionSystem",
    "BuildEvolution",
]


@dataclass
class DesignArtifact:
    title: str
    acronym: str # id
    code: str
    explain: str
    scale: str
    instruct: str
    seed_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, dict: Dict) -> DesignArtifact:
        return cls(**dict)

    def save(self,db_dir: str):
        U.mkdir(U.pjoin(db_dir,self.acronym))
        U.save_json(self.to_dict(),U.pjoin(db_dir,self.acronym,"artifact.json"))

    @classmethod
    def load(cls, db_dir: str, id:str) -> DesignArtifact:
        return cls.from_dict(U.load_json(U.pjoin(db_dir,id,"artifact.json")))


class PhylogeneticTree:
    def __init__(self, db_dir: str):
        self.G = nx.DiGraph()
        self.db_dir = db_dir
        U.mkdir(db_dir)
        self.load()

    def new_design(self, artifact_dict: dict):
        artifact = DesignArtifact.from_dict(artifact_dict)
        acronym = self.unique_acronym(artifact.acronym)
        artifact.acronym = acronym
        artifact.save(self.db_dir)
        self.G.add_node(acronym, data=artifact)
        for seed_id in artifact.seed_ids:
            self.G.add_edge(seed_id, acronym)

    def unique_acronym(self, acronym: str) -> str:
        existing_acronyms = set(self.G.nodes)
        if acronym not in existing_acronyms:
            return acronym
        i = 1
        while f"{acronym}{i}" in existing_acronyms:
            i += 1
        return f"{acronym}{i}"

    def load(self):
        edges_to_add = []
        for id in os.listdir(self.db_dir):
            artifact = DesignArtifact.load(self.db_dir, id)
            self.G.add_node(artifact.acronym, data=artifact)
            for seed_id in artifact.seed_ids:
                edges_to_add.append((seed_id, artifact.acronym))

        for seed_id, design_id in edges_to_add:
            self.G.add_edge(seed_id, design_id)

    def __str__(self):
        return nx.info(self.G)




# @exec_utils.Registry("config","evolution")
# class CustomParams(exec_utils.ModuleParams):
#     strparams: str = exec_utils.ParamField(
#         default='',
#         metadata={
#             "help"         : '";" separated parameters, e.g. "param1=val1;param2=val2", just use it for now',
#             "exclude_hash" : True,
#         }
#     )

@exec_utils.Registry(
    resource_type="system_type",
    name="evolution",
    #cache="query_system",
)
class EvolutionSystem(exec_utils.System):
    def __init__(self,agent_system,config,**kwargs):
        self.agents = agent_system
        self._config = config
        
        # init params, TODO: upgrade to exec_util params, use a simple str params for now
        self.params={}
        for param in self._config.strparams.split(';'):
            key,val=param.split('=')
            self.params[key]=val
        print(self.params)

        # set the name and save dir
        self.evoname=self.params['evoname'] # Provide the name for the whole run including evolutions of all scales, all designs, all agents
        if not 'ckpt_dir' in self.params or not self.params['ckpt_dir']:
            self.ckpt_dir=os.environ.get("CKPT_DIR")
        self.evo_dir=U.pjoin(self.ckpt_dir,self.evoname)
        U.mkdir(self.evo_dir)

        # load or init the state
        self.state=self.load_state() # load the state by evoname
        if 'scales' not in self.state:
            scales=self.params['scales'].split(',') # e.g. "10M:64,35M:16,70M:4,130M:1", scale and budget
            scales=sorted(scales, key=lambda x: U.letternum2num(x.split(':')[0]))
            self.state['scales']=[s.split(':')[0] for s in scales]
        if 'current_scale' not in self.state:
            self.state['current_scale']=0
        if 'budgets' not in self.state: # remaining budget for each scale
            self.state['budgets']={s.split(':')[0]:int(s.split(':')[1]) for s in scales}
        self.save_state() # save the initialized state

        self.scales=[eval(f'GAMConfig_{scale}()') for scale in self.state['scales']]

        print(f"Evolution system initialized with scales: {self.state['scales']}")
        print(f"Current scale: {self.state['current_scale']}")
        print(f"Budgets remaining: {self.state['budgets']}")
        print(f"Checkpoint directory: {self.evo_dir}")

        self.rnd_agent = BuildSystem(
            debug_steps=True,
            cache_type="diskcache", #<-- agent caching method 
            temperature=0.1,
            jupyter=False,
            cache_id=919,
            #from_json='/path/to/config'
        )
        self.ptree=PhylogeneticTree(U.pjoin(self.evo_dir,'db'))

    def query_system(self,
        query: Optional[str] = '',
        stream: Optional[ModuleType] = None,
        frontend: Optional[bool] = False,
        status: Optional[bool] = True,
        **kwargs
    ) -> list:
        """ Talk to the selector agent """
        
        print("Hello from the evolution system")


    def load_state(self):
        return U.load_json(U.pjoin(self.evo_dir,'state.json'))

    def save_state(self):
        U.save_json(self.state,U.pjoin(self.evo_dir,'state.json'))

    def run(self):
        """ Run the scale-climbing evolution """
        for idx,scale in enumerate(self.state['scales']):
            if idx<self.state['current_scale']:
                continue
            self.evolve(idx)
            self.state['current_scale']+=1
            self.save_state()

    def evolve(self,scale_id):
        """ Evolve the design at a given scale """

        scale=self.state['scales'][scale_id]
        budget=self.state['budgets'][scale]
        print(f"Evolve at scale {scale} with budget {budget}")
        for _ in range(budget):
            instruct,seed_ids=self.select() # use the seed_ids to record the phylogenetic tree
            artifact=self.sample(scale_id,instruct) # NOTE: maybe randomly jump up or down to next scale? How to use the budget more wisely?

            # save the design to the phylogenetic tree and update the budget
            artifact['seed_ids']=seed_ids
            self.ptree.new_design(artifact)
            self.state['budgets'][scale]-=1
            self.save_state() # NOTE!!!: handle it carefully in multi-threading

    def sample(self,scale_id,instruct,verbose=True):
        """ Sample a design at a given scale and verify it """
        self.rnd_agent.set_config(self.scales[scale_id])
        title,code,explain=self.rnd_agent(instruct) 
        for i in [' and ',' for ','-']:
            title=title.replace(i,' ')
        acronym=''.join([i[0].upper() for i in title.split(' ') if i.isalpha()])

        artifact={
            'title':title,
            'acronym':acronym,
            'code':code,
            'explain':explain,
            'scale':self.state['scales'][scale_id],
            'instruct':instruct,
        }
        return artifact
    

    def verify(self,artifact):
        pass


    def select(self):
        """ Provide the instruction including seeds and instructs for the next design """
        # TODO
        return '',[]

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
    ) -> EvolutionSystem:
    """Factory for loading evolution system 

    :param config: 
        Configuration object (optional) 

    """
    kwargs["system_type"] = "evolution"
    evolution = NativeBuild(config,**kwargs)
    return evolution



if __name__ == '__main__':
    strparams=[
        "evoname=evolution_test",
        # "scales=10M:64,35M:16,70M:4,130M:1",
        # "scales=10M:16,35M:4,70M:1",
        "scales=10M:4,35M:1",
    ]

    evoname=strparams[0].split('=')[1]
    ckpt_dir=os.environ.get("CKPT_DIR")
    evo_dir=U.pjoin(ckpt_dir,evoname)
    os.removedirs(evo_dir) if os.path.exists(evo_dir) else None

    evolution_system = BuildEvolution(
        strparams=';'.join(strparams),
        cache_type='diskcache',
    )
    print(evolution_system)

    evolution_system('')

    # evolution_system.sample(0,'')
    evolution_system.run()
    

