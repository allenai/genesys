''' System of R & D Agents and Selector for Scale Climbing Evolution '''

from __future__ import annotations

import sys
import exec_utils
import pathlib
import os
import json
import time
import tempfile
from dataclasses import dataclass, field, asdict
import networkx as nx
import pandas as pd
import numpy as np
from io import StringIO
import random
from networkx.drawing.nx_pydot import write_dot

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
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)
from .ve.run import main as ve_main
from .ve.run import parser as ve_parser


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
    seed_ids: List[str]
    # rating: int = None
    # review: str = None

    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, dict: Dict) -> DesignArtifact:
        return cls(**dict)

    def save(self,db_dir: str):
        U.mkdir(U.pjoin(db_dir,self.acronym))
        U.save_json(self.to_dict(),U.pjoin(db_dir,self.acronym,"artifact.json"))
        with open(U.pjoin(db_dir,self.acronym,"gab.py"),'w') as f:
            f.write(self.code)
        with open(U.pjoin(db_dir,self.acronym,"explaination.md"),'w') as f:
            f.write(self.explain)
        if self.instruct:
            with open(U.pjoin(db_dir,self.acronym,"instruct.md"),'w') as f:
                f.write(self.instruct)

    @classmethod
    def load(cls, db_dir: str, id:str) -> DesignArtifact:
        return cls.from_dict(U.load_json(U.pjoin(db_dir,id,"artifact.json")))
    
class PhylogeneticTree:
    # Read from a design base and construct a phylogenetic tree
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

    def export(self):
        G=nx.DiGraph()
        for node in self.G.nodes:
            data=self.G.nodes[node]['data']
            G.add_node(node,title=data.title,scale=data.scale,
                    #    explain=data.explain.replace(':',' '),
                    #    code=f'"{data.code}"'
                    )
        for edge in self.G.edges:
            G.add_edge(edge[0],edge[1])
        write_dot(G, U.pjoin(self.db_dir, '..', "phylogenetic_tree.dot"))
        


def report_reader(report):
    metrics={}
    training_record=report['training_record.csv']
    system_metrics=report['system_metrics.csv']
    trainer_state=report['trainer_state.json']
    eval_results=report['eval_results.json']
    
    metrics['perf']={}
    metrics['perf']['total_train_flos']=trainer_state['total_flos']
    metrics['perf']['total_eval_time']=float(eval_results['total_evaluation_time_seconds'])
    metrics['perf']['num_params']=eval_results['config']['model_num_parameters']
    metrics['eval']={}
    for task in eval_results['results']:
        res=eval_results['results'][task]
        task_alias=res['alias']
        if 'perplexity,none' in res:
            metrics['eval'][f'{task_alias}_ppl']=-np.log2(res['perplexity,none']) # the lower the better
        elif 'acc_norm,none' in res: 
            metrics['eval'][f'{task_alias}_acc_norm']=res['acc_norm,none']
        elif 'acc,none' in res:
            metrics['eval'][f'{task_alias}_acc']=res['acc,none']
    
    report={
        'metrics':metrics,
        'training_record':training_record,
        'system_metrics':system_metrics,
    }
    # TODO: add an analysis of the report by the Selector agent
    return report

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
        self.select_method=self.params['select_method']

        # load or init the state
        self.state=self.load_state() # load the state by evoname
        if 'selection_ratio' not in self.state:
            self.state['selection_ratio']=float(self.params['selection_ratio'])
        if 'scales' not in self.state:
            scales=self.params['scales'].split(',') # e.g. "14M,31M,70M,125M", scales
            self.state['scales']=list(sorted(scales, key=lambda x: U.letternum2num(x))) # sort from small to large
        if 'current_scale' not in self.state:
            self.state['current_scale']=0
        if 'budgets' not in self.state: # remaining budget for each scale
            self.state['budgets']={}
            budget=1
            for scale in self.state['scales'][::-1]:
                self.state['budgets'][scale]=int(np.ceil(budget))
                budget/=self.state['selection_ratio']
        # if 'unverified' not in self.state:
        #     self.state['unverified']=[]
        self.save_state() # save the initialized state

        self.scales=[eval(f'GAMConfig_{scale}()') for scale in self.state['scales']]

        print(f"Evolution system initialized with scales: {self.state['scales']}")
        print(f"Current scale: {self.state['current_scale']}")
        print(f"Budgets remaining: {self.state['budgets']}")
        print(f"Checkpoint directory: {self.evo_dir}")

        self.rnd_agent = BuildSystem(
            debug_steps=True, # True for debugging, but very long
            cache_type="diskcache", #<-- agent caching method 
            temperature=0.1,
            jupyter=False,
            cache_id=919,
            #from_json='/path/to/config'
        )
        self.ptree=PhylogeneticTree(U.pjoin(self.evo_dir,'db'))
        self.ptree.export()

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
        raise NotImplementedError("This method is not implemented for multi-gpu yet")
        while self.evolve():
            self.verify()
        
    def _run(self,mode):
        if mode=='evolve':
            print('\n\n'+'+'*50)
            print("RUNNING IN EVOLUTION MODE...\n\n")
            ret=self.evolve()
            if ret is None:
                print("No budget left for the evolution")
                time.sleep(1)
            else:
                print(f"Design {ret} sampled")
        elif mode=='verify':
            print('\n\n'+'x'*50)
            print("RUNNING IN VERIFICATION MODE...\n\n")
            ret=self.verify()
            if ret is None:
                print("No unverified design left")
                time.sleep(1)
            else: 
                print(f"Design {ret} verified")

    def evolve(self): # run a single evolution step unless no budget left
        scale_id=self.state['current_scale']
        if scale_id>=len(self.state['scales']): # no scale left
            print("No scale left")
            return None
        scale=self.state['scales'][scale_id]
        budget=self.state['budgets'][scale]
        if budget==0: 
            self.state['current_scale']+=1 # Will influence the sampling distribution
            self.save_state()
            self.evolve()
        else:
            return self._evolve(scale_id)

    def _evolve(self,scale_id): # do evolve that produce one design and operate the phylogenetic tree
        if scale_id==0:
            K=np.random.choice([0,1,2,3],p=[0.05,0.4,0.4,0.15]) # TODO: upgrade this to be configurable and better (e.g. decay 0 with scales)
        else:   
            K=np.random.choice([1,2,3],p=[0.4,0.4,0.2])
        instruct,seed_ids=self.select(K) # use the seed_ids to record the phylogenetic tree
        artifact=self.sample(scale_id,instruct) # NOTE: maybe randomly jump up or down to next scale? How to use the budget more wisely?
        if artifact is None:
            print("No design sampled")
            return None
        # save the design to the phylogenetic tree and update the budget
        artifact['seed_ids']=seed_ids
        self.ptree.new_design(artifact)
        scale=self.state['scales'][scale_id]
        self.state['budgets'][scale]-=1
        # self.state['unverified'].append(artifact['acronym'])
        self.save_state() # NOTE!!!: handle it carefully in multi-threading
        return artifact['acronym']

    def sample(self,scale_id,instruct,verbose=True):
        """ Sample a design at a given scale and verify it """
        self.rnd_agent.set_config(self.scales[scale_id])
        title,code,explain=self.rnd_agent(instruct) 
        # title,code,explain,review,rating=self.rnd_agent(instruct) 
        if title is None: # no design sampled
            return None
        for i in [' and ',' for ','-']:
            title=title.replace(i,' ')
        acronym=''.join([i[0].upper() for i in title.split(' ') if i.isalpha()])

        # modify the code to fit the block registry
        # TODO: change the registry name to acronyms
        code+='\n\n\nfrom .block_registry import BlockRegister\n\nBlockRegister(\n    name="default",\n    config=gab_config()\n)(GAB)'

        artifact={
            'title':title,
            'acronym':acronym,
            'code':code,
            'explain':explain,
            'scale':self.state['scales'][scale_id],
            'instruct':instruct,
            # 'review':review,
            # 'rating':rating,
        }
        return artifact

    def make_artifact(self,design_id,report=None):
        code=self.ptree.G.nodes[design_id]['data'].code
        explain=self.ptree.G.nodes[design_id]['data'].explain
        title=self.ptree.G.nodes[design_id]['data'].title
        acronym=self.ptree.G.nodes[design_id]['data'].acronym
        scale=self.ptree.G.nodes[design_id]['data'].scale
        config:GAMConfig=eval(f'GAMConfig_{scale}()')
        config_str=config.to_str()
        artifact_obj=f'## Title: {title}\n## Acronym: {acronym}\n\n## Code:\n\n{code}\n\n## Justification:\n\n{explain}'
        artifact_obj+=f'\\## Config:\n\n{config_str}\n\n'
        if report:
            artifact_obj+=f'## Report:\n\n{json.dumps(report,indent=4)}'
        return artifact_obj
    

    # TODO: upgrade to Selector agent

    def select(self,K: int=1,selector_instruct=''): # K is the number of designs to sample, instruct is the instruction to the selector, select seeds or select populations
        """ Provide the instruction including seeds and instructs for the next design """
        K=min(K,len(self.ptree.G.nodes))
        if K==0: # no design to sample
            return '',[]
        if self.select_method=='heuristic':
            topk,reports = self.heuristic_select(K,selector_instruct)
        elif self.select_method=='random':
            topk,reports = self.random_select(K,selector_instruct)
        if K==1: # Mutate
            artifact_obj=self.make_artifact(topk[0],reports[topk[0]])
            instruct=f'Please improve based on this design for the new design, think of how to overcome its weaknesses and absorb its advantage:\n\n{artifact_obj}'
        else: # Cross-over
            artifact_objs='\n\n\n'.join([self.make_artifact(design_id,reports[design_id]) for design_id in topk])
            instruct=f'Please improve by combining the advantages and mitigating the disadvantages of these designs for the new design:\n\n{artifact_objs}'
        return instruct,list(topk)
        
    def heuristic_select(self,K: int=1,selector_instruct=''):
        alpha=0.1
        sample_metrics={}
        sample_scale={}
        reports={}
        for node in self.ptree.G.nodes:
            artifact=self.ptree.G.nodes[node]['data']
            if U.pexists(U.pjoin(self.evo_dir,'ve',node,'report.json')):
                report=U.load_json(U.pjoin(self.evo_dir,'ve',node,'report.json'))
                report=report_reader(report)
                reports[node]=report
                # TODO: upgrade this thing
                sample_metrics[node]=np.mean([v for k,v in report['metrics']['eval'].items() if 'acc' in k])
                scale_id=self.state['scales'].index(artifact.scale)+1
                sample_scale[node]=scale_id

        # TODO: upgrade this thing
        prob=np.array([v for k,v in sample_metrics.items()])*(1-alpha)+np.random.rand(len(sample_metrics))*alpha
        prob=prob*np.array([v for k,v in sample_scale.items()]) # prefer the higher scale
        prob=prob/np.sum(prob)
        topk=np.random.choice(list(sample_metrics.keys()),size=K,replace=False,p=prob)
        return topk,reports

    def random_select(self,K: int=1,selector_instruct=''):
        topk=random.sample(self.ptree.G.nodes,K)
        reports={}
        for node in topk:
            if U.pexists(U.pjoin(self.evo_dir,'ve',node,'report.json')):
                report=U.load_json(U.pjoin(self.evo_dir,'ve',node,'report.json'))
                report=report_reader(report)
                reports[node]=report
            else:
                reports[node]=None
        return topk,reports


    
    def verify(self): # run a single verify that verify one unverified design
        designed=os.listdir(U.pjoin(self.evo_dir,'db'))
        for design_id in designed:
            report_dir=U.pjoin(self.evo_dir,'ve',design_id,'report.json')
            if U.load_json(report_dir)=={}:
                for _ in range(3): # try 3 times
                    self._verify(design_id) # verify the design until it's done
                    report=U.load_json(report_dir)
                    if report!={}: 
                        return design_id
                return 'FAILED'
        return None
        

    def _verify(self,design_id): # do a single verify
        artifact=U.load_json(U.pjoin(self.evo_dir,'db',design_id,'artifact.json'))
        with open('/home/junyanc/model_discovery/model_discovery/model/gab.py','w') as f:
            f.write(artifact['code'])
        args = ve_parser.parse_args()
        args.evoname=self.evoname
        args.design_id=artifact['acronym']
        args.config=f'GAMConfig_{artifact["scale"]}'
        args.ckpt_dir=self.ckpt_dir
        args.data_dir=os.environ.get("DATA_DIR")
        args.resume=True
        args.training_token_multiplier=self.get_train_budget(artifact)
        ve_main(args)


    def get_train_budget(self,artifact):
        # rating=artifact['rating']
        return 20
    

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












############################################################################################################

def test_evolve(test_name):
    strparams=[
        f"evoname={test_name}",
        "scales=14M,31M,70M",
        "selection_ratio=0.25",
        "select_method=random",
    ]
    evolution_system = BuildEvolution(
        strparams=';'.join(strparams),
        cache_type='diskcache',
    )
    while evolution_system.evolve():
        pass



if __name__ == '__main__':
    strparams=[
        "evoname=evolution_test1",
        "scales=14M,31M,70M",
        "selection_ratio=0.25",
        "select_method=random",
    ]

    # evoname=ve_parser.parse_args().evoname
    # strparams.append(f"evoname={evoname}")

    # evolution_system = BuildEvolution(
    #     strparams=';'.join(strparams),
    #     cache_type='diskcache',
    # )
    # evolution_system._evolve(0)


    test_evolve('evo_test_001')

    