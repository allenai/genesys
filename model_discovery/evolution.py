''' System of R & D Agents and Selector for Scale Climbing Evolution '''

from __future__ import annotations

import sys
import re
import exec_utils
import pathlib
import datetime
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
from networkx.drawing.nx_pydot import to_pydot,warnings
from pyvis.network import Network
import math


try: # a stupid patch for windows
    from .secrets import *
    os.environ['MY_OPENAI_KEY']=MY_OPENAI_KEY
    os.environ['CLAUDE_API_KEY']=CLAUDE_API_KEY
    os.environ['HF_KEY']=HF_KEY
    os.environ['HF_HUB_KEY']=HF_HUB_KEY
    os.environ['GITHUB_TOKEN']=GITHUB_TOKEN
    os.environ['WANDB_API_KEY']=WANDB_API_KEY
    os.environ['S2_API_KEY']=S2_API_KEY
    os.environ['AWS_SECRET_ACCESS_KEY']=AWS_SECRET_ACCESS_KEY
    os.environ['AWS_ACCESS_KEY_ID']=AWS_ACCESS_KEY_ID
    os.environ['DATA_DIR']=DATA_DIR
    os.environ['CKPT_DIR']=CKPT_DIR
    os.environ['HF_DATASETS_TRUST_REMOTE_CODE']='1'    
except:
    pass


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
from .system import BuildSystem,PrintSystem
from exec_utils.factory import _check_config
from exec_utils import BuildSystem as NativeBuild
from exec_utils.aliases import ConfigType

from model_discovery.model.composer import GAUTree
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


LIBRARY_DIR = './model_discovery/model/library'

NODE_COLOR_MAP={
    '14M':'#5698c3',
    '31M':'#1177b0',
    '70M':'#15559a',
    '125M':'#f0a1a8',
    '350M':'#f07c82',
    '760M':'#ee3f4d',
    '1300M':'#fcb70a',
}

ROOT_COLOR='#9eccab'

NODE_SIZE_MAP={
    '14M':15,
    '31M':20,
    '70M':25,
    '125M':30,
    '350M':35,
    '760M':40,
    '1300M':45,
}

CORE_COLOR = '#f0a1a8' # core reference
REFERENCE_COLOR = '#AF47D2'
RWC_COLOR = '#FB773C' # reference with code
EXT_COLOR_1HOC = '#ed556a' # extended 1-hop reference


@dataclass
class NodeObject:
    acronym: str # acronym is the unique identifier for the node
    title: str
    seed_ids: List[str]

    @property
    def type(self) -> str:
        return self.__class__.__name__
    
    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, dict: Dict):
        return cls(**dict)
    
    @classmethod
    def load(cls, save_dir: str, acronym:str):
        with open(U.pjoin(save_dir,acronym+'.json'),'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def save(self,save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(U.pjoin(save_dir,self.acronym+'.json'),'w', encoding='utf-8') as f:
            json.dump(self.to_dict(),f,indent=4)

    def to_desc(self) -> str:
        raise NotImplementedError
    
    def to_prompt(self) -> str:
        raise NotImplementedError

@dataclass
class LibraryReference(NodeObject):
    s2id: str = None
    abstract: str = None
    authors: List[str] = None
    venue: str = None
    year: int = None
    tldr: str = None
    # embedding: list
    citationCount: int = None
    influentialCitationCount: int = None
    code: str = None
    description: str = None
    url: str = None
    tree: GAUTree = None

    def __post_init__(self):
        py_dir=U.pjoin(LIBRARY_DIR,'base',self.acronym,self.acronym+'_edu.py')
        go_dir=U.pjoin(LIBRARY_DIR,'base',self.acronym,self.acronym+'_edu.go')
        core_dir=U.pjoin(LIBRARY_DIR,'core',self.acronym)
        if U.pexists(py_dir):
            self.code=f'# {self.acronym}_edu.py\n\n'+open(py_dir,'r', encoding='utf-8').read()
        elif U.pexists(go_dir):
            self.code=f'// {self.acronym}_edu.go\n\n'+open(go_dir,'r', encoding='utf-8').read()
        elif U.pexists(core_dir):
            code_dir=U.pjoin(core_dir,self.acronym+'_edu.py')
            tree_dir=U.pjoin(core_dir,'units')
            if U.pexists(code_dir):    
                self.code=f'# {self.acronym}_edu.py\n\n'+open(code_dir,'r', encoding='utf-8').read()
            if U.pexists(tree_dir):
                self.tree=GAUTree.load_from_base(tree_dir)
            if self.tree is not None:
                print(f'{self.acronym} tree loaded')
        else:
            self.code=None

    @property
    def type(self) -> str:
        core_dir=U.pjoin(LIBRARY_DIR,'core',self.acronym)
        if self.tree is not None: # as long as there is a tree, it is a core reference
            return 'ReferenceCoreWithTree'
        elif self.code is not None: # if there is code, it is a reference with code
            if U.pexists(core_dir): # if there is a core dir, it is a core reference
                return 'ReferenceCore'
            else: # otherwise, it is a reference with code
                return 'ReferenceWithCode'
        else: # if there is no code, it is a reference
            return 'Reference'

    def to_desc(self, reformat=True) -> str:
        mdtext = f'## {self.title}'
        
        if self.s2id:
            mdtext += f'\n**S2 ID:** {self.s2id}'
        
        if self.authors:
            authors = ', '.join(self.authors)
            mdtext += f'\n**Authors:** {authors}'
        
        if self.tldr:
            mdtext += f'\n\n**TL;DR:** {self.tldr}'
        
        if self.abstract:
            abstract = self.abstract.replace('. ', '.\n') if reformat else self.abstract
            mdtext += f'\n\n**Abstract:**\n{abstract}'
        
        if self.venue:
            mdtext += f'\n\n**Published at:** *{self.venue}* in {self.year}'
        
        if self.citationCount:
            mdtext += f'\n\n**Cited:** {self.citationCount} times'
        
        if self.influentialCitationCount:
            mdtext += f'\n\n**Impactful Citations:** {self.influentialCitationCount}'
        
        if self.description:
            description = self.description.replace('. ', '.\n') if reformat else self.description
            mdtext += f'\n\n### Description:\n{description}'
        
        if self.url:
            mdtext += f'\n\n**[Link to Paper]({self.url})**'

        if reformat:
            return mdtext.replace(':', ' ').replace('e.\ng.\n', 'e.g.').replace('i.\ne.\n', 'i.e.')
        
        return mdtext

    def to_prompt(self) -> str:
        prompt = self.to_desc(reformat=False)
        
        if self.tree:
            prompt += f'\n\n{self.tree.to_prompt()}\n\n'
        elif self.code:
            if self.type == 'ReferenceCore':
                prompt += (
                    f'\n\n## GAB Implementation\n'
                    '<details><summary>Click to expand</summary>\n\n'
                    '```python\n'
                    f'{self.code}\n'
                    '```\n'
                    '</details>\n\n'
                )
            else:
                prompt += (
                    f'\n\n## Reference Code\n'
                    '<details><summary>Click to expand</summary>\n\n'
                    '```python\n'
                    f'{self.code}\n'
                    '```\n'
                    '</details>\n\n'
                )

        return prompt


@dataclass
class LibraryReference1hop(LibraryReference):

    @property
    def type(self) -> str:
        # if self.code is not None:
        #     return 'Reference1hopWithCode'
        # else:
        return 'Reference1hop'

@dataclass
class DesignArtifact(NodeObject):
    code: str
    rawcode: str
    explain: str
    scale: str
    summary: str
    instruct: str
    ratings: dict
    reviews: dict
    session_id: str # store the agent design process
    verify_report: dict = None
    check_results: dict = None

    def save(self,db_dir: str):
        U.mkdir(U.pjoin(db_dir,self.acronym))
        U.save_json(self.to_dict(),U.pjoin(db_dir,self.acronym,"artifact.json"))
        with open(U.pjoin(db_dir,self.acronym,"gab.py"),'w', encoding='utf-8') as f:
            f.write(self.code)
        with open(U.pjoin(db_dir,self.acronym,"explaination.md"),'w', encoding='utf-8') as f:
            f.write(self.explain)
        if self.instruct:
            with open(U.pjoin(db_dir,self.acronym,"instruct.md"),'w', encoding='utf-8') as f:
                f.write(self.instruct)
        with open(U.pjoin(db_dir,self.acronym,"summary.md"),'w', encoding='utf-8') as f:
            f.write(self.summary)
        with open(U.pjoin(db_dir,self.acronym,"reviews.md"),'w', encoding='utf-8') as f:
            f.write(self.get_reviews())
        if self.check_results:
            U.save_json(self.check_results,U.pjoin(db_dir,self.acronym,"check_results.json"))
        
    @classmethod
    def load(cls, db_dir: str, id:str) -> DesignArtifact:
        obj = cls.from_dict(U.load_json(U.pjoin(db_dir,id,"artifact.json")))
        report_dir=U.pjoin(db_dir,'..','ve',id,'report.json')
        if U.pexists(report_dir):
            obj.verify_report=U.load_json(report_dir)
        return obj


    def to_desc(self) -> str:
        title=self.title.replace(':',' ')
        summary=self.summary.replace(':',' ')

        # Split the summary into parts: code blocks and non-code blocks
        code_block_pattern = re.compile(r'(```python[\s\S]*?```)', re.MULTILINE)
        numbered_list_pattern = re.compile(r'(\d+\.\s[^\n]*\n)', re.MULTILINE)
        parts = re.split(r'(```python[\s\S]*?```|\d+\.\s[^\n]*\n)', summary)

        # Replace colons in the non-code block parts
        for i in range(len(parts)):
            if not code_block_pattern.match(parts[i]) and not numbered_list_pattern.match(parts[i]):
                parts[i] = parts[i].replace('.', '.\n').replace(';', ';\n').replace('?', '?\n').replace('!', '!\n').replace(',', '\n')

        # Join the parts back together
        summary = ''.join(parts)
        mdtext=f'# {title} ({self.scale})\n\n{summary}\n\n## Rating\n{self.rating} out of 5'
        return mdtext.replace('e.\ng.\n','e.g.').replace('i.\ne.\n','i.e.')
    
    def get_reviews(self):
        review_ratings=''
        for idx, style in enumerate(self.reviews):
            review=self.reviews[style]
            rating=self.ratings[style]
            review_ratings+=f'# Review of Reviewer {idx+1} ({style}):\n\n{review}\n\n## Rating: {rating} out of 5\n\n'
        return review_ratings

    def to_prompt(self):
        scale=self.scale
        config:GAMConfig=eval(f'GAMConfig_{scale}()')
        config_str=config.to_prompt()
        prompt=f'## Title: {self.title}\n## Acronym: {self.acronym}\n\n## Code:\n\n{self.rawcode}\n\n## Justification:\n\n{self.explain}'
        prompt+=f'\\## Config and Reference:\n\n{config_str}\n\n'
        prompt+=self.get_reviews()
        if self.check_results:
            prompt+=f"## Effectiveness:\n\n{json.dumps(self.check_results['effectiveness'],indent=4)}\n\n"
        if self.verify_report:
            report=report_reader(self.verify_report)
            prompt+=f'## Report:\n\n{json.dumps(report,indent=4)}\n\n'
        return prompt
    

def write_dot(G, path):
    """Write NetworkX graph G to Graphviz dot format on path.

    Path can be a string or a file handle.
    """
    msg = (
        "nx.nx_pydot.write_dot depends on the pydot package, which has"
        "known issues and is not actively maintained. Consider using"
        "nx.nx_agraph.write_dot instead.\n\n"
        "See https://github.com/networkx/networkx/issues/5723"
    )
    warnings.warn(msg, PendingDeprecationWarning, stacklevel=2)
    P = to_pydot(G)
    with open(path, "w", encoding='utf-8') as f:
        f.write(P.to_string())
    return


class PhylogeneticTree: ## TODO: remove redundant edges and reference nodes
    # Read from a design base and construct a phylogenetic tree
    def __init__(self, db_dir: str):
        self.G = nx.DiGraph()
        self.db_dir = db_dir
        self.lib_dir = U.pjoin(LIBRARY_DIR,'tree')
        self.lib_ext_dir = U.pjoin(LIBRARY_DIR,'tree_ext')
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

    def filter_by_type(self,types):
        if isinstance(types, str):
            types=[types]
        nodes=[]
        for node in self.G.nodes:
            if self.G.nodes[node]['data'].type in types:
                nodes.append(node)
        return nodes
    
    def remove_redundant_edges(self):
        G = self.G  # Work directly with self.G
        topological_order = list(nx.topological_sort(G))
        redundant_edges = []

        for node in topological_order:
            # Get all successors of the current node
            successors = list(G.successors(node))
            for succ in successors:
                # Temporarily remove the edge to check for other paths
                G.remove_edge(node, succ)
                if nx.has_path(G, node, succ):
                    redundant_edges.append((node, succ))
                # Re-add the edge
                G.add_edge(node, succ)

        # Remove all redundant edges
        for u, v in redundant_edges:
            G.remove_edge(u, v)
            
    def load(self):
        edges_to_add = []
        for id in os.listdir(self.db_dir):
            artifact = DesignArtifact.load(self.db_dir, id)
            self.G.add_node(artifact.acronym, data=artifact)
            for seed_id in artifact.seed_ids:
                edges_to_add.append((seed_id, artifact.acronym))
        
        # Load core library
        for id in os.listdir(self.lib_dir):
            id=id.split('.')[0]
            ref = LibraryReference.load(self.lib_dir, id)
            self.G.add_node(ref.acronym, data=ref)
            for seed_id in ref.seed_ids:
                edges_to_add.append((seed_id, ref.acronym))

        # load extended library
        dir_ext_1hop = U.pjoin(self.lib_ext_dir,'1hop')
        for i in os.listdir(dir_ext_1hop):
            id=i.split('.')[0]
            ref = LibraryReference1hop.load(dir_ext_1hop, id)
            self.G.add_node(ref.acronym, data=ref)
            for seed_id in ref.seed_ids:
                edges_to_add.append((seed_id, ref.acronym))
        
        for seed_id, product_id in edges_to_add:
            if seed_id == product_id or nx.has_path(self.G, product_id, seed_id):
                continue
            self.G.add_edge(seed_id, product_id)
        
        self.remove_redundant_edges()
        

    def viz(self,G,height=5000,width="100%",layout=False,max_nodes=None): # larger canvas may be needed for large trees
        nt=Network(
            directed=True,height=height,width=width,
            layout=layout, bgcolor="#fafafa", #font_color="#ffffff",
            #select_menu=True, # filter_menu=True,
            # heading=f'Phylogenetic Tree for {self.db_dir.split("/")[-2]}'
        )
        nt.prep_notebook(True)#,'./etc/ptree_template.html')
        nt.from_nx(G)
        fname='PTree' if not layout else 'PTree_layout'
        if max_nodes: fname+=f'_{max_nodes}'
        nt.show(U.pjoin(self.db_dir, '..', fname+'.html'))

    def export(self,with_ext=False,max_nodes=None,height=5000,layout=False):
        G=nx.DiGraph()
        for idx,node in enumerate(self.G.nodes):
            if max_nodes and idx>max_nodes:
                break
            data=self.G.nodes[node]['data']
            if data.type=='DesignArtifact':
                scale=data.scale
                color=NODE_COLOR_MAP[scale]
                if data.seed_ids == []:
                    color=ROOT_COLOR
                size=NODE_SIZE_MAP[scale]
            elif data.type=='Reference':
                color=REFERENCE_COLOR
                citations=data.citationCount
                size=5*max(0,int(math.log(citations,3)))+10 if citations else 10
            elif data.type=='ReferenceWithCode':
                color=RWC_COLOR
                citations=data.citationCount
                size=5*max(0,int(math.log(citations,3)))+10 if citations else 10
            elif data.type in ['ReferenceCore', 'ReferenceCoreWithTree']:
                color=CORE_COLOR
                citations=data.citationCount
                size=5*max(0,int(math.log(citations,3)))+10 if citations else 10
            else: # VERY SLOW TO LOAD
                if not with_ext: continue
                color=EXT_COLOR_1HOC
                citations=data.citationCount
                size=5*max(0,int(math.log(citations,3)))+10 if citations else 10
                # continue # skip the 1hop reference, too much
            G.add_node(
                node,
                title=data.to_desc(),
                size=size,
                color=color,
                # scale=scale,
                # rating=data.rating
            )
        for edge in self.G.edges:
            if edge[0] in G.nodes and edge[1] in G.nodes:
                G.add_edge(edge[0],edge[1])
        fname='phylogenetic_tree'
        if max_nodes: fname+=f'_{max_nodes}'
        write_dot(G, U.pjoin(self.db_dir, '..', fname+".dot"))
        self.viz(G,max_nodes=max_nodes,height=height,layout=layout)


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
        self.stream = PrintSystem(self._config)
        self.load(**kwargs)

    def load(self,**kwargs):
        # init params, TODO: upgrade to exec_util params, use a simple str params for now
        self.params={}
        for param in self._config.strparams.split(';'):
            key,val=param.split('=')
            self.params[key]=val
        self.stream.write(self.params)

        # set the name and save dir
        self.evoname=self.params['evoname'] # Provide the name for the whole run including evolutions of all scales, all designs, all agents
        if not 'ckpt_dir' in self.params or not self.params['ckpt_dir']:
            self.ckpt_dir=os.environ.get("CKPT_DIR")
        else:
            self.ckpt_dir=self.params['ckpt_dir']
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
        self.save_state() # save the initialized state

        self.scales=[eval(f'GAMConfig_{scale}()') for scale in self.state['scales']]

        self.stream.write(f"Evolution system initialized with scales: {self.state['scales']}")
        self.stream.write(f"Current scale: {self.state['current_scale']}")
        self.stream.write(f"Budgets remaining: {self.state['budgets']}")
        self.stream.write(f"Checkpoint directory: {self.evo_dir}")

        self.rnd_agent = BuildSystem(
            debug_steps=False, # True for debugging, but very long
            # cache_type="diskcache", #<-- agent caching method 
            temperature=0.1,
            jupyter=False,
            lib_dir=U.pjoin(self.evo_dir,'lib'),
            # cache_id=919,
            #from_json='/path/to/config'
            **kwargs
        )
        self.ptree=PhylogeneticTree(U.pjoin(self.evo_dir,'db'))
        # self.ptree.export()

    def link_stream(self,stream):
        self.stream=stream

    def reload(self,config):
        self._config = config
        self.load()

    def query_system(self,
        query: Optional[str] = '',
        stream: Optional[ModuleType] = None,
        frontend: Optional[bool] = False,
        status: Optional[bool] = True,
        **kwargs
    ) -> list:
        """ Talk to the selector agent """
        
        self.stream.write("Hello from the evolution system")


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
            self.stream.write('\n\n'+'+'*50)
            self.stream.write("RUNNING IN EVOLUTION MODE...\n\n")
            ret=self.evolve()
            if ret is None:
                self.stream.write("No budget left for the evolution")
                time.sleep(1)
            else:
                self.stream.write(f"Design {ret} sampled")
        elif mode=='verify':
            self.stream.write('\n\n'+'x'*50)
            self.stream.write("RUNNING IN VERIFICATION MODE...\n\n")
            ret=self.verify()
            if ret is None:
                self.stream.write("No unverified design left")
                time.sleep(1)
            else: 
                self.stream.write(f"Design {ret} verified")

    def evolve(self): # run a single evolution step unless no budget left
        scale_id=self.state['current_scale']
        if scale_id>=len(self.state['scales']): # no scale left
            self.stream.write("No scale left")
            return None
        scale=self.state['scales'][scale_id]
        budget=self.state['budgets'][scale]
        if budget==0: 
            self.state['current_scale']+=1 # Will influence the sampling distribution
            self.save_state()
            return self.evolve()
        else:
            return self._evolve(scale_id)
            

    def _evolve(self,scale_id): # do evolve that produce one design and operate the phylogenetic tree
        K=np.random.choice([1,2,3],p=[1,0,0])
        instruct,metadata=self.select(K) # use the seed_ids to record the phylogenetic tree

        artifact=self.sample(scale_id,instruct,metadata) # NOTE: maybe randomly jump up or down to next scale? How to use the budget more wisely?
        if artifact is None:
            self.stream.write("No design sampled")
            return True # no design sampled, continue
        # save the design to the phylogenetic tree and update the budget
        seed=metadata['seed']
        references=metadata['references']
        artifact['seed_ids']=[seed.acronym for seed in seed]
        artifact['references']=[reference.acronym for reference in references]
        self.ptree.new_design(artifact)
        scale=self.state['scales'][scale_id]
        self.state['budgets'][scale]-=1
        # self.state['unverified'].append(artifact['acronym'])
        self.save_state() # NOTE!!!: handle it carefully in multi-threading
        self.ptree.export()
        return artifact['acronym']

    def sample(self,scale_id,instruct,metadata,mode='existing'):
        """ Sample a design at a given scale and verify it """
        self.rnd_agent.set_config(self.scales[scale_id])
        session_id=f'sample_{len(self.ptree.filter_by_type(["DesignArtifact"]))}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
        log_dir=U.pjoin(self.evo_dir,'log',session_id)
        U.mkdir(log_dir)
        response=self.rnd_agent(instruct,log_dir=log_dir,stream=self.stream,metadata=metadata)  # instruct should have a type
        if response is None: # no design sampled
            return None
        title,rawcode,explain,summary,autocfg,reviews,ratings,check_results=response
        for i in [' and ',' for ','-']:
            title=title.replace(i,' ')
        acronym=''.join([i[0].upper() for i in title.split(' ') if i.isalpha()])

        # modify the code to fit the block registry
        # TODO: change the registry name to acronyms
        code=rawcode+f'\n\n\n{autocfg}\nblock_config=gab_config\nblock_config.update(autoconfig)'
        code+='\n\n\nfrom .block_registry import BlockRegister\n\nBlockRegister(\n    name="default",\n    config=block_config\n)(GAB)'

        artifact={
            'title':title,
            'session_id':session_id,
            'acronym':acronym,
            'code':code,
            'rawcode':rawcode,
            'explain':explain,
            'scale':self.state['scales'][scale_id],
            'instruct':instruct,
            'summary':summary,
            'reviews':reviews,
            'ratings':ratings,
            'check_results':check_results,
        }
        return artifact
    

    # TODO: upgrade to Selector agent


    def select(self,K: Union[int,Dict[str,int]],selector_instruct='',mode='existing')->Tuple[str,List[NodeObject]]:
        '''
        K: int or dict of {source_type: num of seeds}, if K is int, then sample from all default sources (the ones with code)
        Return:
            seeds: List[NodeObject]
            instruct: str, the prompt generated from the selector and seeds
        '''
        seeds=[]
        prompt=''
        if isinstance(K,int):
            prompt,seeds=self._select(K,selector_instruct)
        elif isinstance(K,dict):
            for source_type,num in K.items():
                instruct,topk=self._select(num,selector_instruct,source_type)
                prompt+=instruct
                seeds.extend(topk)
        if mode=='existing':
            seed_types = ['DesignArtifact','ReferenceCoreWithTree']
            seed = [i for i in seeds if i.type in seed_types]
            references = [i for i in seeds if i.type not in seed_types]
            assert len(seed)>0, "There must be at least one seed from DesignArtifact or ReferenceCoreWithTree when design from existing"
            if len(seed)>1:
                seed = random.choice(seed) # randomly select for now
                references = [i for i in seeds if i.acronym!=seed.acronym]
            else:
                seed = seeds[0]
        elif mode=='scratch':
            seed_types = ['DesignArtifact','ReferenceCoreWithTree','ReferenceWithCode']
            seed = [i for i in seeds if i.type in seed_types]
            references = [i for i in seeds if i.type not in seed_types]
        metadata={
            'mode': mode,
            'seed': seed,
            'references': references,
        }
        return prompt,metadata

    def _select(self,K: int=1,selector_instruct='',
            filter_type=['DesignArtifact','ReferenceWithCode','ReferenceCoreWithTree','ReferenceCore'])-> Tuple[str,List[NodeObject]]: # K is the number of designs to sample, instruct is the instruction to the selector, select seeds or select populations
        """ Provide the instruction including seeds and instructs for the next design """
        K=min(K,len(self.ptree.filter_by_type(filter_type)))
        if K==0: # no design to sample
            return '',[]
        if self.select_method=='heuristic':
            topk = self.heuristic_select(K,selector_instruct,filter_type)
        elif self.select_method=='random':
            topk = self.random_select(K,selector_instruct,filter_type)
        instruct='' # TODO: leave it to the selector agent
        return instruct,topk

    def nodes2data(self,nodes)->List[NodeObject]: # convert the nodes to data: NodeObject
        return [self.ptree.G.nodes[node]['data'] for node in nodes]

    def heuristic_select(self,K: int=1,selector_instruct='',
            filter_type=['DesignArtifact','ReferenceWithCode','ReferenceCoreWithTree','ReferenceCore'])-> List[NodeObject]:
        alpha=0.1
        sample_metrics={}
        sample_scale={}
        for node in self.ptree.filter_by_type(filter_type):
            artifact=self.ptree.G.nodes[node]['data']
            if artifact.verify_report is not None:
                # TODO: upgrade this thing
                report=report_reader(artifact.verify_report)
                sample_metrics[node]=np.mean([v for k,v in report['metrics']['eval'].items() if 'acc' in k])
                scale_id=self.state['scales'].index(artifact.scale)+1
                sample_scale[node]=scale_id

        # TODO: upgrade this thing
        prob=np.array([v for k,v in sample_metrics.items()])*(1-alpha)+np.random.rand(len(sample_metrics))*alpha
        prob=prob*np.array([v for k,v in sample_scale.items()]) # prefer the higher scale
        prob=prob/np.sum(prob)
        topk=np.random.choice(list(sample_metrics.keys()),size=K,replace=False,p=prob)
        return self.nodes2data(topk)

    def random_select(self,K: int=1,selector_instruct='',
            filter_type=['DesignArtifact','ReferenceWithCode','ReferenceCoreWithTree','ReferenceCore']):
        topk=random.sample(self.ptree.filter_by_type(filter_type),K)
        return self.nodes2data(topk)
    
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
        with open('./model/gab.py','w', encoding='utf-8') as f:
            f.write(artifact['code'])
        args = ve_parser.parse_args()
        args.evoname=self.evoname
        args.design_id=artifact['acronym']
        args.scale=artifact["scale"]
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
        stream: Optional[ModuleType] = None,
        **kwargs
    ) -> EvolutionSystem:
    """Factory for loading evolution system 

    :param config: 
        Configuration object (optional) 

    """
    kwargs["system_type"] = "evolution"
    evolution = NativeBuild(config,**kwargs)
    if stream:
        evolution.link_stream(stream)
    return evolution












############################################################################################################

def test_evolve(test_name,step=False):
    strparams=[
        f"evoname={test_name}",
        "scales=14M,31M,70M",
        "selection_ratio=0.25",
        "select_method=random",
    ]
    evolution_system = BuildEvolution(
        strparams=';'.join(strparams),
        do_cache=True,
        cache_type='diskcache',
    )
    while evolution_system.evolve():
        if step:
            break



if __name__ == '__main__':
    strparams=[
        "evoname=evolution_test1",
        "scales=14M,31M,70M",
        "selection_ratio=0.25",
        "select_method=random",
    ]

    args = ve_parser.parse_args()
    strparams.append(f"evoname={args.evoname}")

    # evolution_system = BuildEvolution(
    #     strparams=';'.join(strparams),
    #     do_cache=False,
    #     # cache_type='diskcache',
    # )
    # evolution_system._run(args.mode)

    test_evolve('test_evo_004',step=True)


#     code_MHA='''
# # gab.py

# import torch
# import torch.nn as nn
# from mamba_ssm.modules.mha import MHA

# from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #

# class GAB(GABBase):
#     """Generalized Autoregressive Block
#         Input:        X: (batch, seqlen, embed_dim)
#         Output:       Y: (batch, seqlen, embed_dim)
#         Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
#     """
#     def __init__(self, embed_dim: int, device=None, dtype=None, n_heads=8, ff_dim=None, dropout=0.1): 
#         factory_kwargs = {"device": device, "dtype": dtype} 
#         super().__init__(embed_dim)
        
#         if ff_dim is None:
#             ff_dim = 4 * embed_dim  # Feed-forward dimension is 4 times the embedding dimension
        
#         self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, **factory_kwargs)
#         # self.attn = MHA(embed_dim, n_heads, causal=False, **factory_kwargs)
#         # self.lstm=nn.LSTM(embed_dim, embed_dim, batch_first=True)
#         # self.bilstm=nn.LSTM(embed_dim, embed_dim//2, batch_first=True, bidirectional=True)
#         # self.causalconv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, groups=4, **factory_kwargs)
#         # self.conv = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)

#     def _forward(self, X, **kwargs): 
#         output,_ = self.attention(X, X, X)
#         # mask=nn.Transformer.generate_square_subsequent_mask(len(X)).to(X.device)
#         # output,_ = self.attention(X, X, X, attn_mask=mask)
#         # output = self.attn(X)
#         # output,_ = self.lstm(X)
#         # output,_ = self.bilstm(X)
#         # output = self.causalconv(X.permute(0,2,1)).permute(0,2,1)[:,:-2]
#         # output = self.conv(X.permute(0,2,1)).permute(0,2,1)
#         return output 
    
# gab_config = {
#     'n_heads': 8,
#     'ff_dim': None,  # This will be set to 4 * embed_dim in the GAB class
#     'dropout': 0.1
# }
# '''


    # checker=evolution_system.rnd_agent.checker
    # cfg=evolution_system.rnd_agent._cfg
    # design_name='test_design'

    # code=code_MHA
    # checkpass,check_report,gabcode,check_results = checker.check(cfg,code,design_name)
    # print(check_results)

    # print('Check the second code')
    # code_retnet_dir='/home/junyanc/model_discovery/model_discovery/model/library/base/retnet/retnet_edu.py'
    # code=open(code_retnet_dir,'r').read()
    # checkpass,check_report,gabcode,check_results = checker.check(cfg,code,design_name)

    # print('Check the third code')
    # code=U.read_file('./draft.py')
    # ret=checker.check(cfg,code,design_name)


