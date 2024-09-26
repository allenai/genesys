''' System of R & D Agents and Selector for Scale Climbing Evolution '''

from __future__ import annotations

import os

try: # a stupid patch for windows
    from .secrets import *
    os.environ['MY_OPENAI_KEY']=MY_OPENAI_KEY
    os.environ['ANTHROPIC_API_KEY']=ANTHROPIC_API_KEY
    os.environ['HF_KEY']=HF_KEY
    os.environ['HF_HUB_KEY']=HF_HUB_KEY
    os.environ['GITHUB_TOKEN']=GITHUB_TOKEN
    os.environ['WANDB_API_KEY']=WANDB_API_KEY
    os.environ['S2_API_KEY']=S2_API_KEY
    os.environ['AWS_SECRET_ACCESS_KEY']=AWS_SECRET_ACCESS_KEY
    os.environ['AWS_ACCESS_KEY_ID']=AWS_ACCESS_KEY_ID
    os.environ['MATHPIX_API_ID']=MATHPIX_API_ID
    os.environ['UNSTRUCTURED_API_ID']=UNSTRUCTURED_API_ID
    os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
    os.environ['COHERE_API_KEY']=COHERE_API_KEY
    os.environ['PERPLEXITY_API_KEY']=PERPLEXITY_API_KEY
    os.environ['DATA_DIR']=DATA_DIR
    os.environ['CKPT_DIR']=CKPT_DIR
    os.environ['HF_DATASETS_TRUST_REMOTE_CODE']='1'    
except:
    pass

import sys
import re
import exec_utils
import pathlib
import datetime
import json
import time
import tempfile
from dataclasses import dataclass, field, asdict
import networkx as nx
import pandas as pd
import numpy as np
from io import StringIO
import hashlib
import random
# from networkx.drawing.nx_pydot import to_pydot,warnings
from pyvis.network import Network
import math
import multiprocessing

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
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
from model_discovery.model.library.tester import check_tune
from .system import BuildSystem,PrintSystem,DesignModes,RunningModes
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


LIBRARY_DIR = U.pjoin(os.path.dirname(__file__),'model','library')

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

# from low to high
TARGET_SCALES = ['14M','31M','70M','125M','350M']#,'760M','1300M']


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


######### Library Reference

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
    verifications: Dict[str, Verification] = field(default_factory=dict)

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
            verification_dir=U.pjoin(core_dir,'verifications')
            if U.pexists(verification_dir):
                for scale in os.listdir(verification_dir):
                    report=U.load_json(U.pjoin(verification_dir,scale))
                    scale=scale.split('.')[0]
                    if 'verification_report' in report:
                        self.verifications[scale]=Verification.from_dict(report['verification_report'])
                    else:
                        self.verifications[scale]=Verification(scale=scale,verification_report=report)
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


##### 1-hop reference, low SNR, not used now, most are just about how to use those models to do applications
# @dataclass
# class LibraryReference1hop(LibraryReference):

#     @property
#     def type(self) -> str:
#         # if self.code is not None:
#         #     return 'Reference1hopWithCode'
#         # else:
#         return 'Reference1hop'



######## Design Artifacts


@dataclass
class Proposal:
    selection:str
    modelname:str
    variantname:str
    proposal:str
    review:str
    rating:int
    passed:bool
    suggestions:str
    reflection:str
    changes:str
    ideation:str
    instructions:str
    search_report:str
    search_references:str
    # traces: List[Dict]
    costs: Dict[str, float]
    design_cfg: Dict[str, Any]
    user_input: str
    abstract:str = None
    search_stack: List[str] = None
    review_search_stack: List[str] = None

    def save(self, design_dir: str, name='proposal.json'):
        dict=asdict(self)
        dict['design_cfg']['running_mode']=self.design_cfg['running_mode'].value
        U.save_json(dict, U.pjoin(design_dir, name))

    @classmethod
    def from_dict(cls, dict: Dict):
        dict['design_cfg']['running_mode']=RunningModes(dict['design_cfg']['running_mode'])
        return cls(**dict)

    @classmethod
    def load(cls, design_dir: str, name='proposal.json'):
        if not U.pexists(U.pjoin(design_dir, name)):
            return None
        dict=U.load_json(U.pjoin(design_dir, name))
        dict['design_cfg']['running_mode']=RunningModes(dict['design_cfg']['running_mode'])
        return cls.from_dict(dict)


@dataclass
class ImplementationAttempt:
    status: str
    rounds: int
    costs: Dict[str, float]
    tree: GAUTree
    design_cfg: Dict[str, Any]
    user_input: str

    def to_dict(self):
        dict=asdict(self)
        dict['design_cfg']['running_mode']=self.design_cfg['running_mode'].value
        dict['tree']=self.tree.to_dict()
        return dict

    @classmethod
    def from_dict(cls, dict: Dict):
        dict['design_cfg']['running_mode']=RunningModes(dict['design_cfg']['running_mode'])
        dict['tree']=GAUTree.from_dict(dict['tree'])
        return cls(**dict)

@dataclass
class Implementation:
    status: str # implemented, failed, or unfinished
    implementation: GAUTree
    history: List[ImplementationAttempt]
    # TODO:consider gaudict management

    def save(self, design_dir: str):
        dict=self.to_dict()
        U.save_json(dict, U.pjoin(design_dir, f'implementation.json'))

    def to_dict(self):
        dict=asdict(self)
        dict['implementation']=self.implementation.to_dict()
        dict['history']=[attempt.to_dict() for attempt in self.history]
        return dict

    @classmethod
    def from_dict(cls, dict: Dict):
        for i in range(len(dict['history'])):
            dict['history'][i]['design_cfg']['running_mode']=RunningModes(dict['history'][i]['design_cfg']['running_mode'])
        dict['history']=[ImplementationAttempt.from_dict(attempt) for attempt in dict['history']]
        dict['implementation']=GAUTree.from_dict(dict['implementation'])
        return cls(**dict)

    @classmethod
    def load(cls, design_dir: str):
        if not U.pexists(U.pjoin(design_dir, 'implementation.json')):
            return None
        dict=U.load_json(U.pjoin(design_dir, 'implementation.json'))
        return cls.from_dict(dict)
    
    def get_cost(self):
        costs={}
        for attempt in self.history:
            for k,v in attempt.costs.items():
                if k not in costs:
                    costs[k]=0
                costs[k]+=v
        return costs

@dataclass
class Verification:
    scale: str
    verification_report: Dict

    def save(self, design_dir: str):
        U.mkdir(U.pjoin(design_dir, 'verifications'))
        U.save_json(asdict(self), U.pjoin(design_dir, 'verifications',f'{self.scale}.json'))

    @classmethod
    def from_dict(cls, dict: Dict):
        return cls(**dict)
    
    @classmethod
    def load(cls, dir: str):
        return cls.from_dict(U.load_json(dir))

@dataclass
class DesignArtifact(NodeObject):
    design_id: str # design session id
    proposal: Proposal
    implementation: Implementation = None # find by modelname/id
    verifications: Dict[str, Verification] = field(default_factory=dict) # find by modelname/id
    codes: Dict[str, str] = field(default_factory=dict) # find by modelname/id

    @property
    def type(self) -> str:
        if self.is_implemented():
            return 'DesignArtifactImplemented'
        else:
            return 'DesignArtifact'
            
    @classmethod
    def load(cls, design_dir: str):
        metadata = U.load_json(U.pjoin(design_dir, 'metadata.json'))
        proposal = Proposal.load(design_dir)
        if proposal is None:
            return None
        implementation = Implementation.load(design_dir)
        verifications = {}
        ver_dir=U.pjoin(design_dir,'verifications')
        if U.pexists(ver_dir):
            for scale in os.listdir(ver_dir):
                scale=scale.split('.')[0]
                dir = U.pjoin(ver_dir,f'{scale}.json')
                if U.pexists(dir):
                    verifications[scale] = Verification.load(dir)
        codes = U.load_json(U.pjoin(design_dir, 'codes.json'))
        return cls(proposal=proposal, implementation=implementation, verifications=verifications, codes=codes, **metadata)

    def to_prompt(self, full_code=True):
        prompt=f"""
# Proposal: {self.proposal.modelname}

{self.proposal.proposal}

## Review

{self.proposal.review}

### Rating: {self.proposal.rating} out of 5

### Reviewer Suggestions

{self.proposal.suggestions}
"""
        if self.is_implemented():
            if full_code:
                prompt+=f"""
# Implementation

{self.implementation.implementation.view()}
            """
            else:
                prompt+=f"""
# Implementation

{self.implementation.implementation.root.spec.document}
            """
        for scale in self.verifications:
            pass # TODO

        return prompt


    def to_desc(self):
        mdtext = f'## {self.proposal.modelname}'
        mdtext += f'\n**Selection:** {self.proposal.selection}'
        mdtext += f'\n**Model:** {self.proposal.modelname}'
        mdtext += f'\n**Variant:** {self.proposal.variantname}'
        mdtext += f'\n**Rating:** {self.proposal.rating}/5'
        mdtext += f'\n**Passed:** {self.proposal.passed}'
        if self.is_implemented():
            mdtext += f'\n**Implementation:**\n\n{self.implementation.implementation.root.spec.document}'
        return mdtext.replace(':', ' ').replace('e.\ng.\n', 'e.g.').replace('i.\ne.\n', 'i.e.')

    def is_implemented(self):
        return self.implementation is not None and self.implementation.status=='implemented'

    def get_cost(self):
        costs=self.proposal.costs
        if self.implementation:
            icosts=self.implementation.get_cost()
            costs={k:v+icosts[k] for k,v in costs.items()}
        # TODO: maybe rerank, selection, etc. cost
        return costs


# def write_dot(G, path):
#     """Write NetworkX graph G to Graphviz dot format on path.

#     Path can be a string or a file handle.
#     """
#     msg = (
#         "nx.nx_pydot.write_dot depends on the pydot package, which has"
#         "known issues and is not actively maintained. Consider using"
#         "nx.nx_agraph.write_dot instead.\n\n"
#         "See https://github.com/networkx/networkx/issues/5723"
#     )
#     warnings.warn(msg, PendingDeprecationWarning, stacklevel=2)
#     P = to_pydot(G)
#     with open(path, "w", encoding='utf-8') as f:
#         f.write(P.to_string())
#     return


class PhylogeneticTree: ## TODO: remove redundant edges and reference nodes
    # Read from a design base and construct a phylogenetic tree
    """
    Physical structure:
    db_dir
    ├── designs
    │   ├── acronym
    |   |   ├── metadata.json
    │   │   ├── proposal.json
    │   │   ├── implementation.json
    │   │   ├── verifications
    │   │   │   ├── scale.json
    │   │   │   └── ...
    │   └── ...
    ├── sessions
    │   ├── design_id
    │   │   ├── metadata.json
    │   │   ├── log
    │   │   │   ├── stream.log
    │   │   │   └── ...
    │   └── ...
    | ... # units, etc.
    """
    def __init__(self, db_dir: str, db_only=False):
        self.db_dir = db_dir
        self.lib_dir = U.pjoin(LIBRARY_DIR,'tree')
        self.lib_ext_dir = U.pjoin(LIBRARY_DIR,'tree_ext')
        self.design_sessions = {}
        U.mkdir(db_dir)
        U.mkdir(U.pjoin(db_dir,'designs'))
        U.mkdir(U.pjoin(db_dir,'sessions'))
        self.db_only=db_only
        self.load()

    # new design: proposal -> implement -> verify

    # def reload(self): # why do we need this at all??
    #     self.load()

    def get_nodes(self,acronyms):
        if isinstance(acronyms,str):
            acronyms=[acronyms]
        return [self.get_node(acronym) for acronym in acronyms]
    
    def get_abstracts(self,acronyms):
        abstracts=[]
        reviews=[]
        ratings=[]
        for acronym in acronyms:
            proposal=self.get_node(acronym).proposal
            if proposal.abstract:
                abstracts.append(proposal.abstract)
            else:
                abstracts.append(proposal.proposal[:1600]+'...') # ~400 tokens
            reviews.append(proposal.review)
            ratings.append(proposal.rating)
        return abstracts,reviews,ratings
    
    def find_sibling_designs(self,parents):
        if isinstance(parents,str):
            parents=[parents]
        siblings=[]
        for acronym in self.filter_by_type(['DesignArtifact','DesignArtifactImplemented']):
            design=self.get_node(acronym)
            if any([p in design.seed_ids for p in parents]):
                siblings.append(acronym)
        return siblings

    def load_design_sessions(self):
        self.design_sessions={}
        for design_id in os.listdir(U.pjoin(self.db_dir,'sessions')):
            metadata = U.load_json(U.pjoin(self.session_dir(design_id), 'metadata.json'))
            metadata['mode']=DesignModes(metadata['mode'])
            self.design_sessions[design_id] = metadata

    @property
    def design_cost(self):
        costs=0
        designs=self.filter_by_type('DesignArtifact')
        for design in designs:
            costs+=sum(self.get_node(design).get_cost().values())
        return costs

    # How to handle variants? i.e., in GPT, there are optional pre-conv and post-conv, maybe just all of them to the tree, let selector to choose

    def new_design(self, seed_ids, ref_ids, instruct, num_samples, mode=None): # new design session, a session explore the steps from a selected node
        # generate unique hash for the design, do not consider the order
        # design_id = hashlib.sha256(f"{sorted(ref_ids)}{sorted(seed_ids)}{instruct}{mode}".encode()).hexdigest()
        if mode is None:
            mode=DesignModes.MUTATION
        hash_tail=hashlib.sha256(f"{sorted(ref_ids)}{sorted(seed_ids)}{instruct}{mode}".encode()).hexdigest()
        design_id = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{hash_tail[-6:]}"
        sessdata = {
            'seed_ids': seed_ids,
            'ref_ids': ref_ids,
            'instruct': instruct,
            'mode': mode,
            'proposed': [],
            'reranked': {},
            'num_samples': num_samples
        }
        self.design_sessions[design_id] = sessdata
        sess_dir=self.session_dir(design_id)
        U.mkdir(sess_dir)
        self.save_session(design_id)
        U.mkdir(U.pjoin(sess_dir, 'log'))
        return design_id
    
    
    def get_unverified_designs(self,scale):
        unverified=[]
        for acronym in self.filter_by_type('DesignArtifactImplemented'):
            design=self.get_node(acronym)
            if scale not in design.verifications:
                unverified.append(acronym)
        return unverified

        
    def get_unverified_scales(self,acronym): # from low to high
        unverified=[]
        design=self.get_node(acronym)
        for scale in TARGET_SCALES:
            if scale not in design.verifications:
                unverified.append(scale)
        unverified.sort(key=lambda x: int(x.replace('M','')))
        return unverified

    def get_gau_tree(self,acronym:str):
        node=self.get_node(acronym)
        if node.type=='ReferenceCoreWithTree':
            tree=node.tree
            return tree
        elif node.type=='DesignArtifactImplemented':
            tree=node.implementation.implementation
            return tree
        else:
            return None 
    
    def get_session_input(self,design_id:str):
        sessdata=self.design_sessions[design_id]
        seeds=[self.get_node(seed_id) for seed_id in sessdata['seed_ids']]
        refs=[self.get_node(ref_id) for ref_id in sessdata['ref_ids']]
        return seeds,refs,sessdata['instruct']
    
    def session_dir(self, design_id: str):
        sess_dir=U.pjoin(self.db_dir, 'sessions', design_id)
        U.mkdir(sess_dir)
        return sess_dir
    
    def session_get(self,design_id:str,key:str):
        return self.design_sessions[design_id].get(key)

    def session_set(self,design_id:str,key:str,value):
        self.design_sessions[design_id][key]=value
        self.save_session(design_id)
    
    def design_dir(self, acronym: str):
        design_dir=U.pjoin(self.db_dir, 'designs', acronym)
        U.mkdir(design_dir)
        return design_dir
    
    def coreref_dir(self, acronym: str):
        coreref_dir=U.pjoin(LIBRARY_DIR,'core',acronym,'reports')
        U.mkdir(coreref_dir)
        return coreref_dir
    
    def get_node(self, acronym: str):
        return self.G.nodes[acronym]['data']
    
    def get_unfinished_designs(self):
        unfinished_designs = []
        for design_id in self.design_sessions:
            sessdata=self.design_sessions[design_id]
            num_samples=sessdata['num_samples']
            passed,_ = self.session_proposals(design_id,passed_only=True)
            implemented,_ = self.session_implementations(design_id,implemented_only=True)
            # print(f"Design {design_id} has {len(passed)}/{num_samples['proposal']} proposals and {len(implemented)}/{num_samples['implementation']} implementations.")
            if len(passed)<num_samples['proposal'] or \
                len(implemented)<num_samples['implementation']:
                unfinished_designs.append(design_id)
        return unfinished_designs

    def get_implementation_checkpoint(self,acronym:str):
        design=self.get_node(acronym)
        if design.implementation:
            return design.implementation.implementation
        else:
            return None
    
    def session_proposals(self,design_id:str,passed_only=False):
        sessdata=self.design_sessions[design_id]
        acronyms=[]
        proposals=[]
        for acronym in sessdata['proposed']:
            design=self.get_node(acronym)
            if passed_only and not design.proposal.passed:
                continue
            proposals.append(design.proposal)
            acronyms.append(acronym)
        return proposals,acronyms

    
    def session_implementations(self,design_id:str,implemented_only=False):
        sessdata=self.design_sessions[design_id]
        acronyms=[]
        implementations=[]
        for acronym in sessdata['proposed']:
            design=self.get_node(acronym)
            if design.implementation:
                if implemented_only and design.implementation.status!='implemented':
                    continue
                implementations.append(design.implementation)
                acronyms.append(acronym)
        return implementations,acronyms

    def propose(self, design_id: str, proposal,proposal_traces,costs,design_cfg,user_input): # create a new design artifact
        sessdata=self.design_sessions[design_id]
        seeds=sessdata['seed_ids']
        proposal['costs']=costs
        proposal['design_cfg']=design_cfg
        proposal['user_input']=user_input
        proposal = Proposal(**proposal)
        title = f'{proposal.modelname}: Refinement of {seeds} by improving {proposal.selection}'
        for line in proposal.proposal.split("\n"):
            if line.startswith("# "):
                title = line[2:]
                break
        acronym = self.unique_acronym(proposal.modelname.replace(' ', '_').lower())
        proposal.modelname = acronym
        metadata = {'design_id': design_id, 'acronym': acronym, 'seed_ids': seeds, 'title': title}
        U.save_json(metadata, U.pjoin(self.design_dir(acronym), 'metadata.json'))
        proposal.save(self.design_dir(acronym))
        traces_dir=U.pjoin(self.design_dir(acronym),'proposal_traces')
        for idx,trace in enumerate(proposal_traces):
            U.mkdir(traces_dir)
            trace['costs']=costs
            trace['design_cfg']=design_cfg
            trace['user_input']=user_input
            proposal_trace=Proposal(**trace)
            proposal_trace.save(traces_dir,f'trace_{idx}.json')
        design_artifact = DesignArtifact(design_id=design_id, acronym=acronym, seed_ids=seeds, title=title, proposal=proposal)
        self.G.add_node(acronym, data=design_artifact)
        self.design_sessions[design_id]['proposed'].append(acronym)
        self.save_session(design_id)

    def save_session(self,design_id: str):
        sessdata=self.design_sessions[design_id]        
        try:
            sessdata['mode']=DesignModes(sessdata['mode'])
        except:
            pass
        sessdata['mode']=sessdata['mode'].value
        U.save_json(sessdata, U.pjoin(self.session_dir(design_id), 'metadata.json'))
        sessdata['mode']=DesignModes(sessdata['mode'])

    def implement(self, acronym: str, tree,ROUNDS,status,costs,design_cfg,user_input): # update a proposal node with implementation
        design_artifact=self.get_node(acronym)
        implementation=design_artifact.implementation
        attempt=ImplementationAttempt(status=status, rounds=ROUNDS, costs=costs, tree=tree, design_cfg=design_cfg, user_input=user_input)
        if implementation is None:
            implementation=Implementation(status=status, implementation=tree, history=[attempt])
        else:
            implementation.status=status
            implementation.implementation=tree
            implementation.history.append(attempt)
        implementation.save(self.design_dir(acronym))
        design_artifact.implementation=implementation
        self.G.nodes[acronym]['data']=design_artifact
        # Tune in all target scales
        if status=='implemented':
            codes = {}
            _code = tree.compose()
            for scale in TARGET_SCALES:
                codes[scale] = check_tune(scale,acronym, code=_code,check_only=True,cpu_only=True,reformat_only=True)
        U.save_json(codes, U.pjoin(self.design_dir(acronym), 'codes.json'))

    def verify(self, acronym: str, scale: str, verification_report): # attach a verification report under a scale to an implemented node
        design_artifact=self.get_node(acronym)
        verification=Verification(scale=scale, verification_report=verification_report)
        design_artifact.verifications[scale]=verification
        self.G.nodes[acronym]['data']=design_artifact
        if design_artifact.type=='DesignArtifactImplemented':
            verification.save(self.design_dir(acronym))
        else:
            verification.save(self.coreref_dir(acronym))

        
    def unique_acronym(self, acronym: str) -> str:
        existing_acronyms = set(self.G.nodes)
        if acronym not in existing_acronyms:
            return acronym
        i = 1
        while f"{acronym}_{i}" in existing_acronyms:
            i += 1
        return f"{acronym}_{i}"

    def filter_by_type(self,types):
        if isinstance(types, str):
            types=[types]
        nodes=[]
        for node in self.G.nodes:
            if self.G.nodes[node]['data'].type in types:
                nodes.append(node)
        return nodes
    
    def remove_redundant_edges(self,G):
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
        
        return G

    def load(self):
        self.G=self.load_graph()
        if not self.db_only:
            self.load_design_sessions()

    def load_graph(self,max_nodes=None):
        edges_to_add = []
        count=0
        G=nx.DiGraph()
        for id in os.listdir(U.pjoin(self.db_dir,'designs')):
            if max_nodes and count>max_nodes:
                break
            artifact = DesignArtifact.load(self.design_dir(id))
            G.add_node(artifact.acronym, data=artifact)
            count+=1
            for seed_id in artifact.seed_ids:
                edges_to_add.append((seed_id, artifact.acronym))

        # Load core library
        for id in os.listdir(self.lib_dir):
            if max_nodes and count>max_nodes:
                break
            id=id.split('.')[0]
            ref = LibraryReference.load(self.lib_dir, id)
            G.add_node(ref.acronym, data=ref)
            count+=1
            for seed_id in ref.seed_ids:
                edges_to_add.append((seed_id, ref.acronym))
         
        if self.db_only:
            return G

        # # load extended library
        # dir_ext_1hop = U.pjoin(self.lib_ext_dir,'1hop')
        # for i in os.listdir(dir_ext_1hop):
        #     id=i.split('.')[0]
        #     ref = LibraryReference1hop.load(dir_ext_1hop, id)
        #     self.G.add_node(ref.acronym, data=ref)
        #     for seed_id in ref.seed_ids:
        #         edges_to_add.append((seed_id, ref.acronym))
        
        for seed_id, product_id in edges_to_add:
            if seed_id not in G.nodes or product_id not in G.nodes:
                continue
            if seed_id == product_id or nx.has_path(G, product_id, seed_id):
                continue
            G.add_edge(seed_id, product_id)
        
        G=self.remove_redundant_edges(G)

        return G

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

    def export(self,max_nodes=None,height=5000,layout=False): #,with_ext=False
        G=nx.DiGraph()
        if not max_nodes or max_nodes==0 or max_nodes>=len(self.G.nodes):
            _G=self.G.copy()
        else:
            _G=self.load_graph(max_nodes)
        for idx,node in enumerate(_G.nodes):
            if max_nodes and idx>max_nodes:
                break
            data=_G.nodes[node]['data']
            if data.type in ['DesignArtifact','DesignArtifactImplemented']:
                scale='31M'   #data.scale # TODO: use the actual scale
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
            # else: # VERY SLOW TO LOAD
            #     if not with_ext: continue
            #     color=EXT_COLOR_1HOC
            #     citations=data.citationCount
            #     size=5*max(0,int(math.log(citations,3)))+10 if citations else 10
            #     # continue # skip the 1hop reference, too much
            G.add_node(
                node,
                title=data.to_desc(),
                size=size,
                color=color,
                # scale=scale,
                # rating=data.rating
            )
        for edge in _G.edges:
            if edge[0] in G.nodes and edge[1] in G.nodes:
                G.add_edge(edge[0],edge[1])
        fname='phylogenetic_tree'
        if max_nodes: fname+=f'_{max_nodes}'
        # write_dot(G, U.pjoin(self.db_dir, '..', fname+".dot"))
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





def _verify(evoname,design_id,scale,resume=True, mult=20): # do a single verify
    args = ve_parser.parse_args()
    args.evoname=evoname
    args.design_id=design_id+f'_{scale}'
    args.scale=scale
    args.ckpt_dir=os.environ.get("CKPT_DIR")
    args.data_dir=os.environ.get("DATA_DIR")
    args.resume=resume
    args.training_token_multiplier=mult
    ve_main(args)




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
        self.params=config.params
        self.stream = PrintSystem(config)
        self.load(**kwargs)

    def load(self,**kwargs):
        # # init params, TODO: upgrade to exec_util params, use a simple str params for now

        # set the name and save dir
        self.evoname=self.params['evoname'] # Provide the name for the whole run including evolutions of all scales, all designs, all agents
        self.ckpt_dir=os.environ.get("CKPT_DIR")
        self.evo_dir=U.pjoin(self.ckpt_dir,self.evoname)
        U.mkdir(self.evo_dir)

        # load or init the state, if it is an existing evolution, load the state, otherwise init the state
        self.state=self.load_state() # load the state by evoname
        if 'select_method' not in self.params:
            self.params['select_method']='random'
        if 'select_method' not in self.state:
            self.state['select_method']=self.params['select_method']
        self.select_method=self.state['select_method']

        # action choose strategy
        if 'action_strategy' not in self.params:
            self.params['action_strategy']='random'
        if 'action_strategy' not in self.state:
            self.state['action_strategy']=self.params['action_strategy']
        self.action_strategy=self.state['action_strategy']

        
        # design verify strategy
        if 'verify_strategy' not in self.params:
            self.params['verify_strategy']='random'
        if 'verify_strategy' not in self.state:
            self.state['verify_strategy']=self.params['verify_strategy']
        self.verify_strategy=self.state['verify_strategy']

        if 'design_budget' not in self.params:
            self.params['design_budget']=0
        if 'design_budget' not in self.state:
            self.state['design_budget']=int(self.params['design_budget'])
        self.design_budget_limit=self.state['design_budget']

        if 'selection_ratio' not in self.params:
            self.params['selection_ratio']=0.25
        if 'selection_ratio' not in self.state:
            self.state['selection_ratio']=float(self.params['selection_ratio'])
        
        if 'scales' not in self.params:
            self.params['scales']='14M,31M,70M,125M,350M'
        if 'budgets' not in self.state: # remaining budget for each scale
            scales=self.params['scales'].split(',') # e.g. "14M,31M,70M,125M", scales
            budget=1
            self.state['budgets']={}    
            for scale in scales[::-1]:
                self.state['budgets'][scale]=int(np.ceil(budget))
                budget/=self.state['selection_ratio']
        scales=list(self.state['budgets'].keys())

        if 'no_agent' not in self.params:
            self.params['no_agent']=False
        if 'db_only' not in self.params:
            self.params['db_only']=False

        self.save_state() # save the initialized state

        self.scales=[eval(f'GAMConfig_{scale}()') for scale in scales]

        self.stream.write(f"Evolution system initialized with scales: {scales}")
        self.stream.write(f"Budgets remaining: {self.state['budgets']}")
        self.stream.write(f"Checkpoint directory: {self.evo_dir}")

        self.ptree=PhylogeneticTree(U.pjoin(self.evo_dir,'db'),self.params['db_only'])
        print(f"Phylogenetic tree loaded with {len(self.ptree.G.nodes)} nodes and {len(self.ptree.design_sessions)} design sessions from {self.ptree.db_dir}.")

        if self.params['no_agent']:
            self.rnd_agent = None
        else:
            self.rnd_agent = BuildSystem(
                debug_steps=False, # True for debugging, but very long
                # cache_type="diskcache", #<-- agent caching method 
                temperature=0.1,
                jupyter=False,
                # cache_id=919,
                #from_json='/path/to/config'
                **kwargs
            )
            self.rnd_agent.bind_ptree(self.ptree,self.stream)
            # self.ptree.export()

    def link_stream(self,stream):
        self.stream=stream
        self.rnd_agent.sss.stream=stream

    def reload(self,params=None):
        if params:
            self.params = params
            self._config.params = params
        self.load()
    
    def switch_ckpt(self,ckpt_name):
        self.reload({'evoname':ckpt_name})

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


    def check_budget(self,action):
        if action=='design': # check the design budget
            if self.design_budget_limit>0 and self.ptree.design_cost>=self.design_budget_limit:
                return False
        elif action=='verify':
            if sum(self.state['budgets'].values())<=0:
                return False
        else:
            raise ValueError(f"Invalid action: {action}")
        return True
        
    def evolve(self): # each time do one step, agent choose what to do, use shell to run it continuously 
        if not self.check_budget('design') and not self.check_budget('verify'):
            self.stream.write(f"No budget for design and verify, evolution stops, system sleeps, please terminate manually")
            time.sleep(1e9)
            return
        act=self.choose()
        if not self.check_budget(act):
            self.stream.write(f"No budget for {act}, will do another action")
            act='design' if act=='verify' else 'verify'

        act='verify' # XXX: for testing

        if act=='design':
            self.design() # FIXME: it needs to be updated with actual selector and design cfg
        elif act=='verify':
            self.verify()

    # TODO: the interface should be updated when selector agent is ready, and design cfg is ready
    def design(self,n_sources=None,design_cfg={},search_cfg={},user_input='',design_id=None,mode=None,resume=True): # select then sample, TODO: n_sources and design_cfg should be configed
        # user_input and design_cfg maybe changed by the user, so we need to pass them in
        # self.ptree.reload() # WHY WE NEED THIS???
        if mode is None:
            mode=DesignModes.MUTATION
        unfinished_designs = self.ptree.get_unfinished_designs()
        self.stream.write(f"Found {len(unfinished_designs)} unfinished designs, allow resume: {resume}")
        if n_sources is None:
            n_sources = {
                'ReferenceCoreWithTree':1,
                # 'DesignArtifact':1,
                'ReferenceWithCode':2,
                'Reference':2,
            }
        if design_id is None:
            if len(unfinished_designs)==0 or not resume:
                instruct,seed,refs=self.select(n_sources,mode=mode) # use the seed_ids to record the phylogenetic tree
                self.sample(instruct,seed,refs,mode=mode,user_input=user_input,design_cfg=design_cfg,search_cfg=search_cfg)
            else:
                design_id = random.choice(unfinished_designs)
                mode=DesignModes(self.ptree.session_get(design_id,'mode'))
                self.stream.write(f"Restoring a session {design_id}, mode: {mode}.")
                self.sample(design_id=design_id,user_input=user_input,design_cfg=design_cfg,mode=mode,search_cfg=search_cfg) # should not change the design_cfg
        else:
            mode=DesignModes(self.ptree.session_get(design_id,'mode'))
            self.stream.write(f"Design id provided, will restore session {design_id}, mode: {mode}")
            self.sample(design_id=design_id,user_input=user_input,design_cfg=design_cfg,mode=mode,search_cfg=search_cfg)

    def sample(self,instruct=None,seed:List[NodeObject]=None,refs:List[NodeObject]=None,design_id=None,mode=None,user_input='',design_cfg={},search_cfg={}):
        """ 
        Sample a design at a given scale and verify it 
        
        Input optional selector instruct and metadata, return a design artifact
        DB should be fully managed by the agent system, likewise, VE should be fully managed by the verification engine
        agent system should only have access to DB, and VE should only have access to VE

        Selector choose which seeds to use, and budget for verification
        Given the seeds which direct the global direction, the agent system should be fully responsible for the best local move
        """
        if mode is None:
            mode=DesignModes.MUTATION

        self.rnd_agent(
            user_input,
            instruct=instruct,
            seed=seed,
            refs=refs,
            design_id=design_id,
            stream=self.stream,
            design_cfg=design_cfg,
            search_cfg=search_cfg,
            mode=mode
        )


    # TODO: upgrade to Selector agent
    def select(self,K: Union[int,Dict[str,int]],selector_instruct='',mode=None)->Tuple[str,List[NodeObject]]:
        '''
        K: int or dict of {source_type: num of seeds}, if K is int, then sample from all default sources (the ones with code)
        Return:
            seeds: List[NodeObject]
            instruct: str, the prompt generated from the selector and seeds
        '''
        if mode is None:
            mode=DesignModes.MUTATION
        seeds=[]
        instruct=''
        if isinstance(K,int):
            instruct,seeds=self._select(K,selector_instruct)
        elif isinstance(K,dict):
            for source_type,num in K.items():
                _instruct,topk=self._select(num,selector_instruct,source_type)
                instruct+=_instruct # NOTE: should not like this
                seeds.extend(topk)
        if mode==DesignModes.MUTATION:
            seed_types = ['DesignArtifactImplemented','ReferenceCoreWithTree']
            seed = [i for i in seeds if i.type in seed_types] # NOTE: need improve 
            refs = [i for i in seeds if i.type not in seed_types]
            assert len(seed)>0, "There must be at least one seed from DesignArtifact or ReferenceCoreWithTree when design from existing"
            if len(seed)>1:
                seed = [random.choice(seed)] # NOTE: randomly select for now, should not happen at all
                # refs = [i for i in seeds if i.acronym!=seed[0].acronym]
            else:
                seed = [seeds[0]]
        elif mode==DesignModes.SCRATCH:
            seed_types = ['DesignArtifactImplemented','ReferenceCoreWithTree','ReferenceWithCode']
            seed = [i for i in seeds if i.type in seed_types]
            refs = [i for i in seeds if i.type not in seed_types]
        elif mode==DesignModes.CROSSOVER:
            seed_types = ['DesignArtifactImplemented','ReferenceCoreWithTree']
            seeds = [i for i in seeds if i.type in seed_types]
            assert len(seeds)>=2, "There must be at least two seeds from DesignArtifactImplemented and ReferenceCoreWithTree when design from existing"
            seed = random.sample(seeds,2)
            refs = [i for i in seeds if i not in seed]
        return instruct,seed,refs

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
        raise NotImplementedError
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
    
    def choose(self):
        """ Choose a move, select a design to verify """
        if self.action_strategy=='random':
            return random.choice(['design','verify'])
        # TODO: max parallel sampling efficiency strategy
        else:
            raise ValueError(f"Invalid action choose strategy: {self.action_strategy}")


    def verify(self,design_id=None,scale=None,resume=True): # choose then verify
        if design_id is None:
            design_id,scale=self._select_verify_design()
        if design_id is None:
            return None
        self.stream.write(f"Verifying design {design_id} at scale {scale}...")
        mult=self.get_train_budget(self.ptree.get_node(design_id))
        _verify(self.evoname,design_id,scale,resume=resume, mult=mult) # verify the design until it's done
        report_dir=U.pjoin(self.evo_dir,'ve',design_id+f'_{scale}','report.json')
        report=U.load_json(report_dir)
        self.ptree.verify(design_id,scale,report)
        if report!={}: 
            return 'SUCCESS'
        return 'FAILED'

    def _select_verify_design(self):
        TARGET_SCALES.sort(key=lambda x: int(x.replace('M','')))
        if self.verify_strategy=='random':
            for scale in TARGET_SCALES:
                if self.state['budgets'][scale]==0:
                    self.stream.write(f"No budget for design verify at scale {scale}.")
                    continue
                unverified=self.ptree.get_unverified_designs(scale)
                if len(unverified)==0:
                    self.stream.write(f"No unverified design at scale {scale}.")
                else:
                    design_id=random.choice(unverified)
                    return design_id,scale
            self.stream.write(f"No unverified design found at any scale.")
            return None,None
        else:
            raise ValueError(f"Invalid design verify strategy: {self.verify_strategy}")
    
    def _prep_verify(self,design_id,scale):
        design=self.ptree.get_node(design_id) # need to ensure this design has not been verified under scale
        ### XXX need manully check then comment it, need to fix, TUNE cause the problem
        if design.type=='DesignArtifactImplemented':
            _code = design.implementation.implementation.compose()
        else:
            code_dir=U.pjoin(LIBRARY_DIR,'core',design_id,design_id+'.py')
            if U.pexists(code_dir):
                _code = U.read_file(code_dir)
            else:
                raise FileNotFoundError(f"Code file not found for design {design_id}")
        code = check_tune(scale,design_id, code=_code,check_only=True,cpu_only=True,reformat_only=True)
        with open('./model_discovery/model/gab.py','w', encoding='utf-8') as f:
            f.write(code)
        return self.get_train_budget(design)


    def get_train_budget(self,artifact): # dynamic budget 
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
    params={
        'evoname':test_name,
        'scales':'14M,31M,70M',
        'selection_ratio':0.25,
        'select_method':'random',
        'design_budget':0,
    }
    evolution_system = BuildEvolution(
        params=params,
        do_cache=True,
        cache_type='diskcache',
    )
    while evolution_system.evolve():
        if step:
            break



if __name__ == '__main__':
    args = ve_parser.parse_args()
    # print('*'*20)
    # print('Parsed args:',args)
    # print('*'*20)
    if args.mode=='test':
        params={
            'evoname':'evolution_test1',
            'scales':'14M,31M,70M',
            'selection_ratio':0.25,
            'select_method':'random',
            'design_budget':0,
        }
        params['evoname']=args.evoname
        args.evoname=params['evoname']

        test_evolve('test_evo_000',step=True)
    else:
        params=json.loads(args.params)
        # print(f'Running with params:\n{params}')
        if args.mode=='verify':
            _verify(args.evoname,args.design_id, args.scale, resume=args.resume)
        else:
            if args.mode=='prep_verify':
                params['no_agent']=True
                params['db_only']=True
            evolution_system = BuildEvolution(
                params=params,
                do_cache=False,
                # cache_type='diskcache',
            )
            if args.mode=='prep_verify':
                evolution_system._prep_verify(args.design_id, args.scale)
            elif args.mode=='design':
                pass
                # evolution_system.design(n_sources,design_cfg,search_cfg,user_input,design_id,mode,resume)
            elif args.mode=='evolve':
                # evolution_system.evolve()
                pass
            else:
                raise ValueError(f"Invalid mode: {args.mode}")
