# Translate GAUBases to the executable GAB
import os
from typing import List, Dict

from dataclasses import dataclass, field
from .utils.modules import GAUBase

import model_discovery.utils as U



ROOT_UNIT_TEMPLATE = '''# UNIT_NAME.py

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GAUBase # DO NOT CHANGE THIS IMPORT STATEMENT #


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #


class UNIT_NAME(GAUBase): 
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim), Z: {dict of all current intermediate variables}
        Output:       Y: (batch, seqlen, embed_dim), Z_: Optional, {dict of *new* intermediate variables to update the current Z}
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """
    def __init__(self, embed_dim: int, device=None, dtype=None,**kwargs): # YOU CAN ADD MORE ARGUMENTS, BUT YOU HAVE TO HAVE embed_dim, device, dtype AS THE ARGUTMENTS #
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to all nn layers
        super().__init__(embed_dim) # DO NOT CHANGE THIS LINE #
        
        # COMPLETING THE CODE HERE #


    # YOU CAN ADD MORE FUNCTIONS HERE #


    def _forward(self, X, **Z): 

        # COMPLETING THE CODE HERE #
        
        return X
'''



@dataclass
class GAUNode: # this is mainly used to 1. track the hierarchies 2. used for the Linker to solve the dependencies 3. fully serialize the GAUTree
    name: str # name of the GAU 
    code: str # code of the GAU
    args: dict # *new* args and default values of the GAU
    desc: str
    path: List[str] # execution path of the GAU children objects
    review: str 
    rating: str
    report: str # report of checks and tests
    children: Dict[str,str] # children of the GAU, key is the object name, value is the unit name
    suggestions: str # suggestions for the GAU from reviewer for further improvement


    def save(self, dir):
        data={
            'name':self.name,
            'code':self.code,
            'args':self.args,
            'desc':self.desc,
            'path':self.path,
            'review':self.review,
            'rating':self.rating,
            'report':self.report,
            'children':self.children,
            'suggestions':self.suggestions
        }
        U.save_json(data,U.pjoin(dir,f'{self.name}.json'))

    @classmethod
    def load(cls, name, dir):
        data=U.load_json(U.pjoin(dir,f'{name}.json'))
        return cls(**data)



class GAUDict: # GAU code book, registry of GAUs, shared by a whole evolution
    def __init__(self, lib_dir=None):
        self.units = {}
        self.units_dir = U.pjoin(lib_dir, 'units')
        U.mkdir(self.units_dir)
        self.load()

    def exist(self, name):
        return name in self.units   

    def load(self):
        for dir in os.listdir(self.units_dir):
            name=dir.split('.')[0]
            dir=U.pjoin(self.units_dir,f'{name}.py')
            self.units[name]=GAUNode.load(name,self.units_dir)

    def register(self, unit: GAUNode):
        name = unit.name
        assert name not in self.units, f"Unit {name} is already registered" # never overwrite for backward compatibility
        self.units[name] = unit
        unit.save(self.units_dir)
    
    def get(self, unit_name):
        if unit_name not in self.units:
            return None
        return self.units[unit_name]


def check_tree_name(name, lib_dir):
    assert lib_dir is not None, "Please provide the database directory"
    flows_dir = U.pjoin(lib_dir, 'flows')
    U.mkdir(flows_dir)
    existing_trees=[f.split('.')[0] for f in os.listdir(flows_dir)]
    assert name not in existing_trees, f"Tree {name} is already in the database"


# ideally the GAUTree is similar to a pseudo-code
class GAUTree:
    def __init__(self, name, proposal, review, rating, suggestions, lib_dir=None):
        self.units = {} 
        self.root = None
        self.name = name # name of a design 
        self.proposal = proposal # proposal of the design
        self.review = review # review of the design
        self.rating = rating
        self.suggestions = suggestions
        self.dict = GAUDict(lib_dir)
        self.flows_dir = U.pjoin(lib_dir, 'flows')
        U.mkdir(self.flows_dir)

    def add_unit(self, name, code, args, desc, path, review, rating, report, children, suggestions):
        assert name not in self.units, f"Unit {name} is already in the tree"
        assert not self.dict.exist(name), f"Unit {name} is already registered"
        self.units[name] = GAUNode(name, code, args, desc, path, review, rating, report, children, suggestions)
        if len(self.units)==0:
            self.root = self.units[name]

    def del_unit(self, name):
        assert name in self.units, f"Unit {name} is not in the tree"
        del self.units[name]

    def register_unit(self, name): # permanently register a unit to the GAUDict, do it only when the unit is fully tested
        assert name in self.units, f"Unit {name} is not in the tree"
        self.dict.register(self.units[name])
    
    def save(self): # save the Tree only when the design is finalized and fully tested
        dir=U.pjoin(self.flows_dir,f'{self.name}.json')
        data = {
            'name':self.name,
            'root':self.root.name,
            'units':list(self.units.keys()),
            'proposal':self.proposal,
            'review':self.review,
            'rating':self.rating,
            'suggestions':self.suggestions
        }
        U.save_json(data,dir)
        for unit in self.units.values(): # Do not overwrite by default, which should be done by the design process
            if not self.dict.exist(unit.name):
                self.dict.register(unit)

    @classmethod
    def load(cls, name, lib_dir):
        dir = U.pjoin(lib_dir, 'flows', f'{name}.json')
        data = U.load_json(dir)
        tree = cls(data['name'], data['proposal'], data['review'], data['rating'], data['suggestions'], lib_dir)
        for unit_name in data['units']:
            tree.units[unit_name] = tree.dict.get(unit_name)
        tree.root = tree.dict.get(data['root'])
        return tree

    def compose_flow(self): # compose the GAB from the GAUTree and test it
        pass


