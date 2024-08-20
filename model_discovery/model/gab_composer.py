# Translate GABUnits to the executable GAB
import os
from typing import List

from dataclasses import dataclass, field
from .utils.modules import GABFlow, GABUnit

import model_discovery.utils as U



ROOT_UNIT_TEMPLATE = '''# {name}.py

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GABUnit # DO NOT CHANGE THIS IMPORT STATEMENT #


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #


class {name}(GABUnit): 
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


class GABBook: # GABUnit code book, registry of GABUnits, shared by a whole evolution
    def __init__(self, db_dir=None, rename=True):
        self.units = {}
        self.log = []
        self.rename = rename
        self.units_dir = U.pjoin(db_dir, 'units')
        U.mkdir(self.units_dir)
        for dir in os.listdir(self.units_dir):
            self.load(dir.split('.')[0])

    def assign_name(self, name):
        if name in self.units:
            if self.rename:
                cnt=1
                while True:
                    if f"{name}_{cnt}" not in self.units:
                        name = f"{name}_{cnt}"
                        break
                    cnt += 1
            else:
                raise ValueError(f"Unit {name} is already registered")
        return name
    
    def check_source(self, source):
        
        # NOT IMPLEMENTED YET

        return source

    def parse_source(self, source, name):
        modules = {}
        try:
            exec(source,modules)
        except Exception as e:
            raise ValueError(f"Failed to parse source code: {e}")
        assert name in modules, f"Unit {name} is not defined in the source code"
        unit = modules[name]
        assert isinstance(unit,GABUnit), f"Unit {name} is not a GABUnit"
        return unit
    
    def load(self, name):
        dir=U.pjoin(self.units_dir,f'{name}.py')
        if U.pexists(dir):
            name=U.psplit(dir)[-1].split('.')[0]
            source = U.read_file(dir)
            unit=self.parse_source(source,name,rename=self.rename)
            assert name not in self.units, f"Unit {name} is not registered"
            self.units[name] = unit
        else:
            raise ValueError(f"File {dir} does not exist")
        
    def register_from_source(self, source, name): # Write
        rname = self.assign_name(name)
        if rname != name:
            source = source.replace(name,rname) # XXX: this is a very naive way to rename the class, may cause problems
        return self._write_from_source(source,rname)
    
    def _write_from_source(self, source, name): # Write
        try:
            source = self.check_source(source)
        except Exception as e:
            raise ValueError(f"Failed to check source code: {e}")
        unit = self.parse_source(source,name)
        self.units[name] = unit
        U.write_file(U.pjoin(self.units_dir,f'{name}.py'),source)
        return unit

    def update_from_source(self, source, name): # Update
        assert name in self.units, f"Unit {name} is not registered"
        return self._write_from_source(source,name)
    
    def retrieve(self, unit_name):
        if unit_name not in self.units:
            raise ValueError(f"Unit {unit_name} is not registered")
        return self.units[unit_name]


@dataclass
class GABNode: # this is mainly used to 1. track the hierarchies 2. used for the Linker to solve the dependencies
    name: str # name of the GABUnit, do we need an alias?
    next: str = None # pointer to the next GABNode, GABTree is always a series 
    child: str = None # pointer to the child GABNode, if there is a child, it will go through the child branch until the end then go back and pass the output to the next 
    config: dict = field(default_factory=dict)

    def save(self, flow_dir):
        data={
            'name':self.name,
            'next':self.next,
            'child':self.child,
            'config':self.config
        }
        U.save_json(U.pjoin(flow_dir,f'{self.name}.json'),data)

    @classmethod
    def load(cls, name, flow_dir):
        data=U.load_json(U.pjoin(flow_dir,f'{name}.json'))
        return cls(**data)


# ideally the GABTree is similar to a pseudo-code
class GABTree:
    def __init__(self, name, db_dir=None):
        self.path = {} 
        self.name = name # name of a design 
        self.book = GABBook(db_dir)
        self.flow_dir = U.pjoin(db_dir, 'flows', self.name)
        self.config = {} # the config of the whole design, avoid assigning configs to individual GABUnits
        U.mkdir(self.flow_dir)
        self.load()
        if len(self.path)==0: # if the tree is empty, create a new one
            self.path[self.name] = GABNode(name=self.name)
            root_template = ROOT_UNIT_TEMPLATE.replace('{name}', 'test_tree')
            self.book.register_from_source(root_template,self.name)
            self.save()

    def save(self):
        dir=U.pjoin(self.flow_dir,f'metadata.json')
        metadata = {
            'name':self.name,
            'path': list(self.path.keys()),
            'config':self.config,
        }
        U.save_json(dir,metadata)
        for i in self.path:
            self.path[i].save(self.flow_dir)
    
    def load(self):
        dir=U.pjoin(self.flow_dir,f'metadata.json')
        if not U.pexists(dir):
            return
        metadata=U.load_json(dir)
        assert self.name==metadata['name'], "The name of the metadata does not match the name of the GABTree" # impossible if the code is correct
        self.config=metadata['config']
        paths=metadata['path']
        self.load_tree()
        assert set(paths)==set(self.path.keys()), "The paths in the metadata does not match the paths in the GABTree" # impossible if the code is correct

    def load_tree(self,path=None):
        if path is None: # load the whole tree
            path=self.name
        name=path.split('.')[-1]
        base=path[:-len(name)-1]
        node=GABNode.load(name,self.flow_dir)
        self.path[path]=node
        if node.next is not None:
            assert base != '', "The root node should not have a next node" # it should be impossible if the code is correct
            self.load_tree(f'{base}.{node.next}')
        if node.child is not None:
            self.load_tree(f'{path}.{node.child}')

    def compose_flow(self): # scan from the root unit, recursively compose the GABFlow by fill in the GABUnit
        pass


