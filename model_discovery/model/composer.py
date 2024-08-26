# Translate GAUBases to the executable GAB
import os
import ast, astor
from typing import List, Dict
import json

from dataclasses import dataclass, field
from .utils.modules import GAUBase

import model_discovery.utils as U





GAB_TEMPLATE='''
# gab.py    # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE #

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #


class GAB(GABBase):
    def __init__(self,embed_dim: int, block_loc: tuple, device=None,dtype=None,**kwargs): # YOU CAN ADD MORE ARGUMENTS, BUT YOU HAVE TO HAVE embed_dim, device, dtype AS THE ARGUTMENTS #
        factory_kwargs = {{"device": device, "dtype": dtype}} # remember to pass it to nn layers
        super().__init__(embed_dim, block_loc) # DO NOT CHANGE THIS LINE #
        self.root = {ROOT_UNIT_NAME}(embed_dim, embed_dim=embed_dim, device=device, dtype=dtype, **kwargs)

    def _forward(self, X, *Z): 
        X, Z = self.root(X, **Z)
        return X, Z
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



class GABComposer:
    
    def generate_gab_code(self,tree):
        root_node = tree.root
        generated_code = []
        
        # Recursively generate code for the root and its children
        self.generate_node_code(root_node.name, generated_code, tree.units)
        
        # Combine all generated code into a single Python file content
        gau_code = "\n".join(generated_code)

        gathered_args={}
        for unit in tree.units.values():
            gathered_args.update(unit.args)
        gab_code=GAB_TEMPLATE.format(ROOT_UNIT_NAME=root_node.name)

        cfg_code=f'gab_config = {json.dumps(gathered_args)}'

        compoesed_code = f'{gab_code}\n\n{gau_code}\n\n{cfg_code}'

        compoesed_code=U.replace_from_second(compoesed_code,'import torch\n','')
        compoesed_code=U.replace_from_second(compoesed_code,'import torch.nn as nn\n','')
        compoesed_code=U.replace_from_second(compoesed_code,'from model_discovery.model.utils.modules import GAUBase\n','')

        return compoesed_code


    # Recursive function to generate code for a node and its children
    def generate_node_code(self, unit_name, generated_code: List[str], units):
        # Check if the node exists in units
        if unit_name not in units:
            # If the node does not exist in units, create a placeholder
            generated_code.append(self.create_placeholder_class(unit_name))
        else:
            node = units[unit_name]
            generated_code.append(node.code)
            
            # Recursively generate code for children
            children_units=set()
            for child_name, child_unit_name in node.children.items():
                children_units.add(child_unit_name)
            for child_unit in children_units:
                self.generate_node_code(child_unit, generated_code, units)

    # Function to create a placeholder class for a GAUNode
    def create_placeholder_class(self, unit_name) -> str:
        class_template = f"""
class {unit_name}(GAUBase): 
    def __init__(self, embed_dim: int, device=None, dtype=None, **kwargs): 
        factory_kwargs = {{"device": device, "dtype": dtype}} 
        super().__init__(embed_dim) 
        
    def _forward(self, X, **Z): 
        return X
"""
        return class_template

    # Function to convert the generated code to AST using ast and astor
    def convert_code_to_ast(self, code: str):
        try:
            return ast.parse(code)
        except SyntaxError as e:
            print(f"Syntax error in code: {code}")
            raise e

    # Function to convert AST back to Python code using astor
    def convert_ast_to_code(self, ast_tree: ast.AST) -> str:
        return astor.to_source(ast_tree)
    



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

    def add_unit(self, name, code, args, desc, path, review, rating, report, children, suggestions, overwrite=False):
        if name in self.units and not overwrite:
            print(f"Unit {name} is already in the tree")
            return
        # assert name not in self.units, f"Unit {name} is already in the tree"
        assert not self.dict.exist(name), f"Unit {name} is already registered"
        node = GAUNode(name, code, args, desc, path, review, rating, report, children, suggestions)
        if len(self.units)==0:
            self.root = node
        self.units[name] = node

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

    def compose(self): # compose the GAB from the GAUTree and test it
        gab_code = GABComposer().generate_gab_code(self)
        return gab_code


