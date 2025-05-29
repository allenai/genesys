# Translate GAUBases to the executable GAB
import os
import ast, astor
from typing import List, Dict
import json
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import torch
import copy
import torch.nn as nn
import numpy as np
import random
from streamlit_flow import streamlit_flow
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout, RadialLayout
from pyvis.network import Network
import math
import networkx as nx

from dataclasses import dataclass, field
from .utils.modules import GAUBase, gau_test

import model_discovery.utils as U

from model_discovery.model.utils.modules import UnitSpec, UnitDecl
from model_discovery.agents.flow.gau_utils import process_core_library_code



@dataclass
class GAUNode: # this is mainly used to 1. track the hierarchies 2. used for the Linker to solve the dependencies 3. fully serialize the GAUTree
    spec: UnitSpec # name of the GAU 
    code: str # code of the GAU
    args: dict # *new* args and default values of the GAU
    desc: str # description of the GAU
    review: str 
    rating: str
    children: list[str] # list of children GAUs
    gautests: Dict[str, str] # unit tests of the GAU
    suggestions: str # suggestions for the GAU from reviewer for further improvement
    design_traces: list = None # traces of the design process
    requirements: str = None # requirements of the GAU from declaration, root GAU has no requirements as the demand is the proposal
    reuse_from: str = None # whether the GAU is reused from another one

    def json(self):
        data = self.to_dict()
        return json.dumps(data, indent=4)

    def save(self, dir):
        U.save_json(json.loads(self.json()), U.pjoin(dir, f'{self.spec.unitname}.json'))

    def to_dict(self):
        data = self.__dict__.copy()
        data['spec'] = self.spec.model_dump_json() 
        return data

    @classmethod
    def from_dict(cls, dict):
        data = cls(**dict)
        data.spec = UnitSpec.model_validate_json(data.spec)  # Ensure proper deserialization
        return data

    @classmethod
    def load(cls, name, dir):
        data = U.load_json(U.pjoin(dir, f'{name}.json'))
        return cls.from_dict(data)
    

@dataclass
class GAUDictTerm:
    name: str
    is_root: bool = False
    variants: Dict[str, GAUNode] = field(default_factory=lambda: {}) # tree name where the unit is registered

    def append(self,variant,node,desc):
        if variant not in self.variants:
            self.variants[variant] = (node,variant,desc)



# TODO: WORK IN PROGRESS
# Reuse: same name; New: different name
# If there is an existing GAU for children, may reuse
# If implementing a new GAU, make sure no similar ones exist
# Same name must be a reuse, different name must be new
class GAUDict: # GAU code book, registry of GAUs, shared by a whole evolution
    def __init__(self,ptree):
        self.terms = {}
        self.ptree=ptree
    
    def new_term(self,acronym,tree):
        if not isinstance(tree,GAUTree):
            return
        for unit_name in tree.units:
            if unit_name not in self.terms:
                self.terms[unit_name]=GAUDictTerm(name=unit_name)
            desc=tree.get_unit_desc(unit_name)
            self.terms[unit_name].append(acronym,tree.units[unit_name],desc)
            if tree.root == unit_name:
                self.terms[unit_name].is_root = True

    @classmethod
    def from_ptree(cls,ptree):
        _cls = cls(ptree)
        nodes=ptree.filter_by_type(['DesignArtifactImplemented','ReferenceCoreWithTree'])
        for acronym in nodes:
            tree = ptree.get_gau_tree(acronym)
            _cls.new_term(acronym,tree)
        return _cls
    
    def get_tree(self,name):
        return self.ptree.get_gau_tree(name)

    def get_unit(self,name,tree_names=None,K=1): # K = None is return all, otherwise K random
        if name not in self.terms:
            return None
        if tree_names is None:
            if K>1: # RETURN K RANDOM non overlapping trees
                tree_names=np.random.choice(list(self.terms[name].variants.keys()),min(K,len(self.terms[name].variants)),replace=False)
            else: # RETURN ALL
                tree_names=self.terms[name].variants.keys()
            tree_names=list(set(tree_names))
        else:
            if isinstance(tree_names,str):
                tree_names=[tree_names]
        matches=[self.terms[name].variants[tree_name] for tree_name in tree_names]
        if K==1:
            return matches[0] # for convenience
        return matches # tuples of (node,tree_name,decl)
    

    def viz(self,G,height=5000,width="100%",layout=False,max_nodes=None,bgcolor="#fafafa",physics=True):
        nt=Network(
            directed=True,height=height,width=width,
            layout=layout, 
            bgcolor=bgcolor, #font_color="#ffffff",
            #select_menu=True, # filter_menu=True,
            # heading=f'Phylogenetic Tree for {self.db_dir.split("/")[-2]}'
        )
        nt.prep_notebook(True)#,'./etc/ptree_template.html')
        nt.from_nx(G)
        fname='UTree' if not layout else 'UTree_layout'
        if max_nodes: fname+=f'_{max_nodes}'
        nt.toggle_physics(physics)
        nt.show(U.pjoin(self.ptree.db_dir, '..', fname+'.html'))

    def export(self,max_nodes=None,height=5000,layout=False,bgcolor="#eeeeee",color_root='yellow',size_root=25,
               color_term='red',size_term=20,color_variant='blue',size_variant=10,no_root=False,no_units=False,physics=True): #,with_ext=False

        degrees = {}
        n_reuses = {}
        for unit in self.terms:
            term = self.terms[unit]
            variants = term.variants
            for variant in variants:
                node = variants[variant][0]
                if node.reuse_from:  # seems wrong?
                    _,reuse_from = node.reuse_from.split('.')
                    if reuse_from not in self.terms:
                        continue
                    if reuse_from not in degrees:
                        degrees[reuse_from] = 0
                    degrees[reuse_from]+=1
                    if unit not in degrees:
                        degrees[unit] = 0
                    degrees[unit]+=1
                    if reuse_from not in n_reuses:
                        n_reuses[reuse_from] = 0
                    n_reuses[reuse_from]+=1
        sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        G=nx.DiGraph()
        variant_edges=[]
        reuse_edges=[]
        cnt=0
        for name,degree in sorted_degrees:
            if max_nodes and cnt>=max_nodes:
                break
            term=self.terms[name]
            if no_root and term.is_root:
                continue
            cnt+=1
            # add the term node
            n_reused = n_reuses.get(name,0)
            size = size_term if not term.is_root else size_root
            if term.is_root:
                color = color_root
            else:
                if n_reused<2:
                    color = '#b9dec9'
                elif n_reused<4:
                    color = '#9eccab'
                elif n_reused<8:
                    color = '#8cc269'
                elif n_reused<16:
                    color = '#229453'
                elif n_reused<32:
                    color = '#41b349'
                else: # n_reused<64:
                    color = '#20a162'
            G.add_node(
                name,
                title=name,
                size=size+max(0,int(math.log(n_reused,2)))*2 if n_reused else size,
                color=color,
            )

            for variant,content in term.variants.items():
                (node,variant,desc)=content
                # add the variant node
                if not no_units:
                    G.add_node(
                        variant,
                        title=U.break_sentence(desc,100),
                        size=size_variant,
                        color=color_variant,
                    )
                    # add the edge
                    variant_edges.append((name,variant))
                # add the edge from the reuse node
                if node.reuse_from is not None:
                    parent = node.reuse_from.split('.')[-1]
                    if not no_units:
                        reuse_edges.append((parent,variant))
                    else:
                        if (parent,name) not in reuse_edges and parent!=name:
                            reuse_edges.append((parent,name))
            
        for edge in variant_edges:
            if edge[0] in G.nodes and edge[1] in G.nodes:
                G.add_edge(edge[0],edge[1],color=color_variant)
        for edge in reuse_edges:
            if edge[0] in G.nodes and edge[1] in G.nodes:
                G.add_edge(edge[0],edge[1])#,color=color_term)
        fname='unit_tree'
        if max_nodes: fname+=f'_{max_nodes}'
        self.viz(G,max_nodes=max_nodes,height=height,layout=layout,bgcolor=bgcolor,physics=physics)
        return G


    
class GABComposer:
    def compose(self,tree):
        root_node = tree.root_node
        declares=tree.declares
        generated_code = []
        processed_units = set()

        # Recursively generate code for the root and its children
        self.generate_node_code(root_node.spec.unitname, generated_code, tree.units, processed_units,declares)
        
        # Combine all generated code into a single Python file content
        gau_code = "\n".join(generated_code)

        gathered_args={}
        for unit in tree.units.values():
            gathered_args.update(unit.args)

        
        gab_code=self.compose_root(root_node)
        cfg_code=f'gab_config = {str(gathered_args)}'

        compoesed_code = f'{gab_code}\n\n{gau_code}\n\n{cfg_code}'

        compoesed_code=U.replace_from_second(compoesed_code,'import torch\n','')
        compoesed_code=U.replace_from_second(compoesed_code,'import torch.nn as nn\n','')
        compoesed_code=U.replace_from_second(compoesed_code,'from model_discovery.model.utils.modules import GAUBase, gau_test, UnitDecl\n','')

        return compoesed_code
    
    def compose_root(self,root_node):
        
        GAB_TEMPLATE='''
# gab.py    # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE #

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #


class GAB(GABBase):
    def __init__(self,embed_dim: int, block_loc: tuple, device=None,dtype=None,**kwargs): # YOU CAN ADD MORE ARGUMENTS, BUT YOU HAVE TO HAVE embed_dim, device, dtype AS THE ARGUTMENTS #
        factory_kwargs = {{"device": device, "dtype": dtype}} # remember to pass it to nn layers
        super().__init__(embed_dim, block_loc) # DO NOT CHANGE THIS LINE #
        self.root = {ROOT_UNIT_NAME}(embed_dim=embed_dim, block_loc=block_loc, kwarg_all=kwargs, **factory_kwargs, **kwargs)

    def _forward(self, X, **Z): 
        X, Z = self.root(X, **Z)
        return X, Z
'''

        gab_code=GAB_TEMPLATE.format(ROOT_UNIT_NAME=root_node.spec.unitname)
        return gab_code
    
    def compose_unit(self, tree, unit_name,declares):
        if unit_name not in tree.units:
            print(f"Unit {unit_name} is not in the tree")
            return None
        
        unit = tree.units[unit_name]
        gau_tests = unit.gautests
        
        generated_code = []
        processed_units = set()
        self.generate_node_code(unit_name, generated_code, tree.units, processed_units,declares)

        gau_code = "\n".join(generated_code)

        run_code = f'def run_{unit_name}_tests():\n'
        for test_name, test_code in gau_tests.items():
            gau_code += f"\n\n{test_code}"
            run_code += f"\ttry:\n\t\ttest_{unit_name}_{test_name}()\n"
            run_code += '\texcept Exception as e:\n'
            run_code += f'\t\tprint("Error in running {test_name}:")\n'
            run_code += '\t\tprint(traceback.format_exc())\n'
        if len(gau_tests)==0:
            run_code += f'\t\tprint("No tests found for {unit_name}, all tests must be decorated with @gau_test")'

        run_code += '\n\nif __name__ == "__main__":'
        run_code += f"\n\trun_{unit_name}_tests()"

        composed_code = f'{gau_code}\n\n{run_code}'
        return composed_code

        
    # Recursive function to generate code for a node and its children
    def generate_node_code(self, unit_name, generated_code: List[str], units, processed_units,declares):
        if unit_name in processed_units:
            return
        processed_units.add(unit_name)
        # Check if the node exists in units
        if unit_name not in units:
            # If the node does not exist in units, create a placeholder
            outputs='{}'
            if unit_name in declares:
                declare=declares[unit_name]
                outputs='{'+','.join([f"'{name}': None" for name in declare.outputs])+'}'
            generated_code.append(self.create_placeholder_class(unit_name,outputs))
        else:
            node = units[unit_name]
            generated_code.append(node.code)
            
            # Recursively generate code for children
            for child_unit in set(node.children):
                self.generate_node_code(child_unit, generated_code, units, processed_units,declares)

    # Function to create a placeholder class for a GAUNode
    def create_placeholder_class(self, unit_name, outputs='{}') -> str:
        class_template = f"""
class {unit_name}(GAUBase): 
    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None, dtype=None, **kwargs): 
        self.factory_kwargs = {{"device": device, "dtype": dtype}} 
        super().__init__(embed_dim, block_loc, kwarg_all)
        
    def _forward(self, X, **Z): 
        Z_={outputs}
        return X, Z_
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
    def __init__(self, name, proposal, review, rating, suggestions, lib_dir=None, proposal_traces=[]):
        self.units:Dict[str,GAUNode] = {} 
        self.declares:Dict[str,UnitDecl] = {} # the declarations of the units in the tree, not including the root as it has spec directly
        self.root = None # the root node name of the tree
        self.name = name # name of a design 
        self.proposal = proposal # proposal of the design
        self.proposal_traces = proposal_traces # traces of the proposal
        self.review = review # review of the design
        self.rating = rating
        self.suggestions = suggestions
        # if lib_dir is not None:
        #     self.dict = GAUDict(lib_dir) # TODO: consider this later
        
    def get_unit_desc(self,unit_name):
        if unit_name in self.units:
            unit=self.units[unit_name]
            decl=self.declares.get(unit_name,None)
            requirements=f'Declaration Requirements: {decl.requirements}' if decl is not None else ''
            desc = unit.spec.to_prompt()
            return f'{desc}\n\n{requirements}'
        elif unit_name in self.declares:
            return self.declares[unit_name].to_prompt()
        else:
            return None
    
    # def set_lib_dir(self,lib_dir):
    #     self.dict = GAUDict(lib_dir)

    def add_unit(self, spec, code, args, desc, review, rating, children, gautests, suggestions, design_traces=None, requirements=None, overwrite=False, reuse_from=None):
        name = spec.unitname
        if name in self.units and not overwrite:
            print(f"Unit {name} is already in the tree")
            return
        # assert name not in self.units, f"Unit {name} is already in the tree"
        # assert not self.dict.exist(name), f"Unit {name} is already registered"
        node = GAUNode(spec, code, args, desc, review, rating, children, gautests, suggestions, design_traces, requirements, reuse_from)
        if len(self.units)==0:
            self.root = name
        self.units[name] = node

    def _replace_unit(self, old: str, new: str): # also need to rename the references in the code, seems hard, so we do not do it for now
        # just clear the old unit, you need to add the new one later manually, its an unsafe method
        # if old == new or old not in self.units:  
        #     return
        # assert new in self.units, f"You must have new unit added to the tree already"
        if old in self.units:
            self.del_unit(old)

        # rename the root 
        # if self.root.spec.unitname == old:
        #     self.root.spec.unitname = new
        if old == self.root:
            self.root = new

        # rename the declares
        if old in self.declares:
            # if new not in self.declares:
            #     self.declares[new] = copy.deepcopy(self.declares[old])
            del self.declares[old]

        # rename children
        if old != new:
            for unit in self.units.values():
                if old in unit.children: # both new and old names are class names
                    unit.children[unit.children.index(old)] = new
                    unit.code=unit.code.replace(f'{old}(',f'{new}(') 
                    unit.code=unit.code.replace(f'{old},',f'{new},') # both new and old names are instance names

        # update the dict should be handled separately, should simply add a new entry for back compatibility, should be handled already

    def get_children(self,name):
        assert name in self.units, f"Unit {name} is not in the tree"
        return self.units[name].children

    def del_unit(self, name):
        assert name in self.units, f"Unit {name} is not in the tree"
        del self.units[name]
    
    def del_declare(self, name):
        assert name in self.declares, f"Unit {name} is not declared"
        del self.declares[name]

    # def register_unit(self, name): # permanently register a unit to the GAUDict, do it only when the unit is fully tested
    #     assert name in self.units, f"Unit {name} is not in the tree"
    #     self.dict.register(self.units[name])

    def descendants(self,name): # recursively get decendants of a unit
        if name not in self.units and name not in self.declares:
            # raise ValueError(f"Unit {name} is neither in the tree nor declared")
            return set()
        if name in self.units:
            node=self.units[name]
            descendants=set(node.children)
            for child in node.children:
                descendants.update(self.descendants(child))
        else:
            descendants=set()
        return descendants

    def get_disconnected(self):
        connected=self.descendants(self.root)
        connected.add(self.root)
        disconnected=set(self.units.keys())-connected
        return disconnected
    
    def clear_disconnected(self):
        disconnected=self.get_disconnected()
        for unit in disconnected:
            del self.units[unit]
        return disconnected  

    def to_dict(self):
        data = {
            'name':self.name,
            'root':self.root,
            'units':{name:self.units[name].to_dict() for name in self.units},  
            'declares': {name:declare.model_dump_json() for name,declare in self.declares.items()},
            'proposal':self.proposal,
            'proposal_traces':self.proposal_traces,
            'review':self.review,
            'rating':self.rating,
            'suggestions':self.suggestions
        }
        return data

    @classmethod
    def from_dict(cls, dict: Dict, lib_dir=None):
        tree = cls(dict['name'], dict['proposal'], dict['review'], dict['rating'], dict['suggestions'], lib_dir)
        tree.root = dict['root']
        for unit_name in dict['declares']:
            tree.declares[unit_name] = UnitDecl.model_validate_json(dict['declares'][unit_name])
        for unit_name in dict['units']:
            tree.units[unit_name] = GAUNode.from_dict(dict['units'][unit_name])
            # if unit_name not in tree.declares:
            #     spec=tree.units[unit_name].spec
            #     decl=UnitDecl(unitname=unit_name,requirements=spec.document,inputs=spec.inputs,outputs=spec.outputs)
            #     tree.declares[unit_name] = decl
        return tree

    @classmethod
    def load(cls, name, lib_dir):
        dir = U.pjoin(lib_dir, 'flows', f'{name}.json')
        data = U.load_json(dir)
        tree = cls.from_dict(data)
        return tree

    def compose(self): # compose the GAB from the GAUTree and test it
        return GABComposer().compose(self)
    
    @property
    def root_node(self):
        return self.units[self.root]
    
    def compose_root(self): # compose the root of the GAUTree
        return GABComposer().compose_root(self.root_node)

    def compose_unit(self, unit_name): # compose a single unit for running unit tests
        return GABComposer().compose_unit(self, unit_name,self.declares)
    
    def test_unit(self, unit_name, return_code=True):
        code = self.compose_unit(unit_name)
        if code is None:
            report = f'Unit {unit_name} not found'
            return (report, None, False) if return_code else (report, False)

        # Prepare to capture output
        _stdout_output = io.StringIO()
        _stderr_output = io.StringIO()

        # Create a custom namespace
        namespace = {
            '__name__': '__main__',
            'print': lambda *args, **kwargs: print(*args, file=_stdout_output, **kwargs),
            'traceback': traceback,
        }

        try:
            # Redirect stdout and stderr to capture all output
            with redirect_stdout(_stdout_output), redirect_stderr(_stderr_output):
                exec(code, namespace)

            # Get the captured output
            captured_stdout = _stdout_output.getvalue()
            captured_stderr = _stderr_output.getvalue()
        except Exception as e:
            error_msg = f"An error occurred while executing the unit test:\n{traceback.format_exc()}"
            if return_code:
                return error_msg, code, False
            return error_msg, False

        passed=True
        if captured_stderr:
            passed=False
        if captured_stdout or captured_stderr:
            new_check_report=[]
            code_lines=code.split('\n')
            need_code_lines=False
            captured_output = captured_stdout+'\n'+captured_stderr
            for line in captured_output.split('\n'):
                if 'File "<string>", line' in line:
                    passed=False
                    need_code_lines=True
                    fname=f'test_{unit_name}.py'
                    line=line.replace('File "<string>", line',f'File "{fname}", line')
                    line_num=int(line.split(f'File "{fname}", line ')[-1].split(',')[0].strip())
                    line=line.replace(f'line {line_num}',f'line {line_num}: {code_lines[line_num-1]}')
                new_check_report.append(line)
            report='\n'.join(new_check_report)                
            if need_code_lines:
                report = f'Exported unit tests script with line number:\n\n{U.add_line_num(code)}\n\n{"-"*100}\n\n{report}'
        else:
            report = f"No output captured for {unit_name} unit tests"

        if return_code:
            return report, code, passed
        return report, passed
        
    def _view(self, _name, path='', node=None, pstr='', unimplemented=set()):
        """
        Create a string representation of the GAU tree structure.
        """
        name = _name
        
        # Handling unimplemented nodes
        if node is None: 
            name += ' (Unimplemented)'
            unimplemented.add(_name)
        else:
            if node.rating is not None:
                name += f" (Rating: {node.rating}/5)"
            if node.reuse_from:
                if name != node.reuse_from.split('.')[-1]:
                    name += f" [Adapted from {node.reuse_from}]"
                else:
                    name += f" [Reused from {node.reuse_from}]"

        # Define levels for tree structure indentation
        if path != '':
            level = len(path.split('.'))
            name = '    ' * level + ' |- ' + name
            path += '.' + _name
        else:
            # pstr += f'GAU Tree Map for {self.name}:\n'
            path = _name
        
        pstr += '  ' + name + '\n'
        
        # Recursively view child nodes
        if node is not None:
            for child_unit in node.children:
                child_node = self.units.get(child_unit, None)
                pstr, unimplemented = self._view(child_unit, path, child_node, pstr, unimplemented)
        
        return pstr, unimplemented

    def check_implemented(self):
        all_children=self.descendants(self.root)
        all_children.add(self.root)
        implemented=set(self.units.keys())
        unimplemented=all_children-implemented
        return implemented,unimplemented
    
    def tree_view(self):
        pstr, _ = self._view(self.root, node=self.root_node)
        return pstr
 
    def view(self, unit_code=True): # XXX: BUGGY! NEED TO CHECK! GOT WRONG TREE SOMETIMES
        """
        Returns a detailed view of the GAU tree, showing implemented and unimplemented units,
        along with their specifications and code if applicable.
        """
        pstr=self.tree_view()
        pstr=f'#### {self.name} Tree Map\n\n```bash\n{pstr}'
        # Collect implemented and unimplemented units
        implemented,unimplemented = self.check_implemented()

        # Display implemented and unimplemented units
        pstr += '\nImplemented Units: ' + ', '.join(implemented)
        if unimplemented:
            pstr += '\nUnimplemented Units: ' + ', '.join(unimplemented)
        else:
            pstr += '\nAll units are implemented.'
        pstr += '\n```\n'
        
        pstr += '\n\n#### Specifications for Implemented Units:\n'
        
        # Append specifications and code for implemented units
        for unit in self.units.values():
            pstr += unit.spec.to_prompt() + '\n'
            if unit_code:
                pstr += f'\n###### Implementation \n\n<details><summary>Click to expand</summary>\n\n```python\n{unit.code}\n```\n</details>\n'
                pstr += '\n---\n'
        
        # Append unimplemented unit declarations
        if unimplemented:
            pstr += '\n\n##### Declarations for Unimplemented Units:\n'
            for unit in unimplemented:
                if unit in self.declares:
                    pstr += self.declares[unit].to_prompt() + '\n---\n'
                else:
                    pstr += f'\n{unit}: declaration not found (Unimplemented)'

        unused=self.get_disconnected()
        if unused:
            pstr += '\n\n##### Unused Units:\n'
            for unit in unused:
                pstr += self.units[unit].spec.to_prompt() + '\n'
                if unit_code:
                    pstr += f'\n###### Implementation \n\n<details><summary>Click to expand</summary>\n\n```python\n{self.units[unit].code}\n```\n</details>\n'
                pstr += '\n---\n'
        
        return pstr, list(implemented), list(unimplemented)

    def to_prompt(self, unit_code=True):
        """
        Generates the final prompt including the GAU tree and the composed LM block code.
        """
        view, _, _ = self.view(unit_code)
        
        # Generate the complete composed LM block code
        code = self.compose_root() if unit_code else self.compose()
        
        return (
            f'## Tree Map of the GAUs\n\n'
            '<details><summary>Click to expand</summary>\n\n'
            f'\n{view}\n'
            '</details>\n\n'
            '## Composed LM Block Code\n'
            '<details><summary>Click to expand</summary>\n\n'
            f'```python\n{code}\n'
            '... # Unit implementations\n```\n'
            '</details>\n\n'
        ) if unit_code else (
            f'## Tree Map of the GAUs\n\n'
            '<details><summary>Click to expand</summary>\n\n'
            f'\n{view}\n'
            '</details>\n\n'
            '## Composed LM Block Code\n'
            '<details><summary>Click to expand</summary>\n\n'
            f'```python\n{code}\n```\n'
            '</details>\n\n'
        )


    @classmethod
    def load_from_base(cls,path,lib_dir=None):
        '''
        load a GAUTree from the core library, the path should include the metadata.yaml and the units directory /src,
        each src file should include SPEC, ARGS, DESC, CHILDREN, and the unit tests
        '''
        if not U.pexists(path):
            return None
        metadata=U.load_yaml(U.pjoin(path,'metadata.yaml'))
        name=metadata['name']
        proposal=metadata['proposal']
        review=metadata.get('review',None)
        rating=metadata.get('rating',None)
        suggestions=metadata.get('suggestions',None)
        tree=cls(name,proposal,review,rating,suggestions,lib_dir=lib_dir)
        for unit in os.listdir(U.pjoin(path,'src')):
            if unit.endswith('.py'):
                code=U.read_file(U.pjoin(path,'src',unit))
                local={}
                exec(code,local)
                spec=UnitSpec.model_validate(local['SPEC'])
                args=local['ARGS']
                desc=local['DESC']
                children=local['CHILDREN']
                review=local.get('REVIEW',None)
                rating=local.get('RATING',None)
                suggestions=local.get('SUGGESTIONS',None)
                requirements=local.get('REQUIREMENTS',None)
                code,gautests,warnings=process_core_library_code(code,spec.unitname)
                if warnings:
                    print(f'Warning {spec.unitname}: {warnings}')
                tree.add_unit(spec,code,args,desc,review,rating,children,gautests,suggestions,requirements)
        tree.root=metadata['root']
        return tree


    def _traverse(self,parent,flow_nodes,flow_edges,with_code=False):
        if parent in self.units: 
            children=self.units[parent].children
            for child in children:
                code = f'\n\n```python\n{self.units[child].code}\n```' if with_code else ''
                node=StreamlitFlowNode(child,(0,0),{'content':f'{child}{code}'})
                flow_nodes[child] = node
                flow_edges.append(StreamlitFlowEdge(f'{parent}-{child}',parent,child,animated=True,
                                                    style={'markerEnd': 'url(#arrow)'}))
                flow_nodes,flow_edges = self._traverse(child,flow_nodes,flow_edges,with_code)
        return flow_nodes,flow_edges


    def get_flow_nodes(self,with_code=False):
        flow_nodes = {}
        flow_edges = []
        code = f'\n\n```python\n{self.units[self.root].code}\n```' if with_code else ''
        flow_st=StreamlitFlowNode(self.root,(0,0),{'content':f'{self.root}{code}'},'input',
                                # source_position='right',target_position='left',
                                style={'backgroundColor': '#20a162'})
        flow_nodes[self.root] = flow_st
        flow_nodes,flow_edges = self._traverse(self.root,flow_nodes,flow_edges,with_code)
        
        return flow_nodes,flow_edges

    def export(self,height=800,with_code=False,light_mode=False):
        flow_nodes,flow_edges = self.get_flow_nodes(with_code)

        if not with_code:
            horizontal_spacing=300
            vertical_spacing=75
            node_node_spacing=150
        else:
            horizontal_spacing=300
            vertical_spacing=150
            node_node_spacing=450
        style = {}
        if light_mode:
            style = {'backgroundColor': '#f0f0f0', 'textColor': '#000000'}
        state = StreamlitFlowState(list(flow_nodes.values()), flow_edges)
        return streamlit_flow(
            self.name, 
            state,
            layout=TreeLayout(
                "down",
                # horizontal_spacing=horizontal_spacing,
                # vertical_spacing=vertical_spacing,
                node_node_spacing=node_node_spacing,
            ), 
            fit_view=True, 
            height=height, 
            enable_node_menu=False, 
            show_minimap=False, 
            enable_pane_menu=False, 
            hide_watermark=True, 
            allow_new_edges=False, 
            get_node_on_click=True,
            get_edge_on_click=True,
            min_zoom=0.1,
            style=style
        )
