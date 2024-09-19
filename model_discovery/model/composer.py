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
import torch.nn as nn

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


# TODO: WORK IN PROGRESS
class GAUDict: # GAU code book, registry of GAUs, shared by a whole evolution
    def __init__(self, lib_dir=None):
        self.units = {}
        if lib_dir is not None:
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
        name = unit.spec.unitname
        assert name not in self.units, f"Unit {name} is already registered" # never overwrite for backward compatibility
        self.units[name] = unit
        unit.save(self.units_dir)
    
    def get(self, unit_name):
        if unit_name not in self.units:
            return None
        return self.units[unit_name]


class GABComposer:
    def compose(self,tree):
        root_node = tree.root
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
        self.root = None
        self.name = name # name of a design 
        self.proposal = proposal # proposal of the design
        self.proposal_traces = proposal_traces # traces of the proposal
        self.review = review # review of the design
        self.rating = rating
        self.suggestions = suggestions
        if lib_dir is not None:
            self.dict = GAUDict(lib_dir) # TODO: consider this later
        
    def set_lib_dir(self,lib_dir):
        self.dict = GAUDict(lib_dir)

    def add_unit(self, spec, code, args, desc, review, rating, children, gautests, suggestions, design_traces=None, requirements=None, overwrite=False):
        name = spec.unitname
        if name in self.units and not overwrite:
            print(f"Unit {name} is already in the tree")
            return
        # assert name not in self.units, f"Unit {name} is already in the tree"
        # assert not self.dict.exist(name), f"Unit {name} is already registered"
        node = GAUNode(spec, code, args, desc, review, rating, children, gautests, suggestions, design_traces, requirements)
        if len(self.units)==0:
            self.root = node
        self.units[name] = node

    def _replace_unit(self, old: str, new: str): # also need to rename the references in the code, seems hard, so we do not do it for now
        # just clear the old unit, you need to add the new one later manually, its an unsafe method
        # if old == new or old not in self.units:  
        #     return
        # assert new in self.units, f"You must have new unit added to the tree already"
        if old in self.units:
            self.del_unit(old)

        # rename the root 
        if self.root.spec.unitname == old:
            self.root.spec.unitname = new

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

        # May also neeed to update the traces
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

    def register_unit(self, name): # permanently register a unit to the GAUDict, do it only when the unit is fully tested
        assert name in self.units, f"Unit {name} is not in the tree"
        self.dict.register(self.units[name])

    def descendants(self,name): # recursively get decendants of a unit
        assert name in self.units or name in self.declares, f"Unit {name} is neither in the tree nor declared"
        if name in self.units:
            node=self.units[name]
            descendants=set(node.children)
            for child in node.children:
                descendants.update(self.descendants(child))
        else:
            descendants=set()
        return descendants

    def get_disconnected(self):
        connected=self.descendants(self.root.spec.unitname)
        connected.add(self.root.spec.unitname)
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
            'root':self.root.spec.unitname,
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
        for unit_name in dict['units']:
            tree.units[unit_name] = GAUNode.from_dict(dict['units'][unit_name])
        tree.root = tree.units[dict['root']]
        for unit_name in dict['declares']:
            tree.declares[unit_name] = UnitDecl.model_validate_json(dict['declares'][unit_name])
        return tree

    @classmethod
    def load(cls, name, lib_dir):
        dir = U.pjoin(lib_dir, 'flows', f'{name}.json')
        data = U.load_json(dir)
        tree = cls.from_dict(data)
        return tree

    def compose(self): # compose the GAB from the GAUTree and test it
        return GABComposer().compose(self)
    
    def compose_root(self): # compose the root of the GAUTree
        return GABComposer().compose_root(self.root)

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
                report = f'Exported unit tests script with line number:\n\n{U.add_line_num(code)}\n\n{'-'*100}\n\n{report}'
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
        all_children=self.descendants(self.root.spec.unitname)
        all_children.add(self.root.spec.unitname)
        implemented=set(self.units.keys())
        unimplemented=all_children-implemented
        return implemented,unimplemented
 
    def view(self, unit_code=True): # XXX: BUGGY! NEED TO CHECK! GOT WRONG TREE SOMETIMES
        """
        Returns a detailed view of the GAU tree, showing implemented and unimplemented units,
        along with their specifications and code if applicable.
        """
        pstr, _ = self._view(self.root.spec.unitname, node=self.root)
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
                pstr += self.declares[unit].to_prompt() + '\n---\n'

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
        tree.root=tree.units[metadata['root']]
        return tree



