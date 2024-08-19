import sys
import torch
import torch.fx
from types import MethodType
from dataclasses import dataclass
import copy
from functools import partial

torch._dynamo.config.cache_size_limit = 64  # Increase the limit as needed

sys.path.append('..')

from model_discovery.model.library import *
from exec_utils import BuildTool
from model_discovery.configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

#!!! BETTER NOT HAVING ANY KWARGS WHEN CALLING AN NN.MODULE
# When using a kwarg wrapper, the kwargs will be hidden in CONSTANTS


def load_gab(model_name: str, scale='14M'):
    gab_code = MODEL2CODE[model_name]
    checker = BuildTool(tool_type="checker")
    try:
        checkpass, gab_code = checker._check_format_and_reformat(gab_code)
        assert checkpass
    except AssertionError as e:
        print('Model does not pass the format checker')
        raise e
    
    # Wrap len inside the executed code
    gab_code = f"{gab_code}"
    
    module = {}
    exec(gab_code.replace("class GAB","class GABCustom"),module)
    assert "GABCustom" in module, "Class GAB not found in module. You should never ever change the class name of GAB and it should always inherit from GABBase."
    GAB = module["GABCustom"]

    cfg = eval(f"GAMConfig_{scale}()")
    gab_config = {} 
    assert "gab_config" in module, "Dictionary gab_config not found in module."
    gab_config = module["gab_config"]

    gab= GAB(cfg.d_model,block_loc=(0,cfg.n_block),device=None,dtype=None, **gab_config)

    return gab,cfg



class ModuleNode:
    def __init__(self, name, graph_module=None,kwarg=None):
        self.name = name
        self.graph_module = graph_module
        self.children = []
        self.kwargs = kwarg

    def print_tree(self, indent=""):
        print(indent + self.name)
        if self.graph_module:
            print(indent + "  (GraphModule captured)")#, self.graph_module)
        for child in self.children:
            child.print_tree(indent + "  ")

@dataclass
class BlockAnalysis:
    root: ModuleNode
    nodes: dict
    config: GAMConfig

class BlockAnalyzer:
    def __init__(self):
        self.module_tree_root = None
        self.current_inputs = {}  # To store inputs for each module during forward pass
        self.current_nodes = {}  # To store ModuleNode instances for each module by path

    def track_input_wrapper(self, original_forward, module_path):
        # Custom wrapper for the forward method to capture both positional and keyword arguments
        def wrapped_forward(module_self, *inputs, **kwargs):
            self.current_inputs[module_path] = (inputs, kwargs)
            # Call the original forward method without re-binding `self`
            return original_forward(*inputs, **kwargs)

        return wrapped_forward

    def wrap_forward_methods(self, model,wrapper):
        # Replace the forward method of each submodule with the custom wrapped version
        for module_path, module in self._get_full_module_paths(model):
            original_forward = module.forward
            module.forward = MethodType(wrapper(original_forward, module_path), module)

    def _get_full_module_paths(self, model):
        # Recursively generate the full path for each module in the model
        module_paths = []

        def recursive_collect_modules(parent, prefix):
            for name, module in parent.named_children():
                full_path = f"{prefix}.{name}" if prefix else name
                module_paths.append((full_path, module))
                recursive_collect_modules(module, full_path)

        recursive_collect_modules(model, "")
        return module_paths

    def analyze_submodule(self, module_path, module):
        # Retrieve the inputs and kwargs for this module captured during the forward pass
        inputs, kwargs = self.current_inputs.get(module_path, (None, None))

        module = copy.deepcopy(module)
        if inputs is None:
            return None
        if kwargs:
            for key in kwargs:
                try:
                    kwargs[key] = kwargs[key].detach()
                except:
                    pass
            module.forward=partial(module.forward,**kwargs)

        # Trace the current module with the captured inputs and kwargs
        if isinstance(inputs,tuple):
            new_inputs = []
            for inp in inputs:
                try:
                    new_inputs.append(inp.detach())
                except:
                    new_inputs.append(inp)
            inputs = tuple(new_inputs)
        else:
            try:
                inputs = inputs.detach()
            except:
                pass
        traced_module = torch.jit.trace(module, inputs)
        # Create a ModuleNode for the current module
        node = ModuleNode(module_path, traced_module, kwargs)
        self.current_nodes[module_path] = node

        # Recursively trace submodules
        for name, submodule in module.named_children():
            child_path = f"{module_path}.{name}"
            child_node = self.analyze_submodule(child_path.replace('root.',''), submodule)
            if child_node is not None:
                node.children.append(child_node)

        return node

    def analyze(self, model, cfg):
        # Wrap the forward methods to capture both positional and keyword arguments
        self.current_inputs = {}
        self.current_nodes = {}
        model_wrap = copy.deepcopy(model)   
        self.wrap_forward_methods(model_wrap,self.track_input_wrapper)

        # Run the model with an example input
        input_tensor = torch.randn(2, 100, cfg.d_model)
        model_wrap(input_tensor)  # This will trigger the wrapped forward methods and capture inputs
        del model_wrap
        
        # Start with the root module and analyze it recursively
        self.current_inputs['root'] = (input_tensor, None)
        self.module_tree_root = self.analyze_submodule('root', model)

        return BlockAnalysis(self.module_tree_root, self.current_nodes, cfg)
    




if __name__ == '__main__':
    gab,cfg = load_gab('ttt')
    analyzer = BlockAnalyzer()
    analysis = analyzer.analyze(gab, cfg)

