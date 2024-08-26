import ast
import astor

class GAUReformer(ast.NodeTransformer):
    def __init__(self, unit_name):
        self.errors = []
        self.gau_class_found = False
        self.gaubase_classes = []
        self.unit_name = unit_name
        self.found_gaubase_import = False

    def is_inheriting_from_gaubase(self, bases):
        # Check if any of the base classes are GAUBase, whether directly as ast.Name or ast.Attribute
        for base in bases:
            if isinstance(base, ast.Name) and base.id == "GAUBase":
                return True
            elif isinstance(base, ast.Attribute) and base.attr == "GAUBase":
                return True
        return False

    def visit_ImportFrom(self, node):
        # Check if 'from model_discovery.model.utils.modules import GAUBase' exists
        if node.module == 'model_discovery.model.utils.modules' and any(alias.name == 'GAUBase' for alias in node.names):
            self.found_gaubase_import = True
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        # Check for classes inheriting from GAUBase
        if self.is_inheriting_from_gaubase(node.bases):
            self.gaubase_classes.append(node)
        
        # Check if there's a class named 'GAU'
        if node.name == "GAU":
            self.gau_class_found = True
            # Rename GAU class to unit_name
            node.name = self.unit_name
            # Ensure GAU inherits from GAUBase
            if not self.is_inheriting_from_gaubase(node.bases):
                node.bases = [ast.Name(id="GAUBase", ctx=ast.Load())]
        
        return self.generic_visit(node)

    def visit_Module(self, node):
        # Add import if not found
        if not self.found_gaubase_import:
            gaubase_import = ast.ImportFrom(module='model_discovery.model.utils.modules', names=[ast.alias(name='GAUBase', asname=None)], level=0)
            node.body.insert(0, gaubase_import)

        # Handle GAU class detection and renaming
        if not self.gau_class_found:
            if len(self.gaubase_classes) == 1:
                # Rename the only GAUBase class to GAU (with unit_name)
                gau_class_node = self.gaubase_classes[0]
                gau_class_node.name = self.unit_name
            elif len(self.gaubase_classes) > 1:
                # Find the class with the name matching unit_name
                matching_class = None
                for cls in self.gaubase_classes:
                    if cls.name == self.unit_name:
                        matching_class = cls
                        break
                
                if matching_class:
                    matching_class.name = self.unit_name
                else:
                    self.errors.append(f"Error: Multiple classes inheriting from GAUBase found, but none match the provided unit_name '{self.unit_name}'.")
            else:
                self.errors.append("Error: No class inheriting from GAUBase found.")
        else:
            # Remove other classes that inherit from GAUBase (other than the renamed class)
            for cls in self.gaubase_classes:
                if cls.name != self.unit_name:
                    node.body.remove(cls)

        return self.generic_visit(node)

def check_and_reformat_code(source_code, unit_name):
    tree = ast.parse(source_code)
    reformer = GAUReformer(unit_name)
    transformed_tree = reformer.visit(tree)
    reformatted_code = astor.to_source(transformed_tree)
    
    return reformatted_code, reformer.errors

# Example usage
source_code = """
# gau.py

import torch
import torch.nn as nn

from model_discovery.model.utils.modules import GAUBase

# Placeholder classes for future implementation
class MemoryAccessUnit(GAUBase):
    def __init__(self, embed_dim, memory_size, device=None, dtype=None):
        super().__init__(embed_dim)

    def _forward(self, X, **Z):
        return X, {}

class DownsamplingUnit(GAUBase):
    def __init__(self, embed_dim, downsample_factor, device=None, dtype=None):
        super().__init__(embed_dim)

    def _forward(self, X, **Z):
        return X, {}

class GAU(GAUBase):  # This class will be renamed to the unit_name
    def __init__(self, embed_dim: int, device=None, dtype=None):
        super().__init__(embed_dim)

    def _forward(self, X, **Z):
        return X, Z
"""

unit_name = "CustomGAU"  # Provide the unit_name to rename GAU class
reformatted_code, errors = check_and_reformat_code(source_code, unit_name)
print("Reformatted Code:\n", reformatted_code)
print("Errors:\n", errors)
