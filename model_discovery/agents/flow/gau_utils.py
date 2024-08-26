import ast
import astor


#############################################################
# GAU Format checker and reformatter
#############################################################


class ImportChecker(ast.NodeVisitor):
    def __init__(self):
        self.found_gaubase_import = False

    def visit_ImportFrom(self, node):
        # Check if 'from model_discovery.model.utils.modules import GAUBase' exists
        if node.module == 'model_discovery.model.utils.modules' and any(alias.name == 'GAUBase' for alias in node.names):
            self.found_gaubase_import = True
        return self.generic_visit(node)


class ClassRenamer(ast.NodeTransformer):
    def __init__(self, unit_name):
        self.unit_name = unit_name
        self.gau_class_found = False
        self.gaubase_classes = []
        self.errors = []

    def visit_ClassDef(self, node):
        # Extract the base class names
        base_names = [base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else None for base in node.bases]

        # Check for classes inheriting from GAUBase
        if any(base == "GAUBase" for base in base_names):
            self.gaubase_classes.append(node)

        # Rename the class named 'GAU' to unit_name
        if node.name == "GAU" or (len(self.gaubase_classes) == 1 and not self.gau_class_found):
            node.name = self.unit_name
            self.gau_class_found = True

        return self.generic_visit(node)


class AttributeChecker(ast.NodeVisitor):
    def __init__(self, unit_name):
        self.unit_name = unit_name
        self.children_units = {} # key: class unit_name, value: object name
        self.errors = []
        self.required_args = ["embed_dim", "device", "dtype", "kwargs"]
        self.new_args = {}
        self.found_init = False
        self.found__forward = False
        self.inside_gau_class = False

    def visit_ClassDef(self, node):
        if node.name == self.unit_name:
            self.inside_gau_class = True  # Start processing only when inside GAU class
            self.generic_visit(node)
            self.inside_gau_class = False  # Reset after processing GAU class
        else:
            # Skip other classes
            self.inside_gau_class = False

    def visit_FunctionDef(self, node):
        if not self.inside_gau_class:
            return node
        # Only process the __init__ method
        if node.name == "__init__":
            self.found_init = True

            # Extract argument names
            arg_names = [arg.arg for arg in node.args.args]
            
            # Check for required arguments in __init__
            missing_args = [arg for arg in self.required_args if arg not in arg_names]
            if missing_args:
                for arg in missing_args:
                    new_arg = ast.arg(arg=arg, annotation=None)
                    node.args.args.append(new_arg)
                    # print(f"Added missing argument {arg} to __init__ method of {self.unit_name}")

            # ensure kwargs is **kwargs 
            for kw in node.args.args:
                if kw.arg == "kwargs":
                    node.args.args.remove(kw)
                    break
            node.args.kwarg = ast.arg(arg="kwargs", annotation=None)

            # Check for new args and default values in __init__
            for arg in node.args.args:
                if arg.arg not in self.required_args and arg.arg != "self":
                    self.new_args[arg.arg] = arg.annotation 
            
            # Look for annotated assignments in the __init__ method
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign):
                    # Look for annotated assignments with GAUBase
                    if isinstance(stmt.annotation, ast.Name) and stmt.annotation.id == "GAUBase":
                        # Process the value (which is a constructor call) to check arguments
                        if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                            stmt.value.keywords = []
                            stmt.value.args = []
                            for arg in self.required_args:
                                if arg == "kwargs":
                                    # Handle **kwargs
                                    stmt.value.keywords.append(ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load())))
                                else:
                                    new_kw = ast.keyword(arg=arg, value=ast.Name(id=arg, ctx=ast.Load()))
                                    stmt.value.keywords.append(new_kw)
                            self.children_units[stmt.target.attr] = stmt.value.func.id

        elif node.name == "_forward":
            self.found__forward = True
        elif node.name=='forward':
            self.errors.append("Error: The forward method in GAUBase should not be overridden.")

        return self.generic_visit(node)


class ForwardChecker(ast.NodeVisitor):
    def __init__(self, unit_name, created_children):
        self.unit_name = unit_name
        self.created_children = created_children
        self.warnings = []
        self.inside_gau_class = False
        self.called_path = []

    def visit_ClassDef(self, node):
        if node.name == self.unit_name:
            self.inside_gau_class = True  # Start processing only when inside GAU class
            self.generic_visit(node)
            self.inside_gau_class = False  # Reset after processing GAU class
        else:
            # Skip other classes
            self.inside_gau_class = False

    def visit_FunctionDef(self, node):
        if not self.inside_gau_class:
            return node
        
        if node.name == "_forward":
            # Traverse the body of the _forward function
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    # Check if the right-hand side of the assignment is a function call
                    if isinstance(stmt.value, ast.Call):
                        # Check if the function call is an attribute of `self` (e.g., `self.token_scorer`)
                        if isinstance(stmt.value.func, ast.Attribute) and isinstance(stmt.value.func.value, ast.Name):
                            if stmt.value.func.value.id == "self":
                                self.called_path.append(stmt.value.func.attr)  # Add the attribute name

            # Check for any children that were not called in _forward
            missing_calls = self.created_children - set(self.called_path)
            if missing_calls:
                self.warnings.append(f"Warning: The following GAUBase children defined in __init__ are not called in _forward: {missing_calls}")

        return self.generic_visit(node)


class ModuleProcessor(ast.NodeTransformer):
    def __init__(self, found_gaubase_import, gaubase_classes, unit_name, gau_class_found):
        self.found_gaubase_import = found_gaubase_import
        self.gaubase_classes = gaubase_classes
        self.unit_name = unit_name
        self.gau_class_found = gau_class_found
        self.errors = []

    def visit_Module(self, node):
        # Add import if not found
        if not self.found_gaubase_import:
            gaubase_import = ast.ImportFrom(module='model_discovery.model.utils.modules', names=[ast.alias(name='GAUBase', asname=None)], level=0)
            node.body.insert(0, gaubase_import)

        # Handle renaming and removing other classes that inherit from GAUBase
        if not self.gau_class_found and len(self.gaubase_classes) == 1:
            gau_class_node = self.gaubase_classes[0]
            gau_class_node.name = self.unit_name
            self.gau_class_found = True
        elif len(self.gaubase_classes) > 1:
            matching_class = None
            for cls in self.gaubase_classes:
                if cls.name == self.unit_name:
                    matching_class = cls
                    break
            if matching_class:
                matching_class.name = self.unit_name
            else:
                self.errors.append(f"Error: Multiple classes inheriting from GAUBase found, but none match the provided unit_name '{self.unit_name}'.")

        # Remove other classes that inherit from GAUBase (other than the renamed class)
        node.body = [cls for cls in node.body if not (isinstance(cls, ast.ClassDef) and "GAUBase" in [base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else None for base in cls.bases] and cls.name != self.unit_name)]

        return self.generic_visit(node)


def check_and_reformat_gau_code(source_code, unit_name):
    # Step 1: Parse the source code into an AST
    tree = ast.parse(source_code)
    errors = []
    warnings = []

    # Step 2: Run an import checker to determine if GAUBase import exists
    import_checker = ImportChecker()
    import_checker.visit(tree)

    # Step 3: Run the first pass to rename classes and gather class-related information
    class_renamer = ClassRenamer(unit_name)
    class_renamer.visit(tree)

    # Step 4: Run the second pass to check the __init__ method for annotated attributes
    attribute_checker = AttributeChecker(unit_name)
    attribute_checker.visit(tree)
    if not attribute_checker.found_init:
        errors.append("Error: No __init__ method found in the GAU.")
    if not attribute_checker.found__forward:
        errors.append("Error: No _forward method found in the GAU.")

    # Step 5: Run the third pass to check the _forward method for all created children
    forward_checker = ForwardChecker(unit_name, attribute_checker.children_units.keys())
    forward_checker.visit(tree)

    # Step 6: Process the module (e.g., imports and removing classes)
    module_processor = ModuleProcessor(import_checker.found_gaubase_import, class_renamer.gaubase_classes, unit_name, class_renamer.gau_class_found)
    module_processor.visit(tree)

    reformatted_code = astor.to_source(tree)
    
    errors += class_renamer.errors + attribute_checker.errors + module_processor.errors
    warnings += forward_checker.warnings
    # return the code, children, args, called, errors, and warnings
    return reformatted_code, attribute_checker.children_units, attribute_checker.new_args, forward_checker.called_path, errors, warnings



#############################################################
# GAB Code Generator and Checker
#############################################################







