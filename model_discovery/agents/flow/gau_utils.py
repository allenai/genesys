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


### TODO: There must be more complicated patterns, need to solve them by recursion, but it is good for now
class InitChecker(ast.NodeVisitor):
    def __init__(self, unit_name):
        self.unit_name = unit_name
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
        if self.inside_gau_class and node.name == "__init__":
            self.found_init = True

            # Check for and remove existing self.factory_kwargs and super() calls
            new_body = []

            for stmt in node.body:
                # Detect and skip the self.factory_kwargs assignment
                if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Attribute):
                    target = stmt.targets[0]
                    if (target.value.id == "self" and target.attr == "factory_kwargs"):
                        factory_kwargs_found = True
                        continue
                
                # Detect and skip the super().__init__ call
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    if isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "__init__":
                        super_init_found = True
                        continue

                new_body.append(stmt)

            # Replace the old body with the filtered body (without the old lines)
            node.body = new_body

            # Add the required lines at the beginning of the body
            factory_kwargs_assign = ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="factory_kwargs", ctx=ast.Store())],
                value=ast.Dict(keys=[ast.Constant(value="device"), ast.Constant(value="dtype")],
                               values=[ast.Name(id="device", ctx=ast.Load()), ast.Name(id="dtype", ctx=ast.Load())])
            )

            super_init_call = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=ast.Call(func=ast.Name(id="super", ctx=ast.Load()), args=[], keywords=[]), attr="__init__", ctx=ast.Load()),
                    args=[ast.Name(id="embed_dim", ctx=ast.Load()), ast.Name(id="block_loc", ctx=ast.Load()), ast.Name(id="kwarg_all", ctx=ast.Load())],
                    keywords=[]
                )
            )

            # Add the factory_kwargs and super().__init__ lines at the start of the method
            node.body.insert(0, factory_kwargs_assign)
            node.body.insert(1, super_init_call)

        return self.generic_visit(node)



class AttributeChecker(ast.NodeVisitor):
    def __init__(self, unit_name, children):
        self.unit_name = unit_name
        self.errors = []
        self.required_args = ["embed_dim", "block_loc", "kwarg_all", "device", "dtype", "kwargs"]
        self.children = children
        self.children_visit_order = []
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
            kwarg_name = node.args.kwarg.arg if node.args.kwarg else None
            
            # Check for required arguments in __init__
            missing_args = [arg for arg in self.required_args if arg not in arg_names and arg != kwarg_name]
            non_default_arg_count = len(node.args.args) - len(node.args.defaults)
            for missing_arg in missing_args:
                new_arg = ast.arg(arg=missing_arg, annotation=None)
                if missing_arg == "kwargs":
                    node.args.kwarg = new_arg
                else:
                    node.args.args.insert(non_default_arg_count, new_arg)
                    non_default_arg_count += 1  

            # ensure kwargs is **kwargs 
            for kw in node.args.args:
                if kw.arg == "kwargs":
                    node.args.args.remove(kw)
                    break
            node.args.kwarg = ast.arg(arg="kwargs", annotation=None)

            # Handle default values
            # `defaults` will be shorter than `args.args` if not all args have default values
            defaults = node.args.defaults
            num_non_default_args = len(node.args.args) - len(defaults)
            
            # Loop through all arguments
            for i, arg in enumerate(node.args.args):
                if arg.arg not in self.required_args and arg.arg != "self":
                    # Check if the argument has a default value
                    if i >= num_non_default_args:
                        # Get the corresponding default value from defaults
                        default_value_node = defaults[i - num_non_default_args]
                        try:
                            self.new_args[arg.arg] = ast.literal_eval(default_value_node)
                        except ValueError:
                            self.new_args[arg.arg] = None  # Handle cases where ast.literal_eval can't evaluate
                    else:
                        # If no default value, set it to None
                        self.new_args[arg.arg] = None
            
            # Look for assignments in the __init__ method
            for stmt in node.body:
                # Handle direct assignments (e.g., self.unit = UnitClass(...))
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    if isinstance(stmt.value, ast.Call):
                        self.process_assignment(stmt)
                    else:
                        self.process_collections(stmt)

        elif node.name == "_forward":
            self.found__forward = True
        elif node.name=='forward':
            self.errors.append("Error: The forward method in GAUBase should not be overridden.")

        return self.generic_visit(node)


    def handle_statements(self, body):
        for stmt in body:
            # Handle direct and annotated assignments
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if isinstance(stmt.value, ast.Call):
                    self.process_assignment(stmt)
                self.process_collections(stmt)

    def process_assignment(self, stmt):
        """Handle direct assignments like self.unit = UnitClass(...)"""
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target

            if isinstance(target, ast.Attribute) and target.value.id == "self":
                if self.is_child_class(stmt.value.func):
                    self.rewrite_instantiation(stmt.value)
                # Process nn cases like nn.Sequential or nn.ModuleList
                self.handle_nn_cases(stmt.value)

    def handle_nn_cases(self, node):
        """Handle specific nn.Module cases like nn.Sequential, nn.ModuleList, and nn.ModuleDict"""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "Sequential":
                self.process_sequential(node)
            elif node.func.attr == "ModuleList":
                self.process_modulelist(node)
            elif node.func.attr == "ModuleDict":
                self.process_moduledict(node)

    def process_sequential(self, node):
        """Handle nn.Sequential initialization"""
        for arg in node.args:
            if isinstance(arg, ast.Call):
                func = arg.func
                if isinstance(func, ast.Name):
                    if func.id in self.children:
                        # self.rewrite_instantiation(arg)
                        self.errors.append("Error: nn.Sequential is not supported in GAU. You may use ModuleList, ModuleDict.")
                    elif func.id == 'OrderedDict':
                        self.process_ordereddict(arg)
                elif isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name):
                        if func.value.id=='nn' and func.attr=='Sequential':
                            self.process_sequential(arg)

    def process_ordereddict(self, node):
        """Handle OrderedDict initialization inside Sequential"""
        # The first argument to OrderedDict is a list of tuples
        if isinstance(node.args[0], ast.List):
            for tuple_elem in node.args[0].elts:
                if isinstance(tuple_elem, ast.Tuple) and len(tuple_elem.elts) == 2:
                    key, value = tuple_elem.elts
                    if isinstance(value, ast.Call):
                        if self.is_child_class(value.func):
                            # self.rewrite_instantiation(value)
                            self.errors.append("Error: nn.Sequential is not supported in GAU. You may use ModuleList, ModuleDict.")
                elif isinstance(tuple_elem, ast.Call):
                    if isinstance(tuple_elem.func, ast.Tuple) and len(tuple_elem.func.elts) == 2:
                        key, value = tuple_elem.func.elts
                        if isinstance(value, ast.Call):
                            if self.is_child_class(value.func):
                                # self.rewrite_instantiation(value)
                                self.errors.append("Error: nn.Sequential is not supported in GAU. You may use ModuleList, ModuleDict.")

    def process_moduledict(self, node):
        """Handle nn.ModuleDict initialization"""
        if isinstance(node.args[0], ast.Dict):
            for key, value in zip(node.args[0].keys, node.args[0].values):
                if isinstance(value, ast.Call):
                    if self.is_child_class(value.func):
                        self.rewrite_instantiation(value)

    def get_full_attribute(self, node):
        """Recursively extract full attribute chain from an ast.Attribute node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self.get_full_attribute(node.value) + "." + node.attr
        return None

    def process_modulelist(self, node):
        if isinstance(node.args[0], ast.List):
            for elem in node.args[0].elts:
                if isinstance(elem, ast.Call) and self.is_child_class(elem.func):
                    self.rewrite_instantiation(elem)
        elif isinstance(node.args[0], ast.ListComp):
            self.process_comprehension(node.args[0])

    def process_collections(self, stmt):
        if isinstance(stmt.value, (ast.List, ast.Tuple, ast.Set)):
            for elem in stmt.value.elts:
                if isinstance(elem, ast.Call) and self.is_child_class(elem.func):
                    self.rewrite_instantiation(elem)
        elif isinstance(stmt.value, ast.Dict):
            self.process_dict(stmt.value)
        elif isinstance(stmt.value, ast.ListComp):
            self.process_comprehension(stmt.value)

    def process_dict(self, node):
        for value in node.values:
            if isinstance(value, ast.Call) and self.is_child_class(value.func):
                self.rewrite_instantiation(value)

    def process_comprehension(self, list_comp):
        if isinstance(list_comp.elt, ast.Call) and self.is_child_class(list_comp.elt.func):
            self.rewrite_instantiation(list_comp.elt)

    def is_child_class(self, func):
        if isinstance(func, ast.Name):
            return func.id in self.children
        elif isinstance(func, ast.Attribute):
            return func.attr in self.children
        return False
    
    def rewrite_instantiation(self, node):
        """Rewrite the instantiation of a GAUBase child"""
        node.keywords = []  # Clear existing keyword arguments
        node.args = []  # Clear existing positional arguments

        # Add keyword arguments
        node.keywords.append(ast.keyword(arg="embed_dim", value=ast.Name(id="embed_dim", ctx=ast.Load())))
        node.keywords.append(ast.keyword(arg="block_loc", value=ast.Name(id="block_loc", ctx=ast.Load())))
        node.keywords.append(ast.keyword(arg="kwarg_all", value=ast.Name(id="kwarg_all", ctx=ast.Load())))

        # Add **self.factory_kwargs and **kwarg_all as unpacked keyword arguments
        node.keywords.append(ast.keyword(arg=None, value=ast.Name(id="self.factory_kwargs", ctx=ast.Load())))
        node.keywords.append(ast.keyword(arg=None, value=ast.Name(id="kwarg_all", ctx=ast.Load())))

        self.children_visit_order.append(node.func.id if isinstance(node.func, ast.Name) else node.func.attr)

        print(f"Rewritten instantiation: {node.func.id if isinstance(node.func, ast.Name) else node.func.attr}(embed_dim, block_loc, kwarg_all, **self.factory_kwargs, **kwarg_all)")



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


def check_and_reformat_gau_code(source_code, unit_name, children):
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

    # Check init 
    init_checker = InitChecker(unit_name)
    init_checker.visit(tree)

    # Step 4: Run the second pass to check the __init__ method for annotated attributes
    attribute_checker = AttributeChecker(unit_name, children)
    attribute_checker.visit(tree)
    if not attribute_checker.found_init:
        errors.append("Error: No __init__ method found in the GAU.")
    if not attribute_checker.found__forward:
        errors.append("Error: No _forward method found in the GAU.")

    # Step 5: Process the module (e.g., imports and removing classes)
    module_processor = ModuleProcessor(import_checker.found_gaubase_import, class_renamer.gaubase_classes, unit_name, class_renamer.gau_class_found)
    module_processor.visit(tree)

    reformatted_code = astor.to_source(tree)
    
    errors += class_renamer.errors + attribute_checker.errors + module_processor.errors
    # return the code, children, args, called, errors, and warnings
    return reformatted_code, attribute_checker.new_args, errors, warnings

