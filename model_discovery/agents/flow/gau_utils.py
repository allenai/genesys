import ast
import astor


#############################################################
# GAU Format checker and reformatter
#############################################################


class FormatChecker(ast.NodeTransformer):
    def __init__(self, unit_name):
        self.unit_name = unit_name
        self.found_gaubase_import = False
        self.found_gautest_import = False
        self.gau_class_found = False
        self.gaubase_classes = []
        self.errors = []
        self.warnings = []

    def visit_ImportFrom(self, node):
        # Remove any import from model_discovery.model.utils.modules
        if node.module == 'model_discovery.model.utils.modules':
            return None  # Remove this import line
        return self.generic_visit(node)

    def visit_If(self, node):
        # Check if this is the if __name__ == "__main__": block
        if (isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == '__name__' and
            isinstance(node.test.comparators[0], ast.Constant) and
            node.test.comparators[0].value == '__main__'):
            self.warnings.append(
                'The if __name__ == "__main__": block is removed by the reformatter.\n'
            )
            return None
        return node

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


class KwargAllChecker(ast.NodeVisitor):
    def __init__(self,unit_name):
        self.inside_gau_class = False
        self.unit_name = unit_name
        self.warnings = []
        self.additional_args = []  # To store arguments that need to be added to __init__
        self.existing_args = []  # To track existing arguments

    def visit_FunctionDef(self, node):
        # Check only the __init__ function
        if node.name == "__init__":
            # Extract argument names and defaults from the __init__ method
            init_arg_names = [arg.arg for arg in node.args.args]
            non_default_arg_count = len(init_arg_names) - len(node.args.defaults)
            new_body = []

            for stmt in node.body:
                # Handle Assign: self.param_name = kwarg_all.get('param_name', default_value)
                if (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "get"):
                    
                    self.process_kwarg_get(stmt, stmt.value, stmt.targets[0], node, init_arg_names, non_default_arg_count)
                    continue

                # Handle AnnAssign: self.param_name: Type = kwarg_all.get('param_name', default_value)
                if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "get"):
                    
                    self.process_kwarg_get(stmt, stmt.value, stmt.target, node, init_arg_names, non_default_arg_count, is_annassign=True)
                    continue

                # If no matching pattern, keep the original statement
                new_body.append(stmt)

            # Replace the old body with the new one
            node.body = new_body

            # Add new arguments to __init__ if needed
            for param_name, default_value in self.additional_args:
                node.args.args.append(ast.arg(arg=param_name, annotation=None))
                node.args.defaults.append(default_value)

        return self.generic_visit(node)

    def process_kwarg_get(self, stmt, get_call, target, node, init_arg_names, non_default_arg_count, is_annassign=False):
        """Process kwarg_all.get('param_name', default_value) in Assign or AnnAssign"""
        if (isinstance(get_call.func.value, ast.Name) and get_call.func.value.id == "kwarg_all"
            and len(get_call.args) == 2 and isinstance(get_call.args[0], ast.Constant)):
            
            param_name = get_call.args[0].value
            default_value = get_call.args[1]

            self.warnings.append(f"Warning: '{param_name}' extracted from kwarg_all. Adding or updating it as an argument to __init__.")

            # If the param already exists, check if it has a default value
            if param_name in init_arg_names:
                param_index = init_arg_names.index(param_name)
                if param_index < non_default_arg_count:  # If it's a non-default argument
                    # Set the default value by moving it to the correct position
                    node.args.defaults.insert(param_index - non_default_arg_count, default_value)
                    non_default_arg_count -= 1  # Adjust the count of non-default args
            else:
                # Add the param to __init__ arguments if not already present
                self.additional_args.append((param_name, default_value))
                init_arg_names.append(param_name)

            # Replace the line with self.param_name = param_name
            new_assign = ast.Assign(
                targets=[target],
                value=ast.Name(id=param_name, ctx=ast.Load())
            )
            if is_annassign:  # Preserve the annotation if it was an AnnAssign
                new_assign = ast.AnnAssign(
                    target=target,
                    annotation=stmt.annotation,
                    value=ast.Name(id=param_name, ctx=ast.Load()),
                    simple=1
                )

            node.body.append(new_assign)

    def visit_ClassDef(self, node):
        if node.name == self.unit_name:
            self.inside_gau_class = True  # Start processing only when inside GAU class
            self.generic_visit(node)
            self.inside_gau_class = False  # Reset after processing GAU class
        else:
            # Skip other classes
            self.inside_gau_class = False


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



### TODO: There must be more complicated patterns, need to solve them by recursion, but it is good for now
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
    def __init__(self, gaubase_classes, unit_name, gau_class_found):
        self.gaubase_classes = gaubase_classes
        self.unit_name = unit_name
        self.gau_class_found = gau_class_found
        self.errors = []
        self.warnings = []

    def visit_Module(self, node):
        # Add the combined import line at the top
        gaubase_import = ast.ImportFrom(
            module='model_discovery.model.utils.modules',
            names=[
                ast.alias(name='GAUBase', asname=None),
                ast.alias(name='gau_test', asname=None)
            ],
            level=0
        )
        node.body.insert(2, gaubase_import) # Insert after the first two lines which should be import torch and nn

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

    # Step 2: Run the format checker which now removes the import lines
    format_checker = FormatChecker(unit_name)
    format_checker.visit(tree)

    # Step 3: Run the KwargAllChecker
    kwarg_all_checker = KwargAllChecker(unit_name)
    kwarg_all_checker.visit(tree)

    # Step 4: Run the InitChecker
    init_checker = InitChecker(unit_name)
    init_checker.visit(tree)

    # Step 5: Run the AttributeChecker
    attribute_checker = AttributeChecker(unit_name, children)
    attribute_checker.visit(tree)
    if not attribute_checker.found_init:
        errors.append("Error: No __init__ method found in the GAU.")
    if not attribute_checker.found__forward:
        errors.append("Error: No _forward method found in the GAU.")

    # Step 6: Process the module (e.g., imports and removing classes)
    module_processor = ModuleProcessor(
        format_checker.gaubase_classes,
        unit_name,
        format_checker.gau_class_found
    )
    module_processor.visit(tree)

    # Step 7: Generate the reformatted source code
    reformatted_code = astor.to_source(tree)
    
    # Step 8: Collect errors and warnings
    errors += format_checker.errors + attribute_checker.errors + module_processor.errors
    warnings += format_checker.warnings

    # Return the reformatted code, any new arguments, errors, and warnings
    return reformatted_code, attribute_checker.new_args, errors, warnings
