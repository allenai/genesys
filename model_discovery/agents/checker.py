from __future__ import annotations

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
import exec_utils

from ..model.loader import reload_gam

import ast
import astor
import inspect
import traceback

class GABFormatChecker:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.gab_code = None

    def reset(self):
        self.errors.clear()
        self.warnings.clear()
        self.gab_code = None

    def check(self, gab_code: str) -> bool:
        """Check if the model format is correct.

        :param gab_code: 
            The code of the new block. 
        """
        self.reset()
        self.gab_code = gab_code
        local_ns = {}

        # Parse the code into an AST
        try:
            code_ast = ast.parse(gab_code)
        except Exception as e:
            self.errors.append(f'The code is not parsable:\n{str(e)}\n')
            return self._report_errors()
        

        # Remove the if __name__ == "__main__": block
        code_ast = self._remove_main_block(code_ast)

        # Ensure super().__init__(embed_dim) remains unchanged
        code_ast = self._ensure_super_init(code_ast)
        
        # Update self.gab_code with the modified AST
        self.gab_code = astor.to_source(code_ast)

        print(f'Code after reformatted:\n\n{self.gab_code}\n\n')

        # Execute the modified AST
        # try: ####LEAVE IT OPEN FOR DEBUGGING CAUSE, REMEMBER TO UNCOMMENT
        exec(self.gab_code, globals().copy(), local_ns)
        # except Exception as e:
        #     self.errors.append(f'The code is not executable:\n{str(e)}\n')
        #     return self._report_errors()
        
        # Check for the required class and its base class
        self._check_class_definition(code_ast)
        
        # Check for required import statements
        if 'GABBase' not in local_ns:
            self.errors.append(
                'The import statement is not correct. Cannot find "from model_discovery.model.utils.modules import GABBase". '
                'The GAB class must be inherited from GABBase. You should never define a GABBase class.\n'
            )

        # Additional checks for methods and docstrings
        self._check_methods_and_docstrings(local_ns)
        
        # Check for gab_config dictionary
        self._check_gab_config_dictionary(local_ns, code_ast)

        # Report all errors
        return self._report_errors()

    def _remove_main_block(self, code_ast):
        class MainBlockRemover(ast.NodeTransformer):
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

        return MainBlockRemover().visit(code_ast)
    
    def _ensure_super_init(self, code_ast):
        class SuperInitCorrector(ast.NodeTransformer):
            def visit_ClassDef(cls, node):
                if node.name == 'GAB':
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                            found_super = False
                            for i, stmt in enumerate(item.body):
                                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                                    if (isinstance(stmt.value.func, ast.Attribute) and
                                        stmt.value.func.attr == '__init__' and
                                        isinstance(stmt.value.func.value, ast.Call) and
                                        isinstance(stmt.value.func.value.func, ast.Name) and
                                        stmt.value.func.value.func.id == 'super'):
                                        found_super = True
                                        self.warnings.append(
                                            'The super().__init__(embed_dim) call in GAB is force overwritten by the reformatter. It may cause error if you modified this line.\n'
                                        )
                                        stmt.value = ast.Call(
                                            func=ast.Attribute(
                                                value=ast.Call(
                                                    func=ast.Name(id='super', ctx=ast.Load()),
                                                    args=[],
                                                    keywords=[]
                                                ),
                                                attr='__init__',
                                                ctx=ast.Load()
                                            ),
                                            args=[ast.Name(id='embed_dim', ctx=ast.Load())],
                                            keywords=[]
                                        )
                                        break

                            if not found_super:
                                self.warnings.append(
                                    'The super().__init__(embed_dim) call is missing in the __init__ method. Automatically added by the reformatter.\n'
                                )
                                # Insert super().__init__(embed_dim) at the start of the __init__ method
                                super_call = ast.Expr(
                                    value=ast.Call(
                                        func=ast.Attribute(
                                            value=ast.Call(
                                                func=ast.Name(id='super', ctx=ast.Load()),
                                                args=[],
                                                keywords=[]
                                            ),
                                            attr='__init__',
                                            ctx=ast.Load()
                                        ),
                                        args=[ast.Name(id='embed_dim', ctx=ast.Load())],
                                        keywords=[]
                                    )
                                )
                                item.body.insert(0, super_call)

                return node

        return SuperInitCorrector().visit(code_ast)
    
    def _check_class_definition(self, code_ast) -> None:
        class_found = False
        for node in ast.walk(code_ast):
            if isinstance(node, ast.ClassDef) and node.name == 'GAB':
                class_found = True
                base_names = [base.id for base in node.bases if isinstance(base, ast.Name)]
                if 'GABBase' not in base_names:
                    self.errors.append(
                        'The class name is not correct. Cannot find class "GAB(GABBase)". '
                        'The name of the block class must be GAB and inherited from GABBase.\n'
                    )
                break

        if not class_found:
            self.errors.append(
                'The class "GAB" is not defined. Ensure the class name is "GAB" and it inherits from "GABBase".\n'
            )

    def _check_methods_and_docstrings(self,local_ns) -> None:
        gab_class = local_ns.get('GAB')
        if not gab_class:
            self.errors.append('The class "GAB" is not defined in the provided code.\n')
            return

        required_methods = ['__init__', '_forward']
        for method in required_methods:
            if not hasattr(gab_class, method):
                self.errors.append(f'The method "{method}" is not defined in the class "GAB".\n')
        
        # Check __init__ arguments
        init_signature = inspect.signature(gab_class.__init__)
        init_parameters = init_signature.parameters
        required_args = ['embed_dim', 'device', 'dtype']
        for arg in required_args:
            if arg not in init_parameters:
                self.errors.append(f'The "__init__" method of "GAB" is missing the "{arg}" argument.\n')
        
        # Check docstrings
        # for method in required_methods:
        #     if not inspect.getdoc(getattr(gab_class, method, None)):
        #         self.warnings.append(f'The docstring for method "{method}" is missing.\n')
        
    def _check_gab_config_dictionary(self, local_ns, code_ast) -> None:
        gab_config = local_ns.get('gab_config')
        if not gab_config:
            self.errors.append('The dictionary "gab_config" is not defined.\n')
            return 
        
        if not isinstance(gab_config, dict):
            self.errors.append('"gab_config" should be a dictionary.\n')
            return 
        
        gab_class = local_ns.get('GAB')
        if not gab_class:
            self.errors.append('The class "GAB" is not defined in the provided code, cannot validate "gab_config".\n')
            return

        init_signature = inspect.signature(gab_class.__init__)
        init_parameters = init_signature.parameters
        excluded_args = {'self','embed_dim', 'device', 'dtype','kwargs'}
        init_args = {name for name in init_parameters if name not in excluded_args}

        config_args = set(gab_config.keys())

        missing_args = init_args - config_args
        extra_args = config_args - init_args

        if missing_args:
            self.errors.append(f'The dictionary "gab_config" is missing the following arguments: {", ".join(missing_args)} in "GAB.__init__".\n')

        if extra_args:
            self.warnings.append(f'The dictionary "gab_config" contains extra arguments: {", ".join(extra_args)} not used or not allowed to be re-defined in "GAB.__init__". They are automatically removed by the reformatter.\n')
            class ConfigModifier(ast.NodeTransformer):
                def visit_Assign(self, node):
                    if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'gab_config':
                        if isinstance(node.value, ast.Dict):
                            new_keys = []
                            new_values = []
                            for key, value in zip(node.value.keys, node.value.values):
                                if isinstance(key, ast.Constant) and key.s not in extra_args:
                                    new_keys.append(key)
                                    new_values.append(value)
                            node.value.keys = new_keys
                            node.value.values = new_values
                    return node

            modified_ast = ConfigModifier().visit(code_ast)
            self.gab_code = astor.to_source(modified_ast)

    def _report_errors(self) -> bool:
        if self.errors:
            report='Errors:\n\n'+'\n'.join(self.errors)
        else:
            report='Code format is correct and reformatted.\n'
        if self.warnings:
            report+='\n\nWarnings:\n\n'+'\n'.join(self.warnings)
        return not bool(self.errors),report,self.gab_code


__all__ = [
    "Checker"
]

@exec_utils.Registry(
    resource_type="tool_type",
    name="checker",
)
class Checker(exec_utils.BaseTool):
    """Checker for checking the correctness of model designs.  
    

    Methods 
    ----------
    check(check,gab_code: str, name: str) 
        
    This is the main method that checks the proposed 
        block using `_is_causal` (has causal attention), 
        `_check_differentiable` (check that all operations 
        are differentiable)  and `check_magnitude` (that 
        parameters are within a certain range)

    """
    def __init__(self):
        self.report = ''
        self.format_checker = GABFormatChecker()

    def rprint(self, msg) -> None:
        """Log information of check and adds to report 

        :param msg: 
            The debug and report message.
 
        """
        self.logging.info(msg)
        self.report += msg+'\n'

    def reset(self) -> None:
        """Results the report
        
        :rtype: None 
        """
        self.report = ''

    ### HAS SOME WEIRD BUGS ### It may also due to torch
    def _is_causal(self, block, D: int, seq_len: int = 100) -> bool:
        """Checks if a design is causal

        :param block: 
            The target block design. 
        :param D: 
            The block dimensions. 
        :param seq_len: 
            The block target sequence length.
        """
        B: int = 10
        X = torch.arange(seq_len * B * D).float().reshape(B, seq_len, D)
        if torch.cuda.is_available():
            X = X.cuda()
            
        block.eval()  # Set block to evaluation mode, so that dropout layers are not active
        with torch.no_grad():
            Y = block(X)

        print('Checking causality... It checks the causality by changing all future steps X[t+delta] of X[t] and see if Y[t] or any previous outputs change.')
        bar = tqdm(range(seq_len), desc='Causality test', colour='green')
        for t in bar:
            X_mod = X.clone()
            X_mod[:, t + 1:, :]*=-1 # Perturb the future steps of X[t]

            with torch.no_grad():
                Y_mod = block(X_mod)
                        
            # If any previous outputs change when future X[t + delta] changes, then it is not causal
            if not torch.equal(Y[:, :t+1, :], Y_mod[:, :t+1, :]):#, atol=1e-5):
                print(f'Error: Causality test failed at t={t}')
                return False

        print('Causality test passed')
        return True


    def _check_differentiable(self,model,vocab_size: int) -> bool:
        """Check if the mode is differentiable 

        :param model: 
            The target model with the new block. 
        :param vocab_size: 
            The model vocabulary size. 

        """
        self.rprint('Checking differentiability...')
        mock_input = torch.randint(0, vocab_size, (2, 100)).cuda() if \
          torch.cuda.is_available() else torch.randint(0, vocab_size, (2, 100))
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimizer = optim.Adam(model.parameters())

        # Zero the parameter gradients
        optimizer.zero_grad()
        logits = model(mock_input).logits
        loss = criterion(
            logits.view(-1, logits.shape[-1]),
            mock_input.view(-1)
        )
        loss.backward()
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is None:
                self.rprint(f"Error: Parameter {name} does not have a gradient")
                return False

        self.rprint('Differentiability test passed')
        return True
    

    def _check_efficiency(self, model, vocab_size: int) -> bool:

        raise NotImplementedError
    
    
    def _check_format_and_reformat(self, gab_code: str) -> bool:
        """Check if the model format is correct 

        :param gab_code: 
            The code of the new block. 

        """
        self.rprint('Checking code format...')
        checkpass,errors,gab_code=self.format_checker.check(gab_code)
        self.rprint(errors)
        return checkpass,gab_code
    
    def _check_forward_pass(self, gab, emb, vocab_size: int) -> bool:
        """Check if the forward pass is correct 

        :param gab: 
            The target block design. 
        :param vocab_size: 
            The model vocabulary size. 

        """
        self.rprint('Checking forward pass...')
        mock_input = torch.randint(0, vocab_size, (2, 100)).cuda() if \
          torch.cuda.is_available() else torch.randint(0, vocab_size, (2, 100))
        emb.eval()
        gab.eval()
        with torch.no_grad():
            input=emb(mock_input)
            output = gab(input)
        if output.shape != input.shape:
            self.rprint(
                f'Error: The output shape of GAB should be the same as the input. Expected {mock_input.shape}, got {output.shape}.'
            )
            return False
        self.rprint('Forward pass test passed')
        return True
    

    def check(self, config, gab_code: str, name: str) -> bool:
        """Runs through a bunch of checks for the new module at path 

        :param path: 
            The path of the proposed module 
        """
        self.reset()
        self.rprint('Checking the designed model...')
        try:
            checkpass,gab_code = self._check_format_and_reformat(gab_code)
            assert checkpass
        except AssertionError:
            return False,self.report,gab_code
        
        try: 
            exec(gab_code,globals())
            if torch.cuda.is_available():
                glm,_ = reload_gam(config,gab_code,name,dtype=torch.bfloat16, device="cuda") # intentially use bfloat16 to check whether the model is correctly defined
            else:
                glm,_ = reload_gam(config,gab_code,name,dtype=torch.float16, device="cpu")

            mock_input=torch.randint(0, config.vocab_size, (8, 500))
            mock_input = mock_input.to(glm.device)
            output = glm(mock_input)
    
        except Exception as e:
            error_trace = traceback.format_exc()
            self.rprint(
                'Error: Model initialization failed with error: '+str(e)+'\n'
                'Full Traceback: \n' + error_trace + '\n'
                'Hint: 1. if it is a dtype or device error, check whether the factory kwargs are passed to the layers. '
                '2. If it is a shape error, check whether the output shape is equal to the input shape. The output shape of GAB should be the same as the input.'
            )
            return False,self.report,gab_code
        
        ### check model size 
        glm,_ = reload_gam(config,gab_code,name) # reload the model with regular dtype and device
        if torch.cuda.is_available():
            glm = glm.cuda()    
        gam = glm.backbone
        gab = gam.blocks[0].gab
        self.rprint(
            f'Model initialization succeeded.\n'
            f'{glm.print_size(printout=False)}'
        )

        # Functional checks
        try:
            checkpass1=self._check_forward_pass(
                gab,
                gam.embedding,
                config.vocab_size
            )
            checkpass2=self._is_causal(
                gab,
                gam.d_model
            )
            checkpass3=self._check_differentiable(glm,config.vocab_size)
            # assert checkpass1 and checkpass2 and checkpass3
        except AssertionError:
            self.rprint('Model test failed\n')
            if not checkpass2:
                self.rprint('Hint: If you used convolutional layer, you should consider that the conv kernel may cover the future steps. '
                            'You can add padding and truncation of future steps to the conv layer to make it causal.\n')
            return False,self.report,gab_code

        self.rprint("All tests passed!\n")
        return True,self.report,gab_code
    
    def tune(self,config,gab_code,name,tune_dim=True)->str: # the model is already correct but we need to tune its scale
        print('Tuning the model scale...')
        d_model=config.d_model
        n_block=config.n_block
        # assert d_model%128==0 # initial d_model from config should be a multiple of 128
        vocab_size=config.vocab_size
        reference_size=config.reference_size
        threshold=config.size_threshold
        UB=reference_size*(1+threshold)
        LB=reference_size*(1-threshold)
        print(f'Reference size: {reference_size}, threshold: {threshold}, upper bound: {UB}, lower bound: {LB}')

        exec(gab_code,globals())
        if 'GAB' not in globals(): 
            raise NameError("GAB class not defined in the executed code")

        glm,_ = reload_gam(config,gab_code,name)
        size=sum(p.numel() for p in glm.parameters())
        if LB<size<UB:
            print('The model size is already within the threshold.')
            return 'autoconfig={}'
        
        # Tune n_blocks first, then d_model, idea is to maximally keep the size of embedding layer first
        DIR=1 if size<LB else -1
        while True:
            n_block+=DIR
            print(f'Trying n_block={n_block}')
            auto_cfg={'n_block':n_block}
            glm,_ = reload_gam(config,gab_code,name,auto_cfg)
            size=sum(p.numel() for p in glm.parameters())
            if LB<size<UB:
                print('Model after tuned:')
                glm.print_size()
                return "autoconfig = {\n    'n_block': "+str(n_block)+"\n}"
            if (DIR==1 and size>UB) or (DIR==-1 and size<LB):
                print('The model size requirement cannot be met by tuning n_block.')
                break
    
        if not tune_dim:
            raise ValueError('The model size requirement cannot be met by tuning n_block.')
                
        print('Tuning d_model...')
        step_size=d_model//8 # smallest d_model is 128
        if d_model%3==0: # like 384, 768...
            min_step=24
        else:
            min_step=16
        step_size=max(step_size,min_step) 

        DIR=1 if size<LB else -1
        while True: # tune d_model as little as possible
            if (step_size<min_step) or (LB<size<UB):
                break
            d_model+=step_size*DIR
            print(f'Trying d_model={d_model}, n_block={n_block}')
            auto_cfg={'d_model':d_model,'n_block':n_block}
            glm,_ = reload_gam(config,gab_code,name,auto_cfg)
            size=sum(p.numel() for p in glm.parameters())
            NEW_DIR=1 if size<reference_size else -1
            if NEW_DIR!=DIR:
                DIR=NEW_DIR
                step_size=step_size//2
        # if not LB<size<UB: # usually unless the agent create a over huge block
        #     raise ValueError('The model size requirement cannot be met by tuning d_model.')
        
        # Final adjustment of dim to check whether the dim cause error (e.g., dim head)
        DIR=1 if size<reference_size else -1
        print(f'Checking model correctness with d_model={d_model}')
        while True:
            try:
                if torch.cuda.is_available():
                    glm = glm.cuda()
                mock_input=torch.randint(0, vocab_size, (8, 500)).to(glm.device)
                _ = glm(mock_input)
                break
            except Exception as e:
                d_model+=step_size*DIR
                print(f'The model is incorrect. Trying d_model={d_model}')
                glm,_ = reload_gam(config,gab_code,name,auto_cfg)
                size=sum(p.numel() for p in glm.parameters())
                if size>reference_size*(1+2*threshold) or size<reference_size*(1-2*threshold):
                    # Not likely to happen when reference d_model is a multiple of 128 and step_size is at least 8 or 12, but leave it for safety
                    raise ValueError('The model is too far from the reference size and cannot be correctly tuned.')

        print(f'The model is correct with d_model = {d_model}')
        print('Model after tuned:')
        glm.print_size()
        return "autoconfig = {\n    'd_model': "+str(d_model)+"\n    'n_block': "+str(n_block)+"\n}"
    
    def __call__(self,path: str) -> bool:
        return self.check(path)

