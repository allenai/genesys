from __future__ import annotations

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
import exec_utils
import numpy as np
import os
import copy
import platform

from scipy.optimize import curve_fit

from ..model.loader import reload_gam

import time
import ast
import astor
import inspect
import traceback

from transformers import TrainingArguments, DataCollatorForLanguageModeling

from model_discovery.configs.gam_config import DEFAULT_CONTEXT_LENGTH,DEFAULT_TOKENIZER
from model_discovery.ve.data_loader import load_datasets_args
from model_discovery.ve.modis_trainer import ModisTrainer
import model_discovery.utils as U


def power_function(n, a, b, c):
    return a * np.power(n, b) + c 

def analyze_complexity(sequence_lengths, runtimes, memory_usages):
    # Convert lists to numpy arrays for fitting
    sequence_lengths = np.array(sequence_lengths)
    runtimes = np.array(runtimes)
    memory_usages = np.array(memory_usages)

    # Fit the runtime data to the runtime_function
    popt_runtime, _ = curve_fit(power_function, sequence_lengths, runtimes)
    a_runtime, b_runtime = popt_runtime[:2]

    # Fit the memory usage data to the memory_function
    popt_memory, _ = curve_fit(power_function, sequence_lengths, memory_usages)
    c_memory, d_memory = popt_memory[:2]

    print(f"Runtime complexity: O({a_runtime}n^{b_runtime:.2f})")
    print(f"Memory usage complexity: O({c_memory}n^{d_memory:.2f})")

    return popt_runtime, popt_memory


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHEPATH = f"{MODULE_DIR}"


#### TODO: Multi-GPU checker


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


def get_system_info_str():
    system_info=''
    # CPU Information
    cpu_info = {
        'Processor': platform.processor(),
        'Machine': platform.machine(),
        'Platform': platform.platform(),
        'System': platform.system(),
        'Version': platform.version(),
    }
    # print("CPU Info:", cpu_info)
    system_info+=f'{cpu_info["Processor"]}'       

    # GPU Information
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                'Device': torch.cuda.get_device_name(i),
                'Memory': f"{torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB",
                'Capability': torch.cuda.get_device_capability(i),
            })
        gpu_info = gpu_info[0] # test on 1 gpu
        system_info+=f' {gpu_info["Device"]} {gpu_info["Memory"]}'

    return system_info


BENCHMARK_MODEL = '''
import torch.nn as nn
from mamba_ssm.modules.mha import MHA
from model_discovery.model.utils.modules import GABBase # DO NOT CHANGE THIS IMPORT STATEMENT #
from model_discovery.model.utils.modules import MLP 

class GAB(GABBase):
    def __init__(self,embed_dim: int, n_heads, device=None,dtype=None,**kwargs):
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim)
        self.fn = MHA(embed_dim, n_heads, causal=True, **factory_kwargs)
        self.fn2 = MLP(embed_dim, 4*embed_dim, embed_dim, **factory_kwargs)
        self.norm1 = nn.LayerNorm(embed_dim, **factory_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim, **factory_kwargs)

    def _forward(self,X,**kwargs): # type hints are optional but recommended
        X = self.fn(self.norm1(X))+X
        X = self.fn2(self.norm2(X))+X
        return X
    
gab_config = {'n_heads':8}
'''



class EffectiveChecker: # WORING IN PROGRESS
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.results = {}
        self.ds, self.tokenizer = load_datasets_args(
            DEFAULT_TOKENIZER,
            DEFAULT_CONTEXT_LENGTH,
            ['wikitext2']
        )
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def get_benchmark(self,config):
        exec(BENCHMARK_MODEL,globals())
        if torch.cuda.is_available():
            glm,_ = reload_gam(config,BENCHMARK_MODEL,'BENCHMARK_MODEL',dtype=torch.bfloat16, device="cuda") # intentially use bfloat16 to check whether the model is correctly defined
            glm = glm.cuda()
            glm=glm.to(torch.bfloat16)
        else:
            glm,_ = reload_gam(config,BENCHMARK_MODEL,'BENCHMARK_MODEL',dtype=torch.float16, device="cpu")
            glm = glm.to(torch.float16)
        runtime, loss, gradient_of_losses,max_memory_allocated,total_flos,train_loss=self.test_training(config,glm)
        return {'run_time':runtime,'loss':loss,'gradient_of_losses':gradient_of_losses,'max_memory_allocated':max_memory_allocated,
                'total_flos':total_flos,'train_loss':train_loss}

    def check_training(self, config, model) -> None:
        benchmarks=U.load_json(f'{CACHEPATH}/checker_benchmark.json')
        if 'training' not in benchmarks:
            benchmarks['training']={}
        cache_key=f'{config.scale}_{get_system_info_str()}'
        if cache_key not in benchmarks['training']:
            benchmark=self.get_benchmark(config)
            benchmarks['training'][cache_key]=benchmark
            U.save_json(benchmarks,f'{CACHEPATH}/checker_benchmark.json')
        else:
            benchmark=benchmarks['training'][cache_key]

        run_time,loss,gradient_of_losses,max_memory_allocated,total_flos,train_loss=self.test_training(config,model)
        if gradient_of_losses>0:
            self.errors.append('The model is not training correctly. The loss is not decreasing. ')
        if loss>1e4: # its already abnormal
            self.errors.append('The model is diverging. The loss is NaN. ')
        if run_time>benchmark['run_time']*10:
            self.errors.append(f"The model is not efficient. The training time is overly long. Its {run_time/benchmark['run_time']:.2f} times of the benchmark.")
        elif run_time>benchmark['run_time']*5:
            self.warnings.append(f"The model is not efficient. The training time is long. Its {run_time/benchmark['run_time']:.2f} times of the benchmark.")
        if max_memory_allocated>benchmark['max_memory_allocated']*5:
            self.errors.append(f"The model is not efficient. The memory usage is overly high. Its {max_memory_allocated/benchmark['max_memory_allocated']:.2f} times of the benchmark.")
        elif max_memory_allocated>benchmark['max_memory_allocated']*2:
            self.warnings.append(f"The model is not efficient. The memory usage is high. Its {max_memory_allocated/benchmark['max_memory_allocated']:.2f} times of the benchmark.")
        if total_flos>benchmark['total_flos']*5:
            self.errors.append(f"The model is not efficient. The FLOPs is overly high. Its {total_flos/benchmark['total_flos']:.2f} times of the benchmark.")
        elif total_flos>benchmark['total_flos']*2:
            self.warnings.append(f"The model is not efficient. The FLOPs is high. Its {total_flos/benchmark['total_flos']:.2f} times of the benchmark.")
        if train_loss>benchmark['train_loss']*5:
            self.errors.append(f"The model is not efficient. The training loss is overly high. Its {train_loss/benchmark['train_loss']:.2f} times of the benchmark.")
        elif train_loss>benchmark['train_loss']*2:
            self.warnings.append(f"The model is not efficient. The training loss is high. Its {train_loss/benchmark['train_loss']:.2f} times of the benchmark.")

        self.results['run_time'] = run_time
        self.results['loss'] = loss
        self.results['gradient_of_losses'] = gradient_of_losses
        self.results['max_memory_allocated'] = max_memory_allocated
        self.results['total_flos'] = total_flos
        self.results['train_loss'] = train_loss


    def check_complexity(self, config, model) -> None:
        return # NOT WORKING YET, HARD TO FIT A GOOD FUNC 

        # let the model inference with different input size and see whether the time is linearly increased
        model.eval()
        gab=model.backbone.blocks[0]
        gam=model.backbone
        runtimes=[]
        memory_usages=[]
        seqlens=[500*k for k in range(20)]
        D=model.backbone.d_model
        for seqlen in seqlens:
            torch.cuda.reset_peak_memory_stats()
            X = torch.randn(2, seqlen, D).to(gam.device).to(gam.dtype)
                
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure GPU computations are done
                output = gab(X)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure GPU computations are done
            end_time = time.time()
            runtime = end_time - start_time

            # Measure memory usage
            memory_usage = torch.cuda.max_memory_allocated() / 1024 ** 2  # Convert bytes to MB

            runtimes.append(runtime)
            memory_usages.append(memory_usage)

        seqlens = np.array(seqlens) / 500 # need to tune this coefficient to get right fit
        time_complexity, space_complexity = analyze_complexity(seqlens, runtimes, memory_usages)
        t_scale, t_power = time_complexity[:2]
        m_scale, m_power = space_complexity[:2]
        if t_power>3:
            self.errors.append(f'The model is not efficient. The time complexity is too high. It is O({t_scale}n^{t_power:.2f}).')
        elif t_power>2:
            self.warnings.append(f'The model is not efficient. The time complexity is high. It is O({t_scale}n^{t_power:.2f}).')
        if m_power>3:
            self.errors.append(f'The model is not efficient. The space complexity is too high. It is O({m_scale}n^{m_power:.2f}).')
        elif m_power>2:
            self.warnings.append(f'The model is not efficient. The space complexity is high. It is O({m_scale}n^{m_power:.2f}).')

        self.results['time_complexity'] = time_complexity
        self.results['space_complexity'] = space_complexity


    def test_training(self, config, model):
        # TODO: maybe use profiler to get more metrics
        model.train()
        torch.cuda.reset_peak_memory_stats()
        with U.CodeTimer("Training check setup"):
            ckpt_dir=os.environ.get("CKPT_DIR")
            training_args=TrainingArguments(
                output_dir=f'{ckpt_dir}/temp/ve/effective_check',
                overwrite_output_dir=True,
                learning_rate=config.learning_rate,
                save_strategy='no',
                max_steps=5,
                per_device_train_batch_size=8,
                auto_find_batch_size=True, # for safety
                optim="adamw_hf",
                logging_steps=1,
                dataloader_num_workers=16,
                dataloader_pin_memory=True,
                tf32=True,
                ddp_find_unused_parameters=False,  # Set this to False
                lr_scheduler_type="cosine_with_min_lr",
                lr_scheduler_kwargs={
                    "min_lr_rate": 0.1, 
                },
                warmup_ratio=0.02,
                report_to="none",
            )
            trainer = ModisTrainer(
                model=model,
                train_dataset=self.ds["train"],
                tokenizer=self.tokenizer,
                args=training_args,
                data_collator=self.data_collator,
            )
            trainer.args._n_gpu = 1
        with U.CodeTimer("Training check running"):
            output=trainer.train()
        
        run_time=output.metrics['train_runtime']
        loss=output.training_loss
        losses=[log['loss'] for log in trainer.state.log_history if 'loss' in log]
        gradient_of_losses=np.gradient(losses).mean()
        total_flos=output.metrics['total_flos']
        train_loss=trainer.state.log_history[-1]['train_loss'] # average the loss across all steps
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2  # Convert bytes to MB
        return run_time,loss,gradient_of_losses,max_memory_allocated,total_flos,train_loss

    def reset(self):
        self.errors.clear()
        self.warnings.clear()

    def _report_errors(self) -> bool:
        if self.errors:
            report='Errors:\n\n'+'\n'.join(self.errors)
        else:
            report='The model is effective.\n'
        if self.warnings:
            report+='\n\nWarnings:\n\n'+'\n'.join(self.warnings)
        return not bool(self.errors),report,self.results
    
    def check(self, config, model) -> bool:
        self.reset()
        with U.CodeTimer(" - Training effectiveness checking"):
            self.check_training(config,model)
        with U.CodeTimer(" - Inference effectiveness checking"):
            self.check_complexity(config,model)
        return self._report_errors()



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
        block using `_check_causality` (has causal attention), 
        `_check_differentiable` (check that all operations 
        are differentiable)  and `check_magnitude` (that 
        parameters are within a certain range)

    """
    def __init__(self):
        self.report = ''
        self.format_checker = GABFormatChecker()
        self.effective_checker = EffectiveChecker()

    def rprint(self, msg) -> None:
        """Log information of check and adds to report 

        :param msg: 
            The debug and report message.
 
        """
        # self.logging.info(msg)
        print(msg)
        self.report += msg+'\n'

    def reset(self) -> None:
        """Results the report
        
        :rtype: None 
        """
        self.report = ''

    ### HAS SOME WEIRD BUGS ### It may also due to torch
    def _check_causality(self, block, D: int, seq_len: int = 100) -> bool:
        """Checks if a design is causal

        :param block: 
            The target block design. 
        :param D: 
            The block dimensions. 
        :param seq_len: 
            The block target sequence length.
        """
        B: int = 2
        X = torch.arange(seq_len * B * D).float().reshape(B, seq_len, D).to(block.device).to(block.dtype)
            
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
        mock_input = torch.randint(0, vocab_size, (2, 2048)).cuda() if \
          torch.cuda.is_available() else torch.randint(0, vocab_size, (2, 2048))
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
    

    def _check_effectiveness(self, model, config) -> bool:
        self.rprint('Checking effectiveness...')
        checkpass,errors,results=self.effective_checker.check(config, model)    
        self.rprint(errors)
        return checkpass,results
    
    
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
        mock_input = torch.randint(0, vocab_size, (2, 2048)).cuda() if \
          torch.cuda.is_available() else torch.randint(0, vocab_size, (2, 2048))
        mock_input = mock_input.to(gab.device)
        emb.eval()
        gab.eval()
        try:
            with torch.no_grad():
                input=emb(mock_input)
                output = gab(input)
        except Exception as e:
            error_trace = traceback.format_exc()
            self.rprint(
                'Error: Forward pass failed with error: '+str(e)+'\n'
                'Full Traceback: \n' + error_trace
            )
            return False
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
        time_start=time.time()
        self.reset()
        with U.CodeTimer("Format checking"):
            self.rprint('Checking the designed model...')
            try:
                checkpass,gab_code = self._check_format_and_reformat(gab_code)
                assert checkpass
            except AssertionError:
                return False,self.report,gab_code,{}
        
        with U.CodeTimer("Model initialization"):
            try: 
                exec(gab_code,globals())
                if torch.cuda.is_available():
                    glm,_ = reload_gam(config,gab_code,name,dtype=torch.bfloat16, device="cuda") # intentially use bfloat16 to check whether the model is correctly defined
                    glm = glm.cuda()
                    glm=glm.to(torch.bfloat16)
                else:
                    glm,_ = reload_gam(config,gab_code,name,dtype=torch.float16, device="cpu")
                    glm = glm.to(torch.float16)
                mock_input=torch.randint(0, config.vocab_size, (8, 2048))
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
                return False,self.report,gab_code,{}
        
            ### check model size 
            gam = glm.backbone
            gab = gam.blocks[0].gab
            gab.device = glm.device
            gab.dtype = glm.dtype
            self.rprint(
                f'Model initialization succeeded.\n'
                f'{glm.print_size(printout=False)}'
            )

        # Functional checks
        with U.CodeTimer("Model tests"):
            checkpass2=False
            try:
                checkpass1=self._check_forward_pass(
                    gab,
                    gam.embedding,
                    config.vocab_size
                )
                assert checkpass1
                checkpass2=self._check_causality(
                    gab,
                    gam.d_model
                )
                checkpass3=self._check_differentiable(glm,config.vocab_size)
                checkpass4,effectiveness=self._check_effectiveness(glm,config)
                assert checkpass2 and checkpass3 and checkpass4
            except AssertionError:
                self.rprint('Model test failed\n')
                if not checkpass2:
                    self.rprint('Hint: If you used convolutional layer, you should consider that the conv kernel may cover the future steps. '
                                'You can add padding and truncation of future steps to the conv layer to make it causal.\n')
                return False,self.report,gab_code,{}

        self.rprint("All tests passed!\n")
        time_end=time.time()
        print(f'Total time for checking: {time_end-time_start:.2f}s')
        return True,self.report,gab_code,effectiveness
    
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
                mock_input=torch.randint(0, vocab_size, (8, 2048)).to(glm.device)
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

