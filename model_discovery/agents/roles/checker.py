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

from ...model.loader import reload_gam

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



class GABCorrector(ast.NodeTransformer):
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.found_import = False
        self.found_GAB = False
        self.found__init__=False
        self.found__forward=False

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
    
    def visit_ImportFrom(self, node): # FIXME: seems not working, always not found, but doesn't matter too much
        if (node.module == 'model_discovery.model.utils.modules' and
                any(alias.name == 'GABBase' for alias in node.names)):
            self.found_import = True
        return node

    def visit_ClassDef(self, node):
        if node.name == 'GABBase':
            self.warnings.append(
                'The GAB class must be inherited from GABBase. You should never define a GABBase class by yourself. Automatically removed by the corrector, may cause errors.\n'
            )
            return None

        if node.name == 'GAB':
            self.found_GAB = True
            # Check if GAB inherits from GABBase
            if not any(base.id == 'GABBase' for base in node.bases if isinstance(base, ast.Name)):
                self.warnings.append('The class "GAB" does not inherit from "GABBase". Automatically adding the inheritance.\n')
                node.bases.append(ast.Name(id='GABBase', ctx=ast.Load()))

            # Remove any forward method to avoid overriding
            new_body = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                    self.warnings.append(f'The "forward" method in "GAB" class is removed to avoid overriding the one in "GABBase". This may cause error.\n')
                    continue
                new_body.append(item)
            node.body = new_body

            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    self.found__init__=True
                    current_args = [arg.arg for arg in item.args.args]
                    required_args = ['self','embed_dim', 'block_loc', 'device', 'dtype']
                    
                    # Create a new list of args with required args in correct order
                    new_args = []
                    added_args = set()
                    for required_arg in required_args:
                        if required_arg not in current_args:
                            self.warnings.append(f'The "__init__" method of "GAB" is missing the "{required_arg}" argument. Automatically added by the reformatter.\n')
                            new_args.append(ast.arg(arg=required_arg, annotation=None))
                            added_args.add(required_arg)
                        else:
                            new_args.append(item.args.args[current_args.index(required_arg)])
                            added_args.add(required_arg)

                    # Add remaining arguments that are not in the required list
                    for arg in item.args.args:
                        if arg.arg not in added_args:
                            new_args.append(arg)

                    item.args.args = new_args

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
                                    'The super().__init__(embed_dim, block_loc) call in GAB is force overwritten by the reformatter. It may cause error if you modified this line.\n'
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
                                    args=[
                                        ast.Name(id='embed_dim', ctx=ast.Load()),
                                        ast.Name(id='block_loc', ctx=ast.Load())
                                    ],
                                    keywords=[]
                                )
                                break

                    if not found_super:
                        self.warnings.append(
                            'The super().__init__(embed_dim, block_loc) call is missing in the __init__ method. Automatically added by the reformatter.\n'
                        )
                        # Insert super().__init__(embed_dim, block_loc) at the start of the __init__ method
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
                                args=[
                                    ast.Name(id='embed_dim', ctx=ast.Load()),
                                    ast.Name(id='block_loc', ctx=ast.Load())
                                ],
                                keywords=[]
                            )
                        )
                        item.body.insert(0, super_call)

                if isinstance(item, ast.FunctionDef) and item.name == '_forward':
                    self.found__forward=True
                    # Check if **kwargs or **intermediate_vars is in the function arguments
                    if item.args.kwarg is None:
                        self.warnings.append(f'The "_forward" method of "GAB" is missing the "**intermediate_vars" argument. Automatically adding the argument.\n')
                        # Add **intermediate_vars to the function arguments
                        item.args.kwarg = ast.arg(arg='intermediate_vars', annotation=None)    
        return node

class GABCleaner(ast.NodeTransformer):
    def __init__(self):
        self.warnings = []
        self.errors = []

    def visit_Module(self, node):
        # Only keep Import, ImportFrom, FunctionDef, ClassDef, and gab_config nodes
        new_body = []
        for n in node.body:
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Try)):
                new_body.append(n)
            elif isinstance(n, ast.Assign) and (len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'gab_config'):
                new_body.append(n)
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                continue
            else:
                self.warnings.append(f'The statement "{astor.to_source(n).strip()}" is removed by the reformatter.\n')
        node.body = new_body
        return node   
    
    def visit_Try(self, node):
        # Visit the body and handlers of the try-except block
        new_body = []
        for n in node.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                new_body.append(n)
            else:
                self.warnings.append(f'The statement in try block "{astor.to_source(n).strip()}" is removed by the reformatter.\n')

        node.body = new_body
        # Keep all except handlers
        for handler in node.handlers:
            handler.body = [n for n in handler.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        
        return node if new_body or node.handlers else None  # Remove the entire try block if empty


class GABFormatChecker:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.hints = []
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
        
        code_ast = self._reformat(code_ast) # NOTE: maybe also check docstrings
        
        # Update self.gab_code with the modified AST
        self.gab_code = astor.to_source(code_ast)
        print(f'Code after reformatted:\n\n{self.gab_code}\n\n')

        # Execute the modified AST
        try: ####NOTE: LEAVE IT OPEN FOR DEBUGGING CAUSE, REMEMBER TO UNCOMMENT
            exec(self.gab_code, globals().copy(), local_ns)
        except Exception as e:
            self.errors.append(f'The code is not executable:\n{str(e)}\nHint: 1. Check whether you have implement redundant part, remember that you only need to implement the autoregressive block, not the whole model like embedding layer and lm head. \n')
            self.hints.append('REFRESH_TEMPLATE')
            return self._report_errors()
        
        # Check for gab_config dictionary
        self._check_gab_config_dictionary(local_ns, code_ast)

        # Report all errors
        return self._report_errors()
    

    def apply_visitor(self, visitor, code_ast):
        code_ast = visitor.visit(code_ast)
        self.warnings.extend(visitor.warnings)
        self.errors.extend(visitor.errors)
        return code_ast, visitor

    def _reformat(self, code_ast):
        code_ast,corrector = self.apply_visitor(GABCorrector(), code_ast)
        code_ast,_ = self.apply_visitor(GABCleaner(), code_ast)

        # Check and add the import if necessary
        if not corrector.found_import:
            self.warnings.append('The import for "GABBase" from "model_discovery.model.utils.modules" is missing. Automatically adding the import.\n')
            import_node = ast.ImportFrom(
                module='model_discovery.model.utils.modules',
                names=[ast.alias(name='GABBase', asname=None)],
                level=0
            )
            code_ast.body.insert(0, import_node)

        if not corrector.found_GAB:
            self.errors.append('The class "GAB" is not defined in the provided code.\n')
            self.hints.append('REFRESH_TEMPLATE')
        
        if not corrector.found__forward:
            self.errors.append(f'The method "_forward" is not defined in the class "GAB" or "GAB" is not defined.\n')
            self.hints.append('REFRESH_TEMPLATE')

        if not corrector.found__init__:
            self.errors.append(f'The method "__init__" is not defined in the class "GAB" or "GAB" is not defined.\n')
            self.hints.append('REFRESH_TEMPLATE')

        return code_ast


    def _check_gab_config_dictionary(self, local_ns, code_ast) -> None:
        if not 'gab_config' in local_ns:
            self.errors.append('The dictionary "gab_config" is not defined.\n')
            self.hints.append('REFRESH_TEMPLATE')
            return 
        
        gab_config = local_ns['gab_config']
        if not isinstance(gab_config, dict):
            self.errors.append('"gab_config" should be a dictionary.\n')
            self.hints.append('REFRESH_TEMPLATE')
            return 
        
        if 'GAB' not in local_ns:
            self.errors.append('The class "GAB" is not defined in the provided code, cannot validate "gab_config".\n')
            self.hints.append('REFRESH_TEMPLATE')
            return

        gab_class = local_ns['GAB']
        init_signature = inspect.signature(gab_class.__init__)
        init_parameters = init_signature.parameters

        excluded_args = {'self','embed_dim', 'block_loc', 'device', 'dtype', 'kwargs'}
        required_args = {name for name, param in init_parameters.items() if name not in excluded_args and param.default == inspect.Parameter.empty}
        optional_args = {name for name, param in init_parameters.items() if name not in excluded_args and param.default != inspect.Parameter.empty}

        config_args = set(gab_config.keys())

        missing_args = required_args - config_args
        redundant_args = config_args - (required_args | optional_args)
        default_args = optional_args - config_args

        if missing_args:
            missing_args = [f'"{arg}"' for arg in missing_args]
            self.errors.append(f'The dictionary "gab_config" is missing the following required arguments: {", ".join(missing_args)} in "GAB.__init__".\n')

        if default_args:
            default_args = [f'"{arg}"' for arg in default_args]
            self.warnings.append(f'These args are not set by gab_config and directly use the default value: {", ".join(default_args)}. Ignore this if it is intended.\n')

        if redundant_args:
            redundant_args = [f'"{arg}"' for arg in redundant_args]
            self.warnings.append(f'The dictionary "gab_config" contains redundant arguments: {", ".join(redundant_args)} not used or not allowed to be re-defined in "GAB.__init__". They are automatically removed by the reformatter.\n')
            class ConfigModifier(ast.NodeTransformer):
                def visit_Assign(self, node):
                    if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'gab_config':
                        if isinstance(node.value, ast.Dict):
                            new_keys = []
                            new_values = []
                            for key, value in zip(node.value.keys, node.value.values):
                                if isinstance(key, ast.Constant) and key.s not in redundant_args:
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
        return not bool(self.errors),report,self.hints,self.gab_code


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
    def __init__(self,embed_dim: int, block_loc: tuple, n_heads, device=None,dtype=None,**kwargs):
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__(embed_dim, block_loc)
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
        self.hints = []
        self.ds, self.tokenizer = load_datasets_args(
            DEFAULT_TOKENIZER,
            DEFAULT_CONTEXT_LENGTH,
            ['wikitext2']
        )
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def get_benchmark(self,config):
        exec(BENCHMARK_MODEL,globals())
        glm,_ = reload_gam(config,BENCHMARK_MODEL,'BENCHMARK_MODEL')#,dtype=torch.bfloat16, device="cuda") # intentially use bfloat16 to check whether the model is correctly defined
        if torch.cuda.is_available():
            glm = glm.cuda()
            glm=glm.to(torch.bfloat16)
        else:
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
        if run_time>benchmark['run_time']*50: # NOTE: MAKE IT REALLY LOOSE NOW
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


    def check_complexity(self, config, model) -> None: # NOTE: NEED TO CONSIDER THAT VARIANT CONTEXT LENGTH MAY IMPACT MODEL WITH *MEMORY*
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
                output,_ = gab(X)
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
        ckpt_dir=os.environ.get("CKPT_DIR")
        training_args=TrainingArguments(
            output_dir=f'{ckpt_dir}/temp/ve/effective_check',
            overwrite_output_dir=True,
            learning_rate=config.learning_rate,
            save_strategy='no',
            max_steps=5,
            per_device_train_batch_size=2,
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
        return not bool(self.errors),report,self.results,self.hints
    
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
        self.hints = []
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
            Y,_ = block(X)

        print('Checking causality... It checks the causality by changing all future steps X[t+delta] of X[t] and see if Y[t] or any previous outputs change.')
        bar = tqdm(range(seq_len), desc='Causality test', colour='green')
        for t in bar:
            X_mod = X.clone()
            X_mod[:, t + 1:, :]*=-1 # Perturb the future steps of X[t]

            with torch.no_grad():
                Y_mod,_ = block(X_mod)
                        
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
        mock_input = torch.randint(0, vocab_size, (2, DEFAULT_CONTEXT_LENGTH)).cuda() if \
          torch.cuda.is_available() else torch.randint(0, vocab_size, (2, DEFAULT_CONTEXT_LENGTH))
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
                self.rprint(f"Error: Parameter {name} does not have a gradient. Hint: Check whether you have any unused parameters.")
                return False

        self.rprint('Differentiability test passed')
        return True
    

    def _check_effectiveness(self, model, config) -> bool:
        self.rprint('Checking effectiveness...')
        checkpass,errors,results,hints=self.effective_checker.check(config, model)    
        self.hints += hints
        self.rprint(errors)
        return checkpass,results
    
    
    def _check_format_and_reformat(self, gab_code: str) -> bool:
        """Check if the model format is correct 

        :param gab_code: 
            The code of the new block. 

        """
        self.rprint('Checking code format...')
        checkpass,errors,hints,gab_code=self.format_checker.check(gab_code)
        self.hints += hints
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
        mock_input = torch.randint(0, vocab_size, (2, DEFAULT_CONTEXT_LENGTH)).cuda() if \
          torch.cuda.is_available() else torch.randint(0, vocab_size, (2, DEFAULT_CONTEXT_LENGTH))
        mock_input = mock_input.to(gab.device)
        emb.eval()
        gab.eval()
        try:
            with torch.no_grad():
                input=emb(mock_input)
                output,_ = gab(input)
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
        # torch.cuda.empty_cache()
        time_start=time.time()
        self.reset()
        with U.CodeTimer("Format checking"):
            self.rprint('Checking the designed model...')
            try:
                checkpass,gab_code = self._check_format_and_reformat(gab_code)
                assert checkpass
            except AssertionError:
                check_report=self.get_report(gab_code)
                return False,check_report,gab_code,{'hints': self.hints}
        
        with U.CodeTimer("Model initialization"): # NOTE: very time consuming for the first time, but luckily only happens for the very first run, maybe reduce the time of first run>
            try: 
                exec(gab_code,globals())
                glm,_ = reload_gam(config,gab_code,name)
                if torch.cuda.is_available():
                    t0=time.time()
                    glm = glm.cuda() # this step super slow!!! why??? Any solution???
                    print(f'Time for moving model to GPU: {time.time()-t0:.2f}s')
                    glm=glm.to(torch.bfloat16) # intentially use bfloat16 to check whether the model is correctly defined
                else:
                    glm=glm.to(torch.float16)
                mock_input=torch.randint(0, config.vocab_size, (2, DEFAULT_CONTEXT_LENGTH))
                mock_input = mock_input.to(glm.device)
                t0=time.time()
                glm(mock_input) # super slow as well, why??? but its only for the first time initialization
                print(f'Time for the first forward pass: {time.time()-t0:.2f}s')

        
            except Exception as e:
                error_trace = traceback.format_exc()
                self.rprint(
                    'Error: Model initialization failed with error: '+str(e)+'\n'
                    'Full Traceback: \n' + error_trace + '\n'
                    'Hint: 1. if it is a dtype or device error, check whether the factory kwargs are passed to the layers, and whether you manually designate a type instead of apply the type from factory kwargs or the input\'s type during conversion or creating of an variable. '
                    '2. If it is a shape error, check whether the output shape is equal to the input shape. The output shape of GAB should be the same as the input. '
                    '3. Always remember to follow the template and do not implement redundant part like embedding layer. '
                )
                self.hints.append('REFRESH_TEMPLATE')
                check_report=self.get_report(gab_code)
                return False,check_report,gab_code,{'hints': self.hints}
        
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
        effectiveness = None
        with U.CodeTimer("Model functional tests"):
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
                assert checkpass2 and checkpass3
                # checkpass4,effectiveness=self._check_effectiveness(glm,config)
                # assert checkpass4
            except Exception as e:# AssertionError:
                self.rprint(f'Model test failed\n{e}')
                if not checkpass2:
                    self.rprint('Hint: If you used convolutional layer, you should consider that the conv kernel may cover the future steps. '
                                'You can add padding and truncation of future steps to the conv layer to make it causal.\n')
                check_report=self.get_report(gab_code)
                return False,check_report,gab_code,{'hints': self.hints}

        self.rprint("All tests passed!\n")
        time_end=time.time()
        print(f'[Total time for checking: {time_end-time_start:.2f}s]')

        results = {
            'log': self.report,
            'effectiveness': effectiveness,
            'hints': self.hints
        }
        check_report=self.get_report(gab_code)
        return True,check_report,gab_code,results
    
    
    def get_report(self,gab_code):
        gabcode_lines=gab_code.split('\n')
        new_check_report=[]
        for line in self.report.split('\n'):
            if 'File "<string>", line' in line:
                line=line.replace('File "<string>", line','File "gab.py", line')
                line_num=int(line.split('File "gab.py", line ')[-1].split(',')[0].strip())
                line=line.replace(f'line {line_num}',f'line {line_num}: {gabcode_lines[line_num-1]}')
            new_check_report.append(line)
        check_report='\n'.join(new_check_report)
        return check_report


    def tune(self,config,gab_code,name,tune_dim=True)->str: # the model is already correct but we need to tune its scale
        print('Tuning the model scale...')
        d_model=config.d_model
        n_block=config.n_block
        # assert d_model%128==0 # initial d_model from config should be a multiple of 128
        vocab_size=config.vocab_size
        reference_size=config.reference_size
        threshold=config.size_threshold
        UB=int(reference_size*(1+threshold))
        LB=int(reference_size*(1-threshold))
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
            min_step=48
        else:
            min_step=32
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
                mock_input=torch.randint(0, vocab_size, (2, DEFAULT_CONTEXT_LENGTH)).to(glm.device)
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

