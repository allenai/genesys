from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from transformers import PretrainedConfig, AutoTokenizer

from .const import *


# NOTE: training_data,eval_tasks,tokenizer,context_length can be overwritten by args                   

@dataclass
class GAMConfig(PretrainedConfig):
    '''Configurations for Generalized Autoregressive Models.'''

    d_model: int
    n_block: int
    scale: str
    reference_size: int # a reference param num based on GPT
    batch_tokens: int 
    learning_rate: float
    context_length: int = DEFAULT_CONTEXT_LENGTH
    training_data: List[str] = field(default_factory=lambda: SMOLLM_125_CORPUS)
    eval_tasks: List[str] = field(default_factory=lambda: [
        *DEFAULT_TASK_LIST1,
    ])
    vocab_size: int = None
    # training_token_multiplier: int = 20 # Now desinated by ve args
    training_weight: Dict[str, List[float]] = None
    size_threshold: float = 0.2 # e.g. Pythia 410M vs GPT 350M
    tokenizer: str = DEFAULT_TOKENIZER
    rms_norm: bool = True ### triton stuff
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    use_template: bool = False
    per_device_batch_size: int = None # Will overwrite batch_tokens if set, SHOULD NEVER BE SET IN EVOLUTIONARY MODE
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        super().__init__()  # Initialize superclass with necessary arguments
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
            self.vocab_size = tokenizer.vocab_size  # around 32000 for llama
        except:
            assert self.tokenizer == DEFAULT_TOKENIZER, f'{self.tokenizer} not found, please check your HF_KEY'
            self.vocab_size = 32000

    def to_str(self): 
        return "\n".join(f"{key}: {value}" for key, value in self.to_dict().items())

    def to_dict(self):
        return asdict(self)
    
    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: '{key}' is not a recognized configuration key and will be ignored.")
        return self
    
    def print_config(self):
        return self.to_str()
    
    def to_prompt(self):
        return (
            f"scale: {self.scale} # The target scale of the GAM model\n"
            f"vocab_size: {self.vocab_size} # The number of terms in the embedding table\n"
            f"reference size: {self.reference_size} # The number of parameters in the reference model with GPT architecture, the designed model should not diviated from this size too much (notice that the param number shown here is based on using a tied embedding param and llama tokenizer which may have a large difference from the target scale when the scale is small)\n"
            f"reference n_block: {self.n_block} # The number of GAB blocks in this GAM model\n"
            f"reference d_model: {self.d_model} # The d_model applied by the GPT reference model\n"
            f"Note: the number of params is computed as d_model * vocab_size + n_block * param_num_of_gab_block\n"
        )

# Configs below are come from GPT-3 paper https://arxiv.org/pdf/2005.14165, applied by Mamba, TTT
# The major difference between GPT-3 and Pythia setting is on 760M, where Pythia used an 1B model instead, other differences including lr, bs
# Notice that due to the use of Llama tokenizer and tied params, the reference size looks different 


@dataclass
class GAMConfig_14M(GAMConfig):
    scale: str = '14M'
    d_model: int = 128
    n_block: int = 6
    reference_size: int = 5280384
    eval_batch_size: int = 512
    learning_rate: float = 1e-3
    batch_tokens: int = 1024*512 # 0.5M tokens



@dataclass
class GAMConfig_31M(GAMConfig):
    scale: str = '31M'
    d_model: int = 256
    n_block: int = 6
    reference_size: int = 12920064
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    batch_tokens: int = 1024*512 # 0.5M tokens


@dataclass
class GAMConfig_70M(GAMConfig):
    scale: str = '70M'
    d_model: int = 512
    n_block: int = 6
    reference_size: int = 35277312
    eval_batch_size: int = 512
    learning_rate: float = 1e-3
    batch_tokens: int = 1024*512 # 0.5M tokens


@dataclass
class GAMConfig_125M(GAMConfig):
    scale: str = '125M'
    d_model: int = 768
    n_block: int = 12
    reference_size: int = 109566720
    eval_batch_size: int = 512
    learning_rate: float = 6e-4
    batch_tokens: int = 1024*512 # 0.5M tokens
    gradient_accumulation_steps: int = 2
    

@dataclass
class GAMConfig_350M(GAMConfig):
    scale: str = '350M'
    d_model: int = 1024
    n_block: int = 24
    reference_size: int = 334906368
    eval_batch_size: int = 512
    learning_rate: float = 3e-4
    batch_tokens: int = 1024*512 # 0.5M tokens
    gradient_accumulation_steps: int = 4


@dataclass
class GAMConfig_760M(GAMConfig):
    scale: str = '760M'
    d_model: int = 1536
    n_block: int = 24
    reference_size: int = 728851968
    eval_batch_size: int = 512
    learning_rate: float = 2.5e-4
    batch_tokens: int = 1024*512 # 0.5M tokens
    gradient_accumulation_steps: int = 8


@dataclass
class GAMConfig_1300M(GAMConfig):
    scale: str = '1300M'
    d_model: int = 2048
    n_block: int = 24
    reference_size: int = 1273792512
    eval_batch_size: int = 512
    learning_rate: float = 2e-4
    batch_tokens: int = 1024*1024 # 1M tokens
    gradient_accumulation_steps: int = 16


@dataclass
class GAMConfig_2700M(GAMConfig):
    scale: str = '2.7B'
    d_model: int = 2560
    n_block: int = 32
    reference_size: int = 2598996480
    eval_batch_size: int = 512
    learning_rate: float = 1.6e-4
    batch_tokens: int = 1024*1024 # 1M tokens

@dataclass
class GAMConfig_6700M(GAMConfig):
    scale: str = '6.7B'
    d_model: int = 4096
    n_block: int = 32
    reference_size: int = 6574313472
    eval_batch_size: int = 512
    learning_rate: float = 1.2e-4
    batch_tokens: int = 2*1024*1024 # 2M tokens


@dataclass
class GAMConfig_13B(GAMConfig):
    scale: str = '13B'
    d_model: int = 5120
    n_block: int = 40
    reference_size: int = 12747985920
    eval_batch_size: int = 512
    learning_rate: float = 1e-4
    batch_tokens: int = 2*1024*1024 # 2M tokens


@dataclass
class GAMConfig_175B(GAMConfig):
    scale: str = '175B'
    d_model: int = 12288
    n_block: int = 96
    reference_size: int = 175e9 # too large to initialize
    eval_batch_size: int = 512
    learning_rate: float = 0.6e-4
    batch_tokens: int = 3200*1024 # 3.2M tokens


@dataclass
class GAMConfig_1T(GAMConfig): # Just for fun
    scale: str = '1T'
    d_model: int = 20480
    n_block: int = 200
    reference_size: int = 1e12 
    eval_batch_size: int = 512
    learning_rate: float = 0.3e-4
    batch_tokens: int = 6400*1024 # 6.4M tokens






# Debugging configuration


@dataclass
class GAMConfig_debug(GAMConfig_14M):
    scale: str = 'debug'
    training_data: List[str] = field(default_factory=lambda: [
        "fineweb-edu-dedup",  
    ])
    eval_tasks: List[str] = field(default_factory=lambda: [
        "smollm125-tiny",  
        # "squad_completion",
    ])
    # eval_tasks: List[str] = field(default_factory=lambda: [
    #     "lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande",
    #     "blimp", # "blimp_filtered","blimp_supplement"
    # ])
    # rms_norm: bool = False 
    # fused_add_norm: bool = False # TRITON BUGGY
    # use_template: bool = False
    

