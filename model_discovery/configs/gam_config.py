from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from transformers import PretrainedConfig, AutoTokenizer



@dataclass
class GAMConfig(PretrainedConfig):
    '''Configurations for Generalized Autoregressive Models.'''

    d_model: int
    n_block: int
    reference_size: int # a reference param num based on GPT
    training_data: List[str]
    batch_tokens: int 
    context_length: int = 2048
    eval_tasks: List[str] = field(default_factory=lambda: ["lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande","blimp_filtered","blimp_supplement"])
    vocab_size: int = None
    training_weight: Dict[str, List[float]] = None
    size_threshold: float = 0.2
    tokenizer: str = 'meta-llama/Llama-2-7b-hf'
    training_token_multiplier: int = 20
    rms_norm: bool = True ### triton stuff
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    use_template: bool = False
    use_rope: bool = True
    per_device_batch_size: int = None # Will overwrite batch_tokens if set

    def __post_init__(self):
        super().__init__()  # Initialize superclass with necessary arguments
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        self.vocab_size = tokenizer.vocab_size  # around 32000 for llama

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


# Configs below are come from GPT-3 paper https://arxiv.org/pdf/2005.14165, applied by Mamba, TTT
# The major difference between GPT-3 and Pythia setting is on 760M, where Pythia used an 1B model instead, other differences including lr, bs
# Notice that due to the use of Llama tokenizer and tied params, the reference size looks different 

@dataclass
class GAMConfig_14M(GAMConfig):
    d_model: int = 128
    n_block: int = 6
    reference_size: int = 5280384
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 1e-3
    batch_tokens: int = 1024*512 # 0.5M tokens


@dataclass
class GAMConfig_31M(GAMConfig):
    d_model: int = 256
    n_block: int = 6
    reference_size: int = 12920064
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    batch_tokens: int = 1024*512 # 0.5M tokens


@dataclass
class GAMConfig_70M(GAMConfig):
    d_model: int = 512
    n_block: int = 6
    reference_size: int = 35277312
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 1e-3
    batch_tokens: int = 1024*256 # 0.5M tokens


@dataclass
class GAMConfig_125M(GAMConfig):
    d_model: int = 768
    n_block: int = 12
    reference_size: int = 109566720
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 6e-4
    batch_tokens: int = 1024*512 # 0.5M tokens
    

@dataclass
class GAMConfig_350M(GAMConfig):
    d_model: int = 1024
    n_block: int = 24
    reference_size: int = 334906368
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 3e-4
    batch_tokens: int = 1024*512 # 0.5M tokens


@dataclass
class GAMConfig_760M(GAMConfig):
    d_model: int = 1536
    n_block: int = 24
    reference_size: int = 728851968
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 2.5e-4
    batch_tokens: int = 1024*512 # 0.5M tokens


@dataclass
class GAMConfig_1300M(GAMConfig):
    d_model: int = 2048
    n_block: int = 24
    reference_size: int = 1273792512
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 2e-4
    batch_tokens: int = 1024*1024 # 1M tokens


@dataclass
class GAMConfig_2700M(GAMConfig):
    d_model: int = 2560
    n_block: int = 32
    reference_size: int = 2598996480
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 1.6e-4
    batch_tokens: int = 1024*1024 # 1M tokens


@dataclass
class GAMConfig_6700M(GAMConfig):
    d_model: int = 4096
    n_block: int = 32
    reference_size: int = 6574313472
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 1.2e-4
    batch_tokens: int = 2*1024*1024 # 2M tokens



@dataclass
class GAMConfig_13B(GAMConfig):
    d_model: int = 5120
    n_block: int = 40
    reference_size: int = 12747985920
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 1e-4
    batch_tokens: int = 2*1024*1024 # 2M tokens



@dataclass
class GAMConfig_175B(GAMConfig):
    d_model: int = 12288
    n_block: int = 96
    reference_size: int = 175e9 # too large to initialize
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 0.6e-4
    batch_tokens: int = 3200*1024 # 3.2M tokens



@dataclass
class GAMConfig_1T(GAMConfig): # Just for fun
    d_model: int = 20480
    n_block: int = 200
    reference_size: int = 1e12 
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    eval_batch_size: int = 512
    learning_rate: float = 0.3e-4
    batch_tokens: int = 6400*1024 # 6.4M tokens



# Debugging configuration


@dataclass
class GAMConfig_debug(GAMConfig_31M):
    context_length: int = 512
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    training_token_multiplier: int = 20
    eval_tasks: List[str] = field(default_factory=lambda: ["blimp_filtered","blimp_supplement","glue",
        "lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande"])
    rms_norm: bool = False 
    fused_add_norm: bool = False # TRITON BUGGY
    batch_tokens: int = 1024*768 
    use_template: bool = True
    per_device_batch_size: int = 256
    


