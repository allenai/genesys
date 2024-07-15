from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from transformers import PretrainedConfig, AutoTokenizer



@dataclass
class GAMConfig(PretrainedConfig):
    '''Configurations for Generalized Autoregressive Models.'''

    d_model: int
    n_layer: int
    param_magnitude: int
    context_length: int
    training_data: List[str]
    eval_tasks: List[str] = field(default_factory=lambda: ["lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande","blimp_filtered","blimp_supplement"])
    vocab_size: int = None
    training_weight: Dict[str, List[float]] = None
    param_threshold: float = 0.2
    tokenizer: str = 'meta-llama/Llama-2-7b-hf'
    training_token_multiplier: int = 2 # Train fewer steps to fast test
    rms_norm: bool = False ### triton stuff,
    residual_in_fp32: bool = True
    fused_add_norm: bool = False
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

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
    

@dataclass
class GAMConfig_10M(GAMConfig):
    '''Configurations for Generalized Autoregressive Model with 10M scale (non-embedding).'''

    d_model: int = 256
    n_layer: int = 6
    param_magnitude: int = 1e7
    context_length: int = 512
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    per_device_train_batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 1e-4



@dataclass
class GAMConfig_35M(GAMConfig):
    '''Configurations for Generalized Autoregressive Model with 10M scale (non-embedding).'''

    d_model: int = 512
    n_layer: int = 8
    param_magnitude: int = 3.5e7
    context_length: int = 512
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    per_device_train_batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 1e-4



@dataclass
class GAMConfig_70M(GAMConfig):
    '''Configurations for Generalized Autoregressive Model with 10M scale (non-embedding).'''

    d_model: int = 256
    n_layer: int = 6
    param_magnitude: int = 1e7
    context_length: int = 512
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    per_device_train_batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 1e-4



@dataclass
class GAMConfig_130M(GAMConfig):
    '''Configurations for Generalized Autoregressive Model with 10M scale (non-embedding).'''

    d_model: int = 256
    n_layer: int = 6
    param_magnitude: int = 1e7
    context_length: int = 512
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    per_device_train_batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 1e-4


# 370M

# 780M

# 1.3B

# 2.7B

# 7B

# 13B or 8x7B (active params)

# 35B or 8x22B

# 70B 

# 175B


@dataclass
class GAMConfig_debug(GAMConfig):
    '''Configurations for Generalized Autoregressive Model with 10M scale (non-embedding).'''

    d_model: int = 256
    n_layer: int = 6
    param_magnitude: int = 1e7
    context_length: int = 512
    training_data: List[str] = field(default_factory=lambda: ['babylm', 'tinystories'])
    per_device_train_batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 5e-3 # LR for BS=256 and 6 GPUs 20x tokens
    training_token_multiplier: int = 20
    eval_tasks: List[str] = field(default_factory=lambda: ["blimp_filtered","blimp_supplement","glue",
        "lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande"])
    rms_norm: bool = False # TRITON BUGGY
    fused_add_norm: bool = False # TRITON BUGGY
