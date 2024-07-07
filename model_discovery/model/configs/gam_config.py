from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict
from transformers import AutoTokenizer


@dataclass
class GAMConfig:
    '''Configurations for Generalized Autoregressive Models.'''

    d_model: int
    n_layer: int # d_model and n_layers are provided, so the agent can focus on the block design instead of HPO
    param_magnitude: int # The magnitude of the non-emb parameters, e.g., 1e7, 3.5e7, param num should not exceed it (or with some threshold) 
    context_length: int
    training_data: List[str]
    training_weight: Dict[str, List[float]] = None, # e.g., {'train':[1.5,1.0]} # reweighting the datasets
    param_threshold: float = 0.2 # ratio of how many param num can exceed or below the magnitude
    tokenizer: str = 'meta-llama/Llama-2-7b-hf' # pretrained tokenizer name
    training_token_multiplier: int = 1000 # Chinchilla suggests 20
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    def __post_init__(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer) 
        self.vocab_size = tokenizer.vocab_size # around 32000 for llama

    def to_str(self):
        return "\n".join(f"{key}: {value}" for key, value in self.to_dict().items())
    
    def to_dict(self):
        return asdict(self)


@dataclass
class GAMConfig_10M(GAMConfig):
    '''Configurations for Generalized Autoregressive Model with 10M scale (non-embedding).'''

    
    d_model: int = 224
    n_layer: int = 6 
    param_magnitude: int = 1e7 # budget for each block is around (param_magnitute - vocab_size x d_model)/n_layer
    context_length: int = 512
    training_data: List[str] = field(default_factory=lambda: ['babylm','tinystories'])
    per_device_train_batch_size: int = 256
    learning_rate: float = 1e-4
