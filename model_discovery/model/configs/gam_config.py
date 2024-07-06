from dataclasses import dataclass, field, asdict


@dataclass
class GAMConfig:
    '''Configurations for Generalized Autoregressive Models.'''

    d_model: int
    n_layer: int # d_model and n_layers are provided, so the agent can focus on the block design instead of HPO
    param_magnitude: int # The magnitude of the non-emb parameters, e.g., 1e7, 3.5e7, param num should not exceed it (or with some threshold) 
    context_length: int
    param_threshold: float = 0.2 # ratio of how many param num can exceed or below the magnitude
    vocab_size: int = 50277
    training_token_multiplier: int = 20
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    def print_config(self):
        prints=f'''
            d_model: {self.d_model}
            n_layer: {self.n_layer}
            param_magnitude: {self.param_magnitude} 
            context_length: {self.context_length}
            param_threshold: {self.param_threshold}
            vocab_size: {self.vocab_size}
            training_token_multiplier: {self.training_token_multiplier}
            rms_norm: {self.rms_norm}
            residual_in_fp32: {self.residual_in_fp32}
            fused_add_norm: {self.fused_add_norm}
            pad_vocab_size_multiple: {self.pad_vocab_size_multiple}
            tie_embeddings: {self.tie_embeddings}
        '''
        return prints
    

    def to_dict(self):
        return asdict(self)



@dataclass
class GAMConfig_10M(GAMConfig):
    '''Configurations for Generalized Autoregressive Model with 10M scale (non-embedding).'''

    d_model: int = 384
    n_layer: int = 6 
    param_magnitude: int = 1e7 
    context_length: int = 1024
    
