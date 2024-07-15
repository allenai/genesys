
# gab.py

import torch
import torch.nn as nn

from .block_registry import BlockRegister

__all__ = [
    "GAB",
]

@BlockRegister(
    name="default",
    config={}
)
class GAB(nn.Module):
    """Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    """
    def __init__(self,embed_dim: int,layer_idx: int,device=None,dtype=None,**kwargs):
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx 
        
        # Define different types of blocks based on layer_idx
        if layer_idx % 3 == 0:
            self.block = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4, **factory_kwargs),
                nn.ReLU(),
                nn.Linear(embed_dim * 4, embed_dim, **factory_kwargs)
            )
        elif layer_idx % 3 == 1:
            self.block = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, padding=1, **factory_kwargs),
                nn.ReLU(),
                nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=3, padding=1, **factory_kwargs)
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2, **factory_kwargs),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim, **factory_kwargs)
            )

    def _forward(self,X,**kwargs): # type hints are optional but recommended
        """Forward pass of the model"""
        assert X.shape[-1] == self.embed_dim
        
        if self.layer_idx % 3 == 1:
            X = X.transpose(1, 2)  # For Conv1d, change shape to (batch, embed_dim, seqlen)
            Y = self.block(X)
            Y = Y.transpose(1, 2)  # Change back to (batch, seqlen, embed_dim)
        else:
            Y = self.block(X)
        
        return Y
     
    def forward(self,X,**kwargs):
        """Forward pass of the model"""
        Y=self._forward(X,**kwargs)
        assert Y.shape[-1] == self.embed_dim
        return Y
    
def gab_config()->dict:
    """Returns a dictionary of hyperparameters for constructing a GAB layer
        embed_dim, layer_idx, device, dtype should not be included in the dictionary which will be provided by the model
    """
    return {
        "param_magnitude": 10000000.0,
        "context_length": 512,
        "training_data": ['babylm', 'tinystories'],
        "eval_tasks": ['lambada_openai', 'hellaswag', 'piqa', 'arc_easy', 'arc_challenge', 'winogrande', 'blimp_filtered', 'blimp_supplement'],
        "vocab_size": 32000,
        "training_weight": None,
        "param_threshold": 0.2,
        "tokenizer": 'meta-llama/Llama-2-7b-hf',
        "training_token_multiplier": 20,
        "rms_norm": False,
        "residual_in_fp32": True,
        "fused_add_norm": False,
        "pad_vocab_size_multiple": 8,
        "tie_embeddings": True,
        "per_device_train_batch_size": 256,
        "eval_batch_size": 512,
        "learning_rate": 0.0001
    }
