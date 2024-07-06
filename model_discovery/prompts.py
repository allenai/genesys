from dataclasses import dataclass, field, asdict


DESIGNER_PROMPT="""Design a novel autoregressive model block by completing the blanks marked in the python code file gab.py below, which includes the initialization where you can define your custom arguments, the forward function where you can define convenient functions in the GAB class such as caches, the configuration with the hyperparameters that correspond to the arguments you defined:

{gab_py}

This code will be used to construct a gam model in gam.py:

{gam_py}

This is the configuration for the model:

{config}

Here are some hints:         
1. You can use layer_idx to create arbitrary model structure with different types of blocks, examples:
    * create 1 type of block for all layers, you can ignore layer_idx
    * create 2 different types of blocks for layers with layer_idx%2=0,1
    * create 3 different types of blocks for layers with layer_idx%3=0,1,2
    * create 3 different types of blocks A,B,C, has AABC structure, then you can let layer_idx%4=0,1 for A, 2 for B, 3 for C
2. Use different types of blocks is not required also there is no preference for heterogeneous or homogeneous blocks, you can choose to create only one type of block or multiple types of blocks.
3. The gam model alraedy wrap the GAB blocks with residual connections and normalization, so when you design the block, you need to consider that and avoid redundant design.
4. The parameter number of the layers should follow the magnitude by param_magnitude, and can not exceed or below it by param_threshold. You can achieve it through adjusting design or tuning hyperparameters. You may need to do math to estimate the parameter number before chosing the hyperparameters. You can estimate multiple times in your response until you find the proper hyperparameters.
5. The model should be able to be parallel trained, which means you should not introduce recurrent operators like RNN or LSTM.
6. The design should be novel, you are not allowed to simply apply an existing design such as transformer block, you need to design your own block.

{instruct}

Now, use the information provided above to complete the code. You are not allowed to change anything besides the GAB class in gab.py.

Your response should include the full gab.py file with the completed code. You should derive your design step by step with detailed analysis and explanation. Specifically, when providing the full gab.py file, please preserve # gab.py at the beginning of the file.
"""

REVIEWER_PROMPT="""This is the proposal of the design of the general autoregressive block (gab) for you to review:

{proposal}

{instruct}

Now, carefully review the design and give the feedback in a step by step way. You must return as a json file with two keys: 'review' and 'rating'. 
The 'review' key should contain a detailed feedback of the design written in markdown, and the 'rating' key should contain the rating of the design from 1 to 5.
"""

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

GAB_TEMPLATE = """
# gab.py

import torch
import torch.nn as nn

class GAB(nn.Module):
    '''Generalized Autoregressive Block
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
        Constraints:  Causal, differentiable, parameter number, complexity, parallelizable
    '''
    def __init__(self,embed_dim: int,layer_idx: int,device=None,dtype=None,**kwargs):
        # argv: list of hyperparameters
        factory_kwargs = {"device": device, "dtype": dtype} # remember to pass it to nn layers
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx 
        # COMPLETING THE CODE HERE #
        raise NotImplementedError
    

    def _forward(self,X,**kwargs): # type hints are optional but recommended
        '''Forward pass of the model'''
        assert X.shape[-1] == self.embed_dim
        # COMPLETING THE CODE HERE #
        raise NotImplementedError
     
    def forward(self,X,**kwargs):
        '''Forward pass of the model'''
        Y=self._forward(X,**kwargs)
        assert Y.shape[-1] == self.embed_dim
        return Y
    
def gab_config()->dict:
    '''Returns a dictionary of hyperparameters for constructing a GAB layer
        embed_dim, layer_idx, device, dtype should not be included in the dictionary which will be provided by the model
    '''
    # COMPLETING THE CODE HERE #
    raise NotImplementedError

"""

GAM_MODEL = """class GAM(nn.Module):
    ''' Generalized Autoregressive Models
        Input:        X: (batch, seqlen, embed_dim)
        Output:       Y: (batch, seqlen, embed_dim)
    '''
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int = 50277,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg = None,
        fused_add_norm = False,
        residual_in_fp32 = False,
        device = None,
        dtype = None,
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, **self.factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        block_config = gab_config()
        # if n_layer is None:
        #     assert 'n_layer' in block_config, "n_layer must be provided if not in block_config"
        #     n_layer = block_config['n_layer']
        self.n_layer = n_layer
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    block_config,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **self.factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **self.factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states
"""

GAB_ERROR = """Please provide the full gab code, and please do not modify other parts of the code. Specifically, please preserve # gab.py at the beginning of gab.py."""
