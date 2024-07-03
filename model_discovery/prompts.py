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
