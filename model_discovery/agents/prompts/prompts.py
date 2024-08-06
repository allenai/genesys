from dataclasses import dataclass, field, asdict


# 1. You can use layer_idx to create arbitrary model structure with different types of blocks, examples:
#     * create 1 type of block for all layers, you can ignore layer_idx
#     * create 2 different types of blocks for layers with layer_idx%2=0,1
#     * create 3 different types of blocks for layers with layer_idx%3=0,1,2
#     * create 3 different types of blocks A,B,C, has AABC structure, then you can let layer_idx%4=0,1 for A, 2 for B, 3 for C
# 1. Use different types of blocks is not required also there is no preference for heterogeneous or homogeneous blocks, you can choose to create only one type of block or multiple types of blocks.

DESIGNER_PROMPT="""Design a novel autoregressive model block by completing the blanks marked in the Python code file gab.py below, which includes the initialization where you can define your custom arguments, the forward function where you can define convenient functions in the GAB class such as caches, the configuration with the hyperparameters that correspond to the arguments you defined:

{gab_py}

The GAB is inherited from this GABBase class, you should always import it by "from model_discovery.model.utils.modules import GABBase", you should never remove this statement from gab.py and you should never define another GABBase class in gab.py:

{gab_base}

This code will be used to construct a gam model in gam.py:

{gam_py}

This is the configuration and references for the target model:

{config}

Here are some hints:      
1. You need to consider the GAM model structure and the default operations like the normalization when designing the GAB block. Always remember that GAB is a part of the GAM model, it should not be designed as a whole model. 
2. You need to consider the magnitute of the model based on the reference model size, d_model and n_blocks. The n_blocks can be automatically adjusted so do not need to worry too much.
3. The model should be able to be parallel trained, which means you should not introduce recurrent operators like RNN or LSTM. The model should always be causal and differentiable.
4. The design should be innovative, you are not allowed to copy a previous design such as transformer block, you need to design your own block.
5. All dimensions of your model should always be a function of d_model (e.g., 2.5 times of d_model), you should never ever manually set a dimension of a layer to a fixed number in your config.
6. The GABBase provides a block_loc to help you locate the current block within the network which allows you to implement topology related operations.
7. The forward method allows you to create intermediate variables in intermediate_vars if you need.
8. You are not allowed to write any global code, e.g. example usage, macro definitions, etc. in gab.py besides gab_config.

{instruct}

Now, use the information provided above to complete the code. You should strictly follow the instructions in gab.py and do not remove anything suggested by the instructions. 

Your response should include the full gab.py file with the completed code. 
Specifically, when providing the full gab.py file, please preserve # gab.py at the beginning of the file.
You should derive your design step by step with detailed analysis and explanation before writing your code. 
"""

REVIEWER_PROMPT="""This is the proposal of the design of the general autoregressive block (gab) for you to review:

{proposal}

The GAB is inhereted from this GABBase class:

{gab_base}

The definition of a gam model in gam.py:

{gam_py}

{instruct}

Now, carefully review the design and give the feedback in a step by step way. You must return as a json file with two keys: 'review' and 'rating'. 
The 'review' key should contain a detailed feedback of the design written in markdown, and the 'rating' key should contain the rating of the design from 1 to 5.
"""
GAB_ERROR = """Please provide the full gab code, and please do not modify other parts of the code. Specifically, please preserve # gab.py at the beginning of gab.py."""
