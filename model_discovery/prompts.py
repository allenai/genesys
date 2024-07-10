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
GAB_ERROR = """Please provide the full gab code, and please do not modify other parts of the code. Specifically, please preserve # gab.py at the beginning of gab.py."""
