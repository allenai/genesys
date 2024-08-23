from dataclasses import dataclass, field, asdict
from ..flow.alang import AgentPrompt

from exec_utils.models.model import ModelOutput
import re
import json
from typing import Dict, Any



#######################################################
# Naive GAB Design Prompts
#######################################################


# 1. You can use layer_idx to create arbitrary model structure with different types of blocks, examples:
#     * create 1 type of block for all layers, you can ignore layer_idx
#     * create 2 different types of blocks for layers with layer_idx%2=0,1
#     * create 3 different types of blocks for layers with layer_idx%3=0,1,2
#     * create 3 different types of blocks A,B,C, has AABC structure, then you can let layer_idx%4=0,1 for A, 2 for B, 3 for C
# 1. Use different types of blocks is not required also there is no preference for heterogeneous or homogeneous blocks, you can choose to create only one type of block or multiple types of blocks.

DESIGNER_PROMPT="""
Design a novel autoregressive model block by completing the blanks marked in the
Python code file gab.py below, which includes the initialization where you can
define your custom arguments, the forward function where you can define
convenient functions in the GAB class such as caches, the configuration with the
hyperparameters that correspond to the arguments you defined:

```python {gab_py} ```

The GAB is inherited from this GABBase class, you should always import it by
"from model_discovery.model.utils.modules import GABBase", you should never
remove this statement from gab.py and you should never define another GABBase
class in gab.py:

```python {gab_base} ```

This code will be used to construct a gam model in gam.py:

```python {gam_py} ```

This is the configuration and references for the target model:

```python {config} ```

Here are some hints: 1. You need to consider the GAM model structure and the
default operations like the normalization when designing the GAB block. Always
remember that GAB is a part of the GAM model, it should not be designed as a
whole model. 2. You need to consider the magnitute of the model based on the
reference model size, d_model and n_blocks. The n_blocks can be automatically
adjusted so do not need to worry too much. 3. The model should be able to be
parallel trained, which means you should not introduce recurrent operators like
RNN or LSTM. The model should always be causal and differentiable. 4. The design
should be innovative, you are not allowed to copy a previous design such as
transformer block, you need to design your own block. 5. All dimensions of your
model should always be a function of d_model (e.g., 2.5 times of d_model), you
should never ever manually set a dimension of a layer to a fixed number in your
config. 6. The GABBase provides a block_loc to help you locate the current block
within the network which allows you to implement topology related operations. 7.
The forward method allows you to create intermediate variables in
intermediate_vars if you need. 8. Only import, import from, class definition,
function definition, and gab_config definition can be shown in the code. Do not
include any other operations in the code. They will be automatically removed.

{instruct}

Now, use the information provided above to complete the code. You should
strictly follow the instructions in gab.py and do not remove anything suggested
by the instructions. 

Your response should include the full gab.py file with the completed code.
Specifically, when providing the full gab.py file, please preserve # gab.py at
the beginning of the file. You can write multiple codes during your analysis
process, but only one with # gab.py at the beginning will be detected as gab.py,
if multiple gab.py are detected in your response, only the last one will be
applied. You should derive your design step by step with detailed analysis and
explanation before writing your code. 
"""

REVIEWER_PROMPT="""This is the proposal of the design of the general autoregressive block (gab) for you to review:

{proposal}

The GAB is inhereted from this GABBase class:

```python
{gab_base}
```

The definition of a gam model in gam.py:

```python
{gam_py}
```

{instruct}

Now, carefully review the design and give the feedback in a step by step way. You must return as a json file with two keys: 'review' and 'rating'. 
The 'review' key should contain a detailed feedback of the design written in markdown, and the 'rating' key should contain the rating of the design from 1 to 5.
"""

GAB_BASE='''
class GABBase(nn.Module):
    """ Base class for Generalized Autoregressive Block """
    def __init__(self,embed_dim: int, block_loc: tuple): 
        super().__init__()
        self.embed_dim = embed_dim
        self.block_loc = block_loc # location of a block within the network, (layer_idx, n_block)

    def _forward(self,X,**kwargs): 
        raise NotImplementedError
     
    # YOU ARE NOT ALLOW TO OVERRIDE THIS METHOD #
    def forward(self,X,**Z):
        """Forward pass of the model"""
        assert X.shape[-1] == self.embed_dim
        Y_=self._forward(X,**Z)
        if isinstance(Y_,tuple):
            Y, Z = Y_
        else:
            Z = {}
        assert Y.shape == X.shape
        return Y, Z
'''

GAB_ERROR = """Please provide the full gab code, and please do not modify other parts of the code. Specifically, please preserve # gab.py at the beginning of gab.py. You can write multiple codes during your analysis process, but only one with # gab.py at the beginning will be detected as gab.py, if multiple gab.py are detected in your response, only the last one will be applied. Please follow the instructions in the prompt and provide the full gab.py file with the completed code. """




#######################################################
# GU GAB Design Prompts
#######################################################





""" ============================= GU System Prompt ========================================== """

# Current GPT4o token num: around 2K, just 0.005 USD in 0806

#region GU System Prompt



GU_DESIGNER_SYSTEM_prompt = """
You are a professional AI researcher focusing on discovering the best
autoregressive language model block. You goal is to design a novel block
following the Generalized Autoregressive Block (GAB) structure defined in the
following base class:

```python {GAB_BASE} ```


The GAB will be used to construct a Generalized Autoregressive Model (GAM)
defined as follows:

```python {GAM_PY} ```

The produced language model will be pretrained with the corpus and then be
applied for downstream tasks. The new model is expected to have a low
perplexity, high accuracy, robustness, efficiency, and most importantly, good
scalability. You have two roles 1) to propose ideas, analyze the problems,
design the model and implement it and; 2) to write the reports that justify your
ideas. You do not need to immediately do everything at one response, following
the provided instructions, and finish those tasks step by step in the coming
multi-round dialog. 

Since the autoregressive model design is complicated, so we will break it down
into smaller parts. We represent a block as multiple nested units, the
Generalized Autoregressive Unit (GAU). Each GAU accepts a sequence of embeddings
X and a dictionary of intermediate variables Z as input, and outputs a sequence
of embeddings Y and a dictionary of new or updated intermediate variables Z_. Z_
is optional, when it is provided, it will be used to update Z for the next unit
by Z.update(Z_). A GAU is defined in the following base class:

```python {GAU_BASE} ```

You will design a GAU by completing the blanks marked in this template, which
includes the initialization where you can define your custom arguments with
optional default values, the forward function where you can define convenient
functions or classes in the GAB class such as caches, notice that you are only
allowed to have only one GAU which inherited from the GAUBase class in the file:
 
```python {GAU_TEMPLATE} ```

In a GAU, you can call other GAUs, as such, you can create a complicated GAB
block by nesting multiple GAUs. However, each GAU should be not too complex, if
you want to create complex block, you should break it down into smaller GAUs and
nest them. As such, you should design a GAB block in a top-down manner. 

Notice that, everytime you are only allowed to edit within one GAU. You can
leave placeholder definition and calls of the GAUs that you wish to implement
later in your GAU. The system will automatically create an initial GAU code for
the placeholders. Once a GAU is provided, it will be inserted into the entire
GAB composed based on the tree of GAUs under your design and tested for
correctness then reviewed for novelty and quality. You will need to ensure the
correctness of all the GAUs in the final GAB at the end.
"""

GU_DESIGNER_SYSTEM = AgentPrompt(GU_DESIGNER_SYSTEM_prompt)


# endregion


""" ============================= Design from scratch init ===================================== """
# Initialize the design from scratch 

#region GU Design from scratch init



GU_DESIGN_SCRATCH_prompt = """
Start by filling in the blanks of this root GAU:

```python {ROOT_UNIT_TEMPLATE} ```

You should strictly follow the instructions in the template and do not remove
the template code. Your response should include: 

1. The intuitions and analysis of the GAU you are designing. You should
   think of how the design can be novel, creative, and powerful. The analysis
   should be detailed and considerable. It should decide a direction of the
   design, the core ideas and the justifications. Remember that you goal is to
   discover the best and novel autoregressive language model block that can
   defeat the existing state of arts.
2. A rough plan of the children GAUs that may need to be designed in the
   future.
3. The pseudo code of the GAU you are designing that capture the high-level
   idea. 
4. The name of the GAU you are designing. For the root GAU, the name
   will the name of the whole block design, so you definitely think of a
   meaningful name or a creative name that conclude your design, such as
   Transformers, Mamba, CausalConv, SSM, S6, YOLO, etc. Never use a meaningless
   name like RootGAB, NewGAB, etc. When you are trying to provide the name, you
   should wrap the name in the this format ```unit_name {{unit_name}}``` in
   order to allow the system to be able to detect it. 
5. The full implementation of the GAU you designed, remember to replece the
   unit_name marks by the actual unit name. Notice that you can contain multiple
   python codes in your response, but only the last one with "#
   GAB_UNIT_IMPLEMENTATION" mark in the first line will be detected as the final
   implementation. If multiple GAUs are detected in your response, only the
   last one will be applied. When you are trying to give the full implementation
   of the GAU you designed, you should always keep this mark at the first
   line, otherwise, the system will not be able to recognize your code.
6. The config of the hyperparameters you introduced in the GAU. You should
   provide the config dict in the following format: ```config {{
        # ADD HYPERPARAMETERS HERE #
    }} ``` in your response.

Here are some guidelines for designing the GAU:

1. You need to think of a meaningful name of the GAU you are designing or
   refering as the placeholder. When you are defining a placeholder, you should
   have an idea of the function of this GAU. By defining placeholders, you
   are defining the outline of the design you want to implement.
2. When calling a GAU, you should pass both the X and Z to the GAU. You
   should pass Z in the kwargs manner, for example {{GAU}}(X, **Z).
3. When defining a GAU object in __init__, you should privide the type hint,
   for example, ```self.unit: GAU = {{unit_name}}(**kwargs) ```, remember to
   pass such a kwargs which allows more customized arguments you will define
   later in your actual implementation to be passed in. When you defines a
   GAU, it should be either a known GAU or a placeholder of a GAU
   you are going to design. It should never be something else such as nn.Module
   or a constant. The system will automatically detect the GAU placeholders
   and create a new empty GAU or fetch it from the base if it is already
   defined.
4. Be sure to design the block in a top-down manner, be patient and think
   long-run, do never think of designing everything at once. Learn to define
   placeholders that may carry out complicated operations and implement them
   later. Especially when you are working on the root GAU. 
5. Be sure to be innovative, do not copy the existing designs such as the
   vanilla Transformer block. Be creative and think of a new design that can
   defeat the existing state of the art models. Try your best to transcend the
   human experts!

Now, give your design, don't be hurry to try to design everything at once, be
patient and focus on the current GAU you are designing. You will be asked
later to finish the remaining parts of the GAB block. Remember to design it step
by step, firstly do an analysis which shows your design intention, make sure the
design is novel, then write down the pseudo code before implement it, then think
of the name of your design, write the full code, and provide the configs.
"""

GU_DESIGN_SCRATCH_prompt_json = """
Start by filling in the blanks of this root GAU:

```python {ROOT_UNIT_TEMPLATE} ```

You should strictly follow the instructions in the template and do not remove
the template code. Your response should include: 

1. The intuitions and analysis of the GAU you are designing. You should
   think of how the design can be novel, creative, and powerful. The analysis
   should be detailed and considerable. It should decide a direction of the
   design, the core ideas and the justifications. Remember that you goal is to
   discover the best and novel autoregressive language model block that can
   defeat the existing state of arts.
2. A rough plan of the children GAUs that may need to be designed in the
   future.
3. The pseudo code of the GAU you are designing that capture the high-level
   idea. 
4. The name of the GAU you are designing. For the root GAU, the name
   will the name of the whole block design, so you definitely think of a
   meaningful name or a creative name that conclude your design, such as
   Transformers, Mamba, CausalConv, SSM, S6, YOLO, etc. Never use a meaningless
   name like RootGAB, NewGAB, etc. 
5. The full implementation of the GAU you designed, remember to replece the
   unit_name marks by the actual unit name. Notice that you can contain multiple
   python codes in your response, but only the last one with "#
   GAB_UNIT_IMPLEMENTATION" mark in the first line will be detected as the final
   implementation. If multiple GAUs are detected in your response, only the
   last one will be applied. When you are trying to give the full implementation
   of the GAU you designed, you should always keep this mark at the first
   line, otherwise, the system will not be able to recognize your code.
6. The config of the hyperparameters you introduced in the GAU, it should be
   a python dict, the keys are the hyperparams you introduced and the values are
   the corresponding default values.

Here are some guidelines for designing the GAU:

1. You need to think of a meaningful name of the GAU you are designing or
   refering as the placeholder. When you are defining a placeholder, you should
   have an idea of the function of this GAU. By defining placeholders, you
   are defining the outline of the design you want to implement.
2. When calling a GAU, you should pass both the X and Z to the GAU. You
   should pass Z in the kwargs manner, for example {{GAU}}(X, **Z).
3. When defining a GAU object in __init__, you should privide the type hint,
   for example, ```self.unit: GAU = {{unit_name}}(**kwargs) ```, remember to
   pass such a kwargs which allows more customized arguments you will define
   later in your actual implementation to be passed in. When you defines a
   GAU, it should be either a known GAU or a placeholder of a GAU
   you are going to design. It should never be something else such as nn.Module
   or a constant. The system will automatically detect the GAU placeholders
   and create a new empty GAU or fetch it from the base if it is already
   defined.
4. Be sure to design the block in a top-down manner, be patient and think
   long-run, do never think of designing everything at once. Learn to define
   placeholders that may carry out complicated operations and implement them
   later. Especially when you are working on the root GAU. 
5. Be sure to be innovative, do not copy the existing designs such as the
   vanilla Transformer block. Be creative and think of a new design that can
   defeat the existing state of the art models. Try your best to transcend the
   human experts!

Now, give your design, don't be hurry to try to design everything at once, be
patient and focus on the current GAU you are designing. You will be asked
later to finish the remaining parts of the GAB block. Remember to design it step
by step, firstly do an analysis which shows your design intention, make sure the
design is novel, then write down the pseudo code before implement it, then think
of the name of your design, write the full code, and provide the configs.
"""

GU_DESIGN_SCRATCH_format = {
  "type": "json_schema",
  "json_schema": {
    "name": "gab_unit_design",
    "description": "Design a Generalized Autoregressive Block Unit (GAU) for an autoregressive model block",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "analysis": {
          "type": "string",
          "description": "Intuitions and analysis behind the design of the GAU"
        },
        "motivation": {
          "type": "string",
          "description": "Core ideas and justifications driving the design"
        },
        "reasoning": {
          "type": "string",
          "description": "Detailed reasoning and how the design can be novel and powerful"
        },
        "pseudo_code": {
          "type": "string",
          "description": "High-level pseudocode of the GAU being designed"
        },
        "future_plan": {
          "type": "string",
          "description": "Plan for future GAUs that may need to be implemented"
        },
        "unit_name": {
          "type": "string",
          "description": "The name of the designed GAU"
        },
        "implementation": {
          "type": "string",
          "description": "Full Python code implementation of the designed GAU"
        },
        "config": {
          "type": "string",
          "description": "Configuration dictionary containing hyperparameters used in the GAU"
        }
      },
      "required": [
        "analysis",
        "motivation",
        "reasoning",
        "pseudo_code",
        "future_plan",
        "unit_name",
        "implementation",
        "config"
      ],
      "additionalProperties": False
    }
  }
}

# https://platform.openai.com/docs/guides/structured-outputs/supported-schemas Recursion supported, but maybe not use, it cannot isolate errors

def GU_DESIGN_SCRATCH_parser(raw_output: ModelOutput) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = {}
      codes = re.findall(r"```python(.*?)```", raw_text, re.DOTALL)
      if codes:
         for code in codes:
               if code.strip().startswith("# gab.py"):
                  output["code"] = code

      output["text"] = raw_text
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.cost
      output["_details"]["running_cost"] = 0
      return output

DESIGN_FROM_SCRATCH = AgentPrompt(GU_DESIGN_SCRATCH_prompt,GU_DESIGN_SCRATCH_parser)#,format=GU_DESIGN_SCRATCH_format)

# endregion


""" ============================= Give analysis proposal ===================================== """


# region GU Give analysis first 

GU_DESIGN_PROPOSAL_prompt = """
The first step is to write down an overal proposal that contains your idea for
the design you want to have. The proposal decides a direction, phylosophy and
the plan of the design. Here are some references for you to consider that may
inspire you:

{SEEDS}

Your response should include but not restrict to: 

1. A title with the name of the design in the level 1 header format. You shuld
   have only one level 1 header in your response which is the name of the
   design.

2. Your motivation of the design. What problem you want to solve based on the
   insights or observations you have about the autoregressive models today, and
   any inspirations you may have from the references. 

3. The analysis of the problem.

4. The core idea and phylosophy behind of your design that may solve the problem
   you proposed. 

5. The plan of the design. You should include subsections of that describe the
   details of each part of the design with the justifications.

6. A conclution of the proposal. 

7. Optional, the references you used in your proposal, should be in the right format.

The proposal will be reviewed and you will be asked to modify it if it is not
passed. You can start to implement the design after the proposal is passed. 

The proposal should be as detailed as possible, DO NOT WORRY THE LENGTH BUT ALSO
DO NOT FILL IN BY REDUNDANT WORDS, USE PRECISE AND CONCRETE LANGUAGE, it will be
the guideline for the whole design process. Now, give your proposal.
"""

def GU_DESIGN_PROPOSAL_parser(raw_output: ModelOutput) -> Dict[Any,Any]:
   title=""
   raw_text = raw_output.text.strip()
   for line in raw_text.split("\n"):
      if line.startswith("# "):
         title = line[2:]
         break
   if title == "":
       title = raw_text.split("\n")[0]
   output = {}
   output["title"] = title
   output["text"] = raw_text
   output["_details"] = {}
   output["_details"]["cost"] = raw_output.cost
   output["_details"]["running_cost"] = 0
   return output


GU_DESIGN_PROPOSAL = AgentPrompt(GU_DESIGN_PROPOSAL_prompt,GU_DESIGN_PROPOSAL_parser)


# endregion







""" ============================= GU Proposal Reviewer System ===================================== """


# region GU Give analysis first 


GU_PROPOSAL_REVIEWER_SYSTEM_prompt = """
You are a an expert in autoregressive language model research, you are asked to
review the proposal of the design of the autoregressive language model blocks. 

Here is the instruction about how to review the proposal:

1. Check if the design is potentially accurate, robust, efficient, and scalable.

2. The designed block must be novel, you need to check whether it is simply
   applying an existing design such as a transformer block.

3. If there is any unclear part, missing part, mistakes, unjustified, unrigorous
   parts in the proposal, you should explicitly point out. 
   
Your evaluation should be fair and comprehensive. 
"""

GU_PROPOSAL_REVIEWER_SYSTEM = AgentPrompt(GU_PROPOSAL_REVIEWER_SYSTEM_prompt)

# endregion



""" ============================= GU Proposal Review ===================================== """


# region GU Give analysis first 


GU_PROPOSAL_REVIEW_prompt = """
Here is a proposal of a design of the autoregressive language model block for
you to review:

{PROPOSAL}

Please give your review and rating of the design in the proposal, and
suggestions for the clarification, correction, or additional information. Your
rating decides whether the proposal can pass or not, rating is a float number
between 0 and 5. 1 is bad, 2 is not good enough, 3 is really good, 4 is excellent, 5 is
an outstanding design you have never seen and highly recommended. 3 is the
boarder for pass.

Be very strict and fair. Do not pass a proposal easily, give a pass only when it is
good enough.
"""

GU_PROPOSAL_REVIEW_format = {
   "type": "json_schema",
   "json_schema": {
         "name": "review_response",
         "strict": True,
         "schema": {
            "type": "object",
            "properties": {
               "review": {
                     "type": "string",
               },
               "rating": {
                     "type": "number",
                     "description": "A float number between 0 and 5."
               },
               "suggestions": {
                     "type": "string",
                     "description": "The suggestions for clarification, correction, or additional information."
               },
            },
            "required": ["review", "rating","suggestions"],
            "additionalProperties": False
         }
   }
}


def GU_PROPOSAL_REVIEW_parser(raw_output: ModelOutput) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = json.loads(raw_text)  
      output["text"] = raw_text
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.cost
      output["_details"]["running_cost"] = 0
      return output

GU_PROPOSAL_REVIEW = AgentPrompt(GU_PROPOSAL_REVIEW_prompt,GU_PROPOSAL_REVIEW_parser,GU_PROPOSAL_REVIEW_format)   

# endregion




""" ============================= GU Proposal Refinement ===================================== """


# region GU Proposal Refinement


GU_PROPOSAL_REFINEMENT_prompt = """
Your proposal has been reviewed and rated by the expert, here is the feedback:

{REVIEW}

Rating: {RATING} out of 5 (passing score is 3)

Suggestions: {SUGGESTIONS}

Please refine your proposal based on the feedback. You should address the issues
and improve the design based on the suggestions. Keep the format instructions. 
"""


GU_PROPOSAL_REFINEMENT = AgentPrompt(GU_PROPOSAL_REFINEMENT_prompt,GU_DESIGN_PROPOSAL_parser)



# endregion

