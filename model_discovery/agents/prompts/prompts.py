from dataclasses import dataclass, field, asdict
from ..flow.alang import AgentPrompt

from exec_utils.models.model import ModelOutput
import re
import json
from typing import Dict, Any


'''
#######################################################
# Naive GAB Design Prompts
#######################################################
'''

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







'''
#######################################################
# GU GAB Design Proposal Prompts
#######################################################
'''




def GENERAL_JSON_parser(raw_output: ModelOutput) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = json.loads(raw_text)  
      output["text"] = raw_text
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.cost
      output["_details"]["running_cost"] = 0
      return output


""" ============================= GU Designer System Prompt ========================================== """

# Current GPT4o token num: around 2K, just 0.005 USD in 0806

#region GU System Prompt



GU_DESIGN_PROPOSER_SYSTEM_prompt = """
You are a professional AI researcher focusing on discovering the best
autoregressive language model block. Your goal is to design a novel block
following the Generalized Autoregressive Block (GAB) structure defined in the
following base class:

```python {GAB_BASE} ```

The GAB will be used to construct a Generalized Autoregressive Model (GAM)
defined as follows:

```python {GAM_PY} ```

The produced language model will be pretrained with the corpus and then be
applied for downstream tasks. The new model is expected to have a low
perplexity, high accuracy, robustness, efficiency, and most importantly, good
scalability. 

Your role is to write down an overal proposal that contains your idea for the
design you want to have. The proposal decides a direction, phylosophy and the
plan of the design. You will be provided with one or multiple references to
consider that may inspire you.

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

7. Optional, the references you used in your proposal, should be in the right
   format.

The proposal will be reviewed and you will be asked to modify it if it is not
passed. You can start to implement the design after the proposal is passed. 

The proposal should be as detailed as possible, DO NOT WORRY IF THE PROPOSAL IS
TOO LONG, BUT ALSO DO NOT FILL IN BY REDUNDANT WORDS, USE PRECISE AND CONCRETE
LANGUAGE, the proposal will be the guideline for the entire design process so it
should be clear and detailed. """

GU_DESIGN_PROPOSER_SYSTEM = AgentPrompt(GU_DESIGN_PROPOSER_SYSTEM_prompt)


# endregion


""" ============================= Give analysis proposal ===================================== """


# region GU Give analysis first 

GU_DESIGN_PROPOSAL_prompt = """
Here are some references for you to consider that may inspire you:

{SEEDS}

Check the references, then give your proposal follow the instructions.
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

3. Find the highlights of the design, think of whether those highlights can lead
   to a successful design described above. Then think of the potential problems,
   limitations, risk or weaknesses of the design. Write down those concerns in
   your review.

3. If there is any unclear part, missing part, mistakes, unjustified, unrigorous
   parts in the proposal, you should explicitly point them out in your
   suggestions. 

4. Notice that, empirical results are not expected in the proposal stage, so you
   should check the design based on the theoretical analysis and the plan of the
   design.
   
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
suggestions for the clarification, correction, or additional information.
Rememeber that the rating and review should be completely based on the *design*
not the writing, any writing suggestions or highlights should be contained in
suggestions. Your rating decides whether the proposal can pass or not, rating is
a float number between 0 and 5. 1 is bad, 2 is not good enough, 3 is really
good, 4 is excellent, 5 is an outstanding design you have never seen and highly
recommended. 3 is the boarder for pass.

Be very strict and fair. Do not pass a proposal easily, give a pass only when it
is good enough.
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

GU_PROPOSAL_REVIEW = AgentPrompt(GU_PROPOSAL_REVIEW_prompt,GENERAL_JSON_parser,GU_PROPOSAL_REVIEW_format)   

# endregion




""" ============================= GU Proposal Refinement ===================================== """


# region GU Proposal Refinement


GU_PROPOSAL_REFINEMENT_prompt = """
Your proposal has been reviewed and rated by the expert, here is the feedback:

{REVIEW}

Rating: {RATING} out of 5 ({PASS_OR_NOT})

Suggestions: {SUGGESTIONS}

Please refine your proposal based on the feedback. You should address the issues
and improve the design based on the suggestions. You need to firstly provide the
reflection of the feedback, then give the full proposal keeping the format
instructions, finally, a summary of the changes you made.
"""

GU_PROPOSAL_REFINEMENT_format = {
   "type": "json_schema",
   "json_schema": {
         "name": "proposal_refinement_response",
         "strict": True,
         "schema": {
            "type": "object",
            "properties": {
               "reflection": {
                     "type": "string",
                     "description": "The reflection based on the review, rating, and suggestions."
               },
               "proposal": {
                     "type": "string",
                     "description": "The fall proposal, keep the format instructions."
               },
               "changes": {
                     "type": "string",
                     "description": "The summary of the changes you made."
               },
            },
            "required": ["reflection", "proposal","changes"],
            "additionalProperties": False
         }
   }
}


def GU_PROPOSAL_REFINEMENT_parser(raw_output: ModelOutput) -> Dict[Any,Any]:
   title=""
   raw_text = raw_output.text.strip()
   output = json.loads(raw_output.text)
   for line in output["proposal"].split("\n"):
      if line.startswith("# "):
         title = line[2:]
         break
   if title == "":
       title = raw_text.split("\n")[0]
   output["title"] = title
   output["text"] = raw_text
   output["_details"] = {}
   output["_details"]["cost"] = raw_output.cost
   output["_details"]["running_cost"] = 0
   return output



GU_PROPOSAL_REFINEMENT = AgentPrompt(GU_PROPOSAL_REFINEMENT_prompt,GU_PROPOSAL_REFINEMENT_parser,GU_PROPOSAL_REFINEMENT_format)

# endregion



""" ============================= GU Proposal Rereview ===================================== """


# region GU Give analysis first 


GU_PROPOSAL_REREVIEW_prompt = """
The designer has modified the proposal based on your review, here is the refined
version for you to review:

{PROPOSAL}

The change log:

{CHANGES}

Read the refined proposal carefully, check the change log, think of whether
those changes can address the concerns you pointed out in the previous review.
Then think is there more concerns or potential problems in the refined proposal.
Then give your review, rating, and suggestions. Rememeber that the rating and
review should be completely based on the *design* not the writing, any writing
suggestions or highlights should be contained in suggestions. Do not boost the
rating simply for "awarding" the address of a concern.

Be very strict and fair. Do not pass a proposal easily, give a pass only when it
is good enough.
"""


GU_PROPOSAL_REREVIEW = AgentPrompt(GU_PROPOSAL_REREVIEW_prompt,GENERAL_JSON_parser,GU_PROPOSAL_REVIEW_format)

# endregion




'''
#######################################################
# GU Implementation Root Prompts
#######################################################
'''


""" ============================= GU Design Implementer System ===================================== """


# region GU Design Implementer System



DESIGN_IMPLEMENTATER_SYSTEM_prompt = """
You are a professional AI researcher focusing on discovering the best
autoregressive language model block. Your goal is to design a novel block
following the Generalized Autoregressive Block (GAB) structure defined in the
following base class:

```python {GAB_BASE} ```


The GAB will be used to construct a Generalized Autoregressive Model (GAM)
defined as follows:

```python {GAM_PY} ```

The produced language model will be pretrained with the corpus and then be
applied for downstream tasks. The new model is expected to have a low
perplexity, high accuracy, robustness, efficiency, and most importantly, good
scalability. 

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

You will be provided with a proposal and the corresponding review and rating of
the design of the GAU, you should follow the proposal to implement the design.
The proposal provides the general guidance of the design, you should continue
thinking how to further improve the design while following the proposal.

As the proposal is high-level, you will still need to think of the details of
the implementation of each block. As a result, when implementing one GAU, you
should follow the following steps and include them in your response:

1. The intuitions and analysis of the GAU you are designing. Start from and go
   beyond the proposal and review, you should think of how the design can be
   novel, creative, and powerful. The analysis should be detailed and
   thoughtful. It should decide a direction of the design, the core ideas and
   the justifications. Remember that you goal is to discover the best and novel
   autoregressive language model block that can defeat the existing state of
   arts.
2. A rough plan of the children GAUs that may need to be designed in the future.
3. The pseudo code of the GAU you are designing that capture the high-level
   idea. 
4. The full implementation of the GAU you designed, remember to replece the
   unit_name marks by the actual unit name. Notice that you can contain multiple
   python codes in your response, but only the last one with "# gau.py" mark in
   the first line will be detected as the final implementation. If multiple GAUs
   are detected in your response, only the last one will be applied. When you
   are trying to give the full implementation of the GAU you designed, you
   should always keep this mark at the first line, otherwise, the system will
   not be able to recognize your code.

Here are some guidelines for designing the GAU:

 - Do not change the class name GAU, the system will automatically rename it
   using the unit_name you provide, and it should be the only GAU class (which
   can be decided by whether it is inherited from GAUBase) defined in your
   implementation, do not define any other GAU classes in your implementation.
   No need to worry about the placeholders, the system will automatically create
   the empty GAU classes for them.
 - You must have the GAU class that inherited from GAUBase defined in your code,
   here is what will happen if it is not found: 1. If there is no GAUBase
   classes detected, the system will report an error. 2. If there is only one
   GAUBase class detected, the system will automatically rename it and regard it
   as the GAU class you designed. 3. If there are multiple GAUBase classes
   detected, the system will take the one with the name "GAU" or the unit_name
   you provided as the GAU class you designed, and remove others. If no such
   name is found, the system will report an error.
 - When calling a GAU, you should pass both the X and Z to the GAU. You should
   pass Z in the kwargs manner, for example {{unit_name}}(X, **Z).
 - When defining a GAU object in __init__, you should privide the type hint, for
   example, ```self.unit: GAUBase = {{unit_name}}(**kwargs) ```, remember to
   pass such a kwargs which allows more customized arguments you will define
   later in your actual implementation to be passed in. When you defines a GAU,
   it should be either a known GAU or a placeholder of a GAU you are going to
   design. It should never be something else such as nn.Module or a constant.
   The system will automatically detect the GAU placeholders and create a new
   empty GAU or fetch it from the base if it is already defined. Do not
   implement any placeholders, you will be asked to implement them later.
 - If multiple GAU classes are detected in your response, only the one named
   "GAU" will be preserved, others will be removed.
 - You need to think of a meaningful name of the GAU you are designing or
   refering as the placeholder. When you are defining a placeholder, you should
   have an idea of the function of this GAU. By defining placeholders, you are
   defining the outline of the design you want to implement.
 - Be sure to design the block in a top-down manner, be patient and think
   long-run, do never think of designing everything at once. Learn to define
   placeholders that may carry out complicated operations and implement them
   later. Especially when you are working on the root GAU. 
 - Be sure to be innovative, do not copy the existing designs such as the
   vanilla Transformer block. Be creative and think of a new design that can
   defeat the existing state of the art models. Try your best to transcend the
   human experts!
"""


DESIGN_IMPLEMENTATER_SYSTEM = AgentPrompt(DESIGN_IMPLEMENTATER_SYSTEM_prompt)





# endregion




""" ============================= GU Implementation Root ===================================== """


# region GU Implementation Root



GU_IMPLEMENTATION_ROOT_prompt = """
Here is the proposal of the design for you to implement:

{PROPOSAL}

Here is the review of the proposal for you to refer:

{REVIEW}

Rating: {RATING} out of 5 (Passing score is >3)

Now, start by implementing a root GAU based on the proposal follow the
instructions, templates, and the format requirements. The GAU will be reviewed
and checked. It will be accepted only when it pass bothe the review and check
process. Your analysis should be as detailed as possible, and the implementation
can introduce new ideas and details that are not covered in the proposal that
can improve the design.

Don't be hurry to try to implemente everything at once, be patient, slowly, and
focus on the current GAU you are designing. And always remember to design it in
a top-down way. You will be asked later to finish the remaining parts of the GAB
block. 
"""


GU_IMPLEMENTATION_format = {
   "type": "json_schema",
   "json_schema": {
         "name": "implement_response",
         "strict": True,
         "schema": {
            "type": "object",
            "properties": {
               "analysis": {
                     "type": "string",
                     "description": "Analysis, plans, and pseudocode of the GAU being designed"
               },
               "unit_name": {
                     "type": "string",
                     'description': "The name of the designed GAU"
               },
               "implementation": {
                     "type": "string",
                     "description": "the full python implementation of the designed GAU"
               },
            },
            "required": ["analysis", "unit_name","implementation"],
            "additionalProperties": False
         }
   }
}


GU_IMPLEMENTATION_ROOT = AgentPrompt(GU_IMPLEMENTATION_ROOT_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_format)

# endregion




""" ============================= GU Implementation Reviewer System ===================================== """


# region GU Give analysis first 


GU_IMPLEMENTATION_REVIEWER_SYSTEM_prompt = """
You are a an expert in autoregressive language model research, you are asked to
review the details and implementations of a novel autoregressive language model
block. 

As the language model design is complicated, so we break it down into smaller
units, called the Generalized Autoregressive Unit (GAU). Each GAU accepts a
sequence of embeddings X and a dictionary of intermediate variables Z as input,
and outputs a sequence of embeddings Y and a dictionary of new or updated
intermediate variables Z_. Z_ is optional, when it is provided, it will be used
to update Z for the next unit by Z.update(Z_). A GAU is defined in the following
base class: 

{GAU_BASE}


By nesting multiple GAUs, we can design arbitrary complex autoregressive
language model blocks. Every time, a model designer agent will design or refine
a GAU following an overall proposal. You are asked to review the design and its
implementation, and give review, rating and suggestions to the designer.

Here is the instruction about how to review the unit design:

1. Check if the design and implementation is potentially accurate, robust,
   efficient, and scalable.

2. The designed block must be novel, you need to check whether the designer is
   simply applying an existing design such as a transformer block. 

3. Find the highlights of the design, think of whether those highlights can lead
   to a successful design described above. Then think of the potential problems,
   limitations, risk or weaknesses of the design. Write down those concerns in
   your review.

3. If there is any unclear part, missing part, mistakes, unjustified, unrigorous
   parts in the design or implementation, you should explicitly point them out
   in your suggestions. 

4. Notice that, empirical results are not expected in the design stage, so you
   should check the design based on the theoretical analysis and the proposal of
   the design.

5. The implementation may have error, such as not causal, or not differentiable,
   such error is not your concern, unless it is a design error. You should focus
   on the design, not the implementation error. But you should check whether the
   idea can be implemented in a causal, differentiable and efficient way.
   
Your evaluation should be fair and comprehensive. 
"""

GU_IMPLEMENTATION_REVIEWER_SYSTEM = AgentPrompt(GU_IMPLEMENTATION_REVIEWER_SYSTEM_prompt)

# endregion



""" ============================= GU Implementation Root Review Prompt ===================================== """


# region GU Implementation Root Review 

GU_IMPLEMENTATION_ROOT_REVIEW_prompt = """
The designer has designed and implemented a root GAU named {UNIT_NAME} following
this proposal:

{PROPOSAL}

Notice that the designer is allowed to introduce new ideas and details that are
not covered in the proposal that can improve the design. As long as the design
does not change the core idea of the proposal, it is acceptable.

Here is the design idea of the root GAU:

{ANALYSIS}

Here is the full implementation of the root GAU:

{IMPLEMENTATION}

Here is the information from the checker, the checker checks the forward,
backward, causality, etc., of the language model constructed using the designed
GAU and possibly other GAUs which are designed and checked previously, the
execution trace is provideded for you to refer:

{CHECKER_REPORT}

Read the proposal first, then the design ideas carefully, check the
implementation and the checker information, then give your review, rating, and
suggestions. Rememeber that the rating and review should be completely based on
the *design* not the writing. Your rating decides whether the design can pass or
not, rating is a float number between 0 and 5. 1 is bad, 2 is not good enough, 3
is really good, 4 is excellent, 5 is an outstanding design you have never seen
and highly recommended. 3 is the boarder for pass. If there is a clear error in
the design or implementation, you should point it out in your review or
suggestions.

Be very strict and fair. Do not pass a design easily, give a pass only when it
is good enough.
"""

GU_IMPLEMENTATION_REVIEW_format = {
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
                     "description": "The suggestions for improving or correcting the design."
               },
            },
            "required": ["review", "rating","suggestions"],
            "additionalProperties": False
         }
   }
}


GU_IMPLEMENTATION_ROOT_REVIEW = AgentPrompt(GU_IMPLEMENTATION_ROOT_REVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion



""" ============================= GU Implementation Retry Prompt ===================================== """

# region GU Implementation Retry


FORMAT_CHECKER_REPORT = """
Your code {RESULT} the format checker:

Errors:

{ERRORS}

Warnings:

{WARNINGS}
"""

FUNCTION_CHECKER_REPORT = """
Your code {RESULT} the functionality checker:

{REPORT}
"""


GU_IMPLEMENTATION_RETRY_prompt = """
Your design has been checked by the format checker, functionality checker, and
reviewed by the expert. However, it is not passed, here are the feedbacks.

Here is the information from the format checker, it checks whether your code
follows the format requirements:

{FORMAT_CHECKER_REPORT}

Here is the information from the functionality checker, it checks the forward,
backward, causality, etc., of the language model constructed using the designed
GAU and possibly other GAUs which are designed and checked previously, the
execution trace is provideded for you to refer:

{FUNCTION_CHECKER_REPORT}


Here is the review of the expert:

{REVIEW}

Rating: {RATING} out of 5 ({PASS_OR_NOT})

Here are the suggestions from the expert:

{SUGGESTIONS}

Please refine your design and implementation based on the feedback. You should
address the issues and improve the design based on the suggestions. You need to
guarantee that the implementation should pass all checkers. You need to firstly
provide the reflection of the feedback including the checkers' if you didnt pass
and the reviewer's, then give the full design including the new analysis, plans,
pseudocode, and the implementations. Keeping the format instructions, finally, a
summary of the changes you made.
"""

GU_IMPLEMENTATION_RETRY_format = {
   "type": "json_schema",
   "json_schema": {
         "name": "implementation_refinement_response",
         "strict": True,
         "schema": {
            "type": "object",
            "properties": {
               "reflection": {
                     "type": "string",
                     "description": "The reflection based on the review, rating, and suggestions."
               },
               "analysis": {
                     "type": "string",
                     "description": "Analysis, plans, and pseudocode of the GAU being designed"
               },
               "implementation": {
                     "type": "string",
                     "description": "the full python implementation of the designed GAU"
               },
               "changes": {
                     "type": "string",
                     "description": "The summary of the changes you made."
               },
            },
            "required": ["reflection", "analysis","implementation","changes"],
            "additionalProperties": False
         }
   }
}

GU_IMPLEMENTATION_RETRY= AgentPrompt(GU_IMPLEMENTATION_RETRY_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_RETRY_format)


# endregion






""" ============================= GU Implementation Rereview Prompt ===================================== """


# region GU Implementation Review 

GU_IMPLEMENTATION_REREVIEW_prompt = """
The designer has refined its design and implemention of the GAU {UNIT_NAME}
following the same proposal based on the feedbacks from you and the checking
results.

Here is the updated design idea of the GAU:

{ANALYSIS}

Here is the updated full implementation of the GAU:

{IMPLEMENTATION}

Here is a summary of the changes made:

{CHANGES}

Here is the information from the checker on this refined design, the checker
checks the forward, backward, causality, etc., of the language model constructed
using the designed GAU and possibly other GAUs which are designed and checked
previously, the execution trace is provideded for you to refer:

{CHECKER_REPORT}

Read the proposal first, then the design ideas carefully, check the
implementation, and the checker information, think of whether those changes can
address the concerns you pointed out in the previous review, then give your
review, rating, and suggestions. 

Rememeber that the rating and review should be completely based on the *design*
not the writing. Your rating decides whether the design can pass or not, rating
is a float number between 0 and 5. 1 is bad, 2 is not good enough, 3 is really
good, 4 is excellent, 5 is an outstanding design you have never seen and highly
recommended. 3 is the boarder for pass. If there is a clear error in the design
or implementation, you should point it out in your review or suggestions. Do not
boost the rating simply for "awarding" the address of a concern.

Be very strict and fair. Do not pass a design easily, give a pass only when it
is good enough.
"""



GU_IMPLEMENTATION_REREVIEW = AgentPrompt(GU_IMPLEMENTATION_REREVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion




'''
#######################################################
# GU Implementation nodes Prompts
#######################################################
'''

'''
Recursively implement the units or optionally refine a unit until no unimplemented unit is left and the designer/reviewer is satisfied.
The inputs are:
1. View of the current design, including a tree of the implemented and unimplemented units
2. The code (full code, the input tokens are cheap)
3. The proposal

Then the agent need to select a unit to work on from the tree
'''





""" ============================= GU Implementation Unit Selection ===================================== """


# region GU Implementation Root

GU_IMPLEMENTATION_UNIT_SELECTION_prompt = """
Here is the proposal of the design for you to implement:

{PROPOSAL}

Here is the review of the proposal for you to refer:

{REVIEW}

Rating: {RATING} out of 5 (Passing score is >3)

Here is a view of the progress of the implementation, its a tree of the GAUs
that composed the GAB block, the unimplemented GAUs are marked with
(unimplemented), it also shows an overview of the execution paths of the
children GAUs inside each implemented units, along with the rating of the unit:

{VIEW}

This is the exported code of the current design:

{GAB_CODE}

Now, select one GAU from the tree to work on, you can either select an
unimplemented GAU to implement or an implemented GAU to refine. And give a
motivation of your selection along with an overall evaluation of the current
view and a rough plan of your implementation. You need to provide the class name
of the GAU you are going to work on in your selection.

You will need to implement all the unimplemented GAUs in the tree. When all the
nodes are implemented, you can choose to terminate the design process when you
feel the design is good enough and no more left to improve. 
"""


def gen_GU_IMPLEMENTATION_UNIT_SELECTION(SELECTIONS):
   GU_IMPLEMENTATION_UNIT_SELECTION_format = {
      "type": "json_schema",
      "json_schema": {
            "name": "implement_response",
            "strict": True,
            "schema": {
               "type": "object",
               "properties": {
                  "motivation": {
                        "type": "string",
                        "description": "Overall view, motivation and plans of the selection."
                  },
                  "selection": {
                        "type": "string",
                        'description': "The name of the GAU you are going to work on.",
                        'enum': SELECTIONS
                  },
                  'termination': {
                     'type': 'boolean',
                     'description': 'Whether to terminate the design process. It will be ignored if there are unimplemented units left.'
                  }
               },
               "required": ["motivation", "selection","termination"],
               "additionalProperties": False
            }
      }
   }
   return AgentPrompt(GU_IMPLEMENTATION_UNIT_SELECTION_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_UNIT_SELECTION_format)

# endregion




""" ============================= GU Implementation Unit ===================================== """


# region GU Implementation Root




def gen_GU_IMPLEMENTATION_UNIT(refine=False):

   if refine:
      GU_IMPLEMENTATION_UNIT_prompt = """
Here is the description of the GAU you are going to refine:

{DESCRIPTION}

Here is the full implementation of the GAU you are going to refine:

{IMPLEMENTATION}

This is the review of the GAU:

{REVIEW}

Rating: {RATING} out of 5 (Passing score is >3)

The suggestions from the reviewer:

{SUGGESTIONS}


Now you need to refine the GAU based on the feedback. You should address the
issues and improve the design based on the suggestions. You need to firstly
provide the reflection of the feedback, then give the full design including the
new analysis, plans, pseudocode, and the implementations. Keeping the format
instructions, finally, a summary of the changes you made.

Your design and implementation should be based on the proposal, following the
instructions, templates, and the format requirements. The GAU will be reviewed
and checked. It will be accepted only when it pass bothe the review and check
process. Your analysis should be as detailed as possible, and the implementation
can introduce new ideas and details that are not covered in the proposal that
can improve the design. 

Don't be hurry to try to implemente everything at once, be patient, slowly, and
focus on the current GAU you are designing. And always remember to design it in
a top-down way. You are encouraged to define more children GAU placeholders that
can be implemented later for more complicated operations. You will be asked
later to finish the remaining parts of the GAB block. 
   """
      GU_IMPLEMENTATION_UNIT_format = {
         "type": "json_schema",
         "json_schema": {
               "name": "implementation_refinement_response",
               "strict": True,
               "schema": {
                  "type": "object",
                  "properties": {
                     "reflection": {
                           "type": "string",
                           "description": "The reflection based on the review, rating, and suggestions."
                     },
                     "analysis": {
                           "type": "string",
                           "description": "Analysis, plans, and pseudocode of the GAU being designed"
                     },
                     "implementation": {
                           "type": "string",
                           "description": "the full python implementation of the designed GAU"
                     },
                     "changes": {
                           "type": "string",
                           "description": "The summary of the changes you made."
                     },
                  },
                  "required": ["reflection", "analysis","implementation","changes"],
                  "additionalProperties": False
               }
         }
      }
   else:
      GU_IMPLEMENTATION_UNIT_prompt = """
Now, please design and implement the GAU you selected. Your design and
implementation should be based on the proposal, following the instructions,
templates, and the format requirements. The GAU will be reviewed and checked. It
will be accepted only when it pass bothe the review and check process. Your
analysis should be as detailed as possible, and the implementation can introduce
new ideas and details that are not covered in the proposal that can improve the
design. 

Don't be hurry to try to implemente everything at once, be patient, slowly, and
focus on the current GAU you are designing. And always remember to design it in
a top-down way. You are encouraged to define more children GAU placeholders that
can be implemented later for more complicated operations. You will be asked
later to finish the remaining parts of the GAB block. 
   """
      GU_IMPLEMENTATION_UNIT_format = {
         "type": "json_schema",
         "json_schema": {
               "name": "implementation_refinement_response",
               "strict": True,
               "schema": {
                  "type": "object",
                  "properties": {
                     "analysis": {
                           "type": "string",
                           "description": "Analysis, plans, and pseudocode of the GAU being designed"
                     },
                     "implementation": {
                           "type": "string",
                           "description": "the full python implementation of the designed GAU"
                     },
                  },
                  "required": ["analysis","implementation"],
                  "additionalProperties": False
               }
         }
      }

   return AgentPrompt(GU_IMPLEMENTATION_UNIT_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_UNIT_format)

# endregion


""" ============================= GU Implementation Unit Review Prompt ===================================== """


# region GU Implementation Root Review 

GU_IMPLEMENTATION_UNIT_REVIEW_prompt = """
Here is the proposal of the designing to implement:

{PROPOSAL}

Here is a view of the progress of the implementation, its a tree of the GAUs
that composed the GAB block, the unimplemented GAUs are marked with
(unimplemented), it also shows an overview of the execution paths of the
children GAUs inside each implemented units, along with the rating of the unit:

{VIEW}

This is the exported code of the current design:

{GAB_CODE}

The designer has chosen to implement a GAU named {UNIT_NAME}.Notice that the
designer is allowed to introduce new ideas and details that are not covered in
the proposal that can improve the design. As long as the design does not change
the core idea of the proposal, it is acceptable.

Here is the design idea of the GAU:

{ANALYSIS}

Here is the full implementation of the GAU:

{IMPLEMENTATION}

Here is the information from the checker, the checker checks the forward,
backward, causality, etc., of the language model constructed using the designed
GAU and possibly other GAUs which are designed and checked previously, the
execution trace is provideded for you to refer:

{CHECKER_REPORT}

Read the proposal first, then the design ideas carefully, check the
implementation and the checker information, then give your review, rating, and
suggestions. Rememeber that the rating and review should be completely based on
the *design* not the writing. Your rating decides whether the design can pass or
not, rating is a float number between 0 and 5. 1 is bad, 2 is not good enough, 3
is really good, 4 is excellent, 5 is an outstanding design you have never seen
and highly recommended. 3 is the boarder for pass. If there is a clear error in
the design or implementation, you should point it out in your review or
suggestions.

Be very strict and fair. Do not pass a design easily, give a pass only when it
is good enough.
"""

GU_IMPLEMENTATION_UNIT_REVIEW = AgentPrompt(GU_IMPLEMENTATION_UNIT_REVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion


""" ============================= GU Implementation Unit Refine Review Prompt ===================================== """


# region GU Implementation Root Review 

GU_IMPLEMENTATION_UNIT_REFINE_REVIEW_prompt = """
Here is the proposal of the designing to implement:

{PROPOSAL}

Here is a view of the progress of the implementation, its a tree of the GAUs
that composed the GAB block, the unimplemented GAUs are marked with
(unimplemented), it also shows an overview of the execution paths of the
children GAUs inside each implemented units, along with the rating of the unit:

{VIEW}

This is the exported code of the current design:

{GAB_CODE}

The designer has chosen to refine a GAU named {UNIT_NAME}.Notice that the
designer is allowed to introduce new ideas and details that are not covered in
the proposal that can improve the design. As long as the design does not change
the core idea of the proposal, it is acceptable.

This is the description of this GAU:

{DESCRIPTION}

Here is the previous review of the GAU:

{REVIEW}

Rating: {RATING} out of 5 (Passing score is >3)

The suggestions from the previous reviewer:

{SUGGESTIONS}

Here is the design idea of the GAU:

{ANALYSIS}

Here is the full implementation of the GAU:

{IMPLEMENTATION}

Here is a summary of the changes made:

{CHANGES}

Here is the information from the checker, the checker checks the forward,
backward, causality, etc., of the language model constructed using the designed
GAU and possibly other GAUs which are designed and checked previously, the
execution trace is provideded for you to refer:

{CHECKER_REPORT}

Read the proposal first, then the design ideas carefully, check the
implementation and the checker information, then give your review, rating, and
suggestions. Rememeber that the rating and review should be completely based on
the *design* not the writing. Your rating decides whether the design can pass or
not, rating is a float number between 0 and 5. 1 is bad, 2 is not good enough, 3
is really good, 4 is excellent, 5 is an outstanding design you have never seen
and highly recommended. 3 is the boarder for pass. If there is a clear error in
the design or implementation, you should point it out in your review or
suggestions.

Be very strict and fair. Do not pass a design easily, give a pass only when it
is good enough.
"""

GU_IMPLEMENTATION_UNIT_REFINE_REVIEW = AgentPrompt(GU_IMPLEMENTATION_UNIT_REFINE_REVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion

