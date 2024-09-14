
from ..flow.alang import AgentPrompt
from model_discovery.model.utils.modules import UnitSpec

import re
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from ..agent_utils import ModelOutputPlus
from enum import Enum



def generate_enum_from_list(enum_name: str, values: list):
    enum_dict = {value: value for value in values}
    return Enum(enum_name, enum_dict)


GAB_ERROR = """Please provide the full gab code, and please do not modify other parts of the code. Specifically, please preserve # gab.py at the beginning of gab.py. You can write multiple codes during your analysis process, but only one with # gab.py at the beginning will be detected as gab.py, if multiple gab.py are detected in your response, only the last one will be applied. Please follow the instructions in the prompt and provide the full gab.py file with the completed code. """


class GU_IMPLEMENTATION_ROOT_format(BaseModel): 
   analysis: str = Field(..., description="The analysis of how to best design and implement the GAU.")
   spec: UnitSpec = Field(..., description="The specification of the GAU.")
   implementation: str = Field(..., description="The full python implementation of the GAU following the GAU format instructions.")


class GU_IMPLEMENTATION_format(BaseModel): 
   analysis: str = Field(..., description="The analysis of how to best design and implement the GAU.")
   implementation: str = Field(..., description="The full python implementation of the GAU following the GAU format instructions.")

class GU_IMPLEMENTATION_ROOT_RETRY_format(BaseModel): # for retry, can update spec 
   reflection: str = Field(..., description="The reflection of the feedback from the reviewer and checkers.")
   analysis: str = Field(..., description="The analysis of how to best design and implement the GAU.")
   spec: UnitSpec = Field(..., description="The specification of the GAU.")
   implementation: str = Field(..., description="The full python implementation of the GAU following the GAU format instructions.")
   changes: str = Field(..., description="The exact changes you have made in the code. It must include detailed code diffs and necessary context with explanation.")


class DebuggingStep(BaseModel):
    diagnosis: str = Field(..., description="The diagnosis of the cause of the error.")
    suggested_action: str = Field(..., description="The suggested action to fix the error. Must be a concrete code about what to modify or print statements that help locate the error.")

class GU_IMPLEMENTATION_RETRY_DEBUG_format(BaseModel): # for retry
   reflection: str = Field(..., description="The reflection of the feedback from the reviewer.")
   # debugging_steps: list[DebuggingStep] = Field(..., description="The debugging steps to fix the error.")
   debugging_steps: str = Field(..., description="The debugging steps to fix the error.")
   analysis: str = Field(..., description="The analysis of how to best design and implement the GAU.")
   implementation: str = Field(..., description="The full python implementation of the GAU following the GAU format instructions.")
   changes: str = Field(..., description="The exact changes you have made in the code. It must include detailed code diffs and necessary context with explanation.")

class GU_IMPLEMENTATION_RETRY_format(BaseModel): # for retry
   reflection: str = Field(..., description="The reflection of the feedback from the reviewer.")
   analysis: str = Field(..., description="The analysis of how to best design and implement the GAU.")
   implementation: str = Field(..., description="The full python implementation of the GAU following the GAU format instructions.")
   changes: str = Field(..., description="The exact changes you have made in the code. It must include detailed code diffs and necessary context with explanation.")

class GU_IMPLEMENTATION_REFINE_format(BaseModel): # for refine, allow to update description, implementation, children, and changes
   newname: str = Field(..., description="The name of this GAU variant, remember that do not apply this new name to your implementation, keep the original name in the implementation.")
   reflection: str = Field(..., description="The reflection of the feedback from the reviewer.")
   analysis: str = Field(..., description="The analysis of how to best design and implement the GAU.")
   implementation: str = Field(..., description="The full python implementation of the GAU following the GAU format instructions.")
   changes: str = Field(..., description="The exact changes you have made in the code. It must include detailed code diffs and necessary context with explanation.")



def GENERAL_JSON_parser(raw_output: ModelOutputPlus) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = json.loads(raw_text)  
      output["text"] = raw_text
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.usage
      output["_details"]["running_cost"] = raw_output.usage['cost']
      return output



'''
###################################################################################################
##                                                                                               ##
## Naive GAB Design Prompts                                                                      ##
##                                                                                               ##
###################################################################################################
'''

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






'''
###################################################################################################
##                                                                                               ##
## GU Design from scratch prompts                                                                ##
##                                                                                               ##
###################################################################################################
'''


# All start with GU_



def build_GU_QUERY(seed,references=None,instruct=None):
   query = "Here is the seed design for you to consider:"
   for i in seed:
      query += f"Seed {idx}:\n{i.to_prompt()}\n\n"
   if references:
      query += '''
Here are some references that may inspire you, the references are from the following libraries:

- DesignArtifact: stores the previous designs that are fully implemented and passed the test
- ReferenceCore: stores the most typical and representative autoregressive language model designs with implementation in GAB format
- ReferenceCoreWithTree: stores the most typical and representative autoregressive language model designs with GAU-based implementation
- References: stores the state-of-the-art language model papers 
- ReferenceWithCode: stores the state-of-the-art language model papers with illustrative code
- Reference1hop: stores the high-impact citations of the state-of-the-art language model papers from References and ReferenceWithCode

The references are listed as follows:
'''
      for idx,reference in enumerate(references):
         query += f"Reference {idx} from library {reference.type}:\n{reference.to_prompt()}\n\n"
   if instruct:
      query += f"Here are some additional instructions that may help you:\n{instruct}\n\n"
   return query


'''
#######################################################
# GU GAB Design Proposal Prompts
#######################################################
'''



""" ============================= GU Designer Proposal System Prompt ========================================== """

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

Your response should include: 

1. A model name, it should be a camel case legal variable name for defining the
   model class in pytorch.

2. The proposal, it should include but not restrict to the following parts: a. A
   title with the name of the design in the level 1 header format. You shuld
      have only one level 1 header in your response which is the name of the
      design.

   b. Your motivation of the design. What problem you want to solve based on the
      insights or observations you have about the autoregressive models today,
      and any inspirations you may have from the references. 

   c. The analysis of the problem.

   d. The core idea and phylosophy behind of your design that may solve the
      problem you proposed. 

   e. The plan of the design. You should include subsections of that describe
      the details of each part of the design with the justifications.

   f. A conclution of the proposal. 

   g. Optional, the references you used in your proposal, should be in the right
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

class GU_DESIGN_PROPOSAL_format(BaseModel):
   modelname: str = Field(..., description="The name of the model. It should be a camel case legal variable name for defining the model class in pytorch.")
   proposal: str = Field(..., description="The full proposal, keep the format instructions.")

GU_DESIGN_PROPOSAL = AgentPrompt(GU_DESIGN_PROPOSAL_prompt,GENERAL_JSON_parser,GU_DESIGN_PROPOSAL_format)


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

class GU_PROPOSAL_REVIEW_format(BaseModel):
   review: str = Field(..., description="The review of the proposal, keep the format instructions.")
   rating: float = Field(..., description="A float number between 0 and 5.")
   suggestions: str = Field(..., description="The suggestions for clarification, correction, or additional information.")

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

class GU_PROPOSAL_REFINEMENT_format(BaseModel):
   reflection: str = Field(..., description="The reflection based on the review, rating, and suggestions.")
   proposal: str = Field(..., description="The fall proposal, keep the format instructions.")
   changes: str = Field(..., description="The summary of the changes you made.")

def GU_PROPOSAL_REFINEMENT_parser(raw_output: ModelOutputPlus) -> Dict[Any,Any]:
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
   output["_details"]["cost"] = raw_output.usage
   output["_details"]["running_cost"] = raw_output.usage['cost']
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

You must write the unit tests for your implementation and the GAU you designed,
the unit tests are decorated by @gau_test, the system will automatically detect
the unit tests based on this decorator and run them in a checker to help you
diagnose and debug the GAU you designed. You should write assertions and make
prints in the unit tests. The unit tests accept only device and dtype as
arguments, you should use them to initialize your GAU and mock inputs. You can
rename the function but never change the arguments and the decorator. The unit
tests should also not return anything, they will be ignored by the system. You
can also write your debugging functions as gau unit tests, they will be executed
by the system and the outputs will be captured and shown in the report.

Write bug free and easy debugging code is important. In order to better locate
the bug, you should also always write assertions and optionally print in your
code that help you to diagnose the system. They can also be the signals for you
to debug other units in future. The outputs will be captured by the
functionality checker and shown in the report.

The GAUs you designed will consist a tree where the children of a node are the
GAUs defined in the __init__ method of the node. And you will be asked to start
your design by designing a root node. The tree will be used to compose the GAB
block using this GABComposer class:

```python {GAB_COMPOSER} ```

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

1. An analysis, it should contain the detailed process of how you design the
   GAU. Start from and go beyond the proposal and review, you should think of
   how the design can be novel, creative, and powerful. The analysis should be
   detailed and thoughtful. It should decide a direction of the design, the core
   ideas and the justifications. Remember that you goal is to discover the best
   and novel autoregressive language model block that can defeat the existing
   state of arts. A rough plan of the children GAUs that may need to be designed
   in the future. You should have a basic idea of its function, input and
   output, the details can be completed later. However the IO should be decided,
   as changing IO is complicated since it can have influence to the parent
   model. The pseudo code of the GAU you are designing that capture the
   high-level idea. 
2. Based on the type of unit you are designing, you should provide the
   following: 
    - If you are designing the non-root units or refining the root unit: You
      should provide a document, which is a description and user guide of the
      GAU you are designing including the desciption of its function, behavior,
      how it works, the idea and key features, the constraints, and the details
      of the inputs and outputs, how to use and illustrative example usages, and
      other information you think is important for the user to understand and
      use the GAU. The document should be clear and detailed, it will be used
      for the users to understand the GAU you designed without looking at the
      implementation. It should allows the user to safely use this GAU and know
      its advantages and limitations when considering to use it.
    - If you are designing a new root unit: You should provide a full
      specification which contains not only the document but also the unit name,
      variable names of expected inputs, and outputs. For optional inputs, you
      can mark them with * at the beginning of the variable name. You only need
      to provide the variables that are really processed and outputted. For
      example, sequence X and Y is required as input and output respectively,
      however the unit may not process it, so it does not need to be listed even
      it will be provided and returned. Notice that root unit may input and
      output intermediate variables, and may vary if you introduced topology
      related designs. Simular to GAU, the intermediate variables Z will be
      accumulatively updated from upper stream blocks to the lower stream
      blocks. 
3. The list of children you need to define. To declare a child GAU, you should
   provide the unit name, variable names of the expected inputs and outputs
   (i.e. the interfaces of the child), the requirements of the child including the
   expected function, behavior and the description of the interfaces. The
   requirements should be clear and detailed, it should be a guideline for the
   implementation of the child GAU. You can reuse an existing GAU or a declared
   GAU as your child, you can do this by declaring the children with the unit
   name of the GAU you want to reuse, and the system will automatically reuse
   the GAU, the other parts of your declaration will be ignored.
4. The full implementation of the GAU you designed, remember to replece the
   unitname marks by the actual unit name. Notice that you can contain multiple
   python codes in your response, but only the last one with "# gau.py" mark in
   the first line will be detected as the final implementation. If multiple GAUs
   are detected in your response, only the last one will be applied. When you
   are trying to give the full implementation of the GAU you designed, you
   should always keep this mark at the first line, otherwise, the system will
   not be able to recognize your code. Notice that you should never write a
   __main__ part in your code which will be automatically removed by the system. 

Here are some guidelines for designing the GAU:

 - Remember to change the class name of the GAU in the template by the unit
   name, the system will automatically rename it using the unitname you provide,
   and it should be the only GAU class (which can be decided by whether it is
   inherited from GAUBase) defined in your implementation, do not define any
   other GAU classes in your implementation. No need to worry about the
   placeholders, the system will automatically create the empty GAU classes for
   them. If you do not rename the class name to the unitname, it may cause
   problem when you refer it in your unit tests.
 - You must have the GAU class that inherited from GAUBase defined in your code,
   here is what will happen if it is not found: 1. If there is no GAUBase
   classes detected, the system will report an error. 2. If there is only one
   GAUBase class detected, the system will automatically rename it and regard it
   as the GAU class you designed. 3. If there are multiple GAUBase classes
   detected, the system will take the one with the name "GAU" or the unitname
   you provided as the GAU class you designed, and remove others. If no such
   name is found, the system will report an error.
 - When calling a GAU, you should pass both the X and Z to the GAU. You should
   pass Z in the kwargs manner, for example {{unitname}}(X, **Z). If you need to
   pass aditional arguments, you should add it to Z then pass it to the GAU. So
   that the code won't cause error even the the GAU has not been implemented
   yet. The output of the GAU is also always two values, Y and dict Z_, X and Y
   are reserved for a path to convey the sequence of embeddings, if you imagine
   the model is a pipeline of operations, X and Y is the main product, while Z
   is the intermediate products. X and Y should always be a sequence of
   embeddings, any other values should be put in Z. You should always have an
   embedding Y as the output of the GAU, even if you just simple pass the input
   X to the output Y. This path is important to guarantee the composed GAB is
   always executable. So if you are designing a block that do not actually
   operate the sequence, you should use Z to pass the expected inputs from the
   upper stream block and pass the expected outputs to Z for the downstream
   blocks.
 - You should never ever using X and Y to input or output anything besides the
   sequence, if you want to do that, pass them to Z!
 - Whenever you define a GAU instance, you should always follow this way:
   ```self.{{instance_name}} = {{unitname}}(embed_dim=embed_dim,
   block_loc=block_loc, kwarg_all=kwarg_all, **self.factory_kwargs, **kwarg_all)
   ```, and ```self.factory_kwargs = {{"device": device, "dtype": dtype}}```. 
 - embed_dim is the dimension of input, it decides the network; block_loc is a
   tuple of (block_idx, n_block) that helps you locate the GAB block to be
   composed within the network, e.g. (0, 6) for the first block in a network
   with 6 blocks in total; kwarg_all is a dictionary of all hyperparameters
   across all units, device and dtype should be passed to any nn layers,
   parameters, tensors, etc. you defined in __init__, _forward, or any other
   places, you can pass it through self.factory_kwargs every where in your GAU.
 - You can use the block_loc to implment the topology related operations,
   example usages: 
    - Initializing the internal states, memories, caches, embeddings, etc. in
      the first block (GAB composed by the unit tree) of the network, and
      updating them in the later blocks. 
    - Using variant operations, or even model architecture in different blocks,
      such as using one kind in odd blocks and another kind in even blocks.
      Using a different structure in the last block of the network. Making a
      hybrid structure that using different structure when the block_idx mod by
      4 is 0, 1, 2, 3. Making a pyramid structure with  the middle block is the
      bottleneck... 
 - When you defines a GAU, it should be either a known GAU or a placeholder of a
   GAU you are going to design. It should never be something else such as
   nn.Module or a constant. You should provide the *full list* of the children
   GAU classes you will use inside the __init__ method in your response to
   ensure the system can detect them correctly when composing the GAB code.
   Notice that you should record the class names of the GAU not the name of the
   instances of the GAU.
 - You are not allowed to define an instance of a GAU inside nn.Sequential.
   nn.ModuleList and nn.ModuleDict are allowed.
 - The system will automatically detect the GAU placeholders and create a new
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
 - Always keep in mind that every time you are only allowed to edit within one
   GAU, and you do not have the access to the other units. It means that if
   there is a certain design dependent on the other units, you should either
   write a placeholder for it, such as if an input is not provided, you can
   ignore it, but never raise an error. You should keep every unit *runnable*
   without dependency of other units. Notice that runnable does not mean the
   function is correct which is relying on the inputs, just make sure it wont
   trigger any error. You can modify this unit later.
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
and checked. It will be accepted only when it pass both the review and check
process. Your analysis should be as detailed as possible, and the implementation
can introduce new ideas and details that are not covered in the proposal that
can improve the design. 

Don't be hurry to try to implemente everything at once, be patient, slowly, and
focus on the current GAU you are designing. And always remember to design it in
a top-down way. You will be asked later to finish the remaining parts of the GAB
block. 
"""


GU_IMPLEMENTATION_ROOT = AgentPrompt(GU_IMPLEMENTATION_ROOT_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_ROOT_format)

# endregion




""" ============================= GU Implementation Reviewer System ===================================== """


# region GU Proposal Reviewer 


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

```python {GAU_BASE} ```

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

This is the specification of the root GAU:

{SPECIFICATION}

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
is good enough. Notice that the designer is expected work on one unit at a time
and use a top-down manner to design the model, thus a unit may include
unimplemented placeholders. This is allowed, the review should be focused on the
unit being designed itself, not the implementation of the placeholders as they
are expected to be implemented later. The empirical results are not expected in
the design stage, so you should check the design based on the theoretical
analysis. 
"""

class GU_IMPLEMENTATION_REVIEW_format(BaseModel):
   review: str = Field(..., description="The review of the proposal, keep the format instructions.")
   rating: float = Field(..., description="A float number between 0 and 5.")
   suggestions: str = Field(..., description="The suggestions for clarification, correction, or additional information.")

GU_IMPLEMENTATION_ROOT_REVIEW = AgentPrompt(GU_IMPLEMENTATION_ROOT_REVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion



""" ============================= GU Implementation Retry Prompt ===================================== """

# region GU Implementation Retry


def gen_FORMAT_CHECKER_REPORT(RESULT,ERRORS,WARNINGS):
   FORMAT_CHECKER_REPORT = f"Your code {RESULT} the format checker."
   if ERRORS:
      FORMAT_CHECKER_REPORT += "Errors:\n"
      for error in ERRORS:
         FORMAT_CHECKER_REPORT += f"\t{error}\n"
   if WARNINGS:
      FORMAT_CHECKER_REPORT += "Warnings:\n"
      for warning in WARNINGS:
         FORMAT_CHECKER_REPORT += f"\t{warning}\n"
   return FORMAT_CHECKER_REPORT


FUNCTION_CHECKER_REPORT_PASS = """
Your code passed the functionality checker:

{REPORT}
"""


FUNCTION_CHECKER_REPORT_FAIL= """
Your code failed the functionality checker:

{REPORT}

Here is the composed LM block code `gab.py` based on the GAUs for you to refer:

{GAB_CODE_WITH_LINE_NUM}
"""


GU_IMPLEMENTATION_RETRY_prompt = """
Your design has undergone checks by the format checker, functionality checker, and has been reviewed by an expert. Unfortunately, it did not pass. Below is the feedback:

- **Format Checker**: This report assesses whether your code adheres to the required format guidelines.
  
  **Format Checker Report**:
  {FORMAT_CHECKER_REPORT}

- **Functionality Checker**: The functionality checker evaluates two critical aspects:
  1. **Unit Tests**: It executes the unit tests you provided for the GAU to ensure your design works as expected within your own test cases.
  2. **Whole Model Integration**: Beyond testing the GAU in isolation, the functionality checker integrates your GAU into the larger language model (LM). It compose the tree of GAUs as the LM block. It generates any necessary placeholder classes for unimplemented units and verifies the functionality of the entire LM, including forward pass, backward pass, and causality.

  **Functionality Checker Report**:
  {FUNCTION_CHECKER_REPORT}

- **Expert Review**: 
  **Review**: {REVIEW}
  **Rating**: {RATING} out of 5 ({PASS_OR_NOT})

- **Suggestions from the Expert**:
  {SUGGESTIONS}

### Next Steps:

Please refine your design and implementation based on the feedback provided. Your updated submission must pass all checks and unit tests. Follow the steps outlined below:

1. **Analyze the Reviewer's Feedback**:
   - Carefully examine any concerns raised by the reviewer. For each concern, provide a thorough analysis and outline how you plan to address it in your design and implementation.

2. **Debugging and Diagnosis**:
   - If bugs were identified in the unit tests, format checker, or functionality checker, systematically analyze each bug. For every step of the debugging process:
     - **Diagnose the root cause**: Investigate and identify the exact source of the issue.
     - **Propose a solution**: Suggest a detailed action, including a specific code change or the addition of print statements to further diagnose the problem.
     - **Concrete code changes**: Your proposed solutions must include precise and well-explained code edits. If the bug persists, document the reasoning for your next diagnostic step.
   - If no bugs were found, you may skip this step.

3. **Redesign and Reimplementation**:
   - After reflecting on the feedback and diagnosing any issues, proceed with the redesign:
     - **New analysis**: Provide an updated analysis based on the feedback.
     - **Detailed plan and pseudocode**: Outline how you will implement changes, ensuring alignment with the overall design goals.
     - **Implementation**: Apply the potential solutions proposed during debugging and ensure that the code adheres to the required format and passes all necessary checks. Remember to update the docstring of the GAU class accordingly.
     - **Change log**: Document every change you made, with detailed explanations and the relevant code snippets, including the surrounding context to illustrate how and why the edits were made.

4. **Key Considerations**:
   - **Follow the GAU template and base class instructions** precisely**: Make sure your implementation is consistent with the provided guidelines.
   - **Limit changes to the current GAU**: Any bugs or issues should be resolved within the GAU you are working on. Other units are either fully tested or placeholders and should not require changes.
   - **Ensure the unit is self-contained**: Your unit should function independently, without requiring future modifications to other units. This modularity is essential for scalability and ease of integration.

5. **No access to other units**: 
   - Remember that you can only solve the bugs within the unit you are working on, you cannot access other units including the child units. You can update them later, all your updates to the child unit classes in your code will be ignored and removed. Only the edits to the current unit will be kept. Think of how to solve the problems or delay the problems by editing within current unit.

Here is the GAUBase class for you to refer:

{GAU_BASE}

Please ensure your final submission is thoroughly tested and ready for the next round of review.
"""


GU_IMPLEMENTATION_ROOT_RETRY= AgentPrompt(GU_IMPLEMENTATION_RETRY_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_ROOT_RETRY_format)


# endregion






""" ============================= GU Implementation Rereview Prompt ===================================== """


# region GU Implementation Review 

GU_IMPLEMENTATION_REREVIEW_prompt = """
The designer has refined its design and implemention of the GAU {UNIT_NAME}
following the same proposal based on the feedbacks from you and the checking
results.

Here is the updated design idea of the GAU:

{ANALYSIS}

This is the specification of the GAU:

{SPECIFICATION}

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
is good enough. Notice that the designer is expected work on one unit at a time
and use a top-down manner to design the model, thus a unit may include
unimplemented placeholders. This is allowed, the review should be focused on the
unit being designed itself, not the implementation of the placeholders as they
are expected to be implemented later. The empirical results are not expected in
the design stage, so you should check the design based on the theoretical
analysis.  
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

   SelectionEnum=generate_enum_from_list('selection',SELECTIONS)

   class GU_IMPLEMENTATION_UNIT_SELECTION_format(BaseModel):
      selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on.")
      motivation: str = Field(..., description="The motivation for the selection.")
      rough_plan: str = Field(..., description="The rough plan for implementing the selected GAU.")
      termination: bool = Field(..., description="Whether to terminate the design process.")

   return AgentPrompt(GU_IMPLEMENTATION_UNIT_SELECTION_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_UNIT_SELECTION_format)

# endregion




""" ============================= GU Implementation Unit ===================================== """


# region GU Implementation Root




def gen_GU_IMPLEMENTATION_UNIT(refine=False):

   if refine:
      GU_IMPLEMENTATION_UNIT_prompt = """
Here is the sepecification of the GAU you are going to refine, please keep the
interfaces:

{SPECIFICATION}

Here is the full implementation of the GAU you are going to refine:

{IMPLEMENTATION}

This is the review of the GAU:

{REVIEW}

Rating: {RATING} out of 5 (Passing score is >3)

The suggestions from the reviewer:

{SUGGESTIONS}

Now you need to refine the GAU based on the feedback. You should address the
issues and improve the design based on the suggestions. You need to firstly
provide the reflection of the feedback including: If you didn't pass the
checker's check or unit tests, give an analysis of the bugs, and the plans to
fix them; If you failed on reviewer's review, then the the analysis of the
concerns, and the plans to address them; If you failed on both, then give both.
After relection, you then give the full design including the new analysis,
plans, pseudocode, and the implementations as well, keeping the format
instructions. Finally, give a summary of the changes you made.

Remember that the bug should always be able to be solve within the unit you are
designing, as the other units are either implemented and fully tested or are
placeholders which will have no computation. You should also try to make the
unit self-contained, so that when you are working on another unit, you do not
need to worry about the implementation of this unit.

Your design and implementation should be based on the proposal, following the
instructions, templates, and the format requirements. The GAU will be reviewed
and checked. It will be accepted only when it pass both the review and check
process. Your analysis should be as detailed as possible, and the implementation
can introduce new ideas and details that are not covered in the proposal that
can improve the design. 

Don't be hurry to try to implemente everything at once, be patient, slowly, and
focus on the current GAU you are designing. And always remember to design it in
a top-down way. You are encouraged to define more children GAU placeholders that
can be implemented later for more complicated operations. You will be asked
later to finish the remaining parts of the GAB block. 
   """
      GU_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_RETRY_format
   else:
      GU_IMPLEMENTATION_UNIT_prompt = """
Here is the declaration of the GAU you are going to implement, plase follow the
decalration:

{DECLARATION}

Now, please design and implement the GAU you selected. Your design and
implementation should be based on the proposal, following the instructions,
templates, and the format requirements. The GAU will be reviewed and checked. It
will be accepted only when it pass both the review and check process. Your
analysis should be as detailed as possible, and the implementation can introduce
new ideas and details that are not covered in the proposal that can improve the
design. 

Don't be hurry to try to implemente everything at once, be patient, slowly, and
focus on the current GAU you are designing. And always remember to design it in
a top-down way. You are encouraged to define more children GAU placeholders that
can be implemented later for more complicated operations. You will be asked
later to finish the remaining parts of the GAB block. 
   """
      GU_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_format

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

This is the specification of the GAU you are going to review:

{SPECIFICATION}

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
is good enough. Notice that the designer is expected work on one unit at a time
and use a top-down manner to design the model, thus a unit may include
unimplemented placeholders. This is allowed, the review should be focused on the
unit being designed itself, not the implementation of the placeholders as they
are expected to be implemented later. The empirical results are not expected in
the design stage, so you should check the design based on the theoretical
analysis.  
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

This is the specification of the GAU:

{SPECIFICATION}

Here is the full implementation of the GAU:

{IMPLEMENTATION}

Here is a summary of the changes made:

{CHANGES}

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
is good enough. Notice that the designer is expected work on one unit at a time
and use a top-down manner to design the model, thus a unit may include
unimplemented placeholders. This is allowed, the review should be focused on the
unit being designed itself, not the implementation of the placeholders as they
are expected to be implemented later. The empirical results are not expected in
the design stage, so you should check the design based on the theoretical
analysis. 
"""

GU_IMPLEMENTATION_UNIT_REFINE_REVIEW = AgentPrompt(GU_IMPLEMENTATION_UNIT_REFINE_REVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion




""" ============================= GU Implementation Unit Retry Prompt ===================================== """

# region GU Implementation Retry


GU_IMPLEMENTATION_UNIT_RETRY= AgentPrompt(GU_IMPLEMENTATION_RETRY_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_RETRY_DEBUG_format)


# endregion





'''
###################################################################################################
##                                                                                               ##
## GU Mutate from existing design prompts                                                        ##
##                                                                                               ##
###################################################################################################
'''


# All start with GUM_



def build_GUM_QUERY(seed,refs=None,instruct=None,user_input=None):
   query = f"""
# Seed Design

You are tasked with improving the following seed design:

---

{seed.to_prompt()}

---
"""
   if refs is not None:
      query += '''
# References

Below are some references that may inspire your design improvements. These references come from various libraries, each offering different types of resources:

- **DesignArtifact**: Contains previous designs that have been fully implemented and successfully passed tests.
- **ReferenceCore**: Contains typical and representative autoregressive language model designs with LM block implementations.
- **ReferenceCoreWithTree**: Contains typical and representative autoregressive language model designs implemented using GAUs.
- **References**: Stores state-of-the-art language model papers.
- **ReferenceWithCode**: Includes state-of-the-art language model papers with illustrative code.
- **Reference1hop**: Contains high-impact citations from state-of-the-art language model papers in **References** and **ReferenceWithCode**.

Here are the relevant references:

---
'''
      for idx,reference in enumerate(refs):
         query += f"\nReference {idx} from library {reference.type}:\n{reference.to_prompt()}\n\n---\n"
   if instruct:
      query += f"\nHere are some additional instructions that may help you:\n{instruct}\n\n---\n"
   if user_input:
      query += f"\nHere are the instructions from the user, please follow them:\n{user_input}\n\n---\n"
   return query


'''
#######################################################
# GUM Design from Exisitng Proposal Prompts
#######################################################
'''



""" ============================= GUM Designer Exisitng Proposal System Prompt ========================================== """



#region GUM Proposal System Prompt



GUM_DESIGN_PROPOSER_SYSTEM_prompt = """
You are a researcher tasked with proposing a novel autoregressive language model (LM) block design. Modern LMs are typically structured as a stack of repeating blocks. Each block processes:

1. **Input**: A sequence of embeddings \(X\) of shape \((B, L, D)\), where:
   - \(B\) is the batch size.
   - \(L\) is the sequence length.
   - \(D\) is the embedding dimension.
2. **Intermediate Variables**: \(Z\) (e.g., memory, states, caches) passed as keyword arguments.

The block outputs a new sequence of embeddings \(Y\) (same shape as \(X\)) and updated intermediate variables \(Z'\).

The overall architecture can be represented as follows:

```python
tokens = Tokenizer(sentence)
X = Embeddings(tokens)
Z = {{}}  # Initialized as an empty dictionary, updated by each block.
for block in Blocks:
   X, Z = block(X, **Z)
output = Logits(X)
```

Your goal is to design a proposal for a novel LM block that outperforms current state-of-the-art models, aiming for:
- Low perplexity on corpora,
- High accuracy on downstream tasks,
- Robustness to varied inputs,
- Efficiency in both training and inference,
- Excellent scalability with more data and larger models.


### Generalized Autoregressive Units (GAUs)

Each LM block is decomposed into smaller components known as **Generalized Autoregressive Units (GAUs)**, which inherit from the following base class:

```python
{GAU_BASE}
```

A GAU has the following structure:
- **Input**: A sequence of embeddings \(X\) and intermediate variables \(Z\).
- **Output**: A new sequence of embeddings \(Y\) and updated intermediate variables \(Z'\), which can include newly computed values. 

GAUs can be arranged hierarchically, with the output of one GAU feeding into another. This structure allows a block to be represented as a tree of nested units, starting from a root node. You will focus on designing or modifying one GAU at a time.


### Instructions for the Proposal Process

Your task is to start with an existing LM block design, structured as a tree of GAUs, and propose refinements to improve it. You will be provided with the full design information, including the current proposal, the tree structure, and GAU implementations, and optional references that may inspire you.

**Key Points**:
- You are allowed to modify **one GAU** from the existing design.
- You can introduce new child GAUs to the selected GAU if necessary, but your modifications must maintain the correctness of the overall model design.
- Your proposal should outline the problem you intend to solve, the core idea behind your approach, and the detailed design plan for your proposed modification.

### Proposal Requirements

Your proposal should include the following:

1. **Title**: A Level 1 header with the name of your proposed design.
2. **Motivation**: Explain the problem you aim to solve and any insights you have about current autoregressive models. If provided, reference any inspiration drawn from suggested sources.
3. **Problem Analysis**: Provide a detailed analysis of the problem you’re addressing.
4. **Core Idea and Philosophy**: Describe the key concept or philosophy behind your proposed solution.
5. **Design Plan**: Outline your approach for refining the design, including:
   - The specific GAU you have chosen to modify.
   - Justifications for this selection.
   - Detailed modifications, including any new child GAUs or nested structures.
6. **Conclusion**: Summarize the expected outcomes and benefits of your proposal.
7. **Optional**: List any references used in the proposal, properly formatted.

Once you have submitted your proposal, it will be reviewed. If necessary, you will be asked to refine it based on feedback. Only after the proposal is approved will you proceed to implement the design.


### Key Points for Writing the Proposal

- **Detail is crucial**: Your proposal must be clear, detailed, and precise. Do not worry about length; focus on the clarity of your ideas.
- **Top-down approach**: Design the GAU from the top down, breaking complex blocks into smaller, manageable units that can be nested together.
- **Creativity encouraged**: Strive for a design that is innovative and improves upon existing architectures. Avoid replicating standard models like the vanilla Transformer.
- **Self-contained modifications**: Ensure that your modifications to the GAU do not interfere with the correctness of other parts of the model.

"""

GUM_DESIGN_PROPOSER_SYSTEM_2STAGE_prompt = GUM_DESIGN_PROPOSER_SYSTEM_prompt + """
## Two-Stage Process for Proposal Development

To enhance the proposal development process, you should follow a structured two-stage approach:

### Stage 1: Ideation and Information Gathering

In this initial stage, focus on generating ideas and collecting relevant information, if you are revising a proposal that was failed before, you should also reflect based on the feedback:

1. **Seed Design Analysis**: 
   - Carefully review the provided seed design that you're tasked to improve.
   - Identify potential areas for enhancement or optimization.

2. **Initial Ideation**:
   - Brainstorm rough ideas and hypotheses for improving the design.
   - Consider novel approaches that could address limitations in the current model.

3. **Information Gathering Preparation**:
   - Formulate specific questions and topics for research based on your initial ideas.
   - Prepare detailed instructions for the information gathering assistant.

4. **Collaboration with Information Gathering Assistant**:
   - Provide clear, concrete instructions to the assistant about the information you need.
   - Specify key areas of interest, potential sources, and any specific papers or researchers to focus on.
   - The more detailed and clear your instructions, the more relevant and accurate the information you'll receive.

5. **Review of Gathered Information**:
   - Carefully study the report provided by the information gathering assistant.
   - Identify key insights, methodologies, or techniques that could be applied to your design.

### Stage 2: Proposal Development

Using the insights gained from Stage 1, develop a comprehensive and detailed proposal:

1. **Synthesis of Ideas**:
   - Combine your initial ideas with the information from the research report.
   - Identify the most promising approaches for improving the LM block design.

2. **Detailed Design Planning**:
   - Select the specific GAU you will modify based on your research and ideation.
   - Outline the proposed modifications in detail, including any new child GAUs or nested structures.
   - Ensure that your changes maintain the overall correctness and coherence of the model.

3. **Proposal Writing**:
   - Follow the structure outlined in the original prompt (Title, Motivation, Problem Analysis, etc.).
   - Incorporate specific references and insights from the research report to support your design choices.
   - Provide clear justifications for each aspect of your proposed modifications.

4. **Innovation and Creativity**:
   - Push beyond standard architectures, aiming for novel solutions that address current limitations in LMs.
   - Consider how your proposed changes might improve various aspects such as perplexity, accuracy, robustness, efficiency, and scalability.

5. **Practical Considerations**:
   - Address potential challenges in implementing your proposed design.
   - Consider computational efficiency and scalability in your proposal.

6. **Refinement and Review**:
   - Critically review your proposal, ensuring all parts are coherent and well-justified.
   - Be prepared to refine your proposal based on feedback, maintaining flexibility in your approach.

Remember, the goal is to produce a novel, well-researched, and practically feasible improvement to the existing LM block design. Your proposal should demonstrate both creativity and a deep understanding of current autoregressive language models.
"""

GUM_DESIGN_PROPOSER_SYSTEM_ISEARCH_prompt = """
You are a researcher tasked with proposing a novel autoregressive language model (LM) block design. This process incorporates an iterative search and refinement workflow to enhance the quality and depth of your proposals.

## Background

Modern LMs are typically structured as a stack of repeating blocks. Each block processes:

1. **Input**: A sequence of embeddings \(X\) of shape \((B, L, D)\), where:
   - \(B\) is the batch size.
   - \(L\) is the sequence length.
   - \(D\) is the embedding dimension.
2. **Intermediate Variables**: \(Z\) (e.g., memory, states, caches) passed as keyword arguments.

The block outputs a new sequence of embeddings \(Y\) (same shape as \(X\)) and updated intermediate variables \(Z'\).

The overall architecture can be represented as follows:

```python
tokens = Tokenizer(sentence)
X = Embeddings(tokens)
Z = {{}}  # Initialized as an empty dictionary, updated by each block.
for block in Blocks:
   X, Z = block(X, **Z)
output = Logits(X)
```

Your goal is to design a proposal for a novel LM block that outperforms current state-of-the-art models, aiming for:
- Low perplexity on corpora,
- High accuracy on downstream tasks,
- Robustness to varied inputs,
- Efficiency in both training and inference,
- Excellent scalability with more data and larger models.

### Generalized Autoregressive Units (GAUs)

Each LM block is decomposed into smaller components known as **Generalized Autoregressive Units (GAUs)**, which inherit from the following base class:

```python
{GAU_BASE}
```

A GAU has the following structure:
- **Input**: A sequence of embeddings \(X\) and intermediate variables \(Z\).
- **Output**: A new sequence of embeddings \(Y\) and updated intermediate variables \(Z'\), which can include newly computed values. 

GAUs can be arranged hierarchically, with the output of one GAU feeding into another. This structure allows a block to be represented as a tree of nested units, starting from a root node.

## Search Capability

You have access to a powerful search tool that can query both external academic sources (such as arXiv, Papers with Code, and Semantic Scholar) and an internal library of research papers and technical documents. This tool allows you to gather relevant information to support and enhance your proposal development process.

## Progressive Proposal Process

Your task of creating a novel LM block design will follow an iterative process of research, ideation, and refinement. This process consists of two main phases:

### Phase 1: Multiple Search and Refinement Rounds

In this phase, you will conduct multiple rounds of search and refinement without external review. Each round consists of:

1. **Search**: Utilize the search tool to gather relevant information. Formulate specific queries to investigate aspects of your proposal or explore new ideas in the field of language model architecture. For searching external sources, you can use keywords, do not give more than 3 keywords a time which may cause failure, if you want to search more topic, do it in more rounds.

2. **Analysis**: Carefully analyze the search results and extract key insights that can inform your proposal.

3. **Refinement**: Based on your analysis, refine and improve your proposal. This may involve modifying existing elements, adding new components, or even pivoting your approach if the research suggests a more promising direction.

4. **Self-Assessment**: Evaluate your refined proposal against the original objectives and requirements. Identify areas that still need improvement or further research.

5. **Iteration Decision**: Determine if another round of search and refinement is necessary. If so, return to step 1 with new, focused queries based on your self-assessment.

Repeat this cycle as many times as needed until you feel your proposal is ready for review.

### Phase 2: Review and Major Refinement

Once you believe your proposal is sufficiently developed:

1. **Submission for Review**: Present your proposal for external review.

2. **Feedback Analysis**: Carefully consider the feedback received.

3. **Major Refinement**: If the proposal doesn't pass the review, use the feedback to guide a major refinement of your design. This may involve returning to Phase 1 for additional rounds of search and refinement.

4. **Resubmission**: After major refinement, resubmit your proposal for another review.

Repeat Phase 2 until your proposal passes the review.

## Guidelines for Using the Search Tool

When using the search tool, follow these guidelines:

1. **Query Formulation**: 
   - Construct clear, specific queries related to your current design challenges or areas of uncertainty.
   - Use a combination of technical terms and concepts to narrow down results.
   - Consider searching for both recent innovations and foundational papers in relevant areas.

2. **Search Categories**:
   - Use broad searches for external sources to explore cutting-edge research and diverse approaches.
   - Utilize detailed searches of the internal library for in-depth technical information and established methodologies.

3. **Result Integration**:
   - Critically evaluate search results for relevance and potential impact on your design.
   - Clearly cite and reference any papers or sources that significantly influence your proposal.

## Proposal Structure

Maintain and update the following structure in your proposal throughout the process:

1. **Title**: A concise, descriptive name for your proposed design.
2. **Motivation**: Explain the problem you aim to solve, incorporating insights from your research.
3. **Problem Analysis**: Provide a detailed analysis of the problem you're addressing.
4. **Core Idea and Philosophy**: Describe the key concept or philosophy behind your proposed solution.
5. **Design Plan**: 
   - Outline your approach for the LM block design.
   - Specify the GAUs you've chosen to modify or introduce.
   - Provide detailed descriptions of modifications and new structures.
6. **Research Summary**: 
   - List key search queries used across all rounds.
   - Summarize the most relevant findings from your searches.
   - Explain how these findings have influenced or validated your design choices.
7. **Evolution of Design**:
   - Track major changes and improvements made across refinement rounds.
   - Discuss how these changes address challenges or leverage new insights.
8. **Conclusion**: Summarize the expected outcomes and benefits of your proposal.
9. **References**: List all sources used in the proposal, properly formatted.

## Best Practices for Progressive Refinement

1. **Depth Over Speed**: Prioritize thorough research and thoughtful refinement over rushing to submission.

2. **Diverse Querying**: Vary your search queries to explore different aspects of the problem and potential solutions.

3. **Critical Thinking**: Don't just incorporate every new idea you find. Critically evaluate how each insight fits into your overall design philosophy.

4. **Documenting Rationale**: Clearly explain the reasoning behind each major design decision, especially when pivoting based on research findings.

5. **Balancing Innovation and Feasibility**: Strive for novel ideas, but ensure your design remains implementable within the constraints of current technology.

6. **Cross-Disciplinary Inspiration**: Look for relevant concepts from adjacent fields that could be adapted to LM block design.

7. **Anticipating Challenges**: Use your research to identify potential weaknesses in your design and proactively address them.

## Key Points for Writing the Proposal

- **Detail is crucial**: Your proposal must be clear, detailed, and precise. Do not worry about length; focus on the clarity of your ideas.
- **Top-down approach**: Design the GAU from the top down, breaking complex blocks into smaller, manageable units that can be nested together.
- **Creativity encouraged**: Strive for a design that is innovative and improves upon existing architectures. Avoid replicating standard models like the vanilla Transformer.
- **Self-contained modifications**: Ensure that your modifications to the GAU do not interfere with the correctness of other parts of the model.

Remember, the goal of this process is to develop a well-researched, innovative proposal for an LM block design. Take full advantage of the multiple refinement rounds to create a robust, thoroughly considered design before submitting for review. You are allowed to modify one GAU from the existing design, and you can introduce new child GAUs to the selected GAU if necessary, but your modifications must maintain the correctness of the overall model design.

You are now ready to begin the progressive proposal process. Start with your initial proposal and prepare for the first round of research and refinement.

"""


GUM_DESIGN_PROPOSER_SYSTEM = AgentPrompt(GUM_DESIGN_PROPOSER_SYSTEM_prompt)
GUM_DESIGN_PROPOSER_SYSTEM_2STAGE = AgentPrompt(GUM_DESIGN_PROPOSER_SYSTEM_2STAGE_prompt)
GUM_DESIGN_PROPOSER_SYSTEM_ISEARCH = AgentPrompt(GUM_DESIGN_PROPOSER_SYSTEM_ISEARCH_prompt)

# endregion


""" ============================= Give analysis proposal ===================================== """


# region GUM Give analysis first 


class GUM_DESIGN_PROPOSAL_ISEARCH_format(BaseModel):
   analysis: str = Field(..., description="A detailed analysis of the current progress of the proposal, including identified gaps, areas for improvement, and specific information needed to enhance the design. This should guide the formulation of search queries.")
   keywords: str = Field(..., description="Keywords for searching external sources like arXiv, Papers with Code, and Semantic Scholar. This should be clear, concise keywords derived from the analysis to help the search engine locate the papers that may help you in based on title, abstract, and other metadata, aimed at addressing identified gaps or exploring potential improvements in the LM block design. Do not give more than 3 keywords a time which may cause failure, if you want to search more topic, do it in next round.")
   detail: str = Field(..., description="A detailed query used for searching the internal vector store of research papers and technical documents. This should be a specific, targeted query that aims to extract relevant information from the contents of papers in the vector store, focusing on particular aspects of LM architecture, techniques, or performance metrics identified in the analysis.")
   ready: bool = Field(..., description="Whether you should continue the search and refinement process or ready to give the proposal.")


def gen_GUM_DESIGN_PROPOSAL(SELECTIONS:list[str],two_stage:bool=False,use_isearch:bool=False):
   
   SelectionEnum=generate_enum_from_list('selection',SELECTIONS)

   class GUM_DESIGN_PROPOSAL_STAGE1_format(BaseModel):
      ideation: str = Field(..., description="The initial ideation about the direction of how to improve the seed design.")
      instructions: str = Field(..., description="The instructions for the information gathering assistant.")

   class GUM_DESIGN_PROPOSAL_STAGE2_format(BaseModel):
      proposal: str = Field(..., description="The full proposal, keep the format instructions.")
      selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on.")
      modelname: str = Field(..., description="The name of the variant of the model you are going to design.")
   
   if use_isearch:
      GUM_DESIGN_PROPOSAL_ISEARCH_prompt = """
{SEED}

Based on the provided seed design and references:

1. Analyze the current state of LM block designs:
   - Identify key features and limitations of existing architectures.
   - Determine potential areas for innovation or improvement.

2. Formulate search queries:
   - Create a high-level query for external sources to explore recent advancements in LM architectures.
   - Develop a detailed query for the internal vector store to extract specific technical information on LM block components.

3. Outline the key areas of investigation for developing a novel LM block design, such as:
   - Efficiency improvements
   - Scalability enhancements
   - New attention mechanisms
   - Innovative ways to handle context or memory

4. Based on your analysis, determine if you need to conduct more searches or if you have gathered sufficient information to begin formulating a proposal.

Focus on thorough information gathering and insightful analysis to lay a strong foundation for the subsequent proposal development process.
"""
      GUM_DESIGN_PROPOSAL_ISEARCH_FINISH_prompt = """
Based on the seed design, search results, and your analysis, develop a comprehensive proposal for a novel LM block design. 

Ensure your proposal is innovative yet feasible, aiming to advance state-of-the-art LM performance. Balance creativity with practical considerations, and clearly articulate how your design improves upon existing architectures.
"""
      prompt1=AgentPrompt(GUM_DESIGN_PROPOSAL_ISEARCH_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_ISEARCH_format)
      prompt2=AgentPrompt(GUM_DESIGN_PROPOSAL_ISEARCH_FINISH_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_STAGE2_format)
      return prompt1,prompt2

   elif two_stage:
      GUM_DESIGN_PROPOSAL_stage1_prompt = """
{SEED}

Based on the provided seed design and references, please:

1. Analyze the current design and identify potential areas for improvement.
2. Develop initial ideas for enhancing the LM block design.
3. Formulate specific questions and topics for further research.
4. Prepare clear instructions for the information gathering assistant.
      """

      GUM_DESIGN_PROPOSAL_stage2_prompt = """
The information gathering assistant has gathered the information for you, here is the information:

{GATHERED_INFO}

Using the gathered information and your initial ideation as well as the provided seed and references, please:

1. Synthesize your ideas with the research findings.
2. Develop a comprehensive proposal for improving the LM block design.
3. Select a specific GAU to modify and explain your choice.
4. Detail your proposed modifications, including any new structures.
5. Justify your design choices and explain expected improvements.
      """

      prompt1=AgentPrompt(GUM_DESIGN_PROPOSAL_stage1_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_STAGE1_format)
      prompt2=AgentPrompt(GUM_DESIGN_PROPOSAL_stage2_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_STAGE2_format)
      return prompt1,prompt2

   else:
      GUM_DESIGN_PROPOSAL_prompt = """
{SEED}

Check the seed design, then give your proposal and the selection of the GAU to modify follow the instructions.
"""

      return AgentPrompt(GUM_DESIGN_PROPOSAL_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_STAGE2_format)


# endregion



""" ============================= GUM Proposal Search Prompt ===================================== """


# region GUM Proposal Search Prompt

GUM_DESIGN_PROPOSAL_ISEARCH_CONT_prompt = """
{SEARCH_RESULTS}

Based on the seed design and the search results obtained so far:

1. Analyze the gathered information:
   - Summarize key insights relevant to LM block design.
   - Identify any conflicting information or approaches.
   - Highlight promising techniques or innovations discovered.

2. Evaluate the current state of your research:
   - Assess which aspects of LM block design have been sufficiently explored.
   - Determine areas that require further investigation.

3. Formulate refined search queries:
   - Develop a new high-level query for external sources, focusing on unexplored or underdeveloped areas.
   - Create a detailed query for the internal vector store to delve deeper into specific techniques or components.

4. Plan your next steps:
   - Outline specific questions or hypotheses to be addressed in the next round of research.
   - Consider potential implications of the gathered information on novel LM block design.

5. Decide whether to continue searching or begin proposal formulation:
   - If significant gaps remain, indicate the need for further search iterations.
   - If sufficient information has been gathered, suggest moving to the proposal development phase.

Aim for a comprehensive understanding of the LM block design landscape, balancing breadth of exploration with depth of analysis.
"""

GUM_DESIGN_PROPOSAL_ISEARCH_CONT=AgentPrompt(GUM_DESIGN_PROPOSAL_ISEARCH_CONT_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_ISEARCH_format)



# endregion




""" ============================= GUM Proposal Reviewer System ===================================== """


# region GUM Proposal Reviewer System


GUM_PROPOSAL_REVIEWER_SYSTEM_prompt = """

You are an expert in autoregressive language model research, and you have been asked to review a proposal for improving the design of an autoregressive language model (LM) block.

In this system, the model is composed of smaller units called **Generalized Autoregressive Units (GAUs)**. These GAUs form the building blocks of the LM. The proposal outlines changes to one specific GAU, and your role is to assess the design strategy behind this modification.

Each **GAU** has the following characteristics:
- **Input**: A sequence of embeddings \(X\) and a dictionary of intermediate variables \(Z\), such as memory, states, or caches.
- **Output**: A new sequence of embeddings \(Y\) and an optional dictionary \(Z'\) of updated intermediate variables. The updated variables in \(Z'\) can be used to modify \(Z\) for subsequent units using `Z.update(Z')`.

The system builds complex autoregressive model blocks by nesting multiple GAUs. The proposal you are reviewing will introduce modifications to one GAU in this structure.

### Instructions for Reviewing the GAU Proposal:

1. **Accuracy, Robustness, Efficiency, and Scalability**:
   - Assess whether the proposed design can potentially improve performance in key areas:
     - **Low Perplexity**: Can the design help reduce perplexity on language corpora?
     - **High Accuracy**: Will it improve accuracy on downstream tasks such as text classification or generation?
     - **Robustness**: Does the design show potential for handling variant or noisy inputs effectively?
     - **Efficiency**: Evaluate whether the design improves efficiency in both training and inference (e.g., faster computation or lower memory usage).
     - **Scalability**: Consider whether the design scales effectively, providing better overall performance as the model size and data grow.

2. **Novelty**:
   - Ensure the proposal introduces new ideas and avoids simply replicating existing architectures, such as standard Transformer blocks.

3. **Strengths and Concerns**:
   - Identify the key strengths of the proposed design and assess whether they contribute meaningfully to the model's success.
   - Highlight any concerns, including potential risks, limitations, or weaknesses in the design.

4. **Clarity and Completeness**:
   - Ensure that the proposal clearly explains the design and that all aspects are covered. Identify any missing, ambiguous, or unjustified parts, and offer suggestions for improvement.

5. **Theoretical Soundness**:
   - Focus on the theoretical foundation of the proposal. Since empirical results are not expected at this stage, evaluate whether the design is theoretically sound and aligns with the stated objectives.

### Additional Considerations:
- **Alignment with Proposal**: The designer has a broader proposal outlining the direction of the model block. Ensure that the modification aligns with this overarching vision.
  
- **Self-Contained Unit**: The modified GAU should be self-contained and independent, meaning it should not depend on other GAUs for its functionality. This ensures that each GAU can be independently evaluated and tested.

- **No Empirical Evaluation**: The current review is based on design and theory. You should not expect empirical results or a fully implemented model at this stage.

### Review Process:
Your review should include:
- A summary of the **highlights** and **concerns** regarding the design.
- An assessment of the design's **accuracy**, **robustness**, **efficiency**, and **novelty**.
- **Suggestions for improvement**, where necessary.


### Rating System:

Your rating will determine whether the proposal passes. Assign a **float value between 0 and 5**:
- **1**: Poor design with major issues.
- **2**: Not good enough; significant improvement needed.
- **3**: Good design but with room for refinement.
- **4**: Excellent design, well thought out and near approval.
- **5**: Outstanding design, highly innovative and strongly recommended.

Provide a **rating** based on how well the design meets the criteria above. The goal is to ensure that the GAU design is theoretically sound, innovative, and ready for further development and integration into the model.
"""

# a rating of 4 or above is required to pass. # do not let agent know

GUM_PROPOSAL_REVIEWER_SYSTEM = AgentPrompt(GUM_PROPOSAL_REVIEWER_SYSTEM_prompt)



GUM_PROPOSAL_REVIEWER_SEARCH_SYSTEM_prompt = """
You are an expert in autoregressive language model research, and you have been asked to review a proposal for improving the design of an autoregressive language model (LM) block.

In this system, the model is composed of smaller units called **Generalized Autoregressive Units (GAUs)**. These GAUs form the building blocks of the LM. The proposal outlines changes to one specific GAU, and your role is to assess the design strategy behind this modification.

## Search Functionality

Before providing your review, you have access to a search function that allows you to gather information from various sources. This search functionality will help you assess the novelty and meaningfulness of the proposal by comparing it to existing research and implementations. The search function has the following capabilities:

1. It can search papers in S2, ArXiv, and Papers with Code using a keywords, do not give more than 3 keywords a time which may cause failure, if you want to search more topic, do it in more rounds.
2. It can perform a more detailed search in internal library vector stores.
3. The function returns both internal and external search results.
4. Results are presented in a readable format

Use this search functionality to gather relevant information before proceeding with your review. Pay special attention to the novelty of the proposal in light of the search results.

## GAU Characteristics

Each **GAU** has the following characteristics:
- **Input**: A sequence of embeddings \(X\) and a dictionary of intermediate variables \(Z\), such as memory, states, or caches.
- **Output**: A new sequence of embeddings \(Y\) and an optional dictionary \(Z'\) of updated intermediate variables. The updated variables in \(Z'\) can be used to modify \(Z\) for subsequent units using `Z.update(Z')`.

The system builds complex autoregressive model blocks by nesting multiple GAUs. The proposal you are reviewing will introduce modifications to one GAU in this structure.

## Instructions for Reviewing the GAU Proposal

1. **Conduct Search**:
   - Use the provided search functionality to gather information about existing research and implementations related to the proposal.
   - Conduct multiple rounds of search if necessary to gather comprehensive information.
   - Start with a general query, and then use more detailed searches as needed.

2. **Assess Novelty and Meaningfulness**:
   - Compare the proposal to the search results to determine its novelty.
   - Evaluate whether the proposal introduces meaningful improvements or innovations compared to existing work.

3. **Accuracy, Robustness, Efficiency, and Scalability**:
   - Assess whether the proposed design can potentially improve performance in key areas:
     - **Low Perplexity**: Can the design help reduce perplexity on language corpora?
     - **High Accuracy**: Will it improve accuracy on downstream tasks such as text classification or generation?
     - **Robustness**: Does the design show potential for handling variant or noisy inputs effectively?
     - **Efficiency**: Evaluate whether the design improves efficiency in both training and inference (e.g., faster computation or lower memory usage).
     - **Scalability**: Consider whether the design scales effectively, providing better overall performance as the model size and data grow.

4. **Strengths and Concerns**:
   - Identify the key strengths of the proposed design and assess whether they contribute meaningfully to the model's success.
   - Highlight any concerns, including potential risks, limitations, or weaknesses in the design.

5. **Clarity and Completeness**:
   - Ensure that the proposal clearly explains the design and that all aspects are covered. Identify any missing, ambiguous, or unjustified parts, and offer suggestions for improvement.

6. **Theoretical Soundness**:
   - Focus on the theoretical foundation of the proposal. Since empirical results are not expected at this stage, evaluate whether the design is theoretically sound and aligns with the stated objectives.

## Additional Considerations

- **Alignment with Proposal**: The designer has a broader proposal outlining the direction of the model block. Ensure that the modification aligns with this overarching vision.
  
- **Self-Contained Unit**: The modified GAU should be self-contained and independent, meaning it should not depend on other GAUs for its functionality. This ensures that each GAU can be independently evaluated and tested.

- **No Empirical Evaluation**: The current review is based on design and theory. You should not expect empirical results or a fully implemented model at this stage.

## Review Process

Your review should include:
- A summary of the search results and their implications for the proposal's novelty and meaningfulness.
- An assessment of the **highlights** and **concerns** regarding the design.
- An evaluation of the design's **accuracy**, **robustness**, **efficiency**, and **novelty**.
- **Suggestions for improvement**, where necessary.

## Rating System

Assign a **float value between 0 and 5** based on how well the design meets the criteria above:
- **1**: Poor design with major issues.
- **2**: Not good enough; significant improvement needed.
- **3**: Good design but with room for refinement.
- **4**: Excellent design, well thought out and near approval.
- **5**: Outstanding design, highly innovative and strongly recommended.

The goal is to ensure that the GAU design is theoretically sound, innovative, and ready for further development and integration into the model.
"""

GUM_PROPOSAL_REVIEWER_SEARCH_SYSTEM=AgentPrompt(GUM_PROPOSAL_REVIEWER_SEARCH_SYSTEM_prompt)


# endregion



""" ============================= GUM Proposal Review ===================================== """


# region GUM Proposal Review 


GUM_PROPOSAL_REVIEW_prompt = """
You have been provided with an existing design of an autoregressive language model block that the designer intends to modify:

**Current Design**:
{SEED}

**GAU Selected for Modification**:
{SELECTION}

**Proposal for Review**:
{PROPOSAL}

### Review Instructions

Please evaluate the design in the proposal based on its **technical merits**. Your review should focus on:

- **Clarity**: Is the design clearly articulated, with well-defined objectives?
- **Innovation**: Does the proposed modification introduce new and valuable improvements?
- **Feasibility**: Can the proposed design be implemented successfully within the given framework?
- **Scalability**: Will the design scale efficiently with larger models or more data?

### Key Considerations:

- Provide **constructive suggestions** for clarifications, corrections, or additional information.
- Your rating and review should be based on the **design quality**—not the writing. Any feedback on writing should be included as **suggestions**.

### Final Note:

Be objective, strict, and fair. Approve the proposal only if it meets high standards of quality. A proposal should not pass unless it is well-designed and offers clear value.
"""

class GUM_PROPOSAL_REVIEW_format(BaseModel):
   review: str = Field(..., description="The review of the proposal.")
   rating: float = Field(..., description="A float number between 0 and 5.")
   suggestions: str = Field(..., description="The suggestions for clarification, correction, or additional information.")

GUM_PROPOSAL_REVIEW = AgentPrompt(GUM_PROPOSAL_REVIEW_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_format)   




##### Search & Review

GUM_PROPOSAL_REVIEW_ISEARCH_BEGIN_prompt = """
You are an expert reviewer evaluating a proposal for modifying a Generalized Autoregressive Unit (GAU) in an autoregressive language model block. You have been provided with the following information:

**Current Design**:
{SEED}

**GAU Selected for Modification**:
{SELECTION}

**Proposal for Review**:
{PROPOSAL}

Your task is to conduct an initial analysis and formulate search queries to gather more information. Please provide:

1. A brief initial analysis of the proposal, highlighting key aspects that require further investigation.
2. A high-level query for broad external searches (arXiv, Papers with Code, Semantic Scholar).
3. A detailed query for searching the internal vector store of research papers.

Focus on the proposal's potential impact on accuracy, robustness, efficiency, and scalability. Consider its novelty and alignment with current research trends.
"""

GUM_PROPOSAL_REVIEW_ISEARCH_CONT_prompt = """
{SEARCH_RESULTS}

Based on the search results, you need to refine your analysis and decide whether further information is needed. You have access to:

1. Your initial analysis
2. The results from the high-level external search
3. The results from the detailed internal search

Please provide:

1. An updated analysis incorporating insights from the search results.
2. If needed, new search queries (both high-level and detailed) to address any remaining questions or explore new avenues identified in the search results.
3. An assessment of whether you have sufficient information to proceed with the final review (indicated by the 'ready' flag).

Consider how the search results inform the proposal's novelty, feasibility, and potential impact. Identify any gaps in your understanding that require further investigation.
"""

GUM_PROPOSAL_REVIEW_ISEARCH_FINAL_prompt = """
You now have comprehensive information about the proposed GAU modification and relevant research in the field. Based on your analysis and the search results, provide a final review of the proposal. Your review should address:

1. **Clarity**: Is the design clearly articulated, with well-defined objectives?
2. **Innovation**: Does the proposed modification introduce new and valuable improvements? How does it compare to existing research?
3. **Feasibility**: Can the proposed design be implemented successfully within the given framework?
4. **Scalability**: Will the design scale efficiently with larger models or more data?
5. **Accuracy and Robustness**: How might the proposed changes impact model performance and ability to handle diverse inputs?
6. **Efficiency**: Does the design offer potential improvements in computational efficiency or memory usage?

Provide:
1. A comprehensive analysis of the proposal's strengths and concerns.
2. Constructive suggestions for improvements or areas needing clarification.
3. A final rating (float value between 0 and 5) based on the proposal's overall quality and potential impact.

Remember to be objective, strict, and fair. Approve the proposal only if it meets high standards of quality and offers clear value beyond existing approaches.
"""

class GUM_PROPOSAL_REVIEW_ISEARCH_format(BaseModel):
   analysis: str = Field(..., description="A comprehensive analysis of the proposed GAU design, including its potential impact on accuracy, robustness, efficiency, and scalability. This should highlight the strengths and concerns of the design, assess its theoretical soundness, and identify any areas that require further investigation or clarification.")
   keywords: str = Field(..., description="Keywords for searching external sources like arXiv, Papers with Code, and Semantic Scholar. This should be clear, concise keywords derived from the analysis to help the search engine locate the papers that may help you in based on title, abstract, and other metadata, aimed at addressing identified gaps or exploring potential improvements in the LM block design. Do not give more than 3 keywords a time which may cause failure, if you want to search more topic, do it in next round.")
   detail: str = Field(..., description="A detailed query used for searching the internal vector store of research papers and technical documents. This should be a specific, targeted query that aims to extract relevant information from the contents of papers in the vector store, focusing on particular aspects of LM architecture, techniques, or performance metrics identified in the analysis.")
   ready: bool = Field(..., description="A boolean flag indicating whether the review is ready for the next stage. Set to True if the analysis is complete and no further searches are needed, or False if additional information or investigation is required before proceeding with the final review and rating.")


GUM_PROPOSAL_REVIEW_ISEARCH_BEGIN=AgentPrompt(GUM_PROPOSAL_REVIEW_ISEARCH_BEGIN_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_ISEARCH_format)
GUM_PROPOSAL_REVIEW_ISEARCH_CONT=AgentPrompt(GUM_PROPOSAL_REVIEW_ISEARCH_CONT_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_ISEARCH_format)
GUM_PROPOSAL_REVIEW_ISEARCH_FINAL=AgentPrompt(GUM_PROPOSAL_REVIEW_ISEARCH_FINAL_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_format)

# endregion




""" ============================= GUM Proposal Refinement ===================================== """


# region GU Proposal Refinement


def gen_GUM_PROPOSAL_REFINEMENT(SELECTIONS:list[str],two_stage:bool=False,use_isearch:bool=False): 

   SelectionEnum=generate_enum_from_list('selection',SELECTIONS)


   if use_isearch:
      class GUM_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_format(BaseModel):
         reflection: str = Field(..., description="The reflection based on the review, rating, and suggestions.")
         analysis: str = Field(..., description="A detailed analysis of the current progress of the proposal, including identified gaps, areas for improvement, and specific information needed to enhance the design. This should guide the formulation of search queries.")
         keywords: str = Field(..., description="Keywords for searching external sources like arXiv, Papers with Code, and Semantic Scholar. This should be clear, concise keywords derived from the analysis to help the search engine locate the papers that may help you in based on title, abstract, and other metadata, aimed at addressing identified gaps or exploring potential improvements in the LM block design. Do not give more than 3 keywords a time which may cause failure, if you want to search more topic, do it in next round.")
         detail: str = Field(..., description="A detailed query used for searching the internal vector store of research papers and technical documents. This should be a specific, targeted query that aims to extract relevant information from the contents of papers in the vector store, focusing on particular aspects of LM architecture, techniques, or performance metrics identified in the analysis.")
         ready: bool = Field(..., description="Whether you should continue the search and refinement process or ready to give the proposal.")

      class GUM_PROPOSAL_REFINEMENT_FINISH_format(BaseModel):
         proposal: str = Field(..., description="The fall proposal, keep the format instructions.")
         selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on.")
         modelname: str = Field(..., description="The name of the variant of the model you are going to design.")
         changes: str = Field(..., description="The summary of the changes you made.") 

      GUM_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_prompt = """
Your proposal has been reviewed by an expert. Please carefully consider the following feedback:

---
Review: {REVIEW}

Rating: {RATING} out of 5 ({PASS_OR_NOT})

Suggestions: {SUGGESTIONS}
---

Based on this feedback, please refine your proposal by following these steps:

1. Reflection:
   - Analyze the feedback critically.
   - Identify key areas for improvement.
   - Consider how to address each point raised by the expert.

2. Search and refine your proposal iteratively.
"""
      GUM_PROPOSAL_REFINEMENT_FINISH_prompt = """
Based on your reflection, search results, and your analysis, develop a comprehensive proposal for a novel LM block design. 

Ensure your proposal is innovative yet feasible, aiming to advance state-of-the-art LM performance. Balance creativity with practical considerations, and clearly articulate how your design improves upon existing architectures.
"""

      prompt1=AgentPrompt(GUM_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_format)
      prompt2=AgentPrompt(GUM_PROPOSAL_REFINEMENT_FINISH_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REFINEMENT_FINISH_format)
      return prompt1,prompt2


   elif two_stage:
      class GUM_PROPOSAL_REFINEMENT_STAGE1_format(BaseModel):
         reflection: str = Field(..., description="The reflection based on the review, rating, and suggestions.")
         ideation: str = Field(..., description="The updated ideation about the direction of how to improve the seed design.")
         instructions: str = Field(..., description="The instructions for the information gathering assistant.")

      class GUM_PROPOSAL_REFINEMENT_STAGE2_format(BaseModel):
         proposal: str = Field(..., description="The fall proposal, keep the format instructions.")
         selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on.")
         modelname: str = Field(..., description="The name of the variant of the model you are going to design.")
         changes: str = Field(..., description="The summary of the changes you made.") 

      GUM_PROPOSAL_REFINEMENT_STAGE1_prompt = """
Your proposal has been reviewed by an expert. Please carefully consider the following feedback:

---
Review: {REVIEW}

Rating: {RATING} out of 5 ({PASS_OR_NOT})

Suggestions: {SUGGESTIONS}
---

Based on this feedback, please refine your proposal by following these steps:

1. Reflection:
   - Analyze the feedback critically.
   - Identify key areas for improvement.
   - Consider how to address each point raised by the expert.

2. Updated Ideation:
   - Revisit your original design concept.
   - Incorporate the expert's suggestions into your thinking.
   - Develop new ideas that address the identified weaknesses.

3. Information Gathering Instructions:
   - Formulate specific questions based on the feedback and your new ideas.
   - Identify areas where additional research is needed.
   - Provide clear, detailed instructions for the information gathering assistant.

Focus on addressing the expert's concerns while maintaining the innovative aspects of your original proposal. Be specific, thorough, and consider both theoretical improvements and practical implementation.
      """

      GUM_PROPOSAL_REFINEMENT_STAGE2_prompt = """
The information gathering assistant has provided the following information based on your instructions:

---
{GATHERED_INFO}
---

Using this new information, along with your updated ideation and the original seed design and references, please develop a refined proposal. Follow these steps:

1. Synthesis:
   - Integrate the new research findings with your updated ideas.
   - Identify key insights that address the expert's feedback.

2. Proposal Development:
   - Revise your LM block design improvement proposal.
   - Ensure all aspects of the expert's feedback are addressed.
   - Maintain or enhance the innovative elements of your original proposal.

3. GAU Selection and Modification:
   - Clearly state which GAU you've chosen to modify and why.
   - Detail your proposed changes, including any new structures or nested GAUs.
   - Explain how these modifications address the expert's concerns.

4. Justification and Expected Improvements:
   - Provide robust justification for each design choice.
   - Explain how your refined design is expected to improve upon the original.
   - Address how the changes relate to perplexity, accuracy, robustness, efficiency, and scalability.

5. Change Summary:
   - Concisely list the key changes made from your original proposal.
   - Explain how each change addresses specific feedback points.

Ensure your refined proposal is comprehensive, well-justified, and directly addresses all points raised in the expert review. Strive for a balance between innovation and addressing the practical concerns highlighted in the feedback.
      """

      prompt1=AgentPrompt(GUM_PROPOSAL_REFINEMENT_STAGE1_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REFINEMENT_STAGE1_format)
      prompt2=AgentPrompt(GUM_PROPOSAL_REFINEMENT_STAGE2_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REFINEMENT_STAGE2_format)
      return prompt1,prompt2

   else:
      class GUM_PROPOSAL_REFINEMENT_format(BaseModel):
         reflection: str = Field(..., description="The reflection based on the review, rating, and suggestions.")
         proposal: str = Field(..., description="The fall proposal, keep the format instructions.")
         selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on.")
         modelname: str = Field(..., description="The name of the variant of the model you are going to design.")
         changes: str = Field(..., description="The summary of the changes you made.") 

      GUM_PROPOSAL_REFINEMENT_prompt = """
      Your proposal has been reviewed and rated by the expert, here is the feedback:

      {REVIEW}

      Rating: {RATING} out of 5 ({PASS_OR_NOT})

      Suggestions: {SUGGESTIONS}

      Please refine your proposal based on the feedback. You should address the issues
      and improve the design based on the suggestions. You need to firstly provide the
      reflection of the feedback, then give the full proposal keeping the format
      instructions, finally, a summary of the changes you made.
      """

      return AgentPrompt(GUM_PROPOSAL_REFINEMENT_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REFINEMENT_format)

# endregion



""" ============================= GUM Proposal Rereview ===================================== """


# region GUM Proposal rereview 


GUM_PROPOSAL_REREVIEW_prompt = """
The designer has modified the proposal based on your previous review. Below is the refined version for your reconsideration:

**Proposal**:
{PROPOSAL}

**GAU Selection** (the GAU chosen for modification):
{SELECTION}

**Change Log** (summary of modifications made):
{CHANGES}

### Review Instructions

1. **Carefully review** the refined proposal and compare it against your original feedback.
2. **Examine the change log** to determine whether the designer has successfully addressed the concerns you raised in your previous review.
3. **Consider any new or remaining concerns** that may have surfaced in the refined proposal.
4. Provide your **review, rating, and suggestions**. Keep in mind:
   - Your evaluation should be based on the **design quality**, not the writing style.
   - Any feedback on writing should be included under **suggestions**, not reflected in the rating.
   - Do not inflate the rating simply because previous concerns were addressed. The rating should reflect the overall merit of the design at this stage.

### Final Note:
Be strict and objective. Approve the proposal only if it meets the necessary standards of quality and innovation. Do not pass a proposal unless it is sufficiently strong.
"""


GUM_PROPOSAL_REREVIEW = AgentPrompt(GUM_PROPOSAL_REREVIEW_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_format)

GUM_PROPOSAL_REREVIEW_ISEARCH_prompt = GUM_PROPOSAL_REREVIEW_prompt+ """
Please review with the help of the search engine.
"""

GUM_PROPOSAL_REREVIEW_ISEARCH = AgentPrompt(GUM_PROPOSAL_REREVIEW_ISEARCH_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_ISEARCH_format)

# endregion







'''
#######################################################
# GUM Proposal Search Assistant Prompt
#######################################################
'''


""" ============================= S2 Search Assistant System ===================================== """

# region S2 Search Assistant System



S2_SEARCH_ASSISTANT_SYSTEM_prompt = """
You are an advanced search assistant designed to help researchers gather relevant information from Semantic Scholar. Your task is to interpret the given ideation and instructions, formulate appropriate search queries, analyze the results, and compile a comprehensive report.

## Input
You will receive:
1. Ideation: The researcher's initial thoughts and concepts.
2. Instructions: Specific guidance on what information to gather.

## Process
Follow these steps iteratively until you have gathered sufficient information:

1. Query Formulation:
   - Analyze the ideation and instructions.
   - Construct a relevant search query for Semantic Scholar.


2. Result Analysis:
   - Review the search results returned by the Semantic Scholar API.
   - Analyze the relevance and quality of each paper based on:
     - Title
     - Authors
     - Venue
     - Year
     - Abstract
     - Citation count
     - Open access availability

3. Iterative Refinement:
   - If the results are not satisfactory, refine your query and repeat steps 1-2.

4. Information Extraction:
   - For relevant papers, extract key information including:
     - Main findings
     - Methodologies
     - Implications for the research topic

5. Continuation Decision:
   - Decide if further searches are needed based on the quality and quantity of gathered information.

## Output
Once you have gathered sufficient information, compile a report with the following structure:

1. Executive Summary:
   - Brief overview of the search process and key findings

2. Relevant Papers:
   - List of most relevant papers with their details:
     - Title
     - Authors
     - Venue
     - Year
     - Brief summary
     - Citation count
     - Link to open access PDF (if available)

3. Key Insights:
   - Synthesis of main findings and their relevance to the original ideation and instructions

4. Methodologies:
   - Overview of notable methodologies found in the literature

5. Research Gaps:
   - Identification of potential gaps or areas for further investigation

6. Recommendations:
   - Suggestions for how the gathered information can be applied to the original research question or idea

Remember to maintain a balance between thoroughness and conciseness in your report. Focus on quality over quantity, ensuring that each included paper and insight is directly relevant to the researcher's needs as expressed in the ideation and instructions.
"""


S2_SEARCH_ASSISTANT_SYSTEM = AgentPrompt(S2_SEARCH_ASSISTANT_SYSTEM_prompt)


# endregion


""" ============================= S2 Search Proposal Query ===================================== """


# region S2 Search Proposal Query


class S2_SEARCH_PROPOSAL_QUERY_format(BaseModel):
   analysis: str = Field(..., description="The analysis of the ideation and instructions.")
   query: str = Field(..., description="The query to search for papers.")

class S2_SEARCH_PROPOSAL_RESPONSE_format(BaseModel):
   report: str = Field(..., description="The report of the search.")
   references: List[str] = Field(..., description="The list of titles of all the reference papers.")
   continue_search: bool = Field(..., description="Whether to continue the search.")

S2_SEARCH_PROPOSAL_QUERY_prompt = """
You are tasked with generating a search query for Semantic Scholar based on the given ideation and instructions. Analyze the input carefully and formulate an effective query.

Here is the ideation including the researcher's initial thoughts and concepts:

{IDEATION}

Here are the instructions from the researcher about specific guidance on what information to gather:

{INSTRUCTIONS}

## Please provide the following information:

1. Analysis:
   - Briefly analyze the ideation and instructions.
   - Identify key concepts, themes, and potential search terms.
   - Consider any specific requirements or constraints mentioned.

2. Query:
   - Formulate a clear and focused search query.
   - Use relevant keywords and phrases from the ideation and instructions.
   - Do not use advanced search operators (e.g., AND, OR, quotation marks for exact phrases).
   - Keep in mind the following Semantic Scholar search tips:
     - No special query syntax is supported.
     - Replace hyphens with spaces to find matches.
     - Be specific to narrow down results.
     - Consider including author names, publication years, or venues if relevant.

Remember, the quality of the search results depends heavily on the query you formulate. Aim for a balance between specificity (to get relevant results) and breadth (to not miss important papers).
"""

S2_SEARCH_PROPOSAL_RESPONSE_prompt = """
You are tasked with analyzing the search results from Semantic Scholar, generating a report, and deciding whether to continue the search. Evaluate the results carefully in relation to the original ideation and instructions.

Here is the search results:

{SEARCH_RESULTS}

## Please provide the following information:

1. Report:
   - Summarize the search results, focusing on the most relevant papers.
   - Highlight key findings, methodologies, and insights related to the original ideation.
   - Identify any gaps in the literature or areas for further investigation.
   - Structure your report as follows:
     a. Overview of search results
     b. Most relevant papers (3-5) with brief summaries
     c. More relevant papers (if any) that are valuable and may help the researcher with brief summaries
     d. Key insights and their relevance to the original ideation
     e. Notable methodologies found
     f. Identified research gaps
     g. Recommendations for applying the findings

2. References
   - List of titles of all the reference papers.

3. Continue Search:
   - Decide whether further searching is necessary based on:
     a. Quality and relevance of the results
     b. Coverage of the topics from the original ideation and instructions
     c. Identification of new areas that require exploration
   - Set to true if more searching is needed, false if sufficient information has been gathered.

In your analysis, prioritize papers that are highly cited, recent, and from reputable venues. Consider the relevance of each paper to the original ideation and instructions. If you decide to continue the search, think about how the query could be refined or expanded to address any gaps in the current results.
"""


S2_SEARCH_PROPOSAL_QUERY = AgentPrompt(S2_SEARCH_PROPOSAL_QUERY_prompt,GENERAL_JSON_parser,S2_SEARCH_PROPOSAL_QUERY_format)
S2_SEARCH_PROPOSAL_RESPONSE = AgentPrompt(S2_SEARCH_PROPOSAL_RESPONSE_prompt,GENERAL_JSON_parser,S2_SEARCH_PROPOSAL_RESPONSE_format)

# endregion




'''
#######################################################
# GUM Implementation System Prompt
#######################################################
'''


""" ============================= GUM Design Implementer System ===================================== """


# region GU Design Implementer System


# About GAB
GUM_DESIGNER_SYSTEM_prompt_part1 = """
You are a researcher designing a new autoregressive language model (LM). Modern
LMs are typically structured as a stack of repeating blocks. Each block accepts: 

1. A sequence of embeddings $X$ of shape $(B, L, D)$, where $B$ is the batch
   size, $L$ is the sequence length, and $D$ is the embedding dimension.
2. Intermediate variables $Z$ (may be passed as keyword arguments), such as
   memory, states, caches, etc.

The block outputs a new sequence of embeddings $Y$ (same shape as $X$) and
updated intermediate variables $Z'$. Such a block can be written as:

```python {GAB_BASE} ```

And a LM can be written as:

```python 
tokens = Tokenizer(sentence)
X = Embeddings(tokens)
Z = {{}} # initialized as an empty dictionary which might be updated by the blocks
for block in Blocks:
   X, Z = block(X, **Z)
output = Logits(X)
```

Your goal is to discover the best novel autoregressive LM block that can defeat
the existing state-of-the-art models, measured in low perplexity in corpora,
high accuracy in downstream tasks, robustness to variant inputs, efficiency in
training and inference, and most importantly, good scalability that providing
better overall performance with more data and larger models.

"""

# About GAU
GUM_DESIGNER_SYSTEM_prompt_part2 = """
## Generalized Autoregressive Units 

To design this block, you break it down into smaller components called
Generalized Autoregressive Units (GAUs), which inherit from the following base
class:

```python {GAU_BASE}```

The key idea is that a language model (LM) block can be decomposed into a series
of nested units. Each unit shares the same interface as the LM block, mapping
the sequence $X$ and intermediate variables $Z$ to a new sequence $Y$ and
updated variables $Z'$, which incorporate the newly computed values or new
variables. These units can be arranged hierarchically, with the outputs of one
unit passing into the next, potentially with intermediate operations in between.
Units can be nested within one another, allowing a block to be represented as a
tree of units with a root node. As such, you can focus on designing one unit at
a time.

For example, a LM block can be represented using a root unit:

```python 
class ExampleRootUnit(GAUBase):
   def __init__(self,...):
      self.norm1 = Norm(...)
      self.unitA = UnitA(...)
      self.norm2 = Norm(...)
      self.unitB = UnitB(...)

   def _forward(self,X,**Z):
      X1,Z=self.norm1(X,**Z) 
      X2,Z=self.unitA(X1,**Z) 
      X=X+X2 
      X3,Z=self.norm2(X,**Z)
      X4,Z=self.unitB(X3,**Z) 
      X=X+X4 
      return X,Z
```

In this example, the block consists of three child units: `Norm`, `UnitA`, and
`UnitB`. Each of these units follows the same interface and structure as the
root unit and may themselves be composed of nested child units.

"""

# About the role  
GUM_DESIGNER_SYSTEM_prompt_part3 = """

### Instructions for the Design Process

You will start by refining an existing language model (LM) block design, structured as a tree of GAUs (Generalized Autoregressive Units). A proposal will be provided, specifying a target GAU for refinement. Your task is to implement changes based on the proposal. You can modify the target GAU's operations or introduce new child GAUs. Remember, the proposal is a high-level guideline—you're encouraged to explore better design variants.

### Key Design Principles:

1. **Decomposition of Complex GAUs**:  
   If a GAU is complex, it is essential to decompose it into smaller child GAUs to make the design and testing process easier. Follow this top-down approach:
   - **Identify complex components**: Decide if a component should be turned into a child GAU.
   - **Perform requirement analysis**: Define the child GAU's name, requirements, inputs, and outputs. Add this information to the `CHILDREN_DECLARATION` list using the `UnitDecl` structure:
     - **Name**: The name of the child GAU.
     - **Requirements**: Functional requirements for the child GAU.
     - **Inputs**: Variables passed to the child GAU through the `Z` dictionary (e.g., `Z['input_name'] = ...`). If `X` is an input, it represents the sequence input (shape: `(B, L, D)`) and is not stored in `Z`.
     - **Outputs**: Variables returned by the child GAU through the `Z` dictionary (e.g., `Z'['output_name'] = ...`). If `Y` is an output, it represents the sequence output and is not stored in `Z`.
   > **Note**: `X` and `Y` are special inputs and outputs for sequences. They are not expected to be passed through `Z`.

2. **Placeholder Declaration and Child GAU Calls**:  
   Declare and instantiate child GAUs in the parent GAU’s `__init__` method as placeholders, like:
   ```python
   self.{{child_instance}} = {{ChildName}}(...)
   ```
   Call the child GAU in the forward pass using this pattern:
   ```python
   Z['arg1'] = ...
   Z['arg2'] = ...

   Y, Z_ = self.{{child_instance}}(X, **Z)

   out1 = Z_.get('out1', None)
   out2 = Z_.get('out2', None)
   ```
   - You can replace `X`, `Y`, `Z`, and `Z_` with other variable names, but ensure the sequences (`X`, `Y`) are always shaped `(B, L, D)`.
   - Ensure all inputs/outputs, other than sequences, are passed via `Z` and `Z_`.

3. **Prepare Inputs and Outputs**:  
   All inputs needed by child GAUs should be prepared in advance. After finalizing the parent GAU, you won’t be able to modify it when implementing the child GAUs. Always retrieve values from `Z` using `Z.get('var', None)` or other default values to avoid errors. Similarly, when implementing a GAU, you should also handle the case if an input argument is not in `Z` or is `None`.

The system will handle placeholders for declared child GAUs by generating empty classes that accept `X` and `Z` as inputs and return the same `X` and `Z` as outputs. Your job is to correctly prepare the inputs and manage outputs for each child GAU.

### Implementation Guidelines:

- **One GAU at a Time**:  
  Each time, you will work on a **single GAU**. Either you’ll be assigned a GAU from the proposal or choose one from the proposal GAU and newly declared child GAUs. So when selecting a GAU to implement, you need to consider the dependency between GAUs as well.
  
- **No Access to Other GAUs**:  
  When working on a GAU, you will only have access to the current GAU’s implementation and not the internal details of other GAUs. Ensure interactions between GAUs are handled through `Z` and `Z_`.

- **Child GAUs**:  
  When decomposing a GAU into child GAUs, ensure that the placeholder instantiation and calls are correct. Though you won’t implement them immediately, placeholders will be provided. Ensure all input/output interfaces for placeholders are properly handled in the current GAU.

- **Docstring**:  
  Provide a **docstring** for the GAU, explaining its inputs, outputs, and purpose. Follow PyTorch’s style guidelines, as the docstring will help others understand the GAU’s role and how it interacts with other units.

- **Unit Tests**:  
  Write at least one **unit test** for each GAU. Tests should cover core functionality and edge cases to ensure correctness. After the GAU is integrated into the model, tests will be run automatically to validate its performance.

- **Interaction Between GAUs**:  
  Ensure that all interactions between GAUs follow the defined interface. You will not be able to modify other GAUs once your current GAU is finalized, so proper input/output management is essential.

- **Focus on One GAU**:  
  Focus on the design of the current GAU without worrying about the internal workings of its parents, siblings, or children. Ensure your design allows the GAU to communicate effectively with its children using their defined interfaces.

- **Iterative Design**:  
  You will receive feedback and go through iterative rounds of design. If your implementation introduces errors or fails tests, you will need to debug and refine your GAU. The system will guide you through this process with error traces and diagnostics.

"""


GUM_DESIGNER_SYSTEM_prompt_part4 = """
## Guidelines for Designing the GAU:

1. **Class Naming & Structure**:
   - Ensure that your GAU class inherits from `GAUBase` and is named as specified in the proposal. You should only define **one** GAU class in your implementation. Do not define any other GAU classes in this block.
   - Ensure all the arguments introduced in the `__init__` function of the GAU class have either a default value or a way to handle missing values. If an argument is optional, handle it gracefully. Missing argument handling is necessary to prevent checker failures unless `None` is a valid value.
   - Ensure you are referring to the right class names in unit tests. 

2. **GAU Call Behavior**:
   - The GAU should always be called in this format:
     ```python
     Y, Z' = self.{{unit_instance}}(X, **Z)
     ```
     If additional inputs are required, pass them through `Z` (e.g., `Z['arg'] = value`). 
     - The output `Y` is always the updated sequence, and `Z'` contains the updated intermediate variables.
     - If extra outputs besides `Y` are expected, retrieve them from `Z'`, e.g.:
     ```python
     var = Z'.get('var', None)
     ```
   
3. **GAU Initialization**:
    - Always initialize a GAU instance as follows:
     ```python self.{{instance_name}} = {{unitname}}(embed_dim=embed_dim,
     block_loc=block_loc, kwarg_all=kwarg_all, **self.factory_kwargs,
     **kwarg_all) ```
   - If you need to pass extra arguments to the unit, include them in `kwarg_all`.
     For example, suppose you introduced two additional arguments, `arg1` and `arg2`, 
     you can pass them as follows:
     ```python
     kwarg_all['arg1']=...
     kwarg_all['arg2']=...
     ... = {{UnitName}}(..., kwarg_all=kwarg_all, ..., **kwarg_all)
     ```

4. **Embedding & Block Location**: - `embed_dim` specifies the input dimension.
   - `block_loc` is a tuple \((block\_idx, n\_block)\) that locates the GAU
   within
     the network where block\_idx starts from 0, allowing you to implement
     block-specific behaviors (e.g., varying architectures or operations between
     blocks, initializing intermediate variables acrossing blocks in the first
     block).

5. **Module Definition**:
    - Avoid using `GAU` instances inside `nn.Sequential`. You can use
      `nn.ModuleList` or `nn.ModuleDict`.
    - Do not define any nn.Module classes in your code. Declare child GAUs instead and do not implement them in your code.

6. **Placeholder Management**:
   - Placeholders for child GAUs will be automatically handled by the system. Avoid manually implementing placeholders at this stage. You will be prompted to implement them later when necessary.
   - When declaring placeholders for child GAUs in your design, follow the proper syntax and ensure correct input-output handling.

7. **Design Approach**:
   - Name GAUs meaningfully. Each GAU should represent a distinct unit with a clear function in the architecture.
   - Follow a top-down design approach: if the operation is complex, decompose it into child GAUs and define their placeholders. Ensure each placeholder aligns with the broader structure of the model, ready for future implementation.

8. **Be Innovative**:
   - Focus on designing GAUs that improve performance and efficiency. Avoid replicating existing architectures (e.g., vanilla Transformers) and aim to transcend current state-of-the-art models.
   - Introduce unique mechanisms or structures that differentiate your GAUs from traditional models.
   - Do not simply copy from the references or existing codebases. You can use their ideas to inspire you for your own original designs.
   
9. **Be Consistent**: 
   - Ensure your design remains consistent and fits seamlessly into the overall system architecture.
   - Avoid introducing errors, inconsistencies, or redundant code. Your GAU should operate smoothly alongside existing GAUs and should not introduce any deviations from the overall design philosophy.

"""


GUM_DESIGNER_SYSTEM_prompt=GUM_DESIGNER_SYSTEM_prompt_part1+GUM_DESIGNER_SYSTEM_prompt_part2+\
   GUM_DESIGNER_SYSTEM_prompt_part3+GUM_DESIGNER_SYSTEM_prompt_part4


GUM_DESIGNER_SYSTEM = AgentPrompt(GUM_DESIGNER_SYSTEM_prompt)





# endregion




'''
#######################################################
# GUM Implementation nodes Prompts
#######################################################
'''



""" ============================= GUM Implementation Unit Selection ===================================== """


# region GUM Implementation Unit Selection


def gen_GUM_IMPLEMENTATION_UNIT_SELECTION(SELECTIONS,post_refining=False):
   GUM_IMPLEMENTATION_UNIT_SELECTION_prompt = """
####  Overall Proposal for Refining the Design:
{PROPOSAL}

#### Review of the Proposal:
{REVIEW}
- **Rating**: {RATING} out of 5 (Passing score: >3)

#### Current Design Overview:
Below is a tree of the GAUs that compose the language model (LM) block and the details of the GAUs:

{VIEW}

#### Log of Progress:
{LOG}

### Instructions:

1. **Select a GAU to Work On**:
   - Choose one GAU from the tree to work on. You can only select a unit that has been proposed for refinement or newly declared during the refinement process.
   - Provide the **class name** of the GAU you are going to work on.

2. **Motivation**:
   - Explain why you selected this specific GAU.
   - Include an overall evaluation of the current design and its progress.

3. **Rough Plan**:
   - Outline a rough plan for implementing the selected GAU, including key aspects you plan to focus on during the implementation.

---

### Key Points:
- You are required to implement all the unimplemented GAUs in the tree. 
- Once all units are implemented, you may choose to **terminate the design process** if you believe the design is complete and there are no further improvements to make.
"""

   SelectionEnum=generate_enum_from_list('selection',SELECTIONS)

   class GUM_IMPLEMENTATION_UNIT_SELECTION_format(BaseModel):
      selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on.")
      motivation: str = Field(..., description="The motivation for the selection.")
      rough_plan: str = Field(..., description="The rough plan for implementing the selected GAU.")
      termination: bool = Field(..., description="Whether to terminate the design process.")

   if post_refining:
      GUM_IMPLEMENTATION_UNIT_SELECTION_prompt+=(
         '\n\nYou have implemented all the unimplemented GAUs, you can choose to terminate the design process if you think the design is complete. '
         'You should continue refining the design only if you have more ideas to improve the design and there must be concrete changes to the design. '
         'So, please also include the reason for you to continue the design process in your motivation. '
         'And in adition, please provide a plan for the changes you will make in your rough plan.'
      )
   return AgentPrompt(GUM_IMPLEMENTATION_UNIT_SELECTION_prompt,GENERAL_JSON_parser,GUM_IMPLEMENTATION_UNIT_SELECTION_format)

# endregion




""" ============================= GUM Implementation Unit ===================================== """


# region GUM Implementation Unit




def gen_GUM_IMPLEMENTATION_UNIT(refine=False,begin=False):

   if refine:
      GUM_IMPLEMENTATION_UNIT_prompt = """
Below is the specification for the GAU you need to refine:

**Specification**: {SPECIFICATION}

**Children list**: {CHILDREN}

**Current Implementation**: {IMPLEMENTATION}

**Review**: {REVIEW}

**Rating**: {RATING} out of 5 (Passing score >3)

**Reviewer Suggestions**: {SUGGESTIONS}

### Refinement Process

If there is a review provided, you should start by reflecting on the feedback.
Otherwise, leave reflection empty. The, proceed with the following:

1. **New Analysis and Design**: - Provide an updated detailed analysis based on
   the feedback, including your new design direction and justifications. -
   Include a high-level pseudocode that captures the core of the new design. 

2. **Implementation**: - Provide the full updated implementation of the GAU,
   following the specified format and templates. Remember to update the
   docstring of the GAU class accordingly. Follow the instruction in the GAU
   template carefully. If the original docstring is missing or incorrect or not 
   following the template, please rewrite the docstring based on the new design.

3. **Children list**: - Provide the list of the children GAUs that are declared
   in the current GAU. You can declare new children GAUs or preserve the
   existing ones. If you do not declare any new children GAUs, you should
   provide the original children GAUs. 

4. **Log of Changes**: - Summarize the key changes you made during the
   refinement process. Including *all* code snippets where you made a change
   wrapped in ```python ```.

### Key Points to Remember: - The bug or issue must always be resolved within
the current GAU, as other units are either fully implemented and tested or
placeholders that do not perform any computation. - Ensure the GAU is
self-contained, so you won't need to adjust it later when working on other
units. - The design must align with the original proposal and follow all
instructions, templates, and format requirements. - Use a top-down approach:
break down complex operations into smaller tasks where necessary and declare
each of them as a child GAU. Do not make a single unit overly complex.

Remember your final goal is to refine the GAU in a way that enhances the overall
design, ensuring both correctness and innovation.
   """
      GUM_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_RETRY_format
      
      if begin:
         GUM_IMPLEMENTATION_UNIT_prompt = """
####  Overall Proposal for Refining the Design:
{PROPOSAL}

#### Review of the Proposal:
{REVIEW}
- **Rating**: {RATING} out of 5 (Passing score: >3)

#### Current Design Overview:
Below is a tree of the GAUs that compose the language model (LM) block and the details of the GAUs:

{VIEW}

""" + GUM_IMPLEMENTATION_UNIT_prompt+" Please also give a new name of this variant of the GAU, but notice that, please do not rename the GAUBase class of the unit in your code."
         GUM_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_REFINE_format
   else:
      GUM_IMPLEMENTATION_UNIT_prompt = """
#### GAU Declaration:
Below is the declaration of the GAU you are tasked with implementing. Please ensure that your design and implementation align with the details provided:

{DECLARATION}

---

### Instructions for Implementation:

1. **Design and Implement the GAU**:
   - Your design should be based on the proposal, following the provided instructions, templates, and format requirements.
   - The implementation will be reviewed and tested. It will only be accepted once it passes both the review and the functionality checks.
   - Write the docstring of the GAU class carefully. Follow the instruction about the format of the docstring in the GAU template carefully.
   
2. **Detailed Analysis**:
   - Your design should include a detailed analysis of how your implementation addresses the declared requirements. 
   - You may introduce new ideas and details not covered in the proposal if they improve the design.

3. **Top-Down Approach**:
   - Focus on designing the current GAU step-by-step rather than trying to implement everything at once. Be patient and thorough.
   - Use a top-down approach: break down complex operations into smaller tasks where necessary and declare each of them as a child GAU. 

4. **Placeholder Definition**:
   - Feel free to define placeholders for more complex child GAUs that can be implemented later. These placeholders will help guide future stages of the design.
   
---

### Final Note:
After completing this GAU, you will be asked to implement any remaining parts of the GAB block. Make sure your GAU is well-structured and self-contained to support the overall model design.
   """
      GUM_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_format


   return AgentPrompt(GUM_IMPLEMENTATION_UNIT_prompt,GENERAL_JSON_parser,GUM_IMPLEMENTATION_UNIT_format)

# endregion




""" ============================= GUM Implementation Unit Refine Prompt ===================================== """

# region GU Implementation Refine


GUM_IMPLEMENTATION_UNIT_REFINE= AgentPrompt(GU_IMPLEMENTATION_RETRY_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REFINE_format)


# endregion




'''
#######################################################
# GUM Implementation Reviewer Prompts
#######################################################
'''





""" ============================= GUM Implementation Reviewer System ===================================== """


# region GUM Implementation Reviewer 


GUM_IMPLEMENTATION_REVIEWER_SYSTEM_prompt = """
You are an expert in autoregressive language model research, and you have been
asked to review the design and implementation of a novel autoregressive language
model (LM) block.

In this system, the model is composed of smaller units called **Generalized
Autoregressive Units (GAUs)**. These GAUs form the building blocks of the LM.
The idea is to break the complex LM block into smaller, manageable units that
are easier to design, refine, and test.

Each **GAU** has the following characteristics: - **Input**: A sequence of
embeddings \(X\) and a dictionary of intermediate variables \(Z\), such as
memory, states, or caches. - **Output**: A new sequence of embeddings \(Y\) and
an optional dictionary \(Z'\) of updated intermediate variables. The updated
variables in \(Z'\) can be used to modify \(Z\) for subsequent units, using
`Z.update(Z')`.

These GAUs are designed to be nested, allowing the system to build increasingly
complex autoregressive language model blocks. The model itself is structured as
a tree of GAUs, starting from a root unit and branching out through its child
GAUs. The system automatically creates placeholders for GAUs that have yet to be
implemented, allowing designers to focus on refining one GAU at a time.

Your task as a reviewer is to examine the **design and implementation** of a
specific GAU that the designer has either created or refined. This GAU will fit
into the larger tree structure of the model, but you are primarily responsible
for evaluating the individual GAU unit.

### Instructions for Reviewing the GAU Design:

1. **Accuracy, Robustness, Efficiency, and Scalability**: - Assess whether the
   design and implementation can lead to accurate, robust, efficient, and
   scalable performance in the language model.

2. **Novelty**: - Ensure the design introduces new ideas and is not just
   replicating existing designs like standard Transformer blocks.

3. **Strengths and Concerns**: - Identify the design's key strengths and explain
   how they may contribute to a successful model. - Highlight any concerns you
   have, such as potential limitations, risks, or weaknesses in the design.

4. **Clarity and Completeness**: - Ensure the design is clear and complete.
   Point out any ambiguous, missing, or incorrect parts in the design or
   implementation. Offer suggestions to address these issues.

5. **Theoretical Soundness**: - Focus on the theoretical foundation of the
   design since empirical results are not expected at this stage. Check whether
   the design aligns with the proposal and whether it seems feasible and
   effective in theory.

6. **Implementation Feasibility**: - Although minor implementation errors (e.g.,
   non-causal behavior or non-differentiability) are not your primary concern,
   make sure the design could realistically be implemented in a causal,
   differentiable, and efficient way.

### Additional Considerations: - **Design Process**: Designers are provided with
a proposal outlining the overall direction of the language model block. They
will refine a GAU or introduce new child GAUs as necessary. It is important to
consider how the GAU you are reviewing fits into this broader design process and
whether it aligns with the proposal's objectives.
  
- **Self-Contained Unit**: The GAU being reviewed should be self-contained,
  meaning that it should not depend on other GAUs for its functionality. This
  ensures that each unit can be independently tested, refined, and debugged.

- **Placeholder Management**: The system may automatically create placeholders
  for future GAUs. While placeholders are part of the design, you should focus
  on reviewing the GAU that has been implemented or refined. The implementation
  of the children GAUs should not be considered in this review. The designer
  should not provide any implementation for the children GAUs.

### Review Process: Your review should include: - A summary of the
**highlights** and **concerns** of the design. - An assessment of the design's
**accuracy**, **robustness**, **efficiency**, and **novelty**. - Suggestions for
**improvement** where necessary.
  
### Rating
- Provide a rating between **0 and 5** based on the design:
   - **1**: Poor design with major issues.
   - **2**: Design is not good enough, requires significant improvement.
   - **3**: Good design that meets the requirements.
   - **4**: Excellent design, well-thought-out and close to approval.
   - **5**: An outstanding and highly innovative design, strongly recommended.

Provide a **rating** based on how well the design meets the criteria above. The
goal is to ensure the GAU is theoretically sound, scalable, novel, and ready for
integration into the broader language model.
"""

GUM_IMPLEMENTATION_REVIEWER_SYSTEM = AgentPrompt(GUM_IMPLEMENTATION_REVIEWER_SYSTEM_prompt)

# endregion




""" ============================= GUM Implementation Unit Refine Review Prompt ===================================== """


# region GUM Implementation Refine Review 

GUM_IMPLEMENTATION_UNIT_REFINE_REVIEW_prompt = """
####  Overall Proposal for Refining the Design:
{PROPOSAL}

#### Review of the Proposal:
{REVIEW}
- **Rating**: {RATING} out of 5 (Passing score: >3)

#### Current Design Overview:
Below is a tree of the GAUs that compose the language model (LM) block and the details of the GAUs:

{VIEW}

---

### GAU Selected for Refinement:

The designer has chosen to refine the GAU named **{UNIT_NAME}**. While the designer must follow the core idea from the proposal, they are allowed to introduce new ideas and details that could improve the design.

#### GAU Description:
{DESCRIPTION}

#### Previous Review:
- **Previous Rating**: {RATING} out of 5 (Passing score: >3)
- **Suggestions from the Previous Reviewer**: {SUGGESTIONS}

#### Design Idea (Analysis):
{ANALYSIS}

#### GAU Specification:
{SPECIFICATION}

#### Full GAU Implementation:
```python
{IMPLEMENTATION}
```

#### Summary of Changes Made:
{CHANGES}


---

### Checker Report:

The checker has evaluated this refined GAU, assessing aspects such as the forward and backward passes, causality, and integration with other previously designed GAUs. The execution trace is provided for your reference:

```bash
{CHECKER_REPORT}
```

---

### Instructions for Review:

1. **Review the Proposal**: Begin by carefully reading the proposal and understanding the design objectives.
   
2. **Evaluate the Design**: 
   - Check the design idea (analysis) to see how well it aligns with the proposal.
   - Examine the implementation and its adherence to the theoretical foundation. 

3. **Review Focus**:
   - **Design**: Base your review and rating on the quality of the **design** itself, not the writing or placeholders. 
   - **Placeholders**: Unimplemented placeholders are allowed and expected to be filled in later. Your review should focus solely on the GAU being refined, not on the placeholders.
   - **Empirical Results**: Since empirical results are not required at this stage, your review should focus on the theoretical soundness of the design.

4. **Suggestions**: If there are any errors or concerns in the design or implementation, point them out in your review and provide constructive suggestions for improvement.

### Final Note:
Be strict, fair, and thorough. Only approve designs that meet a high standard. The designer is working one unit at a time in a top-down manner, so ensure that the unit itself is sound and aligned with the overall model goals.
"""

GUM_IMPLEMENTATION_UNIT_REFINE_REVIEW = AgentPrompt(GUM_IMPLEMENTATION_UNIT_REFINE_REVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion



""" ============================= GUM Implementation Rereview Prompt ===================================== """


# region GUM Implementation Rereview 

GUM_IMPLEMENTATION_REREVIEW_prompt = """The designer has refined the design and implementation of the GAU **{UNIT_NAME}** based on your previous feedback and the results from the checkers. The refinement follows the same proposal, but incorporates changes to address the concerns raised.

---

### Updated Design Details:

- **Updated Design Idea**:
  {ANALYSIS}

- **GAU Specification**:
  {SPECIFICATION}

- **Updated Full Implementation**:
  ```python
  {IMPLEMENTATION}
  ```

- **Summary of Changes**:
  {CHANGES}

---

### Checker Report:

The checker has evaluated this refined GAU, assessing aspects such as the forward and backward passes, causality, and integration with other previously designed GAUs. The execution trace is provided for your reference:

```bash
{CHECKER_REPORT}
```

---

### Instructions for Review:

1. **Review the Proposal**: Begin by revisiting the original proposal to refresh your understanding of the design goals.

2. **Evaluate the Changes**:
   - Carefully examine the updated design idea, implementation, and summary of changes.
   - Consider whether the changes adequately address the concerns you raised in the previous review.

3. **Checker Feedback**:
   - Review the checker report and ensure the GAU functions as expected when integrated with the full language model.

4. **Suggestions**:
   - If there are any remaining issues, concerns, or errors in the design or implementation, provide specific and constructive suggestions in your review.

---

### Final Notes:

- **Focus on the Current Unit**: The designer is working on one unit at a time using a top-down approach. Unimplemented placeholders are allowed and will be filled in later. Your review should concentrate on the GAU being refined, not the placeholders.
- **Theoretical Focus**: Empirical results are not expected at this stage. Base your review on the theoretical soundness of the design.

Be strict and fair in your review. Only approve the design if it meets a high standard and addresses the previous concerns thoroughly.
"""



GUM_IMPLEMENTATION_REREVIEW = AgentPrompt(GUM_IMPLEMENTATION_REREVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion


""" ============================= GUM Implementation Unit Review Prompt ===================================== """


# region GUM Implementation Unit Review 

GUM_IMPLEMENTATION_UNIT_REVIEW_prompt = """
#### Overall Proposal for Refining the Design:
{PROPOSAL}

#### Review of the Proposal:
{REVIEW}
- **Rating**: {RATING} out of 5 (Passing score: >3)

#### Current Design Overview:
Below is a tree of the GAUs that compose the language model (LM) block, along with details about each GAU:

{VIEW}

---

### GAU Specification and Implementation:

- **GAU Specification**:
  {SPECIFICATION}

The designer has implemented the GAU named **{UNIT_NAME}**. Note that the designer is permitted to introduce new ideas and details that go beyond the proposal, provided they improve the design without altering its core concept.

- **Design Idea (Analysis)**:
  {ANALYSIS}

- **Full GAU Implementation**:
  ```python
  {IMPLEMENTATION}
  ```

---

### Checker Report:

The checker has evaluated the GAU's behavior, including aspects such as forward and backward passes, causality, and integration with other previously implemented GAUs. The execution trace is provided for your reference:

```bash
{CHECKER_REPORT}
```

---

### Instructions for Review:

1. **Review the Proposal**: Begin by reviewing the original proposal to understand the objectives behind the design.

2. **Evaluate the Design**:
   - Review the design idea (analysis) and check if the implementation aligns with the theoretical framework.
   - Ensure that any new ideas introduced by the designer enhance the design without straying from the core concept.

3. **Checker Feedback**:
   - Examine the checker report to verify that the GAU functions correctly when integrated into the full LM model.

4. **Suggestions**:
   - Provide constructive feedback, especially if there are any issues or concerns with the design or implementation.

---

### Final Considerations:

- **Focus on the Current Unit**: The designer is working on one unit at a time using a top-down approach. Unimplemented placeholders are acceptable and will be handled later. Your review should focus on the unit being reviewed, not the placeholders.
- **Theoretical Focus**: Empirical results are not expected at this stage, so base your review on the theoretical soundness of the design.

Be strict, fair, and thorough in your evaluation. Only approve the design if it meets a high standard of quality and innovation.
"""

GUM_IMPLEMENTATION_UNIT_REVIEW = AgentPrompt(GUM_IMPLEMENTATION_UNIT_REVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion

