from dataclasses import dataclass, field, asdict
from ..flow.alang import AgentPrompt

from exec_utils.models.model import ModelOutput
import re
import json
from typing import Dict, Any
from pydantic import BaseModel




GAB_BASE='''
class GABBase(nn.Module):
    """ Base class for Generalized Autoregressive Block """
    def __init__(self,embed_dim: int, block_loc: tuple): 
        super().__init__()
        self.embed_dim = embed_dim
        self.block_loc = block_loc # location of the GAB block within the network, (block_idx, n_block), e.g. (0, 6) for the first block in a network with 6 blocks in total

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



class UnitDeclaration(BaseModel):
   unitname: str
   demands: str
   inputs: list[str]
   outputs: list[str]

   def to_prompt(self):
      return f"""
Unit Name: {self.unitname}
- Demands: {self.demands}
- Inputs: {", ".join(self.inputs)}
- Outputs: {", ".join(self.outputs)}
"""

class UnitSpec(BaseModel):
   unitname: str
   document: str
   inputs: list[str]
   outputs: list[str]

   def to_prompt(self):
      return f"""
Unit Name: {self.unitname}
'''
{self.document}  
'''
- Inputs: {", ".join(self.inputs)}
- Outputs: {", ".join(self.outputs)}
"""

class GU_IMPLEMENTATION_ROOT_format(BaseModel): 
   analysis: str
   spec: UnitSpec
   children: list[UnitDeclaration]
   implementation: str

class GU_IMPLEMENTATION_format(BaseModel): 
   analysis: str
   document: str
   children: list[UnitDeclaration]
   implementation: str

class GU_IMPLEMENTATION_ROOT_RETRY_format(BaseModel): # for retry, can update spec 
   reflection: str
   analysis: str
   spec: UnitSpec
   implementation: str
   children: list[UnitDeclaration]
   changes: str

class GU_IMPLEMENTATION_RETRY_format(BaseModel): # for retry, only allow to update document
   reflection: str
   analysis: str
   document: str
   implementation: str
   children: list[UnitDeclaration]
   changes: str

class GU_IMPLEMENTATION_REFINE_format(BaseModel): # for refine, allow to update description, implementation, children, and changes
   newname: str
   reflection: str
   analysis: str
   document: str
   implementation: str
   children: list[UnitDeclaration]
   changes: str



def GENERAL_JSON_parser(raw_output: ModelOutput) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = json.loads(raw_text)  
      output["text"] = raw_text
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.cost
      output["_details"]["running_cost"] = 0
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


GU_DESIGN_PROPOSAL_format = {
   "type": "json_schema",
   "json_schema": {
         "name": "proposal_refinement_response",
         "strict": True,
         "schema": {
            "type": "object",
            "properties": {
               "modelname": {
                     "type": "string",
                     "description": "The name of the model. It should be a camel case legal variable name for defining the model class in pytorch."
               },
               "proposal": {
                     "type": "string",
                     "description": "The fall proposal, keep the format instructions."
               },
            },
            "required": ["modelname", "proposal"],
            "additionalProperties": False
         }
   }
}


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
               "modelname": {
                     "type": "string",
                     "description": "The name of the model. It should be a camel case legal variable name for defining the model class in pytorch."
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
            "required": ["reflection", "modelname", "proposal","changes"],
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
   (i.e. the interfaces of the child), the demands of the child including the
   expected function, behavior and the description of the interfaces. The
   demands should be clear and detailed, it should be a guideline for the
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

FUNCTION_CHECKER_REPORT_PASS = """
Your code passed the functionality checker:

{REPORT}
"""


FUNCTION_CHECKER_REPORT_FAIL= """
Your code failed the functionality checker:

{REPORT}

Here is the composed gab.py file based on your implementation for you to refer:

{GAB_CODE_WITH_LINE_NUM}
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
guarantee that the implementation should pass all checkers and unit tests. You
need to firstly provide the reflection of the feedback including: If you didn't
pass the checker's check, then give an analysis of the bugs, and the plans to
fix them; If you failed on reviewer's review, then the the analysis of the
concerns, and the plans to address them; If you failed on both, then give both.
After relection, you then give the full design including the new analysis,
plans, pseudocode, and the implementations as well, keeping the format
instructions. Finally, give a summary of the changes you made. In your changes,
you need to provide details about your edits on the code with corresponding code
snippets and explanations.

Rememeber to follow the GAU structure, here is the GAUBase class for you to
refresh:

```python {GAU_BASE} ```

Remember that the bug should always be able to be solve within the unit you are
designing, as the other units are either implemented and fully tested or are
placeholders which will have no computation. You should also try to make the
unit self-contained, so that when you are working on another unit, you do not
need to worry about the implementation of this unit.
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

GU_IMPLEMENTATION_UNIT_REFINE_REVIEW = AgentPrompt(GU_IMPLEMENTATION_UNIT_REFINE_REVIEW_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion




""" ============================= GU Implementation Unit Retry Prompt ===================================== """

# region GU Implementation Retry


GU_IMPLEMENTATION_UNIT_RETRY= AgentPrompt(GU_IMPLEMENTATION_RETRY_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_RETRY_format)


# endregion





'''
###################################################################################################
##                                                                                               ##
## GU Design from existing design prompts                                                        ##
##                                                                                               ##
###################################################################################################
'''


# All start with GUE_



def build_GUE_QUERY(seed,references=None,instruct=None):
   query = f"""
Here is the seed design for you to improve:

{seed.to_prompt()}
"""
   if references is not None:
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
   if instruct is not None:
      query += f"Here are some additional instructions that may help you:\n{instruct}\n\n"
   return query


'''
#######################################################
# GUE Design from Exisitng Proposal Prompts
#######################################################
'''



""" ============================= GUE Designer Exisitng Proposal System Prompt ========================================== """



#region GUE System Prompt



GUE_DESIGN_PROPOSER_SYSTEM_prompt = """
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

Instead of starting from scratch, you will start from an existing design and
improve it. You will be provided with the full information of the design,
including the proposal, the tree structure, the implementations of the GAUs. You
are only allowed to modify *one GAU* from the existing design. 

You need to select one GAU to modify, you can define new children GAUs, however
you need to guarantee that your modification wont affect the correctness of the
overall design. 

You will start by writing down an overal proposal for the design you want to
have, the proposal decides a direction, phylosophy and the plan of the design,
and the analysis of the problem and how you gonna solve it by modifying one GAU
from the existing design. You will be provided with one or multiple references
to consider that may inspire you if there are references provided.

Your response should include: 

1. The proposal, it should include but not restrict to the following parts: a. A
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
      the details of each part of the design with the justifications. The
      selection of the GAU to modify and the reasoning of the selection.  

   f. A conclution of the proposal. 

   g. Optional, the references you used in your proposal, should be in the right
      format.
2. The name of the variant of the model you are going to design.
3. The selection of the GAU to modify.

The proposal will be reviewed and you will be asked to modify it if it is not
passed. You can start to implement the design after the proposal is passed. 

The proposal should be as detailed as possible, DO NOT WORRY IF THE PROPOSAL IS
TOO LONG, BUT ALSO DO NOT FILL IN BY REDUNDANT WORDS, USE PRECISE AND CONCRETE
LANGUAGE, the proposal will be the guideline for the entire design process so it
should be clear and detailed. 
"""

GUE_DESIGN_PROPOSER_SYSTEM = AgentPrompt(GUE_DESIGN_PROPOSER_SYSTEM_prompt)


# endregion


""" ============================= Give analysis proposal ===================================== """


# region GUE Give analysis first 



def gen_GUE_DESIGN_PROPOSAL(SELECTIONS):
   GUE_DESIGN_PROPOSAL_prompt = """
   Here is the design query:

   {SEED}

   Check the seed design, then give your proposal and the selection of the GAU to modify follow the instructions.
   """

   GU_DESIGN_PROPOSAL_format = {
      "type": "json_schema",
      "json_schema": {
            "name": "design_proposal_response",
            "strict": True,
            "schema": {
               "type": "object",
               "properties": {
                  "proposal": {
                        "type": "string",
                        "description": "The fall proposal, keep the format instructions."
                  },
                  "selection": {
                        "type": "string",
                        'description': "The name of the GAU you are going to work on.",
                        'enum': SELECTIONS
                  },
                  "modelname": {
                        "type": "string",
                        "description": "The name of the variant of the model you are going to design."
                  },
               },
               "required": ["proposal", "selection", "modelname"],
               "additionalProperties": False
            }
      }
   }

   return AgentPrompt(GUE_DESIGN_PROPOSAL_prompt,GENERAL_JSON_parser,GU_DESIGN_PROPOSAL_format)



# endregion






""" ============================= GUE Proposal Reviewer System ===================================== """


# region GUE Give analysis first 


GUE_PROPOSAL_REVIEWER_SYSTEM_prompt = """
You are a an expert in autoregressive language model research, you are asked to
review the proposal of improving the design of an autoregressive language model
block. 

An autoregressive model block will be represented as nested Generalized
Autoregressive Units (GAUs) defined in the following base class:

```python {GAU_BASE} ```

The designer will select one GAU to modify, and you will review the proposal of
the modification.

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

GUE_PROPOSAL_REVIEWER_SYSTEM = AgentPrompt(GUE_PROPOSAL_REVIEWER_SYSTEM_prompt)

# endregion



""" ============================= GUE Proposal Review ===================================== """


# region GUE Proposal Review 


GUE_PROPOSAL_REVIEW_prompt = """
This is an existing design of the autoregressive language model block that the
designer is going to modify:

{SEED}

The designer choose to modify the following GAU:

{SELECTION}

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

GUE_PROPOSAL_REVIEW_format = {
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

GUE_PROPOSAL_REVIEW = AgentPrompt(GUE_PROPOSAL_REVIEW_prompt,GENERAL_JSON_parser,GUE_PROPOSAL_REVIEW_format)   

# endregion




""" ============================= GUE Proposal Refinement ===================================== """


# region GU Proposal Refinement


def gen_GUE_PROPOSAL_REFINEMENT(SELECTIONS): 

   GUE_PROPOSAL_REFINEMENT_prompt = """
   Your proposal has been reviewed and rated by the expert, here is the feedback:

   {REVIEW}

   Rating: {RATING} out of 5 ({PASS_OR_NOT})

   Suggestions: {SUGGESTIONS}

   Please refine your proposal based on the feedback. You should address the issues
   and improve the design based on the suggestions. You need to firstly provide the
   reflection of the feedback, then give the full proposal keeping the format
   instructions, finally, a summary of the changes you made.
   """

   GUE_PROPOSAL_REFINEMENT_format = {
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
                  "selection": {
                        "type": "string",
                        'description': "The name of the GAU you are going to work on.",
                        'enum': SELECTIONS
                  },
                  "modelname": {
                        "type": "string",
                        "description": "The name of the variant of the model you are going to design."
                  },
                  "changes": {
                        "type": "string",
                        "description": "The summary of the changes you made."
                  },
               },
               "required": ["reflection", "proposal", "selection", "modelname", "changes"],
               "additionalProperties": False
            }
      }
   }


   return AgentPrompt(GUE_PROPOSAL_REFINEMENT_prompt,GENERAL_JSON_parser,GUE_PROPOSAL_REFINEMENT_format)

# endregion



""" ============================= GUE Proposal Rereview ===================================== """


# region GUE Proposal rereview 


GUE_PROPOSAL_REREVIEW_prompt = """
The designer has modified the proposal based on your review, here is the refined
version for you to review:

{PROPOSAL}

The selection of the GAU to modify is:

{SELECTION}

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


GUE_PROPOSAL_REREVIEW = AgentPrompt(GUE_PROPOSAL_REREVIEW_prompt,GENERAL_JSON_parser,GUE_PROPOSAL_REVIEW_format)

# endregion







'''
#######################################################
# GUE Implementation System Prompt
#######################################################
'''


""" ============================= GUE Design Implementer System ===================================== """


# region GU Design Implementer System


# About GAB
GUE_DESIGNER_SYSTEM_prompt_part1 = """
You are a researcher designing a new autoregressive language model (LM). Modern
LMs are typically structured as a stack of repeating blocks. Each block accepts: 

1. A sequence of embeddings $X$ of shape $(B, L, D)$, where $B$ is the batch
   size, $L$ is the sequence length, and $D$ is the embedding dimension.
2. Intermediate variables $Z$ (may be passed as keyword arguments), such as
   memory, states, caches, etc.

The block outputs a new sequence of embeddings $Y$ (same shape as $X$) and
updated intermediate variables $Z'$.

The model architecture can be written as:

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
GUE_DESIGNER_SYSTEM_prompt_part2 = """
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

For example, GPT-2 can be represented using a root unit:

```python 
class GPT2(GAUBase):
   def __init__(self,...):
      self.norm1 = RMSNorm(...)
      self.mha = MHA(...)
      self.norm2 = RMSNorm(...)
      self.mlp = GatedMLP(...)

   def _forward(self,X,**Z):
      X1,Z=self.norm1(X,**Z) 
      X2,Z=self.mha(X1,**Z) 
      X=X+X2 
      X3,Z=self.norm2(X,**Z)
      X4,Z=self.mlp(X3,**Z) 
      X=X+X4 
      return X,Z
```

In this example, the block consists of three child units: `RMSNorm`, `MHA`, and
`GatedMLP`. Each of these units follows the same interface and structure as the
root unit and may themselves be composed of nested child units.

"""

# About the role  
GUE_DESIGNER_SYSTEM_prompt_part3 = """
### Instructions for the Design Process

You will start by refining an existing language model (LM) block design,
structured as a tree of GAUs (Generalized Autoregressive Units). A proposal will
be provided, specifying a target GAU for refinement. Your task is to implement
changes based on the proposal. You can modify the target GAU's operations or
introduce new child GAUs. However, remember that the proposal is a high-level
guideline, you can introduce more variants that make the design better.

When introducing new child GAUs, declare them in your response and call them
within the target GAU. Placeholders, which directly map inputs to outputs, will
be auto-generated by the system and can be implemented later. You are not
required to implement the child units immediately; this can be done during
subsequent stages.

**Key Points**: - You can only modify one GAU at a time. You will have access to
the target GAU and any newly defined child units during the implementation
stage. - If your unit depends on input from its parent, ensure you pick the
parent unit from the beginning, as you will not have access to modify other
units.

You are expected to provide both an **analysis and reasoning** for your design,
as well as the **implementation** of the refined GAU. The implementation must
follow this template:

```python {GAU_TEMPLATE} ```

For each GAU, you must also include at least one unit test. After the unit is
integrated back into the model, it will be checked for **correctness**,
**executability**, and **effectiveness**. Your refinement must not introduce
errors into the model.

**Review Process**: - Your design will be reviewed for approval based on unit
tests, review feedback, and system checks. - If your design fails the review,
you'll need to make revisions based on the feedback. - If your implementation
fails the tests or system checks, you will need to debug and fix any issues. The
system will provide error traces, and you can use print statements to diagnose
problems. Be patient and resolve issues gradually in multiple rounds of dialogs.

"""


GUE_DESIGNER_SYSTEM_prompt_part4 = """
## Guidelines for Designing the GAU:

1. **Class Naming & Structure**: - Ensure it is the only GAU class (inherited
   from `GAUBase`) in your implementation and name it correctly with the name
   you provided. Avoid defining any other GAU classes. - Ensure you are
   referring to the right class names in unit tests. 

2. **GAU Call Behavior**: - Always call the  GAU in this format:
   `Y,Z'={{unitname}}(X, **Z)`. If you want to pass any extra arguments than
   sequence X, pass it through the kwargs $Z$, e.g., Z['arg']=value. The outputs
   are always sequence $Y$ and updated $Z'$. If you expect any extra outputs
   besides sequence $Y$, read it from $Z'$, e.g., var=Z'.get('var',None).

3. **GAU Initialization**: 
   - Always define a GAU instance like this: 
     ```python self.{{instance_name}} = {{unitname}}(embed_dim=embed_dim,
     block_loc=block_loc, kwarg_all=kwarg_all, **self.factory_kwargs,
     **kwarg_all) ```
   - If you want to pass any extra arguments to the unit, pass it through the
     kwarg_all dictionary. For example, you can set kwarg_all['arg']=value before
     passing it to the constructor.

4. **Embedding & Block Location**: - `embed_dim` specifies the input dimension.
   `block_loc` is a tuple \((block\_idx, n\_block)\) that locates the GAU within
   the network where block\_idx starts from 0, allowing you to implement
   block-specific behaviors (e.g., varying architectures or operations between
   blocks, initializing intermediate variables acrossing blocks in the first
   block).

5. **Module Definition**: - Avoid using `GAU` instances inside `nn.Sequential`.
   You can use `nn.ModuleList` or `nn.ModuleDict`.

6. **Placeholder Management**:
    - The system will handle GAU placeholders automatically. Don't manually
      implement placeholders; you will be asked to implement them later.

7. **Design Approach**:
    - Name GAUs meaningfully and define placeholders thoughtfully, outlining the
      design. Develop blocks top-down, with placeholders for complex operations
      to be implemented later. 

8. **Be Innovative**:
    - Design new GAUs to surpass current models. Avoid copying existing
      architectures like the vanilla Transformer—strive for creativity and
      transcendence over existing state-of-the-art models.

"""


GUE_DESIGNER_SYSTEM_prompt=GUE_DESIGNER_SYSTEM_prompt_part1+GUE_DESIGNER_SYSTEM_prompt_part2+GUE_DESIGNER_SYSTEM_prompt_part3+GUE_DESIGNER_SYSTEM_prompt_part4


GUE_DESIGNER_SYSTEM = AgentPrompt(GUE_DESIGNER_SYSTEM_prompt)





# endregion




'''
#######################################################
# GUE Implementation nodes Prompts
#######################################################
'''



""" ============================= GUE Implementation Unit Selection ===================================== """


# region GUE Implementation Unit Selection

GUE_IMPLEMENTATION_UNIT_SELECTION_prompt = """
Here is the proposal for refining the design:

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
of the GAU you are going to work on in your selection. Notice that, you are only
allowed to work on the selected GAU from the proposal and the new declared GAUs
during the refinement.

You will need to implement all the unimplemented GAUs in the tree. When all the
nodes are implemented, you can choose to terminate the design process when you
feel the design is good enough and no more left to improve. 
"""


def gen_GUE_IMPLEMENTATION_UNIT_SELECTION(SELECTIONS):
   GUE_IMPLEMENTATION_UNIT_SELECTION_format = {
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
   return AgentPrompt(GUE_IMPLEMENTATION_UNIT_SELECTION_prompt,GENERAL_JSON_parser,GUE_IMPLEMENTATION_UNIT_SELECTION_format)

# endregion




""" ============================= GUE Implementation Unit ===================================== """


# region GU Implementation Root




def gen_GUE_IMPLEMENTATION_UNIT(refine=False,begin=False):

   if refine:
      GUE_IMPLEMENTATION_UNIT_prompt = """
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

1. **New Analysis and Design**: - Provide an updated analysis based on the
   feedback, including your new design direction and justifications. - Include a
   high-level pseudocode that captures the core of the new design. You should also 
   provide the updated document of the GAU that allows other people to understand 
   the design and implementation without the need to read the code.

2. **Implementation**: - Provide the full updated implementation of the GAU,
   following the specified format and templates. 

3. **Children list**: - Provide the list of the children GAUs that are declared in the 
   current GAU. You can declare new children GAUs or preserve the existing ones. If you do 
   not declare any new children GAUs, you should provide the original children GAUs. 

4. **Log of Changes**: - Summarize the key changes you made during the
   refinement process. Including *all* code snippets where you made a change
   wrapped in ```python ```.

### Key Points to Remember: 
- The bug or issue must always be resolved within the current GAU, as other units are either 
fully implemented and tested or placeholders that do not perform any computation. 
- Ensure the GAU is self-contained, so you won't need to adjust it later when working on other
units. 
- The design must align with the original proposal and follow all instructions, templates, and format requirements.

Remember your final goal is to refine the GAU in a way that enhances the overall
design, ensuring both correctness and innovation.
   """
      GUE_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_RETRY_format
      
      if begin:
         GUE_IMPLEMENTATION_UNIT_prompt = """
This is the proposal for refining the design:

{PROPOSAL}

Here is the review of the proposal for you to refer:

{PREVIEW}

Rating: {PRATING} out of 5 (Passing score is >3)
      
Here is a view of the progress of the implementation, its a tree of the GAUs
that composed the GAB block, the unimplemented GAUs are marked with
(unimplemented), it also shows an overview of the execution paths of the
children GAUs inside each implemented units, along with the rating of the unit:

{VIEW}

This is the full exported code of the current design:

```python {GAB_CODE} ```

""" + GUE_IMPLEMENTATION_UNIT_prompt+" Please also give a new name of the refined GAU, but notice that, you do not need to rename the unit in your code."
         GUE_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_REFINE_format
   else:
      GUE_IMPLEMENTATION_UNIT_prompt = """
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
      GUE_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_format


   return AgentPrompt(GUE_IMPLEMENTATION_UNIT_prompt,GENERAL_JSON_parser,GUE_IMPLEMENTATION_UNIT_format)

# endregion




""" ============================= GUE Implementation Unit Refine Prompt ===================================== """

# region GU Implementation Refine


GUE_IMPLEMENTATION_UNIT_REFINE= AgentPrompt(GU_IMPLEMENTATION_RETRY_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REFINE_format)


# endregion




'''
#######################################################
# GUE Implementation Reviewer Prompts
#######################################################
'''





""" ============================= GUE Implementation Reviewer System ===================================== """


# region GUE Implementation Reviewer 


GUE_IMPLEMENTATION_REVIEWER_SYSTEM_prompt = """
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
  
Provide a **rating** based on how well the design meets the criteria above. The
goal is to ensure the GAU is theoretically sound, scalable, novel, and ready for
integration into the broader language model.
"""

GUE_IMPLEMENTATION_REVIEWER_SYSTEM = AgentPrompt(GUE_IMPLEMENTATION_REVIEWER_SYSTEM_prompt)

# endregion