from dataclasses import dataclass, field, asdict
from ..flow.alang import AgentPrompt

from exec_utils.models.model import ModelOutput
import re
from typing import Dict, Any


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






# -----------------------------------------------------------------------------------------------------------------------




GU_DESIGNER_SYSTEM = """
You are a professional AI researcher focusing on discovering the best
autoregressive language model block. You goal is to design a novel block
following the Generalized Autoregressive Block (GAB) structure. The GAB is
structured in the following base class:

```python {GAB_BASE} ```


This code will be used to construct a gam model in gam.py:

```python {GAM_PY} ```

The language model will be pretrained with the corpus and then be applied for
downstream tasks. The new model is expected to have a low perplexity, high
accuracy, robustness, efficiency, and most importantly, good scalability. You
have two roles 1) to design the model and implement it and; 2) to write a report
and justify your decisions. You do not need to immediately do everything at one
response, following the provided instructions, and finish those tasks step by
step in the coming multi-round dialog. 
"""

GU_DESIGNER_SYSTEM = AgentPrompt(GU_DESIGNER_SYSTEM)

GU_DESIGN_SCRATCH_raw = """
You are now designing an autoregressive model block. The auto-regressive model
is complex, so we will break it down into smaller parts. A block is represented
as multiple nested units which are called the Generalized Autoregressive Block
Unit (GABUnit). Each GABUnit accepts a sequence of embeddings X and a dictionary
of intermediate variables Z as input, and outputs a sequence of embeddings Y and
a dictionary of new or updated intermediate variables Z_. Z_ is optional, when
it is provided, it will be used to update Z by Z.update(Z_). 

A GABUnit is inherited from nn.Module as follows:

```python {GAB_UNIT} ```

You will design a GABUnit by completing the blanks marked in this template:

```python {GU_TEMPLATE} ```

In a GABUnit, you can call other GABUnits, as such, you can create a complicated
GAB block by nesting multiple GABUnits. However, each GABUnit should be not too
complex, if you want to create  complex block, you should break it down into
smaller GABUnits and nest them. As such, you should design a GAB block in a
top-down manner. 

Notice that, everytime you are only allowed to edit within one GABUnit. You can
leave placeholder code of the GABUnits that you wish to implement later. Notice
that, when a GABUnit is detected as fully implemented, it will be tested for
correctness. You will need to ensure the correctness of all the GABUnits in the
final GAB at the end.

Start by filling in the blanks of this root GABUnit:

```python {ROOT_UNIT_TEMPLATE} ```

You should strictly follow the instructions in the template and do not remove
the template code. Your response should include: 

1. The intuitions and analysis of the GABUnit you are designing. You should
   think of how the design can be novel, creative, and powerful. The analysis
   should be detailed and considerable. It should decide a direction of the
   design, the core ideas and the justifications. Remember that you goal is to
   discover the best and novel autoregressive language model block that can
   defeat the existing state of arts.
2. A rough plan of the children GABUnits that may need to be designed in the
   future.
3. The pseudo code of the GABUnit you are designing that capture the high-level
   idea. 
4. The name of the GABUnit you are designing. For the root GABUnit, the name
   will the name of the whole block design, so you definitely think of a
   meaningful name or a creative name that conclude your design, such as
   Transformers, Mamba, CausalConv, SSM, S6, YOLO, etc. Never use a meaningless
   name like RootGAB, NewGAB, etc. When you are trying to provide the name, you
   should wrap the name in the this format ```unit_name {{unit_name}}``` in
   order to allow the system to be able to detect it. 
5. The full implementation of the GABUnit you designed, remember to replece the
   unit_name marks by the actual unit name. Notice that you can contain multiple
   python codes in your response, but only the last one with "#
   GAB_UNIT_IMPLEMENTATION" mark in the first line will be detected as the final
   implementation. If multiple GABUnits are detected in your response, only the
   last one will be applied. When you are trying to give the full implementation
   of the GABUnit you designed, you should always keep this mark at the first
   line, otherwise, the system will not be able to recognize your code.
6. The config of the hyperparameters you introduced in the GABUnit. You should
   provide the config dict in the following format: ```config {{
        # ADD HYPERPARAMETERS HERE #
    }} ``` in your response.

Here are some guidelines for designing the GABUnit:

1. You need to think of a meaningful name of the GABUnit you are designing or
   refering as the placeholder. When you are defining a placeholder, you should
   have an idea of the function of this GABUnit. By defining placeholders, you
   are defining the outline of the design you want to implement.
2. When calling a GABUnit, you should pass both the X and Z to the GABUnit. You
   should pass Z in the kwargs manner, for example {{GABUnit}}(X, **Z).
3. When defining a GABUnit object in __init__, you should privide the type hint,
   for example, ```self.unit: GABUnit = {{unit_name}}(**kwargs) ```, remember to
   pass such a kwargs which allows more customized arguments you will define
   later in your actual implementation to be passed in. When you defines a
   GABUnit, it should be either a known GABUnit or a placeholder of a GABUnit
   you are going to design. It should never be something else such as nn.Module
   or a constant. The system will automatically detect the GABUnit placeholders
   and create a new empty GABUnit or fetch it from the base if it is already
   defined.
4. Be sure to design the block in a top-down manner, be patient and think
   long-run, do never think of designing everything at once. Learn to define
   placeholders that may carry out complicated operations and implement them
   later. Especially when you are working on the root GABUnit. 
5. Be sure to be innovative, do not copy the existing designs such as the
   vanilla Transformer block. Be creative and think of a new design that can
   defeat the existing state of the art models. Try your best to transcend the
   human experts!

Now, give your design, don't be hurry to try to design everything at once, be
patient and focus on the current GABUnit you are designing. You will be asked
later to finish the remaining parts of the GAB block. Remember to design it step
by step, firstly do an analysis which shows your design intention, make sure the
design is novel, then write down the pseudo code before implement it, then think
of the name of your design, write the full code, and provide the configs.
"""


GU_DESIGN_SCRATCH_gpt = """
**Task:** You are designing an autoregressive model block. Due to its
complexity, we will break it into smaller components called **Generalized
Autoregressive Block Units (GABUnits)**. 

Each **GABUnit**: - Accepts a sequence of embeddings `X` and a dictionary of
intermediate variables `Z`. - Outputs a sequence of embeddings `Y` and an
updated dictionary `Z_`. - If `Z_` is provided, it should be merged into `Z` via
`Z.update(Z_)`.

**Structure:** - A **GABUnit** inherits from `nn.Module` as follows:

```python {GAB_UNIT} ```

- To design a **GABUnit**, complete the blanks in this template:

```python {GU_TEMPLATE} ```

- **GABUnits** can call other **GABUnits**, enabling the creation of complex
  blocks by nesting. However, each unit should remain manageable. If the design
  is complex, decompose it into smaller **GABUnits** and nest them. This
  encourages a **top-down** design approach, refining one unit at a time.

**Rules:** 1. You are allowed to modify only one **GABUnit** at a time. 2.
Placeholder code is acceptable for **GABUnits** you plan to implement later. 3.
Once a **GABUnit** is designed, it will be tested for correctness. The
placeholder units will be initialized as an identity mapping. Ensure that the
entire GAB block is correct by the end.

**Steps to Follow:** 1. **Analysis:** Explain the intuition and analysis behind
the **GABUnit** you are designing. Consider how it can be novel, powerful, and
capable of surpassing existing state-of-the-art models. Provide a detailed
justification for your design choices. 2. **Plan:** Outline the potential child
**GABUnits** you may need to design in future iterations. 3. **Pseudocode:**
Present a high-level pseudocode representation of the **GABUnit** you are
designing. 4. **Naming:** Choose a meaningful and creative name for the
**GABUnit**. For the root **GABUnit**, the name should represent the entire
block (e.g., Transformers, Mamba, CausalConv). Wrap the name in this format:
```unit_name {{unit_name}}```. 5. **Implementation:** Provide the full
implementation of the **GABUnit** you designed. Ensure that the implementation
begins with this marker: `# GAB_UNIT_IMPLEMENTATION` for system recognition. 6.
**Config:** List any hyperparameters you introduced in a dict with the following
format:

```config {{
    # ADD HYPERPARAMETERS HERE #
}}
```

**Guidelines:** 1. **Naming and Placeholders:** Choose meaningful names for
**GABUnits** and placeholders. Placeholders should outline the functions of the
units you plan to implement later. 2. **Unit Calls:** When calling a
**GABUnit**, pass both `X` and `Z` (e.g., `GABUnit(X, **Z)`). 3.
**Initialization:** When defining a **GABUnit** in `__init__`, provide type
hints (e.g., `self.unit: GABUnit = {{unit_name}}(**kwargs)`), and allow custom
arguments through `kwargs`. 4. **Top-Down Design:** Be patient and design
incrementally. Define placeholders for complicated operations and implement them
later. Especially for the root **GABUnit**, focus on the big picture first. 5.
**Be Innovative:** Avoid copying existing designs, such as the vanilla
Transformer block. Focus on originality and aim to surpass current
state-of-the-art models.

Now, begin your design. Don't rush to complete everything at once. Start by
analyzing your design intentions, draft the pseudocode, name your unit
creatively, implement the code, and finally, define the configs.
"""





GU_DESIGN_SCRATCH_json = """
You are now designing an autoregressive model block. The auto-regressive model
is complex, so we will break it down into smaller parts. A block is represented
as multiple nested units which are called the Generalized Autoregressive Block
Unit (GABUnit). Each GABUnit accepts a sequence of embeddings X and a dictionary
of intermediate variables Z as input, and outputs a sequence of embeddings Y and
a dictionary of new or updated intermediate variables Z_. Z_ is optional, when
it is provided, it will be used to update Z by Z.update(Z_). 

A GABUnit is inherited from nn.Module as follows:

```python {GAB_UNIT} ```

You will design a GABUnit by completing the blanks marked in this template:

```python {GU_TEMPLATE} ```

In a GABUnit, you can call other GABUnits, as such, you can create a complicated
GAB block by nesting multiple GABUnits. However, each GABUnit should be not too
complex, if you want to create  complex block, you should break it down into
smaller GABUnits and nest them. As such, you should design a GAB block in a
top-down manner. 

Notice that, everytime you are only allowed to edit within one GABUnit. You can
leave placeholder code of the GABUnits that you wish to implement later. Notice
that, when a GABUnit is detected as fully implemented, it will be tested for
correctness. You will need to ensure the correctness of all the GABUnits in the
final GAB at the end.

Start by filling in the blanks of this root GABUnit:

```python {ROOT_UNIT_TEMPLATE} ```

You should strictly follow the instructions in the template and do not remove
the template code. Your response should include: 

1. The intuitions and analysis of the GABUnit you are designing. You should
   think of how the design can be novel, creative, and powerful. The analysis
   should be detailed and considerable. It should decide a direction of the
   design, the core ideas and the justifications. Remember that you goal is to
   discover the best and novel autoregressive language model block that can
   defeat the existing state of arts.
2. A rough plan of the children GABUnits that may need to be designed in the
   future.
3. The pseudo code of the GABUnit you are designing that capture the high-level
   idea. 
4. The name of the GABUnit you are designing. For the root GABUnit, the name
   will the name of the whole block design, so you definitely think of a
   meaningful name or a creative name that conclude your design, such as
   Transformers, Mamba, CausalConv, SSM, S6, YOLO, etc. Never use a meaningless
   name like RootGAB, NewGAB, etc. 
5. The full implementation of the GABUnit you designed, remember to replece the
   unit_name marks by the actual unit name. Notice that you can contain multiple
   python codes in your response, but only the last one with "#
   GAB_UNIT_IMPLEMENTATION" mark in the first line will be detected as the final
   implementation. If multiple GABUnits are detected in your response, only the
   last one will be applied. When you are trying to give the full implementation
   of the GABUnit you designed, you should always keep this mark at the first
   line, otherwise, the system will not be able to recognize your code.
6. The config of the hyperparameters you introduced in the GABUnit, it should be
   a python dict, the keys are the hyperparams you introduced and the values are
   the corresponding default values.

Here are some guidelines for designing the GABUnit:

1. You need to think of a meaningful name of the GABUnit you are designing or
   refering as the placeholder. When you are defining a placeholder, you should
   have an idea of the function of this GABUnit. By defining placeholders, you
   are defining the outline of the design you want to implement.
2. When calling a GABUnit, you should pass both the X and Z to the GABUnit. You
   should pass Z in the kwargs manner, for example {{GABUnit}}(X, **Z).
3. When defining a GABUnit object in __init__, you should privide the type hint,
   for example, ```self.unit: GABUnit = {{unit_name}}(**kwargs) ```, remember to
   pass such a kwargs which allows more customized arguments you will define
   later in your actual implementation to be passed in. When you defines a
   GABUnit, it should be either a known GABUnit or a placeholder of a GABUnit
   you are going to design. It should never be something else such as nn.Module
   or a constant. The system will automatically detect the GABUnit placeholders
   and create a new empty GABUnit or fetch it from the base if it is already
   defined.
4. Be sure to design the block in a top-down manner, be patient and think
   long-run, do never think of designing everything at once. Learn to define
   placeholders that may carry out complicated operations and implement them
   later. Especially when you are working on the root GABUnit. 
5. Be sure to be innovative, do not copy the existing designs such as the
   vanilla Transformer block. Be creative and think of a new design that can
   defeat the existing state of the art models. Try your best to transcend the
   human experts!

Now, give your design, don't be hurry to try to design everything at once, be
patient and focus on the current GABUnit you are designing. You will be asked
later to finish the remaining parts of the GAB block. Remember to design it step
by step, firstly do an analysis which shows your design intention, make sure the
design is novel, then write down the pseudo code before implement it, then think
of the name of your design, write the full code, and provide the configs.
"""

# https://platform.openai.com/docs/guides/structured-outputs/supported-schemas Recursion supported, but maybe not use, it cannot isolate errors

GU_DESIGN_SCRATCH_format = {
  "type": "json_schema",
  "json_schema": {
    "name": "gab_unit_design",
    "description": "Design a Generalized Autoregressive Block Unit (GABUnit) for an autoregressive model block",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "analysis": {
          "type": "string",
          "description": "Intuitions and analysis behind the design of the GABUnit"
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
          "description": "High-level pseudocode of the GABUnit being designed"
        },
        "future_plan": {
          "type": "string",
          "description": "Plan for future GABUnits that may need to be implemented"
        },
        "unit_name": {
          "type": "string",
          "description": "The name of the designed GABUnit"
        },
        "implementation": {
          "type": "string",
          "description": "Full Python code implementation of the designed GABUnit"
        },
        "config": {
          "type": "string",
          "description": "Configuration dictionary containing hyperparameters used in the GABUnit"
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

DESIGN_FROM_SCRATCH = AgentPrompt(GU_DESIGN_SCRATCH_raw,GU_DESIGN_SCRATCH_parser)#,format=GU_DESIGN_SCRATCH_format)



