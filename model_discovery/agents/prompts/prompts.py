
from ..flow.alang import AgentPrompt
from model_discovery.model.utils.modules import UnitSpec,DesignModes

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


class GU_IMPLEMENTATION_format(BaseModel): 
   analysis: str = Field(..., description="The analysis of how to best design and implement the GAU.")
   implementation: str = Field(..., description="The full python implementation of the GAU following the GAU format instructions.")


class DebuggingStep(BaseModel):
    diagnosis: str = Field(..., description="The diagnosis of the cause of the error.")
    suggested_action: str = Field(..., description="The suggested action to fix the error. Must be a concrete code about what to modify or print statements that help locate the error.")

class GU_IMPLEMENTATION_RETRY_DEBUG_format(BaseModel): # for retry
   reflection: str = Field(..., description="The reflection of the feedback from the reviewer.")
   debugging_steps: str = Field(..., description="The debugging steps to fix the error.")
   analysis: str = Field(..., description="The analysis of how to best design and implement the GAU.")
   implementation: str = Field(..., description="The full python implementation of the GAU following the GAU format instructions.")
   changes: str = Field(..., description="The exact changes you have made in the code. It must include detailed code diffs and necessary context with explanation.")

class GU_IMPLEMENTATION_REFINE_format(BaseModel): # for refine, allow to update description, implementation, children, and changes
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

def GENERAL_CODE_parser(raw_output: ModelOutputPlus) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = {}
      codes = re.findall(r"```python(.*?)```", raw_text, re.DOTALL)
      if not codes:
         codes = ['No code is generated, please try again.']
      output["text"] = raw_text
      output["code"] = codes
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.usage
      output["_details"]["running_cost"] = raw_output.usage['cost']
      return output


class GU_IMPLEMENTATION_REVIEW_format(BaseModel):
   review: str = Field(..., description="The review of the proposal, keep the format instructions.")
   rating: float = Field(..., description="A float number between 0 and 5.")
   suggestions: str = Field(..., description="The suggestions for clarification, correction, or additional information.")


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



def gen_GU_IMPLEMENTATION_UNIT_RETRY(use_o1=False):
   GU_IMPLEMENTATION_RETRY_prompt = """
Your design has undergone checks by the format checker, functionality checker, and has been reviewed by the observer. Unfortunately, it did not pass. Below is the feedback:

- **Format Checker**: This report assesses whether your code adheres to the required format guidelines.
  
  **Format Checker Report**:
  {FORMAT_CHECKER_REPORT}

- **Functionality Checker**: The functionality checker evaluates two critical aspects:
  1. **Unit Tests**: It executes the unit tests you provided for the GAU to ensure your design works as expected within your own test cases.
  2. **Whole Model Integration**: Beyond testing the GAU in isolation, the functionality checker integrates your GAU into the larger language model (LM). It compose the tree of GAUs as the LM block. It generates any necessary placeholder classes for unimplemented units and verifies the functionality of the entire LM, including forward pass, backward pass, and causality.

  **Functionality Checker Report**:
  {FUNCTION_CHECKER_REPORT}

- **Observer Review**: 
  **Review**: {REVIEW}
  **Rating**: {RATING} out of 5 ({PASS_OR_NOT})

- **Suggestions from the Observer**:
  {SUGGESTIONS}
"""
   if not use_o1:
      GU_IMPLEMENTATION_RETRY_prompt += """
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
   - **Ensure the unit is self-contained**: Your unit should function independently, without requiring future modifications to other units. This modularity is essential for scalability and ease of integration. And each unit should be able to be initilized by the system automatically without any external interventions.

5. **No access to other units**: 
   - Remember that you can only solve the bugs within the unit you are working on, you cannot access other units including the child units. You can update them later, all your updates to the child unit classes in your code will be ignored and removed. Only the edits to the current unit will be kept. Think of how to solve the problems or delay the problems by editing within current unit.
"""
# Here is the GAUBase class for you to refer:

# {GAU_BASE}

# Please ensure your final submission is thoroughly tested and ready for the next round of review.
# """

   if use_o1:
      GU_IMPLEMENTATION_RETRY_prompt += """
Please try to fix the code based on the information provided. Do not include anything else besides the implementation(s) of the unit(s) in your final response.
Do not worry about the number of tokens in your reasoning, you can use as many as you need to give the best response.
"""
      return AgentPrompt(GU_IMPLEMENTATION_RETRY_prompt,GENERAL_CODE_parser)
   else:
      return AgentPrompt(GU_IMPLEMENTATION_RETRY_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_RETRY_DEBUG_format)


# endregion





'''
###################################################################################################
##                                                                                               ##
## GU Mutation & Crossover
##                                                                                               ##
###################################################################################################
'''


# All start with GUM_



def build_GU_QUERY(seeds,refs=None,instruct=None,user_input=None,mode=DesignModes.MUTATION):
   if mode==DesignModes.MUTATION:
      query = f"""
# Seed Design

You are tasked with improving the following seed design:

---

{seeds[0].to_prompt()}

---
"""
   elif mode==DesignModes.CROSSOVER:
      seeds_prompt='\n\n---\n\n'.join([seed.to_prompt() for seed in seeds])
      query = f"""
# Seed Design

You are tasked to produce a new design by combining the following parent designs:

---

{seeds_prompt}

---
"""
   elif mode==DesignModes.SCRATCH:
      query = f"""
You are tasked to produce a new design from scratch.
"""
   if refs is not None:
      query += '''
# References

Below are some references that may inspire your design. These references come from various libraries, each offering different types of resources:

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

1. **Input**: A sequence of embeddings X of shape (B, L, D), where:
   - B is the batch size.
   - L is the sequence length.
   - D is the embedding dimension.
2. **Intermediate Variables**: Z (e.g., memory, states, caches) passed as keyword arguments.

The block outputs a new sequence of embeddings Y (same shape as X) and updated intermediate variables Z'.

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
- **Input**: A sequence of embeddings X and intermediate variables Z.
- **Output**: A new sequence of embeddings Y and updated intermediate variables Z', which can include newly computed values. 

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

1. **Input**: A sequence of embeddings X of shape (B, L, D), where:
   - B is the batch size.
   - L is the sequence length.
   - D is the embedding dimension.
2. **Intermediate Variables**: Z (e.g., memory, states, caches) passed as keyword arguments.

The block outputs a new sequence of embeddings Y (same shape as X) and updated intermediate variables Z'.

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
- **Input**: A sequence of embeddings X and intermediate variables Z.
- **Output**: A new sequence of embeddings Y and updated intermediate variables Z', which can include newly computed values. 

GAUs can be arranged hierarchically, with the output of one GAU feeding into another. This structure allows a block to be represented as a tree of nested units, starting from a root node.

## Search Capability

You have access to a powerful search assistant that can query both external academic sources (such as arXiv, Papers with Code, and Semantic Scholar) and an internal library of research papers and technical documents. This search assistant will collect information from the internet based on your queries and provide a detailed analysis of the results. This tool allows you to gather relevant information to support and enhance your proposal development process.

## Progressive Proposal Process

Your task of creating a novel LM block design will follow an iterative process of research, ideation, and refinement. This process consists of two main phases:

### Phase 1: Multiple Search and Refinement Rounds

In this phase, you will conduct multiple rounds of search and refinement without external review. Each round consists of:

1. **Search**: Utilize the search assistant to gather relevant information. Formulate specific queries to investigate aspects of your proposal or explore new ideas in the field of language model architecture. For searching external sources, use keywords, but limit to no more than 3 keywords at a time to avoid potential failure. If you want to search more topics, do so in multiple rounds.

2. **Analysis**: Carefully analyze the search results and detailed analysis provided by the search assistant. Extract key insights that can inform your proposal.

3. **Refinement**: Based on your analysis, refine and improve your proposal. This may involve modifying existing elements, adding new components, or adjusting your approach if the research suggests a more promising direction.

4. **Self-Assessment**: Evaluate your refined proposal against the original objectives and requirements. Identify areas that still need improvement or further research.

5. **Iteration Decision**: Determine if another round of search and refinement is necessary. If so, return to step 1 with new, focused queries based on your self-assessment.

Repeat this cycle as many times as needed until you feel your proposal is ready for review.

Note: Throughout this process, ensure that your proposals are supported by mathematical, theoretical, or logical justifications. Each design decision should be backed by sound reasoning and, where applicable, mathematical formulations.

### Phase 2: Review and Major Refinement

Once you believe your proposal is sufficiently developed:

1. **Submission for Review**: Present your proposal for external review.

2. **Feedback Analysis**: Carefully consider the feedback received.

3. **Major Refinement**: If the proposal doesn't pass the review, use the feedback to guide a major refinement of your design. This may involve returning to Phase 1 for additional rounds of search and refinement.

4. **Resubmission**: After major refinement, resubmit your proposal for another review.

Repeat Phase 2 until your proposal passes the review.

## Guidelines for Using the Search Assistant

When using the search assistant, follow these guidelines:

1. **Query Formulation**: 
   - Construct clear, specific queries related to your current design challenges or areas of uncertainty.
   - Use a combination of technical terms and concepts to narrow down results.
   - Consider searching for both recent innovations and foundational papers in relevant areas.

2. **Search Categories**:
   - Use broad searches for external sources to explore cutting-edge research and diverse approaches.
   - Utilize detailed searches of the internal library for in-depth technical information and established methodologies.

3. **Result Integration**:
   - Critically evaluate the search results and analysis provided by the search assistant for relevance and potential impact on your design.
   - Clearly cite and reference any papers or sources that significantly influence your proposal.

4. **Iterative Refinement**:
   - Use the insights from each search to inform subsequent queries, allowing for a more focused and in-depth exploration of relevant topics.

## Proposal Structure

Maintain and update the following structure in your proposal throughout the process:

1. **Title**: A concise, descriptive name for your proposed design.
2. **Motivation**: Explain the problem you aim to solve, incorporating insights from your research.
3. **Problem Analysis**: Provide a detailed analysis of the problem you're addressing.
4. **Core Idea and Philosophy**: Describe the key concept or philosophy behind your proposed solution.
5. **Design Plan**: 
   - Outline your approach for the LM block design.
   - Specify the single GAU you've chosen to modify (excluding the root unit).
   - Provide detailed descriptions of modifications and new structures.
   - Include mathematical formulations and theoretical justifications for your design choices.
6. **Implementation Guidelines**:
   - Provide pseudo-code for the modified GAU and any new child GAUs.
   - Include mathematical formulas necessary for implementation.
   - Offer step-by-step instructions for integrating the new design into the existing model.
7. **Research Summary**: 
   - List key search queries used across all rounds.
   - Summarize the most relevant findings from your searches, including insights from the search assistant's analysis.
   - Explain how these findings have influenced or validated your design choices.
8. **Evolution of Design**:
   - Track major changes and improvements made across refinement rounds.
   - Discuss how these changes address challenges or leverage new insights.
9. **Theoretical Analysis**:
   - Provide mathematical or logical arguments for why your design is expected to improve model performance.
   - Discuss potential trade-offs and how they are addressed.
10. **Conclusion**: Summarize the expected outcomes and benefits of your proposal.
11. **References**: List all sources used in the proposal, properly formatted.

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
- **Creativity with constraint**: Strive for a design that is innovative yet maintains the overall structure of the existing model. Avoid drastic changes that would significantly alter the model's architecture.
- **Local modifications**: Focus on making changes to a single GAU (excluding the root unit) and its potential child GAUs. Ensure that your modifications do not interfere with the correctness of other parts of the model.
- **Simplicity and implementability**: Prioritize designs that are relatively simple and feasible to implement. Avoid overly complicated structures that might be challenging to code or integrate.
- **Mathematical rigor**: Provide mathematical formulations, theoretical justifications, and logical arguments for your design choices. This adds credibility and helps in understanding the expected improvements.
- **Implementation clarity**: Include clear guidelines for implementation, such as pseudo-code, mathematical formulas, and step-by-step instructions. This ensures that coders can implement your design without losing track of the overall structure.
- **Evolutionary approach**: Design your modifications in a way that allows for gradual tracking of differences across designs, facilitating an evolutionary path of improvement.

Remember, the goal of this process is to develop a well-researched, innovative, yet implementable proposal for an LM block design. Take full advantage of the multiple refinement rounds to create a robust, thoroughly considered design before submitting for review. You are allowed to modify one GAU from the existing design (excluding the root unit), and you can introduce new child GAUs to the selected GAU if necessary, but your modifications must maintain the correctness and overall structure of the model design.

You are now ready to begin the progressive proposal process. Start with your initial proposal and prepare for the first round of research and refinement.
"""


GUM_DESIGN_PROPOSER_SYSTEM = AgentPrompt(GUM_DESIGN_PROPOSER_SYSTEM_prompt)
GUM_DESIGN_PROPOSER_SYSTEM_2STAGE = AgentPrompt(GUM_DESIGN_PROPOSER_SYSTEM_2STAGE_prompt)
GUM_DESIGN_PROPOSER_SYSTEM_ISEARCH = AgentPrompt(GUM_DESIGN_PROPOSER_SYSTEM_ISEARCH_prompt)



GUC_DESIGN_PROPOSER_SYSTEM_ISEARCH_prompt = """
You are a researcher tasked with proposing a novel autoregressive language model (LM) block design. This process incorporates an iterative search and refinement workflow to enhance the quality and depth of your proposals.

## Background

Modern LMs are typically structured as a stack of repeating blocks. Each block processes:

1. **Input**: A sequence of embeddings X of shape (B, L, D), where:
   - B is the batch size.
   - L is the sequence length.
   - D is the embedding dimension.
2. **Intermediate Variables**: Z (e.g., memory, states, caches) passed as keyword arguments.

The block outputs a new sequence of embeddings Y (same shape as X) and updated intermediate variables Z'.

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
- **Input**: A sequence of embeddings X and intermediate variables Z.
- **Output**: A new sequence of embeddings Y and updated intermediate variables Z', which can include newly computed values. 

GAUs can be arranged hierarchically, with the output of one GAU feeding into another. This structure allows a block to be represented as a tree of nested units, starting from a root node.

## Search Capability

You have access to a powerful search assistant that can query both external academic sources (such as arXiv, Papers with Code, and Semantic Scholar) and an internal library of research papers and technical documents. This search assistant will collect information from the internet based on your queries and provide a detailed analysis of the results. This tool allows you to gather relevant information to support and enhance your proposal development process.

## Progressive Proposal Process

Your task is to propose a new GAU design by combining multiple parent GAU designs, you will need to reuse the good GAUs from the parents to produce a better design than both. Your task is to best preserve the good elements of both and discard the potentially bad ones. You are not encouraged to introduce brand-new units but to reuse them from the parents.

Your will follow an iterative process of research, ideation, and refinement. This process consists of two main phases:

### Phase 1: Multiple Search and Refinement Rounds

In this phase, you will conduct multiple rounds of search and refinement without external review.
You need to think of analyze the advantage and disadvantage of each parent, and the best way to combine the parents to get a better design. Each round consists of:

1. **Search**: Utilize the search assistant to gather relevant information. Formulate specific queries to investigate aspects that can help you recombine and improve the parents. For searching external sources, use keywords, but limit to no more than 3 keywords at a time to avoid potential failure. If you want to search more topics, do so in multiple rounds.

2. **Analysis**: Carefully analyze the search results and detailed analysis provided by the search assistant. Extract key insights that can inform your proposal.

3. **Refinement**: Based on your analysis, refine and improve your proposal. This may involve modifying existing elements, adding new components, or adjusting your approach if the research suggests a more promising direction.

4. **Self-Assessment**: Evaluate your refined proposal against the original objectives and requirements. Identify areas that still need improvement or further research.

5. **Iteration Decision**: Determine if another round of search and refinement is necessary. If so, return to step 1 with new, focused queries based on your self-assessment.

Repeat this cycle as many times as needed until you feel your proposal is ready for review.

Note: Throughout this process, ensure that your proposals are supported by mathematical, theoretical, or logical justifications. Each design decision should be backed by sound reasoning and, where applicable, mathematical formulations.

### Phase 2: Review and Major Refinement

Once you believe your proposal is sufficiently developed:

1. **Submission for Review**: Present your proposal for external review.

2. **Feedback Analysis**: Carefully consider the feedback received.

3. **Major Refinement**: If the proposal doesn't pass the review, use the feedback to guide a major refinement of your design. This may involve returning to Phase 1 for additional rounds of search and refinement.

4. **Resubmission**: After major refinement, resubmit your proposal for another review.

Repeat Phase 2 until your proposal passes the review.

## Guidelines for Using the Search Assistant

When using the search assistant, follow these guidelines:

1. **Query Formulation**: 
   - Construct clear, specific queries related to your current design challenges or areas of uncertainty.
   - Use a combination of technical terms and concepts to narrow down results.
   - Consider searching for both recent innovations and foundational papers in relevant areas.

2. **Search Categories**:
   - Use broad searches for external sources to explore cutting-edge research and diverse approaches.
   - Utilize detailed searches of the internal library for in-depth technical information and established methodologies.

3. **Result Integration**:
   - Critically evaluate the search results and analysis provided by the search assistant for relevance and potential impact on your design.
   - Clearly cite and reference any papers or sources that significantly influence your proposal.

4. **Iterative Refinement**:
   - Use the insights from each search to inform subsequent queries, allowing for a more focused and in-depth exploration of relevant topics.

## Proposal Structure

Maintain and update the following structure in your proposal throughout the process:

1. **Title**: A concise, descriptive name for your proposed design.
2. **Motivation**: Explain the idea of how to recombine the parents to get a better design.
3. **Problem Analysis**: Provide a detailed analysis of the recombination plan, how can they complement each other, preserve the good parts, and discard the bad ones from the parents.
4. **Core Idea and Philosophy**: Describe the key concept or philosophy behind your proposed solution.
5. **Design Plan**: 
   - Outline your approach for the recombination of the parents.
   - Provide detailed descriptions of modifications and new structures.
   - Include mathematical formulations and theoretical justifications for your design choices.
6. **Implementation Guidelines**:
   - Provide pseudo-code for the recombination of the parents.
   - Include mathematical formulas necessary for implementation.
   - Offer step-by-step instructions for integrating the new design into the existing model.
7. **Research Summary**: 
   - List key search queries used across all rounds.
   - Summarize the most relevant findings from your searches, including insights from the search assistant's analysis.
   - Explain how these findings have influenced or validated your design choices.
8. **Evolution of Design**:
   - Track major changes and improvements made across refinement rounds.
   - Discuss how these changes address challenges or leverage new insights.
9. **Theoretical Analysis**:
   - Provide mathematical or logical arguments for why your design is expected to improve model performance.
   - Discuss potential trade-offs and how they are addressed.
10. **Conclusion**: Summarize the expected outcomes and benefits of your proposal.
11. **References**: List all sources used in the proposal, properly formatted.

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
- **Top-down approach**: Design the recombination from the top down, breaking complex recombination plans into smaller, manageable units that can be nested together.
- **Reusing from parents**: Avoid introducing brand-new units but to reuse them from the parents.

Remember, the goal of this process is to develop a well-researched, innovative, yet implementable proposal for the recombination of parents. Take full advantage of the multiple refinement rounds to create a robust, thoroughly considered design before submitting it for review.

You are now ready to begin the progressive proposal process. Start with your initial proposal and prepare for the first round of research and refinement.
"""

GUC_DESIGN_PROPOSER_SYSTEM_ISEARCH = AgentPrompt(GUC_DESIGN_PROPOSER_SYSTEM_ISEARCH_prompt)



GUS_DESIGN_PROPOSER_SYSTEM_ISEARCH_prompt = """
You are a researcher tasked with proposing a novel autoregressive language model (LM) block design. This process incorporates an iterative search and refinement workflow to enhance the quality and depth of your proposals.

## Background

Modern LMs are typically structured as a stack of repeating blocks. Each block processes:

1. **Input**: A sequence of embeddings X of shape (B, L, D), where:
   - B is the batch size.
   - L is the sequence length.
   - D is the embedding dimension.
2. **Intermediate Variables**: Z (e.g., memory, states, caches) passed as keyword arguments.

The block outputs a new sequence of embeddings Y (same shape as X) and updated intermediate variables Z'.

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
- **Input**: A sequence of embeddings X and intermediate variables Z.
- **Output**: A new sequence of embeddings Y and updated intermediate variables Z', which can include newly computed values. 

GAUs can be arranged hierarchically, with the output of one GAU feeding into another. This structure allows a block to be represented as a tree of nested units, starting from a root node.

## Search Capability

You have access to a powerful search assistant that can query both external academic sources (such as arXiv, Papers with Code, and Semantic Scholar) and an internal library of research papers and technical documents. This search assistant will collect information from the internet based on your queries and provide a detailed analysis of the results. This tool allows you to gather relevant information to support and enhance your proposal development process.

## Progressive Proposal Process

Your task is to propose a new GAU design from scratch using the information provided.

Your will follow an iterative process of research, ideation, and refinement. This process consists of two main phases:

### Phase 1: Multiple Search and Refinement Rounds

In this phase, you will conduct multiple rounds of search and refinement without external review.
You need to think of how to innovate the LM block design that beyond the existing state-of-the-art. Each round consists of:

1. **Search**: Utilize the search assistant to gather relevant information. Formulate specific queries to investigate aspects that can help you innovate the LM block design. For searching external sources, use keywords, but limit to no more than 3 keywords at a time to avoid potential failure. If you want to search more topics, do so in multiple rounds.

2. **Analysis**: Carefully analyze the search results and detailed analysis provided by the search assistant. Extract key insights that can inform your proposal.

3. **Refinement**: Based on your analysis, refine and improve your proposal. This may involve modifying existing elements, adding new components, or adjusting your approach if the research suggests a more promising direction.

4. **Self-Assessment**: Evaluate your refined proposal against the original objectives and requirements. Identify areas that still need improvement or further research.

5. **Iteration Decision**: Determine if another round of search and refinement is necessary. If so, return to step 1 with new, focused queries based on your self-assessment.

Repeat this cycle as many times as needed until you feel your proposal is ready for review.

Note: Throughout this process, ensure that your proposals are supported by mathematical, theoretical, or logical justifications. Each design decision should be backed by sound reasoning and, where applicable, mathematical formulations.

### Phase 2: Review and Major Refinement

Once you believe your proposal is sufficiently developed:

1. **Submission for Review**: Present your proposal for external review.

2. **Feedback Analysis**: Carefully consider the feedback received.

3. **Major Refinement**: If the proposal doesn't pass the review, use the feedback to guide a major refinement of your design. This may involve returning to Phase 1 for additional rounds of search and refinement.

4. **Resubmission**: After major refinement, resubmit your proposal for another review.

Repeat Phase 2 until your proposal passes the review.

## Guidelines for Using the Search Assistant

When using the search assistant, follow these guidelines:

1. **Query Formulation**: 
   - Construct clear, specific queries related to your current design challenges or areas of uncertainty.
   - Use a combination of technical terms and concepts to narrow down results.
   - Consider searching for both recent innovations and foundational papers in relevant areas.

2. **Search Categories**:
   - Use broad searches for external sources to explore cutting-edge research and diverse approaches.
   - Utilize detailed searches of the internal library for in-depth technical information and established methodologies.

3. **Result Integration**:
   - Critically evaluate the search results and analysis provided by the search assistant for relevance and potential impact on your design.
   - Clearly cite and reference any papers or sources that significantly influence your proposal.

4. **Iterative Refinement**:
   - Use the insights from each search to inform subsequent queries, allowing for a more focused and in-depth exploration of relevant topics.

## Proposal Structure

Maintain and update the following structure in your proposal throughout the process:

1. **Title**: A concise, descriptive name for your proposed design.
2. **Motivation**: Explain the idea of how to innovate the LM block design.
3. **Problem Analysis**: Provide a detailed analysis of the innovation plan, how can it outperform the existing state-of-the-art.
4. **Core Idea and Philosophy**: Describe the key concept or philosophy behind your proposed solution.
5. **Design Plan**: 
   - Outline your approach for the recombination of the parents.
   - Provide detailed descriptions of modifications and new structures.
   - Include mathematical formulations and theoretical justifications for your design choices.
6. **Implementation Guidelines**:
   - Provide pseudo-code for the innovation of the LM block design.
   - Include mathematical formulas necessary for implementation.
   - Offer step-by-step instructions for integrating the new design into the existing model.
7. **Research Summary**: 
   - List key search queries used across all rounds.
   - Summarize the most relevant findings from your searches, including insights from the search assistant's analysis.
   - Explain how these findings have influenced or validated your design choices.
8. **Evolution of Design**:
   - Track major changes and improvements made across refinement rounds.
   - Discuss how these changes address challenges or leverage new insights.
9. **Theoretical Analysis**:
   - Provide mathematical or logical arguments for why your design is expected to improve model performance.
   - Discuss potential trade-offs and how they are addressed.
10. **Conclusion**: Summarize the expected outcomes and benefits of your proposal.
11. **References**: List all sources used in the proposal, properly formatted.

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
- **Creativity matters**: Strive for a design that is innovative over the existing models. 
- **Local modifications**: Focus on making changes to a single GAU (excluding the root unit) and its potential child GAUs. Ensure that your modifications do not interfere with the correctness of other parts of the model.
- **Simplicity and implementability**: Prioritize designs that are relatively simple and feasible to implement. Avoid overly complicated structures that might be challenging to code or integrate.
- **Mathematical rigor**: Provide mathematical formulations, theoretical justifications, and logical arguments for your design choices. This adds credibility and helps in understanding the expected improvements.
- **Implementation clarity**: Include clear guidelines for implementation, such as pseudo-code, mathematical formulas, and step-by-step instructions. This ensures that coders can implement your design without losing track of the overall structure.

Remember, the goal of this process is to develop a well-researched, innovative, yet implementable proposal for an LM block design. Take full advantage of the multiple refinement rounds to create a robust, thoroughly considered design before submitting for review. 

You are now ready to begin the progressive proposal process. Start with your initial proposal and prepare for the first round of research and refinement.
"""

GUS_DESIGN_PROPOSER_SYSTEM_ISEARCH = AgentPrompt(GUS_DESIGN_PROPOSER_SYSTEM_ISEARCH_prompt)



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
      abstract: str = Field(..., description="The abstract of the proposal, a concise summary of the core idea of the proposal.")
      proposal: str = Field(..., description="The full proposal, keep the format instructions.")
      selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on. Do not select the root unit to avoid too drastic changes immediately.")
      variantname: str = Field(..., description="The name of the variant of the selected GAU you are going to design.")
      modelname: str = Field(..., description="The name of the resulting model by applying the variant of GAU.")
   
   if use_isearch:
      GUM_DESIGN_PROPOSAL_ISEARCH_prompt = """
{SEED}

Here are the sibling designs with the same seed, avoid proposing the same design as your siblings, think of how to make your design unique and better.

{SIBLINGS}

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
Here is the search results from your last query, you will not be able to access the search assistant again after this, so do not include any more search queries:

{SEARCH_RESULTS}

Based on the seed design, search results, and your analysis, develop a comprehensive proposal for a novel LM block design. Remember to follow the output format strictly.

Ensure your proposal is innovative yet feasible, aiming to advance state-of-the-art LM performance. Balance creativity with practical considerations, and clearly articulate how your design improves upon existing architectures.
"""
      prompt1=AgentPrompt(GUM_DESIGN_PROPOSAL_ISEARCH_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_ISEARCH_format)
      prompt2=AgentPrompt(GUM_DESIGN_PROPOSAL_ISEARCH_FINISH_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_STAGE2_format)
      return prompt1,prompt2

   elif two_stage:
      GUM_DESIGN_PROPOSAL_stage1_prompt = """
{SEED}

Here are the sibling designs with the same seed, avoid proposing the same design as your siblings, think of how to make your design unique and better.

{SIBLINGS}

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

Here are the sibling designs with the same seed, avoid proposing the same design as your siblings, think of how to make your design unique and better.

{SIBLINGS}

Check the seed design, then give your proposal and the selection of the GAU to modify follow the instructions.
"""

      return AgentPrompt(GUM_DESIGN_PROPOSAL_prompt,GENERAL_JSON_parser,GUM_DESIGN_PROPOSAL_STAGE2_format)




   
class GUC_DESIGN_PROPOSAL_STAGE1_format(BaseModel):
   ideation: str = Field(..., description="The initial ideation about the direction of how to recombine the parents.")
   instructions: str = Field(..., description="The instructions for the information gathering assistant.")

class GUC_DESIGN_PROPOSAL_STAGE2_format(BaseModel):
   abstract: str = Field(..., description="The abstract of the proposal, a concise summary of the core idea of the proposal.")
   proposal: str = Field(..., description="The full proposal, keep the format instructions.")
   modelname: str = Field(..., description="The name of the resulting model by recombining the parents.")
   
GUC_DESIGN_PROPOSAL_ISEARCH_prompt = """
{PARENTS}

1. Analyze the parents:
   - Identify key features, advantages and limitations of existing architectures.
   - Determine best ways to combine the parents to get a better design.

2. Formulate search queries:
   - Create a high-level query for external sources to explore recent advancements in LM architectures.
   - Develop a detailed query for the internal vector store to extract specific technical information on LM block components.

3. Outline the key areas of investigation for recombining the parents, such as:
   - Efficiency improvements
   - Scalability enhancements
   - New attention mechanisms
   - Innovative ways to handle context or memory

4. Based on your analysis, determine if you need to conduct more searches or if you have gathered sufficient information to begin formulating a proposal.

Focus on thorough information gathering and insightful analysis to lay a strong foundation for the subsequent proposal development process.
"""


GUC_DESIGN_PROPOSAL_ISEARCH_FINISH_prompt = """
Here is the search results from your last query, you will not be able to access the search assistant again after this, so do not include any more search queries:

{SEARCH_RESULTS}

Based on the seed design, search results, and your analysis, develop a comprehensive proposal for recombining the parents. Remember to follow the output format strictly.

Ensure your proposal is best recombining the parents that aim to advance state-of-the-art LM performance and clearly articulate how your design improves upon existing architectures.
"""

GUC_DESIGN_PROPOSAL_ISEARCH=AgentPrompt(GUC_DESIGN_PROPOSAL_ISEARCH_prompt,GENERAL_JSON_parser,GUC_DESIGN_PROPOSAL_STAGE1_format)
GUC_DESIGN_PROPOSAL_ISEARCH_FINISH=AgentPrompt(GUC_DESIGN_PROPOSAL_ISEARCH_FINISH_prompt,GENERAL_JSON_parser,GUC_DESIGN_PROPOSAL_STAGE2_format)





class GUS_DESIGN_PROPOSAL_STAGE1_format(BaseModel):
   ideation: str = Field(..., description="The initial ideation about the direction of how to propose a novel design.")
   instructions: str = Field(..., description="The instructions for the information gathering assistant.")

class GUS_DESIGN_PROPOSAL_STAGE2_format(BaseModel):
   abstract: str = Field(..., description="The abstract of the proposal, a concise summary of the core idea of the proposal.")
   proposal: str = Field(..., description="The full proposal, keep the format instructions.")
   modelname: str = Field(..., description="The name of the resulting model in your proposal.")
   
GUS_DESIGN_PROPOSAL_ISEARCH_prompt = """
{REFS}

Based on the provided information:

1. Analyze the current state of the arts of LM block designs:
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

GUS_DESIGN_PROPOSAL_ISEARCH_FINISH_prompt = """
Here is the search results from your last query, you will not be able to access the search assistant again after this, so do not include any more search queries:

{SEARCH_RESULTS}

Based on the seed design, search results, and your analysis, develop a comprehensive proposal for a novel LM block design. Remember to follow the output format strictly.

Ensure your proposal is innovative yet feasible, aiming to advance state-of-the-art LM performance. Balance creativity with practical considerations, and clearly articulate how your design improves upon existing architectures.
"""
GUS_DESIGN_PROPOSAL_ISEARCH=AgentPrompt(GUS_DESIGN_PROPOSAL_ISEARCH_prompt,GENERAL_JSON_parser,GUS_DESIGN_PROPOSAL_STAGE1_format)
GUS_DESIGN_PROPOSAL_ISEARCH_FINISH=AgentPrompt(GUS_DESIGN_PROPOSAL_ISEARCH_FINISH_prompt,GENERAL_JSON_parser,GUS_DESIGN_PROPOSAL_STAGE2_format)



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




""" ============================= GU Proposal Reviewer System ===================================== """


# region GU Proposal Reviewer System



GENERAL_REVIEWER_INSTRUCTIONS = """
## Instructions for Reviewing the Proposal

1. **Conduct Investigations before Reviewing**:
   - Use the provided search functionality to gather information about existing research and implementations related to the proposal.
   - You will be asked to conduct multiple rounds of search if necessary to gather comprehensive information.

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

7.  **No Expectation of Empirical Evaluation**: 
   - The current review is based on design and theory. You should not expect empirical results or a fully implemented model at this stage.
"""



GU_REVIEW_PROCESS_INSTRUCTIONS = """

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


GUM_PROPOSAL_REVIEWER_SYSTEM_prompt = """

You are an expert in autoregressive language model research, and you have been asked to review a proposal for improving the design of an autoregressive language model (LM) block.

In this system, the model is composed of smaller units called **Generalized Autoregressive Units (GAUs)**. These GAUs form the building blocks of the LM. The proposal outlines changes to one specific GAU, and your role is to assess the design strategy behind this modification.

Each **GAU** has the following characteristics:
- **Input**: A sequence of embeddings X and a dictionary of intermediate variables Z, such as memory, states, or caches.
- **Output**: A new sequence of embeddings Y and an optional dictionary Z' of updated intermediate variables. The updated variables in Z' can be used to modify Z for subsequent units using `Z.update(Z')`.

The system builds complex autoregressive model blocks by nesting multiple GAUs. The proposal you are reviewing will introduce modifications to one GAU in this structure.

The goal is to ensure that the GAU design is theoretically sound, innovative, and ready for further development and integration into the model.

""" + GENERAL_REVIEWER_INSTRUCTIONS + GU_REVIEW_PROCESS_INSTRUCTIONS

# a rating of 4 or above is required to pass. # do not let agent know


GUC_PROPOSAL_REVIEWER_SYSTEM_prompt = """
You are an expert in autoregressive language model research, and you have been asked to review a proposal for improving the design of an autoregressive language model (LM) block.

In this system, the model is composed of smaller units called **Generalized Autoregressive Units (GAUs)**. These GAUs form the building blocks of the LM. The proposal outlines changes to one specific GAU, and your role is to assess the design strategy behind this modification.

## GAU Characteristics

Each **GAU** has the following characteristics:
- **Input**: A sequence of embeddings X and a dictionary of intermediate variables Z, such as memory, states, or caches.
- **Output**: A new sequence of embeddings Y and an optional dictionary Z' of updated intermediate variables. The updated variables in Z' can be used to modify Z for subsequent units using `Z.update(Z')`.

The system builds complex autoregressive model blocks by nesting multiple GAUs. 

The proposal you are reviewing will try to produce a new design by recombination of given parent designs:

{SEED}

The goal is to reuse the units from the parents to form a new design, and the new design is expected to best preserve the strengths of the parents and also to fix the issues of the parents.

""" + GENERAL_REVIEWER_INSTRUCTIONS + GU_REVIEW_PROCESS_INSTRUCTIONS


GUS_PROPOSAL_REVIEWER_SYSTEM_prompt = """
You are an expert in autoregressive language model research, and you have been asked to review a proposal for improving the design of an autoregressive language model (LM) block.

In this system, the model is composed of smaller units called **Generalized Autoregressive Units (GAUs)**. These GAUs form the building blocks of the LM. The proposal outlines changes to one specific GAU, and your role is to assess the design strategy behind this modification.

Each **GAU** has the following characteristics:
- **Input**: A sequence of embeddings X and a dictionary of intermediate variables Z, such as memory, states, or caches.
- **Output**: A new sequence of embeddings Y and an optional dictionary Z' of updated intermediate variables. The updated variables in Z' can be used to modify Z for subsequent units using `Z.update(Z')`.

The system builds complex autoregressive model blocks by nesting multiple GAUs. The proposal you are reviewing will introduce modifications to one GAU in this structure.

The goal is to ensure that the design is theoretically sound, innovative, and has the potential to improve the performance over the state-of-the-art models.

""" + GENERAL_REVIEWER_INSTRUCTIONS + GU_REVIEW_PROCESS_INSTRUCTIONS


GUM_PROPOSAL_REVIEWER_SYSTEM = AgentPrompt(GUM_PROPOSAL_REVIEWER_SYSTEM_prompt)
GUC_PROPOSAL_REVIEWER_SYSTEM = AgentPrompt(GUC_PROPOSAL_REVIEWER_SYSTEM_prompt)
GUS_PROPOSAL_REVIEWER_SYSTEM = AgentPrompt(GUS_PROPOSAL_REVIEWER_SYSTEM_prompt)



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
- **Input**: A sequence of embeddings X and a dictionary of intermediate variables Z, such as memory, states, or caches.
- **Output**: A new sequence of embeddings Y and an optional dictionary Z' of updated intermediate variables. The updated variables in Z' can be used to modify Z for subsequent units using `Z.update(Z')`.

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

7.  **No Empirical Evaluation**: 
   - The current review is based on design and theory. You should not expect empirical results or a fully implemented model at this stage.

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

**Siblings from Previous Designs with Same Seeds**:
{SIBLINGS}

**Similar Design Proposals from Previous Designs**:
{TOP_K_PPS}

### Review Instructions

Please evaluate the design in the proposal based on its **technical merits**. Your review should focus on:

- **Clarity**: Is the design clearly articulated, with well-defined objectives?
- **Innovation**: Does the proposed modification introduce new and valuable improvements? How does it compare to existing researches and previous design proposals?
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

**Siblings from Previous Designs with Same Seeds**:
{SIBLINGS}

**Similar Design Proposals from Previous Designs**:
{TOP_K_PPS}

Your task is to conduct an initial analysis and formulate search queries to gather more information. Please provide:

1. A brief initial analysis of the proposal, highlighting key aspects that require further investigation.
2. A high-level query for broad external searches (arXiv, Papers with Code, Semantic Scholar).
3. A detailed query for searching the internal vector store of research papers.
4. Check if the proposal is novel or not compared to the previous design proposals and existing researches. 

Focus on the proposal's potential impact on accuracy, robustness, efficiency, and scalability. Consider its novelty and alignment with current research trends.
"""


GUC_PROPOSAL_REVIEW_ISEARCH_BEGIN_prompt = """
You are an expert reviewer evaluating a proposal for modifying a Generalized Autoregressive Unit (GAU) in an autoregressive language model block. You have been provided with the following information:

**Proposal for Review**:
{PROPOSAL}

**Similar Design Proposals from Previous Designs**:
{TOP_K_PPS}

Your task is to conduct an initial analysis and formulate search queries to gather more information. Please provide:

1. A brief initial analysis of the proposal, highlighting key aspects that require further investigation.
2. A high-level query for broad external searches (arXiv, Papers with Code, Semantic Scholar).
3. A detailed query for searching the internal vector store of research papers.
4. Check if the proposal is novel or not compared to the previous design proposals and existing researches. 

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

GUC_PROPOSAL_REVIEW_ISEARCH_BEGIN=AgentPrompt(GUC_PROPOSAL_REVIEW_ISEARCH_BEGIN_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_ISEARCH_format)

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
         abstract: str = Field(..., description="The abstract of the proposal, a concise summary of the core idea of the proposal.")
         proposal: str = Field(..., description="The fall proposal, keep the format instructions.")
         selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on. Do not select the root unit to avoid too drastic changes immediately.")
         variantname: str = Field(..., description="The name of the variant of the selected GAU you are going to design.")
         modelname: str = Field(..., description="The name of the resulting model by applying the variant of GAU.")
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
Here is the search results from your last query, you will not be able to access the search assistant again after this, so do not include any more search queries:

{SEARCH_RESULTS}

Based on your reflection, search results, and your analysis, develop a comprehensive proposal for a novel LM block design. Remember to follow the output format strictly.

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
         abstract: str = Field(..., description="The abstract of the proposal, a concise summary of the core idea of the proposal.")
         selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on. Do not select the root unit to avoid too drastic changes immediately.")
         variantname: str = Field(..., description="The name of the variant of the selected GAU you are going to design.")
         modelname: str = Field(..., description="The name of the resulting model by applying the variant of GAU.")
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
         abstract: str = Field(..., description="The abstract of the proposal, a concise summary of the core idea of the proposal.")
         proposal: str = Field(..., description="The fall proposal, keep the format instructions.")
         selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on. Do not select the root unit to avoid too drastic changes immediately.")
         variantname: str = Field(..., description="The name of the variant of the selected GAU you are going to design.")
         modelname: str = Field(..., description="The name of the resulting model by applying the variant of GAU.")
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






class GUC_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_format(BaseModel):
   reflection: str = Field(..., description="The reflection based on the review, rating, and suggestions.")
   analysis: str = Field(..., description="A detailed analysis of the current progress of the proposal, including identified gaps, areas for improvement, and specific information needed to enhance the design. This should guide the formulation of search queries.")
   keywords: str = Field(..., description="Keywords for searching external sources like arXiv, Papers with Code, and Semantic Scholar. This should be clear, concise keywords derived from the analysis to help the search engine locate the papers that may help you in based on title, abstract, and other metadata, aimed at best recombining the parents. Do not give more than 3 keywords a time which may cause failure, if you want to search more topic, do it in next round.")
   detail: str = Field(..., description="A detailed query used for searching the internal vector store of research papers and technical documents. This should be a specific, targeted query that aims to extract relevant information from the contents of papers in the vector store, focusing on particular aspects of LM architecture, techniques, or performance metrics identified in the analysis.")
   ready: bool = Field(..., description="Whether you should continue the search and refinement process or ready to give the proposal.")

class GUC_PROPOSAL_REFINEMENT_FINISH_format(BaseModel):
   abstract: str = Field(..., description="The abstract of the proposal, a concise summary of the core idea of the proposal.")
   proposal: str = Field(..., description="The fall proposal, keep the format instructions.")
   modelname: str = Field(..., description="The name of the resulting model by recombining the parents.")
   changes: str = Field(..., description="The summary of the changes you made.") 


GUC_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_prompt = """
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

GUC_PROPOSAL_REFINEMENT_FINISH_prompt = """
Here is the search results from your last query, you will not be able to access the search assistant again after this, so do not include any more search queries:

{SEARCH_RESULTS}

Based on your reflection, search results, and your analysis, develop a comprehensive proposal for recombining the parents. Remember to follow the output format strictly.

Ensure your proposal is best recombining the parents that aim to advance state-of-the-art LM performance and clearly articulate how your design improves upon existing architectures.
"""

GUC_DESIGN_PROPOSAL_ISEARCH_REFINEMENT=AgentPrompt(GUC_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_prompt,GENERAL_JSON_parser,GUC_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_format)
GUC_PROPOSAL_REFINEMENT_FINISH=AgentPrompt(GUC_PROPOSAL_REFINEMENT_FINISH_prompt,GENERAL_JSON_parser,GUC_PROPOSAL_REFINEMENT_FINISH_format)







class GUS_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_format(BaseModel):
   reflection: str = Field(..., description="The reflection based on the review, rating, and suggestions.")
   analysis: str = Field(..., description="A detailed analysis of the current progress of the proposal, including identified gaps, areas for improvement, and specific information needed to enhance the design. This should guide the formulation of search queries.")
   keywords: str = Field(..., description="Keywords for searching external sources like arXiv, Papers with Code, and Semantic Scholar. This should be clear, concise keywords derived from the analysis to help the search engine locate the papers that may help you in based on title, abstract, and other metadata, aimed at addressing identified gaps or exploring potential improvements in the LM block design. Do not give more than 3 keywords a time which may cause failure, if you want to search more topic, do it in next round.")
   detail: str = Field(..., description="A detailed query used for searching the internal vector store of research papers and technical documents. This should be a specific, targeted query that aims to extract relevant information from the contents of papers in the vector store, focusing on particular aspects of LM architecture, techniques, or performance metrics identified in the analysis.")
   ready: bool = Field(..., description="Whether you should continue the search and refinement process or ready to give the proposal.")

class GUS_PROPOSAL_REFINEMENT_FINISH_format(BaseModel):
   abstract: str = Field(..., description="The abstract of the proposal, a concise summary of the core idea of the proposal.")
   proposal: str = Field(..., description="The fall proposal, keep the format instructions.")
   modelname: str = Field(..., description="The name of the resulting model by applying the variant of GAU.")
   changes: str = Field(..., description="The summary of the changes you made.") 


GUS_PROPOSAL_REFINEMENT_FINISH_prompt = """
Here is the search results from your last query, you will not be able to access the search assistant again after this, so do not include any more search queries:

{SEARCH_RESULTS}

Based on your reflection, search results, and your analysis, develop a comprehensive proposal for a novel LM block design. Remember to follow the output format strictly.

Ensure your proposal is innovative yet feasible, aiming to advance state-of-the-art LM performance. Balance creativity with practical considerations, and clearly articulate how your design improves upon existing architectures.
"""

GUS_DESIGN_PROPOSAL_ISEARCH_REFINEMENT=AgentPrompt(GUC_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_prompt,GENERAL_JSON_parser,GUS_DESIGN_PROPOSAL_ISEARCH_REFINEMENT_format)
GUS_PROPOSAL_REFINEMENT_FINISH=AgentPrompt(GUS_PROPOSAL_REFINEMENT_FINISH_prompt,GENERAL_JSON_parser,GUS_PROPOSAL_REFINEMENT_FINISH_format)









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

**Similar Design Proposals from Previous Designs**:
{TOP_K_PPS}

### Review Instructions

1. **Carefully review** the refined proposal and compare it against your original feedback.
2. **Examine the change log** to determine whether the designer has successfully addressed the concerns you raised in your previous review.
3. **Consider any new or remaining concerns** that may have surfaced in the refined proposal.
4. Provide your **review, rating, and suggestions**. Keep in mind:
   - Your evaluation should be based on the **design quality**, not the writing style.
   - Any feedback on writing should be included under **suggestions**, not reflected in the rating.
   - Do not inflate the rating simply because previous concerns were addressed. The rating should reflect the overall merit of the design at this stage.
5. Check if the updated proposal is novel or not compared to the previous design proposals and existing researches.
   
### Final Note:
Be strict and objective. Approve the proposal only if it meets the necessary standards of quality and innovation. Do not pass a proposal unless it is sufficiently strong.
"""



GUC_PROPOSAL_REREVIEW_prompt = """
The designer has modified the proposal based on your previous review. Below is the refined version for your reconsideration:

**Proposal**:
{PROPOSAL}

**Change Log** (summary of modifications made):
{CHANGES}

**Similar Design Proposals from Previous Designs**:
{TOP_K_PPS}

### Review Instructions

1. **Carefully review** the refined proposal and compare it against your original feedback.
2. **Examine the change log** to determine whether the designer has successfully addressed the concerns you raised in your previous review.
3. **Consider any new or remaining concerns** that may have surfaced in the refined proposal.
4. Provide your **review, rating, and suggestions**. Keep in mind:
   - Your evaluation should be based on the **design quality**, not the writing style.
   - Any feedback on writing should be included under **suggestions**, not reflected in the rating.
   - Do not inflate the rating simply because previous concerns were addressed. The rating should reflect the overall merit of the design at this stage.
5. Check if the updated proposal is novel or not compared to the previous design proposals and existing researches.
   
### Final Note:
Be strict and objective. Approve the proposal only if it meets the necessary standards of quality and innovation. Do not pass a proposal unless it is sufficiently strong.
"""


GUM_PROPOSAL_REREVIEW = AgentPrompt(GUM_PROPOSAL_REREVIEW_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_format)
GUC_PROPOSAL_REREVIEW = AgentPrompt(GUC_PROPOSAL_REREVIEW_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_format)

GUM_PROPOSAL_REREVIEW_ISEARCH_prompt = GUM_PROPOSAL_REREVIEW_prompt+ """
Please review with the help of the search engine.
"""

GUC_PROPOSAL_REREVIEW_ISEARCH_prompt = GUC_PROPOSAL_REREVIEW_prompt+ """
Please review with the help of the search engine.
"""

GUM_PROPOSAL_REREVIEW_ISEARCH = AgentPrompt(GUM_PROPOSAL_REREVIEW_ISEARCH_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_ISEARCH_format)
GUC_PROPOSAL_REREVIEW_ISEARCH = AgentPrompt(GUC_PROPOSAL_REREVIEW_ISEARCH_prompt,GENERAL_JSON_parser,GUM_PROPOSAL_REVIEW_ISEARCH_format)

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


# region GUM Design Implementer System


# About GAB
GUM_DESIGNER_SYSTEM_prompt_part1 = """
Modern LMs are typically structured as a stack of repeating blocks. Each block accepts: 

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


#### Note

1. **GAU is just a specialized nn.Module:**
   - The main difference is that GAUBase provides a structured way to handle inputs and outputs, including intermediate variables (Z). 
   - You can define layers and implement logic just like in a regular nn.Module.

2. Input and Output structure:
   - Input: X (tensor of shape (batch, seqlen, embed_dim)) and Z (dictionary of intermediate variables)
   - Output: Y (tensor of same shape as X) and updated Z (dictionary)

3. The _forward method:
   - This is where you implement the core logic of your GAU.
   - It should take X and any needed intermediate variables from Z as arguments.
   - It should return Y and a dictionary of updated/new intermediate variables.

4. Nesting GAUs:
   - You can create more complex GAUs by nesting simpler ones.
   - In the _forward method of a complex GAU, you would call the simpler GAUs in sequence, passing the output of one to the input of the next.

5. Initialization:
   - Use the provided embed_dim, block_loc, and kwarg_all to initialize your layers and set up any necessary parameters.



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
   - Ensure that your GAU class inherits from `GAUBase` and is named as specified in the proposal. You should only define **one** GAU class in your implementation. Do not define any other GAU classes in this block. And the name of the class should be the unit name you are implementing.
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
   - `block_loc` is a tuple (block_idx, n_block) that locates the GAU
   within
     the network where block_idx starts from 0, allowing you to implement
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
   - Name GAUs meaningfully. Each GAU should represent a distinct unit with a clear function in the architecture. If you are improving based on an existing GAU, DO NOT use the same name as the existing GAU. Give a new name, and remember to use the new name to the class in your implementation.
   - Follow a top-down design approach: if the operation is complex, decompose it into child GAUs and define their placeholders. Ensure each placeholder aligns with the broader structure of the model, ready for future implementation.

8. **Be Innovative**:
   - Focus on designing GAUs that improve performance and efficiency. Avoid replicating existing architectures (e.g., vanilla Transformers) and aim to transcend current state-of-the-art models.
   - Introduce unique mechanisms or structures that differentiate your GAUs from traditional models.
   - Do not simply copy from the references or existing codebases. You can use their ideas to inspire you for your own original designs.
   
9. **Be Consistent**: 
   - Ensure your design remains consistent and fits seamlessly into the overall system architecture.
   - Avoid introducing errors, inconsistencies, or redundant code. Your GAU should operate smoothly alongside existing GAUs and should not introduce any deviations from the overall design philosophy.

"""


GUM_DESIGNER_SYSTEM_prompt="You are a researcher designing a new autoregressive language model (LM). "+\
   GUM_DESIGNER_SYSTEM_prompt_part1+GUM_DESIGNER_SYSTEM_prompt_part2+GUM_DESIGNER_SYSTEM_prompt_part3+GUM_DESIGNER_SYSTEM_prompt_part4


GUM_DESIGNER_SYSTEM = AgentPrompt(GUM_DESIGNER_SYSTEM_prompt)





# endregion




'''
#######################################################
# GUM Implementation System Prompt for O1Coder
#######################################################
'''


""" ============================= GUM Design Implementer System for O1Coder ===================================== """


# region GUM Design Implementer System


# About GAB
GUM_DESIGNER_SYSTEM_O1_prompt_part1 = """
Modern LMs are typically structured as a stack of repeating blocks. Each block accepts: 

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
GUM_DESIGNER_SYSTEM_O1_prompt_part2 = """
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

#### Note

1. **GAU is just a specialized nn.Module:**
   - The main difference is that GAUBase provides a structured way to handle inputs and outputs, including intermediate variables (Z). 
   - You can define layers and implement logic just like in a regular nn.Module.

2. Input and Output structure:
   - Input: X (tensor of shape (batch, seqlen, embed_dim)) and Z (dictionary of intermediate variables)
   - Output: Y (tensor of same shape as X) and updated Z (dictionary)

3. The _forward method:
   - This is where you implement the core logic of your GAU.
   - It should take X and any needed intermediate variables from Z as arguments.
   - It should return Y and a dictionary of updated/new intermediate variables.

4. Nesting GAUs:
   - You can create more complex GAUs by nesting simpler ones.
   - In the _forward method of a complex GAU, you would call the simpler GAUs in sequence, passing the output of one to the input of the next.

5. Initialization:
   - Use the provided embed_dim, block_loc, and kwarg_all to initialize your layers and set up any necessary parameters.


"""

# About the role  
GUM_DESIGNER_SYSTEM_O1_prompt_part3 = """

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
  Every time you will be asked to work on one **Single GAU**, either the one selected by the proposal or from the planner. You can include multiple codes in your response, in each code, you should implement one GAU, the selected one or the children you declared. You can choose to leave some children to implement in the future or declare an already implemented child to reuse it. You need to consider the dependency between the GAU you are implementing in a code and other GAUs. Every GAU code should strictly follow the format requirement from the template.
  
- **No Access to Other GAUs**:  
  When working on a GAU, you will only have access to the current GAU’s implementation and its childrens' implementations and not the internal details of other GAUs. Ensure interactions between GAUs are handled through `Z` and `Z_`.

- **Child GAUs**:  
  When decomposing a GAU into child GAUs, ensure that the placeholder instantiation and calls are correct. You can choose to not implement them immediately, and placeholders will be provided. Ensure all input/output interfaces for placeholders are properly handled in the current GAU if you choose to implement them later.

- **Docstring**:  
  Provide a **docstring** for the GAU, explaining its inputs, outputs, and purpose. Follow PyTorch’s style guidelines, as the docstring will help others understand the GAU’s role and how it interacts with other units.

- **Unit Tests**:  
  Write at least one **unit test** for each GAU. Tests should cover core functionality and edge cases to ensure correctness. After the GAU is integrated into the model, tests will be run automatically to validate its performance.

- **Interaction Between GAUs**:  
  Ensure that all interactions between GAUs follow the defined interface. You will not be able to modify other GAUs besides the current GAU and its children in your response, so proper input/output management is essential.

- **Focus on One GAU**:  
  Focus on the design of the current GAU without worrying about the internal workings of its parents or siblings. 

- **Iterative Design**:  
  You will receive feedback and go through iterative rounds of design. If your implementation introduces errors or fails tests, you will need to debug and refine your GAU. The system will guide you through this process with error traces and diagnostics.

"""


GUM_DESIGNER_SYSTEM_O1_prompt_part4 = """
## Guidelines for Designing the GAU:

1. **Class Naming & Structure**:
   - Ensure that your GAU class inherits from `GAUBase` and is named as specified in the proposal. You should only define **one** GAU class in your implementation. Do not define any other GAU classes in this block. And the name of the class should be the unit name you are implementing.
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
   - `block_loc` is a tuple (block_idx, n_block) that locates the GAU
   within
     the network where block_idx starts from 0, allowing you to implement
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
It is round {ROUND} for the design implementation. You will need to select the GAU to work on for this round.

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
      selection: SelectionEnum = Field(..., description="The name of the GAU you are going to work on. Do not select the root unit to avoid too drastic changes immediately.")
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
      GUM_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_REFINE_format
      
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

""" + GUM_IMPLEMENTATION_UNIT_prompt+" Please also give a new name of this variant of the GAU."
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
embeddings X and a dictionary of intermediate variables Z, such as
memory, states, or caches. - **Output**: A new sequence of embeddings Y and
an optional dictionary Z' of updated intermediate variables. The updated
variables in Z' can be used to modify Z for subsequent units, using
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
{IMPLEMENTATION}

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
  {IMPLEMENTATION}

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
  {IMPLEMENTATION}

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



'''
#######################################################
##                                                                                               
## Trio architecture for GU Mutate from existing design prompts                                    
##                                                                                               
#######################################################
'''

# region Trio architecture for GU Mutate from existing design prompts


GUT_GUILDLINES=f"""
- Guildline Part 1: Overview of autoregressive language models and block structure:

---

{GUM_DESIGNER_SYSTEM_prompt_part1}

---

- Guildline Part2: Explanation of Generalized Autoregressive Units (GAUs):

---

{GUM_DESIGNER_SYSTEM_prompt_part2}

---

- Guildline Part3: Instructions for the design process:

---

{GUM_DESIGNER_SYSTEM_prompt_part3}

---

- Guildline Part4: Guidelines for designing GAUs:

---

{GUM_DESIGNER_SYSTEM_prompt_part4}

---
"""




GUT_GUILDLINES_O1=f"""
- Guildline Part 1: Overview of autoregressive language models and block structure:

---

{GUM_DESIGNER_SYSTEM_O1_prompt_part1}

---

- Guildline Part2: Explanation of Generalized Autoregressive Units (GAUs):

---

{GUM_DESIGNER_SYSTEM_O1_prompt_part2}

---

- Guildline Part3: Instructions for the design process:

---

{GUM_DESIGNER_SYSTEM_O1_prompt_part3}

---

- Guildline Part4: Guidelines for designing GAUs:

---

{GUM_DESIGNER_SYSTEM_O1_prompt_part4}

---
"""



# def gen_GUT_IMPLEMENTATION_PLANNER_SYSTEM(use_o1=True):
#     if use_o1:
#         guildlines=GUT_GUILDLINES_O1
#     else:
#         guildlines=GUT_GUILDLINES
#     GUT_IMPLEMENTATION_PLANNER_SYSTEM_prompt="""
# # Implementation Planner System Prompt

# You are the Implementation Planner for a team designing a new autoregressive language model (LM) based on Generalized Autoregressive Units (GAUs). Your role is to guide the implementation process by making strategic decisions about which units to implement or refine, and providing high-level instructions to the Implementation Coder.

# ## Background and Context

# Please refer to the following sections from the original system prompt for essential background information:

# """ + guildlines + """

# Ensure that you are familiar with these sections as they provide crucial context for your role.

# ## The proposal and corresponding review for the design to implement

# ###  Overall Proposal for Refining the Design

# {PROPOSAL}

# ### Review of the Proposal

# {REVIEW}

# #### Rating

# {RATING} out of 5 (Passing score: >3)

# #### Proposal Selection of GAU to improve: {SELECTION}

# ## Your Responsibilities:

# 1. **Analyze the Proposal**: Thoroughly understand the proposed LM design, including all GAUs and their relationships.

# 2. **Prioritize Implementation**: Decide the order in which GAUs should be implemented or refined. Consider dependencies between units and the overall structure of the model.

# 3. **Select Next Unit**: Each round, choose either:
#    - An unimplemented GAU
#    - A previously implemented GAU that needs refinement
#    - You will not be asked in the first round, in which the coder will always start by the selected unit in the proposal

# 4. **Provide High-Level Instructions**: For the selected GAU, give clear, concise instructions to the Implementation Coder. These should include:
#    - The purpose and function of the GAU
#    - Key features or operations to implement
#    - Any specific requirements or constraints
#    - Potential challenges or areas that need careful consideration

# 5. **Consider Dependencies**: Ensure that your instructions account for the GAU's place in the overall architecture and its interactions with other units.

# 6. **Promote Innovation**: Encourage the Coder to explore novel approaches that could improve performance or efficiency, while staying true to the overall design philosophy.

# 7. **Iterative Refinement**: Based on feedback from the Implementation Observer, decide when a GAU needs further refinement and what aspects to focus on.

# ## Guidelines:

# - Follow the design principles outlined in Guideline Part 3, particularly regarding the decomposition of complex GAUs and the declaration of child GAUs.
# - Adhere to the guidelines for designing GAUs as specified in Guideline Part 4, including class naming, initialization, and call behavior.
# - Maintain a holistic view of the model architecture as described in Guideline Part 1 and Guideline Part 2.
# - Balance between faithfulness to the proposal and openness to improvements that could enhance model performance.
# - Ensure your instructions promote code that is modular, efficient, and scalable.
# - Consider the principles of good software design, such as DRY (Don't Repeat Yourself) and SOLID principles.
# - Be prepared to adjust the implementation plan based on insights gained during the process.

# Remember, your role is to guide the overall implementation strategy. You don't need to provide detailed code instructions, but rather high-level direction that will enable the Implementation Coder to write effective, innovative code that aligns with the original design principles and goals of the project.
# """

#     return AgentPrompt(GUT_IMPLEMENTATION_PLANNER_SYSTEM_prompt)


def gen_GUT_IMPLEMENTATION_CODER_SYSTEM(use_o1=True,mode=DesignModes.MUTATION):
   if use_o1:
      GUT_IMPLEMENTATION_CODER_SYSTEM_prompt="""
You are the Implementation Coder for a team designing a new autoregressive language model (LM). 

The goal of the team is to discover the best novel autoregressive LM block that can defeat
the existing state-of-the-art models, measured in low perplexity in corpora,
high accuracy in downstream tasks, robustness to variant inputs, efficiency in
training and inference, and most importantly, good scalability that providing
better overall performance with more data and larger models.
Your role is to write the code to implement the given proposal.

## Background

Modern LMs are typically structured as a stack of repeating blocks. Each block processes:

1. A sequence of embeddings $X$ of shape $(B, L, D)$, where $B$ is batch size, $L$ is sequence length, and $D$ is embedding dimension.
2. Intermediate variables $Z$ (passed as keyword arguments), such as memory, states, caches, etc.

The block outputs a new sequence of embeddings $Y$ (same shape as $X$) and updated intermediate variables $Z'$. Such a block can be written as:

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


## Generalized Autoregressive Units (GAUs)

GAUs are smaller components that compose LM blocks. They inherit from this base class:

```python {GAU_BASE}```


Key points:
- LM blocks can be decomposed into nested GAUs
- GAUs share the same interface as LM blocks
- GAUs can be arranged hierarchically and nested within each other

### Note

1. **GAU is just a specialized nn.Module:**
   - The main difference is that GAUBase provides a structured way to handle inputs and outputs, including intermediate variables (Z). 
   - You can define layers and implement logic just like in a regular nn.Module.

2. Input and Output structure:
   - Input: X (tensor of shape (batch, seqlen, embed_dim)) and Z (dictionary of intermediate variables)
   - Output: Y (tensor of same shape as X) and updated Z (dictionary)

3. The _forward method:
   - This is where you implement the core logic of your GAU.
   - It should take X and any needed intermediate variables from Z as arguments.
   - It should return Y and a dictionary of updated/new intermediate variables.

4. Nesting GAUs:
   - You can create more complex GAUs by nesting simpler ones.
   - In the _forward method of a complex GAU, you would call the simpler GAUs in sequence, passing the output of one to the input of the next.

5. Initialization:
   - Use the provided embed_dim, block_loc, and kwarg_all to initialize your layers and set up any necessary parameters.


### Instructions for the Implementation Process

1. You'll receive a proposal of a novel block design.
2. Implement the GAUs based on the proposal.
3. Follow the GAU template:

```python
{GAU_TEMPLATE}
```

### Key Design Principles:

1. **Decomposition of Complex GAUs**:  
   If a GAU is complex, you can consider to decompose it into smaller child GAUs to make the implementation and testing process easier. 

2. **Placeholder Declaration and Child GAU Calls**:  
   You can declare and instantiate child GAUs in the parent GAU’s `__init__` method as placeholders to be implemented later, like:
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

4. **Implementation format**:
You must include full implementations of units in your final response. 
You can provide multiple implementations of *different units* including the selected unit and optionally its children. 
You must wrape each implementation in a block quote as follows:
```python
{{full implementation of a unit, unittests decorated with @gau_test, and children declarations}}
```
. All implementations must follow the format of the GAU template, and remember to keep the first line as the marker `# GAU_IMPLEMENTATION_FILE` to allow the parser detect a GAU implementation file. 
Only the code block wrapped by ```python ``` and kept first line as `# GAU_IMPLEMENTATION_FILE` will be considered as a GAU implementation.
In order to allow the parser successfully detect the code blocks, DO NOT nest any ```python ``` blocks within the code block of a unit implementation, e.g., in examples of the doc string, dont wrap the examples with ```python ```.  
The class name of the GAU will be detected as the unit name of an implementation. Remember to keep the unittests and children declarations of each unit in the same file of the implementation. 
In another word, each file must contain three sections: 1) the unit implementation, 2) the unittests (all unittests must be decorated with @gau_test, otherwise it will be ignored), 3) the children declarations. 
And always remember to declare children GAUs if there is any in your unit, either new, placeholder or reuse existing ones. Otherwise the linker will not be able to find them.  
You can modify based on the implementations from the provided seed, but you should never simply copy them as your response. If you want to reuse a unit, you can simply declare it in the children list without providing the implementation. 


### Implementation Guidelines:
  
- **No Access to Other GAUs**:  
  When working on a GAU, you will only have access to the current GAU’s implementation and its childrens' implementations and not the internal details of other GAUs. Ensure interactions between GAUs are handled through `Z` and `Z_`.

- **Child GAUs**:  
  When decomposing a GAU into child GAUs, ensure that the placeholder instantiation and calls are correct. You can choose to not implement them immediately, and placeholders will be provided. Ensure all input/output interfaces for placeholders are properly handled in the current GAU if you choose to implement them later.

- **Docstring**:  
  Provide a **docstring** for the GAU, explaining its inputs, outputs, and purpose. Follow PyTorch’s style guidelines, as the docstring will help others understand the GAU’s role and how it interacts with other units.

- **Unit Tests**:  
  Write at least one **unit test** for each GAU. Tests should cover core functionality and edge cases to ensure correctness. After the GAU is integrated into the model, tests will be run automatically to validate its performance.

- **Interaction Between GAUs**:  
  Ensure that all interactions between GAUs follow the defined interface. You will not be able to modify other GAUs besides the current GAU and its children in your response, so proper input/output management is essential.

- **Iterative Design**:  
  You will receive feedback and go through iterative rounds of design. If your implementation introduces errors or fails tests, you will need to debug and refine your GAU. The system will guide you through this process with error traces and diagnostics.

- **Reuse Existing GAUs**:  
   If there is an existing GAU in the provided seed that can meet your needs, you should directly reuse it instead of implementing it again. You are encouraged to reuse existing GAUs. Declaring a new GAU only if it is necessary.

## Guidelines for Designing the GAU:

1. **Class Naming & Structure**:
   - Ensure that your GAU class inherits from `GAUBase` and is named as specified in the proposal. You should only define **one** GAU class in each implementation. Do not define any other GAU classes in this block. And the name of the class should be the unit name you are implementing.
   - If you are modifying based on an existing GAU, DO NOT use the original name, give a new name to the new GAU you are implementing.
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
   - `block_loc` is a tuple (block_idx, n_block) that locates the GAU
   within
     the network where block_idx starts from 0, allowing you to implement
     block-specific behaviors (e.g., varying architectures or operations between
     blocks, initializing intermediate variables acrossing blocks in the first
     block).

5. **Module Definition**:
    - Avoid using `GAU` instances inside `nn.Sequential`. You can use
      `nn.ModuleList` or `nn.ModuleDict`.
    - Do not define any nn.Module classes in your code. Declare child GAUs instead and do not implement them in your code.

6. **Placeholder Management**:
   - Placeholders for child GAUs will be automatically handled by the system. Avoid manually implementing placeholders at this stage. You will be prompted to implement them later when necessary.
   - When declaring placeholders for child GAUs in your implementation, follow the proper syntax and ensure correct input-output handling.

7. **Design Approach**:
   - Name GAUs meaningfully. Each GAU should represent a distinct unit with a clear function in the architecture.
   - Follow a top-down design approach: if the operation is complex, decompose it into child GAUs and define their placeholders. Ensure each placeholder aligns with the broader structure of the model, ready for future implementation. Or you can implement the children immediately insperate files wrapped by different python code blocks in your response.

9. **Be Consistent**: 
   - Ensure your implementation(s) remains consistent and fits seamlessly into the overall system architecture.
   - Avoid introducing errors, inconsistencies, or redundant code. Your GAU should operate smoothly alongside other GAUs and should not introduce any deviations from the overall design philosophy.


## Proposal

{PROPOSAL}

## Review 

{REVIEW}

### Rating: {RATING} out of 5

## Implementation Plan

This is the current plan and instructions from the an Implementation Planner in your team:

{PLAN}

"""
   else:
      GUT_IMPLEMENTATION_CODER_SYSTEM_prompt="""
You are the Implementation Coder for a team designing a new autoregressive language model (LM) based on Generalized Autoregressive Units (GAUs). Your role is to write the actual code for each GAU as directed by the Implementation Planner.


Implementation of a GAU should follow this template:

```python
{GAU_TEMPLATE}
```

## Background and Context

Please refer to the following sections from the original system prompt for essential background information:

""" + GUT_GUILDLINES + """

Ensure that you are familiar with these sections as they provide crucial context for your implementation work.

## The proposal and corresponding review for the design to implement

###  Overall Proposal

{PROPOSAL}

### Review of the Proposal

{REVIEW}

#### Rating

{RATING} out of 5 (Passing score: >3)

## Implementation Plan

This is the current plan and instructions from the an Implementation Planner in your team:

{PLAN}

## Your Responsibilities:

1. **Implement GAUs**: Write the Python code for the GAU selected by the Implementation Planner. This includes:
   - Defining the GAU class, inheriting from GAUBase
   - Implementing the `__init__` and `_forward` methods
   - Handling inputs and outputs correctly through the `X`, `Y`, `Z`, and `Z'` interface
   - Declaring and instantiating child GAUs as placeholders when necessary

2. **Follow Design Principles**: Adhere to the design principles outlined in Guideline Part 3 and Guideline Part 4, including:
   - Proper decomposition of complex GAUs into child units
   - Correct placeholder declaration and child GAU calls
   - Proper preparation of inputs and outputs
   - Following the specified class naming and initialization conventions

3. **Write Docstrings**: Provide clear and comprehensive docstrings for each GAU, explaining its purpose, inputs, outputs, and any important details about its operation.

4. **Create Unit Tests**: Write at least one unit test for each GAU to verify its core functionality and edge cases, as specified in Guideline Part 3. The unit tests should be written in the same file as the GAU implementation with @gau_test decorator.

5. **Innovate**: While implementing the GAU as directed, look for opportunities to improve efficiency or introduce novel mechanisms that could enhance the model's performance, as encouraged in Guideline Part 4.

6. **Handle Edge Cases**: Ensure your code gracefully handles potential issues, such as missing inputs or unexpected data types, as mentioned in Guideline Part 4.

## Guidelines:

- Follow Python best practices and PEP 8 style guidelines
- Ensure your code is clean, well-commented, and easy to understand
- Use meaningful variable and function names
- Optimize for both readability and performance
- Don't implement placeholder GAUs; these will be handled by the system as explained in Guideline Part 3
- Focus on the current GAU without worrying about the internal workings of other GAUs
- Be prepared to refine your implementation based on feedback from the Implementation Observer
- Adhere strictly to the GAU call behavior and initialization guidelines provided in Guideline Part 4

Remember, you're implementing one GAU at a time. Your code should be self-contained and interact with other units only through the defined interfaces. Always consider how your implementation fits into the broader architecture of the language model as described in Guideline Part 1 and Guideline Part 2.
"""

   if mode==DesignModes.MUTATION:
      GUT_IMPLEMENTATION_CODER_SYSTEM_prompt+="""
As a background, the proposal is going to improve the following seed design by improving the unit: {SELECTION}.

{SEED}
"""
   elif mode==DesignModes.CROSSOVER:
      GUT_IMPLEMENTATION_CODER_SYSTEM_prompt+="""
As a background, the proposal is going to produce a new design by recombination of the parent designs:

{PARENTS}
"""

   return AgentPrompt(GUT_IMPLEMENTATION_CODER_SYSTEM_prompt)
   


def gen_GUT_IMPLEMENTATION_OBSERVER_SYSTEM(use_o1=True,mode=DesignModes.MUTATION):
   if use_o1:
      guildlines=GUT_GUILDLINES_O1
   else:
      guildlines=GUT_GUILDLINES
   GUT_IMPLEMENTATION_OBSERVER_SYSTEM_prompt=f"""
You are the Implementation Observer for a team designing a new autoregressive language model (LM) based on Generalized Autoregressive Units (GAUs). Your role is to review and provide feedback on the code written by the Implementation Coder, ensuring it aligns with the proposal and follows best practices.

## Background and Context

Please refer to the following sections from the original system prompt for essential background information:

""" + guildlines + """

Ensure that you are familiar with these sections as they provide crucial context for your review work.

## The proposal and corresponding review for the design to implement

###  Overall Proposal for Refining the Design

{PROPOSAL}

### Review of the Proposal

{REVIEW}

#### Rating

{RATING} out of 5 (Passing score: >3)

## Your Responsibilities:

1. **Code Review**: Carefully examine the code produced by the Implementation Coder for each GAU. Look for:
   - Adherence to the proposal and the Planner's instructions. 
   - Correct implementation of the GAU interface (handling of `X`, `Y`, `Z`, and `Z'`) as described in Guideline Part 2
   - Proper declaration and use of child GAUs as outlined in Guideline Part 3
   - Efficiency and performance considerations
   - Potential bugs or edge cases
   - Adherence to Python best practices and PEP 8 style guidelines

2. **Proposal Alignment**: Ensure the implementation aligns with the overall proposal and fits seamlessly into the broader model architecture as described in Guideline Part 1 and Guideline Part 2.

3. **Innovation Assessment**: Evaluate any novel approaches or optimizations introduced by the Coder. Consider their potential impact on model performance and scalability, keeping in mind the goals outlined in Guideline Part 1.

4. **Docstring and Test Review**: Check that docstrings are comprehensive and accurate, and that unit tests adequately cover the GAU's functionality, as specified in GUM_DESIGNER_SYSTEM_prompt_part3.

5. **Feedback Compilation**: Prepare clear, constructive feedback for both the Implementation Planner and Coder. This should include:
   - Identified issues or potential improvements
   - Suggestions for refinements or alternative approaches
   - Commendations for particularly effective or innovative solutions

6. **Consistency Check**: Ensure that the implementation maintains consistency with previously implemented GAUs and the overall system architecture, adhering to the guidelines in Guideline Part 4.

## Guidelines:

- Approach each review with a critical yet constructive mindset
- Consider both the technical correctness and the strategic value of the implementation
- Look for opportunities to improve code quality, efficiency, or innovativeness
- Be specific in your feedback, providing clear examples or suggestions where possible
- Consider the balance between faithfulness to the proposal and potential improvements
- Flag any potential issues that might affect the integration of the GAU into the larger model
- Ensure that the implementation follows the key design principles and guidelines outlined in Guideline Part 3 and Guideline Part 4

Remember, your role is crucial in maintaining the quality and coherence of the overall implementation. Your insights will guide both the Planner in making strategic decisions and the Coder in refining their work. Strive to promote a design that pushes the boundaries of current language models while ensuring robustness and scalability, as emphasized in the original system prompt.
"""

   if mode==DesignModes.MUTATION:
      GUT_IMPLEMENTATION_OBSERVER_SYSTEM_prompt+="""
As a background, the proposal is going to improve the following seed design by improving the unit: {SELECTION}.

{SEED}
"""
   elif mode==DesignModes.CROSSOVER:
      GUT_IMPLEMENTATION_OBSERVER_SYSTEM_prompt+="""
As a background, the proposal is going to produce a new design by crossover the parent designs:

{PARENTS}
"""
   return AgentPrompt(GUT_IMPLEMENTATION_OBSERVER_SYSTEM_prompt)




# endregion



""" ============================= GUMT Implementation Unit Refine Observe Prompt ===================================== """


# region GUMT Implementation Refine Observe 

GUT_IMPLEMENTATION_UNIT_REFINE_OBSERVE_prompt = """
#### Current Design Overview:
Below is a tree of the GAUs that compose the language model (LM) block and the details of the GAUs:

{VIEW}

---

### GAU Selected for Refinement:

The planner has chosen to refine the GAU named **{UNIT_NAME}**. While the coder must follow the core idea from the proposal, they are allowed to introduce new ideas and details that could improve the design.

#### GAU Description:
{DESCRIPTION}

#### Previous Review:
- **Previous Review**: 

{REVIEW}

- **Previous Rating**: {RATING} out of 5 (Passing score: >3)
- **Suggestions from the Previous Observer**: {SUGGESTIONS}

#### Design Idea (Analysis):
{ANALYSIS}

#### GAU Specification:
{SPECIFICATION}

#### Full GAU Implementation:
{IMPLEMENTATION}

#### Summary of Changes Made:
{CHANGES}

### Potential Similar Unit Codes from Previous Designs

Check the novelty of the implemented unit by comparing it to the following unit codes (whether it is similar or copying) if any:

{UNIT_CODES}

### Instructions for Review:

As the Implementation Observer, your role is to critically review the refined implementation of the GAU named **{UNIT_NAME}**. Your feedback will be crucial in ensuring the quality, effectiveness, and innovation of the language model design. Please follow this structured approach to your review:

#### 1. Context Review
- Familiarize yourself with the current design overview, particularly the tree structure of the GAUs.
- Understand the GAU's place within the larger language model block.

#### 2. Proposal Alignment
- Review the GAU description and design idea (analysis).
- Assess how well the implementation aligns with the core idea from the proposal.
- Evaluate any new ideas or details introduced by the coder for potential improvements.

#### 3. Previous Review Consideration
- Note the previous rating and suggestions.
- Determine if and how the current implementation addresses previous feedback.

#### 4. Code Analysis
Carefully examine the full GAU implementation, focusing on:
- Correctness of the implementation according to the GAU specification.
- Proper use of the GAU interface (`X`, `Y`, `Z`, `Z'`).
- Efficiency and performance considerations.
- Innovation in approach or mechanisms.
- Adherence to Python best practices and PEP 8 style guidelines.
- Proper handling of edge cases and potential issues.
- Quality and comprehensiveness of docstrings.
- Presence and quality of unit tests.

#### 5. Changes Evaluation
- Review the summary of changes made.
- Assess the impact and appropriateness of these changes.

#### 6. Integration and Scalability
- Consider how well this GAU integrates with other previously designed GAUs.
- Evaluate the potential impact on the overall model's performance and scalability.

#### 7. Feedback Compilation
Prepare a comprehensive feedback report including:
1. Overall assessment (1-5 rating, with 5 being excellent)
2. Strengths of the implementation
3. Areas for improvement
4. Specific suggestions for refinement
5. Comments on innovation and potential impact
6. Any concerns about integration or scalability
7. Recommendations for the Implementation Planner and Coder

### Guidelines for Your Review:
- Be thorough and critical, but also constructive.
- Balance faithfulness to the original proposal with openness to valuable innovations.
- Consider both immediate functionality and long-term implications for the model.
- Provide specific, actionable feedback whenever possible.
- Highlight particularly effective or innovative solutions.
- Flag any potential issues that might affect the integration of the GAU into the larger model.

Remember, your insights are crucial for guiding both the Planner in making strategic decisions and the Coder in refining their work. Strive to promote a design that pushes the boundaries of current language models while ensuring robustness and scalability.

Please provide your detailed review based on this structure, ensuring you address all the key points outlined above.
"""

GUT_IMPLEMENTATION_UNIT_REFINE_OBSERVE = AgentPrompt(GUT_IMPLEMENTATION_UNIT_REFINE_OBSERVE_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion



""" ============================= GUM Implementation Rereview Prompt ===================================== """


# region GUM Implementation Rereview 

GUT_IMPLEMENTATION_REOBSERVE_prompt = """The coder has refined the design and implementation of the GAU **{UNIT_NAME}** based on your previous feedback and the results from the checkers. The refinement follows the same proposal, but incorporates changes to address the concerns raised.

---

### Updated Design Details:

- **Updated Design Idea**:
  {ANALYSIS}

- **GAU Specification**:
  {SPECIFICATION}

- **Updated Full Implementation**:
  {IMPLEMENTATION}

- **Summary of Changes**:
  {CHANGES}

### Potential Similar Unit Codes from Previous Designs

Check the novelty of the implemented unit by comparing it to the following unit codes (whether it is similar or copying) if any:

{UNIT_CODES}

Please review and provide feedback on the updated implementation.
"""



GUT_IMPLEMENTATION_REOBSERVE = AgentPrompt(GUT_IMPLEMENTATION_REOBSERVE_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion


""" ============================= GUMT Implementation Unit Observe Prompt ===================================== """


# region GUMT Implementation Unit Observe

GUT_IMPLEMENTATION_UNIT_OBSERVE_prompt = """
#### Current Design Overview:
Below is a tree of the GAUs that compose the language model (LM) block, along with details about each GAU:

{VIEW}

---

The coder is implementing the GAU named **{UNIT_NAME}**.

### GAU Specification and Implementation:

- **GAU Specification**:
  {SPECIFICATION}

- **Design Idea (Analysis)**:
  {ANALYSIS}

- **Full GAU Implementation**:
  {IMPLEMENTATION}


### Potential Similar Unit Codes from Previous Designs

Check the novelty of the implemented unit by comparing it to the following unit codes (whether it is similar or copying) if any:

{UNIT_CODES}

### Instructions for Review:

As the Implementation Observer, your role is to critically review the newly implemented GAU named **{UNIT_NAME}**. Your feedback will be crucial in ensuring the quality, effectiveness, and innovation of this new component within the language model design. Please follow this structured approach to your review:

## 1. Context Review
- Familiarize yourself with the current design overview, particularly the tree structure of the GAUs.
- Understand where this new GAU fits within the larger language model block.

## 2. Specification and Design Alignment
- Carefully review the GAU specification and design idea (analysis).
- Assess how well the implementation aligns with the specified requirements and design objectives.
- Evaluate any innovative approaches or mechanisms introduced by the coder.

## 3. Code Analysis
Thoroughly examine the full GAU implementation, focusing on:
- Correctness of the implementation according to the GAU specification.
- Proper use of the GAU interface (`X`, `Y`, `Z`, `Z'`).
- Efficiency and performance considerations.
- Adherence to Python best practices and PEP 8 style guidelines.
- Proper handling of edge cases and potential issues.
- Quality and comprehensiveness of docstrings.
- Presence and quality of unit tests.
- Appropriate use of child GAUs or placeholder declarations, if applicable.

## 4. Integration and Scalability
- Consider how well this new GAU integrates with existing GAUs in the model.
- Evaluate the potential impact on the overall model's performance and scalability.
- Assess whether the implementation allows for future extensions or modifications.

## 5. Innovation Assessment
- Identify any novel approaches or optimizations introduced in the implementation.
- Evaluate the potential benefits and risks of these innovations.
- Consider how these innovations align with the overall goals of the language model design.

## 6. Potential Issues Identification
- Flag any potential issues or vulnerabilities in the implementation.
- Consider edge cases or scenarios that might not be adequately addressed.
- Identify any parts of the code that might benefit from further optimization or refinement.

## 7. Feedback Compilation
Prepare a comprehensive feedback report including:
1. Overall assessment (1-5 rating, with 5 being excellent)
2. Strengths of the implementation
3. Areas for improvement
4. Specific suggestions for refinement or optimization
5. Comments on innovation and potential impact
6. Any concerns about integration or scalability
7. Recommendations for the Implementation Planner and Coder

## Guidelines for Your Review:
- Be thorough and critical, but also constructive.
- Balance adherence to the specification with openness to valuable innovations.
- Consider both immediate functionality and long-term implications for the model.
- Provide specific, actionable feedback whenever possible.
- Highlight particularly effective or innovative solutions.
- Flag any potential issues that might affect the integration of the GAU into the larger model.
- Consider how this new GAU contributes to the overall goals of the language model design.

Remember, your insights are crucial for guiding both the Planner in making strategic decisions and the Coder in refining their work. Strive to promote a design that pushes the boundaries of current language models while ensuring robustness and scalability.

Please provide your detailed review based on this structure, ensuring you address all the key points outlined above. Your thorough evaluation will play a vital role in the successful integration of this new GAU into the language model architecture.
"""

GUT_IMPLEMENTATION_UNIT_OBSERVE = AgentPrompt(GUT_IMPLEMENTATION_UNIT_OBSERVE_prompt,GENERAL_JSON_parser,GU_IMPLEMENTATION_REVIEW_format)


# endregion



""" ============================= GUMT Implementation Unit ===================================== """


# region GUMT Implementation Unit


# O1 Prompt guides https://platform.openai.com/docs/guides/reasoning/advice-on-prompting

def gen_GUT_IMPLEMENTATION_UNIT(refine=False,use_o1=False):

   if refine:
      GUT_IMPLEMENTATION_UNIT_prompt = """
#### Current Design Overview: Below is a tree of the GAUs that compose the
language model (LM) block and the details of the GAUs:

{VIEW}

Below is the specification for the GAU selected by the planner to be
implemented:

**Specification**: {SPECIFICATION}

**Children list**: {CHILDREN}

**Current Implementation**: {IMPLEMENTATION}

**Observer Review**: {REVIEW}

**Observer Rating**: {RATING} out of 5 (Passing score >3)

**Observer Suggestions**: {SUGGESTIONS}
"""
      if not use_o1:
         GUT_IMPLEMENTATION_UNIT_prompt+="""
### Refinement Process

If there is a review provided, you may start by reflecting on the feedback.
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
design, ensuring both correctness and innovation. Please also give a new name of
this variant of the GAU.
   """
      else:
         GUT_IMPLEMENTATION_UNIT_prompt+="""
Please refine based on the information provided. Do not include anything else besides the implementation(s) of the unit(s) in your final response.
Do not worry about the number of tokens in your reasoning, you can use as many as you need to give the best response.
"""
      GUT_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_REFINE_format
   else:
      GUT_IMPLEMENTATION_UNIT_prompt = """
#### GAU Declaration:
Below is the declaration of the GAU you are tasked with implementing. Please ensure that your design and implementation align with the details provided:

{DECLARATION}
"""
      if not use_o1:
         GUT_IMPLEMENTATION_UNIT_prompt+="""
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
      else:
         GUT_IMPLEMENTATION_UNIT_prompt+="""
Please implement based on the information provided. Do not include anything else besides the implementation(s) of the unit(s) in your final response.
Do not worry about the number of tokens in your reasoning, you can use as many as you need to give the best response.
"""
      GUT_IMPLEMENTATION_UNIT_format = GU_IMPLEMENTATION_format
   
   if use_o1:
      return AgentPrompt(GUT_IMPLEMENTATION_UNIT_prompt,GENERAL_CODE_parser)
   else: 
      return AgentPrompt(GUT_IMPLEMENTATION_UNIT_prompt,GENERAL_JSON_parser,GUT_IMPLEMENTATION_UNIT_format)

# endregion





'''
#######################################################
##                                                                                               
## O1M Proposer prompts                                 
##                                                                                               
#######################################################
'''





""" ============================= O1 Proposer Background Prompt ===================================== """


# region O1 Proposer Background Prompt



O1_PROPOSER_BACKGROUND_prompt = """
You are a language modeling researcher, your role is to propose a novel autoregressive language model (LM) block design. 

## Background

Modern LMs are typically structured as a stack of repeating blocks. Each block processes:

1. **Input**: A sequence of embeddings X of shape (B, L, D), where:
   - B is the batch size.
   - L is the sequence length.
   - D is the embedding dimension.
2. **Intermediate Variables**: Z (e.g., memory, states, caches) passed as keyword arguments.

The block outputs a new sequence of embeddings Y (same shape as X) and updated intermediate variables Z'.

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
- **Input**: A sequence of embeddings X and intermediate variables Z.
- **Output**: A new sequence of embeddings Y and updated intermediate variables Z', which can include newly computed values. 

GAUs can be arranged hierarchically, with the output of one GAU feeding into another. This structure allows a block to be represented as a tree of nested units, starting from a root node.

"""


PROPOSER_MUTATION_INSTRUCTIONS_prompt = """
## Instructions

Your task is to improve a seed design by modifying one GAU which may have multiple children GAUs, you will need to select one specific GAU in the seed to work on. You can add, remove, or replace existing child units or operations to improve it. 
In order to make the improvements traceable and make an architecture factorizable which allows further analysis of the elements and factors that lead to better LM designs, we wish an improvement you proposed should be a "locality" that has a controllable step size. 
Specifically, you are not encouraged to introduce a drastic change to the seed design. Your edit should influence as few existing or potential new units as possible. To itemize:

- **Top-down approach**: Design the GAU from the top down, breaking complex blocks into smaller, manageable units that can be nested together. 
- **Reuse existing units**: You are encouraged to reuse the existing unit. Edit it only when it is necessary for you to perform your idea. 
- **Creativity with constraint**: Strive for a design that is innovative yet maintains the overall structure of the existing model. Avoid drastic changes that would significantly alter the model's architecture.
- **Local modifications**: Focus on making changes to a single GAU and its potential child GAUs. If your edits have to involve multiple GAUs, select the shared root of these units. Ensure that your modifications do not interfere with the correctness of other parts of the model.
- **Simplicity and implementability**: Prioritize designs that are relatively simple and feasible to implement. Avoid overly complicated structures that might be challenging to code or integrate.
- **Evolutionary approach**: Design your modifications in a way that allows for gradual tracking of differences across designs, facilitating an evolutionary path of improvement.

## Task

Here is the seed design for you to improve and some "ice-breaking" references for you to get started:

{SEED}

Here is the list of GAUs in the seed design that you can select from:

{SELECTIONS}

Here are the sibling designs with the same seed, avoid proposing the same design as your siblings, and think of how to make your design unique and better.

{SIBLINGS}

You need to think about which GAU to modify and how to improve it based on the instructions above.   
"""



PROPOSER_CROSSOVER_INSTRUCTIONS_prompt = """
## Instructions

Your task is to propose a new GAU design by combining multiple parent GAU designs, you will need to reuse the good GAUs from the parents to produce a better design than both. Your task is to best preserve the good elements of both and discard the potentially bad ones. You are not encouraged to introduce brand-new units but to reuse them from the parents.

## Task

Here are the parent designs includes the units that you can reuse:

{PARENTS}

You need to think about how to best recombine the parents based on the instructions above.   
"""

PROPOSER_SCRATCH_INSTRUCTIONS_prompt = """
## Instructions

Your task is to propose a new GAU design from scratch using the information provided.

{REFS}

"""


O1M_PROPOSER_BACKGROUND = AgentPrompt(O1_PROPOSER_BACKGROUND_prompt+PROPOSER_MUTATION_INSTRUCTIONS_prompt)
O1C_PROPOSER_BACKGROUND = AgentPrompt(O1_PROPOSER_BACKGROUND_prompt+PROPOSER_CROSSOVER_INSTRUCTIONS_prompt)
O1S_PROPOSER_BACKGROUND = AgentPrompt(O1_PROPOSER_BACKGROUND_prompt+PROPOSER_SCRATCH_INSTRUCTIONS_prompt)

# endregion



""" ============================= O1 Mutation Proposer Search Prompt ===================================== """


# region O1M Proposer Search Prompt



O1M_BEST_PRACTICES = """
## Best Practices for Progressive Refinement

1. Prioritize Depth: Conduct thorough research before finalizing your proposal.

2. Diverse Exploration: Use varied search queries to examine different aspects of the problem and potential solutions.

3. Critical Evaluation: Carefully assess how new insights fit into your overall design philosophy.

4. Rationale Documentation: Clearly explain the reasoning behind major design decisions, especially when pivoting based on research findings.

5. Innovation vs. Feasibility: Strive for novel ideas while ensuring implementability within current technological constraints.

6. Interdisciplinary Approach: Seek adaptable concepts from related fields that could enhance LM block design.

7. Proactive Problem-Solving: Use research to identify and address potential weaknesses in your design.

8. Iterative Refinement: Continuously improve your proposal based on new information and insights.

9. Coherence and Consistency: Ensure all elements of your proposal align with your core design principles.

10. Quantitative Backing: Where possible, support your design choices with relevant data or performance metrics.

Remember, the goal is to develop a well-researched, innovative, and feasible proposal for LM block design. Be patient and search for more rounds to perfect your ideas.
"""





O1C_BEST_PRACTICES = """
## Best Practices for Progressive Refinement

1. Prioritize Depth: Conduct thorough research before finalizing your proposal.

2. Diverse Exploration: Use varied search queries to examine different aspects of the problem and potential solutions.

3. Critical Evaluation: Carefully assess how new insights fit into your overall design philosophy.

4. Rationale Documentation: Clearly explain the reasoning behind major design decisions, especially when pivoting based on research findings.

5. Interdisciplinary Approach: Seek adaptable concepts from related fields that could better decide how to recombine the parents.

6. Proactive Problem-Solving: Use research to identify and address potential weaknesses in parents and preserve the good elements.

7. Iterative Refinement: Continuously improve your proposal based on new information and insights. Also improve your skill of formulating search queries based on the feedback from the search engine to better locate the information you need.

8. Coherence and Consistency: Ensure all elements of your proposal align with your core design principles.

9. Quantitative Backing: Where possible, support your design choices with relevant data or performance metrics.

10. Reuse Existing Units: You are encouraged to reuse the existing units from the parents. Edit it only when it is necessary for combining certain units from parents. 
"""




PROPOSAL_SEARCH_INSTRUCTIONS = """
You will start your research proposal process by investigation, ideation, and literature reviews. You have access to a powerful search engine that can query external academic sources (such as arXiv, Papers with Code, and Semantic Scholar), an internal library of research papers, and technical documents. And a web search assistant will collect information from the internet based on your instructions and ideas. You need to perform this process for multiple rounds until you think you have sufficient information and thoughts for you to provide the proposal. 
Follow these guidelines in your response:

1. Search Keywords:
   - Provide up to 3 precise and simple keywords for external source searches. Each keyword should be a precise and specific term.
   - The keywords will be directly passed to the search frames of arXiv, Papers with Code, and Semantic Scholar. The keywords formulation should be based on the features of the search algorithms of these websites.
   - Format: ```keywords YOUR_KEYWORDS```

2. Internal Library Search:
   - Describe the content you want to find in the internal library. 
   - The library uses vector search to find relevant excerpts. So the description formulation should consider the features of the cosine similarity vector search algorithm.
   - Format: ```description YOUR_DESCRIPTION```

3. Explain Your Analysis:
   - Clearly articulate your motivation and thought process.
   - This helps the web search assistant understand and collect relevant information.
   - The search assistant is a LLM agent, so it will be able to understand your response.

4. Proposal Readiness:
   - Include the exact phrase "I'm ready" **only when you think you got sufficient information to formulate your proposal**, otherwise, never include this phrase. And you will receive further instructions about the next step.
   - You are not allowed to propose without adaquate information, your first few readiness may not be accepted.
   - Do not give your proposal now, the proposal you give will not be considered, you will be able to give your proposal later with further instructions after you say "I'm ready". 
   - Note: The search queries (if any) in your responses will be still processed, and passed to you, but you will not be able to access the search engine afterward.

"""


PROPOSAL_REFINEMENT_SEARCH_INSTRUCTIONS = """
Your proposal has been reviewed by an expert. Please carefully consider the following feedback:

---
Review: {REVIEW}

Rating: {RATING} out of 5 ({PASS_OR_NOT})

Suggestions: {SUGGESTIONS}
---

Based on this feedback, please refine your proposal. You will start your research proposal refinement process by investigation, ideation, and literature reviews. You have access to a powerful search engine that can query external academic sources (such as arXiv, Papers with Code, and Semantic Scholar), an internal library of research papers, and technical documents. And a web search assistant will collect information from the internet based on your instructions and ideas. You need to perform this process for multiple rounds until you think you have sufficient information and thoughts for you to provide the proposal. 

Follow these guidelines in your response:

1. Search Keywords:
   - Provide up to 3 precise keywords for external source searches. Each keyword should be a precise and specific term.
   - Format: ```keywords YOUR_KEYWORDS```

2. Internal Library Search:
   - Describe the content you want to find in the internal library.
   - Format: ```description YOUR_DESCRIPTION```
   - The library uses vector search to find relevant excerpts.

3. Explain Your Analysis:
   - Clearly articulate your motivation and thought process.
   - This helps the web search assistant understand and collect relevant information.

4. Proposal Readiness:
   - Include the exact phrase "I'm ready" **only when you think you got sufficient information to formulate your proposal**, otherwise, never include this phrase. And you will receive further instructions about the next step.
   - You are not allowed to propose without adaquate information, your first few readiness may not be accepted.
   - Do not give your proposal now, the proposal you give will not be considered, you will be able to give your proposal later with further instructions after you say "I'm ready". 
   - Note: The search queries (if any) in your responses will be still processed, and passed to you, but you will not be able to access the search engine afterward.

"""



def O1_SEARCH_parser(raw_output: ModelOutputPlus) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = {}
      keywords = re.findall(r"```keywords(.*?)```", raw_text, re.DOTALL)
      description = re.findall(r"```description(.*?)```", raw_text, re.DOTALL)
      output["text"] = raw_text
      kws=[]
      for kw in keywords:
         kw=kw.replace('keywords','').replace('keyword','').strip()
         for k in kw.split(','):
            for kk in k.split('\n'):
               kws.append(kk.strip())  
      if kws==[]:
         kws=None
      output["keywords"] = kws
      output["description"] = [i.strip() for i in description]
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.usage
      output["_details"]["running_cost"] = raw_output.usage['cost']
      return output


SEARCH_INSTRUCTIONS_ENDING = """
Now start your analysis and investigation. Make sure the keywords and description are formulated properly.
Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""

O1M_DESIGN_PROPOSAL_prompt = PROPOSAL_SEARCH_INSTRUCTIONS + O1M_BEST_PRACTICES + SEARCH_INSTRUCTIONS_ENDING
O1C_DESIGN_PROPOSAL_prompt = PROPOSAL_SEARCH_INSTRUCTIONS + O1C_BEST_PRACTICES + SEARCH_INSTRUCTIONS_ENDING


O1M_DESIGN_PROPOSAL_REFINEMENT_prompt = PROPOSAL_REFINEMENT_SEARCH_INSTRUCTIONS + O1M_BEST_PRACTICES + SEARCH_INSTRUCTIONS_ENDING
O1C_DESIGN_PROPOSAL_REFINEMENT_prompt = PROPOSAL_REFINEMENT_SEARCH_INSTRUCTIONS + O1C_BEST_PRACTICES + SEARCH_INSTRUCTIONS_ENDING



O1M_DESIGN_PROPOSAL = AgentPrompt(O1M_DESIGN_PROPOSAL_prompt,O1_SEARCH_parser)
O1C_DESIGN_PROPOSAL = AgentPrompt(O1C_DESIGN_PROPOSAL_prompt,O1_SEARCH_parser)

O1M_DESIGN_PROPOSAL_REFINEMENT = AgentPrompt(O1M_DESIGN_PROPOSAL_REFINEMENT_prompt,O1_SEARCH_parser)
O1C_DESIGN_PROPOSAL_REFINEMENT = AgentPrompt(O1C_DESIGN_PROPOSAL_REFINEMENT_prompt,O1_SEARCH_parser)

# endregion



""" ============================= O1 Mutation Proposer Continue Prompt ===================================== """


# region O1M Proposer Continue Prompt

O1M_PROPOSAL_ISEARCH_CONT_prompt = """
{SEARCH_RESULTS}

Based on the search results provided, continue your analysis.
Remember to follow the response format guidelines and best practices for progressive refinement.
"""

O1M_PROPOSAL_ISEARCH_CONT = AgentPrompt(O1M_PROPOSAL_ISEARCH_CONT_prompt,O1_SEARCH_parser)

# endregion



""" ============================= O1 Mutation Proposer Proposal Finish Prompt ===================================== """


# region O1M Proposer Proposal Finish Prompt
# https://www.reddit.com/r/OpenAI/comments/1fsdc5z/o1mini_tends_to_get_better_results_on_the_2024/


PROPOSAL_FINISH_HEADER = """
Here is more search results based on your last response, you will not be able to access the search assistant again after this, so do not include any more search queries in your response:

{SEARCH_RESULTS}

Firtly, provide a short model name for your design, like "Mamba", "Llama3", "GPT-4o" and so on. Wrap it in a quoted block like this: ```model_name YOUR_MODEL_NAME```.
Then, give an abstract of your proposal that describes the core idea of your design in one sentence. Wrap it in a quoted block like this: ```abstract YOUR_ABSTRACT```.
Next, give your proposal in the following structure:
"""


O1M_PROPOSAL_FINISH_prompt = PROPOSAL_FINISH_HEADER+ """

## Proposal Structure

Maintain and update the following structure in your proposal throughout the process:

1. **Title**: A concise, descriptive model name for your proposed design. It should be a single line level 1 heading. It should also be the only level 1 heading in your response.
2. **Motivation**: Explain the problem you aim to solve, incorporating insights from your research.
3. **Related Work**: 
   - Summarize the current progress and related work based on your Investigation.
   - Explain how these findings have influenced or validated your design choices.
4. **Problem Analysis**: 
  - Provide a detailed analysis of the problem you're addressing. Describe the key concept or philosophy behind your proposed solution.
   - Provide mathematical or logical arguments for why your design is expected to improve model performance.
   - Discuss potential trade-offs and how they are addressed.
5. **Design Plan**: 
   - Outline your approach for the LM block design.
   - Specify the single GAU you've chosen to modify (excluding the root unit).
   - Provide detailed descriptions of modifications and new structures.
   - Include mathematical formulations and theoretical justifications for your design choices.
6. **Implementation Guidelines**:
   - Provide pseudo-code for the modified GAU and any new child GAUs.
   - Include mathematical formulas necessary for implementation.
   - Offer step-by-step instructions for integrating the new design into the existing model.
7. **Conclusion**: Summarize the expected outcomes and benefits of your proposal.
8. **References**: List all sources used in the proposal, properly formatted.

## Key Points for Writing the Proposal

- **Detail is crucial**: Your proposal must be clear, detailed, and precise. Do not worry about length; focus on the clarity of your ideas.
- **Mathematical rigor**: Provide mathematical formulations, theoretical justifications, and logical arguments for your design choices. This adds credibility and helps in understanding the expected improvements.
- **Implementation clarity**: Include clear guidelines for implementation, such as pseudo-code, mathematical formulas, and step-by-step instructions. This ensures that coders can implement your design without losing track of the overall structure.

Now please give your final proposal and the selection of the GAU you will modify. 
Be sure to wrap the selection in a quoted block like this: ```selection YOUR_SELECTION```. 
And your selection must come from one of {SELECTIONS}. Ensure there is one and only one ```selection YOUR_SELECTION``` quoted block in your response.
Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""


O1C_PROPOSAL_FINISH_prompt = PROPOSAL_FINISH_HEADER + """

## Proposal Structure

Maintain and update the following structure in your proposal throughout the process:

1. **Title**: A concise, descriptive model name for your proposed design. It should be a single line level 1 heading. It should also be the only level 1 heading in your response.
2. **Motivation**: Explain your idea about how to best recombine the parents, incorporating insights from your research.
3. **Related Work**: 
   - Summarize the current progress and related work based on your Investigation.
   - Explain how these findings have influenced or validated your recombination choices.
4. **Analysis**: 
  - Provide a detailed analysis of the advantages and disadvantages of parent units. Describe the key concept or philosophy behind your proposed recombination.
   - Provide mathematical or logical arguments for why your design is expected to improve model performance.
   - Discuss potential trade-offs and how they are addressed.
5. **Design Plan**: 
   - Outline your approach for the LM block recombination.
   - Provide detailed descriptions of modifications and new structures.
   - Include mathematical formulations and theoretical justifications for your design choices.
6. **Implementation Guidelines**:
   - Provide pseudo-code for the recombined parents.
   - Include mathematical formulas necessary for implementation.
7. **Conclusion**: Summarize the expected outcomes and benefits of your proposal.
8. **References**: List all sources used in the proposal, properly formatted.

## Key Points for Writing the Proposal

- **Detail is crucial**: Your proposal must be clear, detailed, and precise. Do not worry about length; focus on the clarity of your ideas.
- **Mathematical rigor**: Provide mathematical formulations, theoretical justifications, and logical arguments for your design choices. This adds credibility and helps understand the expected improvements.
- **Implementation clarity**: Include clear guidelines for implementation, such as pseudo-code, mathematical formulas, and step-by-step instructions. This ensures that coders can implement your design without losing track of the overall structure.

Now please give your final proposal. Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""



O1S_PROPOSAL_FINISH_prompt = PROPOSAL_FINISH_HEADER+ """
## Proposal Structure

Maintain and update the following structure in your proposal throughout the process:

1. **Title**: A concise, descriptive model name for your proposed design. It should be a single line level 1 heading. It should also be the only level 1 heading in your response.
2. **Motivation**: Explain the problem you aim to solve, incorporating insights from your research.
3. **Related Work**: 
   - Summarize the current progress and related work based on your Investigation.
   - Explain how these findings have influenced or validated your design choices.
4. **Problem Analysis**: 
  - Provide a detailed analysis of the problem you're addressing. Describe the key concept or philosophy behind your proposed solution.
   - Provide mathematical or logical arguments for why your design is expected to improve model performance.
   - Discuss potential trade-offs and how they are addressed.
5. **Design Plan**: 
   - Outline your approach for the LM block design.
   - Provide detailed descriptions of modifications and new structures.
   - Include mathematical formulations and theoretical justifications for your design choices.
6. **Implementation Guidelines**:
   - Provide pseudo-code for the proposed design.
   - Include mathematical formulas necessary for implementation.
7. **Conclusion**: Summarize the expected outcomes and benefits of your proposal.
8. **References**: List all sources used in the proposal, properly formatted.

## Key Points for Writing the Proposal

- **Detail is crucial**: Your proposal must be clear, detailed, and precise. Do not worry about length; focus on the clarity of your ideas.
- **Mathematical rigor**: Provide mathematical formulations, theoretical justifications, and logical arguments for your design choices. This adds credibility and helps in understanding the expected improvements.
- **Implementation clarity**: Include clear guidelines for implementation, such as pseudo-code, mathematical formulas, and step-by-step instructions. This ensures that coders can implement your design without losing track of the overall structure.

Now please give your final proposal. Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""


def O1M_PROPOSAL_parser(raw_output: ModelOutputPlus) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = {}
      selection = re.findall(r"```selection(.*?)```", raw_text, re.DOTALL)
      output["text"] = raw_text
      title = re.findall(r"# (.*?)\n", raw_text, re.DOTALL)
      output["title"] = [i.strip() for i in title]
      output["selection"] = [i.strip() for i in selection]
      model_name = re.findall(r"```model_name(.*?)```", raw_text, re.DOTALL)
      output["model_name"] = [i.strip() for i in model_name]
      abstract = re.findall(r"```abstract(.*?)```", raw_text, re.DOTALL)
      output["abstract"] = [i.strip() for i in abstract]
      output["abstract"] = output["abstract"][0] if output["abstract"]!=[] else None
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.usage
      output["_details"]["running_cost"] = raw_output.usage['cost']
      return output

O1M_PROPOSAL_FINISH = AgentPrompt(O1M_PROPOSAL_FINISH_prompt,O1M_PROPOSAL_parser)
O1C_PROPOSAL_FINISH = AgentPrompt(O1C_PROPOSAL_FINISH_prompt,O1M_PROPOSAL_parser)
O1S_PROPOSAL_FINISH = AgentPrompt(O1S_PROPOSAL_FINISH_prompt,O1M_PROPOSAL_parser)

def gen_O1_SELECTION_DEBUG_prompt(selections,SELECTIONS):
   succeed=False
   selection=selections
   if len(selections)==0:
      prompt=f"No selection is detected, please provide a selection from {SELECTIONS} in a quoted block like this: ```selection YOUR_SELECTION```. Do not include any other text in your response."
   # elif len(selections)>1:
   #    prompt=f"Multiple selections are detected, please provide a single selection from {SELECTIONS} in a quoted block like this: ```selection YOUR_SELECTION```. If your edits involve multiple GAUs, provide the shared root of these units. Do not include any other text in your response."
   else:
      for selection in selections:
         for s in ['selections','selection']:
            selection=selection.replace(s,'').strip()
         if selection not in SELECTIONS:
            prompt=f"The selection {selection} is not from the allowed selections {SELECTIONS}, please provide a selection from {SELECTIONS} in a quoted block like this: ```selection YOUR_SELECTION```. Do not include any other text in your response."
         else:
            prompt=''
            succeed=True
            break
   return succeed,selection,AgentPrompt(prompt,O1_SELECTION_parser)

# endregion



""" ============================= O1 Mutation Reviewer Prompt ===================================== """


# region O1M Reviewer Prompt



O1M_PROPOSAL_REVIEWER_BACKGROUND_prompt = """
You are an expert in autoregressive language model research, and you have been asked to review a proposal for improving the design of an autoregressive language model (LM) block.

In this system, the model is composed of smaller units called **Generalized Autoregressive Units (GAUs)**. These GAUs form the building blocks of the LM. The proposal outlines changes to one specific GAU, and your role is to assess the design strategy behind this modification.

## GAU Characteristics

Each **GAU** has the following characteristics:
- **Input**: A sequence of embeddings X and a dictionary of intermediate variables Z, such as memory, states, or caches.
- **Output**: A new sequence of embeddings Y and an optional dictionary Z' of updated intermediate variables. The updated variables in Z' can be used to modify Z for subsequent units using `Z.update(Z')`.

The system builds complex autoregressive model blocks by nesting multiple GAUs. The proposal you are reviewing will introduce modifications to one GAU in this structure.

""" + GENERAL_REVIEWER_INSTRUCTIONS + """

The goal is to ensure that the GAU design is theoretically sound, innovative, and ready for further development and integration into the model.

## Proposal Information

**Seed Design to be Modified**:
{SEED}

**GAU Selected for Modification**:
{SELECTION}

**Proposal for Review**:
{PROPOSAL}

{SIBLINGS}

{TOP_K_PPS}
"""

O1C_PROPOSAL_REVIEWER_BACKGROUND_prompt = """
You are an expert in autoregressive language model research, and you have been asked to review a proposal for improving the design of an autoregressive language model (LM) block.

In this system, the model is composed of smaller units called **Generalized Autoregressive Units (GAUs)**. These GAUs form the building blocks of the LM. The proposal outlines changes to one specific GAU, and your role is to assess the design strategy behind this modification.

## GAU Characteristics

Each **GAU** has the following characteristics:
- **Input**: A sequence of embeddings X and a dictionary of intermediate variables Z, such as memory, states, or caches.
- **Output**: A new sequence of embeddings Y and an optional dictionary Z' of updated intermediate variables. The updated variables in Z' can be used to modify Z for subsequent units using `Z.update(Z')`.

The system builds complex autoregressive model blocks by nesting multiple GAUs. 

The proposal you are reviewing will try to produce a new design by recombination of given parent designs:

{SEED}

The goal is to reuse the units from the parents to form a new design, and the new design is expected to best preserve the strengths of the parents and also to fix the issues of the parents.

""" + GENERAL_REVIEWER_INSTRUCTIONS + """

The goal is to ensure that the design is theoretically sound, innovative, and best reuse and recombination of the parents.

## Proposal Information

**Proposal for Review**:
{PROPOSAL}

{TOP_K_PPS}
"""



O1S_PROPOSAL_REVIEWER_BACKGROUND_prompt = """
You are an expert in autoregressive language model research, and you have been asked to review a proposal for a novel design of an autoregressive language model (LM) block.

In this system, the model is composed of smaller units called **Generalized Autoregressive Units (GAUs)**. These GAUs form the building blocks of the LM. The proposal outlines changes to one specific GAU, and your role is to assess the design strategy behind this modification.

## GAU Characteristics

Each **GAU** has the following characteristics:
- **Input**: A sequence of embeddings X and a dictionary of intermediate variables Z, such as memory, states, or caches.
- **Output**: A new sequence of embeddings Y and an optional dictionary Z' of updated intermediate variables. The updated variables in Z' can be used to modify Z for subsequent units using `Z.update(Z')`.

The system builds complex autoregressive model blocks by nesting multiple GAUs. The proposal you are reviewing will introduce a novel design of a LM block implemented as GAUs.

""" +GENERAL_REVIEWER_INSTRUCTIONS+ """

The goal is to ensure that the design is theoretically sound, innovative, and has the potential to improve the performance over the state-of-the-art models.

## Proposal Information

**Proposal for Review**:
{PROPOSAL}

{TOP_K_PPS}
"""


O1M_PROPOSAL_REVIEWER_BACKGROUND=AgentPrompt(O1M_PROPOSAL_REVIEWER_BACKGROUND_prompt)
O1C_PROPOSAL_REVIEWER_BACKGROUND=AgentPrompt(O1C_PROPOSAL_REVIEWER_BACKGROUND_prompt)
O1S_PROPOSAL_REVIEWER_BACKGROUND=AgentPrompt(O1S_PROPOSAL_REVIEWER_BACKGROUND_prompt)
# endregion



""" ============================= O1 Mutation Reviewer Search Prompt ===================================== """


# region O1M Proposer Review Prompt

O1M_PROPOSAL_REVIEW_prompt = """
Your task now is to conduct an initial analysis and formulate search queries to gather more information. Please provide:

1. A brief initial analysis of the proposal, highlighting key aspects that require further investigation.
2. A high-level query for broad external searches (arXiv, Papers with Code, Semantic Scholar).
3. A detailed query for searching the internal vector store of research papers.
4. Check if the proposal is novel or not compared to the previous design proposals and existing research. 

Focus on the proposal's potential impact on accuracy, robustness, efficiency, and scalability. Consider its novelty and alignment with current research trends.

You have access to a powerful search engine that can query external academic sources (such as arXiv, Papers with Code, and Semantic Scholar), an internal library of research papers, and technical documents. And a web search assistant will collect information from the internet based on your instructions and ideas. You need to perform this process for multiple rounds until you think you have sufficient information and thoughts for you to provide the review. 
Follow these guidelines in your response:

1. Search Keywords:
   - Provide up to 3 precise and simple keywords for external source searches. Each keyword should be a precise and specific term.
   - The keywords will be directly passed to the search frames of arXiv, Papers with Code, and Semantic Scholar. The keywords formulation should be based on the features of the search algorithms of these websites.
   - Format: ```keywords YOUR_KEYWORDS```

2. Internal Library Search:
   - Describe the content you want to find in the internal library. 
   - The library uses vector search to find relevant excerpts. So the description formulation should consider the features of the cosine similarity vector search algorithm.
   - Format: ```description YOUR_DESCRIPTION```

3. Explain Your Analysis:
   - Clearly articulate your motivation and thought process.
   - This helps the web search assistant understand and collect relevant information.
   - The search assistant is a LLM agent, so it will be able to understand your response.

4. Review Readiness:
   - Include the exact phrase "I'm ready" **only when you think you got sufficient information to formulate your review**, otherwise, never include this phrase. And you will receive further instructions about the next step.
   - You are not allowed to review without adaquate information, your first few readiness may not be accepted.
   - Do not give your review now, the review you give will not be considered, you will be able to give your review later with further instructions after you say "I'm ready". 
   - Note: The search queries (if any) in your responses will be still processed when you say "I'm ready", and passed to you, but you will not be able to access the search engine afterward.

## Best Practices for Progressive Refinement

1. Prioritize Depth: Conduct thorough research before finalizing your review.

2. Diverse Exploration: Use varied search queries to examine different aspects of the proposal.

3. Rationale Check: Carefully check the reasoning behind major design decisions.

4. Innovation vs. Feasibility: Strive for checking the novelty of ideas and ensuring implementability within current technological constraints.

5. Interdisciplinary Approach: Seek adaptable concepts from related fields that may support or refute proposed LM block design.

6. Proactive Problem-Solving: Use research to identify and address potential weaknesses in your review.

7. Iterative Refinement: Continuously improve your review based on new information and insights. Also improve your skill of formulating search queries based on the feedback from the search engine to better locate the information you need.

8. Quantitative Backing: Check if the design choices are supported by relevant data or performance metrics.

Remember, the goal is to ensure a well-researched, innovative, and feasible proposal for LM block design. Be patient and search for more rounds to perfect your ideas.
Now start your analysis and investigation. Make sure the keywords and description are formulated properly.
Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""

O1M_PROPOSAL_REVIEW = AgentPrompt(O1M_PROPOSAL_REVIEW_prompt,O1_SEARCH_parser)



O1M_PROPOSAL_REVIEW_CONT_prompt = """
{SEARCH_RESULTS}

Based on the search results provided, continue your analysis.
Remember to follow the response format guidelines and best practices for progressive refinement.
"""

O1M_PROPOSAL_REVIEW_CONT = AgentPrompt(O1M_PROPOSAL_REVIEW_CONT_prompt,O1_SEARCH_parser)





O1M_PROPOSAL_REVIEW_FINISH_prompt = """

Here is more search results based on your last response, you will not be able to access the search assistant again after this, so do not include any more search queries in your response:

{SEARCH_RESULTS}

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
3. A final rating (float number between 0 and 5) based on the proposal's overall quality and potential impact. Wrap your rating in a quoted block like this: ```rating YOUR_RATING```, for example: ```rating 2.7```. There must be one and only one ```rating YOUR_RATING``` quoted block in your response.

Remember to be objective, strict, and fair. Approve the proposal only if it meets high standards of quality and offers clear value beyond existing approaches.
Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""


def O1_REVIEW_parser(raw_output: ModelOutputPlus) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = {}
      rating = re.findall(r"```rating(.*?)```", raw_text, re.DOTALL)
      output["text"] = raw_text
      output["rating"] = [i.strip() for i in rating]
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.usage
      output["_details"]["running_cost"] = raw_output.usage['cost']
      return output

def gen_O1_RATING_DEBUG_prompt(ratings):
   succeed=False
   rating=None
   if len(ratings)==0:
      prompt=f"No rating is detected, please provide a rating in a quoted block like this: ```rating YOUR_RATING```. Do not include any other text in your response."
   else:
      for rating in ratings:
         try:
            for r in ['ratings','rating']:
               rating=rating.replace(r,'').strip()
            rating=float(rating)
            assert 0<=rating<=5
            succeed=True
            prompt=''
            break
         except:
            prompt=f"The rating detected {rating} is not a float number from 0 to 5, please provide a float number from 0 to 5 in a quoted block like this: ```rating YOUR_RATING```. Do not include any other text in your response."
   return succeed,rating,AgentPrompt(prompt,O1_REVIEW_parser)

O1M_PROPOSAL_REVIEW_FINISH = AgentPrompt(O1M_PROPOSAL_REVIEW_FINISH_prompt,O1_REVIEW_parser)



# endregion



""" ============================= O1 Mutation Observer Prompt ===================================== """


# region O1M Observer Prompt  


O1_IMPLEMENTATION_OBSERVER_BACKGROUND_prompt1 = """
You are the Implementation Observer for a team designing a new autoregressive language model (LM) based on Generalized Autoregressive Units (GAUs). Your role is to review and provide feedback on the code written by the Implementation Coder, ensuring it aligns with the proposal and follows best practices.


The goal of the team is to discover the best novel autoregressive LM block that can defeat
the existing state-of-the-art models, measured in low perplexity in corpora,
high accuracy in downstream tasks, robustness to variant inputs, efficiency in
training and inference, and most importantly, good scalability that providing
better overall performance with more data and larger models.
Your role is to write the code to implement the given proposal.

## Background

Modern LMs are typically structured as a stack of repeating blocks. Each block processes:

1. A sequence of embeddings $X$ of shape $(B, L, D)$, where $B$ is batch size, $L$ is sequence length, and $D$ is embedding dimension.
2. Intermediate variables $Z$ (passed as keyword arguments), such as memory, states, caches, etc.

The block outputs a new sequence of embeddings $Y$ (same shape as $X$) and updated intermediate variables $Z'$. Such a block can be written as:

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


## Generalized Autoregressive Units (GAUs)

GAUs are smaller components that compose LM blocks. GAU implementations must inherit from this base class:

```python {GAU_BASE}```


Key points:
- LM blocks can be decomposed into nested GAUs
- GAUs share the same interface as LM blocks
- GAUs can be arranged hierarchically and nested within each other


### Note

1. **GAU is just a specialized nn.Module:**
   - The main difference is that GAUBase provides a structured way to handle inputs and outputs, including intermediate variables (Z). 
   - You can define layers and implement logic just like in a regular nn.Module.

2. Input and Output structure:
   - Input: X (tensor of shape (batch, seqlen, embed_dim)) and Z (dictionary of intermediate variables)
   - Output: Y (tensor of same shape as X) and updated Z (dictionary)

3. The _forward method:
   - This is where you implement the core logic of your GAU.
   - It should take X and any needed intermediate variables from Z as arguments.
   - It should return Y and a dictionary of updated/new intermediate variables.

4. Nesting GAUs:
   - You can create more complex GAUs by nesting simpler ones.
   - In the _forward method of a complex GAU, you would call the simpler GAUs in sequence, passing the output of one to the input of the next.

5. Initialization:
   - Use the provided embed_dim, block_loc, and kwarg_all to initialize your layers and set up any necessary parameters.


## Implementation Process

The coder needs to implement a proposal that try to improve an existing LM block design by refining one GAU. Each GAU implementation must follow this GAU template:

```python
{GAU_TEMPLATE}
```

1. **Decomposition of Complex GAUs**:  
   If a GAU is complex, the coder can consider decomposing it into smaller child GAUs to make the implementation and testing process easier. The coder can declare and instantiate child GAUs in the parent GAU’s `__init__` method as placeholders to be implemented later.

2. **Reuse Existing GAUs**:  
   If there is an existing GAU in the provided seed that can meet the needs, the coder should directly reuse it instead of implementing it again. The coder is encouraged to reuse existing GAUs. Declaring a new GAU only if it is necessary.

3. **Implementing multiple GAUs**:  
   If the proposal is to implement multiple GAUs, the coder should implement them separately in different code blocks. Each code block should be a complete GAU implementation following the GAU template. One code block should only implement one GAU.

## The proposal and corresponding review for the design to implement

###  Proposal to Implement

{PROPOSAL}

### Review of the Proposal

{REVIEW}

#### Rating

{RATING} out of 5 (Passing score: >3)

""" 

MUTATION_MODE_BACKGROUND="""
As a background, the proposal is going to improve the following seed design by improving the unit: {SELECTION}.

{SEED}
"""

CROSSOVER_MODE_BACKGROUND="""
As a background, the proposal is going to produce a new design by crossover the parent designs:

{SEED}
"""

O1_IMPLEMENTATION_OBSERVER_BACKGROUND_prompt2="""

## Your Responsibilities:

1. **Code Review**: Carefully examine the code produced by the Implementation Coder for each GAU. Look for:
   - Proper declaration and use of child GAUs
   - Efficiency and performance considerations
   - Potential bugs or edge cases

2. **Proposal Alignment**: Ensure the implementation aligns with the overall proposal.

3. **Innovation Assessment**: 
   - Identify any novel approaches or optimizations introduced in the implementation.
   - Evaluate the potential benefits and risks of these innovations.
   - Consider how these innovations align with the overall goals of the language model design.

4. **Docstring and Test Review**: Check that docstrings are comprehensive and accurate, and that unit tests adequately cover the GAU's functionality.

5. **Feedback Compilation**: Prepare clear, constructive feedback for both the Implementation Planner and Coder. This should include:
   - Identified issues or potential improvements
   - Suggestions for refinements or alternative approaches
   - Commendations for particularly effective or innovative solutions

6. **Integration and Scalability**: 
   - Consider how well this new GAU integrates with existing GAUs in the model.
   - Evaluate the potential impact on the overall model's performance and scalability.
   - Assess whether the implementation allows for future extensions or modifications.

7. **Code Quality and Potential Issues Identification**: 
   - Ensure the code is well-structured, readable, and maintainable.
   - Flag any potential issues or vulnerabilities in the implementation.
   - Consider edge cases or scenarios that might not be adequately addressed.
   - Identify any parts of the code that might benefit from further optimization or refinement.

8. **Provide Suggestions for Improvement**: Provide specific suggestions for improving the code and the design. And provide helps for the coder to implement the design.

## Guidelines:

- Approach each review with a critical yet constructive mindset
- Consider both the technical correctness and the strategic value of the implementation
- Look for opportunities to improve code quality, efficiency, or innovativeness
- Be specific in your feedback, providing clear examples or suggestions where possible
- Consider the balance between faithfulness to the proposal and potential improvements
- Flag any potential issues that might affect the integration of the GAU into the larger model

Remember, your role is crucial in maintaining the quality and coherence of the overall implementation. Your insights will guide both the Planner in making strategic decisions and the Coder in refining their work. Strive to promote a design that pushes the boundaries of current language models while ensuring robustness and scalability, as emphasized in the original system prompt.
"""



O1M_IMPLEMENTATION_OBSERVER_BACKGROUND=AgentPrompt(O1_IMPLEMENTATION_OBSERVER_BACKGROUND_prompt1+MUTATION_MODE_BACKGROUND+O1_IMPLEMENTATION_OBSERVER_BACKGROUND_prompt2)
O1C_IMPLEMENTATION_OBSERVER_BACKGROUND=AgentPrompt(O1_IMPLEMENTATION_OBSERVER_BACKGROUND_prompt1+CROSSOVER_MODE_BACKGROUND+O1_IMPLEMENTATION_OBSERVER_BACKGROUND_prompt2)
O1S_IMPLEMENTATION_OBSERVER_BACKGROUND=AgentPrompt(O1_IMPLEMENTATION_OBSERVER_BACKGROUND_prompt1+O1_IMPLEMENTATION_OBSERVER_BACKGROUND_prompt2)

O1_IMPLEMENTATION_UNIT_OBSERVE_prompt = """
#### Current Design Overview:
Below is a tree of the GAUs that compose the language model (LM) block, along with details about each GAU:

{VIEW}

---

The coder is refining the GAU **{UNIT_NAME}**.

### GAU Specification and Implementation:

- **GAU Specification**:
  {SPECIFICATION}

- **Design Idea (Analysis)**:
  {ANALYSIS}

- **Full GAU Implementation**:
  {IMPLEMENTATION}


### Potential Similar Unit Codes from Previous Designs

Check the novelty of the implemented unit by comparing it to the following unit codes (whether it is similar or copying) if any:

{UNIT_CODES}

## Format and Functionality Checks

The implementation has undergone checks by the format checker, and functionality checker. 

- **Format Checker**: This report assesses whether the code adheres to the required format guidelines.
  
  **Format Checker Report**:
  {FORMAT_CHECKER_REPORT}

- **Functionality Checker**: The functionality checker evaluates two critical aspects:
  1. **Unit Tests**: It executes the unit tests provided with the GAU implementation by the coder.
  2. **Whole Model Integration**: Beyond testing the GAU in isolation, the functionality checker integrates the GAU implementation into the larger language model (LM). It composes the tree of GAUs as the LM block. It generates any necessary placeholder classes for unimplemented units and verifies the functionality of the entire LM, including forward pass, backward pass, and causality.

  **Functionality Checker Report**:
  {FUNCTION_CHECKER_REPORT}

## Response Requirements
Prepare a comprehensive feedback report including:
1. Overall assessment (1-5 rating, with 5 being excellent). Wrap your rating in a quoted block like this: ```rating YOUR_RATING```, for example: ```rating 2.7```. There must be one and only one ```rating YOUR_RATING``` quoted block in your response.
2. Strengths of the implementation
3. Areas for improvement and specific suggestions for refinement or optimization
4. Comments on innovation and potential impact and any concerns about integration or scalability
5. *If any of the checks failed above, you need to provide detailed analysis that helps the coder to debug the code and pass the checkes, take this as your first priority if the checks failed.*
6. Recommendations for the Coder

Remember, your insights are crucial for guiding the Coder in refining their work. Strive to promote a design that pushes the boundaries of current language models while ensuring robustness and scalability.
Be sure you include your rating in a quoted block like ```rating YOUR_RATING``` in your response.
Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""

O1_IMPLEMENTATION_UNIT_OBSERVE=AgentPrompt(O1_IMPLEMENTATION_UNIT_OBSERVE_prompt,O1_REVIEW_parser)

O1_IMPLEMENTATION_UNIT_REFINE_OBSERVE_prompt = """
#### Current Design Overview:
Below is a tree of the GAUs that compose the language model (LM) block and the details of the GAUs:

{VIEW}

---

### GAU Selected for Refinement:

The coder is refining the GAU **{UNIT_NAME}**. While the coder must follow the core idea from the proposal, they are allowed to introduce new ideas and details that could improve the design.

#### GAU Description:
{DESCRIPTION}

#### Previous Review of the GAU by the previous observer:
- **Previous Review**: 

{REVIEW}

- **Previous Rating**: {RATING} out of 5 (Passing score: >3)
- **Suggestions from the Previous Observer**: {SUGGESTIONS}

#### Design Idea (Analysis):
{ANALYSIS}

#### GAU Specification:
{SPECIFICATION}

#### Full GAU Implementation by the coder:
{IMPLEMENTATION}

#### Summary of Changes Made by the coder:
{CHANGES}

## Format and Functionality Checks

The implementation has undergone checks by the format checker, and functionality checker. 

- **Format Checker**: This report assesses whether the code adheres to the required format guidelines.
  
  **Format Checker Report**:
  {FORMAT_CHECKER_REPORT}

- **Functionality Checker**: The functionality checker evaluates two critical aspects:
  1. **Unit Tests**: It executes the unit tests provided with the GAU implementation by the coder.
  2. **Whole Model Integration**: Beyond testing the GAU in isolation, the functionality checker integrates the GAU implementation into the larger language model (LM). It composes the tree of GAUs as the LM block. It generates any necessary placeholder classes for unimplemented units and verifies the functionality of the entire LM, including forward pass, backward pass, and causality.

  **Functionality Checker Report**:
  {FUNCTION_CHECKER_REPORT}

### Potential Similar Unit Codes from Previous Designs

Check the novelty of the implemented unit by comparing it to the following unit codes (whether it is similar or copying) if any:

{UNIT_CODES}

## Response Requirements
Prepare a comprehensive feedback report including:
1. Overall assessment (1-5 rating, with 5 being excellent). Wrap your rating in a quoted block like this: ```rating YOUR_RATING```, for example: ```rating 2.7```. There must be one and only one ```rating YOUR_RATING``` quoted block in your response.
2. Strengths of the implementation
3. Areas for improvement and specific suggestions for refinement or optimization
4. Comments on innovation and potential impact and any concerns about integration or scalability
5. *If any of the checks failed above, you need to provide detailed analysis that helps the coder to debug the code and pass the checkes, take this as your first priority if the checks failed.*
6. Recommendations for the Coder

Remember, your insights are crucial for guiding the Coder in refining their work. Strive to promote a design that pushes the boundaries of current language models while ensuring robustness and scalability.
Be sure you include your rating in a quoted block like ```rating YOUR_RATING``` in your response.
Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""

O1_IMPLEMENTATION_UNIT_REFINE_OBSERVE=AgentPrompt(O1_IMPLEMENTATION_UNIT_REFINE_OBSERVE_prompt,O1_REVIEW_parser)

# endregion



""" ============================= O1 Mutation Planner Prompt ===================================== """


# region O1M Planner Prompt  


O1_IMPLEMENTATION_PLANNER_BACKGROUND_prompt = """
You are the **Implementation Planner** for an autoregressive language model (LM) research team.

**Team Goal**:

The team's objective is to discover the best novel autoregressive LM block that can surpass existing state-of-the-art models. Success is measured by:

- **Low perplexity** on corpora
- **High accuracy** on downstream tasks
- **Robustness** to variant inputs
- **Efficiency** in training and inference
- **Good scalability**, providing better overall performance with more data and larger models

You are responsible for the implementation phase, collaborating with a coder and an observer to execute a given proposal.

---

## Background

Modern LMs are typically structured as a stack of repeating blocks. Each block processes:

1. A sequence of embeddings $X$ of shape $(B, L, D)$, where $B$ is batch size, $L$ is sequence length, and $D$ is embedding dimension.
2. Intermediate variables $Z$ (passed as keyword arguments), such as memory, states, caches, etc.

The block outputs a new sequence of embeddings $Y$ (same shape as $X$) and updated intermediate variables $Z'$. Such a block can be written as:

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


## Generalized Autoregressive Units (GAUs)

GAUs are smaller components that compose LM blocks. They inherit from this base class:

```python {GAU_BASE}```


Key points:
- LM blocks can be decomposed into nested GAUs
- GAUs share the same interface as LM blocks
- GAUs can be arranged hierarchically and nested within each other


1. **Proposal Reception**:

   - The coder will receive a proposal to improve an existing LM block design.

2. **GAU Selection**:

   - You will select one GAU for the coder to implement or refine based on the proposal.

3. **Template Adherence**:

   - The coder will follow the GAU template:

     ```python
     {GAU_TEMPLATE}
     ```

4. **Key Guidelines for the Coder**:

   a. **Decomposition of Complex GAUs**:

      - If a GAU is complex, the coder can decompose it into smaller child GAUs to simplify implementation and testing.

   b. **Reuse of Existing GAUs**:

      - If an existing GAU meets the requirements, the coder should reuse it instead of re-implementing. The coder is encouraged to reuse existing GAUs and declare new ones only when necessary.

   c. **Implementing Multiple GAUs**:

      - If the proposal involves multiple GAUs, the coder should implement them separately in different code blocks.
      - Each code block should contain a complete GAU implementation following the GAU template.
      - One code block should implement only one GAU.

   d. **Limited Access to Other GAUs**:

      - When working on a GAU, the coder will have access only to the current GAU’s implementation and its children's implementations, not be able to edit other GAUs.

   e. **Code Testing**:

      - The code will be tested by the format checker and functionality checker as well as the unit tests provided by the coder. 
      - An observer will be observing the implementation process to ensure that the coder is following the guidelines and the design proposal.

## Your Role as Planner

- **Progress Monitoring**:

  - Review the current implementation status, including which GAUs have been implemented.

- **Implementation Sequencing**:

  - Decide the optimal order for implementing the remaining GAUs, considering dependencies and priorities.
  - Detect if there is a chance to reuse existing GAUs and point out to the coder.
  
- **Task Assignment**:

  - Determine which GAU should be implemented next.

- **Guidance**:

  - Provide clear instructions to the coder for the next implementation task.

---

## Instructions for the Planning Process

1. **Review Current Status**:

   - **Overview Provided**: You will receive an updated overview of the implementation progress, including:
     - A list of units (GAUs) that have been implemented.
     - Any relevant notes on completed units.
     - Dependencies between units.
   - **Analysis**:
     - Identify which units are pending.
     - Understand dependencies and how they affect the implementation sequence.

2. **Decide the Next Unit to Implement**:

   - **Consider Dependencies**:
     - Prioritize units that unblock other units.
     - Ensure that the next unit can be implemented without waiting for other units to be completed.
   - **Assess Priorities**:
     - Focus on units critical to the core functionality of the LM.
     - Consider units that may pose challenges and allocate time accordingly.
   - **Enable Parallel Development**:
     - Where possible, identify units that can be developed concurrently by different coders.

3. **Provide Instructions to the Coder**:

   - **Specify the Next Unit**:
     - Clearly state which unit the coder should implement next.
   - **Include Implementation Details**:
     - Provide any specific instructions or considerations for the unit.
     - Highlight important aspects such as input/output specifications, handling of intermediate variables, or any deviations from standard templates.
   - **Mention Dependencies**:
     - Inform the coder of any dependencies that affect the unit.
     - Specify if the unit relies on outputs from other units or if it provides essential functionality for upcoming units.

4. **Communicate Effectively**:

   - **Clarity**:
     - Use clear and concise language.
     - Avoid technical jargon unless necessary and ensure it's well-defined.
   - **Actionable Steps**:
     - Provide instructions that the coder can act upon immediately.
     - Include any deadlines or time considerations if relevant.

5. **Update the Implementation Plan**:

   - **Documentation**:
     - Record the decision and instructions for transparency.
     - Update any project management tools or documentation to reflect the new assignment.
   - **Monitor Progress**:
     - Plan to review the coder's progress and be ready to adjust the plan as needed.


---

## Key Guidelines

- **Alignment with Project Goals**:
  - Ensure that the chosen unit aligns with the overall objectives of improving the LM as per the proposal.
- **Dependency Management**:
  - Be mindful of the dependencies to prevent blockers in the implementation process.
- **Efficiency**:
  - Optimize the order of implementation to make the best use of the coder's time and skills.
- **Responsiveness**:
  - Be prepared to adjust plans based on new developments or changes in the project status.

---

## Additional Considerations

- **Implementation Guidelines Reminder**:

  - Remind the coder to adhere to the implementation guidelines, including:

    - Use of the GAU template.
    - Proper handling of inputs and outputs.
    - Maintaining documentation standards.

- **Encourage Reuse**:

  - Urge the coder to reuse existing GAUs when appropriate.

- **Error Handling**:

  - Instruct the coder to handle missing arguments or edge cases.

- **Future Dependencies**:

  - Mention upcoming GAUs that depend on the current task.

---

**Final Notes**:

Your careful planning ensures that the implementation proceeds smoothly and efficiently. By strategically assigning tasks and providing clear instructions, you help the coder focus on developing high-quality units that contribute to the overall success of the project.

---

**Remember**:

- **Your decisions directly impact the team's productivity**. Thoughtful planning and clear communication are key.
- **Stay adaptable**. Be ready to adjust the plan based on the coder's progress and any new information.
- **Facilitate collaboration**. Your guidance helps coordinate efforts and keeps the project on track.
"""

O1M_IMPLEMENTATION_PLANNER_BACKGROUND_prompt = O1_IMPLEMENTATION_PLANNER_BACKGROUND_prompt + """
The following is the proposal to improve the seed design by improving a selected GAU: {SELECTION}.

## Seed Design Overview

{SEED}

## Proposal to Implement

{PROPOSAL}

### Review of the Proposal

{REVIEW}

### Rating: {RATING} out of 5
"""

O1C_IMPLEMENTATION_PLANNER_BACKGROUND_prompt = O1_IMPLEMENTATION_PLANNER_BACKGROUND_prompt + """
The following is the proposal to produce a new design by recombining the parents:

{PARENTS}

## Proposal to Implement

{PROPOSAL}

### Review of the Proposal

{REVIEW}

### Rating: {RATING} out of 5
"""


O1S_IMPLEMENTATION_PLANNER_BACKGROUND_prompt = O1_IMPLEMENTATION_PLANNER_BACKGROUND_prompt + """
The following is the proposal of a novel LM block design:

## Proposal to Implement

{PROPOSAL}

### Review of the Proposal

{REVIEW}

### Rating: {RATING} out of 5
"""


O1M_IMPLEMENTATION_PLANNER_BACKGROUND=AgentPrompt(O1M_IMPLEMENTATION_PLANNER_BACKGROUND_prompt)
O1C_IMPLEMENTATION_PLANNER_BACKGROUND=AgentPrompt(O1C_IMPLEMENTATION_PLANNER_BACKGROUND_prompt)
O1S_IMPLEMENTATION_PLANNER_BACKGROUND=AgentPrompt(O1S_IMPLEMENTATION_PLANNER_BACKGROUND_prompt)

def O1_SELECTION_parser(raw_output: ModelOutputPlus) -> Dict[Any,Any]:
      raw_text = raw_output.text
      output = {}
      selection = re.findall(r"```selection(.*?)```", raw_text, re.DOTALL)
      output["text"] = raw_text
      output["selection"] = [i.strip() for i in selection]
      output["_details"] = {}
      output["_details"]["cost"] = raw_output.usage
      output["_details"]["running_cost"] = raw_output.usage['cost']
      return output

O1_IMPLEMENTATION_PLANNER_SELECTION_prompt = """
It is round {ROUND} for the design implementation. Please make your plan.

#### Current Design Overview

{VIEW}

#### Log of Progress:
{LOG}

#### GAUs Available for Selection

{SELECTIONS}

- **Implemented GAUs ({IMPLEMENTED})**: Can be refined.
- **Unimplemented GAUs ({UNIMPLEMENTED})**: Need to be implemented.

*Note*: {PROTECTED} are protected and cannot be modified. You can only work under the subtree rooted at the selected GAU from proposer's selection.

*Reminder*: All unimplemented GAUs must be implemented eventually.

Please wrap your selection in a quoted block like this: ```selection YOUR_SELECTION```, for example: ```selection GAU_NAME```. 
You must include one and only one selection quoted block in your response.
Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""

O1_IMPLEMENTATION_PLANNER_SELECTION=AgentPrompt(O1_IMPLEMENTATION_PLANNER_SELECTION_prompt,O1_SELECTION_parser)


O1_IMPLEMENTATION_PLANNER_POST_REFINE_prompt = """
It is round {ROUND} for the design implementation. Please make your plan.

#### Current Design Overview

{VIEW}


#### Log of Progress:
{LOG}

#### GAUs Available for Selection

{SELECTIONS}

*Note*: {PROTECTED} are protected and cannot be modified. You can only work under the subtree rooted at the selected GAU from proposer's selection.

Now you have implemented all the GAUs, you can choose to refine them or terminate the implementation process.

Please wrap your selection in a quoted block like this: ```selection YOUR_SELECTION```, for example: ```selection GAU_NAME```. 
You must include one and only one selection quoted block in your response. 
Or if you want to terminate the implementation process, you can include a ```terminate``` quoted block in your response.
Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""

O1_IMPLEMENTATION_PLANNER_POST_REFINE=AgentPrompt(O1_IMPLEMENTATION_PLANNER_POST_REFINE_prompt,O1_SELECTION_parser)



O1_IMPLEMENTATION_PLANNER_BEGIN_prompt = """
It is the beginning of the design implementation. Please make your plan.

#### Current Design Overview

{VIEW}

You do not need to select any GAUs at the beginning as you will work on the selected unit at the beginning.
Please analyze the design proposal and give your plan, and providing guidance for the coder.
"""

O1_IMPLEMENTATION_PLANNER_BEGIN=AgentPrompt(O1_IMPLEMENTATION_PLANNER_BEGIN_prompt)

O1_IMPLEMENTATION_PLANNER_BEGIN_RETRY_prompt = """
The agent failed in It is the beginning of the design implementation. Please make your plan.

#### Current Design Overview

{VIEW}

You do not need to select any GAUs at the beginning as you will work on the selected unit at the beginning.
Please analyze the design proposal and give your plan, and providing guidance for the coder.
Do not worry about the number of tokens in your reasoning and your response, you can use as many as you need to give the best response.
"""

O1_IMPLEMENTATION_PLANNER_BEGIN=AgentPrompt(O1_IMPLEMENTATION_PLANNER_BEGIN_prompt)



# endregion

