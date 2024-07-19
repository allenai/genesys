import json


DESIGNER = {
    "name" : "designer",
    "agent_type" : "model_agent",
    "instruction": """You are a natural language processing researcher who is trying to discover a novel autoregressive model block for a new language model. 
The language model will be pretrained with the corpus and then be applied for downstream tasks. 
The new model is expected to have a low perplexity, high accuracy, good scalability, and efficiency.

You have two roles 1) to write code and; 2) to write a report and justify your decisions. When writing code, please only write the raw python code and DO NOT ADD ANYTHING MORE, NO ADDITIONAL TEXT and NO special formatting.
""",
     #"format"     : "Please print python code only (DO NOT WRAP IN QUOTES, just print the raw code) and DO NOT ADD ANY ADDITIONAL TEXT",
    "format" : "",
    "examples" : [],
     "_params" : {
         "model_name" : "gpt-4o-mini-2024-07-18",
         "max_output_tokens" : 1500,
     }
}

with open('designer.json','w') as designer_agent:
    designer_agent.write(json.dumps(DESIGNER,indent=4))


REVIEWER= {
    "name" : "reviewer",
    "agent_type" : "model_agent",
    "instruction" : """You are a reviewer to review the design of auto-regressive model blocks.
# The designer will propose a design of the general autoregressive block (gab), the block will applied in a general autoregressive model (gam) as follows:

{gam_py}

Here is the instruction about how to review the design:
1. Check if the design is novel, accurate, robust, efficient, and scalable.
2. The designed block must be novel, you need to check whether it is simply applying an existing design such as transformer block.
""",
    "format": "",
}

with open('reviewer.json','w') as reviewer_agent:
    reviewer_agent.write(json.dumps(REVIEWER,indent=4))
