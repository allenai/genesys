from typing import (
    List,
    Tuple,
    Any
)

from ..base import (
    UtilityModel,
    Validator,
    ModelField
)

__all__ = [
    "SimplePrompt",
    "format_prompt"
]

class PromptBase(UtilityModel):
    """Base class for prompt instances
    
    :param instruction: 
        The (optional) instruction associated with the prompt. 
    :param examples: 
        A list of balanced (i.e., human-ai) feedback to add to 
        the input as in-context examples. 
    :param prompt: 
        The underlying prompt or task asked to the agent. 

    """
    instruction: str = ""
    examples: List[str] = ModelField(default_factory=list)
    history: List[str]  = ModelField(default_factory=list)
    prompt: str

    @Validator('examples')
    def examples_wellformed(cls,v):
        """Check that examples are provided in pairs 

        :raises: AssertionError
        """
        assert len(v) % 2 == 0, "Examples should be balanced"
        
class SimplePrompt(PromptBase):
    pass 


####################
# PROMPTING UTILS  #
####################

def format_prompt(prompt: str,**kwargs) -> SimplePrompt:
    """Factory method for formatting prompts and created prompt objects 
    based on passed input. 


    :param prompt: 
        The input string prompt. 
        
    """
    examples = kwargs.get("examples",[])
    instruction = kwargs.get("instruction","")
    history = kwargs.get("history",[])
    
    return SimplePrompt(
        prompt=prompt,
        examples=examples,
        instruction=instruction,
        history=history
    )
