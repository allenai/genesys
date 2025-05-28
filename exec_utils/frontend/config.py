### utilities for the demo frontend

from ..register import Registry
from ..param import (
    ModuleParams,
    ParamField
)

@Registry("config","llm_sim.frontend.config")
class DemoParams(ModuleParams):
        
    demo_name: str = ParamField(
        default='My demo',
        metadata={
            "help" : 'the name of the frontend demo',
             "exclude_hash" : True,
        }
    )
    speed: int = ParamField(
        default=10,
        metadata={
            "help" : 'The speed of streaming output',
             "exclude_hash" : True,
        }
    )
    wait_text: str = ParamField(
        default='Running query',
        metadata={
            "help" : 'The text do display when waiting for output',
             "exclude_hash" : True,
        }
    )
    filler_text: str = ParamField(
        default='What can I help you with?',
        metadata={
            "help" : 'The text do display in the text box',
             "exclude_hash" : True,
        }
    ) 
    downloadable: bool = ParamField(
        default=False,
        metadata={
            "help" : 'Download output of your front execution',
             "exclude_hash" : True,
        }
    ) 
    frontend_dataset: str = ParamField(
        default='',
        metadata={
            "help" : 'Dataset for the frontend',
             "exclude_hash" : True,
        }
    ) 
