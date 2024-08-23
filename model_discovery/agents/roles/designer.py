from __future__ import annotations

from typing import (
    Dict,
    Tuple,
    List,
    Optional,
    Any
)
import re
import exec_utils 
from exec_utils.models.model import ModelOutput
import random
from ..agent_utils import structured__call__,ModelOutput


__all__ = [
    "DesignerAgent",
]

@exec_utils.Registry(
    resource_type="agent_model_type",
    name="designer_agent",
    cache="query"
)
class DesignerAgent(exec_utils.SimpleLMAgent):
    """Agent for designing new models. 
    
    Methods 
    ----------
    query(query: str) -> dict
        The main method for handling input queries to the agent. 
    parse_output(raw_output: str) -> Any
        The main method for parsing and normalizing the output 
    """
    def parse_output(self,raw_output: ModelOutput) -> Dict[Any,Any]:
        """Execution of the output produced by the agent.  

        :param raw_output: 
            The raw output of the model
        :returns: 
            A dictionary containing the formatted output plus 
            additional details `_details` with information about 
            the running costs of the model.
        """
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
        output["_details"]["running_cost"] = self.cost

        return output
    

    def query(
        self,
        query: str,
        source: Optional[str] = "user",
        manual_history: Optional[Tuple[str,str]] = [], 
        **kwargs
        ) -> Dict[str,Any]:
        """Main method for querying agent 

        :param query: 
            The target query to pass to the agent. 
        :param source: 
            The source of the query (optional) that indicates where 
            the query is coming from (e.g., human, assistant, etc)
        :param intermediate: 
            Indicates whether the query is part of a series of queries
            to the agent (which requires using the history)  
        """
        self.model_state.query_state = source
        if not self.response_format:
            raw_response = self.model(
                prompt=query,
                model_state=self.model_state,
                history=tuple(manual_history)
            )
        else:
            raw_response = structured__call__(
                self.model,
                response_format=self.response_format,
                prompt=query,
                model_state=self.model_state,
                history=tuple(manual_history)
            )
        response = self.parse_output(raw_response)

        return response

    