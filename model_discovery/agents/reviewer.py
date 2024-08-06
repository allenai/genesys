from __future__ import annotations

from typing import (
    Dict,
    Tuple,
    List,
    Optional,
    Any
)
import exec_utils
import json
import re

__all__ = [
    "ReviewerAgent"
]


@exec_utils.Registry(
    resource_type="agent_model_type",
    name="reviewer_agent",
    cache="query" #<--- set to `None` if you don't want caching
)
class ReviewerAgent(exec_utils.SimpleLMAgent):
    """Agent for reviewing designs. 

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

        js = re.findall(r"```json(.*?)```", raw_text, re.DOTALL)
        json_output = json.loads(js[0])

        output["text"] = raw_text
        output["rating"] = json_output['rating']
        output["review"] = json_output['review']
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
        num_max_retry=5

        success = False
        for _ in range(num_max_retry):
            raw_response = self.model(
                prompt=query,
                model_state=self.model_state,
                history=tuple(manual_history)
            )
            try:
                response = self.parse_output(raw_response)
                success = True
                break
            except Exception as e: 
                manual_history=list(manual_history)
                manual_history.append((raw_response.text,'assistant'))
                query = f"Error parsing output from model: {e}\n\nThe output must be a json with keys 'rating' and 'review'"
        
        if not success:
            raise ValueError(f"Error parsing output from model: {e}\n\nThe output must be a json with keys 'rating' and 'review'")
        return response

    