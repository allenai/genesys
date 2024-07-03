from typing import (
    Dict,
    Tuple,
    List,
    Optional,
    Any
)
import exec_utils 

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

    """
    # def query(
    #     self,
    #     query: str,
    #     source: Optional[str] = "user",
    #     manual_history: Optional[Tuple[str,str]] = [], 
    #     **kwargs
    #     ) -> Dict[str,Any]:
    #     """Main method for querying agent 
    #     """
    #     raise NotImplementedError
