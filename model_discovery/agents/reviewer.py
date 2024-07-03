from typing import (
    Dict,
    Tuple,
    List,
    Optional,
    Any
)
import exec_utils

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
