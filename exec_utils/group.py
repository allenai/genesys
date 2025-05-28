import json
import threading

from typing import (
    Union,
    TypeVar,
    List,
    Dict,
    Tuple,
    Callable,
    Type,
    Any,
    Optional
)
from queue import Queue

from .register import Registry
from .base import UtilResource
from .aliases import ConfigType
from .utils import *
from .param import (
    ModuleParams,
    ParamField
)

__all__ = [
    "AsyncGroup"
]

class GroupBase(UtilResource):
    """Base class for groups of agents and tools 
    
    Attributes
    ----------
    :param size: 
        The number of agents in the group 
    :param cost: 
        The running cost of the agent (alias for __call__)

    Methods 
    ----------
    query_group(query: str)
        The main method for querying members of the group 
    
    """
    def query_group(self,query:str) -> Dict[str,Any]:
        """Main query method

        :param query: 
            The input (str) query to the group. 
        
        """
        raise NotImplementedError
    
    def __call__(self,query : str) -> Dict[str,Any]:
        return self.query_group(query)

    def __iter__(self):
        for agent in self._agents.values():
            yield agent 
            
    @property
    def size(self) -> int:
        """Returns the size of the society, or number of agents
        """
        return len(self._agents)
    
    @property
    def cost(self) -> float:
        """Returns the running cost of the society"""
        self.logging.warning(
            'Cost function not explicit implemented, be careful'
        )
        return 0.

def _query_agents(
        agent_list: List,
        query: Union[str,List[Union[str,Dict[str,str]]]],
        num_threads: Optional[int]=5,
        timeout: Optional[int]=600,
    ):
    """General method for querying agents asychronously using threads 

    :see: https://github.com/jiangjiechen/auction-arena
    
    :param agent_list: 
        The list of agents. 
    :param query: 
        The query or input to the agents.  
    :param num_threads: 
        The number of threads to run.  
    :param timeout: 
        The timeout for the independent threads.  
    """
    result_queue = Queue()
    threads = []
    semaphore = threading.Semaphore(num_threads)

    if isinstance(query,str):
        query = [
            {"query" : query} for _ in range(len(agent_list))
        ]
    elif isinstance(query,list):
        query = [
            {"query" : q} if isinstance(q,str) else q \
            for q in query
        ]
            
    def call_agent(agent,agent_query):
        try: 
            result = agent(**agent_query)
            result_queue.put((True, agent, result))
        finally:
            semaphore.release()

    for i,agent in enumerate(agent_list):
        aquery = query[i] 
        
        args = (agent,aquery) 
        thread = threading.Thread(
            target=call_agent,
            args=args
        )

        thread.start()
        threads.append(thread)
        
    for thread in threads:
        thread.join(timeout=timeout)

    results = [
        result_queue.get() for _ in range(len(agent_list))
    ]
        
    return results

@Registry(
    resource_type="group_type",
    name="simple",
    cache="query_group"
)
class AsyncGroup(GroupBase):
    """A simple agent group that makes asychronous query calls to an agent.  

    Attributes
    ----------
    :param size: 
        The number of agents in the group 
    :param cost: 
        The running cost of the agent (alias for __call__)

    Methods 
    ----------
    query_group(query: str) -> dict
        The main method for querying members of the group 

    """
    def __init__(
            self,
            agent_list,
            config
    ) -> None:
        """Initializes an agent group. 

        :param agent_list: 
            The list of agents. 
        :param config: 
            The global configuration, contianing information 
            about agents, multi-processing, etc. 

        """
        self._agents = {a.name : a for a in agent_list}
        self._config = config
        self._group_board = []
        
        self.num_threads = self._config.thread_num
        self.timeout = self._config.thread_timeout
            
    def query_group(self,query:str) -> Dict[str,Any]:
        """Main query method. In this class, it simply makes 
        an asynchronous call to members of the group 

        :param query: 
            The input (str) query to the group. 
        """
        raw_out = _query_agents(
            list(self._agents.values()),
            query,
            timeout=self.timeout,
            num_threads=self.num_threads
        )
        group_output = {} 

        for (status,agent,output) in raw_out:
            name = agent.name
            group_output[name] = output
                    
        return group_output

    @property
    def agents(self):
        return self._agents
    
    @classmethod
    def from_config(cls,config,**kwargs):
        """Loads a group from configuration. 

        :param config: 
            The global configuration used to build an instanc. 
        :raises: 
            ValueError 
        """
        group_specs = get_group_specs(config.group_loc)
        agent_list = []

        agent_names = set()
        for i,component in enumerate(group_specs):
            kwargs.update({"spec": component})
            agent = Registry.build_model("agent",config,**kwargs)
            
            if agent.name in agent_names:
                raise ValueError(
                    f"Repeat agent name: {agent.name}"
                )
            
            agent_list.append(agent)
            agent_names.add(agent.name)
            
        return cls(agent_list,config)


@Registry("config","exec_utils.group")
class Params(ModuleParams):
    """Parameters for agent class"""

    group_type: str = ParamField(
        default='simple',
        metadata={
            "help" : 'The type of group to use',
        }
    )
    max_components: int = ParamField(
        default=10,
        metadata={"help" : 'Maximum number of components'}
    )
    thread_num: int = ParamField(
        default=5,
        metadata={
            "help": "The number of threads to run",
            "exclude_hash" : True,
        }
    )
    thread_timeout: int = ParamField(
        default=600,
        metadata={
            "help": "The timeout for the thread",
            "exclude_hash" : True,
        }
    )
    group_loc: str = ParamField(
        default='',
        metadata={"help": "The location of the group specifications"}
    )
