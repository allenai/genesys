''' System of Review & Design Agents for Model Discovery '''

import exec_utils
import pathlib
import os
import json
import time
import tempfile
import copy
import numpy as np
# import uuid
import pandas as pd

#from IPython.display import display, Markdown, Latex
from types import ModuleType
from typing import (
    Type,
    List,
    Dict,
    Any,
    TypeVar,
    Tuple,
    Optional,
    Union
)
from dataclasses import dataclass

from exec_utils.aliases import ConfigType
from exec_utils import (
    BuildAgent,
    BuildTool
)
from .agents import *
from .agents.prompts.prompts import (
    DESIGNER_PROMPT,
    REVIEWER_PROMPT,
    GAB_ERROR
)

import model_discovery.utils as U

C = TypeVar("C",bound="ModelDiscoverySystem")

from .configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

# import multiprocessing as mp

__all__ = [
    "ModelDiscoverySystem",
    "BuildSystem",
]

PROJ_SRC = os.path.abspath(os.path.dirname(__file__))
SYSTEM_OUT = os.path.abspath(f"{PROJ_SRC}/../_runs")

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
    def forward(self,X,**intermediate_vars):
        """Forward pass of the model"""
        assert X.shape[-1] == self.embed_dim
        Y=self._forward(X,**kwargs)
        if isinstance(Y,tuple):
            intermediate_vars = Y[1:]
            Y = Y[0]
        else:
            intermediate_vars = {}
        assert Y.shape == X.shape
        return Y
'''


@exec_utils.Registry("config","discovery_system")
class CustomParams(exec_utils.ModuleParams):
    """Parameters for working with this discovery system 
    
    :param max_design_attempts: 
       The number of attempts that the designer agent ca 
        make. 
    :param max_design_refines: 
       The number of times the designer can refine a design 
    :param reviwer_threshold: 
       The threshold for accepting a design for the reviewer. 
    :param designer_spec: 
       Pointer to the designer specification 
    :param reviewer_spec: 
       Pointer to the reviewer specification. 
    :param block_template: 
       Points to the file for the GAB block for prompting. 
    :param gam_template: 
       Points to the file for the GAM template used in the prompt 
    :param debug_step: 
       Print the system steps when running. 
    :param gam_config: 
       The target configuration for the GAM model being explored.
    :param run_name: 
       The name identifier of the model search session. 

    """
    max_design_attempts: int = exec_utils.ParamField(
        default=10,
        metadata={
            "help"         : 'The maximum number of designer queries',
            "exclude_hash" : True,
        }
    )
    max_design_refines: int = exec_utils.ParamField(
        default=10,
        metadata={
            "help"         : 'The maximum number of designer refinements',
            "exclude_hash" : True,
        }
    )
    reviewer_threshold: int = exec_utils.ParamField(
        default=3,
        metadata={
            "help"         : 'The threshold for accepting a design',
            "exclude_hash" : True,
        }
    )
    ### agent profiles
    designer_spec: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/designer.json'
        ),
        metadata={
            "help"         : 'Specification of design agent',
            "exclude_hash" : True,
        }
    )
    reviewer_spec_creative: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/reviewer_creative.json'
        ),
        metadata={
            "help"         : 'Specification of creative reviewer agent',
            "exclude_hash" : True,
        }
    )
    reviewer_spec_balance: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/reviewer_balance.json'
        ),
        metadata={
            "help"         : 'Specification of balance reviewer agent',
            "exclude_hash" : True,
        }
    )
    reviewer_spec_rigorous: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/reviewer_rigorous.json'
        ),
        metadata={
            "help"         : 'Specification of rigorous reviewer agent',
            "exclude_hash" : True,
        }
    )
    debugger_spec: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/../etc/agent_spec/debugger.json'
        ),
        metadata={
            "help"         : 'Specification of debugger agent',
            "exclude_hash" : True,
        }
    )
    ### code information
    block_template: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/agents/prompts/gab_template.py'
        ),
        metadata={
            "help"         : 'Location of block for prompting ',
        }
    )
    gam_template: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/agents/prompts/gam_prompt.py'
        ),
        metadata={
            "help" : 'Location of code prompting ',
        }
    )
    gam_implementation: str = exec_utils.ParamField(
        default=os.path.abspath(
            f'{PROJ_SRC}/model/gam.py'
        ),
        metadata={
            "help" : 'Location of code prompting ',
        }
    )
    ### debugging
    debug_steps: bool = exec_utils.ParamField(
        default=True,
        metadata={
            "help"         : 'Debug the steps of the system',
            "exclude_hash" : True,
        }
    )
    jupyter: bool = exec_utils.ParamField(
        default=False,
        metadata={
            "help"         : 'Inside of jupyter',
            "exclude_hash" : True,
        }
    )
    gam_config: str = exec_utils.ParamField(
        default='GAMConfig_14M',
        metadata={
            "help"         : 'Debug the steps of the system',
            "exclude_hash" : True,
        }
    )
    #### name of the system run 
    run_name: str = exec_utils.ParamField(
        default='demo',
        metadata={
            "help"         : 'The name of the run',
            "exclude_hash" : True,
        }
    )
    from_json: str = exec_utils.ParamField(
        default='',
        metadata={
            "help"         : 'The location of json',
            "exclude_hash" : True,
        }
    )


def get_context_info(config,templated=False) -> Tuple[str,str]:
    """Grabs the block and model implementation details for the prompt 

    :param config: 
        The global configuration 
    :raises: ValueError 

    """
    if not os.path.isfile(config.block_template):
        raise ValueError(f'Cannot find the block template: {config.block_template}')
    if templated:
        config.gam_template = config.gam_template.replace('.py','_templated.py')
    if not os.path.isfile(config.gam_template):
        raise ValueError(f'Cannot find the code context')
    block = open(config.block_template).read()
    code = open(config.gam_template).read()
    
    return (block,code) 

class EmptyHandler: 
    def __init__(self,*args,**kwargs):
        pass 
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class PrintSystem:
    def __init__(self,config):
        self.jupyter = config.jupyter        
    
    def write(self,msg,**kwargs):
        print(msg)
    def markdown(self,msg,**kwargs):
        print(msg)


class AgentContext:
    def __init__(self):
        self.data = []
    
    def append(self,query,source,metadata):
        self.data.append((query,source,metadata))
    
    def get(self):
        return [(query,source) for query,source,_ in self.data]


class DialogThread: # Maybe one thread one context? No, e.g., multiple agents can be in same thread with different context
    def __init__(self,name,did,parent,log_dir=None):
        self.did = did
        self.parent = parent
        self.logs = []
        self.mid = 0 # message id
        self.name=name # The name of the thread
        self.log_dir = None
        self.return_message = None # set it means the thread has returned
        if log_dir:
            self.log_dir = U.pjoin(log_dir,f"thread_{did}_{name}")
            U.mkdir(self.log_dir)

    def log(self,type,data):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        activity = (timestamp,type,data)
        self.logs.append(activity)
        log_content = {'timestamp':timestamp,'type':type,'data':data}
        if self.log_dir: # save to seperate json files to avoid write conficts/lag
            U.save_json(log_content,f"{self.log_dir}/{timestamp}.json")
    
    def message(self,sender,receiver,content,context):
        data={'mid':self.mid,'sender':sender,'receiver':receiver,'content':content,'context':context}
        self.log('message',data)
        self.mid += 1
        
    def query_agent(self,caller,agent,query,source,context=None):
        self.message(caller,agent.name,query,'CALLER')
        history = context.get() if context else []
        response = agent(
            query,
            source=source,
            manual_history=tuple(history)
        )
        if context:
            context=context.data
        self.message(agent.name,caller,response,context)
        return response
    
    def fork(self,name,did,call=None): # a fork return a new thread, each thread expose a call and return message to the parent thread, the fork call and return
        self.log('fork',{'did':did,'name':name,'call':call})
        return DialogThread(name,did,self.did,self.log_dir)
    
class DialogManager:
    def __init__(self,log_dir,system_info):
        self.log_dir = log_dir
        self.threads = {}
        self.threads[0] = DialogThread('root',0,-1,self.log_dir) # create a root thread
        if log_dir:
            U.save_json(system_info,f"{log_dir}/system_info.json")
    
    def assign_did(self):
        return len(self.threads)
    
    def fork(self,parent_did,name,call_message=None):
        parent = self.threads[parent_did]
        did = self.assign_did()
        self.threads[did] = parent.fork(name,did,call_message)
        return did
    
    def close_thread(self,did,content):
        thread = self.threads[did]
        thread.return_message = content
    
    def get_thread(self,did):
        return self.threads[did]

    def get_active_threads(self):
        return [did for did in self.threads if self.threads[did].return_message is None]

    
@exec_utils.Registry(
    resource_type="system_type",
    name="model_discovery_system",
    #cache="query_system",
)
class ModelDiscoverySystem(exec_utils.System):
    """Overall system for discovery
    
    Attributes
    --------
    :param designer: 
        The designer LLM agent. 
    :param reviewers: 
        The reviewer LLM agents. 
    :param checker: 
        The checker tool agent. 
    :param debugger:
        The debugger LLM agent.
    :param block_template: 
        The templated code to be filled in during 
        the block search. 
    :param gam_config: 
        the global model configuration 
    :param config: 
        The system level configuration. 
        
    Methods 
    --------
    query_sysetem(query, stream, frontend, status) 
        The main method for querying the system with the 
        different agents. 

    Example 
    --------

    >>> from model_search import BuildSystem  
    >>> system = BuildSystem()

    """

    def __init__(
        self,
        designer : Type[exec_utils.SimpleLMAgent],
        reviewers : Dict[str,Type[exec_utils.SimpleLMAgent]],
        checker  : Type[exec_utils.BaseTool], 
        debugger : Type[exec_utils.SimpleLMAgent],
        *,
        block_template: str,
        gam_template: str,
        gam_config: Type[GAMConfig],
        config: ConfigType 
    ) -> None:
        """Create a `DiscoverySystem` instance 

        
        :param designer: 
            The designer agent. 
        :param reviewers: 
           The reviewer agents. 
        :param block_template: 
           The autoregressive block that the model has to fill in.
        :param model_implementation: 
           The full model implementation for context 
        :param config: 
           System global configuration. 

        """
        ### modules 
        self.designer = designer
        self.reviewers = reviewers
        self.checker = checker
        self.debugger = debugger
        
        self.gam_py = gam_template
        self.gab_py = block_template
        self._config = config
        self._cfg = gam_config

        ###
        self._queries = [] 
    
    def get_system_info(self):
        system_info = {}
        system_info['agents']={
            'designer':self.designer.config,
            'reviewers':{style:agent.config for style,agent in self.reviewers.items()},
            'debugger':self.debugger.config
        }

    def new_session(self,log_dir=None):
        U.mkdir(log_dir)
        self.log_dir = log_dir
        self.states = {}
        self.states['refresh_template'] = 0 
        self.dialog = DialogManager(log_dir,self.get_system_info())

    def close_session(self):
        self.dialog.close_thread(0,'Session closed')
        if self.dialog.get_active_threads():
            raise ValueError('There are active threads, can not close the session')

    def set_config(self,cfg:GAMConfig): # reset the gam config of the system
        self._cfg = cfg

    def design(self,query,designer_context,stream,status_handler,thread_did): # input query, context, output design and explanation
        designer_cost = 0
        problem_history = copy.deepcopy(designer_context) # a new dialog branch
        source='user'
        thread: DialogThread=self.dialog.get_thread(thread_did)

        debugger_context = AgentContext()
        initial_error = None
        for attempt in range(self._config.max_design_attempts):
            self.logging.info(f'Attempting design, attempt={attempt}')
            
            design_name = f"{self._config.run_name}_{attempt}_{len(self._queries)}"

            with status_handler(f"Attempt {attempt+1}"): 
                
                caller='system' if attempt==0 else 'debugger'
                agent=self.designer if attempt==0 else self.debugger
                designer_out = thread.query_agent(
                    caller, agent, query, source,
                    context=problem_history if attempt==0 else debugger_context,
                )
                metadata = {'sender':caller, 'did': thread_did, 'mid': thread.mid-2} # generated by which agent which thread which message
                if attempt == 0:
                    problem_history.append(query,source,metadata)
                else:
                    debugger_context.append(query,source,metadata)

                try:
                    code = designer_out.get("code",None)
                    text = designer_out.get("text")
                    designer_cost += designer_out["_details"]["running_cost"]
                    metadata = {'sender':agent.name, 'did': thread_did, 'mid': thread.mid-1} # generated by which agent which thread which message
                    if attempt == 0:
                        problem_history.append(str(text),"assistant",metadata)
                        debugger_context.append(f'The designer designed the model: {text}',"user",metadata)
                    else:
                        debugger_context.append(f'{text}',"assistant",metadata)

                    assert code is not None
                    assert "# gab.py" in code 
                
                    if stream: #and code:
                        stream.write('Model authored code block...')
                        stream.markdown(str(text))

                except AssertionError:
                    source = "user"
                    query  = GAB_ERROR
                    continue 
                
                checkpass,check_report,code,check_results = self.checker.check(self._cfg,code,design_name)
                checker_hints = check_results['hints']
                if 'REFRESH_TEMPLATE' in checker_hints:
                    self.states['refresh_template'] += 1
                
                if stream:
                    stream.write(
                        f"""<details><summary>code check</summary>{check_report}</details>""",
                        unsafe_allow_html=True
                    )
                
                if checkpass:
                    report_query = (
                        "The designed model passed the tests, now please generate a text report explaining and justifying your design."
                        " Generate a creative name of your design as the title of your report in the first line of your response."
                        " Do not include abbreviations or acronyms of your design in the title. You can use them in the body of the report."
                        f" Here is the code of the designed model after degugging:\n\n{code}" # FIXME: what is the code after debugging is not the same as the idea before debugging
                    )
                    if initial_error is not None:
                        error_info=f"Your design didn't pass the checker initially:\n\n{initial_error}\n\nIt has been fixed by the assistant already as follows:\n\n{code}"
                    with status_handler(f"Querying agent for report..."):
                        self.logging.info('Now trying to compile self report...')
                        self_report = thread.query_agent(
                            caller='system',
                            agent=self.designer,
                            query=report_query if initial_error is None else f'{error_info}\n\n{report_query}',
                            source=source,
                            context=problem_history,
                        )
                        if stream:
                            stream.markdown(self_report["text"]) #<-- change

                    explain=self_report['text']
                    return code,explain,designer_cost,check_results
                else:
                    query = f"The designed model didn't pass, you need to try again. Here is the report:\n{check_report}. Please fix."
                    if self.states['refresh_template'] >=1:
                        query+=f'\nHere is the template for the GAB block for you to refresh:\n\n{self.gab_py}'
                    if self.states['refresh_template'] >= 2:
                        query+=f'\nHere is the definition of GABBase for you to refresh:\n\n{GAB_BASE}'
                    if self.states['refresh_template'] >= 3:
                        query+=f'\nHere is the definition for the GAM model for you to refresh:\n\n{self.gam_py}'
                    source = 'user'
                    if attempt == 0:
                        initial_error = check_report
                    self.checker.reset()

        return None,None,designer_cost,None
        

    def review(self,proposal,costs,stream,status_handler,thread_did): 
        thread = self.dialog.get_thread(thread_did)
        reviewer_query = REVIEWER_PROMPT.format(
            proposal=proposal,
            gab_base=GAB_BASE,
            gam_py=self.gam_py,
            instruct='' # TODO: add references from S2 etc.
        )
        reviews = {}
        ratings = {}
        with status_handler(f"Querying reviewer agent for review..."):
            for style in self.reviewers:
                reviewer = self.reviewers[style]
                response = thread.query_agent(
                    caller='system',
                    agent=reviewer,
                    query=reviewer_query,
                    source='user',
                    # No history for reviewer or would some history be useful?
                )
                if stream:
                    stream.markdown(response["text"])
                ratings[style] = response["rating"]
                reviews[style] = response["review"]
                costs['review'] += response["_details"]["running_cost"]
        return ratings,reviews,costs

    def query_system(
        self,
        query: Optional[str] = '',
        log_dir: Optional[str] = None,
        stream: Optional[ModuleType] = None,
        frontend: Optional[bool] = False,
        status: Optional[bool] = True,
        **kwargs
    ) -> list:
        """Main function for implementing system calls.

        :param query: 
            The query to the overall system. 
        :param stream: 
            The (optional) streamlit module for writing to frontend 
        :param frontend: 
            Switch indiciating whether system is being used 
            with a frontend.
        
        """
        self.new_session(log_dir)
        main_did = self.dialog.fork(0,'main','Starting a new session...')

        costs={ # NOTE: costs in exec_utils need to be updated
            'design':0,
            'review':0,
            'summary':0,
        }
        status_handler = stream.status if stream and status else EmptyHandler
        if stream is None and self._config.debug_steps:
            stream = PrintSystem(self._config)
            
        designer_context = AgentContext() # should be input, design, review, design, review, ...

        # query = f"{query}\nPlease only write raw Python code and nothing more, no special formatting or extra text."
        designer_query = DESIGNER_PROMPT.format(
            gab_base=GAB_BASE,
            gam_py=self.gam_py,
            gab_py=self.gab_py,
            config=self._cfg.to_prompt(), #<--- need to parameterize 
            instruct=query,
        )
        found_design = False
        self._queries.append(designer_query)

        for i in range(self._config.max_design_refines):
            refine_thread_did = self.dialog.fork(main_did,'design_refine',f'Design refinement round {i+1}')

            design_thread_did = self.dialog.fork(refine_thread_did,'design_attempt',f'Starting design attempt...')
            designer_thread = self.dialog.get_thread(design_thread_did)
            code,explain,designer_cost,check_results = self.design(query,designer_context,stream,status_handler,design_thread_did)
            costs['design'] += designer_cost
            if code is None: continue

            proposal=f'{explain}\n\nImplementation:\n\n{code}\n\n'
            self.dialog.close_thread(design_thread_did,f'The designer has designed the model:\n\n{proposal}')
            designer_context.append(query,"user",{'sender':'system','did':design_thread_did,'mid':0})
            designer_context.append(proposal,"assistant",{'sender':'designer','did':design_thread_did,'mid':designer_thread.mid-1})

            review_thread_did = self.dialog.fork(refine_thread_did,'review',f'Sending the design to reviewers for review...')
            ratings,reviews,costs = self.review(proposal,costs,stream,status_handler,review_thread_did)
            review_ratings=''
            for idx, style in enumerate(self.reviewers):
                review=reviews[style]
                rating=ratings[style]
                review_ratings+=f'# Review of Reviewer {idx+1}:\n\n{review}\n\n## Rating: {rating} out of 5\n\n'
            self.dialog.close_thread(review_thread_did,f'The reviewers have returned the reviews:\n\n{review_ratings}')
            rating=np.mean(list(ratings.values()))
            if rating >= self._config.reviewer_threshold:
                found_design = True
                self.dialog.close_thread(refine_thread_did,f'The design passed the review process')
                break
            else: # next round design
                query=f'The design didn\'t pass the review process, here is the feedback from the reviewer, please improve your design based on the reviews: {review_ratings}'
                self.dialog.close_thread(refine_thread_did,f'The design didn\'t pass the review process')

        if not found_design:
            self.dialog.close_thread(main_did,'The design process has ended without a successful design')
            return None
        
        title=explain.split('\n')[0].replace('#','').strip()#+'_'+str(uuid.uuid4().hex[:6])

        ### Leave it open for now for debugging, the model only fails if it designs a really huge block
        # try:
        autocfg = self.checker.tune(self._cfg,code,title)
        # except Exception as e:
        #     print(f"Error tuning the scale of designed model: {e}")
        #     return None
        
        ### Generate a summary
        with status_handler(f"Generating summary..."):
            self.logging.info('Generating summary of the design...')
            summary_query = (
                "Here is a design of an autoregressive language model block. "
                "The code and explanation of the design are provided below:\n\n"
                f"{explain}\n\nImplementation of {title}:\n\n{code}\n\n"
                "Please summarize the design with a description of the design and a simple pseudo code that conclude the core idea in few sentences."
            )
            summary_thread_did = self.dialog.fork(main_did,'summary')
            summary_thread = self.dialog.get_thread(summary_thread_did)
            response = summary_thread.query_agent(
                caller='system',
                agent=self.designer,
                query=summary_query,
                source='user',
            )
            summary=response['text']
            self.dialog.close_thread(summary_thread_did,f'The summary of the design is:\n\n{summary}')
            costs['summary'] += response["_details"]["running_cost"]
            if stream:
                stream.markdown(summary)
        
        if self.log_dir:
            U.save_json(costs,f"{self.log_dir}/costs.json")
        
        self.dialog.close_thread(main_did,f'The design process has ended with a successful design:\n\nTitle: {title}\n\nCode:\n\n{code}\n\nExplaination:\n\n{explain}\n\nSummary:\n\n{summary}\n\nReviews:\n\n{review_ratings}')

        self.close_session()
        return title,code,explain,summary,autocfg,reviews,ratings,check_results

    @classmethod
    def from_config(cls: Type[C],config: ConfigType,**kwargs) -> C:
        """The main method for instantiating system instances from configuration. 

        :param config: 
            The global configuration used to create instance. 
        :returns: 
            A `DiscoverySystem` instance from configuration. 

        """
        ### creates designer and reviewer agents
        
        designer = BuildAgent(
            config,
            agent_file=config.designer_spec,
            agent_model_type="designer_agent"
        )
        checker = BuildTool(
            tool_type="checker",
        )
        reviewers = {
            'balance': BuildAgent(
                config,
                agent_file=config.reviewer_spec_balance,
                agent_model_type="reviewer_agent"
            ),
            'creative': BuildAgent(
                config,
                agent_file=config.reviewer_spec_creative,
                agent_model_type="reviewer_agent"
            ),
            'rigorous': BuildAgent(
                config,
                agent_file=config.reviewer_spec_rigorous,
                agent_model_type="reviewer_agent"
            )
        }
        debugger = BuildAgent(
            config,
            agent_file=config.debugger_spec,
            agent_model_type="designer_agent"
        )
        
        ### get the model information for context
        cfg = eval(f"{config.gam_config}()")
        block, code = get_context_info(config,templated=cfg.use_template)
        
        return cls(
            designer,
            reviewers,
            checker,
            debugger,
            block_template=block,
            gam_template=code,
            config=config,
            gam_config=cfg
        )

def BuildSystem(
        config: Optional[ConfigType] = None,
        **kwargs
    ) -> ModelDiscoverySystem:
    """Factory for building an overall system

    :param config: 
        The optional configuration object. 

        
    >>> from model_discovery import BuildSystem 
    >>> system = BuildSystem() 

    """
    from exec_utils import BuildSystem
    kwargs["system_type"] = "model_discovery_system"
    
    # if config and not config.wdir:
    #     wdir = f"{SYSTEM_OUT}/{time.strftime('%Y%m%d_%H%M%S')}"
    #     kwargs["wdir"] = wdir
    # elif "wdir" not in kwargs:
    #     wdir = f"{SYSTEM_OUT}/{time.strftime('%Y%m%d_%H%M%S')}"
    #     kwargs["wdir"] = wdir
        
    return BuildSystem(config,**kwargs)

