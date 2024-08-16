''' System of Review & Design Agents for Model Discovery '''

import exec_utils
import pathlib
import os
import json
import time
from datetime import datetime
import tempfile
import copy
import numpy as np
import uuid
import pandas as pd
import functools as ft
import networkx as nx
import inspect
import markdown
import ast
import re
import astor
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout, RadialLayout
import pyflowchart as pfc

#from IPython.display import display, Markdown, Latex
from types import ModuleType, CodeType, FunctionType
from typing import (
    Type,
    List,
    Dict,
    Any,
    TypeVar,
    Tuple,
    Optional,
    Union,
    NamedTuple
)
from dataclasses import dataclass

from exec_utils.aliases import ConfigType
from exec_utils import (
    BuildAgent,
    BuildTool
)
from exec_utils.models import SimpleLMAgent
from .agents.roles import *
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

FAILED = "FAILED"

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
        self.status = EmptyHandler     
    
    def write(self,msg,**kwargs):
        print(msg)
    def markdown(self,msg,**kwargs):
        print(msg)


class AgentContext:
    def __init__(self):
        self.data = []
    
    def append(self,query,role,metadata):
        assert isinstance(query,str) and isinstance(role,str), f'Query and role must be strings'
        self.data.append((query,role,metadata))
    
    def get(self):
        return [(query,role) for query,role,_ in self.data]
    

@dataclass
class ROLE:
    name: str
    obj: Any = None
    role: str = None
    def __post_init__(self):
        if self.role is None:
            if isinstance(self.obj,SimpleLMAgent):
                self.role = 'assistant'
            elif isinstance(self.obj,AgentDialogFlow) or isinstance(self.obj,AgentDialogFlowNaive):
                self.role = 'system'
            elif isinstance(self.obj,AgentDialogThread):
                self.role = 'system'

SYSTEM_CALLER = ROLE('system',role='system')


class AgentDialogThread: # TODO: let runable thread be CFG
    '''
    An Agent program is nested by threads
    Thread types:
        Chat: Query -> Agent -> message, none, the most basic thread 
        Flow: Query -> Flow -> message, return, a chat composed of control flow
        Pipe: Query -> [LThread -> RThread] -> RRes, (LRes, LRet, RRet), a pipe composed of two threads
        Empty: No agent or function, only forking
    '''
    def __init__(self,alias,tid,parent,stream,caller=None,callee=None,log_dir=None,context=AgentContext()): 
        self.tid = tid
        self.parent = parent
        self.mid = 0 # message id
        self.alias=alias # The alias of the thread
        self.log_dir = None
        # self.return_message = None # set it means the thread has closed
        self.stream = stream   
        self.history = context # dialog history of the thread
        self.flow=None # the call message to the agent
        self.agent = None
        self.rthread = None
        self.type = None
        self.log_count = 0
        self._cost = 0
        self._carry(caller,callee)
        if log_dir:
            self.log_dir = U.pjoin(log_dir,f"thread_{tid}_{alias}")
            U.mkdir(self.log_dir)

    def _carry(self,caller,callee):
        self.caller = caller
        if caller:
            assert isinstance(caller,ROLE), f'Caller must be a ROLE object if caller is provided'
        self.callee = callee
        if callee:
            assert isinstance(callee,ROLE), f'Callee must be a ROLE object if callee is provided'
            if isinstance(callee.obj,AgentDialogFlow) or isinstance(callee.obj,AgentDialogFlowNaive):
                self.flow = callee.obj
                self.type = 'flow'
            elif isinstance(callee.obj,SimpleLMAgent):
                self.agent = callee.obj
                self.type = 'chat'
                self.hinter = ROLE('SYSTEM',role='system') # optional system hint between query and response
            elif isinstance(callee.obj,AgentDialogThread):
                assert isinstance(self.caller.obj,AgentDialogThread), f'Pipe should be between two threads'
                self.lthread = self.caller.obj
                self.rthread = callee.obj
                self.type = 'pipe'

    def _log(self,type,data,timestamp=None,verbose=False): #directly log to disk
        if not timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_content = {'timestamp':timestamp,'type':type,'data':data}
        if verbose:
            if type=='message':
                self.stream.markdown(data['content'])
            elif type=='fork':
                self.stream.markdown(f'Thread {self.tid}:{self.alias} forked {data["tid"]}:{data["alias"]} with note: {data["note"]}')
        if self.log_dir: # save to seperate json files to avoid write conficts/lag
            U.save_json(log_content,U.pjoin(f"{self.log_dir}",f"{timestamp}_{self.log_count}.json"))
        self.log_count += 1
    
    def _message(self,sender,receiver,content,timestamp=None,cost=0): # log message to the history
        data={'mid':self.mid,'sender':sender.name,'receiver':receiver.name,'content':content,'cost':cost}
        metadata = {'sender':sender.name, 'tid': self.tid, 'mid': self.mid} # generated by which agent which thread which message
        self.history.append(content,sender.role,metadata)
        self._log('message',data,timestamp)
        self.mid += 1
        
    def _chat(self,query,system_hint=None): # Only two types of call, Query->Response or Query->Hint->Response
        assert self.type=='chat', f'Thread {self.tid}:{self.alias} is not chatable'
        if system_hint is not None: # Query->Hint->Response, only for user
            # assert self.caller.role == 'user'
            self._message(self.caller,self.callee,query)
            input=system_hint
            queryer=self.hinter
        else: # Query->Response
            input=query
            queryer=self.caller
        assert isinstance(input,str), f'Input must be a string'
        query_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        response = self.agent(
            input,
            source=queryer.role, # NOTE: Neet to modify exec_utils to allow it use system as a role
            manual_history=tuple(self.history.get())
        )
        text = response['text']
        cost = response["_details"]["running_cost"] # only chat thread trigger a cost
        self._message(queryer,self.callee,query,timestamp=query_time)
        self._message(self.callee,self.caller,text,cost=cost)
        return text,response
    
    def _flow(self,query,**kwargs):
        assert self.type=='flow', f'Thread {self.tid}:{self.alias} is not runable'
        query_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        try:
            flow=ft.partial(self.flow,query=query,parent_tid=self.tid)
        except Exception as e:
            raise ValueError(f'Thread function must have query and parent_tid as argument, error: {e}')
        output = flow(**kwargs)
        assert isinstance(output,tuple) and len(output)==2, f'Thread function should return a tuple of (message,return)'
        msg,ret = output
        self._message(self.caller,self.callee,query,query_time)
        self._message(self.callee,self.caller,msg)
        return msg,ret

    def _call(self,query,**kwargs):
        if self.type=='flow':
            msg,ret = self._flow(query,**kwargs)
        elif self.type=='chat':
            msg,ret = self._chat(query,**kwargs)
        else:
            raise ValueError(f'Thread {self.tid}:{self.alias} is not callable: it should be either chatable or runable.')
        assert isinstance(msg,str), f'Message should be a string'
        assert isinstance(ret,dict), f'Return should be a dict'
        return msg, ret
    
    def _pipe(self,input,largs,rargs):
        assert self.type=='pipe', f'Thread {self.tid}:{self.alias} is not a pipe'
        query_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        left_msg,left_ret = self.lthread._call(input,**largs)
        right_msg,right_ret = self.rthread._call(left_msg,**rargs)
        assert isinstance(left_msg,str) and isinstance(right_msg,str), f'Pipe should return message'
        if right_msg != FAILED:
            self._message(self.caller,self.callee,left_msg,query_time)
            self._message(self.callee,SYSTEM_CALLER,right_msg)
        return right_msg,(left_msg,left_ret,right_ret)
    
    def __call__(self,query,**kwargs):
        assert self.type!='empty', f'Thread {self.tid}:{self.alias} is not callable'
        if self.type=='pipe':
            return self._pipe(query,**kwargs)
        else:
            return self._call(query,**kwargs)

    def _fork(self,alias,tid,caller=None,callee=None,context=AgentContext(),note=None): # a fork return a new thread, each thread expose a call and return message to the parent thread, the fork call and return
        self._log('fork',{'tid':tid,'alias':alias,'note':note,'context':context.data}) 
        return AgentDialogThread(alias,tid,self.tid,self.stream,caller,callee,self.log_dir,context)
    
class AgentDialogManager: # all dialogs should be go through the dialog manager
    def __init__(self,log_dir,system_info,stream):
        self.log_dir = log_dir
        self.threads = {}
        self.threads[0] = AgentDialogThread('root',0,-1,stream,log_dir=self.log_dir) # create a root thread
        self.stream = stream
        if log_dir:
            session_id=log_dir.split('/')[-1]
            self.stream.markdown(f'Session created. ID: {session_id}')
            U.save_json(system_info,f"{log_dir}/system_info.json")
    
    def _assign_tid(self):
        return len(self.threads)
    
    def fork(self,parent_tid,caller=None,callee=None,context=AgentContext(),alias=None,note=None):
        parent_thread = self.threads[parent_tid]
        tid = self._assign_tid()
        if not alias: alias = f'thread_{tid}'
        else: alias = f'{tid}_{alias}'
        self.threads[tid] = parent_thread._fork(alias,tid,caller,callee,context=context,note=note)
        return tid
    
    def carry(self,pipe_tid,lthread_tid,rthread_tid):
        LROLE = ROLE(self.threads[lthread_tid].alias,self.threads[lthread_tid])
        RROLE = ROLE(self.threads[rthread_tid].alias,self.threads[rthread_tid])
        self.threads[pipe_tid]._carry(LROLE,RROLE)
    
    def access(self,tid):
        return self.threads[tid]
    
    def call(self,tid,query,**kwargs):
        return self.threads[tid](query,**kwargs)
    
    def context(self,tid):
        return self.threads[tid].history

@dataclass
class DialogTreeNode:
    tid: int
    parent: int
    alias: str
    logs: List[Dict[str,Any]]
    children: list
    fork_note: Optional[str]

    def to_mark(self):
        md=f'# {self.alias}\n'
        for child in self.children:
            childmark=child.to_mark()
            for line in childmark.split('\n'):
                md+=f'#{line}\n'
        return md
    
    def to_timeline(self):
        timeline={}
        timeline['title']={
            'text': {
                'headline': f'Dialog: {self.alias}, ID: {self.tid}',
                'text': f'{markdown.markdown(self.fork_note)}'
            },
        }
        timeline['events']=[]
        for log in self.logs:
            timestamp = log['timestamp'] 
            YMD,HMS = timestamp.split('_')
            Y,M,D = map(int,YMD.split('-'))
            H,min,S = map(int,HMS.split('-'))
            timeobj = {'year':Y,'month':M,'day':D,'hour':H,'minute':min,'second':S}
            if log['type']=='message':
                content=log['data']['content']
                codes = re.findall(r"```python(.*?)```", content, re.DOTALL)
                replace = {}
                for code in codes:
                    mark='CODE_REPLACE_TEMP_'+uuid.uuid4().hex.upper()
                    replace[mark]=code
                    content = content.replace(f"```python{code}```",mark)
                content = markdown.markdown(content)
                for mark,code in replace.items():
                    content = content.replace(mark,f"<code style='display: block; white-space: pre-wrap;'>{code}</code>")
                timeline['events'].append({
                    'start_date': timeobj,
                    'text': {
                        'headline': f"Message from {log['data']['sender']} to {log['data']['receiver']}",
                        'text': f"{content}<p style='float: right;'><b><i>Running cost: {float(log['data']['cost'])}</i></b></p>"
                    }
                })
            elif log['type']=='fork':
                timeline['events'].append({
                    'start_date': timeobj,
                    'text': {
                        'headline': f"Thread forked: {log['data']['alias']}",
                        'text': f"{markdown.markdown(log['data']['note'])}"
                    }
                })
        return timeline


# TODO: update to compactible with the Threads
class DialogTreeViewer: # only for viewing and anlyzing the agents dialogs
    def __init__(self,log_dir):
        self.log_dir = log_dir
        self.threads = {}
        self.system_info = U.load_json(f"{log_dir}/system_info.json")
        self.root = self.load_thread(U.pjoin(log_dir,f'thread_0_root'),0,'root',-1)

    def load_thread(self,log_dir,tid,alias,parent_tid,fork_note=None):
        logs=[]
        childrens=[]
        for log_file in os.listdir(log_dir): # should be already sorted by timestamp
            if log_file.endswith('.json'):
                log=U.load_json(U.pjoin(log_dir,log_file))
                logs.append(log)
                if log['type']=='fork':
                    childtid = log['data']['tid']
                    childalias = log['data']['alias']
                    childnote = log['data']['note']
                    childrens.append(self.load_thread(
                        U.pjoin(log_dir,f'thread_{childtid}_{childalias}'),
                        childtid,childalias,tid,childnote))
        node = DialogTreeNode(tid,parent_tid,alias,logs,childrens,fork_note)
        self.threads[alias] = node
        return node

    def to_markmap(self):
        treemd=self.root.to_mark()
        cleaned_lines=[]
        for line in treemd.split('\n'):
            if line.replace('#','').replace(' ','')!='':
                cleaned_lines.append(line)
        clean_md = '\n'.join(cleaned_lines)
        return clean_md
    

#region Prompt CFG


class AgentDialogFlowNaive:
    def __init__(self,name,prog):
        self.name = name
        self._call = prog

    def __call__(self,query,parent_tid,**kwargs):
        try:
            flow=ft.partial(self._call,query=query,parent_tid=parent_tid)
        except Exception as e:
            raise ValueError(f'Thread function must have query and parent_tid as argument, error: {e}')
        output = flow(**kwargs)
        assert isinstance(output,tuple) and len(output)==2 and isinstance(output[0],str), f'Thread function should return a tuple of (message,return)'
        return output
    
    def export(self,dir):
        source = inspect.getsource(self._call)
        source=U.remove_leading_indent(source)
        fc=pfc.Flowchart.from_code(source)
        
        fc_dir=U.pjoin(dir,f'{self.name.replace(' ','_')}_flowchart.html')
        pfc.output_html(fc_dir,f'Flowchart of {self.name}',fc.flowchart())
        return fc_dir



class AgentFlowNode:
    def __init__(self,id,alias,prog,hints=None) -> None:
        self.id=id
        self.alias=alias
        self.prog=prog
        self.children = {}
        self.hints = hints # a dict of hints for the children written on the edge, tell people what excectly each path means

    def __call__(self,query,state,**kwargs):
        print(f'Calling {self.alias}')
        return self._call(query,state,**kwargs)
    
    def _call(self,query,state,**kwargs):
        raise NotImplementedError

    def link(self,children):
        assert isinstance(children,dict), f'Children must be a dict of flow nodes'
        for id,child in children.items():
            assert isinstance(id,int) and id>=0, f'Children id must be an integer >=0'
            assert isinstance(child,AgentFlowNode), f'Children must be a flow node'
        self.children = children
    
    def inspect(self,remove_indent=True):
        source=inspect.getsource(self.prog)
        if remove_indent:
            source=U.remove_leading_indent(source)
        return source


class CONDNode(AgentFlowNode): # It will check a condition and return true or false, the order of the children is the order of the selections
    """
    A COND node input query and kwargs, output a selection index and updated state, it routes to another block
    COND Node can only have multiple children
    """
    def _call(self,query,state,**kwargs):
        assert self.children!=[], f'CONDNode {self.alias}: COND node cannot be a terminal node'
        input_state = copy.deepcopy(state)  
        rets = self.prog(query,state,**kwargs)
        if isinstance(rets,int):
            ret = rets
        elif isinstance(rets,tuple):
            assert len(rets)==2, f'CONDNode {self.alias}: Condition must return a positive integer, and optionally the updated state'
            ret,state = rets
            assert isinstance(state,dict), f'CONDNode {self.alias}: State must be a dict'
            for key in input_state:
                assert key in state, f'CONDNode {self.alias}: {key} lost in state'
        assert isinstance(ret,int) and ret>=0, f'CONDNode {self.alias}: Condition must return a boolean or a positive integer'
        assert ret<len(self.children), f'CONDNode {self.alias}: Condition must return a value less than the number of selections'
        child = self.children[ret]
        assert isinstance(child,AgentFlowNode), f'CONDNode {self.alias}: Children must be a flow node'
        return child(query,state,**kwargs)

class LOOPNode(AgentFlowNode): # It will loop until a condition is met
    """
    LOOP Node will run a condition which return a boolean and updated state, then run the loop body or exit
    LOOP Node can only have 2 children, the first is the loop body, the second is the exit
    """
    def _call(self,query,state,**kwargs):
        assert len(self.children)==2, f'LOOPNode {self.alias}: Children of a LOOP node must be two, the first is the loop body, the second is the exit'
        # while True: # TODO: now seems just a boolean condition, and full of goto, the loop can easily be lost, need to be more like a loop
        input_state = copy.deepcopy(state)
        rets = self.prog(query,state,**kwargs)
        if isinstance(rets,bool):
            cont = rets
        elif isinstance(rets,tuple):
            assert len(rets)==2, f'LOOPNode {self.alias}: Condition must return a boolean and optionally the updated state'
            cont,state=rets
            assert isinstance(state,dict), f'CONDNode {self.alias}: State must be a dict'
            for key in input_state:
                assert key in state, f'CONDNode {self.alias}: {key} lost in state'
        assert isinstance(cont,bool), f'LOOPNode {self.alias}: Condition must return a boolean'
        if cont:
            # query,state,kwargs = self.children[0](query,state,**kwargs)
            return self.children[0](query,state,**kwargs)
        else:
            return self.children[1](query,state,**kwargs)
    
class PROCNode(AgentFlowNode): # It will call an agent and return a response
    """
    PROC Node will really process the query and update the flow of kwargs, the flow will increment monotonicly
    All nodes can update state but only PROC node can update the query and kwargs
    PROC Node can only have one child
    """
    def _call(self, query,state, **kwargs):
        try:
            query,state,ret = self.prog(query,state,**kwargs)
        except Exception as e:
            raise ValueError(f'Error in PROCNode: {self.alias}: {e}')
        assert isinstance(query,str), f'A PROC node must return a string response message'
        assert isinstance(ret,dict), f'A PROC node must return a dict of additional returns'
        kwargs.update(ret) # update the flow of kwargs
        if self.children!=[]:
            assert len(self.children)==1, f'PROCNode {self.alias}: Children of a PROC node must be one, but got {self.children}'
            child = self.children[0]
            return child(query,state,**kwargs) 
        return query,state,kwargs

class AgentDialogFlow:
    """
    input query and kwargs, output a response message and a dict of additional returns
    """
    def __init__(self,name,args=[],outputs=[],init_state={}):
        self.nodes={}
        self.name=name
        self.args=args
        self.outputs=outputs # what to expect from the return
        self.state=init_state ### global vars, the state allows COND and LOOP pass signals without editing the query and flow of kwargs which is only allowed by PROC 
        self.init_state=copy.deepcopy(init_state)
        id_entry=self.assign_id()
        self.id_entry=id_entry
        self.entry = PROCNode(id_entry,'entry',lambda x,y,**kwargs: (x,y,kwargs))
        self.nodes[id_entry] = self.entry
        self.alias_to_id = {'entry':id_entry}
        self.flow_nodes={}
        self.flow_nodes_simple={}
        self.flow_edges=[]
        self.flow_st=StreamlitFlowNode(str(id_entry),(0,0),{'content':f'###### Entry of the {self.name}'},'input',
                                       source_position='right',target_position='left',
                                       style={'backgroundColor': '#20a162'})
        self.flow_nodes[id_entry] = self.flow_st
        id_input=self.assign_id()
        self.flow_nodes[id_input] = StreamlitFlowNode(str(id_input),(0,0),{'content':f'###### Inputs: query, {args}'},
                                                      source_position='right',target_position='left',
                                                      style={'backgroundColor': '#f0d695'})
        self.flow_edges.append(StreamlitFlowEdge(f'{id_entry}-{id_input}',str(id_entry),str(id_input),animated=True))
        self.nodes_to_flowtail = {id_entry:id_input} # map node id to flowchart node id (for linking)

    def __call__(self,query,**kwargs):
        state = copy.deepcopy(self.init_state)
        kwargs = {} if not kwargs else kwargs
        missing_args=[]
        for arg in self.args:
            if arg not in kwargs:
                missing_args.append(arg)
        assert len(missing_args)==0, f'Missing arguments: {missing_args}'
        query,state,ret = self.entry(query,state,**kwargs)
        out={}
        for output in self.outputs:
            assert output in ret, f'Missing output: {output}'
            out[output] = ret[output]
        self.state = state
        return query,out

    def _new_node(self,alias,prog,type,hints=None,is_end=False):
        assert alias not in self.alias_to_id, f'Alias `{alias}` already exists'
        id=self.assign_id()
        self.nodes_to_flowtail[id] = id
        source=inspect.getsource(prog)
        # source=U.remove_leading_indent(source)
        if type=='PROC':
            self.nodes[id] = PROCNode(id,alias,prog,hints)
            self.flow_nodes[id] = StreamlitFlowNode(str(id),(0,0),{'content':f'###### {alias}\n```python\n{source}\n```'},
                                                                source_position='right',target_position='left',
                                                                style={'textAlign': 'left','backgroundColor':'#c0c4c3'})
            self.flow_nodes_simple[id] = copy.deepcopy(self.flow_nodes[id])
            self.flow_nodes_simple[id].data['content'] = f'###### {alias}'
            if is_end:
                id_output = self.assign_id()
                id_end = self.assign_id()
                self.flow_nodes[id_output] = StreamlitFlowNode(str(id_output),(0,0),{'content':f'###### Outputs: query, {self.outputs}'},
                                                                source_position='right',target_position='left',
                                                                style={'backgroundColor': '#f0d695'})
                self.flow_nodes[id_end] = StreamlitFlowNode(str(id_end),(0,0),{'content':f'###### Exit flow by Node [{alias}]'},'output',
                                                                source_position='right',target_position='left',
                                                                style={'backgroundColor': '#e9d7df'})
                self.flow_edges.append(StreamlitFlowEdge(f'{id}-{id_output}',str(id),str(id_output),animated=True,style={'markerEnd': 'url(#arrow)'}))
                self.flow_edges.append(StreamlitFlowEdge(f'{id_output}-{id_end}',str(id_output),str(id_end),animated=True,style={'markerEnd': 'url(#arrow)'}))
                self.nodes_to_flowtail[id] = id_end
        elif type=='COND':
            self.nodes[id] = CONDNode(id,alias,prog,hints)
            self.flow_nodes[id] = StreamlitFlowNode(str(id),(0,0),{'content':f'###### {alias}\n```python\n{source}\n```'},
                                                                source_position='right',target_position='left',
                                                                style={'textAlign': 'left','backgroundColor':'#b7d07a'})
            self.flow_nodes_simple[id] = copy.deepcopy(self.flow_nodes[id])
            self.flow_nodes_simple[id].data['content'] = f'###### {alias}'
        elif type=='LOOP':
            self.nodes[id] = LOOPNode(id,alias,prog,hints)
            self.flow_nodes[id] = StreamlitFlowNode(str(id),(0,0),{'content':f'###### {alias}\n```python\n{source}\n```'},
                                                                source_position='right',target_position='left',
                                                                style={'textAlign': 'left','backgroundColor':'#f9d367'})
            self.flow_nodes_simple[id] = copy.deepcopy(self.flow_nodes[id])
            self.flow_nodes_simple[id].data['content'] = f'###### {alias}'
            
        self.alias_to_id[alias] = id
        return id
    
    def new_proc(self,alias,prog,hint=None,is_end=False):
        hints = {0:hint} if hint else None
        return self._new_node(alias,prog,'PROC',hints,is_end=is_end)
    def new_cond(self,alias,prog,hints=None):
        return self._new_node(alias,prog,'COND',hints)
    def new_loop(self,alias,prog,hints=None):
        return self._new_node(alias,prog,'LOOP',hints)
    
    def assign_id(self):
        id = len(self.nodes)
        self.nodes[id] = None # placeholder
        return id

    def link(self,id_or_alias,children):
        if isinstance(children,AgentFlowNode):
            children = {0:children}
        elif isinstance(children,list):
            _children = {}
            for i,child in enumerate(children):
                if isinstance(child,AgentFlowNode):
                    _children[i] = child
                elif isinstance(child,int):
                    assert child in self.nodes, f'Children id {child} does not exist'
                    _children[i] = self.nodes[child]
                elif isinstance(child,str):
                    assert child in self.alias_to_id, f'Children alias {child} does not exist'
                    _children[i] = self.nodes[self.alias_to_id[child]]
                else:
                    raise ValueError(f'Children must be a dict of flow nodes, or a list of flow nodes, or a single flow node')
            children = _children
        elif isinstance(children,dict):
            for i,child in children.items():
                assert isinstance(i,int) and i>=0, f'Children id must be a positive integer if dict is provided'
                if isinstance(child,AgentFlowNode):
                    pass
                elif isinstance(child,int):
                    assert child in self.nodes, f'Children id {child} does not exist'
                    children[i] = self.nodes[child]
                elif isinstance(child,str):
                    assert child in self.alias_to_id, f'Children alias {child} does not exist'
                    children[i] = self.nodes[self.alias_to_id[child]]
                else:
                    raise ValueError(f'Children must be a dict of flow nodes, or a list of flow nodes, or a single flow node')
        elif isinstance(children,int):
            assert children in self.nodes, f'Children id {children} does not exist'
            children = {0:self.nodes[children]}
        elif isinstance(children,str):
            assert children in self.alias_to_id, f'Children alias {children} does not exist'
            children = {0:self.nodes[self.alias_to_id[children]]}
        else:
            raise ValueError(f'Children must be a dict of flow nodes, or a list of flow nodes, or a single flow node')
        if isinstance(id_or_alias,str):
            id = self.alias_to_id[id_or_alias]
        else:
            id = id_or_alias
        self.nodes[id].link(children)
        node=self.nodes[id]
        flow_id=self.nodes_to_flowtail[id]
        if isinstance(node,PROCNode):
            assert len(children)==1, f'PROCNode {node.alias}-{node.id}: Children of a PROC node must be one'
            child = children[0]
            hint = node.hints[0] if node.hints else None
            self.flow_edges.append(StreamlitFlowEdge(f'{flow_id}-{child.id}',str(flow_id),str(child.id),animated=True,label=hint))
        elif isinstance(node,CONDNode):
            assert len(children)>1, f'CONDNode {node.alias}-{node.id}: COND node cannot be a terminal node'
            hints = node.hints
            for i,child in children.items():
                label = f'{hints[i]}' if hints else f'Selection {i}'
                self.flow_edges.append(StreamlitFlowEdge(f'{flow_id}-{child.id}',str(flow_id),str(child.id),animated=True,label=label))
        elif isinstance(node,LOOPNode):
            assert len(children)==2, f'LOOPNode {node.alias}-{node.id}: Children of a LOOP node must be two, the first is the loop body, the second is the exit'
            hint0,hint1 = (node.hints[0],node.hints[1]) if node.hints else ('Loop Body','Exit')
            self.flow_edges.append(StreamlitFlowEdge(f'{flow_id}-{children[0].id}',str(flow_id),str(children[0].id),animated=True,label=hint0))
            self.flow_edges.append(StreamlitFlowEdge(f'{flow_id}-{children[1].id}',str(flow_id),str(children[1].id),animated=True,label=hint1))

    def export(self,height=1000,simplify=False):
        if simplify:
            for id in self.flow_nodes:
                if id not in self.flow_nodes_simple:
                    self.flow_nodes_simple[id] = copy.deepcopy(self.flow_nodes[id])
            flow_nodes = self.flow_nodes_simple
            horizontal_spacing=300
            vertical_spacing=75
            node_node_spacing=150
        else:
            flow_nodes = self.flow_nodes
            horizontal_spacing=300
            vertical_spacing=150
            node_node_spacing=450
        return streamlit_flow(
            self.name, 
            list(flow_nodes.values()), 
            self.flow_edges, 
            layout=TreeLayout(
                "right",
                horizontal_spacing=horizontal_spacing,
                vertical_spacing=vertical_spacing,
                node_node_spacing=node_node_spacing,
            ), 
            fit_view=True, 
            height=height, 
            enable_node_menu=True, 
            show_minimap=True, 
            enable_pane_menu=True, 
            hide_watermark=True, 
            allow_new_edges=True, 
            get_node_on_click=True,
            # get_edge_on_click=True,
            min_zoom=0.1
        )
    
#endregion

class ALangCompiler:
    """
    Agent Language for Agent Dialog Flow Definition, the key idea is that an agent dialog flow is an assembly line for processing the message.
    This assembly line is composed of PROC node that can really process the message and two type of control nodes: COND and LOOP, one for branching and the other for looping.
    Right now, the language is super primitive, the advantage is simply make the design super modularized and easy to understand, the disadvantage is that it is not very flexible and powerful.
    Main issue it that the definition of the modules are not so elegant, and the grammers are simply instructions.
    
    
    Primitive calls:

    FLOW name arg0|arg1|... output0|output1|... # must be the first line and only once
    PROC node alias prog [hint] # node is var name of the node
    EXIT node alias prog [hint]
    COND node alias prog [hint0|hint1|...]
    LOOP node alias prog [hintLoop|hintExit] # TODO: need to improve, right now it is essentially a COND node and you need to manually go back
    LINK or ->: node_or_alias -> chil0|child1|... # child can be node name or alias

    Other system calls not implemented yet:
    FORK, PIPE

    - name, alias and hints must be bracketed by ``
    - ENTRY is a special predefined constant for the entry node
    - -> and | are special characters, never use them in name, alias or hints

    flow = Acompiler.compile(ALANG, modules) # the compiler returns an AgentDialogFlow object
    or Acompiler.compile(ALANG, modules, init_state)
    or Acompiler.compile(build_flow_func) where build_flow_func has all module definitions and return the ALANG string and optionally the init_state
    """
    def _create_flow(self,line,init_state):
        _line,maps = self._preprocess_line(line)
        parts = _line.split(' ')
        assert len(parts)==4, f'A flow definition line must have 4 parts, found {len(parts)}, line: {line}'
        _,name,args,outputs = parts
        name = maps[name]
        flow = AgentDialogFlow(name,args.split('|'),outputs.split('|'),init_state)
        self._nodes['ENTRY']=flow.id_entry
        self._aliases['ENTRY']=flow.entry.alias
        return flow

    def _preprocess_line(self,line):
        line = self.remove_comments(line)
        maps = {}
        for i,match in enumerate(re.finditer(r'`[^`]+`',line)):
            mark = f'BRACKET_REPLACE_TEMP_{i}'
            maps[mark] = match.group()
            line = line.replace(match.group(),mark)
        return line,maps
    
    def map_back(self,var,map):
        if var in map:
            return map[var][1:-1]
        return var

    def _parse_node(self,line):
        _line,maps = self._preprocess_line(line)
        splits = _line.split(' ')
        assert len(splits)==4 or len(splits)==5, f'A node definition line must have 4 or 5 parts, found {len(splits)}, line: "{line}"'
        for i in range(len(splits)):
            splits[i] = splits[i].strip()
        nodetype,name,_alias,prog = splits[:4]
        alias = self.map_back(_alias,maps)
        is_end = False
        if nodetype=='EXIT':
            nodetype='PROC'
            is_end=True
        hints={}
        if len(splits)==5:
            for i,hint in enumerate(splits[4].split('|')):
                hints[i] = self.map_back(hint,maps)
        prog = self._modules[prog]
        self._nodes[name]=self._flow._new_node(alias,prog,nodetype,hints,is_end)
        self._aliases[name]=alias

    def _parse_link(self,line):
        _line,maps = self._preprocess_line(line)
        if line.startswith('LINK'):
            splits = _line.split(' ')
            assert len(splits)==3, f'A link definition line must have 3 parts, found {len(splits)}, line: "{line}"'
            node_or_alias,_children = splits[1:]
        else:
            node_or_alias,_children = _line.split('->')
        node_or_alias = node_or_alias.strip()
        _children = _children.strip()
        if node_or_alias in maps:
            node_or_alias = self.map_back(node_or_alias,maps)
        children = []
        for child in _children.split('|'):
            child=self.map_back(child,maps)
            if child in self._nodes:
                child = self._nodes[child]
            children.append(child)
        if node_or_alias in self._nodes:
            node_or_alias = self._nodes[node_or_alias]
        self._flow.link(node_or_alias,children)

    def remove_comments(self,line):
        return line.split('#')[0].strip()

    def _convert(self,ALANG,init_state):
        links_def=[]
        nodes_def=[]
        ALANG_reformated = '# Flow Definition\n'
        for i,line in enumerate(ALANG.split('\n')):
            line = line.strip()
            if line=='': continue
            if i==0:
                assert line.startswith('FLOW'), f'ALANG must start with FLOW definition'
                self._flow = self._create_flow(line,init_state)
                ALANG_reformated += line+'\n\n'
            else:
                if line.startswith('PROC') or line.startswith('COND') or line.startswith('LOOP') or line.startswith('EXIT'):
                    assert '->' not in line, f'ALANG line {i+1}: "{line}"\nDo not use -> besides define a LINK'
                    try:
                        self._parse_node(line)
                    except Exception as e:
                        raise ValueError(f'ALANG line {i+1}: "{line}"\n{e}')
                    nodes_def.append(line)
                elif line.startswith('LINK') or '->' in line:
                    if line.startswith('LINK'):
                        assert '->' not in line, f'ALANG line {i+1}: "{line}"\nUse either LINK or ->, not both'
                    try:
                        self._parse_link(line)
                    except Exception as e:
                        raise ValueError(f'ALANG line {i+1}: "{line}"\n{e}')
                    links_def.append(line)
                elif line.startswith('#'):
                    pass # ignore it
                else:
                    if line.startswith('FLOW'):
                        raise ValueError(f'ALANG line {i+1}: "{line}"\nFLOW can only be the first line')
                    else:
                        raise ValueError(f'ALANG line {i+1}: "{line}"\nInvalid syntax, must be PROC, COND, LOOP, LINK or EXIT')
        ALANG_reformated+='# Node Definitions\n'
        ALANG_reformated+='\n'.join(nodes_def)+'\n\n'
        ALANG_reformated+='# Link Definitions\n'
        ALANG_reformated+='\n'.join(links_def)
        return ALANG_reformated

    def _fn_to_modules(self,func):
        _modules = {}
        for const in func.__code__.co_consts:
            if isinstance(const, CodeType):
                function_name = const.co_name
                function_obj = FunctionType(const, globals())
                _modules[function_name] = function_obj
        return _modules
    
    def _module_to_modules(self,module):
        return {name: obj for name, obj in vars(module).items() if callable(obj)}

    def _source_to_modules(self,source):
        # if it is a path of a file
        if U.pexists(source):
            source = U.read_file(source)
        ns = {}
        try:
            exec(source, ns)
        except Exception as e:
            raise ValueError(f'Error in source you provided as modules: {e}')
        return {name: obj for name, obj in ns.items() if callable(obj)}

    def compile(self,ALANG,modules,init_state={},reformat=False):
        self._flow=None
        self._nodes={} # node var name to id
        self._aliases={} # node var name to alias
        if isinstance(modules,FunctionType):
            self._modules = self._fn_to_modules(modules)
        elif isinstance(modules,dict):
            for i,fn in modules.items():
                assert isinstance(fn,FunctionType), f'Modules must be a dict of functions, found {type(fn)} for {i}'
                assert i==fn.__name__, f'Module function name must be the same as the key, found {fn.__name__} for {i}'
            self._modules=modules
        elif isinstance(modules,ModuleType):
            self._modules = self._module_to_modules(modules)
        elif isinstance(modules,str): # path or source code
            self._modules = self._source_to_modules(modules)
        else:
            raise ValueError(f'Type of modules not supported: {type(modules)}, currently supported types are: a function with all module definitions, dict of modules, an imported module where the functions are defined, str of source code, path to the source file')
        ALANG_reformated=self._convert(ALANG,init_state)
        if reformat:
            return self._flow,ALANG_reformated
        return self._flow



def design_flow_definition():
    # args=['cls','stream','status_handler','parent_tid','context']
    # outputs=['code','text','check_results']
    # design_flow = AgentDialogFlow(name='Model Design Flow',args=args,outputs=outputs)

    ALANG = 'FLOW `Model Design Flow` cls|stream|status_handler|parent_tid|context code|text|check_results\n'


    def design_initializer(query,state,cls,parent_tid,context,**kwargs):
        assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        state['initial_error'] = None
        state['design_attemps'] = 0
        DESIGNER = ROLE('designer',cls.designer)
        design_thread_tid=cls.dialog.fork(parent_tid,SYSTEM_CALLER,DESIGNER,context=context,
                                            alias='designing',note=f'Starting design...')
        debug_thread_tid=None
        return query,state,{'design_thread_tid':design_thread_tid,'debug_thread_tid':debug_thread_tid}
    # init_design_node = design_flow.new_proc('Initialize Design Flow',design_initializer)
    # design_flow.link(design_flow.id_entry,init_design_node)
    ALANG += 'PROC init_design_node `Initialize Design Flow` design_initializer\n'
    ALANG += 'ENTRY -> init_design_node\n'


    def design_loop_controller(query,state,cls,**kwargs):
        assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        cont = state['design_attemps'] < cls._config.max_design_attempts
        attempt = state['design_attemps']
        cls.logging.info(f'Attempting design, attempt={attempt}')
        state['design_attemps'] += 1
        return cont,state
    # design_loop_controller_node = design_flow.new_loop('Design Loop Controler',design_loop_controller,hints={0:'Enter the design loop',1:'The design loop should terminate'})
    # design_flow.link(init_design_node,design_loop_controller_node)
    ALANG += 'LOOP design_loop_controller_node `Design Loop Controler` design_loop_controller `Enter the design loop`|`The design loop should terminate`\n'
    ALANG += 'init_design_node -> design_loop_controller_node\n'

    def design_thread_switch(query,state,**kwargs):
        attempt = state['design_attemps']
        state['current_thread']=kwargs['design_thread_tid']
        return 0 if attempt == 1 else 1,state
    # design_switch_node = design_flow.new_cond('Design Switch',design_thread_switch,{0:'Sample initial design',1:'Debug the design'})
    ALANG += 'COND design_switch_node `Design Switch` design_thread_switch `Sample initial design`|`Debug the design`\n'

    def switch_to_debug(query,state,text,cls,**kwargs):
        assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        debug_thread_tid = kwargs['debug_thread_tid']
        if debug_thread_tid is None:
            query = f'The designer designed the model: {text}\n\nThe checker failed the model: {query}\n\nPlease debug the model.'
            DEBUGGER = ROLE('debugger',cls.debugger)
            debug_thread_tid = cls.dialog.fork(state['current_thread'],SYSTEM_CALLER,DEBUGGER,
                                                alias='debugging',note='Starting debugging...')
        state['current_thread']=debug_thread_tid
        return query,state,{'debug_thread_tid':debug_thread_tid}
    # switch_to_debug_node = design_flow.new_proc('Switch to Debug Thread',switch_to_debug,hint='Debugger take over')
    ALANG += 'PROC switch_to_debug_node `Switch to Debug Thread` switch_to_debug `Debugger take over`\n'

    # Define design loop body
    def design_loop_body(query,state,cls,status_handler,stream,**kwargs):
        assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        attempt = state['design_attemps']
        thread_tid = state['current_thread']
        # Block 2: Input thread_tid, query, and get checked code
        with status_handler(f"Design Attempt {attempt+1}"): 
            
            # BLOCK 
            _,out=cls.dialog.call(thread_tid,query)

            try:
                code = out.get("code",None)
                text = out.get("text")
                assert code is not None
                assert "# gab.py" in code 
            
                if stream: #and code:
                    stream.write('Model authored code block...')
                    stream.markdown(str(text))
                generated = True
            except AssertionError:
                generated = False
        ret={'code':code,'text':text,'generated':generated}
        return query,state,ret
    # design_loop_body_node = design_flow.new_proc('Design Loop Body',design_loop_body,hint='Agent response')
    # design_flow.link(design_switch_node,{0:design_loop_body_node,1:switch_to_debug_node})
    # design_flow.link(switch_to_debug_node,design_loop_body_node)
    ALANG += 'PROC design_loop_body_node `Design Loop Body` design_loop_body `Agent response`\n'
    ALANG += 'design_switch_node -> design_loop_body_node|switch_to_debug_node\n'
    ALANG += 'switch_to_debug_node -> design_loop_body_node\n'

    def gocheck_or_goback(query,state,generated,**kwargs):
        return 1 if generated else 0
    # gocheck_or_goback_node = design_flow.new_cond('Whether code is generated?',gocheck_or_goback,{0:'No, go back and retry',1:'Yes, pass to the checker'})
    ALANG += 'COND gocheck_or_goback_node `Whether code is generated?` gocheck_or_goback `No, go back and retry`|`Yes, pass to the checker`\n'

    def check_design(query,state,cls,stream,code,**kwargs):
        assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        attempt = state['design_attemps']
        design_name = f"{cls._config.run_name}_{attempt}"
        checkpass,check_report,code,check_results = cls.checker.check(cls._cfg,code,design_name)
        checker_hints = check_results['hints']
        if 'REFRESH_TEMPLATE' in checker_hints:
            cls.sess_state['refresh_template'] += 1
        
        if stream:
            stream.write(
                f"""<details><summary>code check</summary>{check_report}</details>""",
                unsafe_allow_html=True
            )
        ret={'checkpass':checkpass,'check_report':check_report,'code':code,'check_results':check_results}
        return query,state,ret
    # check_design_node = design_flow.new_proc('Checker checks design',check_design)
    # design_flow.link(design_loop_body_node,gocheck_or_goback_node)
    # design_flow.link(gocheck_or_goback_node,{0:design_loop_controller_node,1:check_design_node})
    ALANG += 'PROC check_design_node `Checker checks design` check_design\n'
    ALANG += 'design_loop_body_node -> gocheck_or_goback_node\n'
    ALANG += 'gocheck_or_goback_node -> design_loop_controller_node|check_design_node\n'

    def check_pass(query,state,checkpass,**kwargs):
        return 1 if checkpass else 0
    # check_pass_node = design_flow.new_cond('Pass or not?',check_pass,{0:'Failed, retry or end of the loop.',1:'Passed, go to report generation.'})
    # design_flow.link(check_design_node,check_pass_node)
    ALANG += 'COND check_pass_node `Pass or not?` check_pass `Failed, retry or end of the loop.`|`Passed, go to report generation.`\n'
    ALANG += 'check_design_node -> check_pass_node\n'

    def design_failed(query,state,cls,check_report,**kwargs):
        assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        query = f"The designed model didn't pass, you need to try again. Here is the report:\n{check_report}. Please fix."
        if cls.sess_state['refresh_template'] >=1:
            query+=f'\nHere is the template for the GAB block for you to refresh:\n\n```python\n{cls.gab_py}```'
        if cls.sess_state['refresh_template'] >= 2:
            query+=f'\nHere is the definition of GABBase for you to refresh:\n\n```python\n{GAB_BASE}```'
        if cls.sess_state['refresh_template'] >= 3:
            query+=f'\nHere is the definition for the GAM model for you to refresh:\n\n```python\n{cls.gam_py}```'
        return query,state,{}
    # design_failed_node = design_flow.new_proc('Design failed prompt',design_failed,hint='Prompt to retry or end of the loop.')
    # design_flow.link(design_failed_node,design_loop_controller_node)
    ALANG += 'PROC design_failed_node `Design failed prompt` design_failed `Prompt to retry or end of the loop.`\n'
    ALANG += 'design_failed_node -> design_loop_controller_node\n'

    def design_succeed_return(query,state,cls,check_results,status_handler,code,stream,**kwargs):
        assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        design_thread_tid = kwargs['design_thread_tid']
        initial_error = state['initial_error']
        report_query = (
            "The designed model passed the tests, now please generate a text report explaining and justifying your design."
            " Generate a creative name of your design as the title of your report in the first line of your response."
            " Do not include abbreviations or acronyms of your design in the title. You can use them in the body of the report."
            f" Here is the code of the designed model after degugging:\n\n{code}" 
            # FIXME: what is the code after debugging is not the same as the idea before debugging
        )
        if initial_error is not None:
            error_info=(
                f"Your design didn't pass the checker initially:\n\n{initial_error}"
                f"\n\nIt has been fixed by the assistant already as follows:\n\n{code}"
            )
            report_query = f"{error_info}\n\n{report_query}"
        with status_handler(f"Querying agent for report..."):
            cls.logging.info('Now trying to compile self report...')
            explain,_ = cls.dialog.call(design_thread_tid,report_query)
            if stream:
                stream.markdown(explain) #<-- change

        proposal=f'{explain}\n\nImplementation:\n\n{code}\n\n'
        return proposal, state, {'code':code,'text':explain,'check_results':check_results}
    # design_succeed_node = design_flow.new_proc('Design succeed & report generation',design_succeed_return, is_end=True)
    # design_flow.link(check_pass_node,{0:design_failed_node,1:design_succeed_node})
    ALANG += 'EXIT design_succeed_node `Design succeed & report generation` design_succeed_return\n'
    ALANG += 'check_pass_node -> design_failed_node|design_succeed_node\n'

    def design_terminal_check(query,state,checkpass,**kwargs):
        return 1 if checkpass else 0
    # design_terminal_check_node = design_flow.new_cond('Loop terminated.',design_terminal_check,{0:'Design failed',1:'Design succeed'})
    ALANG += 'COND design_terminal_check_node `Loop terminated.` design_terminal_check `Design failed`|`Design succeed`\n'

    def design_failure_exit(query,state,**kwargs):
        return FAILED,state,{'code':None,'text':None,'check_results':None}
    # design_failure_exit_node = design_flow.new_proc('Exit with failure',design_failure_exit, hint='Output FAILED and "None"s', is_end=True)
    # design_flow.link(design_terminal_check_node,{0:design_failure_exit_node,1:design_succeed_node})
    ALANG += 'EXIT design_failure_exit_node `Exit with failure` design_failure_exit `Output FAILED and Nones`\n'
    ALANG += 'design_terminal_check_node -> design_failure_exit_node|design_succeed_node\n'
    
    # design_flow.link(design_loop_controller_node,{0:design_switch_node,1:design_terminal_check_node})
    ALANG += 'design_loop_controller_node -> design_switch_node|design_terminal_check_node\n'
    return ALANG #, design_flow # Notice that this return is not required, the compiler only looks at the functions defined within



def _design_naive(cls,query,stream,status_handler,parent_tid,context): # input query, context, output design and explanation, thread_tid is the id of the thread in which the design is running
    assert isinstance(cls,ModelDiscoverySystem), f'cls must be an instance of ModelDiscoverySystem'
    initial_error = None
    DESIGNER = ROLE('designer',cls.designer)
    DEBUGGER = ROLE('debugger',cls.debugger)
    design_thread_tid=cls.dialog.fork(parent_tid,SYSTEM_CALLER,DESIGNER,context=context,
                                        alias='designing',note=f'Starting design...')
    debug_thread_tid=None
    for attempt in range(cls._config.max_design_attempts):
        cls.logging.info(f'Attempting design, attempt={attempt}')
        
        if attempt == 0:
            thread_tid = design_thread_tid
        else:
            if debug_thread_tid is None:
                query = f'The designer designed the model: {text}\n\nThe checker failed the model: {query}\n\nPlease debug the model.'
                debug_thread_tid = cls.dialog.fork(design_thread_tid,SYSTEM_CALLER,DEBUGGER,
                                                    alias='debugging',note='Starting debugging...')
            thread_tid = debug_thread_tid

        # Block 2: Input thread_tid, query, and get checked code
        with status_handler(f"Design Attempt {attempt+1}"): 
            
            # BLOCK 
            _,out=cls.dialog.call(thread_tid,query)

            try:
                code = out.get("code",None)
                text = out.get("text")
                assert code is not None
            
                if stream: #and code:
                    stream.write('Model authored code block...')
                    stream.markdown(str(text))
                success = True
            except AssertionError:
                success = False
            
            if not success:  # if the code is not generated,
                query  = GAB_ERROR
                continue 

            design_name = f"{cls._config.run_name}_{attempt}"
            checkpass,check_report,code,check_results = cls.checker.check(cls._cfg,code,design_name)
            checker_hints = check_results['hints']
            if 'REFRESH_TEMPLATE' in checker_hints:
                cls.sess_state['refresh_template'] += 1
            
            if stream:
                stream.write(
                    f"""<details><summary>code check</summary>{check_report}</details>""",
                    unsafe_allow_html=True
                )

        # COND block
        if checkpass:  
            break # goto next block
        else:
            # BLOCK return constant, input attempt, output initial_error query
            query = f"The designed model didn't pass, you need to try again. Here is the report:\n{check_report}. Please fix."
            if cls.sess_state['refresh_template'] >=1:
                query+=f'\nHere is the template for the GAB block for you to refresh:\n\n```python\n{cls.gab_py}```'
            if cls.sess_state['refresh_template'] >= 2:
                query+=f'\nHere is the definition of GABBase for you to refresh:\n\n```python\n{GAB_BASE}```'
            if cls.sess_state['refresh_template'] >= 3:
                query+=f'\nHere is the definition for the GAM model for you to refresh:\n\n```python\n{cls.gam_py}```'
            if attempt == 0:
                initial_error = check_report

    if checkpass:
        report_query = (
            "The designed model passed the tests, now please generate a text report explaining and justifying your design."
            " Generate a creative name of your design as the title of your report in the first line of your response."
            " Do not include abbreviations or acronyms of your design in the title. You can use them in the body of the report."
            f" Here is the code of the designed model after degugging:\n\n{code}" 
            # FIXME: what is the code after debugging is not the same as the idea before debugging
        )
        if initial_error is not None:
            error_info=f"Your design didn't pass the checker initially:\n\n{initial_error}\n\nIt has been fixed by the assistant already as follows:\n\n{code}"
            report_query = f"{error_info}\n\n{report_query}"
        with status_handler(f"Querying agent for report..."):
            cls.logging.info('Now trying to compile self report...')
            explain,_ = cls.dialog.call(design_thread_tid,report_query)
            if stream:
                stream.markdown(explain) #<-- change

        proposal=f'{explain}\n\nImplementation:\n\n{code}\n\n'
        return proposal, {'code':code,'text':explain,'check_results':check_results}
    else:
        return FAILED, {'code':None,'text':None,'check_results':None}
    

def _review_naive(cls,query,stream,status_handler,parent_tid,context): 
    if query == FAILED:
        return FAILED, {'review_pass':False,'ratings':None,'reviews':None}
    reviewer_query = REVIEWER_PROMPT.format(
        proposal=query,
        gab_base=GAB_BASE,
        gam_py=cls.gam_py,
        instruct='' # TODO: add references from S2 etc.
    )
    reviews = {}
    ratings = {}
    for style in cls.reviewers:
        with status_handler(f"Querying {style} reviewer for review..."):
            REVIWER_CALLEE = ROLE('reviewer',cls.reviewers[style])
            review_thread_tid = cls.dialog.fork(parent_tid,SYSTEM_CALLER,REVIWER_CALLEE,note=f'Starting review process: {style}')
            _,response = cls.dialog.call(review_thread_tid,reviewer_query)

            if stream:
                stream.write(f'Review of {style} reviewer...')
                stream.markdown(response["text"])
            ratings[style] = response["rating"]
            reviews[style] = response["review"]
    review_ratings=''
    for idx, style in enumerate(cls.reviewers):
        review=reviews[style]
        rating=ratings[style]
        review_ratings+=f'# Review of Reviewer {idx+1}:\n\n{review}\n\n## Rating: {rating} out of 5\n\n'
    rating=np.mean(list(ratings.values()))
    review_pass = rating >= cls._config.reviewer_threshold
    if review_pass:
        response=f'The design passed the review process with an average rating of {rating} out of 5. Review details:\n\n{review_ratings}'
    else:
        response=f'The design didn\'t pass the review process with an average rating of {rating} out of 5. Review details:\n\n{review_ratings}'
    return response,{'review_pass':review_pass,'ratings':ratings,'reviews':reviews}


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

        self.review_flow=AgentDialogFlowNaive('Model Review Flow',_review_naive)
        self.DESIGN_ALANG =design_flow_definition()
        self.design_flow,self.DESIGN_ALANG_reformatted=ALangCompiler().compile(self.DESIGN_ALANG,design_flow_definition,reformat=True)

        self.design_flow_naive=AgentDialogFlowNaive('Model Design Flow',_design_naive)


    def get_system_info(self):
        system_info = {}
        system_info['agents']={
            'designer':self.designer.config,
            'reviewers':{style:agent.config for style,agent in self.reviewers.items()},
            'debugger':self.debugger.config
        }

    def new_session(self,log_dir=None,stream=None):
        U.mkdir(log_dir)
        self.log_dir = log_dir
        self.sess_state = {} 
        self.sess_state['refresh_template'] = 0 
        self.dialog = AgentDialogManager(log_dir,self.get_system_info(),stream)

    def set_config(self,cfg:GAMConfig): # reset the gam config of the system
        self._cfg = cfg

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
        status_handler = stream.status if stream and status else EmptyHandler
        if stream is None:# and self._config.debug_steps:
            stream = PrintSystem(self._config)
        self.new_session(log_dir,stream)
        main_tid = self.dialog.fork(0,note='Starting a new session...',alias='main')

        design_query = DESIGNER_PROMPT.format(
            gab_base=GAB_BASE,
            gam_py=self.gam_py,
            gab_py=self.gab_py,
            config=self._cfg.to_prompt(), #<--- need to parameterize 
            instruct=query,
        )
        query=design_query
        
        refine_pipe_tid = self.dialog.fork(main_tid,SYSTEM_CALLER,SYSTEM_CALLER,note='Design refinement pipe.',alias='refine')
        for i in range(self._config.max_design_refines):
            DESIGN_CALLEE = ROLE('designer',self.design_flow)
            design_pipe_tid = self.dialog.fork(refine_pipe_tid,SYSTEM_CALLER,DESIGN_CALLEE,note=f'launch design flow',alias=f'design_{i}')
            REVIEW_CALLEE = ROLE('reviewer',self.review_flow) 
            review_pipe_tid = self.dialog.fork(refine_pipe_tid,SYSTEM_CALLER,REVIEW_CALLEE,note=f'launch review flow',alias=f'review_{i}')
            self.dialog.carry(refine_pipe_tid,design_pipe_tid,review_pipe_tid)
            rres,(lres,lret,rret) = self.dialog.call(refine_pipe_tid,query,
                                                     largs={'cls':self,'stream':stream,'status_handler':status_handler,'context':self.dialog.context(refine_pipe_tid)},
                                                     rargs={'cls':self,'stream':stream,'status_handler':status_handler,'context':None})
            review_pass,ratings,reviews = rret['review_pass'],rret['ratings'],rret['reviews']
            code,explain,check_results = lret['code'],lret['text'],lret['check_results']
            if lres == FAILED:
                query = design_query
            else:
                query=rres


            if review_pass: break

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
            SUMMARY_CALLER = ROLE('designer',self.designer)
            summary_thread_tid = self.dialog.fork(main_tid,SYSTEM_CALLER,SUMMARY_CALLER,note='Starting summary process...')
            _,response = self.dialog.call(summary_thread_tid,query=summary_query)
            summary=response['text']
            if stream:
                stream.markdown(summary)
        
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



