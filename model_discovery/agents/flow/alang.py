import os
import time
from dataclasses import dataclass
from datetime import datetime
import tempfile
import copy
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
from types import ModuleType, CodeType, FunctionType, MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

from exec_utils.models import SimpleLMAgent
from ..agent_utils import ModelOutputPlus
import model_discovery.utils as U

FAILED = "FAILED"



@dataclass
class AgentPrompt:
    prompt: str
    parser: Any = None
    format: Any = None

    def __call__(self,**kwargs):
        return self.prompt.format(**kwargs)
    
    def parse(self,raw_output: ModelOutputPlus) -> Dict[Any,Any]:
        if not self.parser:
            raw_text = raw_output.text
            output = {}
            output["text"] = raw_text
            output["_details"] = {}
            output["_details"]["cost"] = raw_output.usage
            output["_details"]["running_cost"] = raw_output.usage['cost']
            return output
        return self.parser(raw_output)
    
    def apply(self,agent,logprobs=False):
        agent.parse_output = self.parse
        agent.response_format = self.format
        agent.logprobs = logprobs
        return agent

class AgentContext:
    def __init__(self):
        self.data = []
    
    def append(self,query,role,metadata):
        assert isinstance(query,str) and isinstance(role,str), f'Query and role must be strings'
        self.data.append((query,role,metadata))
    
    def get(self):
        return [(query,role) for query,role,_ in self.data] 

    def truncate(self,n):
        # keep the last n messages and optionally the system message
        _self=AgentContext()
        _self.data=self.data[-n:]
        return _self


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
USER_CALLER = ROLE('user',role='user')

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
            query=input,
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
            self.stream.markdown(f'Session created. Save path: {log_dir}')
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

    def get_alias(self,tid):
        return self.threads[tid].alias
    
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

def inspect_module(prog):
    if isinstance(prog,ft.partial):
        source=inspect.getsource(prog.func)
    else:
        source=inspect.getsource(prog)
    
    return source


class AgentDialogFlowNaive:
    def __init__(self,name,prog):
        self.name = name
        self.prog = prog

    def __call__(self,query,parent_tid,**kwargs):
        try:
            flow=ft.partial(self.prog,query=query,parent_tid=parent_tid)
        except Exception as e:
            raise ValueError(f'Thread function must have query and parent_tid as argument, error: {e}')
        output = flow(**kwargs)
        assert isinstance(output,tuple) and len(output)==2 and isinstance(output[0],str), f'Thread function should return a tuple of (message,return)'
        return output
    
    def export(self,dir):
        source=inspect_module(self.prog)
        source=U.remove_leading_indent(source)
        fc=pfc.Flowchart.from_code(source)
        
        fc_dir=U.pjoin(dir,f'{self.name.replace(" ","_")}_flowchart.html')
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
        source=inspect_module(self.prog)
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
        _params = inspect.signature(self.prog).parameters
        _kwargs = {k: v for k, v in kwargs.items() if k in _params}
        rets = self.prog(query=query,state=state,**_kwargs)
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
        _params = inspect.signature(self.prog).parameters
        _kwargs = {k: v for k, v in kwargs.items() if k in _params}
        rets = self.prog(query=query,state=state,**_kwargs)
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
        # try:
        _params = inspect.signature(self.prog).parameters
        _kwargs = {k: v for k, v in kwargs.items() if k in _params}
        query,state,ret = self.prog(query=query,state=state,**_kwargs)
        # except Exception as e:
        #     raise ValueError(f'Error in PROCNode: {self.alias}: {e}')
        assert isinstance(query,str), f'A PROC node must return a string response message'
        assert isinstance(ret,dict), f'A PROC node must return a dict of additional returns'
        kwargs.update(ret) # update the flow of kwargs
        if self.children!={}:
            assert len(self.children)==1, f'PROCNode {self.alias}: Children of a PROC node must be one, but got {self.children}'
            child = self.children[0]
            return child(query,state,**kwargs) 
        return query,state,kwargs

class AgentDialogFlow:
    """
    input query and kwargs, output a response message and a dict of additional returns
    """
    def __init__(self,name,args=[],outs=[],init_state={}):
        self.nodes={}
        self.name=name
        self.args=args
        self.outs=outs # what to expect from the return
        self.state=init_state ### global vars, the state allows COND and LOOP pass signals without editing the query and flow of kwargs which is only allowed by PROC 
        self.init_state=copy.deepcopy(init_state)
        id_entry=self.assign_id()
        self.id_entry=id_entry
        self.entry = PROCNode(id_entry,'entry',lambda query,state,**kwargs: (query,state,kwargs))
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
        for output in self.outs:
            assert output in ret, f'Missing output: {output}'
            out[output] = ret[output]
        self.state = state
        return query,out

    def _new_node(self,alias,prog,type,hints=None,is_end=False):
        assert alias not in self.alias_to_id, f'Alias `{alias}` already exists'
        id=self.assign_id()
        self.nodes_to_flowtail[id] = id
        source=inspect_module(prog)
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
                self.flow_nodes[id_output] = StreamlitFlowNode(str(id_output),(0,0),{'content':f'###### Outputs: query, {self.outs}'},
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

    def export(self,height=1000,simplify=False,light_mode=False):
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
        style = {}
        if light_mode:
            style = {'backgroundColor': '#f0f0f0', 'textColor': '#000000'}
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
            min_zoom=0.1,
            style=style
        )
    
#endregion

class ALangCompiler:
    """
    Agent Language for Agent Dialog Flow Definition, the key idea is that an agent dialog flow is an assembly line for processing the message.
    This assembly line is composed of PROC node that can really process the message and two type of control nodes: COND and LOOP, one for branching and the other for looping.
    Right now, the language is super primitive, the advantage is simply make the design super modularized and easy to understand, the disadvantage is that it is not very flexible and powerful.
    Main issue it that the definition of the modules are not so elegant, and the grammers are simply instructions.
    
    
    Primitive calls:

    FLOW name arg0|arg1|... output0|output1|... # must be the first line and only once, use None if args or outs are empty
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
        _,name,args,outs = parts
        name = maps[name]
        args = [] if args=='None' else args.split('|')
        outs = [] if outs=='None' else outs.split('|')
        flow = AgentDialogFlow(name,args,outs,init_state)
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
                    # try:
                    self._parse_node(line)
                    # except Exception as e:
                    #     raise ValueError(f'ALANG line {i+1}: "{line}"\n{e}')
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
        if U.pexists(ALANG): # path of a ALANG script file
            ALANG = U.read_file(ALANG)
        if isinstance(init_state,str): # path of a json file
            init_state = U.load_json(init_state)
        self._flow=None
        self._nodes={} # node var name to id
        self._aliases={} # node var name to alias
        if isinstance(modules,FunctionType):
            self._modules = self._fn_to_modules(modules)
        elif isinstance(modules,dict):
            for i,fn in modules.items():
                assert isinstance(fn,FunctionType) or isinstance(fn,ft.partial) or isinstance(fn,MethodType), f'Modules must be a dict of functions, found {type(fn)} for {i}'
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




def register_module(type='PROC',alias=None,hints=None,node=None,links=None):
    def decorator(func):
        assert type in ['PROC','COND','LOOP','EXIT'], f'Type must be PROC, COND, LOOP, or EXIT found {type}'
        # Extract the function's signature
        sig = inspect.signature(func)
        # Define the required arguments
        required_args = {'query', 'state'}

        # Check if the required arguments are in the function's signature
        func_args = set(sig.parameters.keys())
        missing_args = required_args - func_args

        if missing_args:
            raise ValueError(f"Missing required arguments: {missing_args} in function '{func.__name__}'")

        # Attach the hint to the function
        func._is_registered = True
        func._hints = hints  # Store the hint in the function's attribute
        func._alias = alias if alias else func.__name__  
        func._type = type
        func._node = node if node else func.__name__
        func._links = links 
        if links:
            assert isinstance(links,str) or isinstance(links,list), f'Links must be a string or a list of strings if provided, found {type(links)}'
        return func

    return decorator

class FlowCreator:
    def __init__(self,system,name,args=[],outs=[]):
        self._modules = {}
        self.system = system # An AgentSystem object
        self.name = name
        self.args:List[str] = args
        self.outs:List[str] = outs
        self.nodes_def=[]
        self.links_def=self._links()

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and getattr(attr, '_is_registered', False):
                self._modules[attr_name] = attr #ft.partial(attr, self)
                # self._modules[attr_name].__name__ = attr_name
                alias = getattr(attr, '_alias')
                hints = getattr(attr, '_hints')
                type = getattr(attr, '_type')
                node = getattr(attr, '_node')
                links = getattr(attr, '_links')
                self.nodes_def.append(f'{type} {node} `{alias}` {attr_name} `{hints}`')
                if links:
                    if isinstance(links,str):
                        self.links_def.append(f'{node} -> {links}')
                    elif isinstance(links,list):
                        self.links_def.append(f'{node} -> {"|".join(links)}')
                    else:
                        raise ValueError(f'Links must be a string or a list of strings, found {type(links)}')
        
        self.script = self._alang()
        self.flow = self._compile()
        
    # either a single string or a list of strings
    def _links(self)->List[str]:
        raise NotImplementedError
    
    def _alang(self)->str:
        args = '|'.join(self.args) if self.args else 'None'
        outs = '|'.join(self.outs) if self.outs else 'None'
        ascript = f'FLOW `{self.name}` {args} {outs}\n\n '
        ascript += '\n'.join(self.nodes_def)+'\n\n'
        if isinstance(self.links_def,str):
            ascript += self.links_def
        elif isinstance(self.links_def,list):
            ascript += '\n'.join(self.links_def)
        else:
            raise ValueError(f'Links definition must be a string or a list of strings, found {type(self.links_def)}')
        return ascript
    
    def _compile(self,init_state={}):
        _flow,_script= ALangCompiler().compile(self._alang(),self._modules,init_state,True)
        self.script = _script
        return _flow
    