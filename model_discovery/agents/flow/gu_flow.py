import os
import numpy as np
from typing import Any,Dict, List
import inspect

from exec_utils.models.model import ModelOutput
from .alang import FlowCreator,register_module,ROLE,SYSTEM_CALLER,AgentContext

# from model_discovery.system import ModelDiscoverySystem
import model_discovery.agents.prompts.prompts as P
from model_discovery.model.gab_composer import ROOT_UNIT_TEMPLATE,GABUnit


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

gam_prompt_path = os.path.join(current_dir,'..','prompts','gam_prompt.py')
gu_template_path = os.path.join(current_dir,'..','prompts','gu_template.py')
GAM_TEMPLATE=open(gam_prompt_path).read()
GU_TEMPLATE=open(gu_template_path).read()

GAB_UNIT=inspect.getsource(GABUnit)




class GUFlow(FlowCreator):
    def __init__(self,system,status_handler,stream):
        super().__init__(system,'GAB Unit Design Flow')
        self.system=system
        self.status_handler=status_handler
        self.stream=stream
        self.args=['main_tid']

        # prepare roles
        self.system.designer.model_state.static_message=self.system.designer.model_state.fn(
             instruction=P.GU_DESIGNER_SYSTEM(GAB_BASE=P.GAB_BASE,GAM_PY=GAM_TEMPLATE),
             examples=[]
        )
        self.DESIGNER = ROLE('designer',self.system.designer)


    def _links(self):
        links_def=[
            'ENTRY->design_initializer',
        ]
        return links_def

    @register_module(
        "PROC",
        hints="output the initial threads",
        links='sample_initial_design',
    )
    def design_initializer(self,query,state,main_tid,context=AgentContext()):
        design_thread_tid=self.system.dialog.fork(main_tid,SYSTEM_CALLER,self.DESIGNER,context=context,
                                            alias='designing',note=f'Starting design...')
        self.dialog=self.system.dialog
        RET={
            'design_thread_tid':design_thread_tid
        }
        return query,state,RET


    @register_module(
        "PROC",
        hints="output the initial threads",
        links='end_of_design',
    )
    def sample_initial_design(self,query,state,design_thread_tid):
        design_scratch_prompt=P.DESIGN_FROM_SCRATCH(GAB_UNIT=GAB_UNIT,GU_TEMPLATE=GU_TEMPLATE,ROOT_UNIT_TEMPLATE=ROOT_UNIT_TEMPLATE)
        self.system.designer = P.DESIGN_FROM_SCRATCH.apply(self.system.designer)
        thread_tid=design_thread_tid

        with self.status_handler('Start designing from scratch...'):
            _,out=self.dialog.call(thread_tid,design_scratch_prompt)
            self.stream.write(out)

        return query,state,{}
    

    @register_module(
        "EXIT",
        hints="output the initial threads",
    )
    def end_of_design(self,query,state):
        
        return query,state,{}


def gu_design(cls,query,stream,status_handler):
    main_tid = cls.dialog.fork(0,note='Starting a new session...',alias='main')
    gu_flow = GUFlow(cls, status_handler, stream)
    GU_CALLEE = ROLE('GAB Unit Designer',gu_flow.flow)
    gu_tid = cls.dialog.fork(main_tid,SYSTEM_CALLER,GU_CALLEE,note=f'launch design flow',alias=f'gu_design')
    res,ret=cls.dialog.call(gu_tid,query,main_tid=main_tid)

    