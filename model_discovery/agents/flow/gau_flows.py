import os
import numpy as np
from typing import Any,Dict, List
import inspect
import copy

from exec_utils.models.model import ModelOutput
from .alang import FlowCreator,register_module,ROLE,SYSTEM_CALLER,USER_CALLER,AgentContext

# from model_discovery.system import ModelDiscoverySystem
import model_discovery.agents.prompts.prompts as P
from model_discovery.model.composer import ROOT_UNIT_TEMPLATE,GAUBase


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

gam_prompt_path = os.path.join(current_dir,'..','prompts','gam_prompt.py')
gau_template_path = os.path.join(current_dir,'..','prompts','gau_template.py')
GAM_TEMPLATE=open(gam_prompt_path).read()
GAU_TEMPLATE=open(gau_template_path).read()

GAU_BASE=inspect.getsource(GAUBase)


def load_system_prompt(agent,prompt):
    agent.model_state.static_message=agent.model_state.fn(
            instruction=prompt,examples=[]
    )
    return agent

def reload_role(name,agent,prompt): # reload the role of an agent, it will change the role
    agent=load_system_prompt(agent,prompt)
    return ROLE(name,agent)

def apply_prompt(role,prompt,**kwargs):
    _prompt=prompt(**kwargs)
    role.obj = prompt.apply(role.obj)
    return role,_prompt


def print_details(stream,agent,context,prompt):
    if not hasattr(stream,'_isprintsystem'):
        stream.write('Details of the input:')
        stream.write(
            f"""<details><summary>Agent system prompt</summary>{agent.model_state.static_message}</details>""",
            unsafe_allow_html=True
        )
        stream.write(
            f"""<details><summary>Agent input context</summary>{context.get()}</details>""",
            unsafe_allow_html=True
        )
        stream.write(
            f"""<details><summary>Agent input prompt</summary>{prompt}</details>""",
            unsafe_allow_html=True
        )
        stream.write('---')

def print_raw_output(stream,out):
    if not hasattr(stream,'_isprintsystem'):
        stream.write(
            f"""<details><summary>Raw output</summary>{out}</details>""",
            unsafe_allow_html=True
        )


class GUFlowScratch(FlowCreator): 
    """
    The flow for designing a GAB Flow nested of GAB Units from scratch.
    the input query should be the seeds from the root tree for the design
    """
    def __init__(self,system,status_handler,stream):
        super().__init__(system,'GAU Design Flow')
        self.system=system
        self.status_handler=status_handler
        self.stream=stream
        self.args=['main_tid']#,'max_attemps']
        self.outs=[]
        self.max_attemps={
            'design_proposal':10,
        }

        # prepare roles
        self.gpt4o0806_agent=self.system.designer # as we replaced the system prompt, essential its just a base agent
        self.gpt4omini_agent=self.system.debugger 

    def _links(self):
        links_def=[
            'ENTRY->design_initializer',
        ]
        return links_def
    
    def print_details(self,agent,context,prompt):
        print_details(self.stream,agent,context,prompt)

    def print_raw_output(self,out):
        print_raw_output(self.stream,out)

    @register_module(
        "PROC",
        hints="output the initial threads",
        links='generate_proposal',
    )
    def design_initializer(self,query,state):
        return query,state,{}

    @register_module(
        "PROC",
        hints="output the proposal after review",
        links='implement_proposal',
    )
    def generate_proposal(self,query,state,main_tid):
        self.dialog=self.system.dialog

        with self.status_handler('Starting the design process, seeds sampled.'):
            self.stream.write(query,unsafe_allow_html=True)

        self.stream.write(f'#### Start design process by generating a design proposal')


        traces=[]
        context_design_proposer=AgentContext()
        context_proposal_reviewer=AgentContext()
        for i in range(self.max_attemps['design_proposal']):
            DESIGN_PROPOSER=reload_role('designer',self.gpt4o0806_agent,P.GU_DESIGN_PROPOSER_SYSTEM(
                GAB_BASE=P.GAB_BASE,GAM_PY=GAM_TEMPLATE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE))
            design_proposer_tid=self.dialog.fork(main_tid,USER_CALLER,DESIGN_PROPOSER,context=context_design_proposer,
                                                alias='design_proposal',note=f'Starting design proposal...')
            if i==0:
                status_info=f'Initial design proposal...'
                proposal_prompt=P.GU_DESIGN_PROPOSAL(SEEDS=query)
                P.GU_DESIGN_PROPOSAL.apply(DESIGN_PROPOSER.obj)
            else:
                status_info=f'Refining design proposal (attempt {i})...'
                proposal_prompt=P.GU_PROPOSAL_REFINEMENT(REVIEW=review,RATING=rating,SUGGESTIONS=suggestions)
                P.GU_PROPOSAL_REFINEMENT.apply(DESIGN_PROPOSER.obj)
            
            with self.status_handler(status_info):
                self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_prompt)
                _,out=self.dialog.call(design_proposer_tid,proposal_prompt)
                title=out['title']
                if 'proposal' in out:
                    proposal=out['proposal']
                    reflection=out['reflection']
                    changes=out['changes']
                    self.stream.write(f'# Proposal\n{proposal}')
                    self.stream.write(f'# Reflection\n{reflection}')
                    self.stream.write(f'# Changes\n{changes}')
                else:
                    proposal=out['text']
                    self.stream.write(proposal)
                context_design_proposer=self.dialog.context(design_proposer_tid)
                self.print_raw_output(out)


            PROPOSAL_REVIEWER=reload_role('proposal_reviewer',self.gpt4o0806_agent, 
                                                P.GU_PROPOSAL_REVIEWER_SYSTEM())
            proposal_reviewer_tid=self.dialog.fork(main_tid,USER_CALLER,PROPOSAL_REVIEWER,context=context_proposal_reviewer,
                                                alias='proposal_review',note=f'Reviewing proposal...')
            with self.status_handler(f'Reviewing design proposal (version {i})...'):
                proposal_review_prompt=P.GU_PROPOSAL_REVIEW(PROPOSAL=proposal)
                P.GU_PROPOSAL_REVIEW.apply(PROPOSAL_REVIEWER.obj)
                self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,proposal_review_prompt)
                _,out=self.dialog.call(proposal_reviewer_tid,proposal_review_prompt)
                review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                context_proposal_reviewer=self.dialog.context(proposal_reviewer_tid)
                self.stream.write(review)
                self.stream.write(suggestions)
                passornot='Pass' if rating>3 else 'Fail'
                self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                self.print_raw_output(out)


            trace={
                'title':title,
                'proposal':proposal,
                'review':review,
                'rating':rating,
                'suggestions':suggestions,
            }
            traces.append(trace)

            if rating>3:
                self.stream.write(f'#### Proposal passed with rating {rating} out of 5, starting implementation')
                break
        
        if rating<=3:
            self.stream.write(f'#### Proposal failed with rating {rating} out of 5, stopping design process')
            raise Exception('Design proposal failed, stopping design process')
        RET={
            'proposal':trace,
            'traces':traces,
        }
        return query,state,RET
    

    def check_GAU(self,gau_code):

        # check format
        # 1. if multiple GAUBase classes are defined
        # 2. if the class is named GAU
        # 3. if the imports are correct

        # check functionality
        # 1. create placeholders
        # 2. run checker to check the whole model


        return True

    
    @register_module(
        "PROC",
        hints="output the initial threads",
        links='end_of_design',
    )
    def implement_proposal(self,query,state,main_tid,proposal):
        self.dialog=self.system.dialog
        
        traces=[]
        context_design_implementer=AgentContext()
        DESIGN_IMPLEMENTER=reload_role('design_implementer',self.gpt4o0806_agent,P.DESIGN_IMPLEMENTATER_SYSTEM(
            GAB_BASE=P.GAB_BASE,GAM_PY=GAM_TEMPLATE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE))
        design_implementer_tid=self.dialog.fork(main_tid,USER_CALLER,DESIGN_IMPLEMENTER,context=context_design_implementer,
                                            alias='design_proposal',note=f'Starting design proposal...')
        
        status_info=f'Starting design implementation of root unit...'
        with self.status_handler(status_info):
            gu_design_root_prompt=P.GU_IMPLEMENTATION_ROOT(
                PROPOSAL=proposal['proposal'],REVIEW=proposal['review'],RATING=proposal['rating'])
            P.GU_IMPLEMENTATION_ROOT.apply(DESIGN_IMPLEMENTER.obj)
            self.print_details(DESIGN_IMPLEMENTER.obj,context_design_implementer,gu_design_root_prompt)
            _,out=self.dialog.call(design_implementer_tid,gu_design_root_prompt)

            self.stream.write(f'## Implementation of {out['unit_name']}')
            self.stream.write(out['analysis'])
            self.stream.write(f'### Code\n```python\n{out['implementation']}\n```')
                              
            self.print_raw_output(out)

        


    
    @register_module(
        "EXIT",
        hints="output the initial threads",
    )
    def end_of_design(self,query,state):
        
        return query,state,{}


def gu_design_scratch(cls,query,stream,status_handler):
    main_tid = cls.dialog.fork(0,note='Starting a new session...',alias='main')
    gu_flow = GUFlowScratch(cls, status_handler, stream)
    GU_CALLEE = ROLE('GAB Unit Designer',gu_flow.flow)
    gu_tid = cls.dialog.fork(main_tid,SYSTEM_CALLER,GU_CALLEE,note=f'launch design flow',alias=f'gu_design')
    res,ret=cls.dialog.call(gu_tid,query,main_tid=main_tid)

    # return title,code,explain,summary,autocfg,reviews,ratings,check_results





