import os
from enum import Enum
import inspect
import copy
from dataclasses import dataclass
from typing import Optional

from exec_utils.models.model import ModelOutput
from exec_utils import SimpleLMAgent
from .alang import FlowCreator,register_module,ROLE,SYSTEM_CALLER,USER_CALLER,AgentContext
from .gau_utils import check_and_reformat_gau_code

# from model_discovery.system import ModelDiscoverySystem
import model_discovery.agents.prompts.prompts as P
from model_discovery.model.composer import GAUBase, GAUTree, check_tree_name, GABComposer
from model_discovery.model.utils.modules import GABBase
import model_discovery.utils as U


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

gam_prompt_path = os.path.join(current_dir,'..','prompts','gam_prompt.py')
gau_template_path = os.path.join(current_dir,'..','prompts','gau_template.py')
GAM_TEMPLATE=open(gam_prompt_path).read()
GAU_TEMPLATE=open(gau_template_path).read()

GAU_BASE=inspect.getsource(GAUBase)
GAB_BASE=inspect.getsource(GABBase)


@dataclass
class AgentModelDef:
    agent: SimpleLMAgent
    model_name: Optional[str] = None

class DesignModes(Enum):
    MUTATION = 'Mutate existing design'
    SCRATCH = 'Design from scratch (unstable)'
    CROSSOVER = 'Crossover designs (unsupported)'


def reload_role(name,agentdef:AgentModelDef,prompt):# reload the role of an agent, it will change the role
    model_name=agentdef.model_name
    agentdef.agent.model_state.static_message=agentdef.agent.model_state.fn(
        instruction=prompt,examples=[]
    )
    if model_name is not None:
        agentdef.agent.model._config.model_name=model_name
    return ROLE(name,agentdef.agent)

def collapse_write(stream,summary,content):
    stream.write(f'<details><summary>{summary}</summary>\n{content}</details>',unsafe_allow_html=True)


def print_details(stream,agent,context,prompt):
    if not hasattr(stream,'_isprintsystem'):
        stream.write('Details of the input:')
        stream.write(
            f"""<details><summary>Agent system prompt</summary>{agent.model_state.static_message[0]['content']}</details>""",
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



###################################################################
# Design Flow from Existing Design
###################################################################


# region MUTATION


class EndReasons(Enum):
    PROPOSAL_FAILED='Proposal Failed'
    MAX_POST_REFINEMENT_REACHED='Max Post Refinement Reached'
    AGENT_TERMINATION='Agent Choose Termination'
    MAX_TOTAL_BUDGET_REACHED='Max Total Budget Reached'
    MAX_DEBUG_BUDGET_REACHED='Max Debug Budget Reached'
    MAX_FAILED_ROUNDS_REACHED='Max Failed Rounds Reached'
    UNFINISHED='Unfinished'

class RunningModes(Enum):
    PROPOSAL_ONLY='Proposal Only'
    IMPLEMENTATION_ONLY='Implementation Only'
    BOTH='Proposal + Implementation'

END_REASONS_LABELS = {
    EndReasons.PROPOSAL_FAILED:'Failed',
    EndReasons.MAX_POST_REFINEMENT_REACHED:'Success',
    EndReasons.AGENT_TERMINATION:'Success',
    EndReasons.MAX_TOTAL_BUDGET_REACHED:'Failed',
    EndReasons.MAX_DEBUG_BUDGET_REACHED:'Failed',
    EndReasons.MAX_FAILED_ROUNDS_REACHED:'Failed',
    EndReasons.UNFINISHED:'Unfinished',
}

class GUFlowMutation(FlowCreator): 
    """
    The flow for designing a GAB Flow nested of GAB Units from scratch.
    the input query should be the seeds from the root tree for the design
    """
    def __init__(self,system,status_handler,stream,design_id,design_cfg,user_input=''):
        self.costs={
            'DESIGN_PROPOSER':0,
            'PROPOSAL_REVIEWER':0,
            'IMPLEMENTATION_PLANNER':0, 
            'IMPLEMENTATION_OBSERVER':0, 
            'IMPLEMENTATION_CODER':0,
            'SEARCH_ASSISTANT':0,  # optional role
        }

        super().__init__(system,'GAU Design Flow from Existing Tree')
        self.system=system
        self.sss=system.sss # search_utils
        self.status_handler=status_handler
        self.stream=stream
        self.args=['main_tid']
        self.outs=['design_stack']

        # prepare roles

        AGENT_TYPES = {
            'claude3.5_sonnet':self.system.claude,
            'gpt4o_0806':self.system.designer,
            'gpt4o_mini':self.system.designer,
            'o1_preview':self.system.designer,
            'o1_mini':self.system.designer,
            'None':None, # None for search assistiant 
        }
        AGENT_TYPES_MODEL_NAMES = {
            'claude3.5_sonnet':'claude-3.5-sonnet-20240620', # maybe incompatible with exec_utils
            'gpt4o_0806':'gpt-4o-2024-08-06',
            'gpt4o_mini':'gpt-4o-mini',
            'o1_preview':'o1-preview',
            'o1_mini':'o1-mini',
            'None':None,
        }

        self.failed_rounds=0
        self.max_attemps=design_cfg['max_attemps']
        self.agent_types=design_cfg['agent_types']     
        self.agents={}
        for name,value in self.agent_types.items():
            self.agents[name]=AgentModelDef(AGENT_TYPES[value],AGENT_TYPES_MODEL_NAMES[value])

        self.termination=design_cfg['termination']
        self.threshold=design_cfg['threshold']
        self.search_settings=design_cfg['search_settings']
        self.mode=design_cfg['running_mode']
        self.num_samples=design_cfg['num_samples']
        self.design_cfg=design_cfg
        self.user_input=user_input

        # assert any(self.termination.values())>0, 'At least one of the termination conditions should be set'

        self.design_id = design_id
        self.ptree=system.ptree
        seeds,refs,instruct=self.ptree.get_session_input(design_id)
        self.seed_tree = self.ptree.get_gau_tree(seeds[0].acronym)
        self.seed_input=P.build_GUM_QUERY(seeds[0],refs,instruct,user_input)


    def _links(self):
        links_def=[
            'ENTRY->design_initializer',
        ]
        return links_def

    def print_details(self,agent,context,prompt):
        print_details(self.stream,agent,context,prompt)

    @property
    def total_cost(self):
        return sum(self.costs.values())
    
    @property
    def implementation_cost(self):
        return self.costs['IMPLEMENTATION_PLANNER']+self.costs['IMPLEMENTATION_OBSERVER']+self.costs['IMPLEMENTATION_CODER']

    @property
    def proposal_cost(self):
        return self.costs['DESIGN_PROPOSER']+self.costs['PROPOSAL_REVIEWER']

    def print_raw_output(self,out,agent):
        print_raw_output(self.stream,out)
        self.costs[agent]+=out["_details"]["running_cost"]
        usage=out["_details"]["cost"]
        if isinstance(usage,float):
            self.stream.write(f'##### **Usage**\n {usage}')
        else:
            self.stream.write(f'##### **Usage**')
            for k,v in usage.items():
                self.stream.write(f' - *{k}*: {v}')
        self.stream.write(f'###### **Running Cost**')
        for agent,cost in self.costs.items():
            self.stream.write(f' - *{agent} Cost*: {cost}')
        self.stream.write(f'###### **Session Total Cost**: {self.total_cost}')


    @register_module(
        "PROC",
        hints="output the initial threads",
        links='generate_proposal',
    )
    def design_initializer(self,query,state,proposal=None):
        if proposal:
            if self.mode!=RunningModes.IMPLEMENTATION_ONLY:
                self.stream.write(f'Proposal is provided, running mode will be forced switched to implementation only from {self.mode}')
                self.mode=RunningModes.IMPLEMENTATION_ONLY
        else:
            if self.mode==RunningModes.IMPLEMENTATION_ONLY:
                raise Exception('Proposal is required for implementation only mode')
        return query,state,{}
    

    def call_search_assistant(self,main_tid,ideation,instructions):
        self.stream.write(f'Warning: Search Assistant prompt has not been updated, performance may degrade.')
        S2_SEARCH_SYSTEM=P.S2_SEARCH_ASSISTANT_SYSTEM()
        S2_SEARCH_ASSISTANT=reload_role('search_assistant',self.agents['SEARCH_ASSISTANT'],S2_SEARCH_SYSTEM)
        context_search_assistant=AgentContext()
        search_assistant_tid=self.dialog.fork(main_tid,USER_CALLER,S2_SEARCH_ASSISTANT,context=context_search_assistant,
                                                alias='search_proposal',note=f'Starting search S2...')
        
        for i in range(self.max_attemps['max_search_rounds']):
            self.stream.write(f'## Searching from S2, round {i+1}...')
            # S2 Search Query
            s2_search_query_prompt=P.S2_SEARCH_PROPOSAL_QUERY(IDEATION=ideation,INSTRUCTIONS=instructions)
            P.S2_SEARCH_PROPOSAL_QUERY.apply(S2_SEARCH_ASSISTANT.obj)
            self.print_details(S2_SEARCH_ASSISTANT.obj,context_search_assistant,s2_search_query_prompt)
            _,out=self.dialog.call(search_assistant_tid,s2_search_query_prompt)
            analysis,query=out['analysis'],out['query']
            self.stream.write(f'### Analysis\n{analysis}')
            self.stream.write(f'### Query\n{query}')
            rets=self.sss(query,analysis) # TODO: specifically prompts details for search assistant
            self.stream.markdown(rets,unsafe_allow_html=True)
            self.print_raw_output(out,'SEARCH_ASSISTANT')

            # S2 Search Response
            s2_search_response_prompt=P.S2_SEARCH_PROPOSAL_RESPONSE(SEARCH_RESULTS=rets)
            P.S2_SEARCH_PROPOSAL_RESPONSE.apply(S2_SEARCH_ASSISTANT.obj)
            self.print_details(S2_SEARCH_ASSISTANT.obj,context_search_assistant,s2_search_response_prompt)
            _,out=self.dialog.call(search_assistant_tid,s2_search_response_prompt)
            report,references,continue_search=out['report'],out['references'],out['continue_search']
            self.stream.write(f'### Report\n{report}')
            self.stream.write(f'### References\n{references}')
            self.print_raw_output(out,'SEARCH_ASSISTANT')
            self.stream.write('---')
            if not continue_search:
                self.stream.write(f'### Search Assistant chose to stop search')
                break
        self.stream.write(f'### Search Finished')
        return report,references


    @register_module(
        "PROC",
        hints="output the proposal after review",
        links='implement_proposal_recursive',
    )
    def generate_proposal(self,query,state,main_tid):
        '''
        Overally evaluate the current ptree, and generate a proposal for the next step, and pick one unit to work on
        '''
        if self.mode==RunningModes.IMPLEMENTATION_ONLY:
            self.stream.write('Implementation only mode, skipping proposal generation...')
            return query,state,{}
        passed_proposals=self.ptree.session_proposals(self.design_id,passed_only=True)
        for _ in range(self.num_samples['proposal']-len(passed_proposals)):
            cost_raw=copy.deepcopy(self.costs)
            query,state,RET=self._generate_proposal(self.seed_input,state,main_tid)
            proposal,proposal_traces=RET['proposal'],RET['proposal_traces']
            costs={k:v-cost_raw[k] for k,v in self.costs.items()}
            self.ptree.propose(self.design_id,proposal,proposal_traces,costs,self.design_cfg,self.user_input)
        return query,state,{}

    def _generate_proposal(self,query,state,main_tid):
        '''
        Sample one proposal based on the current tree and add it to the tree
        '''
        self.dialog=self.system.dialog

        with self.status_handler('Starting the design process, seeds sampled.'):
            self.stream.write(query,unsafe_allow_html=True)

        self.stream.write(f'#### Start design process by generating a design proposal')

        USE_2STAGE=self.search_settings['proposal_search']
        USE_ISEARCH=self.agent_types['SEARCH_ASSISTANT']=='None' and USE_2STAGE 
        USE_2STAGE=USE_2STAGE and not USE_ISEARCH

        traces=[]
        context_design_proposer=AgentContext()
        context_proposal_reviewer=AgentContext()
        SELECTIONS=list(self.seed_tree.units.keys())
        for attempt in range(self.max_attemps['design_proposal']):
            DESIGN_PROPOSER_SYSTEM=P.GUM_DESIGN_PROPOSER_SYSTEM 
            if USE_ISEARCH:
                DESIGN_PROPOSER_SYSTEM=P.GUM_DESIGN_PROPOSER_SYSTEM_ISEARCH
            elif USE_2STAGE:
                DESIGN_PROPOSER_SYSTEM=P.GUM_DESIGN_PROPOSER_SYSTEM_2STAGE
            DESIGN_PROPOSER=reload_role('design_proposer',self.agents['DESIGN_PROPOSER'],DESIGN_PROPOSER_SYSTEM(GAU_BASE=GAU_BASE))
            design_proposer_tid=self.dialog.fork(main_tid,USER_CALLER,DESIGN_PROPOSER,context=context_design_proposer,
                                                alias='design_proposal',note=f'Starting design proposal...')
            if attempt==0:
                status_info=f'Initial design proposal...'
                GUM_DESIGN_PROPOSAL=P.gen_GUM_DESIGN_PROPOSAL(SELECTIONS=SELECTIONS,two_stage=USE_2STAGE,use_isearch=USE_ISEARCH)
                if USE_ISEARCH:
                    GUM_DESIGN_PROPOSAL,GUM_DESIGN_PROPOSAL_FINISH=GUM_DESIGN_PROPOSAL
                elif USE_2STAGE:
                    GUM_DESIGN_PROPOSAL,GUM_DESIGN_PROPOSAL_STAGE2=GUM_DESIGN_PROPOSAL
                proposal_prompt=GUM_DESIGN_PROPOSAL(SEED=query)
                GUM_DESIGN_PROPOSAL.apply(DESIGN_PROPOSER.obj)
            else:
                status_info=f'Refining design proposal (attempt {attempt})...'
                GUM_PROPOSAL_REFINEMENT=P.gen_GUM_PROPOSAL_REFINEMENT(SELECTIONS=SELECTIONS,two_stage=USE_2STAGE,use_isearch=USE_ISEARCH)
                if USE_ISEARCH:
                    GUM_PROPOSAL_REFINEMENT,GUM_DESIGN_PROPOSAL_FINISH=GUM_PROPOSAL_REFINEMENT
                elif USE_2STAGE:
                    GUM_PROPOSAL_REFINEMENT,GUM_DESIGN_PROPOSAL_STAGE2=GUM_PROPOSAL_REFINEMENT
                proposal_prompt=GUM_PROPOSAL_REFINEMENT(REVIEW=review,RATING=rating,SUGGESTIONS=suggestions,
                                                         PASS_OR_NOT='Pass' if rating>=4 else 'Fail')
                GUM_PROPOSAL_REFINEMENT.apply(DESIGN_PROPOSER.obj)
            
            ideation,instructions,search_report,search_references=None,None,None,None
            search_stack=[]
            if USE_ISEARCH:
                with self.status_handler(status_info):
                    self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_prompt)
                    _,out=self.dialog.call(design_proposer_tid,proposal_prompt)
                    analysis,keywords,detail,ready,reflection=out['analysis'],out['keywords'],out['detail'],out['ready'],out.get('reflection',None)
                    if reflection:
                        self.stream.write(f'# Reflection\n{reflection}')
                    self.stream.write(f'# Analysis\n{analysis}')
                    self.stream.write(f'# Query\n{keywords}')
                    self.stream.write(f'# Detail\n{detail}')
                    self.stream.write(f'# Ready\n{ready}')

                for i in range(self.max_attemps['max_search_rounds']):
                    with self.status_handler(f'Searching... round {i+1}...'):
                        search_ret=self.sss(keywords,detail)
                        search_stack.append({
                                'analysis':analysis,
                                'query':keywords,
                                'detail':detail,
                                'ready':ready,
                                'search_ret':search_ret,
                            })
                        search_cont_prompt=P.GUM_DESIGN_PROPOSAL_ISEARCH_CONT(SEARCH_RESULTS=search_ret)
                        P.GUM_DESIGN_PROPOSAL_ISEARCH_CONT.apply(DESIGN_PROPOSER.obj)
                        self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,search_cont_prompt)
                        _,out=self.dialog.call(design_proposer_tid,search_cont_prompt)
                        analysis,keywords,detail,ready=out['analysis'],out['keywords'],out['detail'],out['ready']
                        self.stream.write(f'### Analysis\n{analysis}')
                        self.stream.write(f'### Query\n{keywords}')
                        self.stream.write(f'### Detail\n{detail}')
                        self.stream.write(f'### Ready\n{ready}')
                        self.print_raw_output(out,'DESIGN_PROPOSER')
                        self.stream.write('---')
                    if ready: break # the first ready will be ignored, at least one search is required
                
                with self.status_handler('Finishing design proposal...'):
                    search_finish_prompt=GUM_DESIGN_PROPOSAL_FINISH()
                    GUM_DESIGN_PROPOSAL_FINISH.apply(DESIGN_PROPOSER.obj)
                    self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,search_finish_prompt)
                    _,out=self.dialog.call(design_proposer_tid,search_finish_prompt)
                    selection,proposal,modelname,changes=out['selection'],out['proposal'],out['modelname'],out.get('changes',None)
                    self.stream.write(f'### Design Name: {modelname}')
                    self.stream.write(f'### Selection: {selection}')
                    self.stream.write(f'# Proposal\n{proposal}')
                    if changes:
                        self.stream.write(f'# Changes\n{changes}')
                    context_design_proposer=self.dialog.context(design_proposer_tid)
                    self.print_raw_output(out,'DESIGN_PROPOSER')

            elif USE_2STAGE: # use search or not
                with self.status_handler(status_info):
                    self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_prompt)
                    _,out=self.dialog.call(design_proposer_tid,proposal_prompt)
                    reflection,ideation,instructions=out.get('reflection',None),out['ideation'],out['instructions']
                    if reflection:
                        self.stream.write(f'# Reflection\n{reflection}')
                    self.stream.write(f'# Ideation\n{ideation}')
                    self.stream.write(f'# Instructions to Search Assistant\n{instructions}')
                    self.print_raw_output(out,'DESIGN_PROPOSER')

                with self.status_handler('Searching...'):
                    search_report,search_references=self.call_search_assistant(main_tid,ideation,instructions)

                with self.status_handler('Generating proposal...'):
                    proposal_prompt_stage2=GUM_DESIGN_PROPOSAL_STAGE2(GATHERED_INFO=search_report)
                    GUM_DESIGN_PROPOSAL_STAGE2.apply(DESIGN_PROPOSER.obj)
                    self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_prompt_stage2)
                    _,out=self.dialog.call(design_proposer_tid,proposal_prompt_stage2)
                    selection,proposal,modelname,changes=out['selection'],out['proposal'],out['modelname'],out.get('changes',None)
                    self.stream.write(f'### Design Name: {modelname}')
                    self.stream.write(f'### Selection: {selection}')
                    self.stream.write(f'# Proposal\n{proposal}')
                    if changes:
                        self.stream.write(f'# Changes\n{changes}')
                    context_design_proposer=self.dialog.context(design_proposer_tid)
                    self.print_raw_output(out,'DESIGN_PROPOSER')
            else:
                with self.status_handler(status_info):
                    self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_prompt)
                    _,out=self.dialog.call(design_proposer_tid,proposal_prompt)
                    selection,proposal,modelname=out['selection'],out['proposal'],out['modelname']
                    self.stream.write(f'### Design Name: {modelname}')
                    self.stream.write(f'### Selection: {selection}')
                    reflection,changes=out.get('reflection',None),out.get('changes',None)
                    self.stream.write(f'# Proposal\n{proposal}')
                    if reflection:
                        self.stream.write(f'# Reflection\n{reflection}')
                    if changes:
                        self.stream.write(f'# Changes\n{changes}')
                    context_design_proposer=self.dialog.context(design_proposer_tid)
                    self.print_raw_output(out,'DESIGN_PROPOSER')


            ### Review
            USE_ISEARCH_REVIEW=self.search_settings['proposal_review_search']

            SYSTEM_PROMPT=P.GUM_PROPOSAL_REVIEWER_SYSTEM() if not USE_ISEARCH_REVIEW else P.GUM_PROPOSAL_REVIEWER_SEARCH_SYSTEM()
            PROPOSAL_REVIEWER=reload_role('proposal_reviewer',self.agents['PROPOSAL_REVIEWER'],SYSTEM_PROMPT)
            proposal_reviewer_tid=self.dialog.fork(main_tid,USER_CALLER,PROPOSAL_REVIEWER,context=context_proposal_reviewer,
                                                alias='proposal_review',note=f'Reviewing proposal...')
            if attempt==0:
                status_info=f'Reviewing initial proposal...'
                REVIEW_PROMPT=P.GUM_PROPOSAL_REVIEW_ISEARCH_BEGIN if USE_ISEARCH_REVIEW else P.GUM_PROPOSAL_REVIEW
                proposal_review_prompt=REVIEW_PROMPT(
                    SEED=query,SELECTION=selection,PROPOSAL=proposal)
                REVIEW_PROMPT.apply(PROPOSAL_REVIEWER.obj)
            else:
                status_info=f'Reviewing refined proposal (version {attempt})...'
                REREVIEW_PROMPT=P.GUM_PROPOSAL_REREVIEW_ISEARCH if USE_ISEARCH_REVIEW else P.GUM_PROPOSAL_REREVIEW
                proposal_review_prompt=REREVIEW_PROMPT(
                    SELECTION=selection,PROPOSAL=proposal,CHANGES=changes)
                REVIEW_PROMPT.apply(PROPOSAL_REVIEWER.obj)

            review_search_stack=[]
            if USE_ISEARCH_REVIEW:
                with self.status_handler(status_info):
                    self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,proposal_review_prompt)
                    _,out=self.dialog.call(proposal_reviewer_tid,proposal_review_prompt)
                    analysis,keywords,detail,ready=out['analysis'],out['keywords'],out['detail'],out['ready']
                    self.stream.write(f'### Analysis\n{analysis}')
                    self.stream.write(f'### Query\n{keywords}')
                    self.stream.write(f'### Detail\n{detail}')
                    self.stream.write(f'### Ready\n{ready}')
                    self.print_raw_output(out,'PROPOSAL_REVIEWER')
                
                for i in range(self.max_attemps['max_search_rounds']):
                    with self.status_handler(f'Searching... round {i+1}...'):
                        search_ret=self.sss(keywords,detail)
                        review_search_stack.append({
                                'analysis':analysis,
                                'query':keywords,
                                'detail':detail,
                                'ready':ready,
                                'search_ret':search_ret,
                            })
                        search_cont_prompt=P.GUM_PROPOSAL_REVIEW_ISEARCH_CONT(SEARCH_RESULTS=search_ret)
                        P.GUM_PROPOSAL_REVIEW_ISEARCH_CONT.apply(PROPOSAL_REVIEWER.obj)
                        self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,search_cont_prompt)
                        _,out=self.dialog.call(proposal_reviewer_tid,search_cont_prompt)
                        analysis,keywords,detail,ready=out['analysis'],out['keywords'],out['detail'],out['ready']
                        self.stream.write(f'### Analysis\n{analysis}')
                        self.stream.write(f'### Query\n{keywords}')
                        self.stream.write(f'### Detail\n{detail}')
                        self.stream.write(f'### Ready\n{ready}')
                        self.print_raw_output(out,'PROPOSAL_REVIEWER')
                        self.stream.write('---')
                    if ready: break # the first ready will be ignored, at least one search is required
                
                with self.status_handler('Finishing proposal review...'):
                    search_finish_prompt=P.GUM_PROPOSAL_REVIEW_ISEARCH_FINAL()
                    P.GUM_PROPOSAL_REVIEW_ISEARCH_FINAL.apply(PROPOSAL_REVIEWER.obj)
                    self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,search_finish_prompt)
                    _,out=self.dialog.call(proposal_reviewer_tid,search_finish_prompt)
                    review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                    context_proposal_reviewer=self.dialog.context(proposal_reviewer_tid)
                    passornot='Pass' if rating>=self.threshold['proposal_rating'] else 'Fail'
                    self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                    self.stream.write(review)
                    self.stream.write(suggestions)
                    self.print_raw_output(out,'PROPOSAL_REVIEWER')
            else:
                with self.status_handler(status_info):
                    self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,proposal_review_prompt)
                    _,out=self.dialog.call(proposal_reviewer_tid,proposal_review_prompt)
                    review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                    context_proposal_reviewer=self.dialog.context(proposal_reviewer_tid)
                    passornot='Pass' if rating>=self.threshold['proposal_rating'] else 'Fail'
                    self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                    self.stream.write(review)
                    self.stream.write(suggestions)
                    self.print_raw_output(out,'PROPOSAL_REVIEWER')

            trace={
                # proposal content
                'selection':selection,
                'modelname':modelname,
                'proposal':proposal,
                # review process
                'review':review,
                'rating':rating,
                'passed':passornot=='Pass',
                'suggestions':suggestions,
                'reflection':reflection,
                'changes':changes,
                # search process
                'ideation':ideation,
                'instructions':instructions,
                'search_report':search_report,
                'search_references':search_references,
                'search_stack':search_stack,
                'review_search_stack':review_search_stack,
            }
            traces.append(trace)

            if rating>=self.threshold['proposal_rating']:
                self.stream.write(f'#### Proposal passed with rating {rating} out of 5, starting implementation')
                break
        
        if rating<self.threshold['proposal_rating']:
            self.stream.write(f'#### Proposal failed with rating {rating} out of 5, stopping design process')
            # raise Exception('Design proposal failed, stopping design process')
        RET={
            'proposal':trace,
            'proposal_traces':traces,
            'proposal_passed':rating>=self.threshold['proposal_rating'],
        }
        return query,state,RET
        

    def rerank_proposals(self,proposals): # now simply rank by rating, TODO: improve it
        rerank={}
        rank=sorted(proposals,key=lambda x:x.rating,reverse=True)
        rerank['rank']=[proposal.modelname for proposal in rank]
        return rerank

    def reranked_proposal(self):
        # select the highest rated unimplemented proposal
        proposals=[]
        rerank=self.ptree.session_get(self.design_id,'reranked')
        if not rerank:
            proposals=self.ptree.session_proposals(self.design_id,passed_only=True)
            rerank=self.rerank_proposals(proposals)
            self.ptree.session_set(self.design_id,'reranked',rerank)
        for acronym in rerank['rank']:
            design=self.ptree.get_node(acronym)
            if not design.is_implemented():
                proposals.append(design.proposal)
        return proposals

    @register_module(
        "PROC",
        hints="output the design stack",
        links='end_of_design',
    )
    def implement_proposal_recursive(self,query,state,main_tid,proposal=None): # if provide a proposal, then design session should be taken care of manually
        if self.mode==RunningModes.PROPOSAL_ONLY:
            self.stream.write('Proposal only mode, skipping implementation...')
            return query,state,{}
        elif self.mode==RunningModes.IMPLEMENTATION_ONLY:
            self.stream.write('Implementation only mode, will select from unimplemented passed proposals or using provided proposal if any')
        if proposal is None:
            proposals=self.reranked_proposal()
        else:
            proposals=[proposal]
        for proposal in proposals:
            self.stream.write(f'Implementing proposal: {proposal.modelname} with rating {proposal.rating} out of 5')
            self.tree=copy.deepcopy(self.seed_tree)
            self.tree.name=proposal.modelname
            cost_raw=copy.deepcopy(self.costs)
            RETS=self._implement_proposal_recursive(main_tid,proposal)
            costs={k:v-cost_raw[k] for k,v in self.costs.items()}
            ROUNDS,new_unit_name,SUCCEED=RETS['ROUNDS'],RETS['new_unit_name'],RETS['SUCCEED']
            self.ptree.implement(self.design_id,self.tree,new_unit_name,ROUNDS,SUCCEED,costs,self.design_cfg,self.user_input)

        return query,state,{}
   
    def _implement_proposal_recursive(self,main_tid,proposal=None):
        '''
        1. Implement the selected unit first
        2. Implement any unimplemented newly declared units
        3. Do post refinement, if new units defined, go to 2, post refinement count will not be refreshed
        '''

        self.dialog=self.system.dialog

        RETS={}
        RETS['ROUNDS']=[]
        SUCCEED=False
        LOG=[]
        round=0
        PROTECTED_UNITS=list(set(self.tree.units.keys())-set([proposal.selection])) # the units besides the current one, they should not be *modified*, can be removed as descendants
        self.stream.write(f'##### Protected Units: {PROTECTED_UNITS}')
        USE_PAIRING=self.agent_types['IMPLEMENTATION_OBSERVER']!='None'
        # o1 beta does not support structured outputs, so let it output the code directly
        USE_O1_CODER=self.agents['IMPLEMENTATION_CODER'].model_name in ['o1-mini','o1-preview']

        post_refinement=0 # TODO: introduce self-evaluate to post-refinement
        while True:
            round+=1 # Each round works on one unit at a time, start counting from beginning so that we dont wrap it in wrong place
            traces=[]
            context_design=AgentContext() # context accummulated for all attempts in one unit
            context_implementation_reviewer=AgentContext()
            succeed=False
            _,IMPLEMENTED,UNIMPLEMENTED=self.tree.view()
            # GAB_CODE=self.tree.compose()
            VIEW_DETAILED=self.tree.to_prompt(unit_code=True)
            UNIMPLEMENTED=list(set(UNIMPLEMENTED)-set(PROTECTED_UNITS)) # although its impossible to have unavailable units
            IMPLEMENTED=list(set(IMPLEMENTED)-set(PROTECTED_UNITS))

            self.stream.write(f'Round {round}. Unprotected Implemented: {IMPLEMENTED}, Unimplemented: {UNIMPLEMENTED}')

            if len(UNIMPLEMENTED)==0 and round>1: # round 1 is selected unit, naturally no unimplemented units, consume the post refinement count only if round>1
                post_refinement+=1

            ################# SELECTING THE NEXT UNIT TO WORK ON #################

            DESIGN_IMPLEMENTER=reload_role('design_implementer',self.agents['DESIGN_IMPLEMENTER'],P.GUM_DESIGNER_SYSTEM(
                GAB_BASE=GAB_BASE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE))
            design_implementer_tid=self.dialog.fork(main_tid,USER_CALLER,DESIGN_IMPLEMENTER,context=context_design_implementer,
                                                alias='design_implementation',note=f'Starting design implementation...')
            
            if round>1: # if round > 1, let the agent choose the next unit to work on, TODO: maybe more background about previous rounds
                with self.status_handler('Selecting the next unit to work on...'):
                    GUM_IMPLEMENTATION_UNIT_SELECTION=P.gen_GUM_IMPLEMENTATION_UNIT_SELECTION(
                        IMPLEMENTED+UNIMPLEMENTED,post_refining=len(UNIMPLEMENTED)==0)
                    gu_implementation_unit_selection_prompt=GUM_IMPLEMENTATION_UNIT_SELECTION(
                        PROPOSAL=proposal.proposal,REVIEW=proposal.review,RATING=proposal.rating,
                        VIEW=VIEW_DETAILED,LOG='\n'.join(LOG)
                    )
                    GUM_IMPLEMENTATION_UNIT_SELECTION.apply(DESIGN_IMPLEMENTER.obj)
                    self.print_details(DESIGN_IMPLEMENTER.obj,context_design_implementer,gu_implementation_unit_selection_prompt)
                    self.stream.write(f'{VIEW_DETAILED}\n\nNow selecting the next unit to work on...')
                    
                    _,out=self.dialog.call(design_implementer_tid,gu_implementation_unit_selection_prompt)
                    selection,motivation,rough_plan,termination=out['selection'],out['motivation'],out['rough_plan'],out['termination']
                    context_design_implementer=self.dialog.context(design_implementer_tid) # update context with tree view background
                    self.stream.write(f'### Selection: {selection}')
                    self.stream.write(f'### Motivation\n{motivation}')    
                    self.stream.write(f'### Rough Plan\n{rough_plan}')
            else: # round 1, work on the selected unit
                selection=proposal.selection
                termination=False
            LOG.append(f'Round {round} started. Implementing unit {selection}.')

            if selection in IMPLEMENTED:
                self.stream.write(f'##### Start design refinement of {selection}')
            else:
                self.stream.write(f'##### Start design implementation of {selection}')

            ### Termination

            # 1. design succeeded, post refinement count reached
            if post_refinement>self.max_attemps['post_refinement']:
                self.stream.write(f'#### All units have been implemented and maximal refinements are reached, stopping design process')
                SUCCEED=True
                self.stream.log(EndReasons.MAX_POST_REFINEMENT_REACHED,'end')
                break
            
            # 2. design succeeded, agent choose to terminate
            if termination and len(UNIMPLEMENTED)==0:
                self.stream.write(f'#### All units have been implemented, the agent choose to terminate the design process')
                SUCCEED=True
                self.stream.log(EndReasons.AGENT_TERMINATION,'end')
                break

            # 3. design failed, force stop conditions
            if (self.termination['max_total_budget']>0 and self.total_cost>self.termination['max_total_budget']):
                self.stream.write(f'#### Max total budget reached, stopping design process')
                self.stream.log(EndReasons.MAX_TOTAL_BUDGET_REACHED,'end')
                break
            if (self.termination['max_debug_budget']>0 and self.implementation_cost>self.termination['max_debug_budget']):
                self.stream.write(f'#### Max debug budget reached, stopping design process')
                self.stream.log(EndReasons.MAX_DEBUG_BUDGET_REACHED,'end')
                break
            if (self.termination['max_failed_rounds']>0 and self.failed_rounds>self.termination['max_failed_rounds']):
                self.stream.write(f'#### Max failed rounds reached, stopping design process')
                self.stream.log(EndReasons.MAX_FAILED_ROUNDS_REACHED,'end')
                break

            ################# UNIT IMPLEMENTATION INNER LOOP #################

            tree_backup=copy.deepcopy(self.tree) # backup the tree for rollback
            for attempt in range(self.max_attemps['implementation_debug']):
                DESIGN_IMPLEMENTER=reload_role('design_implementer',self.agents['DESIGN_IMPLEMENTER'],P.GUM_DESIGNER_SYSTEM(
                    GAB_BASE=GAB_BASE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE))
                design_implementer_tid=self.dialog.fork(main_tid,USER_CALLER,DESIGN_IMPLEMENTER,context=context_design_implementer,
                                                    alias='design_implementation',note=f'Starting design implementation...')
                if attempt==0: # first attempt, implement the unit
                    status_info=f'Starting design implementation of {selection}...'
                    if selection in IMPLEMENTED:
                        REFINE=True 
                        GUM_IMPLEMENTATION_UNIT=P.gen_GUM_IMPLEMENTATION_UNIT(refine=True,begin=round==1) # first round can only be an implemented unit
                        node=self.tree.units[selection]
                        if round>1: # round > 1, use unit implementation prompt, tree view background is already in context
                            gu_implement_unit_prompt=GUM_IMPLEMENTATION_UNIT(
                                SPECIFICATION=node.spec.to_prompt(),
                                IMPLEMENTATION=node.code,
                                REVIEW=node.review,
                                RATING=node.rating,
                                SUGGESTIONS=node.suggestions, 
                                CHILDREN=node.children
                            )
                        else: # round 1, use unit implementation prompt with tree view background, context is empty
                            gu_implement_unit_prompt=GUM_IMPLEMENTATION_UNIT(
                                SPECIFICATION=node.spec.to_prompt(),
                                IMPLEMENTATION=node.code,
                                REVIEW=node.review,
                                RATING=node.rating,
                                SUGGESTIONS=node.suggestions,
                                VIEW=VIEW_DETAILED, 
                                CHILDREN=node.children,
                                PROPOSAL=proposal.proposal,
                                PREVIEW=proposal.review,
                                PRATING=proposal.rating,
                            )
                    else:
                        REFINE=False # implement a new unit
                        GUM_IMPLEMENTATION_UNIT=P.gen_GUM_IMPLEMENTATION_UNIT(refine=False)
                        declaration=self.tree.declares[selection]
                        gu_implement_unit_prompt=GUM_IMPLEMENTATION_UNIT(DECLARATION=declaration.to_prompt())
                    GUM_IMPLEMENTATION_UNIT.apply(DESIGN_IMPLEMENTER.obj)
                else: # Debugging or refining the implementation
                    status_info=f'Refining design implementation of {selection} (attempt {attempt})...'
                    REFINE=True
                    # if round>1: 
                    RETRY_RPOMPT=P.GU_IMPLEMENTATION_UNIT_RETRY
                    # else:
                    #     RETRY_RPOMPT=P.GUM_IMPLEMENTATION_UNIT_REFINE
                    gu_implement_unit_prompt=RETRY_RPOMPT(
                        FORMAT_CHECKER_REPORT=FORMAT_CHECKER_REPORT,
                        FUNCTION_CHECKER_REPORT=FUNCTION_CHECKER_REPORT,
                        REVIEW=review if USE_PAIRING else 'REVIEW_DISABLED',
                        RATING=rating if USE_PAIRING else 'REVIEW_DISABLED',
                        SUGGESTIONS=suggestions if USE_PAIRING else 'REVIEW_DISABLED',
                        PASS_OR_NOT='Accept' if rating>3 else 'Reject' if USE_PAIRING else 'REVIEW_DISABLED',
                        GAU_BASE=GAU_BASE
                    )
                    RETRY_RPOMPT.apply(DESIGN_IMPLEMENTER.obj)


                with self.status_handler(status_info): # calling the agent
                    self.print_details(DESIGN_IMPLEMENTER.obj,context_design_implementer,gu_implement_unit_prompt)
                    _,out=self.dialog.call(design_implementer_tid,gu_implement_unit_prompt)
                    context_design_implementer=self.dialog.context(design_implementer_tid)
                    reflection,changes,debugging_steps=None,None,None
                    if REFINE: # 1. working on an existing unit 2. all >0 attempts
                        reflection,analysis,implementation,changes=out['reflection'],out['analysis'],out['implementation'],out['changes']
                        self.stream.write(f'### Reflection\n{reflection}')
                        self.stream.write(f'## Refinement of {selection}')
                        if 'debugging_steps' in out:
                            debugging_steps=out['debugging_steps']
                            if isinstance(debugging_steps,list):
                                self.stream.write(f'### Debugging Steps\n')
                                for idx,step in enumerate(debugging_steps):
                                    self.stream.write(f'##### Diagnosis {idx+1}\n{step["diagnosis"]}\n')
                                    self.stream.write(f'##### Suggested Action {idx+1}\n{step["suggested_action"]}\n')
                            else:
                                self.stream.write(f'### Debugging Steps\n{debugging_steps}')
                        if round==1 and attempt==0:
                            NEWNAME=out['newname']
                            self.stream.write(f'#### New Name: {NEWNAME}')
                        if selection in IMPLEMENTED: # update the unit spec for now, unit name update at the end
                            spec=self.tree.units[selection].spec
                        else: # possible when debugging the implementation of a new unit
                            spec = P.UnitSpec(
                                unitname=selection,
                                document='',
                                inputs=declaration.inputs,
                                outputs=declaration.outputs
                            )
                    else: # only for the first attempt of a new unit, the reason we do this is that the response format is different for this case
                        implementation,analysis=out['implementation'],out['analysis']
                        self.stream.write(f'## Implementation of {selection}')
                        spec = P.UnitSpec(
                            unitname=selection,
                            document='',
                            inputs=declaration.inputs,
                            outputs=declaration.outputs
                        )                    
                    self.stream.write(analysis)
                    self.stream.write(f'### Code\n```python\n{implementation}\n```')
                    if REFINE:
                        self.stream.write(f'### Changes\n{changes}')
                    
                    self.print_raw_output(out,'DESIGN_IMPLEMENTER')


                # Run all checks for every implementations, optimize both grammar and semantics at the same time 
                # avoid redundant debugging steps, i.e. only the debug for the passed plans are needed
                with self.status_handler('Checking the implementation of the selected unit...'):
                    # 1. check the format code for GAU
                    reformatted_code,new_args,gau_tests,format_errors,format_warnings,fetal_errors,docstring,children_decl=check_and_reformat_gau_code(implementation,selection)
                    spec.document=docstring


                    children = {child.unitname: child for child in children_decl}
                    # never overwrite existing ones, as the children might be reused
                    NEW_DECLARED = []
                    for childname,child in children.items():
                        if childname not in self.tree.declares and childname not in self.tree.units: # only add new ones
                            self.tree.declares[childname]=child
                            NEW_DECLARED.append(childname)

                    self.stream.write(f'### Children')
                    for childname,child in children.items():
                        self.stream.write(f'##### {childname}\n'+child.to_prompt())
                    
                    collapse_write(
                        self.stream,
                        'Code format check',
                        (
                            f'\n\n#### Format Check Passed: {len(format_errors+fetal_errors)==0}\n\n'
                            f'#### Document\n{docstring}\n\n'
                            f'#### Reformatted Code\n\n```python\n{reformatted_code}\n```\n\n'
                            f'#### New Arguments\n{new_args}\n\n'
                            f'#### Format Errors\n{format_errors+fetal_errors}\n\n'
                            f'#### Format Warnings\n{format_warnings}\n\n'
                        )
                    )
                    format_checks = {
                        'format_errors':format_errors+fetal_errors,
                        'format_warnings':format_warnings,
                    }
                    # !!!TODO: the reformatter should rename the existing kwargs from the tree, okey for root node
                    # !!!TODO: remove any possible if __name__=='__main__' method from the code
                    # 2. check the functionality of the composed GAB
                    checkpass=False
                    func_checks = {}
                    if selection not in self.tree.units:
                        self.tree.add_unit(
                            spec,reformatted_code,new_args,analysis,None,None,list(children.keys()),gau_tests,None,requirements=declaration.requirements
                        )
                    else:
                        self.tree.units[selection].code=reformatted_code
                        self.tree.units[selection].args=new_args
                        self.tree.units[selection].analysis=analysis
                        self.tree.units[selection].children=list(children.keys())
                        self.tree.units[selection].gau_tests=gau_tests
                        # TODO: remove disconnected units
                    if fetal_errors==[]:
                        # run unit tests
                        _unit_test_results, _unit_test_code, _unit_test_passed = self.tree.test_unit(spec.unitname, True)
                        self.stream.write(f'### Unit Tests Passed: {_unit_test_passed}')
                        # self.stream.write(f'### Unit Tests Code\n```python\n{_unit_test_code}\n```')
                        self.stream.write(f'### Unit Tests Results\n```bash\n{_unit_test_results}\n```')

                        gabcode = self.tree.compose()
                        checkpass,check_report,gabcode_reformat,check_results = self.system.checker.check(self.system._cfg,gabcode,selection)

                        if not _unit_test_passed:
                            if 'All tests passed!' in check_report:
                                check_report = check_report.replace('All tests passed!','Checker checks passed, but unit tests failed. You must implement the unit tests and pass them.')
                    
                        self.stream.write(f'### Check passed: {checkpass}')
                        self.stream.write(f'### Check Report\n```python\n{check_report}\n```')
                        self.stream.write(f'### Check Output\n```python\n{check_results}\n```')
                        self.stream.write(f'### Reformatted GAB Code\n```python\n{gabcode_reformat}\n```')
                        
                        checkpass = checkpass and _unit_test_passed
                        checker_report = check_report
                        check_report = f'### Unit tests\n```bash\n{_unit_test_results}\n```\n\n### Checkers report\n```bash\n{check_report}\n```\n\n'
                    else:
                        check_report = 'Format check failed with fetal errors, please fix the format errors and try again.'
                        checker_report = 'Format check failed with fetal errors, please fix the format errors and try again.'
                        check_results={}
                        gabcode_reformat=None

                    func_checks = {
                        'checkpass':checkpass,
                        'check_report':check_report,
                        'check_results':check_results,
                    }

                ########################### Review the implementation ###########################
                
                if USE_PAIRING:
                    IMPLEMENTATION_REVIEWER=reload_role('implementation_reviewer',self.agents['IMPLEMENTATION_REVIEWER'], 
                                                        P.GUM_IMPLEMENTATION_REVIEWER_SYSTEM())
                    implementation_reviewer_tid=self.dialog.fork(main_tid,USER_CALLER,IMPLEMENTATION_REVIEWER,context=context_implementation_reviewer,
                                                        alias='implementation_review',note=f'Reviewing implementation...')
                    if REFINE:
                        if attempt==0:
                            status_info=f'Reviewing refinement of {selection}...'
                            gum_implementation_unit_review_prompt=P.GUM_IMPLEMENTATION_UNIT_REFINE_REVIEW(
                                UNIT_NAME=selection,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                                CHANGES=changes,PROPOSAL=proposal.proposal, CHECKER_REPORT=checker_report,
                                VIEW=VIEW_DETAILED, DESCRIPTION=node.desc,REVIEW=node.review,
                                RATING=node.rating,SUGGESTIONS=node.suggestions,SPECIFICATION=node.spec.to_prompt()
                            )
                            P.GUM_IMPLEMENTATION_UNIT_REFINE_REVIEW.apply(IMPLEMENTATION_REVIEWER.obj)
                        else:
                            status_info=f'Reviewing refined implementation of {selection} (version {attempt})...'
                            gum_implementation_unit_review_prompt=P.GUM_IMPLEMENTATION_REREVIEW(
                                UNIT_NAME=selection,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                                CHANGES=changes,SPECIFICATION=spec.to_prompt(), CHECKER_REPORT=checker_report
                            )
                            P.GUM_IMPLEMENTATION_REREVIEW.apply(IMPLEMENTATION_REVIEWER.obj)
                    else: # first attempt of a new unit
                        status_info=f'Reviewing implementation of {selection}...'
                        gum_implementation_unit_review_prompt=P.GUM_IMPLEMENTATION_UNIT_REVIEW(
                            UNIT_NAME=selection,PROPOSAL=proposal.proposal,REVIEW=proposal.review,
                            RATING=proposal.rating,VIEW=VIEW_DETAILED,SPECIFICATION=spec.to_prompt(),
                            ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,CHECKER_REPORT=checker_report,
                        )
                        P.GUM_IMPLEMENTATION_UNIT_REVIEW.apply(IMPLEMENTATION_REVIEWER.obj)

                    with self.status_handler(status_info):
                        self.print_details(IMPLEMENTATION_REVIEWER.obj,context_implementation_reviewer,gum_implementation_unit_review_prompt)
                        _,out=self.dialog.call(implementation_reviewer_tid,gum_implementation_unit_review_prompt)
                        review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                        context_implementation_reviewer=self.dialog.context(implementation_reviewer_tid)
                        passornot='Accept' if rating>3 else 'Reject'
                        self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                        self.stream.write(review)
                        self.stream.write(suggestions)
                        self.print_raw_output(out,'IMPLEMENTATION_REVIEWER')

                        self.tree.units[selection].rating=rating
                        self.tree.units[selection].review=review
                        self.tree.units[selection].suggestions=suggestions
                else:
                    review=None
                    rating=None
                    suggestions=None

                ########################### Attempt finished ###########################  
                if USE_PAIRING:
                    review_pass=rating>3
                else:
                    review_pass=True
                design = {
                    'unit': self.tree.units[selection].json(),
                    'gab_code':gabcode_reformat,
                    'format_checks':format_checks,
                    'func_checks':func_checks,
                    'reflection':reflection,
                    'debugging_steps':debugging_steps,
                    'changes':changes,
                }
                traces.append(design)

                if not checkpass or not review_pass or len(format_errors)>0: # failed  
                    succeed=False
                    self.tree=tree_backup # restore the tree
                    # if selection in UNIMPLEMENTED: 
                    #     self.tree.del_unit(selection) # remove the unit 
                    # else:
                    #     self.tree.units[selection]=node_backup # restore the unit
                    # for childname in new_declared: 
                    #     self.tree.del_declare(childname) # remove the new declared children to restore the tree
                    FORMAT_CHECKER_REPORT = P.gen_FORMAT_CHECKER_REPORT(
                        RESULT='failed' if len(format_errors+fetal_errors)>0 else 'passed',
                        ERRORS=format_errors+fetal_errors,
                        WARNINGS=format_warnings
                    )
                    if len(fetal_errors)>0:
                        FUNCTION_CHECKER_REPORT = 'Functionality check skipped due to fetal format errors.'
                    else:
                        if checkpass:
                            FUNCTION_CHECKER_REPORT = P.FUNCTION_CHECKER_REPORT_PASS.format(
                                REPORT=check_report,
                            )
                        else:
                            gabcode_reformat_with_line_num = '\n'.join([f'{i+1}: {line}' for i,line in enumerate(gabcode_reformat.split('\n'))])
                            FUNCTION_CHECKER_REPORT = P.FUNCTION_CHECKER_REPORT_FAIL.format(
                                REPORT=check_report,
                                GAB_CODE_WITH_LINE_NUM=gabcode_reformat_with_line_num
                            )
                else:
                    succeed=True
                    self.tree.units[selection].design_traces=traces
                    # removed=self.tree.clear_disconnected() # there might be some disconnected units, leave it for now
                    # PROTECTED_UNITS=list(set(PROTECTED_UNITS)-set(removed))
                    self.stream.write(f'#### Implementation passed, starting the next unit')
                    break
            
            ########################### Round finished ###########################  
            if not succeed:
                self.stream.write(f'#### Implementation failed, trying the next unit')
                RET={
                    'round':round,
                    'succeed':False,
                    'unit_design':design,
                    'unit_design_traces':traces,
                }
                RETS['ROUNDS'].append(RET)
                self.failed_rounds+=1
                LOG.append(f'Round {round} finished. Failed to implement unit {selection}.')
            else:
                RET={
                    'round':round,
                    'succeed':True,
                    'unit_design':design,
                    'unit_design_traces':traces,
                }
                RETS['ROUNDS'].append(RET)
                LOG.append(f'Round {round} finished. Successfully implemented unit {selection}.')
                if NEW_DECLARED:
                    LOG.append(f'Newly declared units in Round {round}: {NEW_DECLARED}.')
                
        ########################### Design finished ###########################  
        # self.tree.rename_unit(proposal['selection'],NEWNAME)
        self.tree.clear_disconnected() 
        RETS['new_unit_name']=NEWNAME
        RETS['SUCCEED']=SUCCEED
        if SUCCEED:
            self.stream.write('#### Design Implementation succeeded!')
            try:
                self.stream.balloons()
            except:
                pass
        else:
            self.stream.write('#### Design Implementation failed!')
            try:
                self.stream.snow()
            except:
                pass
        return RETS
    
    
    @register_module(
        "EXIT",
        hints="output the initial threads",
    )
    def end_of_design(self,query,state):

        # nothing to do here, as the ptree is already updated

        return query,state,{}

def gu_design_mutation(system,stream,design_id,design_cfg,user_input='',proposal=None):
    main_tid = system.dialog.fork(0,note='Starting a new session...',alias='main')
    status_handler = stream.status
    gu_flow = GUFlowMutation(system, status_handler,stream,design_id,design_cfg,user_input)
    GU_CALLEE = ROLE('GAB Unit Designer',gu_flow.flow)
    gum_tid = system.dialog.fork(main_tid,SYSTEM_CALLER,GU_CALLEE,note=f'launch design flow',alias=f'gu_mutate')
    system.dialog.call(gum_tid,'',main_tid=main_tid,proposal=proposal) 

    # costs=ret['costs']
    # succeed=ret['succeed']
    # design_stack=ret['design_stack']
    # new_tree=gu_flow.tree
    # new_name=design_stack['new_name']
    # return new_tree,new_name,design_stack,costs,succeed

# endregion
