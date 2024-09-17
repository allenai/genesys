import os
from enum import Enum
import inspect
import copy
from dataclasses import dataclass
from typing import Optional
import datetime

from exec_utils.models.model import ModelOutput
from exec_utils import SimpleLMAgent
from .alang import FlowCreator,register_module,ROLE,SYSTEM_CALLER,USER_CALLER,AgentContext
from .gau_utils import check_and_reformat_gau_code

# from model_discovery.system import ModelDiscoverySystem
import model_discovery.agents.prompts.prompts as P
from model_discovery.model.composer import GAUBase, GAUTree, GABComposer
from model_discovery.configs.gam_config import GAMConfig_14M
from model_discovery.model.utils.modules import GABBase, UnitDecl
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
        stream.write('---')
        stream.write(
            f"""<details><summary>Raw output</summary>{out}</details>""",
            unsafe_allow_html=True
        )
        stream.write(
            f"""<details><summary>Raw text</summary>{out['text']}</details>""",
            unsafe_allow_html=True
        )



###################################################################
# Design Flow from Existing Design
###################################################################




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
    Do not allow root for now
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
        passed_proposals,_=self.ptree.session_proposals(self.design_id,passed_only=True)
        remaining_samples=self.num_samples['proposal']-len(passed_proposals)
        self.stream.write(f'{len(passed_proposals)} proposals passed yet. Remaining {remaining_samples} proposal{"s" if remaining_samples>1 else ""} to generate.')
        for _ in range(remaining_samples):
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
        REVIEW_THRESHOLD=self.threshold['proposal_rating']

        with self.status_handler('Starting the design process, seeds sampled.'):
            self.stream.write(query,unsafe_allow_html=True)

        self.stream.write(f'#### Start design process by generating a design proposal')
        self.stream.write(f'###### **Current time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}**')

        USE_2STAGE=self.search_settings['proposal_search']
        USE_ISEARCH=self.agent_types['SEARCH_ASSISTANT']=='None' and USE_2STAGE 
        USE_2STAGE=USE_2STAGE and not USE_ISEARCH

        traces=[]
        context_design_proposer=AgentContext()
        context_proposal_reviewer=AgentContext()
        SELECTIONS=list(set(self.seed_tree.units.keys())-{self.seed_tree.root.spec.unitname})
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
                    # TODO: perplexity context maintainance
                    with self.status_handler(f'Searching... round {i+1}...'):
                        search_ret=self.sss(keywords,detail,analysis=analysis)
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
                    selection,proposal,modelname,variantname,changes=out['selection'],out['proposal'],out['modelname'],out['variantname'],out.get('changes',None)
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
                    selection,proposal,modelname,variantname,changes=out['selection'],out['proposal'],out['modelname'],out['variantname'],out.get('changes',None)
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
                    selection,proposal,modelname,variantname=out['selection'],out['proposal'],out['modelname'],out['variantname']
                    self.stream.write(f'### Design Name: {modelname}')
                    self.stream.write(f'### Selection: {selection}')
                    self.stream.write(f'### Variant Name: {variantname}')
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
            _,top_k_pps=self.sss.query_design_proposals(proposal)
            self.stream.write(top_k_pps)
            if attempt==0:
                status_info=f'Reviewing initial proposal...'
                REVIEW_PROMPT=P.GUM_PROPOSAL_REVIEW_ISEARCH_BEGIN if USE_ISEARCH_REVIEW else P.GUM_PROPOSAL_REVIEW
                proposal_review_prompt=REVIEW_PROMPT(
                    SEED=query,SELECTION=selection,PROPOSAL=proposal,TOP_K_PPS=top_k_pps)
                REVIEW_PROMPT.apply(PROPOSAL_REVIEWER.obj)
            else:
                status_info=f'Reviewing refined proposal (version {attempt})...'
                REREVIEW_PROMPT=P.GUM_PROPOSAL_REREVIEW_ISEARCH if USE_ISEARCH_REVIEW else P.GUM_PROPOSAL_REREVIEW
                proposal_review_prompt=REREVIEW_PROMPT(
                    SELECTION=selection,PROPOSAL=proposal,CHANGES=changes,TOP_K_PPS=top_k_pps)
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
                        search_ret=self.sss(keywords,detail,analysis=analysis)
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
                    passornot='Pass' if rating>=REVIEW_THRESHOLD else 'Fail'
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
                    passornot='Pass' if rating>=REVIEW_THRESHOLD else 'Fail'
                    self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                    self.stream.write(review)
                    self.stream.write(suggestions)
                    self.print_raw_output(out,'PROPOSAL_REVIEWER')

            trace={
                # proposal content
                'selection':selection,
                'modelname':modelname,
                'variantname':variantname,
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

            if rating>=REVIEW_THRESHOLD:
                self.stream.write(f'#### Proposal passed with rating {rating} out of 5, starting implementation')
                break
        
        if rating<REVIEW_THRESHOLD:
            self.stream.write(f'#### Proposal failed with rating {rating} out of 5, stopping design process')
            # raise Exception('Design proposal failed, stopping design process')
        RET={
            'proposal':trace,
            'proposal_traces':traces,
            'proposal_passed':rating>=REVIEW_THRESHOLD,
        }
        return query,state,RET
        

    def rerank_proposals(self,proposals,acronyms): # now simply rank by rating, TODO: improve it
        rerank={}
        proposal_dict=zip(acronyms,proposals)
        rank=sorted(proposal_dict,key=lambda x:x[1].rating,reverse=True)
        rerank['rank']=[x[0] for x in rank]
        return rerank

    def reranked_proposal(self):
        # select the highest rated unimplemented proposal
        proposals=[]
        acronyms=[] 
        rerank=self.ptree.session_get(self.design_id,'reranked')
        if not rerank:
            proposals,acronyms=self.ptree.session_proposals(self.design_id,passed_only=True)
            rerank=self.rerank_proposals(proposals,acronyms)
            self.ptree.session_set(self.design_id,'reranked',rerank)
        for acronym in rerank['rank']:
            design=self.ptree.get_node(acronym)
            if not design.is_implemented():
                proposals.append(design.proposal)
                acronyms.append(acronym)
        return proposals,acronyms

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
            proposals,acronyms=self.reranked_proposal()
        else:
            proposals=[proposal]
            acronyms=[proposal.acronym]
        self.stream.write(f'Implementing {len(proposals)} proposals.')
        for proposal,acronym in zip(proposals,acronyms):
            self.stream.write(f'Implementing proposal: {proposal.modelname} with rating {proposal.rating} out of 5')
            tree_ckpt=self.ptree.get_implementation_checkpoint(acronym)
            if tree_ckpt is None:
                self.tree=copy.deepcopy(self.seed_tree)
                self.tree.name=proposal.modelname
            else:
                self.tree=copy.deepcopy(tree_ckpt)
            cost_raw=copy.deepcopy(self.costs)
            RETS=self._implement_proposal_recursive(main_tid,proposal,acronym,resume=tree_ckpt is not None)
            costs={k:v-cost_raw[k] for k,v in self.costs.items()}
            ROUNDS,SUCCEED=RETS['ROUNDS'],RETS['SUCCEED']
            status='implemented' if SUCCEED else 'failed'
            self.ptree.implement(acronym,self.tree,ROUNDS,status,costs,self.design_cfg,self.user_input)
        return query,state,{}
   
    def check_code_format(self,code,selection=None,spec=None,analysis=None):
        # 1. check the format code for GAU
        reformatted_code,new_args,gau_tests,format_errors,format_warnings,fetal_errors,docstring,children_decl, unit_name=check_and_reformat_gau_code(code,selection)
        
        format_checks = {
            'format_errors':format_errors+fetal_errors,
            'format_warnings':format_warnings,
        }

        if children_decl is not None:
            # never overwrite existing ones, as the children might be reused
            NEW_DECLARED = []
            for child_decl in children_decl:
                if child_decl.unitname not in self.tree.declares and child_decl.unitname not in self.tree.units: # only add new ones
                    self.tree.declares[child_decl.unitname]=child_decl
                    NEW_DECLARED.append(child_decl.unitname)
        else: # must be fetal error
            NEW_DECLARED=[]

        if fetal_errors:
            return format_checks,format_errors,format_warnings,fetal_errors, unit_name, reformatted_code, docstring, new_args, gau_tests, children_decl, NEW_DECLARED
        
        if spec is not None:
            spec.document=docstring


        self.stream.write(f'#### Children in {unit_name}')
        if children_decl==[]:
            self.stream.write('No children declared.')
        else:
            for child in children_decl:
                self.stream.write(f'##### {child.unitname}\n'+child.to_prompt())
        
        # collapse_write(
        #     self.stream,
        #     'Code format check for '+unit_name,
        #     (
        #         f'\n\n#### Format Check Passed: {len(format_errors+fetal_errors)==0}\n\n'
        #         f'#### Document\n{docstring}\n\n'
        #         f'#### Reformatted Code\n\n```python\n{reformatted_code}\n```\n\n'
        #         f'#### New Arguments\n{new_args}\n\n'
        #         f'#### Format Errors\n{format_errors+fetal_errors}\n\n'
        #         f'#### Format Warnings\n{format_warnings}\n\n'
        #     )
        # )

        # !!!TODO: the reformatter should rename the existing kwargs from the tree, okey for root node
        # !!!TODO: remove any possible if __name__=='__main__' method from the code
        if selection is not None: # for non o1 mode
            if unit_name not in self.tree.units:
                if spec is not None:
                    self.tree.add_unit(
                        spec,reformatted_code,new_args,analysis,None,None,[child.unitname for child in children_decl],gau_tests,None,requirements=declaration.requirements
                    )
            else:
                self.tree.units[unit_name].code=reformatted_code
                self.tree.units[unit_name].args=new_args
                self.tree.units[unit_name].analysis=analysis
                self.tree.units[unit_name].children=[child.unitname for child in children_decl]
                self.tree.units[unit_name].gau_tests=gau_tests
            
        return format_checks,format_errors,format_warnings,fetal_errors,unit_name, reformatted_code, docstring, new_args, gau_tests, children_decl, NEW_DECLARED
                        

    def _implement_proposal_recursive(self,main_tid,proposal,acronym,resume=False):
        '''
        1. Implement the selected unit first
        2. Implement any unimplemented newly declared units
        3. Do post refinement, if new units defined, go to 2, post refinement count will not be refreshed
        '''

        self.dialog=self.system.dialog
        OBSERVE_THRESHOLD=self.threshold['implementation_rating']
        cost_raw=copy.deepcopy(self.costs)

        RETS={}
        RETS['ROUNDS']=[]
        SUCCEED=False
        LOG=[]
        round=0
        # XXX: Protected units should be self.seed_tree.units.keys() - self.tree.units.keys(), but this make things simpler
        PROTECTED_UNITS=list(set(self.tree.units.keys())-set([proposal.selection])) # the units besides the current one, they should not be *modified*, can be removed as descendants
        self.stream.write(f'##### Protected Units: {PROTECTED_UNITS}')
        USE_PAIRING=self.agent_types['IMPLEMENTATION_OBSERVER']!='None'
        # o1 beta does not support structured outputs, so let it output the code directly
        USE_O1_CODER=self.agents['IMPLEMENTATION_CODER'].model_name in ['o1-mini','o1-preview']

        context_implementation_planner=AgentContext() # context accummulated for all attempts in one unit
        

        post_refinement=0 # TODO: introduce self-evaluate to post-refinement
        while True:
            round+=1 # Each round works on one unit at a time, start counting from beginning so that we dont wrap it in wrong place
            if resume: # resume from a checkpoint, do not force selecting the selection
                round+=1
            traces=[]
            context_implementation_coder=AgentContext()
            context_implementation_observer=AgentContext()

            succeed=False
            IMPLEMENTED,UNIMPLEMENTED=self.tree.check_implemented()
            # GAB_CODE=self.tree.compose()
            VIEW_DETAILED=self.tree.to_prompt(unit_code=True)
            UNIMPLEMENTED=list(set(UNIMPLEMENTED)-set(PROTECTED_UNITS)) # although its impossible to have unavailable units
            IMPLEMENTED=list(set(IMPLEMENTED)-set(PROTECTED_UNITS))

            self.stream.write(f'Round {round}. Unprotected Implemented: {IMPLEMENTED}, Unimplemented: {UNIMPLEMENTED}')

            if len(UNIMPLEMENTED)==0 and round>1: # round 1 is selected unit, naturally no unimplemented units, consume the post refinement count only if round>1
                post_refinement+=1
                

            # 1. design succeeded, post refinement count reached
            if post_refinement>self.max_attemps['post_refinement']:
                self.stream.write(f'#### All units have been implemented and maximal refinements are reached, stopping design process')
                SUCCEED=True
                self.stream.log(EndReasons.MAX_POST_REFINEMENT_REACHED,'end')
                break
            

            ################# SELECTING THE NEXT UNIT TO WORK ON #################
            
            GUMT_IMPLEMENTATION_PLANNER_SYSTEM=P.gen_GUMT_IMPLEMENTATION_PLANNER_SYSTEM(use_o1=USE_O1_CODER)
            IMPLEMENTATION_PLANNER=reload_role('implementation_planner',self.agents['IMPLEMENTATION_PLANNER'],GUMT_IMPLEMENTATION_PLANNER_SYSTEM(
                GAB_BASE=GAB_BASE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE,PROPOSAL=proposal.proposal,REVIEW=proposal.review,RATING=proposal.rating))
            implementation_planner_tid=self.dialog.fork(main_tid,USER_CALLER,IMPLEMENTATION_PLANNER,context=context_implementation_planner,
                                                alias='implementation_planner',note=f'Starting implementation planning...')
            
            if round>1: # if round > 1, let the agent choose the next unit to work on, TODO: maybe more background about previous rounds
                with self.status_handler('Selecting the next unit to work on...'):
                    SELECTIONS=set(IMPLEMENTED+UNIMPLEMENTED)-{self.tree.root.spec.unitname}
                    GUM_IMPLEMENTATION_UNIT_SELECTION=P.gen_GUM_IMPLEMENTATION_UNIT_SELECTION(
                        SELECTIONS,post_refining=len(UNIMPLEMENTED)==0)
                    gu_implementation_unit_selection_prompt=GUM_IMPLEMENTATION_UNIT_SELECTION(
                        VIEW=VIEW_DETAILED,LOG='\n'.join(LOG),ROUND=round
                    )
                    GUM_IMPLEMENTATION_UNIT_SELECTION.apply(IMPLEMENTATION_PLANNER.obj)
                    self.print_details(IMPLEMENTATION_PLANNER.obj,context_implementation_planner,gu_implementation_unit_selection_prompt)
                    self.stream.write(f'{VIEW_DETAILED}\n\nNow selecting the next unit to work on...')
                    
                    _,out=self.dialog.call(implementation_planner_tid,gu_implementation_unit_selection_prompt)
                    selection,motivation,rough_plan,termination=out['selection'],out['motivation'],out['rough_plan'],out['termination']
                    context_implementation_planner=self.dialog.context(implementation_planner_tid) # update context with tree view background
                    self.stream.write(f'### Selection: {selection}')
                    self.stream.write(f'### Motivation\n{motivation}')    
                    self.stream.write(f'### Rough Plan\n{rough_plan}')
            else: # round 1, work on the selected unit
                selection=proposal.selection
                termination=False
            LOG.append(f'Round {round} started. Implementing unit {selection}.')

            if selection in IMPLEMENTED:
                self.stream.write(f'##### Start implementing refined unit {selection}')
            else:
                self.stream.write(f'##### Start implementing new unit {selection}')
            self.stream.write(f'###### **Current time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}**')


            ### Termination

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
                GUMT_IMPLEMENTATION_CODER_SYSTEM=P.gen_GUMT_IMPLEMENTATION_CODER_SYSTEM(use_o1=USE_O1_CODER)
                IMPLEMENTATION_CODER=reload_role('implementation_coder',self.agents['IMPLEMENTATION_CODER'],GUMT_IMPLEMENTATION_CODER_SYSTEM(
                    GAB_BASE=GAB_BASE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE,PROPOSAL=proposal.proposal,REVIEW=proposal.review,RATING=proposal.rating))
                implementation_coder_tid=self.dialog.fork(main_tid,USER_CALLER,IMPLEMENTATION_CODER,context=context_implementation_coder,
                                                    alias='implementation_coder',note=f'Starting design implementation...')
                if attempt==0: # first attempt, implement the unit
                    status_info=f'Starting design implementation of {selection}...'
                    if selection in IMPLEMENTED:
                        REFINE=True 
                        GUM_IMPLEMENTATION_UNIT=P.gen_GUMT_IMPLEMENTATION_UNIT(refine=True,use_o1=USE_O1_CODER) # first round can only be an implemented unit
                        node=self.tree.units[selection]
                        gu_implement_unit_prompt=GUM_IMPLEMENTATION_UNIT(
                            SPECIFICATION=node.spec.to_prompt(),
                            IMPLEMENTATION=f'```python\n{node.code}\n```',
                            REVIEW=node.review,
                            RATING=node.rating,
                            SUGGESTIONS=node.suggestions,
                            VIEW=VIEW_DETAILED, 
                            CHILDREN=node.children,
                        )
                    else:
                        REFINE=False # implement a new unit
                        GUM_IMPLEMENTATION_UNIT=P.gen_GUMT_IMPLEMENTATION_UNIT(refine=False,use_o1=USE_O1_CODER)
                        declaration=self.tree.declares[selection]
                        gu_implement_unit_prompt=GUM_IMPLEMENTATION_UNIT(DECLARATION=declaration.to_prompt())
                    GUM_IMPLEMENTATION_UNIT.apply(IMPLEMENTATION_CODER.obj)
                else: # Debugging or refining the implementation
                    status_info=f'Refining design implementation of {selection} (attempt {attempt})...'
                    REFINE=True
                    RETRY_RPOMPT=P.gen_GU_IMPLEMENTATION_UNIT_RETRY(use_o1=USE_O1_CODER)
                    if USE_PAIRING:
                        pass_or_not='Accept' if rating>=OBSERVE_THRESHOLD else 'Reject'
                    else:
                        pass_or_not='IMPLEMENTATION OBSERVER NOT AVAILABLE'
                    gu_implement_unit_prompt=RETRY_RPOMPT(
                        FORMAT_CHECKER_REPORT=FORMAT_CHECKER_REPORT,
                        FUNCTION_CHECKER_REPORT=FUNCTION_CHECKER_REPORT,
                        REVIEW=review if USE_PAIRING else 'IMPLEMENTATION OBSERVER NOT AVAILABLE',
                        RATING=rating if USE_PAIRING else 'IMPLEMENTATION OBSERVER NOT AVAILABLE',
                        SUGGESTIONS=suggestions if USE_PAIRING else 'IMPLEMENTATION OBSERVER NOT AVAILABLE',
                        PASS_OR_NOT=pass_or_not,
                        # GAU_BASE=GAU_BASE
                    )
                    RETRY_RPOMPT.apply(IMPLEMENTATION_CODER.obj)


                with self.status_handler(status_info): # calling the agent
                    self.print_details(IMPLEMENTATION_CODER.obj,context_implementation_coder,gu_implement_unit_prompt)
                    _,out=self.dialog.call(implementation_coder_tid,gu_implement_unit_prompt)
                    context_implementation_coder=self.dialog.context(implementation_coder_tid)
                    reflection,changes,debugging_steps=None,None,None
                    if REFINE: # 1. working on an existing unit 2. all >0 attempts
                        if USE_O1_CODER:
                            reflection,analysis,changes=None,None,None
                            codes=out['code']
                            implementations=[]
                            for code in codes:
                                if code.strip().startswith('# GAU_IMPLEMENTATION_FILE'):
                                    implementations.append(code)
                        else:
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
                        if USE_O1_CODER:
                            codes,analysis=out['code'],None
                            implementations=[]
                            for code in codes:
                                if code.strip().startswith('# GAU_IMPLEMENTATION_FILE'):
                                    implementations.append(code)
                        else:
                            implementation,analysis=out['implementation'],out['analysis']
                        self.stream.write(f'## Implementation of {selection}')
                        spec = P.UnitSpec(
                            unitname=selection,
                            document='',
                            inputs=declaration.inputs,
                            outputs=declaration.outputs
                        )                    
                    if analysis:
                        self.stream.write(analysis)
                    if USE_O1_CODER:
                        for idx,implementation in enumerate(implementations):
                            self.stream.write(f'### Code {idx+1}\n```python\n{implementation}\n```')
                    else:
                        self.stream.write(f'### Code\n```python\n{implementation}\n```')
                    if REFINE and changes:
                        self.stream.write(f'### Changes\n{changes}')
                    
                    self.print_raw_output(out,'IMPLEMENTATION_CODER')


                # Run all checks for every implementations, optimize both grammar and semantics at the same time 
                # avoid redundant debugging steps, i.e. only the debug for the passed plans are needed
                with self.status_handler('Checking the implementation of the selected unit...'):
                    if USE_O1_CODER:
                        format_checks={}
                        docstrings={}
                        new_args={}
                        gau_tests={}
                        children_decls={}
                        reformatted_codes={}
                        fetal_errors=[]
                        format_warnings=[]
                        format_errors=[]
                        reformatted_code=''
                        parents={}
                        NEW_DECLARED=set()
                        for idx,implementation in enumerate(implementations):
                            _format_checks,_format_errors,_format_warnings,_fetal_errors, unit_name, _reformatted_code, _docstring,_new_args,_gau_tests,_children_decl,_NEW_DECLARED=self.check_code_format(implementation)
                            NEW_DECLARED.update(_NEW_DECLARED)
                            fetal_errors.extend([f'Code block {idx+1} of {unit_name}: {e}' for e in _fetal_errors])
                            format_warnings.extend([f'Code block {idx+1} of {unit_name}: {e}' for e in _format_warnings])
                            format_errors.extend([f'Code block {idx+1} of {unit_name}: {e}' for e in _format_errors])
                            if not _docstring: 
                                continue
                            docstrings[unit_name]=_docstring
                            new_args[unit_name]=_new_args
                            gau_tests[unit_name]=_gau_tests
                            children_decls[unit_name]=_children_decl
                            for c in _children_decl:
                                parents[c.unitname]=unit_name
                            reformatted_codes[unit_name]=_reformatted_code
                            unit_name=f'None_{idx+1}' if not unit_name else unit_name
                            format_checks[unit_name]=_format_checks
                            reformatted_code+=f'### {unit_name} Reformatted Code\n```python\n{_reformatted_code}\n```\n\n'
                            collapse_write(
                                self.stream,
                                'Code format check for '+unit_name,
                                (
                                    f'\n\n#### Format Check Passed: {len(format_errors+fetal_errors)==0}\n\n'
                                    f'#### Format Errors\n{format_errors+fetal_errors}\n\n'
                                    f'#### Format Warnings\n{format_warnings}\n\n'
                                    f'\n\n{reformatted_code}\n\n'
                                )
                            )
                        root_name=[]
                        for unit_name,docstring in docstrings.items():
                            if unit_name not in parents:
                                root_name.append(unit_name)
                        self.stream.write(f'##### Found local root units: {root_name}')
                        if len(docstrings)==0:
                            fetal_errors.append(f'There is no valid GAU implementation found.')
                        else:
                            if len(root_name)==0:
                                fetal_errors.append(f'There is no root unit found, please check if there is any cycle in the dependency graph of units.')
                            elif len(root_name)>1:
                                fetal_errors.append(f'There are multiple root units found: {", ".join(root_name)}, please check if there is any cycle in the dependency graph of units.')
                            else:
                                unit_name=root_name[0]
                                reformatted_code=f'### {unit_name} Reformatted Code\n```python\n{reformatted_code}\n```\n\n'
                                # replace selection with unit_name
                                self.tree._replace_unit(selection,unit_name) # do replace first
                                LOG.append(f'Replace unit {selection} with {unit_name}')
                                self.stream.write(f'Replace local root unit {selection} with {unit_name}')
                                self.tree.declares[unit_name]=UnitDecl(
                                    unitname=unit_name,
                                    requirements='',
                                    inputs=spec.inputs,
                                    outputs=spec.outputs
                                )
                        for unit_name,docstring in docstrings.items():
                            if unit_name not in self.tree.declares and unit_name not in self.tree.units: # if it is a new local root, then it is declared above, otherwise, it should be declared as a child
                                format_errors.append(f'A new unit {unit_name} has not been declared. May cause errors when linking the units.')
                                declaration=UnitDecl(
                                    unitname=unit_name,
                                    requirements='',
                                    inputs=[],
                                    outputs=[]
                                )
                            else:
                                declaration=self.tree.declares[unit_name]
                            # XXX: whether we allow overwrite the existing unit besides selection???? 
                            if unit_name in self.tree.units: # overwrite the existing unit, remove the old one first
                                self.stream.write(f'Overwriting unit {unit_name} in the tree...')
                            else:
                                self.stream.write(f'Adding unit {unit_name} to the tree...')
                            _spec = P.UnitSpec(
                                unitname=unit_name,
                                document=docstring,
                                inputs=declaration.inputs,
                                outputs=declaration.outputs
                            )           
                            _children=[c.unitname for c in children_decls[unit_name]]
                            self.tree.add_unit( # all new units are added 
                                _spec,reformatted_codes[unit_name],new_args[unit_name],None,None,None,_children,gau_tests[unit_name],None,requirements=declaration.requirements, overwrite=True
                            )   
                        unit_name=root_name[0] if len(root_name)>0 else selection
                        
                    else:   
                        format_checks,format_errors,format_warnings,fetal_errors, unit_name, reformatted_code, _, _, _, _, NEW_DECLARED=self.check_code_format(implementation,selection,spec,analysis)
                        unit_name=selection
                        reformatted_code=f'### {unit_name} Reformatted Code\n```python\n{reformatted_code}\n```\n\n'
                        collapse_write(
                            self.stream,
                            'Code format check for '+unit_name,
                            (
                                f'\n\n#### Format Check Passed: {len(format_errors+fetal_errors)==0}\n\n'
                                f'#### Format Errors\n{format_errors+fetal_errors}\n\n'
                                f'#### Format Warnings\n{format_warnings}\n\n'
                                f'\n\n{reformatted_code}\n\n'
                            )
                        )

                    if fetal_errors==[]:
                        # run unit tests
                        if USE_O1_CODER:
                            _unit_test_passed=True
                            _unit_test_results=''
                            for unitname in format_checks.keys():
                                __unit_test_results, _unit_test_code, __unit_test_passed = self.tree.test_unit(unitname, True)
                                self.stream.write(f'### {unitname} Unit Tests Passed: {__unit_test_passed}')
                                self.stream.write(f'### {unitname} Unit Tests Results\n```bash\n{__unit_test_results}\n```')
                                collapse_write(
                                    self.stream,
                                    'Unit Tests Code for '+unitname,
                                    f'\n\n```python\n{_unit_test_code}\n```\n\n'
                                )
                                _unit_test_passed = _unit_test_passed and __unit_test_passed
                                _unit_test_results += f'### {unitname} Unit Tests Results\n```bash\n{__unit_test_results}\n```\n\n'
                        else:
                            _unit_test_results, _unit_test_code, _unit_test_passed = self.tree.test_unit(spec.unitname, True)
                            self.stream.write(f'### Unit Tests Passed: {_unit_test_passed}')
                            self.stream.write(f'### Unit Tests Results\n```bash\n{_unit_test_results}\n```')
                            collapse_write(
                                self.stream,
                                'Unit Tests Code for '+unitname,
                                f'```python\n{_unit_test_code}\n```\n'
                            )   
                            _unit_test_results += f'### Unit Tests Results\n```bash\n{_unit_test_results}\n```\n\n'

                        gabcode = self.tree.compose()
                        checkpass,check_report,gabcode_reformat,check_results = self.system.checker.check(GAMConfig_14M(),gabcode,selection)

                        if not _unit_test_passed:
                            if 'All tests passed!' in check_report:
                                check_report = check_report.replace('All tests passed!','Checker checks passed, but unit tests failed. You must implement the unit tests and pass them.')
                    
                        self.stream.write(f'### Check passed: {checkpass}')
                        self.stream.write(f'### Check Report\n```python\n{check_report}\n```')
                        self.stream.write(f'### Check Output\n```python\n{check_results}\n```')
                        self.stream.write(f'### Reformatted GAB Code\n```python\n{gabcode_reformat}\n```')
                        
                        checkpass = checkpass and _unit_test_passed
                        checker_report = check_report # Too long in the prompt
                        check_report = f'{_unit_test_results}### Checkers report\n```bash\n{check_report}\n```\n\n'
                    else:
                        check_report = 'Format check failed with fetal errors, please fix the format errors and try again.'
                        checker_report = 'Format check failed with fetal errors, please fix the format errors and try again.'
                        check_results={}
                        gabcode_reformat=None
                        checkpass=False
                        self.stream.write(f'#### Functionality check skipped due to fetal format errors \n {fetal_errors}')

                    func_checks = {
                        'checkpass':checkpass,
                        'check_report':check_report,
                        'check_results':check_results,
                    }

                ########################### Review the implementation ###########################
                
                if USE_PAIRING:
                    GUMT_IMPLEMENTATION_OBSERVER_SYSTEM=P.gen_GUMT_IMPLEMENTATION_OBSERVER_SYSTEM(use_o1=USE_O1_CODER)
                    IMPLEMENTATION_OBSERVER=reload_role('implementation_reviewer',self.agents['IMPLEMENTATION_OBSERVER'], GUMT_IMPLEMENTATION_OBSERVER_SYSTEM(
                        GAB_BASE=GAB_BASE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE,PROPOSAL=proposal.proposal,REVIEW=proposal.review,RATING=proposal.rating))
                    implementation_observer_tid=self.dialog.fork(main_tid,USER_CALLER,IMPLEMENTATION_OBSERVER,context=context_implementation_observer,
                                                        alias='implementation_observer',note=f'Reviewing implementation...')
                    if REFINE:
                        if attempt==0:
                            status_info=f'Observing refinement of {selection}...'
                            gum_implementation_unit_review_prompt=P.GUMT_IMPLEMENTATION_UNIT_REFINE_OBSERVE(
                                UNIT_NAME=selection,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                                CHANGES=changes, VIEW=VIEW_DETAILED, DESCRIPTION=node.desc,
                                REVIEW=node.review,RATING=node.rating,SUGGESTIONS=node.suggestions,
                                SPECIFICATION=node.spec.to_prompt()
                            )
                            P.GUMT_IMPLEMENTATION_UNIT_REFINE_OBSERVE.apply(IMPLEMENTATION_OBSERVER.obj)
                        else:
                            status_info=f'Observing refined implementation of {selection} (version {attempt})...'
                            gum_implementation_unit_review_prompt=P.GUMT_IMPLEMENTATION_REOBSERVE(
                                UNIT_NAME=selection,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                                CHANGES=changes,SPECIFICATION=spec.to_prompt()
                            )
                            P.GUMT_IMPLEMENTATION_REOBSERVE.apply(IMPLEMENTATION_OBSERVER.obj)
                    else: # first attempt of a new unit
                        status_info=f'Reviewing implementation of {selection}...'
                        gum_implementation_unit_review_prompt=P.GUMT_IMPLEMENTATION_UNIT_OBSERVE(
                            UNIT_NAME=selection,VIEW=VIEW_DETAILED,SPECIFICATION=spec.to_prompt(),
                            ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                        )
                        P.GUMT_IMPLEMENTATION_UNIT_OBSERVE.apply(IMPLEMENTATION_OBSERVER.obj)

                    with self.status_handler(status_info):
                        self.print_details(IMPLEMENTATION_OBSERVER.obj,context_implementation_observer,gum_implementation_unit_review_prompt)
                        _,out=self.dialog.call(implementation_observer_tid,gum_implementation_unit_review_prompt)
                        review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                        context_implementation_observer=self.dialog.context(implementation_observer_tid)
                        passornot='Accept' if rating>=OBSERVE_THRESHOLD else 'Reject'
                        self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                        self.stream.write(review)
                        self.stream.write(suggestions)
                        self.print_raw_output(out,'IMPLEMENTATION_OBSERVER')
                        
                        if unit_name in self.tree.units:
                            self.tree.units[unit_name].rating=rating
                            self.tree.units[unit_name].review=review
                            self.tree.units[unit_name].suggestions=suggestions
                else:
                    review=None
                    rating=None
                    suggestions=None

                ########################### Attempt finished ###########################  
                if USE_PAIRING:
                    review_pass=rating>=OBSERVE_THRESHOLD
                else:
                    review_pass=True
                design = {
                    'unit': self.tree.units[unit_name].json() if unit_name in self.tree.units else None,
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
                    self.tree=copy.deepcopy(tree_backup) # restore the tree
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
                    costs={k:v-cost_raw[k] for k,v in self.costs.items()}
                    self.ptree.implement(acronym,self.tree,RETS['ROUNDS'],'unfinished',costs,self.design_cfg,self.user_input)
                    cost_raw=copy.deepcopy(self.costs)
                    # self.tree.units[unit_name].design_traces=traces
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
                LOG.append(f'Round {round} finished. Failed to implement unit {unit_name} of selection {selection}.')
            else:
                RET={
                    'round':round,
                    'succeed':True,
                    'unit_design':design,
                    'unit_design_traces':traces,
                }
                RETS['ROUNDS'].append(RET)
                LOG.append(f'Round {round} finished. Successfully implemented unit {unit_name} of selection {selection}.')
                if NEW_DECLARED:
                    LOG.append(f'Newly declared units in Round {round}: {NEW_DECLARED}.')
                
        ########################### Design finished ###########################  
        self.tree.clear_disconnected() 
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

