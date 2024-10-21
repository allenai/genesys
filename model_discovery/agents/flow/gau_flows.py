import os
from enum import Enum
import inspect
import copy
from dataclasses import dataclass
from typing import Optional
import datetime
import random

from exec_utils.models.model import ModelOutput
from exec_utils import SimpleLMAgent
from .alang import FlowCreator,register_module,ROLE,SYSTEM_CALLER,USER_CALLER,AgentContext
from .gau_utils import check_and_reformat_gau_code

# from model_discovery.system import ModelDiscoverySystem
import model_discovery.agents.prompts.prompts as P
from model_discovery.model.composer import GAUBase, GAUTree, GABComposer
from model_discovery.configs.gam_config import GAMConfig_14M
from model_discovery.model.utils.modules import GABBase, UnitDecl,DesignModes
import model_discovery.utils as U


DESIGN_ZOMBIE_THRESHOLD = 300 # O1 is really really slow

LOG_STATES={
    'BEGIN':'Begin',
    'RUNNING':'Running',
    'ERROR':'Error',
    'TERMINATED':'Terminated',
    'PROPOSAL': 'Generating Proposal',
    'IMPLEMENTATION': 'Implementing Proposal',
    'EXIT': 'Design Finished',
}

DESIGN_TERMINAL_STATES=['EXIT','TERMINATED','ERROR','ZOMBIE']
DESIGN_ACTIVE_STATES=['BEGIN','RUNNING','PROPOSAL','IMPLEMENTATION']


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

gam_prompt_path = os.path.join(current_dir,'..','prompts','gam_prompt.py')
gau_template_path = os.path.join(current_dir,'..','prompts','gau_template.py')
GAM_TEMPLATE=open(gam_prompt_path).read()
GAU_TEMPLATE=open(gau_template_path).read()

GAU_BASE=inspect.getsource(GAUBase)
GAB_BASE=inspect.getsource(GABBase)

AGENT_TYPES = ['claude3.5_sonnet','gpt4o_0806','gpt4o_mini','o1_preview','o1_mini']
AGENT_OPTIONS = {
    'DESIGN_PROPOSER':AGENT_TYPES,
    'PROPOSAL_REVIEWER':AGENT_TYPES,
    'IMPLEMENTATION_PLANNER':AGENT_TYPES,
    'IMPLEMENTATION_CODER':['o1_preview','o1_mini'],
    'IMPLEMENTATION_OBSERVER':AGENT_TYPES+['None'],
    'SEARCH_ASSISTANT':['claude3.5_sonnet','gpt4o_0806','gpt4o_mini','None']
}

DEFAULT_AGENT_WEIGHTS = {
    'DESIGN_PROPOSER': [0.6,0.1,0,0.2,0.1],
    'PROPOSAL_REVIEWER': [0.6,0.1,0,0.2,0.1],
    'IMPLEMENTATION_PLANNER': [0.3,0.1,0,0.2,0.4],
    'IMPLEMENTATION_CODER': [1,0],
    'IMPLEMENTATION_OBSERVER': [0.1,0.1,0,0.1,0.7,0.0],
} # THE ORDER MUST BE CORRESPONDING TO AGENT_OPTIONS

@dataclass
class AgentModelDef:
    agent: SimpleLMAgent
    model_name: Optional[str] = None


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
        try:
            stream.write(
                f"""<details><summary>Agent input context (Length: {len(context.get())})</summary>{context.get()}</details>""",
                unsafe_allow_html=True
            )
        except:
            stream.write(
                f"""<details><summary>Agent input context (Length: {len(context.get())})</summary>{context.get()}</details>""",
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
    IMPLEMENTATION_SUCCESS='Implementation Success'
    IMPLEMENTATION_FAILURE='Implementation Failure'
    PROPOSAL_FAILED='Proposal Failed'
    MAX_POST_REFINEMENT_REACHED='Max Post Refinement Reached'
    AGENT_TERMINATION='Agent Choose Termination'
    MAX_TOTAL_BUDGET_REACHED='Max Total Budget Reached'
    MAX_DEBUG_BUDGET_REACHED='Max Debug Budget Reached'
    MAX_FAILED_ROUNDS_REACHED='Max Failed Rounds Reached'

class RunningModes(Enum):
    PROPOSAL_ONLY='Proposal Only'
    IMPLEMENTATION_ONLY='Implementation Only'
    BOTH='Proposal + Implementation'

END_REASONS_LABELS = {
    EndReasons.PROPOSAL_FAILED:'Failed',
    EndReasons.IMPLEMENTATION_SUCCESS:'Success',
    EndReasons.IMPLEMENTATION_FAILURE:'Failed',
    EndReasons.MAX_POST_REFINEMENT_REACHED:'Success',
    EndReasons.AGENT_TERMINATION:'Success',
    EndReasons.MAX_TOTAL_BUDGET_REACHED:'Failed',
    EndReasons.MAX_DEBUG_BUDGET_REACHED:'Failed',
    EndReasons.MAX_FAILED_ROUNDS_REACHED:'Failed',
}

class GUFlow(FlowCreator): 
    """
    The flow for designing a GAB Flow nested of GAB Units from scratch.
    the input query should be the seeds from the root tree for the design
    Do not allow root for now
    """
    def __init__(self,system,status_handler,stream,sess_id,design_cfg,user_input='',cpu_only=False,log_fn=None,design_mode=DesignModes.MUTATION):
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
        self.cpu_only=cpu_only
        self.log_fn=log_fn if log_fn else lambda x,y=None: None
        
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
            'claude3.5_sonnet':'claude-3-5-sonnet-20240620', # maybe incompatible with exec_utils
            'gpt4o_0806':'gpt-4o-2024-08-06',
            'gpt4o_mini':'gpt-4o-mini',
            'o1_preview':'o1-preview',
            'o1_mini':'o1-mini',
            'None':None,
        }

        self.failed_rounds=0
        self.max_attemps=design_cfg['max_attemps']
        self.agent_types=design_cfg['agent_types']     
        self.unittest_pass_required=design_cfg['unittest_pass_required']
        self.agents={}

        with self.stream.status('Setting up agents'):
            for name,agent_type in self.agent_types.items():
                if agent_type == 'hybrid':
                    weights=design_cfg['agent_weights'][name]
                    agent_type=random.choices(AGENT_OPTIONS[name],weights=weights,k=1)[0]
                    self.stream.write(f'Agent ```{name}``` is randomly selected as ```{agent_type}``` with hybrid weights ```{weights}```')
                self.agents[name]=AgentModelDef(AGENT_TYPES[agent_type],AGENT_TYPES_MODEL_NAMES[agent_type])

        self.termination=design_cfg['termination']
        self.threshold=design_cfg['threshold']
        self.search_settings=design_cfg['search_settings']
        self.running_mode=design_cfg['running_mode']
        self.num_samples=design_cfg['num_samples']
        self.use_unlimited_prompt=design_cfg['use_unlimited_prompt']
        self.design_cfg=design_cfg
        self.user_input=user_input

        # assert any(self.termination.values())>0, 'At least one of the termination conditions should be set'

        self.sess_id = sess_id
        self.ptree=system.ptree
        seeds,refs,instruct,self.design_mode=self.ptree.get_session_input(sess_id)
        self.stream.write(f'Number of seeds sampled: :green[{len(seeds)}].\tNumber of references: :orange[{len(refs)}].\tWorking in design mode: :violet[{self.design_mode}]')
        self.seed_input=P.build_GU_QUERY(seeds,refs,instruct,user_input,mode=self.design_mode,mutation_no_tree=design_cfg['mutation_no_tree'])
        if self.design_mode==DesignModes.MUTATION:
            self.seed_tree = self.ptree.get_gau_tree(seeds[0].acronym)
            self.seed = seeds[0].to_prompt()
            self.seed_ids=[seeds[0].acronym]
        elif self.design_mode==DesignModes.CROSSOVER:
            self.seed_tree = self.ptree.new_gau_tree()
            seeds_prompt='\n\n'
            for idx,seed in enumerate(seeds):
                seeds_prompt+=f'---\n\n<details><summary>Parent {idx+1}</summary>{seed.to_prompt()}</details>\n\n'
            self.seed = seeds_prompt
            self.seed_ids=[seed.acronym for seed in seeds]
        elif self.design_mode==DesignModes.SCRATCH:
            self.seed_tree = self.ptree.new_gau_tree()
            self.seed = None
        else:
            raise ValueError(f'Invalid design mode: {self.design_mode}')
        
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

    def call_dialog(self,tid,prompt,context=None):
        callee=self.dialog.get_alias(tid)
        status='IMPLEMENTATION' if 'implementation' in callee else 'PROPOSAL'
        self.log_fn(f'Calling {tid} ({callee})...',status) 
        if context:
            print(f'[DEBUG] Switching context of {tid} ({callee}) to new context with length {len(context.get())}')
            self.dialog.switch_context(tid,copy.deepcopy(context))
        if self.use_unlimited_prompt:
            prompt += P.REASONING_TOKEN_UNLIMITED
        msg,out=self.dialog.call(tid,prompt)
        self.log_fn(f'{tid} ({callee}) returned.',status)
        return msg,out

    @register_module(
        "PROC",
        hints="output the initial threads",
        links='generate_proposal',
    )
    def design_initializer(self,query,state,proposal=None):
        if proposal:
            if self.running_mode!=RunningModes.IMPLEMENTATION_ONLY:
                self.stream.write(f'Proposal is provided, running mode will be forced switched to implementation only from {self.running_mode}')
                self.running_mode=RunningModes.IMPLEMENTATION_ONLY
        else:
            if self.running_mode==RunningModes.IMPLEMENTATION_ONLY:
                self.log_fn('No proposal is provided, running mode is set to proposal only','ERROR')
                raise Exception('Proposal is required for implementation only mode')
        self.log_fn(f'Starting design flow with running mode: {self.running_mode}','BEGIN')
        return query,state,{}
    
    def call_search_assistant(self,main_tid,ideation,instructions):
        raise NotImplementedError('Independent Search Assistant need to be updated')
        # self.stream.write(f'Warning: Search Assistant prompt has not been updated, performance may degrade.')
        # S2_SEARCH_SYSTEM=P.S2_SEARCH_ASSISTANT_SYSTEM()
        # S2_SEARCH_ASSISTANT=reload_role('search_assistant',self.agents['SEARCH_ASSISTANT'],S2_SEARCH_SYSTEM)
        # context_search_assistant=AgentContext()
        # search_assistant_tid=self.dialog.fork(main_tid,USER_CALLER,S2_SEARCH_ASSISTANT,context=context_search_assistant,
        #                                         alias='search_assistant',note=f'Starting search S2...')
        
        # for i in range(self.max_attemps['max_search_rounds']):
        #     self.stream.write(f'## Searching from S2, round {i+1}...')
        #     # S2 Search Query
        #     s2_search_query_prompt=P.S2_SEARCH_PROPOSAL_QUERY(IDEATION=ideation,INSTRUCTIONS=instructions)
        #     P.S2_SEARCH_PROPOSAL_QUERY.apply(S2_SEARCH_ASSISTANT.obj)
        #     self.print_details(S2_SEARCH_ASSISTANT.obj,context_search_assistant,s2_search_query_prompt)
        #     _,out=self.call_dialog(search_assistant_tid,s2_search_query_prompt)
        #     analysis,query=out['analysis'],out['query']
        #     self.stream.write(f'### Analysis\n{analysis}')
        #     self.stream.write(f'### Query\n{query}')
        #     self.log_fn(f'Searching papers...')
        #     rets=self.sss(query,analysis) # TODO: specifically prompts details for search assistant
        #     self.log_fn(f'Search finished.')
        #     self.stream.markdown(rets,unsafe_allow_html=True)
        #     self.print_raw_output(out,'SEARCH_ASSISTANT')

        #     # S2 Search Response
        #     s2_search_response_prompt=P.S2_SEARCH_PROPOSAL_RESPONSE(SEARCH_RESULTS=rets)
        #     P.S2_SEARCH_PROPOSAL_RESPONSE.apply(S2_SEARCH_ASSISTANT.obj)
        #     self.print_details(S2_SEARCH_ASSISTANT.obj,context_search_assistant,s2_search_response_prompt)
        #     _,out=self.call_dialog(search_assistant_tid,s2_search_response_prompt)
        #     report,references,continue_search=out['report'],out['references'],out['continue_search']
        #     self.stream.write(f'### Report\n{report}')
        #     self.stream.write(f'### References\n{references}')
        #     self.print_raw_output(out,'SEARCH_ASSISTANT')
        #     self.stream.write('---')
        #     if not continue_search:
        #         self.stream.write(f'### Search Assistant chose to stop search')
        #         break
        # self.stream.write(f'### Search Finished')
        # return report,references


    @register_module(
        "PROC",
        hints="output the proposal after review",
        links='implement_proposal_recursive',
    )
    def generate_proposal(self,query,state,main_tid):
        '''
        Overally evaluate the current ptree, and generate a proposal for the next step, and pick one unit to work on
        '''
        if self.running_mode==RunningModes.IMPLEMENTATION_ONLY:
            self.stream.write('Implementation only mode, skipping proposal generation...')
            return query,state,{}
        
        i=0
        while True:
            passed_proposals,_=self.ptree.session_proposals(self.sess_id,passed_only=True)
            remaining_samples=self.num_samples['proposal']-len(passed_proposals)
            info=f'{len(passed_proposals)} proposals passed yet. Remaining {remaining_samples} proposal{"s" if remaining_samples>1 else ""} to generate.'
            self.stream.write(info)
            self.log_fn(info,'PROPOSAL')
            if len(passed_proposals)>=self.num_samples['proposal']:
                break
            if not self.ptree.acquire_design_lock(self.sess_id): # For benchmark, active sessions + finished designs < max designs
                self.log_fn(f'Design lock not acquired, design is full, stopping proposal generation...','PROPOSAL')
                break
            self.log_fn(f'Generating proposal {i+1}...','PROPOSAL')
            cost_raw=copy.deepcopy(self.costs)
            query,state,RET=self._generate_proposal(self.seed_input,state,main_tid)
            costs={k:v-cost_raw[k] for k,v in self.costs.items()} # run cost
            proposal,proposal_traces=RET['proposal'],RET['proposal_traces']
            self.ptree.propose(self.sess_id,proposal,proposal_traces,costs,self.design_cfg,self.user_input)
            self.log_fn(f'Proposal {i+1} generated: {proposal.modelname} (Passed: {proposal.passed}).','PROPOSAL')
            i+=1
        self.log_fn(f'All proposals generated.','PROPOSAL')
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
        # Iterative Search is default now, designed for claude, utilizing cache mechanism
        USE_ISEARCH=self.agent_types['SEARCH_ASSISTANT']=='None' and USE_2STAGE 
        if self.design_mode!=DesignModes.MUTATION:
            # only isearch is supported for crossover and scratch
            USE_ISEARCH=True
        if USE_ISEARCH:
            USE_2STAGE=False
        UNSTRUCT_PROPOSER='o1' in self.agent_types['DESIGN_PROPOSER']
        UNSTRUCT_REVIEWER='o1' in self.agent_types['PROPOSAL_REVIEWER']

        traces=[]

        if self.design_mode!=DesignModes.SCRATCH:
            units=set(self.seed_tree.units.keys())
            SELECTIONS=units if UNSTRUCT_PROPOSER else units-{self.seed_tree.root} 
            SELECTIONS=list(SELECTIONS)
            self.log_fn(f'Searching sibling designs...','PROPOSAL')
            _,SIBLINGS=self.sss.query_sibling_designs(self.seed_ids)
            self.log_fn(f'Search finished.','PROPOSAL')

        cross_attempt_context=AgentContext()
        for attempt in range(self.max_attemps['design_proposal']):
            context_design_proposer=copy.deepcopy(cross_attempt_context.truncate(1))
            if UNSTRUCT_PROPOSER:
                if self.design_mode==DesignModes.MUTATION:
                    DESIGN_PROPOSER=reload_role('design_proposer',self.agents['DESIGN_PROPOSER'],P.O1M_PROPOSER_BACKGROUND(
                        GAU_BASE=GAU_BASE,SEED=query,SELECTIONS=SELECTIONS,SIBLINGS=SIBLINGS))
                elif self.design_mode==DesignModes.CROSSOVER:
                    DESIGN_PROPOSER=reload_role('design_proposer',self.agents['DESIGN_PROPOSER'],P.O1C_PROPOSER_BACKGROUND(
                        GAU_BASE=GAU_BASE,PARENTS=query,SIBLINGS=SIBLINGS))
                else:
                    DESIGN_PROPOSER=reload_role('design_proposer',self.agents['DESIGN_PROPOSER'],P.O1S_PROPOSER_BACKGROUND(
                        GAU_BASE=GAU_BASE,REFS=query))
            else:
                if self.design_mode==DesignModes.MUTATION:
                    DESIGN_PROPOSER_SYSTEM=P.GUM_DESIGN_PROPOSER_SYSTEM 
                    if USE_ISEARCH:
                        DESIGN_PROPOSER_SYSTEM=P.GUM_DESIGN_PROPOSER_SYSTEM_ISEARCH
                    elif USE_2STAGE:
                        DESIGN_PROPOSER_SYSTEM=P.GUM_DESIGN_PROPOSER_SYSTEM_2STAGE
                elif self.design_mode==DesignModes.CROSSOVER:
                    DESIGN_PROPOSER_SYSTEM=P.GUC_DESIGN_PROPOSER_SYSTEM_ISEARCH
                else:
                    DESIGN_PROPOSER_SYSTEM=P.GUS_DESIGN_PROPOSER_SYSTEM_ISEARCH
                DESIGN_PROPOSER=reload_role('design_proposer',self.agents['DESIGN_PROPOSER'],DESIGN_PROPOSER_SYSTEM(GAU_BASE=GAU_BASE))
            design_proposer_tid=self.dialog.fork(main_tid,USER_CALLER,DESIGN_PROPOSER,context=context_design_proposer,
                                                alias='design_proposer',note=f'Starting design proposal...')
            if attempt==0:
                status_info=f'Initial design proposal...'
                if self.design_mode==DesignModes.MUTATION:
                    if UNSTRUCT_PROPOSER:
                        GUM_DESIGN_PROPOSAL=P.O1M_DESIGN_PROPOSAL
                        proposal_prompt=GUM_DESIGN_PROPOSAL()
                    else:
                        GUM_DESIGN_PROPOSAL=P.gen_GUM_DESIGN_PROPOSAL(SELECTIONS=SELECTIONS,two_stage=USE_2STAGE,use_isearch=USE_ISEARCH)
                        if USE_ISEARCH:
                            GUM_DESIGN_PROPOSAL,GU_DESIGN_PROPOSAL_FINISH=GUM_DESIGN_PROPOSAL
                        elif USE_2STAGE:
                            GUM_DESIGN_PROPOSAL,GUM_DESIGN_PROPOSAL_STAGE2=GUM_DESIGN_PROPOSAL
                        proposal_prompt=GUM_DESIGN_PROPOSAL(SEED=query,SIBLINGS=SIBLINGS)
                    GUM_DESIGN_PROPOSAL.apply(DESIGN_PROPOSER.obj)
                elif self.design_mode==DesignModes.CROSSOVER:
                    if UNSTRUCT_PROPOSER:
                        GUC_DESIGN_PROPOSAL=P.O1C_DESIGN_PROPOSAL
                        proposal_prompt=GUC_DESIGN_PROPOSAL()
                    else:
                        GUC_DESIGN_PROPOSAL=P.GUC_DESIGN_PROPOSAL_ISEARCH
                        GU_DESIGN_PROPOSAL_FINISH=P.GUC_DESIGN_PROPOSAL_ISEARCH_FINISH
                        proposal_prompt=GUC_DESIGN_PROPOSAL(PARENTS=query,SIBLINGS=SIBLINGS)
                    GUC_DESIGN_PROPOSAL.apply(DESIGN_PROPOSER.obj)
                else:
                    if UNSTRUCT_PROPOSER:
                        GUS_DESIGN_PROPOSAL=P.O1M_DESIGN_PROPOSAL # reuse it
                        proposal_prompt=GUS_DESIGN_PROPOSAL()
                    else:
                        GUS_DESIGN_PROPOSAL=P.GUS_DESIGN_PROPOSAL_ISEARCH
                        GU_DESIGN_PROPOSAL_FINISH=P.GUS_DESIGN_PROPOSAL_ISEARCH_FINISH
                        proposal_prompt=GUS_DESIGN_PROPOSAL(REFS=query)
                    GUS_DESIGN_PROPOSAL.apply(DESIGN_PROPOSER.obj)
            else:
                status_info=f'Refining design proposal (attempt {attempt})...'
                if self.design_mode==DesignModes.MUTATION:
                    if UNSTRUCT_PROPOSER:
                        GUM_PROPOSAL_REFINEMENT=P.O1M_DESIGN_PROPOSAL_REFINEMENT
                    else:
                        GUM_PROPOSAL_REFINEMENT=P.gen_GUM_PROPOSAL_REFINEMENT(SELECTIONS=SELECTIONS,two_stage=USE_2STAGE,use_isearch=USE_ISEARCH)
                        if USE_ISEARCH:
                            GUM_PROPOSAL_REFINEMENT,GU_DESIGN_PROPOSAL_FINISH=GUM_PROPOSAL_REFINEMENT
                        elif USE_2STAGE:
                            GUM_PROPOSAL_REFINEMENT,GUM_DESIGN_PROPOSAL_STAGE2=GUM_PROPOSAL_REFINEMENT
                    proposal_prompt=GUM_PROPOSAL_REFINEMENT(REVIEW=review,RATING=rating,SUGGESTIONS=suggestions,
                                                            PASS_OR_NOT='Pass' if rating>=4 else 'Fail')
                    GUM_PROPOSAL_REFINEMENT.apply(DESIGN_PROPOSER.obj)
                elif self.design_mode==DesignModes.CROSSOVER:
                    if UNSTRUCT_PROPOSER:
                        GUC_PROPOSAL_REFINEMENT=P.O1C_DESIGN_PROPOSAL_REFINEMENT
                    else:
                        GUC_PROPOSAL_REFINEMENT = P.GUC_DESIGN_PROPOSAL_ISEARCH_REFINEMENT
                        GU_DESIGN_PROPOSAL_FINISH = P.GUC_PROPOSAL_REFINEMENT_FINISH
                    proposal_prompt=GUC_PROPOSAL_REFINEMENT(REVIEW=review,RATING=rating,SUGGESTIONS=suggestions,
                                                            PASS_OR_NOT='Pass' if rating>=4 else 'Fail')
                    GUC_PROPOSAL_REFINEMENT.apply(DESIGN_PROPOSER.obj)
                else:
                    if UNSTRUCT_PROPOSER:
                        GUS_PROPOSAL_REFINEMENT=P.GUS_PROPOSAL_REFINEMENT_FINISH # reuse it
                    else:
                        GUS_PROPOSAL_REFINEMENT=P.GUS_DESIGN_PROPOSAL_ISEARCH_REFINEMENT
                        GU_DESIGN_PROPOSAL_FINISH=P.GUS_PROPOSAL_REFINEMENT_FINISH
                    proposal_prompt=GUS_PROPOSAL_REFINEMENT(REVIEW=review,RATING=rating,SUGGESTIONS=suggestions,
                                                            PASS_OR_NOT='Pass' if rating>=4 else 'Fail')
                    GUS_PROPOSAL_REFINEMENT.apply(DESIGN_PROPOSER.obj)
            
            ideation,instructions,search_report,search_references=None,None,None,None
            thoughts,keywords,description=None,None,None
            search_stack=[]
            if USE_ISEARCH or UNSTRUCT_PROPOSER:
                with self.status_handler(status_info):
                    self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_prompt)
                    _,out=self.call_dialog(design_proposer_tid,proposal_prompt)
                    context_design_proposer=self.dialog.context(design_proposer_tid) # past + begin_prompt + first query
                    if UNSTRUCT_PROPOSER:
                        thoughts,keywords,description=out['text'],out['keywords'],out['description']
                        ready="I'm ready" in thoughts
                        self.stream.markdown(thoughts,unsafe_allow_html=True)
                    else:
                        analysis,keywords,detail,ready,reflection=out['analysis'],out['keywords'],out['detail'],out['ready'],out.get('reflection',None)
                        if reflection:
                            self.stream.write(f'# Reflection\n{reflection}')
                        self.stream.write(f'# Analysis\n{analysis}')
                        self.stream.write(f'# Query\n{keywords}')
                        self.stream.write(f'# Detail\n{detail}')
                        self.stream.write(f'# Ready\n{ready}')
                    self.print_raw_output(out,'DESIGN_PROPOSER')

                _MIN_ROUNDS=max(2-attempt,0)
                _MAX_ROUNDS=max(self.max_attemps['max_search_rounds'],_MIN_ROUNDS)
                for i in range(_MAX_ROUNDS):
                    # TODO: perplexity context maintainance
                    with self.status_handler(f'Searching... round {i+1}...'):
                        if UNSTRUCT_PROPOSER:
                            detail=thoughts if description==[] else '\n'.join(description)
                            self.log_fn(f'Searching for {keywords}...','PROPOSAL')
                            search_ret=self.sss(keywords,detail,instruct=thoughts)
                            self.log_fn(f'Search finished.','PROPOSAL')
                            analysis=thoughts
                            if not keywords:
                                search_ret+='\n\nWarning: No keywords detected, external search skipped, please wrap your keywords in a quoted block like this: ```keywords {{Your keywods}} ``` in your response next time.'
                            if description==[]:
                                search_ret+='\n\nWarning: No description detected, will use full response to search internal library, please wrap your description in a quoted block like this: ```description {{Your description}}``` in your response next time.'
                        else:
                            self.log_fn(f'Searching for {keywords}...','PROPOSAL')
                            search_ret=self.sss(keywords,detail,analysis=analysis)
                            self.log_fn(f'Search finished.','PROPOSAL')
                        search_stack.append({
                            'analysis':analysis,
                            'query':keywords,
                            'detail':detail,
                            'ready':ready,
                            'search_ret':search_ret,
                        })
                        PROPOSAL_ISEARCH_CONT=P.O1M_PROPOSAL_ISEARCH_CONT if UNSTRUCT_PROPOSER else P.GUM_DESIGN_PROPOSAL_ISEARCH_CONT
                        search_cont_prompt=PROPOSAL_ISEARCH_CONT(SEARCH_RESULTS=search_ret)
                        if i<=_MIN_ROUNDS:
                            search_cont_prompt+=f'\n\nNote: This is your {i+1}th set of search results, you are not allowed to propose without adaquate information, you need to propose with at least {_MIN_ROUNDS+1} sets of search results. And your first {_MIN_ROUNDS} readiness will not be accepted.'
                        
                        PROPOSAL_ISEARCH_CONT.apply(DESIGN_PROPOSER.obj)
                        self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,search_cont_prompt)
                        _,out=self.call_dialog(design_proposer_tid,search_cont_prompt,context_design_proposer)
                        context_design_proposer.append(out['text'],'assistant',{}) # append new note,ignore search results
                        
                        if UNSTRUCT_PROPOSER:
                            thoughts,keywords,description=out['text'],out['keywords'],out['description']
                            ready="I'm ready" in thoughts
                            self.stream.markdown(thoughts,unsafe_allow_html=True)
                        else:
                            analysis,keywords,detail,ready,reflection=out['analysis'],out['keywords'],out['detail'],out['ready'],out.get('reflection',None)
                            if reflection:
                                self.stream.write(f'# Reflection\n{reflection}')
                            self.stream.write(f'# Analysis\n{analysis}')
                            self.stream.write(f'# Query\n{keywords}')
                            self.stream.write(f'# Detail\n{detail}')
                            self.stream.write(f'# Ready\n{ready}')
                        self.print_raw_output(out,'DESIGN_PROPOSER')
                        self.stream.write('---')
                    if ready and i>=_MIN_ROUNDS: break # at least request 3 times
                
                # final search 
                
                variantname,changes,reflection,abstract,selection=None,None,None,None,None
                with self.status_handler('Finishing design proposal...'):
                    # final search, so at least two searches, first, and second in first ready
                    if UNSTRUCT_PROPOSER:
                        detail=thoughts if description==[] else '\n'.join(description)
                        self.log_fn(f'Searching for {keywords}...','PROPOSAL')
                        search_ret=self.sss(keywords,detail,instruct=thoughts)
                        self.log_fn(f'Search finished.','PROPOSAL')
                        analysis=thoughts
                        if not keywords:
                            search_ret+='\n\nWarning: No keywords detected, external search skipped, please wrap your keywords in a quoted block like this: ```keywords {{Your keywods}} ``` in your response next time.'
                        if description==[]:
                            search_ret+='\n\nWarning: No description detected, will use full response to search internal library, please wrap your description in a quoted block like this: ```description {{Your description}}``` in your response next time.'
                    else:
                        self.log_fn(f'Searching for {keywords}...','PROPOSAL')
                        search_ret=self.sss(keywords,detail,analysis=analysis)
                        self.log_fn(f'Search finished.','PROPOSAL')
                    search_stack.append({
                        'analysis':analysis,
                        'query':keywords,
                        'detail':detail,
                        'ready':ready,
                        'search_ret':search_ret,
                    })
                    if UNSTRUCT_PROPOSER:
                        if self.design_mode==DesignModes.MUTATION:
                            o1_finish_prompt=P.O1M_PROPOSAL_FINISH(SELECTIONS=SELECTIONS,SEARCH_RESULTS=search_ret)
                            PROPOSAL_FINISH=P.O1M_PROPOSAL_FINISH
                        elif self.design_mode==DesignModes.CROSSOVER:
                            o1_finish_prompt=P.O1C_PROPOSAL_FINISH(SEARCH_RESULTS=search_ret)
                            PROPOSAL_FINISH=P.O1C_PROPOSAL_FINISH
                        else:
                            o1_finish_prompt=P.O1S_PROPOSAL_FINISH(SEARCH_RESULTS=search_ret)
                            PROPOSAL_FINISH=P.O1S_PROPOSAL_FINISH
                        self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,o1_finish_prompt)
                        PROPOSAL_FINISH.apply(DESIGN_PROPOSER.obj)
                        _,out=self.call_dialog(design_proposer_tid,o1_finish_prompt,context_design_proposer) # all notes + final query
                        proposal,title,modelname,abstract=out['text'],out['title'],out['model_name'],out['abstract']
                        if self.design_mode==DesignModes.MUTATION:
                            selection=out['selection']
                            self.stream.write(f'### Selection: {selection}')
                        self.stream.markdown(f'### Abstract\n{abstract}')
                        self.stream.markdown(f'### Proposal\n{proposal}')
                        self.print_raw_output(out,'DESIGN_PROPOSER')
                        if self.design_mode==DesignModes.MUTATION:
                            context_design_proposer_bkup=copy.deepcopy(context_design_proposer)
                            for _ in range(5):
                                context_design_proposer=copy.deepcopy(context_design_proposer_bkup)
                                succeed,selection,RETRY_PROMPT=P.gen_SELECTION_DEBUG_prompt(selection,SELECTIONS)
                                if succeed:
                                    break
                                self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,RETRY_PROMPT())
                                RETRY_PROMPT.apply(DESIGN_PROPOSER.obj)
                                self.stream.write(f'Error in output, retry...') # TODO: very costly and wasteful, need to fix
                                _,out=self.call_dialog(design_proposer_tid,RETRY_PROMPT(),context_design_proposer)
                                selection=out['selection']
                                self.stream.write(f'##### Correcting selection: {selection}')
                                self.print_raw_output(out,'DESIGN_PROPOSER')
                            if not succeed:
                                info = 'Failed to generate design proposal with right format with O1, stopping design process'
                                self.log_fn(info,'ERROR')
                                raise Exception(info)
                            context_design_proposer=context_design_proposer_bkup
                        if len(modelname)>0:
                            modelname=modelname[0]
                        elif len(title)>0:
                            modelname=title[0]
                        else:
                            modelname=f'An Improved {selection}'
                        self.stream.write(f'### Design Name: {modelname}')
                    else:
                        proposal_finish_prompt=GU_DESIGN_PROPOSAL_FINISH(SEARCH_RESULTS=search_ret)
                        self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_finish_prompt)
                        GU_DESIGN_PROPOSAL_FINISH.apply(DESIGN_PROPOSER.obj)
                        _,out=self.call_dialog(design_proposer_tid,proposal_finish_prompt,context_design_proposer)
                        proposal,modelname,changes,abstract=out['proposal'],out['modelname'],out.get('changes',None),out.get('abstract',None)
                        self.stream.write(f'### Design Name: {modelname}')
                        if self.design_mode==DesignModes.MUTATION:
                            selection,variantname=out['selection'],out['variantname']
                            self.stream.write(f'### Selection: {selection}')
                            self.stream.write(f'### Variant Name: {variantname}')
                        self.stream.write(f'### Abstract\n{abstract}')
                        self.stream.write(f'# Proposal\n{proposal}')
                        if changes:
                            self.stream.write(f'# Changes\n{changes}')
                        self.print_raw_output(out,'DESIGN_PROPOSER')

            elif USE_2STAGE: # use search or not
                with self.status_handler(status_info):
                    self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_prompt)
                    _,out=self.call_dialog(design_proposer_tid,proposal_prompt)
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
                    _,out=self.call_dialog(design_proposer_tid,proposal_prompt_stage2)
                    selection,proposal,modelname,variantname,changes,abstract=out['selection'],out['proposal'],out['modelname'],out['variantname'],out.get('changes',None),out.get('abstract',None)
                    self.stream.write(f'### Design Name: {modelname}')
                    self.stream.write(f'### Selection: {selection}')
                    self.stream.write(f'### Abstract\n{abstract}')
                    self.stream.write(f'# Proposal\n{proposal}')
                    if changes:
                        self.stream.write(f'# Changes\n{changes}')
                    context_design_proposer=self.dialog.context(design_proposer_tid)
                    self.print_raw_output(out,'DESIGN_PROPOSER')
            else:
                with self.status_handler(status_info):
                    self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_prompt)
                    _,out=self.call_dialog(design_proposer_tid,proposal_prompt)
                    selection,proposal,modelname,variantname,abstract=out['selection'],out['proposal'],out['modelname'],out['variantname'],out.get('abstract',None)
                    self.stream.write(f'### Design Name: {modelname}')
                    self.stream.write(f'### Selection: {selection}')
                    self.stream.write(f'### Variant Name: {variantname}')
                    reflection,changes=out.get('reflection',None),out.get('changes',None)
                    self.stream.write(f'### Abstract\n{abstract}')
                    self.stream.write(f'# Proposal\n{proposal}')
                    if reflection:
                        self.stream.write(f'# Reflection\n{reflection}')
                    if changes:
                        self.stream.write(f'# Changes\n{changes}')
                    context_design_proposer=self.dialog.context(design_proposer_tid)
                    self.print_raw_output(out,'DESIGN_PROPOSER')


            ### Review
            USE_ISEARCH_REVIEW=self.search_settings['proposal_review_search']
            self.log_fn(f'Searching for similar designs...','PROPOSAL')
            _,top_k_pps=self.sss.query_design_proposals(proposal)
            self.log_fn(f'Search finished.','PROPOSAL')
            _proposal=f'Abstract: {abstract}\n\n{proposal}' if abstract else proposal
            # self.stream.markdown(top_k_pps)

            if UNSTRUCT_REVIEWER:
                # o1_review_context=AgentContext()
                if self.design_mode==DesignModes.MUTATION:
                    SYSTEM_PROMPT=P.O1M_PROPOSAL_REVIEWER_BACKGROUND(
                        SEED=query,SELECTION=selection,PROPOSAL=_proposal,TOP_K_PPS=top_k_pps,SIBLINGS=SIBLINGS)
                elif self.design_mode==DesignModes.CROSSOVER:
                    SYSTEM_PROMPT=P.O1C_PROPOSAL_REVIEWER_BACKGROUND(SEED=query,PROPOSAL=_proposal,TOP_K_PPS=top_k_pps,SIBLINGS=SIBLINGS)
                else:
                    SYSTEM_PROMPT=P.O1S_PROPOSAL_REVIEWER_BACKGROUND(PROPOSAL=_proposal,TOP_K_PPS=top_k_pps)
            else:
                if self.design_mode==DesignModes.MUTATION:
                    SYSTEM_PROMPT=P.GUM_PROPOSAL_REVIEWER_SYSTEM() if not USE_ISEARCH_REVIEW else P.GUM_PROPOSAL_REVIEWER_SEARCH_SYSTEM()
                elif self.design_mode==DesignModes.CROSSOVER:
                    SYSTEM_PROMPT=P.GUC_PROPOSAL_REVIEWER_SYSTEM(SEED=query)
                else:
                    SYSTEM_PROMPT=P.GUS_PROPOSAL_REVIEWER_SYSTEM()
            PROPOSAL_REVIEWER=reload_role('proposal_reviewer',self.agents['PROPOSAL_REVIEWER'],SYSTEM_PROMPT)
            context_proposal_reviewer=copy.deepcopy(cross_attempt_context.truncate(1))
            proposal_reviewer_tid=self.dialog.fork(main_tid,USER_CALLER,PROPOSAL_REVIEWER,context=context_proposal_reviewer,
                                                alias='proposal_reviewer',note=f'Reviewing proposal...')
            if attempt==0:
                status_info=f'Reviewing initial proposal...'
                if UNSTRUCT_REVIEWER:
                    REVIEW_PROMPT=P.O1M_PROPOSAL_REVIEW
                    proposal_review_prompt=REVIEW_PROMPT()
                else:
                    if self.design_mode==DesignModes.MUTATION:
                        REVIEW_PROMPT=P.GUM_PROPOSAL_REVIEW_ISEARCH_BEGIN if USE_ISEARCH_REVIEW else P.GUM_PROPOSAL_REVIEW
                        proposal_review_prompt=REVIEW_PROMPT(
                            SEED=query,SELECTION=selection,PROPOSAL=_proposal,TOP_K_PPS=top_k_pps,SIBLINGS=SIBLINGS)
                    elif self.design_mode==DesignModes.CROSSOVER:
                        REVIEW_PROMPT=P.GUC_PROPOSAL_REVIEW_ISEARCH_BEGIN
                        proposal_review_prompt=REVIEW_PROMPT(
                            SEED=query,PROPOSAL=_proposal,TOP_K_PPS=top_k_pps,SIBLINGS=SIBLINGS)
                    else:
                        REVIEW_PROMPT=P.GUS_PROPOSAL_REVIEW_ISEARCH_BEGIN
                        proposal_review_prompt=REVIEW_PROMPT(PROPOSAL=_proposal,TOP_K_PPS=top_k_pps)
                REVIEW_PROMPT.apply(PROPOSAL_REVIEWER.obj)
            else:
                status_info=f'Reviewing refined proposal (version {attempt})...'
                if UNSTRUCT_REVIEWER:
                    REREVIEW_PROMPT=P.O1M_PROPOSAL_REVIEW # context free
                    proposal_review_prompt=REREVIEW_PROMPT()
                else:
                    if self.design_mode==DesignModes.MUTATION:
                        REREVIEW_PROMPT=P.GUM_PROPOSAL_REREVIEW_ISEARCH if USE_ISEARCH_REVIEW else P.GUM_PROPOSAL_REREVIEW
                    elif self.design_mode==DesignModes.CROSSOVER:
                        REREVIEW_PROMPT=P.GUC_PROPOSAL_REREVIEW_ISEARCH if USE_ISEARCH_REVIEW else P.GUC_PROPOSAL_REREVIEW
                    else:
                        REREVIEW_PROMPT=P.GUS_PROPOSAL_REREVIEW_ISEARCH if USE_ISEARCH_REVIEW else P.GUS_PROPOSAL_REREVIEW
                    if not changes:
                        changes='Please refer to the proposal for changes.'
                    if self.design_mode==DesignModes.MUTATION:
                        proposal_review_prompt=REREVIEW_PROMPT(
                            SELECTION=selection,PROPOSAL=_proposal,CHANGES=changes,TOP_K_PPS=top_k_pps,SIBLINGS=SIBLINGS)
                    elif self.design_mode==DesignModes.CROSSOVER:
                        proposal_review_prompt=REREVIEW_PROMPT(
                            PROPOSAL=_proposal,CHANGES=changes,TOP_K_PPS=top_k_pps,SIBLINGS=SIBLINGS)
                    else:
                        proposal_review_prompt=REREVIEW_PROMPT(
                            PROPOSAL=_proposal,CHANGES=changes,TOP_K_PPS=top_k_pps)
                REVIEW_PROMPT.apply(PROPOSAL_REVIEWER.obj)

            review_search_stack=[]
            if USE_ISEARCH_REVIEW or UNSTRUCT_REVIEWER:
                with self.status_handler(status_info):
                    self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,proposal_review_prompt)
                    _,out=self.call_dialog(proposal_reviewer_tid,proposal_review_prompt)
                    context_proposal_reviewer=self.dialog.context(proposal_reviewer_tid)
                    if UNSTRUCT_REVIEWER:
                        thoughts,keywords,description=out['text'],out['keywords'],out['description']
                        ready="I'm ready" in thoughts
                        self.stream.markdown(thoughts,unsafe_allow_html=True)
                    else:
                        analysis,keywords,detail,ready=out['analysis'],out['keywords'],out['detail'],out['ready']
                        self.stream.write(f'### Analysis\n{analysis}')
                        self.stream.write(f'### Query\n{keywords}')
                        self.stream.write(f'### Detail\n{detail}')
                        self.stream.write(f'### Ready\n{ready}')
                    self.print_raw_output(out,'PROPOSAL_REVIEWER')
                
                _MIN_ROUNDS=2
                _MAX_ROUNDS=max(self.max_attemps['max_search_rounds'],_MIN_ROUNDS)
                for i in range(_MAX_ROUNDS):
                    with self.status_handler(f'Searching... round {i+1}...'):
                        if UNSTRUCT_REVIEWER:
                            detail=thoughts if description==[] else '\n'.join(description)
                            self.log_fn(f'Searching for {keywords}...','PROPOSAL')
                            search_ret=self.sss(keywords,detail,instruct=thoughts)
                            self.log_fn(f'Search finished.','PROPOSAL')
                            analysis=thoughts
                            if not keywords:
                                search_ret+='\n\nWarning: No keywords detected, external search skipped, please wrap your keywords in a quoted block like this: ```keywords {{Your keywods}} ``` in your response next time.'
                            if description==[]:
                                search_ret+='\n\nWarning: No description detected, will use full response to search internal library, please wrap your description in a quoted block like this: ```description {{Your description}}``` in your response next time.'
                        else:
                            self.log_fn(f'Searching for {keywords}...','PROPOSAL')
                            search_ret=self.sss(keywords,detail,analysis=analysis)
                            self.log_fn(f'Search finished.','PROPOSAL')
                        review_search_stack.append({
                            'analysis':analysis,
                            'query':keywords,
                            'detail':detail,
                            'ready':ready,
                            'search_ret':search_ret,
                        })
                        PROPOSAL_REVIEW_ISEARCH_CONT=P.GUM_PROPOSAL_REVIEW_ISEARCH_CONT if not UNSTRUCT_REVIEWER else P.O1M_PROPOSAL_REVIEW_CONT
                        search_cont_prompt=PROPOSAL_REVIEW_ISEARCH_CONT(SEARCH_RESULTS=search_ret)
                        if i<=_MIN_ROUNDS:
                            search_cont_prompt+=f'\n\nNote: This is your {i+1}th set of search results, you are not allowed to propose without adaquate information, you need to propose with at least {_MIN_ROUNDS+1} sets of search results. And your first {_MIN_ROUNDS} readiness will not be accepted.'
                        PROPOSAL_REVIEW_ISEARCH_CONT.apply(PROPOSAL_REVIEWER.obj)
                        self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,search_cont_prompt)
                        _,out=self.call_dialog(proposal_reviewer_tid,search_cont_prompt,context_proposal_reviewer)
                        context_proposal_reviewer.append(out['text'],'assistant',{})
                        if UNSTRUCT_REVIEWER:
                            thoughts,keywords,description=out['text'],out['keywords'],out['description']
                            ready="I'm ready" in thoughts
                            self.stream.markdown(thoughts,unsafe_allow_html=True)
                        else:
                            analysis,keywords,detail,ready=out['analysis'],out['keywords'],out['detail'],out['ready']
                            self.stream.write(f'### Analysis\n{analysis}')
                            self.stream.write(f'### Query\n{keywords}')
                            self.stream.write(f'### Detail\n{detail}')
                            self.stream.write(f'### Ready\n{ready}')
                        self.print_raw_output(out,'PROPOSAL_REVIEWER')
                        self.stream.write('---')
                    if ready and i>=_MIN_ROUNDS: break # at least request 3 times
                
                suggestions=None
                with self.status_handler('Finishing proposal review...'):
                    if UNSTRUCT_REVIEWER:
                        detail=thoughts if description==[] else '\n'.join(description)
                        self.log_fn(f'Searching for {keywords}...','PROPOSAL')
                        search_ret=self.sss(keywords,detail,instruct=thoughts)
                        self.log_fn(f'Search finished.','PROPOSAL')
                        analysis=thoughts
                        if not keywords:
                            search_ret+='\n\nWarning: No keywords detected, external search skipped, please wrap your keywords in a quoted block like this: ```keywords {{Your keywods}} ``` in your response next time.'
                        if description==[]:
                            search_ret+='\n\nWarning: No description detected, will use full response to search internal library, please wrap your description in a quoted block like this: ```description {{Your description}}``` in your response next time.'
                    else:
                        self.log_fn(f'Searching for {keywords}...','PROPOSAL')
                        search_ret=self.sss(keywords,detail,analysis=analysis)
                        self.log_fn(f'Search finished.','PROPOSAL')
                    search_stack.append({
                        'analysis':analysis,
                        'query':keywords,
                        'detail':detail,
                        'ready':ready,
                        'search_ret':search_ret,
                    })
                    if UNSTRUCT_REVIEWER:
                        o1m_finish_prompt=P.O1M_PROPOSAL_REVIEW_FINISH(SEARCH_RESULTS=search_ret)
                        self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,o1m_finish_prompt)
                        P.O1M_PROPOSAL_REVIEW_FINISH.apply(PROPOSAL_REVIEWER.obj)
                        # o1_review_context.append(o1m_finish_prompt,'user',{})
                        _,out=self.call_dialog(proposal_reviewer_tid,o1m_finish_prompt,context_proposal_reviewer)
                        review,rating=out['text'],out['rating']
                        self.stream.write(review)
                        self.print_raw_output(out,'PROPOSAL_REVIEWER')
                        for _ in range(5):
                            succeed,rating,RETRY_PROMPT=P.gen_O1_RATING_DEBUG_prompt(rating)
                            if succeed:
                                break
                            self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,RETRY_PROMPT())
                            RETRY_PROMPT.apply(PROPOSAL_REVIEWER.obj)
                            self.stream.write(f'Error in output, retry...') # TODO: very costly and wasteful, need to fix
                            _,out=self.call_dialog(proposal_reviewer_tid,RETRY_PROMPT())
                            rating=out['rating']
                            self.stream.write(f'##### Correcting rating: {rating}')
                            self.print_raw_output(out,'PROPOSAL_REVIEWER')
                        if not succeed:
                            info = 'Failed to generate design proposal with right format with O1, stopping design process'
                            self.log_fn(info,'ERROR')
                            raise Exception(info)
                        passornot='Pass' if rating>=REVIEW_THRESHOLD else 'Fail'
                        self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                    else:
                        search_finish_prompt=P.GUM_PROPOSAL_REVIEW_ISEARCH_FINAL()
                        P.GUM_PROPOSAL_REVIEW_ISEARCH_FINAL.apply(PROPOSAL_REVIEWER.obj)
                        self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,search_finish_prompt)
                        _,out=self.call_dialog(proposal_reviewer_tid,search_finish_prompt,context_proposal_reviewer)
                        review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                        passornot='Pass' if rating>=REVIEW_THRESHOLD else 'Fail'
                        self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                        self.stream.write(review)
                        self.stream.write(suggestions)
                        self.print_raw_output(out,'PROPOSAL_REVIEWER')
            else:
                with self.status_handler(status_info):
                    self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,proposal_review_prompt)
                    _,out=self.call_dialog(proposal_reviewer_tid,proposal_review_prompt)
                    review,rating,suggestions=out['review'],out['rating'],out['suggestions']
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
                'abstract':abstract,
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
                self.stream.write(f'#### Proposal passed with rating {rating} out of 5')
                self.log_fn(f'Proposal passed with rating {rating} out of 5','PROPOSAL')
                break
            else:
                _cross_attempt_context=f"This is a past proposal attempt that failed the review with rating {rating} out of 5."
                if abstract:
                    _cross_attempt_context+=f"\n\n## Abstract\n{abstract}\n"
                _cross_attempt_context+=f"\n\n### Model Name\n{modelname}\n"    
                if selection:
                    _cross_attempt_context+=f"\n\n### Selection\n{selection}\n"
                if variantname:
                    _cross_attempt_context+=f"\n\n### Variant Name\n{variantname}\n"
                _cross_attempt_context+=f"\n\n## Proposal\n{proposal}\n"
                if reflection or changes:
                    _cross_attempt_context+='\n\n#### The proposal is a refinement of its previous proposal which failed the review.\n'
                if reflection:
                    _cross_attempt_context+=f"\n\n### The proposer's Reflection on its previous proposal\n{reflection}\n"
                if changes:
                    _cross_attempt_context+=f"\n\n### Changes made compared to its previous proposal \n{changes}\n"
                _cross_attempt_context+=f"\n\n## Review\n{review}\n"
                _cross_attempt_context+=f"\n\n## Suggestions\n{suggestions}\n"
                cross_attempt_context.append(_cross_attempt_context,'user',{})
        
        if rating<REVIEW_THRESHOLD:
            self.stream.write(f'#### Proposal failed with rating {rating} out of 5')
            self.log_fn(f'Proposal failed with rating {rating} out of 5','PROPOSAL')
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
        rerank=self.ptree.session_get(self.sess_id,'reranked')
        if not rerank:
            _proposals,_acronyms=self.ptree.session_proposals(self.sess_id,passed_only=True)
            proposals=[]
            acronyms=[]
            for i,acronym in enumerate(_acronyms):
                if acronym not in acronyms: # why it happens at all???
                    proposals.append(_proposals[i])
                    acronyms.append(acronym)
            rerank=self.rerank_proposals(proposals,acronyms)
            self.ptree.session_set(self.sess_id,'reranked',rerank)
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
        if self.running_mode==RunningModes.PROPOSAL_ONLY:
            self.stream.write('Proposal only mode, skipping implementation...')
            return query,state,{}
        elif self.running_mode==RunningModes.IMPLEMENTATION_ONLY:
            self.stream.write('Implementation only mode, will select from unimplemented passed proposals or using provided proposal if any')
        if proposal is None:
            self.log_fn('Reranking proposals to implement...','IMPLEMENTATION')
            proposals,acronyms=self.reranked_proposal()
            self.log_fn('Reranking finished.','IMPLEMENTATION')
        else:
            self.log_fn('Using provided proposal for implementation...','IMPLEMENTATION')
            proposals=[proposal]
            acronyms=[proposal.acronym]
        if len(proposals) == 0:
            self.stream.write('No remaining proposals to implement, stopping design process')
            end_reason = EndReasons.PROPOSAL_FAILED
            self.stream.log(end_reason,'end')
            return query,state,{}
        self.stream.write(f'Implementing {len(proposals)} proposals: {", ".join([f"{i+1}. {proposal.modelname} with rating {proposal.rating} out of 5" for i,proposal in enumerate(proposals)])}')
        for proposal,acronym in zip(proposals,acronyms):
            if not self.ptree.acquire_design_lock(self.sess_id): # For benchmark, active sessions + finished designs < max designs
                self.log_fn(f'Design lock not acquired, design is full, stopping implementation...','IMPLEMENTATION')
                break
            if self.ptree.is_challenging(acronym):
                self.stream.write(f'Design {acronym} is too challenging, retried for {self.ptree.challenging_threshold} times already, skipping implementation...')
                continue
            self.log_fn(f'Implementing proposal: {proposal.modelname} with rating {proposal.rating} out of 5','IMPLEMENTATION')
            self.stream.write(f'Implementing proposal: {proposal.modelname} with rating {proposal.rating} out of 5')
            tree_ckpt,status=self.ptree.get_implementation_checkpoint(acronym)
            initial_pass=False
            if tree_ckpt is None:
                self.tree=copy.deepcopy(self.seed_tree)
                self.tree.name=proposal.modelname
            else:
                self.tree=copy.deepcopy(tree_ckpt)
                if status in ['initial_pass','unfinished']:
                    initial_pass=True
                self.stream.write(f'Resuming implementation checkpoint of {acronym}...')

            cost_raw=copy.deepcopy(self.costs)
            RETS=self._implement_proposal_recursive(main_tid,proposal,acronym,resume=tree_ckpt is not None,initial_pass=initial_pass)
            costs={k:v-cost_raw[k] for k,v in self.costs.items()}
            ROUNDS,SUCCEED,INITIAL_PASS=RETS['ROUNDS'],RETS['SUCCEED'],RETS['INITIAL_PASS']
            if SUCCEED:
                status='implemented'
            else:
                status='initial_pass' if INITIAL_PASS else 'failed'
            self.log_fn(f'Adding implementation to tree, tuning and exporting full LM codes...','IMPLEMENTATION')
            with self.status_handler(f'Adding implementation to tree, tuning and exporting full LM codes...'):
                self.ptree.implement(acronym,self.tree,ROUNDS,status,costs,self.design_cfg,self.user_input)
        return query,state,{}
   
    def check_code_format(self,code,selection=None,spec=None,analysis=None,declaration=None):
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
                if child_decl.unitname not in self.tree.declares:# and child_decl.unitname not in self.tree.units: # only add new ones
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

        # TODO: the reformatter should rename the existing kwargs from the tree, okey for root node
        # TODO: remove any possible if __name__=='__main__' method from the code
        if selection is not None: # for non o1 mode
            raise NotImplementedError('Non o1 mode need to be updated')
            # if unit_name not in self.tree.units:
            #     if spec is not None:
            #         self.tree.add_unit(
            #             spec,reformatted_code,new_args,analysis,None,None,[child.unitname for child in children_decl],gau_tests,None,requirements=declaration.requirements
            #         )
            # else:
            #     self.tree.units[unit_name].code=reformatted_code
            #     self.tree.units[unit_name].args=new_args
            #     self.tree.units[unit_name].analysis=analysis
            #     self.tree.units[unit_name].children=[child.unitname for child in children_decl]
            #     self.tree.units[unit_name].gau_tests=gau_tests
            
        return format_checks,format_errors,format_warnings,fetal_errors,unit_name, reformatted_code, docstring, new_args, gau_tests, children_decl, NEW_DECLARED


    def recommend_reuses(self,unimplemented_units): # all unimplemented units should have declarations
        # return dict of dicts of units {unit_name: [unit_name, ...]}
        _items={}
        prt=''
        for unit in unimplemented_units:
            desc=self.tree.get_unit_desc(unit)
            if desc is not None:
                items,_=self.system.sss.query_units_by_desc(desc)
                if items:
                    prt+=f'#### Recommended Unit Reuses for unimplemented unit: {unit}\n\n'
                for unit_name in items:
                    code,desc,tree_name,score=items[unit_name]
                    _items[f'{tree_name}.{unit_name}']=code
                    prt+=f'##### {unit_name} from {tree_name} (Score: {score:.2f})\n\n<details><summary>Unit Implementation</summary>\n\n```\n{code}\n```\n\n</details>\n\n'
        if _items=={}:
            prt=None
        return _items,prt
    
    def reuse_parents(self,seed_ids):
        _items={}
        for seed_id in seed_ids:
            parent=self.ptree.get_node(seed_id)
            for unit in parent.units:
                _items[f'{seed_id}.{unit}']=parent.units[unit].code
        return _items

    def _implement_proposal_recursive(self,main_tid,proposal,acronym,resume=False,initial_pass=False):
        '''
        1. Implement the selected unit first
        2. Implement any unimplemented newly declared units
        3. Do post refinement, if new units defined, go to 2, post refinement count will not be refreshed
        '''

        self.dialog=self.system.dialog
        OBSERVE_THRESHOLD=self.threshold['implementation_rating']
        cost_raw=copy.deepcopy(self.costs)

        end_reason = None
        RETS={}
        RETS['ROUNDS']=[]
        SUCCEED=False
        LOG=[]
        INITIAL_PASS=initial_pass
        round=0
        # XXX: Protected units should be self.seed_tree.units.keys() - self.tree.units.keys(), but this make things simpler
        if self.design_mode==DesignModes.MUTATION:
            PROTECTED_UNITS=list(set(self.tree.units.keys())-set([proposal.selection])) # the units besides the current one, they should not be *modified*, can be removed as descendants
        else:
            PROTECTED_UNITS=[]
        self.stream.write(f'##### Protected Units: {PROTECTED_UNITS}')
        USE_PAIRING=self.agent_types['IMPLEMENTATION_OBSERVER']!='None'
        # o1 beta does not support structured outputs, so let it output the code directly
        # UNSTRUCT_CODER='o1' in self.agents['IMPLEMENTATION_CODER'].model_name
        # UNSTRUCT_OBSERVER='o1' in self.agents['IMPLEMENTATION_OBSERVER'].model_name
        # UNSTRUCT_PLANNER='o1' in self.agents['IMPLEMENTATION_PLANNER'].model_name

        UNSTRUCT_PLANNER=True # always use o1 planner prompts for now, i.e. no structured outputs
        UNSTRUCT_OBSERVER=True # always use o1 observer prompts for now, i.e. no structured outputs
        UNSTRUCT_CODER=True # always use o1 coder prompts for now, i.e. no structured outputs

        # NOTE: Better use all o1 here, others are not stable

        context_implementation_planner=AgentContext() 
        planner_context=AgentContext()

        post_refinement=0 # TODO: introduce self-evaluate to post-refinement
        while True:
            round+=1 # Each round works on one unit at a time, start counting from beginning so that we dont wrap it in wrong place
            if resume: # resume from a checkpoint, do not force selecting the selection
                round+=1
            traces=[]
            context_implementation_coder=AgentContext()
            context_implementation_observer=AgentContext()

            succeed=False
            if self.design_mode==DesignModes.MUTATION or INITIAL_PASS:
                IMPLEMENTED,UNIMPLEMENTED=self.tree.check_implemented()
                # GAB_CODE=self.tree.compose()
                UNIMPLEMENTED=list(set(UNIMPLEMENTED)-set(PROTECTED_UNITS)) # although its impossible to have unavailable units
                IMPLEMENTED=list(set(IMPLEMENTED)-set(PROTECTED_UNITS))
            else:
                UNIMPLEMENTED=['root']
                IMPLEMENTED=[]


            VIEW_DETAILED=self.tree.to_prompt(unit_code=True)

            self.stream.write(f'Round {round-1 if resume else round}. Unprotected Implemented: {IMPLEMENTED}, Unimplemented: {UNIMPLEMENTED}')

            if len(UNIMPLEMENTED)==0 and round>1: # round 1 is selected unit, naturally no unimplemented units, consume the post refinement count only if round>1
                post_refinement+=1

            # 1. design succeeded, post refinement count reached
            if INITIAL_PASS and post_refinement>self.max_attemps['post_refinement'] and len(UNIMPLEMENTED)==0:
                self.stream.write(f'#### All units have been implemented and maximal refinements are reached, stopping design process')
                SUCCEED=True
                end_reason = EndReasons.MAX_POST_REFINEMENT_REACHED
                break
            

            ################# SELECTING THE NEXT UNIT TO WORK ON #################
            
            context_implementation_planner=copy.deepcopy(planner_context)
            if self.design_mode==DesignModes.MUTATION: 
                GUT_IMPLEMENTATION_PLANNER_SYSTEM=P.O1M_IMPLEMENTATION_PLANNER_BACKGROUND
                _background_prompt={'SEED':self.seed,'SELECTION':proposal.selection}
            elif self.design_mode==DesignModes.CROSSOVER:
                GUT_IMPLEMENTATION_PLANNER_SYSTEM=P.O1C_IMPLEMENTATION_PLANNER_BACKGROUND
                _background_prompt={'PARENTS':self.seed}
            else:
                GUT_IMPLEMENTATION_PLANNER_SYSTEM=P.O1S_IMPLEMENTATION_PLANNER_BACKGROUND
                _background_prompt={}

            IMPLEMENTATION_PLANNER=reload_role('implementation_planner',self.agents['IMPLEMENTATION_PLANNER'],GUT_IMPLEMENTATION_PLANNER_SYSTEM(
                GAB_BASE=GAB_BASE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE,
                PROPOSAL=proposal.proposal,REVIEW=proposal.review,RATING=proposal.rating,**_background_prompt))
            
            implementation_planner_tid=self.dialog.fork(main_tid,USER_CALLER,IMPLEMENTATION_PLANNER,context=context_implementation_planner,
                                                alias='implementation_planner',note=f'Starting implementation planning...')
            

            if INITIAL_PASS:
                potential_reuses={}
                if self.design_mode==DesignModes.CROSSOVER:
                    parents_reuses=self.reuse_parents(self.seed_ids)
                    REUSE_PROMPT=P.gen_REUSE_PROMPT(self.design_mode,parents_reuses=parents_reuses)
                else: 
                    potential_reuses,reuse_prompt=self.recommend_reuses(UNIMPLEMENTED)
                    if self.design_mode==DesignModes.MUTATION:
                        parents_reuses=self.reuse_parents(self.seed_ids)
                        REUSE_PROMPT=P.gen_REUSE_PROMPT(self.design_mode,parents_reuses=parents_reuses,potential_reuses=potential_reuses,reuse_prompt=reuse_prompt)
                    else:
                        REUSE_PROMPT=P.gen_REUSE_PROMPT(self.design_mode,potential_reuses=potential_reuses,reuse_prompt=reuse_prompt)


            if round>1 or UNSTRUCT_PLANNER: # if round > 1, let the agent choose the next unit to work on, TODO: maybe more background about previous rounds
                with self.status_handler('Planning for the next round...'):
                    SELECTIONS=set(IMPLEMENTED+UNIMPLEMENTED)#-{self.tree.root}
                    SKIP_PLANNING=False
                    if UNSTRUCT_PLANNER:
                        if round==1: # working on the selected unit, no reuse
                            if self.design_mode==DesignModes.MUTATION:
                                GUM_IMPLEMENTATION_UNIT_SELECTION=P.O1_IMPLEMENTATION_PLANNER_BEGIN_MUTATION
                            elif self.design_mode==DesignModes.CROSSOVER:
                                GUM_IMPLEMENTATION_UNIT_SELECTION=P.O1_IMPLEMENTATION_PLANNER_BEGIN_CROSSOVER
                            else:
                                GUM_IMPLEMENTATION_UNIT_SELECTION=P.O1_IMPLEMENTATION_PLANNER_BEGIN_SCRATCH
                            gu_implementation_unit_selection_prompt=GUM_IMPLEMENTATION_UNIT_SELECTION(
                                VIEW=VIEW_DETAILED
                            )
                        elif not INITIAL_PASS:
                            SKIP_PLANNING=True
                        elif len(UNIMPLEMENTED)==0:
                            GUM_IMPLEMENTATION_UNIT_SELECTION=P.O1_IMPLEMENTATION_PLANNER_POST_REFINE
                            gu_implementation_unit_selection_prompt=GUM_IMPLEMENTATION_UNIT_SELECTION(
                                ROUND=round,VIEW=VIEW_DETAILED,SELECTIONS=SELECTIONS,
                                PROTECTED=PROTECTED_UNITS,LOG='\n'.join(LOG)
                            )
                        else: # SELECTION of unimplemented units
                            GUM_IMPLEMENTATION_UNIT_SELECTION=P.O1_IMPLEMENTATION_PLANNER_SELECTION
                            gu_implementation_unit_selection_prompt=GUM_IMPLEMENTATION_UNIT_SELECTION(
                                ROUND=round,VIEW=VIEW_DETAILED,SELECTIONS=SELECTIONS,LOG='\n'.join(LOG),
                                IMPLEMENTED=IMPLEMENTED,UNIMPLEMENTED=UNIMPLEMENTED,PROTECTED=PROTECTED_UNITS,
                                REUSE_PROMPT=REUSE_PROMPT
                            )
                    else:
                        raise NotImplementedError('Structural prompts need to be updated.')
                        # GUM_IMPLEMENTATION_UNIT_SELECTION=P.gen_GUM_IMPLEMENTATION_UNIT_SELECTION(
                        #     SELECTIONS,post_refining=len(UNIMPLEMENTED)==0)
                        # gu_implementation_unit_selection_prompt=GUM_IMPLEMENTATION_UNIT_SELECTION(
                        #     VIEW=VIEW_DETAILED,LOG='\n'.join(LOG),ROUND=round
                        # )

                    if not SKIP_PLANNING:
                        GUM_IMPLEMENTATION_UNIT_SELECTION.apply(IMPLEMENTATION_PLANNER.obj)
                        self.print_details(IMPLEMENTATION_PLANNER.obj,context_implementation_planner,gu_implementation_unit_selection_prompt)
                        # self.stream.write(f'{VIEW_DETAILED}\n\nNow selecting the next unit to work on...')
                        planner_context.append(gu_implementation_unit_selection_prompt,'user',{})
                        _,out=self.call_dialog(implementation_planner_tid,gu_implementation_unit_selection_prompt)
                    motivation,rough_plan,reuse_code,reuse_from,reuse_type=None,None,None,None,None
                    termination=False
                    if UNSTRUCT_PLANNER:
                        if not SKIP_PLANNING: # reuse the results last round, only happen when failed in the begining
                            if round==1:
                                plan=out['text']
                                if self.design_mode==DesignModes.MUTATION:
                                    selection=proposal.selection
                                else:
                                    selection='root'
                            else:
                                plan,selection,reuse_unit=out['text'],out['selection'],out['reuse']
                                for _reuse in reuse_unit:
                                    if _reuse in parents_reuses:
                                        reuse_code = parents_reuses[_reuse]
                                        reuse_from = _reuse
                                        reuse_type = 'parents'
                                        break
                                    elif _reuse in potential_reuses:
                                        reuse_code = potential_reuses[_reuse]
                                        reuse_from = _reuse
                                        reuse_type = 'recommended'
                                        break

                            planner_context.append(plan,'assistant',{})
                            if round>1 and len(UNIMPLEMENTED)==0:
                                termination="```terminate```" in plan
                            self.stream.write(f'### Selection: {selection}')
                            self.stream.write(f'### Plan\n{plan}')
                            self.print_raw_output(out,'IMPLEMENTATION_PLANNER')
                            if not termination and round>1:
                                succeed,selection,_=P.gen_O1_SELECTION_DEBUG_prompt(selection,SELECTIONS)
                                if not succeed:
                                    if len(UNIMPLEMENTED)==0:
                                        termination=True
                                    else:
                                        selection = random.choice(UNIMPLEMENTED)
                                        plan+=f'\n\n### NOTE: The selection does not successfully parsed, randomly select {selection} from {UNIMPLEMENTED} instead. You may ignore the plan if not useful.'
                        else:
                            if self.design_mode==DesignModes.MUTATION:
                                selection=proposal.selection
                            else:
                                selection='root'
                            termination=False
                            plan='Not available yet.'
                    else:
                        raise NotImplementedError('Structural prompts need to be updated.')
                        # selection,motivation,rough_plan,termination=out['selection'],out['motivation'],out['rough_plan'],out['termination']
                        # plan=out['text']
                        # self.stream.write(f'### Selection: {selection}')
                        # self.stream.write(f'### Motivation\n{motivation}')    
                        # self.stream.write(f'### Rough Plan\n{rough_plan}')
                        # self.print_raw_output(out,'IMPLEMENTATION_PLANNER')
                        # planner_context.append(plan,'assistant',{})
            else: # round 1, work on the selected unit
                raise NotImplementedError('Structural prompts need to be updated.')
                # if self.design_mode==DesignModes.MUTATION:
                #     selection=proposal.selection
                # else:
                #     selection='root'
                # termination=False
                # plan='Not available for the first round.'
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
                end_reason = EndReasons.AGENT_TERMINATION
                break

            # 3. design failed, force stop conditions
            if (self.termination['max_total_budget']>0 and self.total_cost>self.termination['max_total_budget']):
                self.stream.write(f'#### Max total budget reached, stopping design process')
                end_reason = EndReasons.MAX_TOTAL_BUDGET_REACHED
                break
            if (self.termination['max_debug_budget']>0 and self.implementation_cost>self.termination['max_debug_budget']):
                self.stream.write(f'#### Max debug budget reached, stopping design process')
                end_reason = EndReasons.MAX_DEBUG_BUDGET_REACHED
                break
            if (self.termination['max_failed_rounds']>0 and self.failed_rounds>self.termination['max_failed_rounds']):
                self.stream.write(f'#### Max failed rounds reached, stopping design process')
                end_reason = EndReasons.MAX_FAILED_ROUNDS_REACHED
                break

            ################# UNIT IMPLEMENTATION INNER LOOP #################

            tree_backup=copy.deepcopy(self.tree) # backup the tree for rollback
            for attempt in range(self.max_attemps['implementation_debug']):
                GUT_IMPLEMENTATION_CODER_SYSTEM=P.gen_GUT_IMPLEMENTATION_CODER_SYSTEM(
                    unstruct=UNSTRUCT_CODER,mode=self.design_mode,INITAL_PASS=INITIAL_PASS)
                if self.design_mode==DesignModes.MUTATION:
                    _background_prompt={'SELECTION':proposal.selection,'SEED':self.seed}
                elif self.design_mode==DesignModes.CROSSOVER:
                    _background_prompt={'PARENTS':self.seed}
                else:
                    _background_prompt={}
                IMPLEMENTATION_CODER=reload_role('implementation_coder',self.agents['IMPLEMENTATION_CODER'],GUT_IMPLEMENTATION_CODER_SYSTEM(
                    GAB_BASE=GAB_BASE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE,PLAN=plan,
                    PROPOSAL=proposal.proposal,REVIEW=proposal.review,RATING=proposal.rating,**_background_prompt))
                implementation_coder_tid=self.dialog.fork(main_tid,USER_CALLER,IMPLEMENTATION_CODER,context=context_implementation_coder, # keep last 2 messages and system message
                                                    alias='implementation_coder',note=f'Starting design implementation...')
                if attempt==0: # first attempt, implement the unit
                    status_info=f'Starting design implementation of {selection}...'
                    if selection in IMPLEMENTED:
                        REFINE=True 
                        GUM_IMPLEMENTATION_UNIT=P.gen_GUT_IMPLEMENTATION_UNIT(refine=True,unstruct=UNSTRUCT_CODER) # first round can only be an implemented unit
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
                        REUSE_UNIT_PROMPT = P.gen_REUSE_UNIT_PROMPT(reuse_code,reuse_from,reuse_type,self.design_mode)
                        GUM_IMPLEMENTATION_UNIT=P.gen_GUT_IMPLEMENTATION_UNIT(refine=False,unstruct=UNSTRUCT_CODER)
                        declaration=self.tree.declares[selection].to_prompt()
                        gu_implement_unit_prompt=GUM_IMPLEMENTATION_UNIT(
                            DECLARATION=declaration,
                            VIEW=VIEW_DETAILED, 
                            REUSE_UNIT_PROMPT=REUSE_UNIT_PROMPT
                        )
                    GUM_IMPLEMENTATION_UNIT.apply(IMPLEMENTATION_CODER.obj)
                else: # Debugging or refining the implementation
                    status_info=f'Refining design implementation of {selection} (attempt {attempt})...'
                    REFINE=True
                    RETRY_RPOMPT=P.gen_GU_IMPLEMENTATION_UNIT_RETRY(unstruct=UNSTRUCT_CODER)
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
                        REUSE_UNIT_PROMPT=REUSE_UNIT_PROMPT
                        # GAU_BASE=GAU_BASE
                    )
                    RETRY_RPOMPT.apply(IMPLEMENTATION_CODER.obj)


                with self.status_handler(status_info): # calling the agent
                    context_implementation_coder=context_implementation_coder.truncate(4)
                    self.print_details(IMPLEMENTATION_CODER.obj,context_implementation_coder,gu_implement_unit_prompt)
                    _,out=self.call_dialog(implementation_coder_tid,gu_implement_unit_prompt,context_implementation_coder)
                    context_implementation_coder=self.dialog.context(implementation_coder_tid)
                    reflection,changes,debugging_steps=None,None,None
                    if REFINE: # 1. working on an existing unit 2. all >0 attempts
                        if UNSTRUCT_CODER:
                            reflection,analysis=None,None
                            codes=out['code']
                            implementations=[]
                            for code in codes:
                                if code.strip().startswith('# GAU_IMPLEMENTATION_FILE'):
                                    implementations.append(code)
                            changes="The coder didn't provide the summary of changes."
                        else:
                            raise NotImplementedError('Structural prompts need to be updated.')
                            # reflection,analysis,implementation,changes=out['reflection'],out['analysis'],out['implementation'],out['changes']
                            # self.stream.write(f'### Reflection\n{reflection}')
                            # self.stream.write(f'## Refinement of {selection}')
                            # if 'debugging_steps' in out:
                            #     debugging_steps=out['debugging_steps']
                            #     if isinstance(debugging_steps,list):
                            #         self.stream.write(f'### Debugging Steps\n')
                            #         for idx,step in enumerate(debugging_steps):
                            #             self.stream.write(f'##### Diagnosis {idx+1}\n{step["diagnosis"]}\n')
                            #             self.stream.write(f'##### Suggested Action {idx+1}\n{step["suggested_action"]}\n')
                            #     else:
                            #         self.stream.write(f'### Debugging Steps\n{debugging_steps}')
                        if selection in IMPLEMENTED: # update the unit spec for now, unit name update at the end
                            spec=self.tree.units[selection].spec
                        else: # possible when debugging the implementation of a new unit
                            if isinstance(declaration,str):
                                declaration = UnitDecl(
                                    unitname=str(selection),
                                    requirements='N/A',
                                    inputs=['N/A'],
                                    outputs=['N/A']
                                )
                            spec = P.UnitSpec(
                                unitname=str(selection),
                                document='',
                                inputs=declaration.inputs,
                                outputs=declaration.outputs
                            )
                    else: # only for the first attempt of a new unit, the reason we do this is that the response format is different for this case
                        if UNSTRUCT_CODER:
                            codes,analysis=out['code'],None
                            implementations=[]
                            for code in codes:
                                if code.strip().startswith('# GAU_IMPLEMENTATION_FILE'):
                                    implementations.append(code)
                        else:
                            raise NotImplementedError('Structural prompts need to be updated.')
                            # implementation,analysis=out['implementation'],out['analysis']
                        self.stream.write(f'## Implementation of {selection}')
                        if isinstance(declaration,str):
                            declaration = UnitDecl(
                                unitname=str(selection),
                                requirements='N/A',
                                inputs=['N/A'],
                                outputs=['N/A']
                            )
                        spec = P.UnitSpec(
                            unitname=str(selection),
                            document='',
                            inputs=declaration.inputs,
                            outputs=declaration.outputs
                        )                    
                    if analysis:
                        self.stream.write(analysis)
                    if UNSTRUCT_CODER:
                        for idx,implementation in enumerate(implementations):
                            self.stream.write(f'### Code {idx+1}\n```python\n{implementation}\n```')
                    else:
                        raise NotImplementedError('Structural prompts need to be updated.')
                        # self.stream.write(f'### Code\n```python\n{implementation}\n```')
                    if REFINE and changes:
                        self.stream.write(f'### Changes\n{changes}')
                    
                    self.print_raw_output(out,'IMPLEMENTATION_CODER')


                # Run all checks for every implementations, optimize both grammar and semantics at the same time 
                # avoid redundant debugging steps, i.e. only the debug for the passed plans are needed
                with self.status_handler('Checking the implementation of the selected unit...'):
                    if UNSTRUCT_CODER:
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
                                self.tree.declares[c.unitname]=c
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
                                fetal_errors.append(f'There are multiple root units found: {", ".join(root_name)}, please check if you miss declare some child units.')
                            else:
                                unit_name=root_name[0]
                                if self.design_mode in [DesignModes.SCRATCH,DesignModes.CROSSOVER] and not INITIAL_PASS:
                                    self.tree.root = unit_name
                                    if 'root' in unit_name.lower() or 'gau' in unit_name.lower():
                                        format_warnings.append(f'Detected words like "root" or "gau" in the root unit name, which seems not a meaningful name, please use the model *block* name in the proposal as the root unit name.')
                                
                                reformatted_code=f'### {unit_name} Reformatted Code\n```python\n{reformatted_code}\n```\n\n'
                                # replace selection with unit_name
                                self.tree._replace_unit(selection,unit_name) # do replace first
                                LOG.append(f'Replace unit {selection} with {unit_name}')
                                self.stream.write(f'Replace local root unit {selection} with {unit_name}')
                                if unit_name not in self.tree.declares:
                                    self.tree.declares[unit_name]=UnitDecl(
                                        unitname=str(unit_name),
                                        requirements='N/A',
                                        inputs=spec.inputs,
                                        outputs=spec.outputs
                                    )
                        for unit_name,docstring in docstrings.items():
                            if unit_name in self.tree.units: # the unit is already implemented
                                format_errors.append(f'The unit {unit_name} has already been implemented. Please do not implement the same unit twice. If the existing unit can already meet your needs, please reuse it. If you are going to modify the existing unit, please provide a new name for the unit.')
                            if unit_name not in self.tree.declares:# and unit_name not in self.tree.units: # if it is a new local root, then it is declared above, otherwise, it should be declared as a child
                                format_errors.append(f'A new implemented unit {unit_name} has not been declared. May cause errors when linking the units.')
                                declaration=UnitDecl( # make a temoral declaration placeholder for checking
                                    unitname=str(unit_name),
                                    requirements='N/A',
                                    inputs=['N/A'],
                                    outputs=['N/A']
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
                                _spec,reformatted_codes[unit_name],new_args[unit_name],None,None,None,_children,gau_tests[unit_name],None,
                                requirements=declaration.requirements, overwrite=True, reuse_from=reuse_from
                            )   
                        unit_name=root_name[0] if len(root_name)>0 else selection
                        
                    else:   
                        raise NotImplementedError('Structural prompts need to be updated.')
                        # unit_name=selection
                        # if unit_name not in self.tree.declares:
                        #     if unit_name not in self.tree.units: # if it is a new local root, then it is declared above, otherwise, it should be declared as a child
                        #         format_warnings.append(f'A new implemented unit {unit_name} has not been declared. May cause errors when linking the units.')
                        #         declaration=UnitDecl(unitname=str(unit_name),requirements='N/A',inputs=['N/A'],outputs=['N/A'])
                        #     else:
                        #         spec=self.tree.units[unit_name].spec
                        #         declaration=UnitDecl(unitname=str(unit_name),requirements=spec.document,inputs=spec.inputs,outputs=spec.outputs)
                        # else:
                        #     declaration=self.tree.declares[unit_name]
                        # format_checks,format_errors,format_warnings,fetal_errors, unit_name, reformatted_code, docstring,new_args,gau_tests,children_decl,NEW_DECLARED=self.check_code_format(implementation,selection,spec,analysis,declaration)
                        # reformatted_code=f'### {unit_name} Reformatted Code\n```python\n{reformatted_code}\n```\n\n'
                        # if self.design_mode in [DesignModes.SCRATCH,DesignModes.CROSSOVER] and not INITIAL_PASS:
                        #     self.tree.root = unit_name
                        
                        # collapse_write(
                        #     self.stream,
                        #     'Code format check for '+unit_name,
                        #     (
                        #         f'\n\n#### Format Check Passed: {len(format_errors+fetal_errors)==0}\n\n'
                        #         f'#### Format Errors\n{format_errors+fetal_errors}\n\n'
                        #         f'#### Format Warnings\n{format_warnings}\n\n'
                        #         f'\n\n{reformatted_code}\n\n'
                        #     )
                        # )

        
                    if self.design_mode in [DesignModes.SCRATCH,DesignModes.CROSSOVER] and not INITIAL_PASS:
                        if len(self.tree.root_node.children)==0:
                            no_children_error=f'FETAL ERROR: No child units declared or detected under this root node, which is not allowed. You cannot have only one unit in your implementation, root node must have children.'  
                            format_errors.append(no_children_error)
                            self.stream.write(f':red[{no_children_error}]')

                    _func_checkpass = False
                    if fetal_errors==[]:
                        # run unit tests
                        if UNSTRUCT_CODER:
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
                            raise NotImplementedError('Structural prompts need to be updated.')
                            # _unit_test_results, _unit_test_code, _unit_test_passed = self.tree.test_unit(spec.unitname, True)
                            # self.stream.write(f'### Unit Tests Passed: {_unit_test_passed}')
                            # self.stream.write(f'### Unit Tests Results\n```bash\n{_unit_test_results}\n```')
                            # collapse_write(
                            #     self.stream,
                            #     'Unit Tests Code for '+unitname,
                            #     f'```python\n{_unit_test_code}\n```\n'
                            # )   
                            # _unit_test_results += f'### Unit Tests Results\n```bash\n{_unit_test_results}\n```\n\n'

                        gabcode = self.tree.compose()
                        self.log_fn(f'Checking the implementation of {selection}...','IMPLEMENTATION')
                        checkpass,check_report,gabcode_reformat,check_results = self.system.checker.check(GAMConfig_14M(),gabcode,selection,cpu_only=self.cpu_only)
                        self.log_fn(f'Checker checks result: {checkpass}','IMPLEMENTATION')
                        _func_checkpass = checkpass

                        if not _unit_test_passed:
                            if 'All tests passed!' in check_report:
                                check_report = check_report.replace('All tests passed!','Checker checks passed, but unit tests failed. You must implement the unit tests and pass them.')
                    
                        self.stream.write(f'### Check passed: {checkpass}')
                        self.stream.write(f'### Check Report\n```python\n{check_report}\n```')
                        self.stream.write(f'### Check Output\n```python\n{check_results}\n```')
                        self.stream.write(f'### Reformatted GAB Code\n')
                        self.stream.code(gabcode_reformat,language='python',line_numbers=True)
                        
                        checkpass = checkpass and _unit_test_passed if self.unittest_pass_required else checkpass
                        checker_report = check_report # XXX: Too long in the prompt
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

                ########################### Review the implementation ###########################
                
                if USE_PAIRING:
                    if len(format_errors+fetal_errors)>0:
                        _FORMAT_CHECKER_REPORT=FORMAT_CHECKER_REPORT
                    else:
                        _FORMAT_CHECKER_REPORT='Format check passed.'
                        if len(format_warnings)>0:
                            _FORMAT_CHECKER_REPORT+=f'\n\n#### Format Warnings\n{format_warnings}'
                    if not _func_checkpass:
                        _FUNCTION_CHECKER_REPORT=FUNCTION_CHECKER_REPORT
                    else:
                        _FUNCTION_CHECKER_REPORT='Functionality check passed.'
                    
                    if UNSTRUCT_OBSERVER:
                        if self.design_mode==DesignModes.MUTATION:
                            GUT_IMPLEMENTATION_OBSERVER_SYSTEM=P.O1M_IMPLEMENTATION_OBSERVER_BACKGROUND
                        elif self.design_mode==DesignModes.CROSSOVER:
                            GUT_IMPLEMENTATION_OBSERVER_SYSTEM=P.O1C_IMPLEMENTATION_OBSERVER_BACKGROUND
                        else:
                            GUT_IMPLEMENTATION_OBSERVER_SYSTEM=P.O1S_IMPLEMENTATION_OBSERVER_BACKGROUND
                    else:
                        raise NotImplementedError('Structural prompts need to be updated.')
                        # GUT_IMPLEMENTATION_OBSERVER_SYSTEM=P.gen_GUT_IMPLEMENTATION_OBSERVER_SYSTEM(unstruct=UNSTRUCT_CODER,mode=self.design_mode)
                    if self.design_mode==DesignModes.MUTATION:
                        _background_prompt={'SELECTION':proposal.selection,'SEED':self.seed}
                    elif self.design_mode==DesignModes.CROSSOVER:
                        _background_prompt={'PARENTS':self.seed}
                    else:
                        _background_prompt={}
                    IMPLEMENTATION_OBSERVER=reload_role('implementation_reviewer',self.agents['IMPLEMENTATION_OBSERVER'], GUT_IMPLEMENTATION_OBSERVER_SYSTEM(
                        GAB_BASE=GAB_BASE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE,
                        PROPOSAL=proposal.proposal,REVIEW=proposal.review,RATING=proposal.rating,**_background_prompt))
                    implementation_observer_tid=self.dialog.fork(main_tid,USER_CALLER,IMPLEMENTATION_OBSERVER,context=context_implementation_observer,
                                alias='implementation_observer',note=f'Observing implementation...')
                    self.log_fn(f'Searching similar units for {selection}...','IMPLEMENTATION')
                    unit_codes=self.system.sss.query_units_by_code(reformatted_code)[1]
                    self.log_fn(f'Searching finished.','IMPLEMENTATION')
                    REUSE_UNIT_PROMPT = P.gen_REUSE_UNIT_PROMPT(reformatted_code,reuse_from,reuse_type,self.design_mode,TYPE='observer')
                    if REFINE:
                        if attempt==0:
                            status_info=f'Observing refinement of {selection}...'
                            prompt_kwargs={}
                            if UNSTRUCT_OBSERVER:
                                GUT_IMPLEMENTATION_UNIT_REFINE_OBSERVE=P.O1_IMPLEMENTATION_UNIT_REFINE_OBSERVE
                                prompt_kwargs['FORMAT_CHECKER_REPORT']=_FORMAT_CHECKER_REPORT
                                prompt_kwargs['FUNCTION_CHECKER_REPORT']=_FUNCTION_CHECKER_REPORT
                            else:
                                raise NotImplementedError('Structural prompts need to be updated.')
                                # GUT_IMPLEMENTATION_UNIT_REFINE_OBSERVE=P.GUT_IMPLEMENTATION_UNIT_REFINE_OBSERVE
                            gum_implementation_unit_review_prompt=GUT_IMPLEMENTATION_UNIT_REFINE_OBSERVE(
                                UNIT_NAME=selection,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                                CHANGES=changes, VIEW=VIEW_DETAILED, DESCRIPTION=node.desc,
                                REVIEW=node.review,RATING=node.rating,SUGGESTIONS=node.suggestions,
                                SPECIFICATION=node.spec.to_prompt(),UNIT_CODES=unit_codes,
                                REUSE_UNIT_PROMPT=REUSE_UNIT_PROMPT,
                                **prompt_kwargs
                            )
                            GUT_IMPLEMENTATION_UNIT_REFINE_OBSERVE.apply(IMPLEMENTATION_OBSERVER.obj)
                        else:
                            status_info=f'Observing refined implementation of {selection} (version {attempt})...'
                            if UNSTRUCT_OBSERVER:
                                GUT_IMPLEMENTATION_REOBSERVE=P.O1_IMPLEMENTATION_UNIT_OBSERVE
                                gum_implementation_unit_review_prompt=GUT_IMPLEMENTATION_REOBSERVE(
                                    UNIT_NAME=selection,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                                    VIEW=VIEW_DETAILED, SPECIFICATION=spec.to_prompt(),UNIT_CODES=unit_codes,
                                    FORMAT_CHECKER_REPORT=_FORMAT_CHECKER_REPORT,
                                    FUNCTION_CHECKER_REPORT=_FUNCTION_CHECKER_REPORT,
                                    REUSE_UNIT_PROMPT=REUSE_UNIT_PROMPT,
                                )
                            else:
                                raise NotImplementedError('Structural prompts need to be updated.')
                                # GUT_IMPLEMENTATION_REOBSERVE=P.GUT_IMPLEMENTATION_REOBSERVE
                                # gum_implementation_unit_review_prompt=GUT_IMPLEMENTATION_REOBSERVE(
                                #     UNIT_NAME=selection,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                                #     CHANGES=changes,SPECIFICATION=spec.to_prompt(),UNIT_CODES=unit_codes,
                                # )
                            GUT_IMPLEMENTATION_REOBSERVE.apply(IMPLEMENTATION_OBSERVER.obj)
                    else: # first attempt of a new unit
                        status_info=f'Observing implementation of {selection}...'
                        prompt_kwargs={}
                        if UNSTRUCT_OBSERVER:
                            GUT_IMPLEMENTATION_UNIT_OBSERVE=P.O1_IMPLEMENTATION_UNIT_OBSERVE
                            prompt_kwargs['FORMAT_CHECKER_REPORT']=_FORMAT_CHECKER_REPORT
                            prompt_kwargs['FUNCTION_CHECKER_REPORT']=_FUNCTION_CHECKER_REPORT
                        else:
                            raise NotImplementedError('Structural prompts need to be updated.')
                            # GUT_IMPLEMENTATION_UNIT_OBSERVE=P.GUT_IMPLEMENTATION_UNIT_OBSERVE
                        gum_implementation_unit_review_prompt=GUT_IMPLEMENTATION_UNIT_OBSERVE(
                            UNIT_NAME=selection,VIEW=VIEW_DETAILED,SPECIFICATION=spec.to_prompt(),
                            ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,UNIT_CODES=unit_codes,
                            REUSE_UNIT_PROMPT=REUSE_UNIT_PROMPT,
                            **prompt_kwargs
                        )
                        GUT_IMPLEMENTATION_UNIT_OBSERVE.apply(IMPLEMENTATION_OBSERVER.obj)

                    with self.status_handler(status_info):
                        # if UNSTRUCT_OBSERVER: # no history  
                        context_implementation_observer=AgentContext() # no history for all
                        self.print_details(IMPLEMENTATION_OBSERVER.obj,context_implementation_observer,gum_implementation_unit_review_prompt)
                        _,out=self.call_dialog(implementation_observer_tid,gum_implementation_unit_review_prompt)
                        context_implementation_observer=self.dialog.context(implementation_observer_tid)
                        if UNSTRUCT_OBSERVER:
                            suggestions=None
                            review,rating=out['text'],out['rating']
                            self.stream.write(review)
                            self.print_raw_output(out,'IMPLEMENTATION_OBSERVER')
                            for _ in range(5):
                                succeed,rating,RETRY_PROMPT=P.gen_O1_RATING_DEBUG_prompt(rating)
                                if succeed:
                                    break
                                self.print_details(IMPLEMENTATION_OBSERVER.obj,context_implementation_observer,RETRY_PROMPT())
                                RETRY_PROMPT.apply(IMPLEMENTATION_OBSERVER.obj)
                                self.stream.write(f'Error in output, retry...') # TODO: very costly and wasteful, need to fix
                                _,out=self.call_dialog(implementation_observer_tid,RETRY_PROMPT())
                                rating=out['rating']
                                self.stream.write(f'##### Correcting rating: {rating}')
                                self.print_raw_output(out,'PROPOSAL_REVIEWER')
                            if not succeed:
                                info = 'Failed to generate a valid design proposal, stopping design process'
                                self.log_fn(info,'ERROR')
                                raise Exception(info)
                            passornot='Accept' if rating>=OBSERVE_THRESHOLD else 'Reject'
                            self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                        else:
                            raise NotImplementedError('Structural prompts need to be updated.')
                            # review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                            # passornot='Accept' if rating>=OBSERVE_THRESHOLD else 'Reject'
                            # self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                            # self.stream.write(review)
                            # self.stream.write(suggestions)
                            # self.print_raw_output(out,'IMPLEMENTATION_OBSERVER')
                            
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
                else:
                    succeed=True
                    costs={k:v-cost_raw[k] for k,v in self.costs.items()}
                    self.ptree.implement(acronym,self.tree,RETS['ROUNDS'],'unfinished',costs,self.design_cfg,self.user_input)
                    INITIAL_PASS=True
                    cost_raw=copy.deepcopy(self.costs)
                    # self.tree.units[unit_name].design_traces=traces
                    # removed=self.tree.clear_disconnected() # there might be some disconnected units, leave it for now
                    # PROTECTED_UNITS=list(set(PROTECTED_UNITS)-set(removed))
                    self.stream.write(f'#### Implementation passed, starting the next unit')
                    self.log_fn(f'Implementation passed, starting the next unit','IMPLEMENTATION')
                    break
            
            ########################### Round finished ###########################  
            if not succeed:
                self.stream.write(f'#### Implementation failed, trying the next unit')
                self.log_fn(f'Implementation failed, trying the next unit','IMPLEMENTATION')
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
                self.log_fn(f'Implementation passed, starting the next unit','IMPLEMENTATION')
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
        RETS['INITIAL_PASS']=INITIAL_PASS
        if end_reason is None:
            end_reason = EndReasons.IMPLEMENTATION_SUCCESS if SUCCEED else EndReasons.IMPLEMENTATION_FAILURE
        self.stream.log(end_reason,'end')
        if SUCCEED:
            self.log_fn(f'Design Implementation succeeded!','IMPLEMENTATION')
            self.stream.write('#### Design Implementation succeeded!')
            self.stream.balloons()
        else:
            self.log_fn(f'Design Implementation failed!','IMPLEMENTATION')
            self.stream.write('#### Design Implementation failed!')
            self.stream.snow()
        return RETS
    
    
    @register_module(
        "EXIT",
        hints="output the initial threads",
    )
    def end_of_design(self,query,state):
        self.log_fn('Design flow ended.','EXIT')
        # nothing to do here, as the ptree is already updated

        return query,state,{}

def gu_design(system,stream,sess_id,design_cfg,user_input='',proposal=None,cpu_only=False,log_fn=None):
    main_tid = system.dialog.fork(0,note='Starting a new session...',alias='main')
    status_handler = stream.status
    gu_flow = GUFlow(system, status_handler,stream,sess_id,design_cfg,user_input,cpu_only=cpu_only,log_fn=log_fn)
    GU_CALLEE = ROLE('GAB Unit Designer',gu_flow.flow)
    gum_tid = system.dialog.fork(main_tid,SYSTEM_CALLER,GU_CALLEE,note=f'launch design flow',alias=f'gu_flow')
    system.dialog.call(gum_tid,'',main_tid=main_tid,proposal=proposal) 


