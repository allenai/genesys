import os
import numpy as np
from typing import Any,Dict, List
import inspect
import copy

from exec_utils.models.model import ModelOutput
from .alang import FlowCreator,register_module,ROLE,SYSTEM_CALLER,USER_CALLER,AgentContext
from .gau_utils import check_and_reformat_gau_code

# from model_discovery.system import ModelDiscoverySystem
import model_discovery.agents.prompts.prompts as P
from model_discovery.model.composer import GAUBase, GAUTree, check_tree_name, GABComposer
import model_discovery.utils as U


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

gam_prompt_path = os.path.join(current_dir,'..','prompts','gam_prompt.py')
gau_template_path = os.path.join(current_dir,'..','prompts','gau_template.py')
GAM_TEMPLATE=open(gam_prompt_path).read()
GAU_TEMPLATE=open(gau_template_path).read()

GAU_BASE=inspect.getsource(GAUBase)
GAB_COMPOSER=inspect.getsource(GABComposer)


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

def collapse_write(stream,summary,content):
    stream.write(f'<details><summary>{summary}</summary>{content}</details>',unsafe_allow_html=True)


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
        self.outs=['design_stack']
        self.max_attemps={
            'design_proposal':10,
            'implementation_debug':10,
            'post_refinement':10,
        }
        self.lib_dir=system.lib_dir

        # prepare roles
        self.gpt4o0806_agent=self.system.designer # as we replaced the system prompt, essential its just a base agent
        self.gpt4omini_agent=self.system.debugger 

        self.tree = None

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
        links='implement_proposal_root',
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
                proposal_prompt=P.GU_PROPOSAL_REFINEMENT(REVIEW=review,RATING=rating,SUGGESTIONS=suggestions,
                                                         PASS_OR_NOT='Pass' if rating>3 else 'Fail')
                P.GU_PROPOSAL_REFINEMENT.apply(DESIGN_PROPOSER.obj)
            
            with self.status_handler(status_info):
                self.print_details(DESIGN_PROPOSER.obj,context_design_proposer,proposal_prompt)
                _,out=self.dialog.call(design_proposer_tid,proposal_prompt)
                modelname,proposal=out['modelname'],out['proposal']
                self.stream.write(f'### Model Name: {modelname}')
                modelname=U.to_camel_case_gab_class_name(modelname)
                reflection,changes=None,None
                if 'reflection' in out:
                    reflection,changes=out['reflection'],out['changes']
                    self.stream.write(f'# Proposal\n{proposal}')
                    self.stream.write(f'# Reflection\n{reflection}')
                    self.stream.write(f'# Changes\n{changes}')
                else:
                    self.stream.write(proposal)
                context_design_proposer=self.dialog.context(design_proposer_tid)
                self.print_raw_output(out)


            PROPOSAL_REVIEWER=reload_role('proposal_reviewer',self.gpt4o0806_agent, 
                                                P.GU_PROPOSAL_REVIEWER_SYSTEM())
            proposal_reviewer_tid=self.dialog.fork(main_tid,USER_CALLER,PROPOSAL_REVIEWER,context=context_proposal_reviewer,
                                                alias='proposal_review',note=f'Reviewing proposal...')
            if i==0:
                status_info=f'Reviewing initial proposal...'
                proposal_review_prompt=P.GU_PROPOSAL_REVIEW(PROPOSAL=proposal)
                P.GU_PROPOSAL_REVIEW.apply(PROPOSAL_REVIEWER.obj)
            else:
                status_info=f'Reviewing refined proposal (version {i})...'
                proposal_review_prompt=P.GU_PROPOSAL_REREVIEW(PROPOSAL=proposal,CHANGES=changes)
                P.GU_PROPOSAL_REREVIEW.apply(PROPOSAL_REVIEWER.obj)
            
            with self.status_handler(status_info):
                self.print_details(PROPOSAL_REVIEWER.obj,context_proposal_reviewer,proposal_review_prompt)
                _,out=self.dialog.call(proposal_reviewer_tid,proposal_review_prompt)
                review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                context_proposal_reviewer=self.dialog.context(proposal_reviewer_tid)
                passornot='Pass' if rating>3 else 'Fail'
                self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                self.stream.write(review)
                self.stream.write(suggestions)
                self.print_raw_output(out)

            trace={
                'modelname':modelname,
                'proposal':proposal,
                'review':review,
                'rating':rating,
                'suggestions':suggestions,
                'reflection':reflection,
                'changes':changes,
            }
            traces.append(trace)

            if rating>3:
                self.stream.write(f'#### Proposal passed with rating {rating} out of 5, starting implementation')
                check_tree_name(modelname,self.lib_dir) # TODO: error handling
                self.tree=GAUTree(name=modelname,proposal=proposal,review=review,rating=rating,suggestions=suggestions,lib_dir=self.lib_dir,proposal_traces=traces)
                break
        
        if rating<=3:
            self.stream.write(f'#### Proposal failed with rating {rating} out of 5, stopping design process')
            raise Exception('Design proposal failed, stopping design process')
        RET={
            'proposal':trace,
            'proposal_traces':traces,
        }
        return query,state,RET
    
    
    @register_module(
        "PROC",
        hints="output the root design",
        links='implement_proposal_recursive',
    )
    def implement_proposal_root(self,query,state,main_tid,proposal):
        self.dialog=self.system.dialog
        
        traces=[]
        context_design_implementer=AgentContext()
        context_implementation_reviewer=AgentContext()
        succeed=False
        for i in range(self.max_attemps['implementation_debug']):
            DESIGN_IMPLEMENTER=reload_role('design_implementer',self.gpt4o0806_agent,P.DESIGN_IMPLEMENTATER_SYSTEM(
                GAB_BASE=P.GAB_BASE,GAM_PY=GAM_TEMPLATE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE,GAB_COMPOSER=GAB_COMPOSER))
            design_implementer_tid=self.dialog.fork(main_tid,USER_CALLER,DESIGN_IMPLEMENTER,context=context_design_implementer,
                                                alias='design_implementation',note=f'Starting design implementation...')
            
            if i==0:
                status_info=f'Starting design implementation of root unit...'
                gu_design_root_prompt=P.GU_IMPLEMENTATION_ROOT(
                    PROPOSAL=proposal['proposal'],REVIEW=proposal['review'],RATING=proposal['rating'],
                )
                P.GU_IMPLEMENTATION_ROOT.apply(DESIGN_IMPLEMENTER.obj)
            else:
                status_info=f'Refining design implementation of root unit (attempt {i})...'
                gu_design_root_prompt=P.GU_IMPLEMENTATION_ROOT_RETRY(
                    FORMAT_CHECKER_REPORT=FORMAT_CHECKER_REPORT,
                    FUNCTION_CHECKER_REPORT=FUNCTION_CHECKER_REPORT,
                    REVIEW=review,RATING=rating,SUGGESTIONS=suggestions,
                    PASS_OR_NOT='Pass' if rating>3 else 'Fail',
                    GAU_BASE=GAU_BASE
                )
                P.GU_IMPLEMENTATION_ROOT_RETRY.apply(DESIGN_IMPLEMENTER.obj)
            with self.status_handler(status_info): 
                self.print_details(DESIGN_IMPLEMENTER.obj,context_design_implementer,gu_design_root_prompt)
                _,out=self.dialog.call(design_implementer_tid,gu_design_root_prompt)
                context_design_implementer=self.dialog.context(design_implementer_tid)

                reflection,changes=None,None
                if i==0:
                    analysis,spec,children,implementation=out['analysis'],out['spec'],out['children'],out['implementation']
                    spec=P.UnitSpec.model_validate(spec)
                    unitname=spec.unitname
                    self.stream.write(f'## Implementation of {unitname}')
                else:
                    reflection,analysis,spec,implementation,children,changes=out['reflection'],out['analysis'],out['spec'],out['implementation'],out['children'],out['changes']
                    spec=P.UnitSpec.model_validate(spec)
                    unitname=spec.unitname
                    self.stream.write(f'### Reflection\n{reflection}')
                    self.stream.write(f'## Refinement of {unitname}')
                self.stream.write(analysis)
                self.stream.write('### *Specification*\n'+spec.to_prompt())
                self.stream.write(f'### Code\n```python\n{implementation}\n```')
                if i>0:
                    self.stream.write(f'### Changes\n{changes}')
                
                children = {child['unitname']: P.UnitDeclaration.model_validate(child) for child in children}
                self.tree.declares=children # directly overwrite, as the root unit is the first one
                self.stream.write(f'### Children')
                for childname,child in children.items():
                    self.stream.write(f'##### {childname}\n'+child.to_prompt())

                self.print_raw_output(out)

            # Run all checks for every implementations, optimize both grammar and semantics at the same time 
            # avoid redundant debugging steps, i.e. only the debug for the passed plans are needed
            with self.status_handler('Checking the implementation of the root unit...'):
                # 1. check the format code for GAU
                reformatted_code,new_args,gau_tests,format_errors,format_warnings=check_and_reformat_gau_code(implementation,unitname,children)
                collapse_write(
                    self.stream,
                    'Code format check',
                    (
                        f'### Reformatted Code\n```python\n{reformatted_code}\n```\n\n'
                        f'#### New Arguments\n{new_args}\n\n'
                        f'#### Format Errors\n{format_errors}\n\n'
                        f'#### Format Warnings\n{format_warnings}\n\n'
                    )
                )
                format_checks = {
                    'format_errors':format_errors,
                    'format_warnings':format_warnings,
                }
                # !!!TODO: the reformatter should rename the existing kwargs from the tree, okey for root node
                # !!!TODO: remove any possible if __name__=='__main__' method from the code
                # FIXME: The checker seems buggy! It some times gives two same classes
                # 2. check the functionality of the composed GAB
                checkpass=False
                func_checks = {}
                self.tree.add_unit(
                    spec,reformatted_code,new_args,analysis,None,None,list(children.keys()),gau_tests,None,
                )
                if format_errors==[]:
                    # run unit tests
                    _unit_test_results, _unit_test_code, _unit_test_passed = self.tree.test_unit(spec.unitname, True)
                    self.stream.write(f'### Unit Tests Passed: {_unit_test_passed}')
                    self.stream.write(f'### Unit Tests Code\n```python\n{_unit_test_code}\n```')
                    self.stream.write(f'### Unit Tests Results\n```bash\n{_unit_test_results}```')

                    gabcode = self.tree.compose()
                    checkpass,check_report,gabcode_reformat,check_results = self.system.checker.check(self.system._cfg,gabcode,unitname)
                     
                    if not _unit_test_passed:
                        if 'All tests passed!' in check_report:
                            check_report = check_report.replace('All tests passed!','Checker checks passed, but unit tests failed. You must implement the unit tests and pass them.')
                    
                    self.stream.write(f'### Check passed: {checkpass}')
                    self.stream.write(f'### Check Report\n```python\n{check_report}\n```')
                    self.stream.write(f'### Check Output\n```python\n{check_results}\n```')
                    self.stream.write(f'### Reformatted GAB Code\n```python\n{gabcode_reformat}\n```')
                   
                    checkpass = checkpass and _unit_test_passed
                    check_report = _unit_test_results + '\n\n' + check_report
                    func_checks = {
                        'checkpass':checkpass,
                        'check_report':check_report,
                        'check_results':check_results,
                    }

            # 3. Review the code for GAU
            IMPLEMENTATION_REVIEWER=reload_role('implementation_reviewer',self.gpt4o0806_agent, 
                                                P.GU_IMPLEMENTATION_REVIEWER_SYSTEM(GAU_BASE=GAU_BASE))
            implementation_reviewer_tid=self.dialog.fork(main_tid,USER_CALLER,IMPLEMENTATION_REVIEWER,context=context_implementation_reviewer,
                                                alias='implementation_review',note=f'Reviewing implementation...')
            if i==0:
                status_info=f'Reviewing implementation of root unit...'
                gu_implementation_root_review_prompt=P.GU_IMPLEMENTATION_ROOT_REVIEW(
                    UNIT_NAME=unitname,PROPOSAL=proposal['proposal'],ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,CHECKER_REPORT=check_report,
                    SPECIFICATION=spec.to_prompt())
                P.GU_IMPLEMENTATION_ROOT_REVIEW.apply(IMPLEMENTATION_REVIEWER.obj)
            else:
                status_info=f'Reviewing refined implementation of root unit (version {i})...'
                gu_implementation_root_review_prompt=P.GU_IMPLEMENTATION_REREVIEW(
                    UNIT_NAME=unitname,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                    CHANGES=changes,CHECKER_REPORT=check_report,SPECIFICATION=spec.to_prompt()
                )
                P.GU_IMPLEMENTATION_REREVIEW.apply(IMPLEMENTATION_REVIEWER.obj)
            with self.status_handler(status_info):
                self.print_details(IMPLEMENTATION_REVIEWER.obj,context_implementation_reviewer,gu_implementation_root_review_prompt)
                _,out=self.dialog.call(implementation_reviewer_tid,gu_implementation_root_review_prompt)
                review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                context_implementation_reviewer=self.dialog.context(implementation_reviewer_tid)
                passornot='Accept' if rating>3 else 'Reject'
                self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                self.stream.write(review)
                self.stream.write(suggestions)
                self.print_raw_output(out)

                self.tree.units[unitname].rating=rating
                self.tree.units[unitname].review=review
                self.tree.units[unitname].suggestions=suggestions

            design = {
                'unit': self.tree.units[unitname].json(),
                'gab_code':gabcode_reformat,
                'format_checks':format_checks,
                'func_checks':func_checks,
                'reflection':reflection,
                'changes':changes,
            }
            traces.append(design)
            if not checkpass or rating<=3 or len(format_errors)>0:
                self.tree.del_unit(unitname) # didnt remove declares as there might be reuses
                FORMAT_CHECKER_REPORT = P.FORMAT_CHECKER_REPORT.format(
                    RESULT='failed' if len(format_errors)>0 else 'passed',
                    ERRORS=format_errors,
                    WARNINGS=format_warnings
                )
                if len(format_errors)>0:
                    FUNCTION_CHECKER_REPORT = 'Functionality check skipped due to format errors.'
                else:
                    if checkpass:
                        FUNCTION_CHECKER_REPORT = P.FUNCTION_CHECKER_REPORT_PASS.format(
                            REPORT=check_report,
                        )
                    else:
                        gabcode_reformat_with_line_num = U.add_line_num(gabcode_reformat)
                        FUNCTION_CHECKER_REPORT = P.FUNCTION_CHECKER_REPORT_FAIL.format(
                            REPORT=check_report,
                            GAB_CODE_WITH_LINE_NUM=gabcode_reformat_with_line_num
                        )
            else:
                succeed=True
                self.tree.units[unitname].design_traces=traces
                self.stream.write(f'#### Implementation of root passed, designing remaning units recursively')
                break
        
        if not succeed:
            self.stream.write(f'#### Implementation failed, stopping design process')
            # raise Exception('Design implementation failed, stopping design process')
            RET={
                'root_design':None,
                'root_design_traces':traces,
            }
        else:
            RET={
                'root_design':design,
                'root_design_traces':traces,
            }
        return query,state,RET


    
    @register_module(
        "PROC",
        hints="output the initial threads",
        links='self_evaluation',
    )
    def implement_proposal_recursive(self,query,state,main_tid,proposal,root_design):
        if not root_design:
            self.stream.write(f'#### Root design failed, stopping design process')
            return query,state,{'unit_designs':None}
        
        self.dialog=self.system.dialog

        RETS={}
        RETS['/FAILED']=[]
        GAB_CODE=root_design['gab_code']

        post_refinement=0
        while True:
            # Working on one unit at a time
            traces=[]
            context_design_implementer=AgentContext()
            context_implementation_reviewer=AgentContext()
            succeed=False
            VIEW,IMPLEMENTED,UNIMPLEMENTED=self.tree.view()
            if len(UNIMPLEMENTED)==0:
                post_refinement+=1

            DESIGN_IMPLEMENTER=reload_role('design_implementer',self.gpt4o0806_agent,P.DESIGN_IMPLEMENTATER_SYSTEM(
                GAB_BASE=P.GAB_BASE,GAM_PY=GAM_TEMPLATE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE,GAB_COMPOSER=GAB_COMPOSER))
            design_implementer_tid=self.dialog.fork(main_tid,USER_CALLER,DESIGN_IMPLEMENTER,context=context_design_implementer,
                                                alias='design_implementation',note=f'Starting design implementation...')
            with self.status_handler('Selecting the next unit to work on...'):
                GU_IMPLEMENTATION_UNIT_SELECTION=P.gen_GU_IMPLEMENTATION_UNIT_SELECTION(IMPLEMENTED+UNIMPLEMENTED)
                gu_implementation_unit_selection_prompt=GU_IMPLEMENTATION_UNIT_SELECTION(
                    PROPOSAL=proposal['proposal'],REVIEW=proposal['review'],RATING=proposal['rating'],
                    VIEW=VIEW, GAB_CODE=GAB_CODE
                )
                GU_IMPLEMENTATION_UNIT_SELECTION.apply(DESIGN_IMPLEMENTER.obj)
                self.print_details(DESIGN_IMPLEMENTER.obj,context_design_implementer,gu_implementation_unit_selection_prompt)
                self.stream.write(f'#### Current Tree Map and Units\n```\n{VIEW.replace('```python','').replace('```','')}\n```\n\nNow selecting the next unit to work on...')
                _,out=self.dialog.call(design_implementer_tid,gu_implementation_unit_selection_prompt)
                selection,motivation,termination=out['selection'],out['motivation'],out['termination']
                context_design_implementer=self.dialog.context(design_implementer_tid)
                self.stream.write(f'### Selection: {selection}')
                self.stream.write(f'### Motivation\n{motivation}')   

            declaration=self.tree.declares[selection]

            self.stream.write(f'##### Start design implementation of {selection}')

            if post_refinement>self.max_attemps['post_refinement']:
                self.stream.write(f'#### All units have been implemented and maximal refinements are reached, stopping design process')
                break
            
            if termination and len(UNIMPLEMENTED)==0:
                self.stream.write(f'#### All units have been implemented, the agent choose to terminate the design process')
                break
            
            for i in range(self.max_attemps['implementation_debug']):
                DESIGN_IMPLEMENTER=reload_role('design_implementer',self.gpt4o0806_agent,P.DESIGN_IMPLEMENTATER_SYSTEM(
                    GAB_BASE=P.GAB_BASE,GAM_PY=GAM_TEMPLATE,GAU_BASE=GAU_BASE,GAU_TEMPLATE=GAU_TEMPLATE,GAB_COMPOSER=GAB_COMPOSER))
                design_implementer_tid=self.dialog.fork(main_tid,USER_CALLER,DESIGN_IMPLEMENTER,context=context_design_implementer,
                                                    alias='design_implementation',note=f'Starting design implementation...')
                REFINE=False
                if i==0:
                    status_info=f'Starting design implementation of root unit...'
                    if selection in IMPLEMENTED:
                        REFINE=True
                        GU_IMPLEMENTATION_UNIT=P.gen_GU_IMPLEMENTATION_UNIT(refine=True)
                        node=self.tree.units[selection]
                        gu_implement_unit_prompt=GU_IMPLEMENTATION_UNIT(
                            SPECIFICATION=node.spec.to_prompt(),IMPLEMENTATION=node.code,REVIEW=node.review,RATING=node.rating,
                            SUGGESTIONS=node.suggestions
                        )
                        node_backup=copy.deepcopy(self.tree.units[selection])
                    else:
                        GU_IMPLEMENTATION_UNIT=P.gen_GU_IMPLEMENTATION_UNIT(refine=False)
                        gu_implement_unit_prompt=GU_IMPLEMENTATION_UNIT(DECLARATION=declaration.to_prompt())
                    GU_IMPLEMENTATION_UNIT.apply(DESIGN_IMPLEMENTER.obj)
                else: # Debugging or refining the implementation
                    status_info=f'Refining design implementation of root unit (attempt {i})...'
                    REFINE=True
                    gu_implement_unit_prompt=P.GU_IMPLEMENTATION_UNIT_RETRY(
                        FORMAT_CHECKER_REPORT=FORMAT_CHECKER_REPORT,
                        FUNCTION_CHECKER_REPORT=FUNCTION_CHECKER_REPORT,
                        REVIEW=review,RATING=rating,SUGGESTIONS=suggestions,
                        PASS_OR_NOT='Accept' if rating>3 else 'Reject',
                        GAU_BASE=GAU_BASE
                    )
                    P.GU_IMPLEMENTATION_UNIT_RETRY.apply(DESIGN_IMPLEMENTER.obj)
                with self.status_handler(status_info): 
                    self.print_details(DESIGN_IMPLEMENTER.obj,context_design_implementer,gu_implement_unit_prompt)
                    _,out=self.dialog.call(design_implementer_tid,gu_implement_unit_prompt)
                    context_design_implementer=self.dialog.context(design_implementer_tid)
                    reflection,changes=None,None
                    if REFINE:
                        reflection,analysis,implementation,changes,children,document=out['reflection'],out['analysis'],out['implementation'],out['changes'],out['children'],out['document']
                        self.stream.write(f'### Reflection\n{reflection}')
                        self.stream.write(f'## Refinement of {selection}')
                        if selection in IMPLEMENTED:
                            spec=self.tree.units[selection].spec
                            spec.document=document
                        else:
                            spec = P.UnitSpec(
                                unitname=selection,
                                document=document,
                                inputs=declaration.inputs,
                                outputs=declaration.outputs
                            )
                    else:
                        implementation,analysis,children,document=out['implementation'],out['analysis'],out['children'],out['document']
                        self.stream.write(f'## Implementation of {selection}')
                        spec = P.UnitSpec(
                            unitname=selection,
                            document=document,
                            inputs=declaration.inputs,
                            outputs=declaration.outputs
                        )                    
                    self.stream.write(analysis)
                    self.stream.write(f'### Document\n{document}')
                    self.stream.write(f'### Code\n```python\n{implementation}\n```')
                    if REFINE:
                        self.stream.write(f'### Changes\n{changes}')
                    
                    children = {child['unitname']: P.UnitDeclaration.model_validate(child) for child in children}
                    # never overwrite existing ones, as the children might be reused
                    new_declared = []
                    for childname,child in children.items():
                        if childname not in self.tree.declares and childname not in self.tree.units: # only add new ones
                            self.tree.declares[childname]=child
                            new_declared.append(childname)

                    self.stream.write(f'### Children')
                    for childname,child in children.items():
                        self.stream.write(f'##### {childname}\n'+child.to_prompt())

                    self.print_raw_output(out)


                # Run all checks for every implementations, optimize both grammar and semantics at the same time 
                # avoid redundant debugging steps, i.e. only the debug for the passed plans are needed
                with self.status_handler('Checking the implementation of the selected unit...'):
                    # 1. check the format code for GAU
                    reformatted_code,new_args,gau_tests,format_errors,format_warnings=check_and_reformat_gau_code(implementation,selection,children)
                    collapse_write(
                        self.stream,
                        'Code format check',
                        (
                            f'### Reformatted Code\n```python\n{reformatted_code}\n```\n\n'
                            f'#### New Arguments\n{new_args}\n\n'
                            f'#### Format Errors\n{format_errors}\n\n'
                            f'#### Format Warnings\n{format_warnings}\n\n'
                        )
                    )
                    format_checks = {
                        'format_errors':format_errors,
                        'format_warnings':format_warnings,
                    }
                    # !!!TODO: the reformatter should rename the existing kwargs from the tree, okey for root node
                    # !!!TODO: remove any possible if __name__=='__main__' method from the code
                    # 2. check the functionality of the composed GAB
                    checkpass=False
                    func_checks = {}
                    self.tree.add_unit(
                        spec,reformatted_code,new_args,analysis,None,None,list(children.keys()),gau_tests,None,demands=declaration.demands
                    )
                    if format_errors==[]:
                        # run unit tests
                        _unit_test_results, _unit_test_code, _unit_test_passed = self.tree.test_unit(spec.unitname, True)
                        self.stream.write(f'### Unit Tests Passed: {_unit_test_passed}')
                        self.stream.write(f'### Unit Tests Code\n```python\n{_unit_test_code}\n```')
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
                        check_report = _unit_test_results + '\n\n' + check_report
                    else:
                        check_report = 'Format check failed, please fix the format errors and try again.'
                        checkpass=False
                        check_results={}
                        
                    func_checks = {
                        'checkpass':checkpass,
                        'check_report':check_report,
                        'check_results':check_results,
                    }

                # 3. Review the code for GAU
                IMPLEMENTATION_REVIEWER=reload_role('implementation_reviewer',self.gpt4o0806_agent, 
                                                    P.GU_IMPLEMENTATION_REVIEWER_SYSTEM(GAU_BASE=GAU_BASE))
                implementation_reviewer_tid=self.dialog.fork(main_tid,USER_CALLER,IMPLEMENTATION_REVIEWER,context=context_implementation_reviewer,
                                                    alias='implementation_review',note=f'Reviewing implementation...')
                if REFINE:
                    if i==0:
                        status_info=f'Reviewing refinement of selected unit...'
                        gu_implementation_unit_review_prompt=P.GU_IMPLEMENTATION_UNIT_REFINE_REVIEW(
                            UNIT_NAME=selection,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                            CHANGES=changes,CHECKER_REPORT=check_report,PROPOSAL=proposal['proposal'],
                            VIEW=VIEW, GAB_CODE=GAB_CODE,DESCRIPTION=node.desc,REVIEW=node.review,
                            RATING=node.rating,SUGGESTIONS=node.suggestions,SPECIFICATION=node.spec.to_prompt()
                        )
                        P.GU_IMPLEMENTATION_UNIT_REFINE_REVIEW.apply(IMPLEMENTATION_REVIEWER.obj)
                    else:
                        status_info=f'Reviewing refined implementation of root unit (version {i})...'
                        gu_implementation_unit_review_prompt=P.GU_IMPLEMENTATION_REREVIEW(
                            UNIT_NAME=selection,ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,
                            CHANGES=changes,CHECKER_REPORT=check_report,SPECIFICATION=spec.to_prompt()
                        )
                        P.GU_IMPLEMENTATION_REREVIEW.apply(IMPLEMENTATION_REVIEWER.obj)
                else:
                    status_info=f'Reviewing implementation of root unit...'
                    gu_implementation_unit_review_prompt=P.GU_IMPLEMENTATION_UNIT_REVIEW(
                        UNIT_NAME=selection,PROPOSAL=proposal['proposal'],ANALYSIS=analysis,IMPLEMENTATION=reformatted_code,CHECKER_REPORT=check_report,
                        VIEW=VIEW, GAB_CODE=GAB_CODE,SPECIFICATION=spec.to_prompt())
                    P.GU_IMPLEMENTATION_UNIT_REVIEW.apply(IMPLEMENTATION_REVIEWER.obj)
                with self.status_handler(status_info):
                    self.print_details(IMPLEMENTATION_REVIEWER.obj,context_implementation_reviewer,gu_implementation_unit_review_prompt)
                    _,out=self.dialog.call(implementation_reviewer_tid,gu_implementation_unit_review_prompt)
                    review,rating,suggestions=out['review'],out['rating'],out['suggestions']
                    context_implementation_reviewer=self.dialog.context(implementation_reviewer_tid)
                    passornot='Accept' if rating>3 else 'Reject'
                    self.stream.write(f'### Rating: {rating} out of 5 ({passornot})')
                    self.stream.write(review)
                    self.stream.write(suggestions)
                    self.print_raw_output(out)

                    self.tree.units[selection].rating=rating
                    self.tree.units[selection].review=review
                    self.tree.units[selection].suggestions=suggestions

                design = {
                    'unit': self.tree.units[selection].json(),
                    'gab_code':gabcode_reformat,
                    'format_checks':format_checks,
                    'func_checks':func_checks,
                    'reflection':reflection,
                    'changes':changes,
                }
                traces.append(design)
                if not checkpass or rating<=3 or len(format_errors)>0:
                    if selection in UNIMPLEMENTED: 
                        self.tree.del_unit(selection) # remove the unit 
                    else:
                        self.tree.units[selection]=node_backup # restore the unit
                    for childname in new_declared: 
                        self.tree.del_declare(childname) # remove the new declared children to restore the tree
                    FORMAT_CHECKER_REPORT = P.FORMAT_CHECKER_REPORT.format(
                        RESULT='failed' if len(format_errors)>0 else 'passed',
                        ERRORS=format_errors,
                        WARNINGS=format_warnings
                    )
                    if len(format_errors)>0:
                        FUNCTION_CHECKER_REPORT = 'Functionality check skipped due to format errors.'
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
                    self.stream.write(f'#### Implementation passed, starting the next unit')
                    break
            
            if not succeed:
                self.stream.write(f'#### Implementation failed, trying the next unit')
                RET={
                    'unit_design':design,
                    'unit_design_traces':traces,
                }
                RETS['/FAILED'].append(RET)
            else:
                RET={
                    'unit_design':design,
                    'unit_design_traces':traces,
                }
                RETS[selection]=RET
        
        return query,state,{'unit_designs':RETS}


    @register_module(
        "PROC",
        hints="output the designs",
        links='end_of_design',
    )
    def self_evaluation(self,query,state,main_tid):
        # self evaluate then maybe redesign but the units can be reused
        self.dialog=self.system.dialog

        # TODO

        return query,state,{}


    
    @register_module(
        "EXIT",
        hints="output the initial threads",
    )
    def end_of_design(self,query,state,proposal,proposal_traces,root_design,root_design_traces,unit_designs):
        
        design_stack={
            'proposal':proposal,
            'proposal_traces':proposal_traces,
            'root_design':root_design,
            'root_design_traces':root_design_traces,
            'unit_designs':unit_designs,
        }
        RET={
            'design_stack':design_stack,
        }

        return query,state,RET


def gu_design_scratch(cls,query,stream,status_handler):
    main_tid = cls.dialog.fork(0,note='Starting a new session...',alias='main')
    gu_flow = GUFlowScratch(cls, status_handler, stream)
    GU_CALLEE = ROLE('GAB Unit Designer',gu_flow.flow)
    gu_tid = cls.dialog.fork(main_tid,SYSTEM_CALLER,GU_CALLEE,note=f'launch design flow',alias=f'gu_design')
    res,ret=cls.dialog.call(gu_tid,query,main_tid=main_tid)

    # actually tree comprises everything
    # return self.tree

    # return title,code,explain,summary,autocfg,reviews,ratings,check_results





