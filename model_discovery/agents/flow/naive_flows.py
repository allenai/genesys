import numpy as np

from .alang import ROLE,SYSTEM_CALLER,FAILED
from ..prompts.prompts import GAB_ERROR,REVIEWER_PROMPT,GAB_BASE,DESIGNER_PROMPT

# from model_discovery.system import ModelDiscoverySystem


def design_flow_definition():
    # args=['cls','stream','status_handler','parent_tid','context']
    # outs=['code','text','check_results']
    # design_flow = AgentDialogFlow(name='Model Design Flow',args=args,outs=outs)

    ALANG = 'FLOW `Model Design Flow` cls|stream|status_handler|parent_tid|context code|text|check_results\n'


    def design_initializer(query,state,cls,parent_tid,context,**kwargs):
        # assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        state['initial_error'] = None
        state['design_attemps'] = 0
        state['refresh_template'] = 0 
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
        # assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
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
        # assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
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
        # assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
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
        # assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        attempt = state['design_attemps']
        design_name = f"{cls._config.run_name}_{attempt}"
        checkpass,check_report,code,check_results = cls.checker.check(cls._cfg,code,design_name)
        checker_hints = check_results['hints']
        if 'REFRESH_TEMPLATE' in checker_hints:
            state['refresh_template'] += 1
        
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
        # assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        query = f"The designed model didn't pass, you need to try again. Here is the report:\n{check_report}. Please fix."
        if state['refresh_template'] >=1:
            query+=f'\nHere is the template for the GAB block for you to refresh:\n\n```python\n{cls.gab_py}```'
        if state['refresh_template'] >= 2:
            query+=f'\nHere is the definition of GABBase for you to refresh:\n\n```python\n{GAB_BASE}```'
        if state['refresh_template'] >= 3:
            query+=f'\nHere is the definition for the GAM model for you to refresh:\n\n```python\n{cls.gam_py}```'
        return query,state,{}
    # design_failed_node = design_flow.new_proc('Design failed prompt',design_failed,hint='Prompt to retry or end of the loop.')
    # design_flow.link(design_failed_node,design_loop_controller_node)
    ALANG += 'PROC design_failed_node `Design failed prompt` design_failed `Prompt to retry or end of the loop.`\n'
    ALANG += 'design_failed_node -> design_loop_controller_node\n'

    def design_succeed_return(query,state,cls,check_results,status_handler,code,stream,**kwargs):
        # assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
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




def design_naive(cls,query,stream,status_handler,parent_tid,context): # input query, context, output design and explanation, thread_tid is the id of the thread in which the design is running
    # assert isinstance(cls,ModelDiscoverySystem), f'cls must be an instance of ModelDiscoverySystem'
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

        with status_handler(f"Design Attempt {attempt+1}"): 
            
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

        if checkpass:  
            break # goto next block
        else:
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
    

def review_naive(cls,query,stream,status_handler,parent_tid,context): 
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



def naive_design_review(cls,query,stream,status_handler):
    main_tid = cls.dialog.fork(0,note='Starting a new session...',alias='main')
    design_query = DESIGNER_PROMPT.format(
        gab_base=GAB_BASE,
        gam_py=cls.gam_py,
        gab_py=cls.gab_py,
        config=cls._cfg.to_prompt(), #<--- need to parameterize 
        instruct=query,
    )
    query=design_query
    
    refine_pipe_tid = cls.dialog.fork(main_tid,SYSTEM_CALLER,SYSTEM_CALLER,note='Design refinement pipe.',alias='refine')
    for i in range(cls._config.max_design_refines):
        DESIGN_CALLEE = ROLE('designer',cls.design_flow)
        design_pipe_tid = cls.dialog.fork(refine_pipe_tid,SYSTEM_CALLER,DESIGN_CALLEE,note=f'launch design flow',alias=f'design_{i}')
        REVIEW_CALLEE = ROLE('reviewer',cls.review_flow) 
        review_pipe_tid = cls.dialog.fork(refine_pipe_tid,SYSTEM_CALLER,REVIEW_CALLEE,note=f'launch review flow',alias=f'review_{i}')
        cls.dialog.carry(refine_pipe_tid,design_pipe_tid,review_pipe_tid)
        rres,(lres,lret,rret) = cls.dialog.call(refine_pipe_tid,query,
                                                    largs={'cls':cls,'stream':stream,'status_handler':status_handler,'context':cls.dialog.context(refine_pipe_tid)},
                                                    rargs={'cls':cls,'stream':stream,'status_handler':status_handler,'context':None})
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
    autocfg = cls.checker.tune(cls._cfg,code,title)
    # except Exception as e:
    #     print(f"Error tuning the scale of designed model: {e}")
    #     return None
    
    ### Generate a summary
    with status_handler(f"Generating summary..."):
        cls.logging.info('Generating summary of the design...')
        summary_query = (
            "Here is a design of an autoregressive language model block. "
            "The code and explanation of the design are provided below:\n\n"
            f"{explain}\n\nImplementation of {title}:\n\n{code}\n\n"
            "Please summarize the design with a description of the design and a simple pseudo code that conclude the core idea in few sentences."
        )
        SUMMARY_CALLER = ROLE('designer',cls.designer)
        summary_thread_tid = cls.dialog.fork(main_tid,SYSTEM_CALLER,SUMMARY_CALLER,note='Starting summary process...')
        _,response = cls.dialog.call(summary_thread_tid,query=summary_query)
        summary=response['text']
        if stream:
            stream.markdown(summary)
    
    return title,code,explain,summary,autocfg,reviews,ratings,check_results