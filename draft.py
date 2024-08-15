
def design_initializer(query,state,cls,parent_tid,context,**kwargs):
    assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
    state['initial_error'] = None
    state['design_attemps'] = 0
    DESIGNER = ROLE('designer',cls.designer)
    design_thread_tid=cls.dialog.fork(parent_tid,SYSTEM_CALLER,DESIGNER,context=context,
                                        alias='designing',note=f'Starting design...')
    debug_thread_tid=None
    return query,state,{'design_thread_tid':design_thread_tid,'debug_thread_tid':debug_thread_tid}


def design_loop_controller(query,state,cls,**kwargs):
    assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
    cont = state['design_attemps'] < cls._config.max_design_attempts
    attempt = state['design_attemps']
    cls.logging.info(f'Attempting design, attempt={attempt}')
    state['design_attemps'] += 1
    return cont,state

def design_thread_switch(query,state,**kwargs):
    attempt = state['design_attemps']
    state['current_thread']=kwargs['design_thread_tid']
    return 0 if attempt == 1 else 1,state

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

def gocheck_or_goback(query,state,generated,**kwargs):
    return 1 if generated else 0

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

def check_pass(query,state,checkpass,**kwargs):
    return 1 if checkpass else 0

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

def design_terminal_check(query,state,checkpass,**kwargs):
    return 1 if checkpass else 0

def design_failure_exit(query,state,**kwargs):
    return FAILED,state,{'code':None,'text':None,'check_results':None}




def design_flow_definition():
    args=['cls','stream','status_handler','parent_tid','context']
    outputs=['code','text','check_results']
    design_flow = AgentDialogFlow(name='Model Design Flow',args=args,outputs=outputs)

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

    init_design_node = design_flow.new_proc('Initialize Design Flow',design_initializer)
    design_flow.link(design_flow.id_entry,init_design_node)
    ALANG += 'PROC init_design_node `Initialize Design Flow` design_initializer\n'
    ALANG += 'LINK ENTRY init_design_node\n'


    def design_loop_controller(query,state,cls,**kwargs):
        assert isinstance(cls,ModelDiscoverySystem), f'cls must be a ModelDiscoverySystem object'
        cont = state['design_attemps'] < cls._config.max_design_attempts
        attempt = state['design_attemps']
        cls.logging.info(f'Attempting design, attempt={attempt}')
        state['design_attemps'] += 1
        return cont,state

    design_loop_controller_node = design_flow.new_loop('Design Loop Controler',design_loop_controller,hints={0:'Enter the design loop',1:'The design loop should terminate'})
    design_flow.link(init_design_node,design_loop_controller_node)
    ALANG += 'LOOP design_loop_controller_node `Design Loop Controler` design_loop_controller `Enter the design loop`|`The design loop should terminate`\n'
    ALANG += 'LINK init_design_node design_loop_controller_node\n'

    def design_thread_switch(query,state,**kwargs):
        attempt = state['design_attemps']
        state['current_thread']=kwargs['design_thread_tid']
        return 0 if attempt == 1 else 1,state
    design_switch_node = design_flow.new_cond('Design Switch',design_thread_switch,{0:'Sample initial design',1:'Debug the design'})
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
    switch_to_debug_node = design_flow.new_proc('Switch to Debug Thread',switch_to_debug,hint='Debugger take over')
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
    design_loop_body_node = design_flow.new_proc('Design Loop Body',design_loop_body,hint='Agent response')
    design_flow.link(design_switch_node,{0:design_loop_body_node,1:switch_to_debug_node})
    design_flow.link(switch_to_debug_node,design_loop_body_node)
    ALANG += 'PROC design_loop_body_node `Design Loop Body` design_loop_body `Agent response`\n'
    ALANG += 'LINK design_switch_node design_loop_body_node|switch_to_debug_node\n'
    ALANG += 'LINK switch_to_debug_node design_loop_body_node\n'

    def gocheck_or_goback(query,state,generated,**kwargs):
        return 1 if generated else 0
    gocheck_or_goback_node = design_flow.new_cond('Whether code is generated?',gocheck_or_goback,{0:'No, go back and retry',1:'Yes, pass to the checker'})
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
    check_design_node = design_flow.new_proc('Checker checks design',check_design)
    design_flow.link(design_loop_body_node,gocheck_or_goback_node)
    design_flow.link(gocheck_or_goback_node,{0:design_loop_controller_node,1:check_design_node})
    ALANG += 'PROC check_design_node `Checker checks design` check_design\n'
    ALANG += 'LINK design_loop_body_node gocheck_or_goback_node\n'
    ALANG += 'LINK gocheck_or_goback_node design_loop_controller_node|check_design_node\n'

    def check_pass(query,state,checkpass,**kwargs):
        return 1 if checkpass else 0
    check_pass_node = design_flow.new_cond('Pass or not?',check_pass,{0:'Failed, retry or end of the loop.',1:'Passed, go to report generation.'})
    design_flow.link(check_design_node,check_pass_node)
    ALANG += 'COND check_pass_node `Pass or not?` check_pass `Failed, retry or end of the loop.`|`Passed, go to report generation.`\n'
    ALANG += 'LINK check_design_node check_pass_node\n'

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
    design_failed_node = design_flow.new_proc('Design failed prompt',design_failed,hint='Prompt to retry or end of the loop.')
    design_flow.link(design_failed_node,design_loop_controller_node)
    ALANG += 'PROC design_failed_node `Design failed prompt` design_failed `Prompt to retry or end of the loop.`\n'
    ALANG += 'LINK design_failed_node design_loop_controller_node\n'

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
    design_succeed_node = design_flow.new_proc('Design succeed & report generation',design_succeed_return, is_end=True)
    design_flow.link(check_pass_node,{0:design_failed_node,1:design_succeed_node})
    ALANG += 'EXIT design_succeed_node `Design succeed & report generation` design_succeed_return\n'
    ALANG += 'LINK check_pass_node design failed_node|design_succeed_node\n'

    def design_terminal_check(query,state,checkpass,**kwargs):
        return 1 if checkpass else 0
    design_terminal_check_node = design_flow.new_cond('Loop terminated.',design_terminal_check,{0:'Design failed',1:'Design succeed'})
    ALANG += 'COND design_terminal_check_node `Loop terminated.` design_terminal_check `Design failed`|`Design succeed`\n'

    def design_failure_exit(query,state,**kwargs):
        return FAILED,state,{'code':None,'text':None,'check_results':None}
    design_failure_exit_node = design_flow.new_proc('Exit with failure',design_failure_exit, hint='Output FAILED and "None"s', is_end=True)
    design_flow.link(design_terminal_check_node,{0:design_failure_exit_node,1:design_succeed_node})
    ALANG += 'EXIT design_failure_exit_node `Exit with failure` design_failure_exit `Output FAILED and "None"s`\n'
    ALANG += 'LINK design_terminal_check_node design_failure_exit_node|design_succeed_node\n'
    
    design_flow.link(design_loop_controller_node,{0:design_switch_node,1:design_terminal_check_node})
    ALANG += 'LINK design_loop_controller_node design_switch_node|design_terminal_check_node\n'
    return design_flow, ALANG


import types

def extract_functions_from_function(func):
    extracted_functions = {}

    # Loop through the constants in the code object of the given function
    for const in func.__code__.co_consts:
        if isinstance(const, types.CodeType):
            # Retrieve the function name
            function_name = const.co_name
            
            # Create the function object
            function_obj = types.FunctionType(const, globals())
            
            # Add the function to the dictionary
            extracted_functions[function_name] = function_obj
    
    return extracted_functions

print(extract_functions_from_function(design_flow_definition))