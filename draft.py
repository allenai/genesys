
class AgentFlowNode:
    def __init__(self,id,alias,prog) -> None:
        self.id=id
        self.alias=alias
        self._call=prog
        self.children = None

    def __call__(self,query,states,**kwargs):
        raise NotImplementedError

    def type(self):
        raise NotImplementedError
    
    def link(self,children):
        assert isinstance(children,Dict[int,AgentFlowNode]), f'Children must be a dict of flow nodes'
        for id,child in children.items():
            assert isinstance(id,int) and id>=0, f'Children id must be an integer >=0'
            assert isinstance(child,AgentFlowNode), f'Children must be a flow node'
        self.children = children
    

class CONDNode(AgentFlowNode): # It will check a condition and return true or false, the order of the children is the order of the selections
    """
    A COND node input query and kwargs, output a selection index, it routes to another block
    """
    def __call__(self,query,states,**kwargs):
        assert self.children, f'CONDNode {self.alias}-{self.id}: COND node cannot be a terminal node'
        ret = self._call(query,states,**kwargs)
        assert isinstance(ret,int) and ret>=0, f'CONDNode {self.alias}-{self.id}: Condition must return a boolean or a positive integer'
        assert ret<len(self.children), f'CONDNode {self.alias}-{self.id}: Condition must return a value less than the number of selections'
        child = self.children[ret]
        assert isinstance(child,AgentFlowNode), f'CONDNode {self.alias}-{self.id}: Children must be a flow node'
        return child(query,states,**kwargs)
    
    def type(self):
        return 'COND'
    
class LOOPNode(AgentFlowNode): # It will loop until a condition is met
    """
    _call is a condition function, it returns a boolean value, and states, 
    """
    def __call__(self,query,states,**kwargs):
        assert self.children, f'LOOPNode {self.alias}-{self.id}: LOOP node cannot be a terminal node'
        assert len(self.children)==2, f'LOOPNode {self.alias}-{self.id}: Children of a LOOP node must be two, the first is the loop body, the second is the exit'
        while True:
            cont,states = self._call(query,states,**kwargs)
            assert isinstance(cont,bool), f'LOOPNode {self.alias}-{self.id}: Condition must return a boolean'
            if cont:
                query,states,kwargs = self.children[0](query,states,**kwargs)
            else:
                return self.children[1](query,states,**kwargs)
        
    def type(self):
        return 'LOOP'

class PROCNode(AgentFlowNode): # It will call an agent and return a response
    
    def __call__(self, query,states, **kwargs):
        query,states,ret = self._call(query,states,**kwargs)
        assert isinstance(query,str), f'A PROC node must return a string response message'
        assert isinstance(ret,dict), f'A PROC node must return a dict of additional returns'
        if self.children:
            assert len(self.children)==1, f'PROCNode {self.alias}-{self.id}: Children of a PROC node must be one'
            child = self.children[0]
            return child(query,states,**ret) 
        return query,states,ret
    
    def type(self):
        return 'PROC'

class AgentFlow:
    """
    input query and kwargs, output a response message and a dict of additional returns
    """
    def __init__(self,states={}):
        self.nodes={}
        self.states=states # global vars
        self.entry = PROCNode(0,'entry',lambda x,**kwargs: (x,kwargs))
        self.nodes[0] = self.entry
        self.alias_to_id = {'entry':0}

    def __call__(self,query,**kwargs):
        query,states,ret = self.entry(query,self.states,**kwargs)
        self.states = states
        return query,ret
    
    def new_node(self,alias,prog,type):
        assert alias not in self.alias_to_id, f'Alias {alias} already exists'
        id=self.assign_id()
        if type=='PROC':
            self.nodes[id] = PROCNode(id,alias,prog)
        elif type=='COND':
            self.nodes[id] = CONDNode(id,alias,prog)
        elif type=='LOOP':
            self.nodes[id] = LOOPNode(id,alias,prog)
        self.alias_to_id[alias] = id
        return id
    
    def new_proc(self,alias,prog):
        return self.new_node(alias,prog,'PROC')
    def new_cond(self,alias,prog):
        return self.new_node(alias,prog,'COND')
    def new_loop(self,alias,prog):
        return self.new_node(alias,prog,'LOOP')
    
    def assign_id(self):
        return len(self.nodes)

    def link(self,id_or_alias,children):
        if isinstance(children,AgentFlowNode):
            children = {0:children}
        elif isinstance(children,List[AgentFlowNode]):
            children = {i:child for i,child in enumerate(children)}
        elif isinstance(children,Dict[int,AgentFlowNode]):
            pass
        elif isinstance(children,int):
            assert children in self.nodes, f'Children id {children} does not exist'
            children = {0:self.nodes[children]}
        elif isinstance(children,str):
            assert children in self.alias_to_id, f'Children alias {children} does not exist'
            children = {0:self.nodes[self.alias_to_id[children]]}
        else:
            raise ValueError(f'Children must be a dict of flow nodes, or a list of flow nodes, or a single flow node')
        if isinstance(id_or_alias,str):
            id_or_alias = self.alias_to_id[id_or_alias]
        self.nodes[id_or_alias].link(children)






def _design(cls,query,states,stream,status_handler,parent_tid,context): # input query, context, output design and explanation, thread_tid is the id of the thread in which the design is running
    
    query,states,ret = initialize_design(query,states,cls,parent_tid,context)
    for attempt in range(cls._config.max_design_attempts):
        cls.logging.info(f'Attempting design, attempt={attempt}')
        
        if attempt == 0:
            thread_tid = design_thread_tid
        else:
            if debug_thread_tid is None:
                query = f'The designer designed the model: {text}\n\nThe checker failed the model: {query}\n\nPlease debug the model.'
                DEBUGGER = ROLE('debugger',cls.debugger)
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
                assert "# gab.py" in code 
            
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
                cls.states['refresh_template'] += 1
            
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
            if cls.states['refresh_template'] >=1:
                query+=f'\nHere is the template for the GAB block for you to refresh:\n\n```python\n{cls.gab_py}```'
            if cls.states['refresh_template'] >= 2:
                query+=f'\nHere is the definition of GABBase for you to refresh:\n\n```python\n{GAB_BASE}```'
            if cls.states['refresh_template'] >= 3:
                query+=f'\nHere is the definition for the GAM model for you to refresh:\n\n```python\n{cls.gam_py}```'
            if attempt == 0:
                initial_error = check_report

    if checkpass:
        report_query = (
            "The designed model passed the tests, now please generate a text report explaining and justifying your design."
            " Generate a creative name of your design as the title of your report in the first line of your response."
            " Do not include abbreviations or acronyms of your design in the title. You can use them in the body of the report."
            f" Here is the code of the designed model after degugging:\n\n{code}" # FIXME: what is the code after debugging is not the same as the idea before debugging
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

    return FAILED, {'code':None,'text':None,'check_results':None}
    
