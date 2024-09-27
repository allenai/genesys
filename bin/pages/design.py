import json
import time
import pathlib
import datetime
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import sys,os
import threading
import queue
sys.path.append('.')
import model_discovery.utils as U
from model_discovery.agents.flow.gau_flows import EndReasons,RunningModes,END_REASONS_LABELS,DesignModes
import bin.app_utils as AU



def get_thread_safe_session_state():
    try:
        return st.session_state
    except RuntimeError:
        return {}


#################################################

def stat_logs(logs):
    end_reasons = {}
    end_labels = {}
    for reason in EndReasons:
        end_reasons[reason] = 0
    for _,label in END_REASONS_LABELS.items():
        end_labels[label] = 0
    for log in logs:
        unfinished = True
        for _,text,type in log:
            if type=='end':
                unfinished=False
                end_reasons[text] += 1
        if unfinished:
            end_reasons[EndReasons.UNFINISHED] += 1
    
    end_ratios = {i:num/sum(end_reasons.values()) for i,num in end_reasons.items()}
    for i in end_reasons:
        end_labels[END_REASONS_LABELS[i]] += end_reasons[i]
    end_label_ratios = {i:num/sum(end_labels.values()) for i,num in end_labels.items()}


    col1,col2=st.columns([1,1])
    
    with col1:
        st.write('###### Total number of runs: ',sum(end_reasons.values()))
        st.write('###### Success rate')
        for reason in end_label_ratios:
            count = end_labels[reason]
            ratio = end_label_ratios[reason]
            st.write(f"{reason}: {count} ({ratio*100:.2f}%)")
            
    with col2:
        st.write('###### End reasons')
        for reason in end_ratios:
            count = end_reasons[reason]
            ratio = end_ratios[reason]
            st.write(f"{reason.value}: {count} ({ratio*100:.2f}%)")
    

def show_log(log):
    # four types: enter, exit, write, markdown
    in_status = False
    code = []
    for dt, text, type in log:
        line = []
        if in_status:
            line.append('\t')
        
        # Use repr() to get a string literal representation
        text_repr = repr(text)
        
        if type == 'enter':
            in_status = True
            line.append(f"with st.status({text_repr}):")
        elif type == 'exit':
            in_status = False
        elif type == 'write':
            line.append(f"st.write({text_repr}, unsafe_allow_html=True)")
        elif type == 'warning':
            line.append(f"st.warning({text_repr})")
        elif type == 'markdown':
            line.append(f"st.markdown({text_repr}, unsafe_allow_html=True)")
        
        code.append(''.join(line))
    
    final_code = '\n'.join(code)
    exec(final_code)

def load_log(log_file):
    log=[]
    for line in U.read_file(log_file).split('\n'):
        try:
            log.append(eval(line.replace('/NEWLINE/','\n').replace('/TAB/','\t')))
        except:
            log.append((datetime.datetime.now(),f'MISSING LINE: ERROR IN LOG FILE: {line}','warning'))
    return log




###########################################################################

def _run_design(evosys, thread_id, n_sources=None, design_cfg={}, search_cfg={}, user_input='', design_id=None, mode=None, resume=True, placeholder=None):
    session_state = get_thread_safe_session_state()
    stop_event = session_state.get('design_threads', {}).get(thread_id, {}).get('stop_event', threading.Event())
    output_queue = session_state.get('design_threads', {}).get(thread_id, {}).get('queue', queue.Queue())
    
    # Create a custom StreamlitOutputStream that puts messages in the queue
    class ThreadSafeStream:
        def write(self, text):
            output_queue.put(text)
        
        def __getattr__(self, attr):
            return lambda *args, **kwargs: None  # No-op for other methods

    thread_safe_stream = ThreadSafeStream()
    
    while not stop_event.is_set():
        # Run the design function with the thread-safe stream
        select_cfg={'n_sources':n_sources}
        evosys.design(select_cfg, design_cfg, search_cfg, user_input=user_input, mode=mode, design_id=design_id, resume=resume, stream=thread_safe_stream)
        time.sleep(1)  # Simulate some processing time
        print(f"Thread {thread_id} is running")

    print(f"Thread {thread_id} has stopped.")

# Function to start a new thread
def start_design_thread(evosys, n_sources=None, design_cfg={}, search_cfg={}, user_input='', design_id=None, mode=None, resume=True):
    thread_id = len(st.session_state.get('design_threads', {})) + 1  # Unique thread ID
    stop_event = threading.Event()
    
    # Create a placeholder for this thread's output
    placeholder = st.empty()
    
    design_thread = threading.Thread(target=_run_design, 
                                     args=(evosys, thread_id, n_sources, design_cfg, search_cfg, user_input, design_id, mode, resume, placeholder))
    
    # Add Streamlit's script run context to the thread
    add_script_run_ctx(design_thread)
    
    # Store the thread, stop event, and placeholder in session state
    if 'design_threads' not in st.session_state:
        st.session_state['design_threads'] = {}
    st.session_state['design_threads'][thread_id] = {
        "thread": design_thread, 
        "stop_event": stop_event,
        "placeholder": placeholder,
        "queue": queue.Queue()
    }
    
    # Start the thread
    design_thread.start()
    return thread_id


def _design_tuning(evosys,project_dir):
    ### build the system 
    st.title("Design Engine Playground")

    system = evosys.rnd_agent
    db_dir = evosys.ptree.db_dir
    design_cfg = {}
    n_sources = {}

    with st.sidebar:
        st.write('Controls')
        if st.button("Reset design query"):#,use_container_width=True):
            st.rerun()

    #### Configure design

    with st.expander("**Playground Settings**",expanded=True):#,icon='⚙️'):

        col1, col2 = st.columns([1, 5])
        with col1:
            mode = st.selectbox(label="Design Mode",options=[i.value for i in DesignModes],disabled=True)
        with col2:
            # st.markdown("#### Configure the base models for each agent")
            AGENT_TYPES = ['claude3.5_sonnet','gpt4o_0806','gpt4o_mini']
            agent_type_labels = {
                'DESIGN_PROPOSER':'Proposal Agent',
                'PROPOSAL_REVIEWER':'Proposal Reviewer',
                'IMPLEMENTATION_PLANNER':'Implementation Planner',
                'IMPLEMENTATION_CODER':'Implementation Coder',
                'IMPLEMENTATION_OBSERVER':'Implementation Observer',
                'SEARCH_ASSISTANT': '*Separate Search Assistant*'
            }
            agent_types = {}
            cols = st.columns(len(agent_type_labels))
            for i,agent in enumerate(agent_type_labels):
                with cols[i]:
                    index=0 
                    options=AGENT_TYPES
                    if agent in ['SEARCH_ASSISTANT']:
                        options=AGENT_TYPES+['None']
                        index=len(options)-1
                    elif agent in ['IMPLEMENTATION_OBSERVER']:
                        options=AGENT_TYPES+['o1_preview','o1_mini','None']
                        index=len(options)-2
                    elif agent in ['IMPLEMENTATION_CODER','DESIGN_PROPOSER','PROPOSAL_REVIEWER','IMPLEMENTATION_PLANNER']: 
                        options=AGENT_TYPES+['o1_preview','o1_mini']
                        if agent in ['IMPLEMENTATION_CODER']:
                            index=len(options)-1
                        else:
                            index=len(options)-1
                    agent_types[agent] = st.selectbox(label=agent_type_labels[agent],options=options,index=index,disabled=agent=='SEARCH_ASSISTANT')
            design_cfg['agent_types'] = agent_types

        if mode!=DesignModes.MUTATION.value:
            st.toast("WARNING!!!: Only mutation mode is supported now. Other modes are not stable or unimplemented.")

        sources = ['ReferenceCoreWithTree', 'DesignArtifactImplemented', 'DesignArtifact', 'ReferenceCore', 'ReferenceWithCode', 'Reference']
        sources={i:len(evosys.ptree.filter_by_type(i)) for i in sources}
        
        st.markdown("##### Configure the number of seeds to sample from each source")
        cols = st.columns(len(sources))
        for i,source in enumerate(sources):
            with cols[i]:
                if mode==DesignModes.MUTATION.value and source=='ReferenceCoreWithTree':
                    n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=1)#,max_value=1,disabled=True)
                else:
                    init_value=0 if source in ['DesignArtifact','ReferenceCore'] else min(2,sources[source])
                    if source == 'DesignArtifactImplemented':
                        init_value = min(1,sources[source])
                    # disabled=True if source == 'DesignArtifact' else False
                    n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=init_value,max_value=sources[source])#,disabled=disabled)
        if mode==DesignModes.MUTATION.value:
            st.write('**ReferenceCoreWithTree and DesignArtifactImplemented are seed types. Will randomly sample one from samples from them as seed.*')

        col1,col2=st.columns([3,2])
        termination={}
        threshold={}
        max_attempts = {}
        with col1:
            st.markdown("##### Configure termination conditions and budgets")
            cols=st.columns(4)
            with cols[0]:
                termination['max_failed_rounds'] = st.number_input(label="Max failed rounds (0 is no limit)",min_value=1,value=3)
            with cols[1]:
                termination['max_total_budget'] = st.number_input(label="Max total budget (0 is no limit)",min_value=0,value=0)
            with cols[2]:
                termination['max_debug_budget'] = st.number_input(label="Max debug budget (0 is no limit)",min_value=0,value=0)
            with cols[3]:
                max_attempts['max_search_rounds'] = st.number_input(label="Max search rounds",min_value=0,value=4)
        with col2:
            st.markdown("##### Configure the threshold for rating the design")
            cols=st.columns(2)
            with cols[0]:
                threshold['proposal_rating'] = st.slider(label="Proposal rating",min_value=0,max_value=5,value=4)
            with cols[1]:
                threshold['implementation_rating'] = st.slider(label="Implementation rating",min_value=0,max_value=5,value=3)
        design_cfg['termination'] = termination
        design_cfg['threshold'] = threshold 


        col1,col2=st.columns([4,5])
        with col1:
            st.markdown("##### Configure max number of attempts")
            cols=st.columns(3)
            with cols[0]:
                max_attempts['design_proposal'] = st.number_input(label="Max proposal attempts",min_value=3,value=5)
            with cols[1]:
                max_attempts['implementation_debug'] = st.number_input(label="Max debug attempts",min_value=3,value=5)
            with cols[2]:
                max_attempts['post_refinement'] = st.number_input(label="Max post refinements",min_value=0,value=0)
        with col2:
            st.markdown("##### Re-show previous runs (work in progress)")
            cols=st.columns([2,2,3])
            with cols[0]:
                ckpts=os.listdir(evosys.ckpt_dir)
                current_ckpt = evosys.evoname
                selected_ckpt = st.selectbox(label="Select a ckpt",options=ckpts,index=ckpts.index(current_ckpt),disabled=True)
                db_dir = U.pjoin(evosys.ckpt_dir,selected_ckpt,'db')
            with cols[1]:
                folders=['']
                if os.path.exists(U.pjoin(db_dir,'sessions')):
                    for i in os.listdir(U.pjoin(db_dir,'sessions')):
                        if os.path.isdir(U.pjoin(db_dir,'sessions',i,'log')):
                            folders.append(i)
                selected_folder = st.selectbox(label="**View folder stats**",options=folders,disabled=True)
                selected_folder_dir = U.pjoin(db_dir,'sessions',selected_folder,'log')
            with cols[2]:
                designs=['']
                if selected_folder and os.path.exists(selected_folder_dir):
                    designs += os.listdir(selected_folder_dir)
                folder_name = selected_folder if selected_folder else 'No folder selected'
                selected_design = st.selectbox(label=f"***View runs in selected folder: {folder_name}***",options=designs,disabled=True)
        design_cfg['max_attempts'] = max_attempts

    with st.expander("Search Settings",expanded=False):
        search_cfg={}
        search_cfg['result_limits']={}
        search_cfg['perplexity_settings']={}
        search_cfg['proposal_search_cfg']={}

        cols=st.columns([2,2,2,3,2,3])
        with cols[0]:
            search_cfg['result_limits']['lib']=st.number_input("Library Primary",value=5,min_value=0,step=1)
        with cols[1]:
            search_cfg['result_limits']['lib2']=st.number_input("Library Secondary",value=0,min_value=0,step=1,disabled=True)
        with cols[2]:
            search_cfg['result_limits']['libp']=st.number_input("Library Plus",value=0,min_value=0,step=1,disabled=True)
        with cols[3]:
            search_cfg['rerank_ratio']=st.slider("Rerank Scale Ratio (0 means no rerank)",min_value=0.0,max_value=1.0,value=0.2,step=0.01)
        with cols[4]:
            search_cfg['proposal_search_cfg']['top_k']=st.number_input("Proposal Top K",value=3,min_value=0,step=1)
        with cols[5]:
            search_cfg['proposal_search_cfg']['cutoff']=st.slider("Proposal Search Cutoff",min_value=0.0,max_value=1.0,value=0.5,step=0.01)

        cols=st.columns([2,2,2,2,2])
        with cols[0]:
            search_cfg['result_limits']['s2']=st.number_input("S2 Search Result Limit",value=5,min_value=0,step=1)
        with cols[1]:
            search_cfg['result_limits']['arxiv']=st.number_input("Arxiv Search Result Limit",value=3,min_value=0,step=1)
        with cols[2]:
            search_cfg['result_limits']['pwc']=st.number_input("Papers With Code Search Result Limit",value=3,min_value=0,step=1)
        with cols[3]:
            search_cfg['perplexity_settings']['model_size']=st.selectbox("Perplexity Model Size",options=['none','small','large','huge'],index=2)
        with cols[4]:
            search_cfg['perplexity_settings']['max_tokens']=st.number_input("Perplexity Max Tokens",value=4000,min_value=500,step=100,disabled=search_cfg['perplexity_settings']['model_size']=='none')
       

    #### Run design

    # cols = st.columns([7,2.5,1.8,1.2])
    cols = st.columns([6,2,2,1,1])
    with cols[0]:
        user_input = st.text_input(label = "Add any additional instructions (optional)" )
    with cols[1]:
        running_mode = st.selectbox(label="Running Mode",options=[i.value for i in RunningModes],index=2,disabled=True)
        design_cfg['running_mode'] = RunningModes(running_mode)
    with cols[2]:
        EXPERIMENT_RUNS = st.number_input(label="Number of design runs",min_value=1,value=1,disabled=True)
    with cols[3]:
        st.write('')
        st.write('')
        submit = st.button(label="***Run design***",disabled=mode!=DesignModes.MUTATION.value)
    with cols[4]:
        st.write('')
        st.write('')
        resume = st.checkbox(label="Resume",value=True)

    if submit:
        if mode==DesignModes.MUTATION.value:
            assert n_sources['ReferenceCoreWithTree']+n_sources['DesignArtifactImplemented']>0, "You must select at least one ReferenceCoreWithTree orDesignArtifactImplemented as seeds."
        for i in range(EXPERIMENT_RUNS):
            with st.empty():
                if EXPERIMENT_RUNS>1:
                    st.subheader(f'Running {EXPERIMENT_RUNS} designs, save to *{evosys.ptree.db_dir}*...')

            if sum(n_sources.values())==0:
                st.write("You selected no seed, the agent will randomly generate a design for you.")

            if EXPERIMENT_RUNS==1:
                spinner_text = "running model design loop"
            else:
                spinner_text = f"running model design loop ({i+1}/{EXPERIMENT_RUNS})"
            with st.spinner(text=spinner_text):
                _mode = DesignModes(mode)
                design_id=None
                select_cfg={'n_sources':n_sources}
                evosys.design(select_cfg,design_cfg,search_cfg,user_input=user_input,mode=_mode,design_id=design_id,resume=resume)
    
    elif selected_design:
        with st.empty():
            st.markdown(f'### Viewing design log for session: *{selected_design}*...')
        log=eval(U.read_file(U.pjoin(selected_folder_dir,'stream.log')))
        show_log(log)
    
    elif selected_folder:
        with st.empty():
            st.subheader(f'Viewing statistics for experiment folder: *{selected_folder}*')
        if not U.pexists(selected_folder_dir):
            st.write('No runs in this folder.')
        else:
            logs=[]
            for session in os.listdir(selected_folder_dir):
                log=eval(U.read_file(U.pjoin(selected_folder_dir,'stream.log')))
                logs.append(log)
            stat_logs(logs)
    else:
        st.info(f'**NOTE:** All settings here will only be applied to this design playground session. The playground session will directly work on the selected running namespace ```{evosys.evoname}```.')



    

def _design_engine(evosys,project_dir):
    st.title("Model Design Engine")

    st.info('**NOTE:** Remember to configure the design engine settings in the config page before running.')


    db_dir = evosys.ptree.db_dir
    design_cfg = {}
    n_sources = {}









def design(evosys,project_dir):

    if 'design_threads' not in st.session_state:
        st.session_state['design_threads'] = {}  # Format: {thread_id: {"thread": thread_object, "stop_event": event_object}}

    if 'design_tab' not in st.session_state:
        st.session_state['design_tab'] = 'design_tunner'

    ### side bar 
    with st.sidebar:
        st.write(f'**Running Namespace:\n```{evosys.evoname}```**')

        if st.button("**Design Engine**" if st.session_state['design_tab']=='design_runner' else "Design Engine",use_container_width=True):
            st.session_state['design_tab'] = 'design_runner'
            st.rerun()
        if st.button("**Design Playground**" if st.session_state['design_tab']=='design_tunner' else "Design Playground",use_container_width=True):
            st.session_state['design_tab'] = 'design_tunner'
            st.rerun()

    if st.session_state['design_tab']=='design_tunner':
        _design_tuning(evosys,project_dir)
    if st.session_state['design_tab']=='design_runner':
        _design_engine(evosys,project_dir)