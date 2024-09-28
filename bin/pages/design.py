import json
import time
import pathlib
import datetime
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import sys,os
sys.path.append('.')
import uuid

import subprocess
import shlex
import psutil


import model_discovery.utils as U
from model_discovery.agents.flow.gau_flows import EndReasons,RunningModes,END_REASONS_LABELS,DesignModes
import bin.app_utils as AU





###########################################################################



def _gen_sess_id():
    return f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{uuid.uuid4().hex[:6]}"

def _run_design_thread(evosys,sess_id=None):
    params_str = shlex.quote(json.dumps(evosys.params))
    if sess_id is None:
        sess_id = _gen_sess_id()
    cmd = f"python -m model_discovery.evolution --mode design --params {params_str} --sess_id {sess_id}"
    process = subprocess.Popen(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    return process,sess_id


def run_design_thread(evosys,sess_id=None):
    polls=[p.poll() for p in st.session_state['design_threads'].values()]
    n_running = sum([p is None for p in polls])
    if n_running < st.session_state['max_design_threads']:
        process,sess_id = _run_design_thread(evosys,sess_id)
        st.session_state['design_threads'][sess_id] = process
        st.toast(f"Design thread launched for {evosys.evoname}.",icon="🚀")
    else:
        st.toast(f"Max number of design threads reached ({st.session_state['max_design_threads']}). Please wait for some threads to finish.",icon="🚨")

def _design_engine(evosys,project_dir):
    st.title("Model Design Engine")

    with st.sidebar:
        st.session_state['max_design_threads'] = st.number_input(label="Max Design Threads",min_value=1,value=3,step=1)
        st.write(f'Controls')
        if st.button("Refresh",key='refresh_btn_design'):
            st.session_state['viewing_log'] = None
            st.rerun()

    
    st.subheader(f"Design Sessions under ```{evosys.evoname}```")
    unfinished_designs,finished_designs = evosys.ptree.get_unfinished_designs(return_finished=True)

    with st.expander("Finished Design Sessions (View details in Viewer)"):
        if len(finished_designs)>0:
            for sess_id in finished_designs:
                sessdata=evosys.ptree.design_sessions[sess_id]
                passed,implemented=evosys.ptree.get_session_state(sess_id)
                cols=st.columns([2.5,1,1,0.1,0.5])
                with cols[0]:
                    st.write(f'Session ID: ```{sess_id}```')
                with cols[1]:
                    state='❌' if passed<sessdata["num_samples"]["proposal"] else '✅'
                    st.write(f'Proposals sampled: ```{passed}/{sessdata["num_samples"]["proposal"]}``` {state}')
                with cols[2]:
                    state='❌' if implemented<sessdata["num_samples"]["implementation"] else '✅'
                    st.write(f'Implementations sampled: ```{implemented}/{sessdata["num_samples"]["implementation"]}``` {state}')
                with cols[4]:
                    if st.button('View Log',key=f'btn_{sess_id}_view_log'):
                        st.session_state['viewing_log'] = sess_id
        else:
            st.info('No finished design sessions.')

    with st.expander("Unfinished Design Sessions (View details in Viewer)",expanded=False):
        if len(unfinished_designs)>0:
            for sess_id in unfinished_designs:
                sessdata=evosys.ptree.design_sessions[sess_id]
                passed,implemented=evosys.ptree.get_session_state(sess_id)
                cols=st.columns([3,3,3,1,1,1])
                with cols[0]:
                    st.write(f'Session ID: ```{sess_id}```')
                with cols[1]:
                    state='❌' if passed<sessdata["num_samples"]["proposal"] else '✅'
                    st.write(f'Proposals Progress: ```{passed}/{sessdata["num_samples"]["proposal"]}``` passed {state}')
                with cols[2]:
                    state='❌' if implemented<sessdata["num_samples"]["implementation"] else '✅'
                    st.write(f'Implementations Progress: ```{implemented}/{sessdata["num_samples"]["implementation"]}``` succeeded {state}')
                with cols[3]:
                    if st.button('View Log',key=f'btn_{sess_id}_view_log'):
                        st.session_state['viewing_log'] = sess_id
                with cols[4]:
                    if st.button('Resume',key=f'btn_{sess_id}_resume'):
                        st.session_state['viewing_log'] = None
                        run_design_thread(evosys,sess_id)
                with cols[5]:
                    delete_btn = st.button('Delete',key=f'btn_{sess_id}_delete',disabled=True)
        else:
            st.success('No unfinished design sessions.')


    st.subheader("Run deisgn thread")
    col1,col2=st.columns([4,1])
    with col1:
        with st.expander("Check current design settings"):
            cols=st.columns(3)
            with cols[0]:
                st.write('**Selection Config**')
                st.write(evosys.select_cfg)
            with cols[1]:
                st.write('**Design Config**')
                st.write(evosys.design_cfg)
            with cols[2]:
                st.write('**Search Config**')
                st.write(evosys.search_cfg)
    with col2:
        new_thread_btn = st.button('***Launch a new design session***',use_container_width=True)

    if new_thread_btn:
        st.session_state['viewing_log'] = None
        run_design_thread(evosys)

    running_process={key:process for key,process in st.session_state['design_threads'].items() if process.poll() is None}
    if len(running_process)>0:
        st.subheader("🐎 *Running Threads*")
        with st.expander("Design Sessions",expanded=True):
            for key,process in running_process.items():
                cols=st.columns([3,2,2,1,1,1])
                sessmeta = U.load_json(U.pjoin(evosys.ptree.session_dir(key), 'metadata.json'))
                passed,implemented=evosys.ptree.get_session_state(key)
                with cols[0]:
                    st.write(f'⏩ Session ID: ```{key}```')
                with cols[1]:
                    state='❎' if passed<sessmeta.get("num_samples",{}).get("proposal",0) else '✅'
                    st.write(f'Proposals progress: ```{passed}/{sessmeta.get("num_samples",{}).get("proposal",0)}``` {state}')
                with cols[2]:
                    state='❎' if implemented<sessmeta.get("num_samples",{}).get("implementation",0) else '✅'
                    st.write(f'Implementations progress: ```{implemented}/{sessmeta.get("num_samples",{}).get("implementation",0)}``` {state}')
                with cols[4]:
                    if st.button('View Log',key=f'btn_{key}_view'):
                        st.session_state['viewing_log'] = key
                with cols[5]:
                    if st.button(f"Terminate",key=f'btn_{key}_term'):
                        try:
                            parent = psutil.Process(process.pid)
                            children = parent.children(recursive=True)
                            for child in children:
                                child.terminate()
                            parent.terminate()
                            
                            gone, alive = psutil.wait_procs(children + [parent], timeout=5)
                            
                            for p in alive:
                                p.kill()
                            
                            st.toast(f"Verification process for {key} terminated.",icon="🛑")
                            del st.session_state['design_threads'][key]
                            st.session_state['viewing_log'] = None
                            time.sleep(0.5)
                            st.rerun()
                        except psutil.NoSuchProcess:
                            st.toast(f"Process for {key} has already ended.",icon="🚨")
                        except Exception as e:
                            st.toast(f"Error terminating process: {str(e)}",icon="🚨")
    else:
        st.info(f'**NOTE:** Remember to configure the design engine settings in the config page before running. The engine will work based on the namespace: ```{evosys.evoname}```')

    if st.session_state['viewing_log']:
        sess_id = st.session_state['viewing_log']
        log_dir = U.pjoin(evosys.ptree.db_dir,'sessions',sess_id,'log')
        logs = []
        for log_file in os.listdir(log_dir):
            if log_file.endswith('.log'):
                logs.append(log_file.split('.')[0])
        logs = logs[::-1] # most recent logs first
        st.write(f'#### Viewing log for session: ```{sess_id}```')
        col1,_,col2,col3=st.columns([5,1,1,1])
        with col1:
            selected_log = st.selectbox(label='Select a log file',options=logs)
            log=load_log(U.pjoin(log_dir,f'{selected_log}.log'))
        with col2:
            st.write('')
            st.write('')
            if st.button('Refresh'):
                st.rerun()
        with col3:
            st.write('')
            st.write('')
            if st.button('Clear'):
                st.session_state['viewing_log'] = None
                st.rerun()
        show_log(log)
    else:
        st.write('')


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
        line = ''
        if in_status:
            status_count += 1
            line += '\t'
        # Use repr() to get a string literal representation
        text = text.replace('/NEWLINE/','\n').replace('/TAB/','\t')
        text_repr = f"'''{text}'''"
        
        if type == 'enter':
            in_status = True
            line += f"with st.status({text_repr}):"
            status_count = 0
        elif type == 'enterspinner':
            in_status = True
            line += f"with st.spinner({text_repr}):"
            status_count = 0
        elif type == 'exit':
            in_status = False
            if status_count==1:
                line += 'st.write("No output inside status.")'
                code.append(line)
            status_count = 0
            continue
        elif type == 'write':
            line += f"st.write({text_repr}, unsafe_allow_html=True)"
        elif type == 'warning':
            line += f"st.warning({text_repr})"
        elif type == 'error':
            line += f"st.error({text_repr})"
        elif type == 'markdown':
            line += f"st.markdown({text_repr}, unsafe_allow_html=True)"
        elif type == 'end':
            line += f"st.write('End with reason: '+{text_repr})"
        elif type == 'balloons':
            line += f"st.balloons()"
        elif type == 'snow':
            line += f"st.snow()"
        else:
            line += f"st.write('{type}'+': '+{text_repr})"
        code.append(line)
    
    final_code = '\n'.join(code)
    exec(final_code)

def load_log(log_file):
    log=[]
    for line in U.read_file(log_file).split('\n'):
        if not line:
            continue
        try:  
            log.append(eval(line))
        except Exception as e:
            log.append((datetime.datetime.now(),f'ERROR IN LOG LINE: {line}\n\n{e}','error'))
    return log


########################################################################################



def _design_tuning(evosys,project_dir):
    ### build the system 
    st.title("Model Design Agents")

    system = evosys.rnd_agent
    db_dir = evosys.ptree.db_dir
    design_cfg = {}
    n_sources = {}

    with st.sidebar:
        st.write('Controls')
        if st.button("Reset design query"):#,use_container_width=True):
            st.rerun()

    #### Configure design

    with st.expander("**Agents Settings**",expanded=True):#,icon='⚙️'):

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
                sess_id=None
                select_cfg={'n_sources':n_sources}
                evosys.design(select_cfg,design_cfg,search_cfg,user_input=user_input,mode=_mode,sess_id=sess_id,resume=resume)
    
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
        st.info(f'**NOTE:** All settings here will only be applied to this design session. The design will be based on and saved to the running namespace ```{evosys.evoname}```.')





def design(evosys,project_dir):

    if 'design_threads' not in st.session_state:
        st.session_state['design_threads'] = {}  

    if 'design_tab' not in st.session_state:
        st.session_state['design_tab'] = 'design_runner'

    if 'max_design_threads' not in st.session_state:
        st.session_state['max_design_threads'] = 5

    if 'viewing_log' not in st.session_state:
        st.session_state['viewing_log'] = None

    ### side bar 
    with st.sidebar:
        AU.running_status(st,evosys)

        if st.button("***Design Engine***" if st.session_state['design_tab']=='design_runner' else "Design Engine",use_container_width=True):
            st.session_state['design_tab'] = 'design_runner'
            st.rerun()
        if st.button("***Design Agents***" if st.session_state['design_tab']=='design_tunner' else "Design Agents",use_container_width=True):
            st.session_state['design_tab'] = 'design_tunner'
            st.rerun()

    if st.session_state['design_tab']=='design_tunner':
        _design_tuning(evosys,project_dir)
    if st.session_state['design_tab']=='design_runner':
        _design_engine(evosys,project_dir)