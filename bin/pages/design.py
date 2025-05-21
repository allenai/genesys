import json
import time
import pathlib
import datetime
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import sys,os
sys.path.append('.')
import uuid
import random

import subprocess
import shlex
import psutil
import time


import model_discovery.utils as U
from model_discovery.agents.flow.gau_flows import EndReasons,RunningModes,AGENT_OPTIONS,\
    END_REASONS_LABELS,DesignModes,DESIGN_TERMINAL_STATES,DESIGN_ACTIVE_STATES,DESIGN_ZOMBIE_THRESHOLD
from model_discovery.system import DEFAULT_AGENTS
import bin.app_utils as AU


DESIGN_LOCK_TIMEOUT = 30 


def do_log(log_ref,log):
    timestamp = time.time()
    backup_ref = log_ref.collection('backup')
    try:
        log_ref.set({str(timestamp): log}, merge=True)
    except Exception as e:
        backup_doc_ref = backup_ref.document(str(timestamp))
        log_original = log_ref.get().to_dict()
        while True:
            try:
                backup_doc_ref.set(log_original)
                break
            except:
                log_original.pop(next(iter(log_original)))
        log_ref.set({str(timestamp): log}) # overwrite the original log
    real_time_utc = datetime.datetime.fromtimestamp(timestamp)
    print(f'[{real_time_utc}] {log}')


# Do not need a lock, as each node will maintain its own design sessions, so
# unless a node is permanently lost (never join the network again, unlikely
# happen), the design sessions will always be finished by resumes. 
def design_command(node_id, evosys, evoname, resume=True, cli=False, cpu_only=False, silent=False, running_sessions=[],sess_id=None):
    sess_id,pid =_design_command(node_id, evosys, evoname, resume, cli, cpu_only, silent, running_sessions,sess_id)
    exp_log_ref = evosys.CM.get_log_ref()
    if sess_id:
        log=f'Node {node_id} running design thread with session id {sess_id}'
        do_log(exp_log_ref,log)
        print('Starting Design Daemon Process...')
        daemon_cmd = f"python -m bin.pages.design --daemon --evoname {evoname} --sess_id {sess_id} --node_id {node_id} --pid {pid}"
        subprocess.Popen(daemon_cmd, shell=True)
    else:
        log=f'Node {node_id} failed to run design thread with error: {pid}'
        do_log(exp_log_ref,log)
    return sess_id,pid

def acquire_design_lock(evosys, node_id):
    exp_log_ref = evosys.CM.get_log_ref()
    lock_ref = evosys.CM.get_design_lock_ref()
    while True:
        lock_doc = lock_ref.get()
        if not lock_doc.exists:
            break
        else:
            lock_data = lock_doc.to_dict()
            if not lock_data['locked']:
                break
            else:
                if lock_data['node_id']==node_id:
                    break
                else:
                    if time.time()-float(lock_data['timestamp'])>DESIGN_LOCK_TIMEOUT:
                        break
        time.sleep(1)
    do_log(exp_log_ref,f'Node {node_id} acquired design lock')

def release_design_lock(evosys,node_id):
    exp_log_ref = evosys.CM.get_log_ref()
    lock_ref = evosys.CM.get_design_lock_ref()
    lock_ref.set({'timestamp':str(time.time()),'locked':False,'node_id':None},merge=True)
    do_log(exp_log_ref,f'Node {node_id} released design lock')

def _design_command(node_id, evosys, evoname, resume=True, cli=False, cpu_only=False, silent=False, running_sessions=[],sess_id=None):
    acquire_design_lock(evosys, node_id)
    sess_id = None
    params = {'evoname': evoname}
    if evosys.evoname != evoname: # FIXME: initialize evosys inside
        evosys.switch_ckpt(evoname) 
    exp_log_ref = evosys.CM.get_log_ref()
    if resume:
        unfinished_designs = set(evosys.ptree.get_unfinished_designs())
        unfinished_designs -= set(running_sessions)
        if len(unfinished_designs) > 0:
            sess_id = random.choice(list(unfinished_designs))
    sess_id,pid = run_design_thread(evosys, sess_id, params, cli=cli, cpu_only=cpu_only, silent=silent)
    if sess_id:
        log=f'Node {node_id} running design thread with session id {sess_id}'
        do_log(exp_log_ref,log)
    else:
        log=f'Node {node_id} failed to run design thread with error: {pid}'
        do_log(exp_log_ref,log)
    time.sleep(3)
    release_design_lock(evosys,node_id)
    return sess_id,pid


def design_daemon(evosys, evoname, sess_id, node_id, pid):
    # monitor the design process by sess_id, kill the process if the session is exited or error or zombie
    if evosys.evoname != evoname: # FIXME: initialize evosys inside
        evosys.switch_ckpt(evoname)
    exp_log_ref = evosys.CM.get_log_ref()
    index_ref,_ = evosys.CM.get_design_sessions_index()
    index_ref.set({sess_id:{'node_id':node_id,'pid':pid}},merge=True)
    # Start heartbeat
    try:
        while True:
            _,status,heartbeat = evosys.CM.get_session_log(sess_id)
            if status is None:
                do_log(exp_log_ref,f'Daemon: Node {node_id} design session {sess_id} not found, wait for it to be created...')
                time.sleep(60)
                continue
            if status in DESIGN_TERMINAL_STATES:
                do_log(exp_log_ref,f'Daemon: Node {node_id} design session {sess_id} terminated with status {status}')
                break
            elif time.time()-float(heartbeat)>DESIGN_ZOMBIE_THRESHOLD:
                    log = f'Daemon: Node {node_id} detected zombie process {pid} for {sess_id}'
                    do_log(exp_log_ref,log)
                    index_ref.set({sess_id:{
                        'status':'ZOMBIE',
                        'timestamp':str(time.time())
                    }},merge=True)
                    break
            else:
                do_log(exp_log_ref,f'Daemon: Node {node_id} design session {sess_id} is running with status {status}')
            time.sleep(60)  # Check every minute for active processes
        try:
            process=psutil.Process(pid)
            process.kill()
            do_log(exp_log_ref,f'Daemon: Node {node_id} forcefully killed design process {pid} for {sess_id}')
        except Exception as e:
            do_log(exp_log_ref,f'Daemon: Node {node_id} failed to forcefully kill design process {pid} for {sess_id}')
            print(f'Error killing process {pid}: {e}')
        
    except Exception as e:
        log = f'Daemon: Node {node_id} encountered an error during design process {pid} on {sess_id}: {str(e)}'
        do_log(exp_log_ref,log)
    finally:
        index_ref.set({sess_id:{
            'status':'TERMINATED',
            'timestamp':str(time.time())
        }},merge=True)
    
    return sess_id, pid


###########################################################################



def _gen_sess_id():
    return f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{uuid.uuid4().hex[:6]}"

def _run_design_thread(evosys,sess_id=None,params=None, cpu_only=False, silent=False):
    if params is None:
        params = evosys.params
    params_str = shlex.quote(json.dumps(params))
    if sess_id is None:
        sess_id = _gen_sess_id()
    cmd = f"python -m model_discovery.evolution --mode design --params {params_str} --sess_id {sess_id}"
    if cpu_only:
        cmd += " --cpu_only"
    if silent:
        cmd += " --silent"
    process = subprocess.Popen(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    return process,sess_id


def run_design_thread(evosys,sess_id=None,params=None,cli=False, cpu_only=False, silent=False):
    if not cli:
        polls=[p.poll() for p in st.session_state['design_threads'].values()]
        n_running = sum([p is None for p in polls])
    if cli or n_running < st.session_state['max_design_threads']:
        process,sess_id = _run_design_thread(evosys,sess_id,params, cpu_only=cpu_only, silent=silent)
        time.sleep(10) # wait for the thread to start and session to be created
        if not cli:
            st.session_state['design_threads'][sess_id] = process
            st.toast(f"Design thread launched for {evosys.evoname}.",icon="🚀")
            # time.sleep(3) # wait for the thread to start and session to be created
            st.session_state['viewing_log'] = None
            # st.rerun()
        else:
            print(f'Success: Design thread launched for {evosys.evoname}.')
        return sess_id,process.pid
    else:
        msg=f"Max number of design threads reached ({st.session_state['max_design_threads']}). Please wait for some threads to finish."
        st.toast(msg,icon="🚨")
        return None,msg

def _design_engine(evosys,project_dir):
    st.title("Design Engine Playground")

    if st.session_state.is_demo:
        st.warning("***Demo mode:** design engine is disabled.*")
    
    if st.session_state.listening_mode:
        st.warning("**WARNING:** You are running in listening mode. If there is an evolution running, running design threads in the same namespace will cause unexpected errors.")

    if st.session_state.evo_running:
        st.warning("**NOTE:** Evolution system is running. Design engine is taken over by the system.")

    with st.sidebar:
        st.session_state['max_design_threads'] = st.number_input(label="Max Design Threads (for here)",min_value=1,value=5,step=1)
        # st.write(f'Controls')
        if st.button("🔄 Refresh",key='refresh_btn_design',use_container_width=True):
            st.session_state['viewing_log'] = None
            st.rerun()

    
    st.subheader(f"Design Sessions in this node")
    unfinished_designs,finished_designs = evosys.ptree.get_unfinished_designs(return_finished=True)

    with st.expander("Finished Design Sessions (braket indicates # of given up challenging implementations)"):
        if len(finished_designs)>0:
            for sess_id in finished_designs:
                sessdata=evosys.ptree.get_design_session(sess_id)
                passed,implemented,challenging,unfinished=evosys.ptree.get_session_state(sess_id)
                cols=st.columns([3,2,3,0.1,1])
                with cols[0]:
                    st.write(f'Session ID: ```{sess_id[:32]}```')
                with cols[1]:
                    state='❌' if len(passed)<sessdata["num_samples"]["proposal"] else '✅'
                    st.write(f'Proposal passed: ```{len(passed)}/{sessdata["num_samples"]["proposal"]}``` {state}')
                with cols[2]:
                    state='❌' if len(implemented)+len(challenging)<sessdata["num_samples"]["implementation"] else '✅'
                    # challenging_state=f'({len(challenging)})' if len(challenging)>0 else ''
                    st.write(f'Implementation succeed: ```{len(implemented)+len(challenging)}({len(challenging)})/{sessdata["num_samples"]["implementation"]}``` {state}')
                with cols[4]:
                    log_dir = U.pjoin(evosys.ptree.db_dir,'sessions',sess_id,'log')
                    if st.button('View Log',key=f'btn_{sess_id}_view_log',disabled=not os.path.exists(log_dir)):
                        st.session_state['viewing_log'] = sess_id
        else:
            st.info('No finished design sessions.')

    with st.expander("Unfinished Design Sessions (braket indicates # of given up challenging implementations)",expanded=False):
        if len(unfinished_designs)>0:
            for sess_id in unfinished_designs:
                sessdata=evosys.ptree.get_design_session(sess_id)
                passed,implemented,challenging,unfinished=evosys.ptree.get_session_state(sess_id)
                cols=st.columns([3,2,3,1,1,1])
                with cols[0]:
                    st.write(f'Session ID: ```{sess_id}```')
                with cols[1]:
                    state='❌' if len(passed)<sessdata["num_samples"]["proposal"] else '✅'
                    st.write(f'Proposal passed: ```{len(passed)}/{sessdata["num_samples"]["proposal"]}``` {state}')
                with cols[2]:
                    state='❌' if len(implemented)+len(challenging)<sessdata["num_samples"]["implementation"] else '✅'
                    # challenging_state=f'({len(challenging)})' if len(challenging)>0 else ''
                    st.write(f'Implementation succeed: ```{len(implemented)+len(challenging)}({len(challenging)})/{sessdata["num_samples"]["implementation"]}``` {state}')
                with cols[3]:
                    log_dir = U.pjoin(evosys.ptree.db_dir,'sessions',sess_id,'log')
                    if st.button('View Log',key=f'btn_{sess_id}_view_log',disabled=not os.path.exists(log_dir)):
                        st.session_state['viewing_log'] = sess_id
                with cols[4]:
                    if st.button('Resume',key=f'btn_{sess_id}_resume',disabled=st.session_state.evo_running):
                        with st.status(f'Resuming session: ```{sess_id}```...'):
                            run_design_thread(evosys,sess_id)
                with cols[5]:
                    delete_btn = st.button('Delete',key=f'btn_{sess_id}_delete',disabled=True)
        else:
            st.success('No unfinished design sessions.')


    st.subheader("Run deisgn thread")
    col1,col2,col3=st.columns([4.2,1,1])
    with col1:
        with st.expander("Check current design settings"):
            _to_check=st.selectbox(label='Select a design config to check',
                    options=['Selection Config','Design Config','Search Config'])
            if _to_check=='Selection Config':
                st.write('**Selection Config**')
                st.write(evosys.select_cfg)
            elif _to_check=='Design Config':
                st.write('**Design Config**')
                st.write(evosys.design_cfg)
            elif _to_check=='Search Config':
                st.write('**Search Config**')
                st.write(evosys.search_cfg)
    with col2:
        rand_resume_btn = st.button('***Random Resume***',use_container_width=True,disabled=len(unfinished_designs)==0 or st.session_state.evo_running or st.session_state.is_demo)
    with col3:
        new_session_btn = st.button('***Launch New Session***',use_container_width=True,disabled=st.session_state.evo_running or st.session_state.is_demo)
        
    if rand_resume_btn:
        sess_id = random.choice(unfinished_designs)
        with st.status(f'Resuming random unfinished session: ```{sess_id}```...'):
            run_design_thread(evosys,sess_id)
    if new_session_btn:
        with st.status('Starting a new design session...'):
            run_design_thread(evosys)

    running_process={key:process for key,process in st.session_state['design_threads'].items() if process.poll() is None}
    if len(running_process)>0:
        st.subheader("🐎 *Running Threads*")
        with st.expander("Design Sessions",expanded=True):
            for key,process in running_process.items():
                cols=st.columns([3,2,2,1,1,1])
                sessmeta = U.load_json(U.pjoin(evosys.ptree.session_dir(key), 'metadata.json'))
                passed,implemented,challenging,unfinished=evosys.ptree.get_session_state(key)
                with cols[0]:
                    st.write(f'⏩ Session ID: ```{key}```')
                with cols[1]:
                    state='❎' if len(passed)<sessmeta.get("num_samples",{}).get("proposal",0) else '✅'
                    st.write(f'Proposals progress: ```{len(passed)}/{sessmeta.get("num_samples",{}).get("proposal",0)}``` {state}')
                with cols[2]:
                    state='❎' if len(implemented)+len(challenging)<sessmeta.get("num_samples",{}).get("implementation",0) else '✅'
                    challenging_state=f'(:red[{len(challenging)}])'
                    st.write(f'Implementations progress: ```{len(implemented)+len(challenging)}{challenging_state}/{sessmeta.get("num_samples",{}).get("implementation",0)}``` {state}')
                with cols[4]:
                    if st.button('View Log',key=f'btn_{key}_view'):
                        st.session_state['viewing_log'] = key
                with cols[5]:
                    if st.button(f"Terminate",key=f'btn_{key}_term',disabled=st.session_state.evo_running):
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
            if st.button('🔄 Refresh',use_container_width=True):
                st.rerun()
        with col3:
            st.write('')
            st.write('')
            if st.button('🧹 Clear',use_container_width=True):
                st.session_state['viewing_log'] = None
                st.rerun()
        show_log(log)
    else:
        st.write('')

    with st.sidebar:
        analyze_btn = st.button('🔍 *Analyze Logs*',use_container_width=True)

    if analyze_btn:
        all_logs,local_sessions = load_logs(U.pjoin(evosys.ptree.db_dir,'sessions'))
        with st.status('Analyzing Logs...',expanded=True):
            st.write(f'###### Design sessions in this node for ```{evosys.evoname}```')
            st.write(f'Total number of sessions: {len(local_sessions)}')
            st.write(f'Total number of logs: {len(all_logs)}')
            stat_logs(all_logs)


#################################################

def load_logs(sess_dir):
    all_logs = []
    local_sessions = os.listdir(sess_dir)
    for sess in local_sessions:
        log_dir = U.pjoin(sess_dir,sess,'log')
        for log_file in os.listdir(log_dir):
            if log_file.endswith('.log'):
                all_logs.append(load_log(U.pjoin(log_dir,log_file)))
    return all_logs,local_sessions


def stat_logs(logs):
    end_reasons = {}
    end_labels = {}
    for _,label in END_REASONS_LABELS.items():
        end_labels[label] = 0
    end_reasons['Unfinished'] = 0
    if len(logs) == 0:
        st.warning('There is no logs detected')
        return
    for log in logs:
        unfinished = True
        for _log in log:
            if _log[2]=='end':
                unfinished=False
                reason = str(_log[1])
                if reason not in end_reasons:
                    end_reasons[reason] = 0
                end_reasons[reason] += 1
        if unfinished:
            end_reasons['Unfinished'] += 1
    
    st.write(end_reasons)

    # end_ratios = {i:num/sum(end_reasons.values()) for i,num in end_reasons.items()}
    # for i in end_reasons:
    #     end_labels[END_REASONS_LABELS[i]] += end_reasons[i]
    # end_label_ratios = {i:num/sum(end_labels.values()) for i,num in end_labels.items()}


    # col1,col2=st.columns([1,1])
    
    # with col1:
    #     st.write('###### Total number of runs: ',sum(end_reasons.values()))
    #     st.write('###### Success rate')
    #     for reason in end_label_ratios:
    #         count = end_labels[reason]
    #         ratio = end_label_ratios[reason]
    #         st.write(f"{reason}: {count} ({ratio*100:.2f}%)")
            
    # with col2:
    #     st.write('###### End reasons')
    #     for reason in end_ratios:
    #         count = end_reasons[reason]
    #         ratio = end_ratios[reason]
    #         st.write(f"{reason.value}: {count} ({ratio*100:.2f}%)")
    
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
        text_repr = f"r'''{text}'''"
        
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
        elif type == 'code':
            line += f"st.code({text_repr},language='python',line_numbers=True)"
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

    if st.session_state.listening_mode:
        st.warning("**WARNING:** You are running in listening mode. Design engine is taken over by the system.")

    if st.session_state.evo_running:
        st.warning("**NOTE:** Evolution system is running. Design engine is taken over by the system.")

    if st.session_state.is_demo:
        st.warning("***Demo mode:** Some search features are disabled, all in simplest setup.*")

    db_dir = evosys.ptree.db_dir
    design_cfg = evosys.design_cfg.copy()
    design_cfg['agent_types']=U.safe_get_cfg_dict(design_cfg,'agent_types',DEFAULT_AGENTS)
    n_sources = {}

    with st.sidebar:
        # st.write('Controls')
        if st.button("🔄 *Reset design query*",use_container_width=True):
            st.rerun()

    #### Configure design

    with st.expander("**Agents Settings**",expanded=True):#,icon='⚙️'):

        col1, col2 = st.columns([1, 5])
        with col1:
            n_seeds = st.number_input(label="Number of seeds",min_value=0,value=1,
                                   help='Number of seed designs, it decides the mode of design, design from scratch: 0 seed, mutation: 1 seed, crossover: >=2 seeds')
            if n_seeds==0:
                mode = DesignModes.SCRATCH
            elif n_seeds==1:
                mode = DesignModes.MUTATION
            else:
                mode = DesignModes.CROSSOVER
        with col2:
            # st.markdown("#### Configure the base models for each agent")
            # AGENT_TYPES = ['claude3.5_sonnet','gpt4o_0806','gpt4o_mini','gpt-4.1-nano']
            agent_type_labels = {
                'DESIGN_PROPOSER':'Proposal Agent',
                'PROPOSAL_REVIEWER':'Proposal Reviewer',
                'IMPLEMENTATION_PLANNER':'Impl. Planner',
                'IMPLEMENTATION_CODER':'Implementation Coder',
                'IMPLEMENTATION_OBSERVER':'Impl. Observer',
                'SEARCH_ASSISTANT': '*Sep. Search Assistant*' # no need at all, can be integrated into search engine
            }
            agent_types = {}
            cols = st.columns(len(agent_type_labels))
            for i,agent in enumerate(agent_type_labels):
                with cols[i]:
                    options=AGENT_OPTIONS[agent].copy()
                    options += ['gpt-4.1-nano']
                    if agent in ['IMPLEMENTATION_PLANNER','IMPLEMENTATION_OBSERVER']:
                        options+=['None']
                    options+=['hybrid']
                    index = options.index(design_cfg['agent_types'][agent]) if design_cfg['agent_types'][agent] in options else 0

                    if st.session_state.is_demo:
                        _disabled = agent not in ['IMPLEMENTATION_PLANNER','IMPLEMENTATION_OBSERVER']
                        if agent in ['IMPLEMENTATION_PLANNER','IMPLEMENTATION_OBSERVER']:
                            options = ['gpt-4.1-nano','None']
                        if agent!='SEARCH_ASSISTANT':
                            index = options.index('gpt-4.1-nano')
                    else:
                        _disabled= agent=='SEARCH_ASSISTANT'
                    agent_types[agent] = st.selectbox(label=agent_type_labels[agent],options=options,index=index,disabled=_disabled)
            design_cfg['agent_types'] = agent_types
        if any(['hybrid' in design_cfg['agent_types'][i] for i in design_cfg['agent_types']]):
            st.caption('***NOTE:** Hybrid agent is a weighted random selection of the options at the beginning of design. It will apply the default weights which will be presented later when it runs.*')

        sources = ['ReferenceCoreWithTree', 'DesignArtifactImplemented', 'DesignArtifact', 'ReferenceCore', 'ReferenceWithCode', 'Reference']
        if st.session_state.use_cache:
            sources={i:len(st.session_state.filter_by_types[i]) for i in sources}
        else:
            sources={i:len(evosys.ptree.filter_by_type(i)) for i in sources}
        
        _scol1,_scol2 = st.columns([5,1])
        with _scol1:
            st.markdown("##### Configure the number of *References* to sample from each source")
            cols = st.columns(len(sources))
            for i,source in enumerate(sources):
                with cols[i]:
                    value=min(1,sources[source])
                    display_name = f'{source} ({sources[source]})'
                    if source=='ReferenceCoreWithTree':
                        display_name = f'RefCoreWithTree ({sources[source]})'
                        value=0
                    elif source=='DesignArtifactImplemented':
                        display_name = f'DesignArtImp ({sources[source]})'
                        value=0
                    elif source=='ReferenceWithCode':
                        display_name = f'RefWithCode ({sources[source]})'
                    n_sources[source] = st.number_input(label=display_name,min_value=0,value=value,disabled=st.session_state.is_demo)#,max_value=1,disabled=True)
        with _scol2:
            # st.markdown("##### Input Settings")
            design_cfg['crossover_no_ref'] = st.checkbox("Crossover no ref",value=design_cfg.get('crossover_no_ref',True),
                help='If true, will not use references in crossover mode, it is recommended as crossover does not need cold start, and context length can be over long.')
            design_cfg['mutation_no_tree'] = st.checkbox("Mutation no tree",value=design_cfg.get('mutation_no_tree',True),
                help='If true, will not show full tree but only the document for types with tree (i.e., ReferenceCoreWithTree, DesignArtifactImplemented) in mutation mode, it is recommended as context length can be over long.')
            design_cfg['scratch_no_tree'] = st.checkbox("Scratch no tree",value=design_cfg.get('scratch_no_tree',False),
                help='If true, will not show full tree but only the document for types with tree (i.e., ReferenceCoreWithTree, DesignArtifactImplemented) in scratch mode, it is recommended as context length can be over long.')

        col1,col2=st.columns([3,2])
        termination={}
        threshold={}
        max_attempts = {}
        with col1:
            st.markdown("##### Configure termination conditions and budgets (0 is no limit)")
            cols=st.columns(4)
            with cols[0]:
                _value = 1 if not st.session_state.is_demo else 3
                termination['max_failed_rounds'] = st.number_input(label="Max failed rounds",min_value=1,value=_value,disabled=st.session_state.is_demo)
            with cols[1]:
                _value = 0.05 if st.session_state.is_demo else 0.0
                termination['max_total_budget'] = st.number_input(label="Max total budget",min_value=0.0,value=_value,disabled=st.session_state.is_demo)
            with cols[2]:
                termination['max_debug_budget'] = st.number_input(label="Max debug budget",min_value=0.0,value=0.0,disabled=st.session_state.is_demo)
            with cols[3]:
                _value = 3 if not st.session_state.is_demo else 1
                max_attempts['max_search_rounds'] = st.number_input(label="Max search rounds",min_value=0,value=_value,disabled=st.session_state.is_demo)
        with col2:
            st.markdown("##### Configure the threshold for rating the design")
            cols=st.columns(2)
            with cols[0]:
                threshold['proposal_rating'] = st.slider(label="Proposal rating",min_value=0,max_value=5,value=1,disabled=st.session_state.is_demo)
            with cols[1]:
                threshold['implementation_rating'] = st.slider(label="Implementation rating",min_value=0,max_value=5,value=1,disabled=st.session_state.is_demo)
        design_cfg['termination'] = termination
        design_cfg['threshold'] = threshold 


        col1,col2,col3=st.columns([4,2,5])
        with col1:
            st.markdown("##### Configure max number of attempts")
            cols=st.columns(3)
            with cols[0]:
                max_attempts['design_proposal'] = st.number_input(label="Proposal attempts",min_value=3,value=3,disabled=st.session_state.is_demo)
            with cols[1]:
                max_attempts['implementation_debug'] = st.number_input(label="Debug attempts",min_value=3,value=3,disabled=st.session_state.is_demo)
            with cols[2]:
                max_attempts['post_refinement'] = st.number_input(label="Post refinements",min_value=0,value=0,disabled=st.session_state.is_demo)
        design_cfg['max_attemps'] = max_attempts

        with col2:
            # st.markdown("##### Other settings")
            design_cfg['use_unlimited_prompt']=st.checkbox('Use unlimited prompt',value=design_cfg.get('use_unlimited_prompt',False),disabled=st.session_state.is_demo)
            design_cfg['unittest_pass_required']=st.checkbox('Unittests pass required',value=design_cfg.get('unittest_pass_required',False))
                # help='Whether require the coder to pass self-generated unit tests before the code is accepted.')
            design_cfg['no_f_checkers']=st.checkbox('No F-Checkers',value=design_cfg.get('no_f_checkers',False),
                help='If true, will turn off the Functional Checkers when checking the implementation code. Only applicable in benchmark mode.')
            
        with col3:
            color=AU.theme_aware_options(st,'orange','violet','violet')
            st.markdown(f"##### :{color}[*View previous runs*]")
            cols=st.columns([2,2,2,1])
            with cols[0]:
                ckpts=[i for i in os.listdir(evosys.ckpt_dir) if i!='.node.json']
                current_ckpt = evosys.evoname
                selected_ckpt = st.selectbox(label="Select folder",options=ckpts,index=ckpts.index(current_ckpt),disabled=st.session_state.is_demo)
                db_dir = U.pjoin(evosys.ckpt_dir,selected_ckpt,'db')
            with cols[1]:
                folders=[]
                if os.path.exists(U.pjoin(db_dir,'sessions')):
                    for i in os.listdir(U.pjoin(db_dir,'sessions')):
                        if os.path.isdir(U.pjoin(db_dir,'sessions',i,'log')):
                            folders.append(i)
                selected_folder = st.selectbox(label="Select session",options=folders)
                if selected_folder:
                    selected_folder_dir = U.pjoin(db_dir,'sessions',selected_folder,'log')
                else:
                    selected_folder_dir = None
            with cols[2]:
                design_logs=['']
                if selected_folder and os.path.exists(selected_folder_dir):
                    design_logs += [i for i in os.listdir(selected_folder_dir) if i.endswith('.log')]
                folder_name = selected_folder if selected_folder else 'No folder selected'
                selected_design_log = st.selectbox(label=f"Select log file",options=design_logs)
                if selected_design_log:
                    selected_design_log_path = U.pjoin(selected_folder_dir,selected_design_log)
            with cols[3]:
                st.write('')
                st.write('')
                view_log_btn = st.button("*View*",disabled=selected_design_log=='')

    with st.expander("Simplified Search Settings",expanded=False):
        search_cfg={}
        search_cfg['result_limits']={}
        search_cfg['perplexity_settings']={}
        search_cfg['proposal_search_cfg']={}

        cols=st.columns([2,2,2,3,2,3])
        with cols[0]:
            search_cfg['result_limits']['lib']=st.number_input("Library Primary",value=5,min_value=0,step=1,disabled=st.session_state.is_demo)
        with cols[1]:
            search_cfg['result_limits']['lib2']=st.number_input("Library Secondary",value=0,min_value=0,step=1,disabled=st.session_state.is_demo)
        with cols[2]:
            search_cfg['result_limits']['libp']=st.number_input("Library Plus",value=0,min_value=0,step=1,disabled=st.session_state.is_demo)
        with cols[3]:
            _value = 0.2 if not st.session_state.is_demo else 0.0
            search_cfg['rerank_ratio']=st.slider("Rerank Scale Ratio (0 is disabled)",min_value=0.0,max_value=1.0,value=_value,step=0.01,disabled=st.session_state.is_demo)
        with cols[4]:
            _value = 3 if not st.session_state.is_demo else 0
            search_cfg['proposal_search_cfg']['top_k']=st.number_input("Proposal Top K",value=_value,min_value=0,step=1,disabled=st.session_state.is_demo)
        with cols[5]:
            search_cfg['proposal_search_cfg']['cutoff']=st.slider("Proposal Search Cutoff",min_value=0.0,max_value=1.0,value=0.5,step=0.01,disabled=st.session_state.is_demo)

        cols=st.columns([2,2,2,2,2])
        with cols[0]:
            search_cfg['result_limits']['s2']=st.number_input("S2 Search Result Limit",value=5,min_value=0,step=1,disabled=st.session_state.is_demo)
        with cols[1]:
            search_cfg['result_limits']['arxiv']=st.number_input("Arxiv Search Result Limit",value=3,min_value=0,step=1,disabled=st.session_state.is_demo)
        with cols[2]:
            search_cfg['result_limits']['pwc']=st.number_input("PwC Result Limit",value=3,min_value=0,step=1,disabled=st.session_state.is_demo)
        with cols[3]:
            _index = 2 if not st.session_state.is_demo else 0
            search_cfg['perplexity_settings']['model_size']=st.selectbox("Perplexity Model Size",options=['none','small','large','huge'],index=_index,disabled=st.session_state.is_demo)
        with cols[4]:
            search_cfg['perplexity_settings']['max_tokens']=st.number_input("Perplexity Max Tokens",value=4000,min_value=500,step=100,disabled=search_cfg['perplexity_settings']['model_size']=='none')
       

    #### Run design

    EXPERIMENT_RUNS=1
    
    # cols = st.columns([7,2.5,1.8,1.2])
    cols = st.columns([5,1.5,2,0.8,1.3,1.2])
    with cols[0]:
        user_input = st.text_input(label = "Add any additional instructions (optional)", help='Will be combined with selector\'s instructions (if any)')
    with cols[1]:
        _options = ['None']
        if st.session_state.use_cache:
            _options += list(st.session_state.filter_by_types['ReferenceCoreWithTree']) + list(st.session_state.filter_by_types['DesignArtifactImplemented'])
        else:
            _options += list(evosys.ptree.filter_by_type(['ReferenceCoreWithTree','DesignArtifactImplemented']))
        _options = sorted(_options)
        manual_seed = st.selectbox(label="Manual seed",options=_options,
            help='Will override selector\'s selection')
    with cols[2]:
        # EXPERIMENT_RUNS = st.number_input(label="Number of design runs",min_value=1,value=1,disabled=True)
        manual_refs = st.text_input(label="Manual References",value='None',help='Comma separated ids, will override selector\'s recommendations')
    with cols[3]:
        st.write('')  
        st.write('')
        if st.session_state.is_demo and AU.daily_usage_status(st)>=1.0:
            submit = st.button(label="***Run***",use_container_width=True,disabled=True)
            st.warning("**NOTE:** Daily limit of the demo reached. Please try again tomorrow.")
        else:
            submit = st.button(label="***Run***",use_container_width=True)

    with cols[4]:
        st.write('')
        st.write('')
        resume = st.checkbox(label="Resume",value=True, help='If checked, will randomly resume the unfinished design session if any.')
    with cols[5]:
        st.write('')
        st.write('')
        _is_naive = design_cfg.get('flow_type','gau')=='naive'
        design_cfg['flow_type'] = 'naive' if st.checkbox(label="Naive",value=_is_naive,help='Use the Naive GAB Coder and Observer') else 'gau'


    if submit:
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
                st.write(f"Design Mode:  {_mode}")
                sess_id=None
                select_cfg=evosys.select_cfg.copy()
                select_cfg['n_sources']=n_sources
                manual_seed = None if manual_seed == 'None' else manual_seed
                manual_refs = None if manual_refs == 'None' else manual_refs.split(',')
                cpu_only = st.session_state.is_demo
                evosys.design(select_cfg,design_cfg,search_cfg,user_input=user_input,n_seeds=n_seeds,
                    sess_id=sess_id,resume=resume,cpu_only=cpu_only)
    
    elif view_log_btn:
        show_log(load_log(selected_design_log_path))

    else:
        st.info(f'**NOTE:** All settings here will only be applied to this design session. The design will be based on and saved to the running namespace ```{evosys.evoname}```.')

    
    # elif selected_folder:
    #     with st.empty():
    #         st.subheader(f'Viewing statistics for experiment folder: *{selected_folder}*')
    #     if not U.pexists(selected_folder_dir):
    #         st.write('No runs in this folder.')
    #     else:
    #         logs=[]
    #         for session in os.listdir(selected_folder_dir):
    #             log=eval(U.read_file(U.pjoin(selected_folder_dir,'stream.log')))
    #             logs.append(log)
    #         stat_logs(logs)


def design(evosys,project_dir):

    if 'design_tab' not in st.session_state:
        st.session_state['design_tab'] = 'design_runner'

    if 'viewing_log' not in st.session_state:
        st.session_state['viewing_log'] = None

    ### side bar 
    with st.sidebar:
        AU.running_status(st,evosys)
        # st.write('Select a playground')
        # # btn_text = "***Design Engine***" if st.session_state['design_tab']=='design_runner' else "Design Engine"
        # btn_text = "Design Engine"
        # if st.button(btn_text,use_container_width=True):
        #     st.session_state['design_tab'] = 'design_runner'
        #     # st.rerun()
        # # btn_text = "***Design Agents***" if st.session_state['design_tab']=='design_tunner' else "Design Agents"
        # btn_text = "Design Agents"
        # if st.button(btn_text,use_container_width=True):
        #     st.session_state['design_tab'] = 'design_tunner'
        #     # st.rerun()
        choose_mode=st.selectbox("Sub-tabs",options=['Design Agents','Design Engine'],index=0)

    # if st.session_state['design_tab']=='design_tunner':
    #     _design_tuning(evosys,project_dir)
    # if st.session_state['design_tab']=='design_runner':
    #     _design_engine(evosys,project_dir)
    if choose_mode=='Design Engine':
        _design_engine(evosys,project_dir)
    elif choose_mode=='Design Agents':
        _design_tuning(evosys,project_dir)



if __name__ == '__main__':
    from model_discovery.evolution import BuildEvolution
    import argparse

    AU.print_cli_title()

    setting=AU.get_setting()
    default_namespace=setting.get('default_namespace','test_evo_000')

    parser = argparse.ArgumentParser()
    parser.add_argument("--evoname", default=default_namespace, type=str)
    parser.add_argument("--resume", action='store_true') # the name of the whole evolution
    parser.add_argument("--daemon", action='store_true')
    parser.add_argument("--sess_id", default=None, type=str)
    parser.add_argument("--node_id", default=None, type=str)
    parser.add_argument("--pid", default=None, type=int)
    args = parser.parse_args()

    args.sess_id = None if args.sess_id == 'None' else args.sess_id
    args.node_id = None if args.node_id == 'None' else args.node_id
    args.pid = None if args.pid == 'None' else args.pid
    
    if args.daemon:
        evosys = BuildEvolution(
            params={'evoname':args.evoname,'db_only':True,'no_agent':True}, 
            do_cache=False,
        )
        design_daemon(evosys,args.evoname,args.sess_id,args.node_id,args.pid)
    else:
        evosys = BuildEvolution(
            params={'evoname':args.evoname}, 
            do_cache=False,
        )
        node_id= args.node_id if args.node_id else str(uuid.uuid4())[:8]
        design_command(node_id,evosys,args.evoname,resume=args.resume,cli=True)
    