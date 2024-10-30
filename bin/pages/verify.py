import json
import time
import pathlib
import copy
import streamlit as st
import sys
import os
import torch
import platform
import psutil
import subprocess
import pytz
import shlex
import numpy as np
import select
import pandas as pd
import multiprocessing
import random
from datetime import datetime, timedelta
import uuid
import subprocess
from google.cloud.firestore import DELETE_FIELD

import pynvml



sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)


from model_discovery.ve.data_loader import load_datasets
from model_discovery.configs.const import TARGET_SCALES, SMOLLM_125_CORPUS,DEFAULT_TOKEN_MULTS,\
    VERIFY_TERMINAL_STATES,VERIFY_ACTIVE_STATES,VERIFY_ZOMBIE_THRESHOLD,LOCK_ZOMBIE_THRESHOLD
from model_discovery.agents.agent_utils import OPENAI_COSTS_DICT, ANTHROPIC_COSTS_DICT




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
    real_time_utc = datetime.utcfromtimestamp(timestamp)
    print(f'[{real_time_utc}] {log}')


def update_local_doc(evosys, sess_id, status, delete=False):
    local_doc_dir = U.pjoin(evosys.ckpt_dir, '.node.json')
    local_doc = U.load_json(local_doc_dir)
    if 'running_verifies' not in local_doc:
        local_doc['running_verifies'] = {}
    if sess_id not in local_doc['running_verifies']:
        local_doc['running_verifies'][sess_id] = {}
    if delete:
        local_doc['running_verifies'].pop(sess_id)
    else:
        local_doc['running_verifies'][sess_id]['status'] = status
        local_doc['running_verifies'][sess_id]['last_heartbeat'] = str(time.time())
    U.save_json(local_doc, local_doc_dir)

def check_local_availability(evosys):
    local_doc_dir = U.pjoin(evosys.ckpt_dir, '.node.json')
    local_doc = U.load_json(local_doc_dir)
    running_verifies = local_doc.get('running_verifies',{})
    to_pop = []
    for sess_id in running_verifies:
        if time.time()-float(running_verifies[sess_id]['last_heartbeat']) > VERIFY_ZOMBIE_THRESHOLD:
            to_pop.append(sess_id)
    for sess_id in to_pop:
        running_verifies.pop(sess_id)
    if to_pop:
        local_doc['running_verifies'] = running_verifies
        U.save_json(local_doc, local_doc_dir)
    return len(running_verifies)==0

def verify_command(node_id, evosys, evoname, design_id=None, scale=None, resume=True, cli=False, RANDOM_TESTING=False, accept_baselines=False, free_verifier=False):
    if not check_local_availability(evosys):
        st.error('There is already a verification job running. Please wait for it to finish.')
        return None, None
    
    sess_id, pid, _design_id, _scale = _verify_command(node_id, evosys, evoname, design_id, scale, resume, cli, 
        ret_id=True, RANDOM_TESTING=RANDOM_TESTING, accept_baselines=accept_baselines, free_verifier=free_verifier)
    exp_log_ref = evosys.CM.get_log_ref()
    
    if sess_id:
        log = f'Node {node_id} running verification on {_design_id}_{_scale}'
        do_log(exp_log_ref,log)
        # Start the daemon in a separate process
        print('Starting Verify Daemon Process...')
        daemon_cmd = f"python -m bin.pages.verify --daemon --evoname {evoname} --sess_id {sess_id} --design_id {_design_id} --scale {_scale} --node_id {node_id} --pid {pid}"
        subprocess.Popen(daemon_cmd, shell=True)
    else:
        log = f'Node {node_id} failed to run verification on {design_id}_{scale} with error: {pid}'
        do_log(exp_log_ref,log)

    return sess_id, pid

# python -m bin.pages.verify --daemon --evoname {evoname} --sess_id {sess_id} --design_id {design_id} --scale {scale} --node_id {node_id} --pid {pid}


def _verify_command(node_id, evosys, evoname, design_id=None, scale=None, resume=True, cli=False,
                    ret_id=False, RANDOM_TESTING=False, accept_baselines=False, free_verifier=False):
    
    if not RANDOM_TESTING:
        if evosys.evoname != evoname:
            evosys.switch_ckpt(evoname)
        
        verify_ref = evosys.remote_db.collection('verifications').document(evoname)

        # Acquire lock
        while True:
            doc = verify_ref.get().to_dict()
            current_time = datetime.now()

            acquireable = False
            if doc is None:
                acquireable = True
            elif 'lock' not in doc:
                acquireable = True
            else:
                lock_info = doc['lock']
                if lock_info['locked']==False:
                    acquireable = True
                else:
                    if lock_info['node_id'] == node_id:
                        acquireable = True
                    else:
                        lock_time = lock_info['timestamp'].replace(tzinfo=None)  # Remove timezone info for comparison
                        if (current_time - lock_time).total_seconds() > LOCK_ZOMBIE_THRESHOLD:
                            acquireable = True
            
            if acquireable:
                verify_ref.set({
                    'lock': {'locked': True, 'node_id': node_id, 'timestamp': current_time}
                }, merge=True)
                break
            else:
                time.sleep(1)  # Wait before trying again
                    

        doc = verify_ref.get().to_dict()
        timestamp = datetime.now()
        try:
            verifying_dict = evosys.CM.get_running_verifications() # sess_ids: design_id_scale
            unfinished_verifies = evosys.get_unfinished_verifies() # local folders
            available_resumes = [v for v in unfinished_verifies if v not in verifying_dict]
            if design_id is None or scale is None:
                doc = verify_ref.get().to_dict()

                if resume and available_resumes:
                    exp = random.choice(available_resumes)
                    design_id, scale = U.parse_verify_id(exp)
                else:
                    exclude_list = []
                    for key in verifying_dict:
                        design_id, scale = U.parse_verify_id(key)
                        exclude_list.append((design_id,scale))  
                    print(f'Selecting with exclude_list: {exclude_list}')
                    design_id, scale = evosys.selector.select_verify(exclude_list=exclude_list, accept_baselines=accept_baselines, free_verifier=free_verifier)
                    if design_id is None:
                        msg = "No unverified design found at any scale."
                        if not cli:
                            st.error(msg)
                        if ret_id:
                            return None, msg, None, None
                        else:
                            return None, msg
                    else:
                        print(f'$$$ Selected design: {design_id}_{scale}')

        finally:
            # create index term
            sess_id = f'{design_id}_{scale}'
            index_ref,_ = evosys.CM.get_verifications_index()
            index_ref.set({sess_id:{
                'timestamp':str(time.time()),
                'status':'RUNNING',
                'latest_log':None,
                'node_id':node_id,
            }},merge=True)
            # Release lock
            time.sleep(random.randint(1,10)) 
            verify_ref.update({'lock': {'locked': False, 'node_id': None}})


    params = {'evoname': evoname}
    sess_id, pid = run_verification(params, design_id, scale, resume, cli=cli, RANDOM_TESTING=RANDOM_TESTING)
    if ret_id:
        return sess_id, pid, design_id, scale
    else:
        return sess_id, pid


def verify_daemon(evoname, evosys, sess_id, design_id, scale, node_id, pid):
    exp_log_ref = evosys.CM.get_log_ref()
    index_ref,_ = evosys.CM.get_verifications_index()
    index_ref.set({sess_id:{'node_id':node_id,'pid':pid}},merge=True)

    # Start heartbeat
    # verify_ref.set({'heartbeats': {node_id: datetime.now()}}, merge=True)
    
    try:
        while True:
            _,status,heartbeat = evosys.CM.get_verification_log(sess_id)
            if status is None:
                do_log(exp_log_ref,f'Daemon: Node {node_id} verification {sess_id} not found, wait for it to be created...')
                time.sleep(60)
                continue
            if status in VERIFY_TERMINAL_STATES:
                do_log(exp_log_ref,f'Daemon: Node {node_id} verification {sess_id} terminated with status {status}')
                update_local_doc(evosys, sess_id, status, delete=True)
                break
            elif time.time()-float(heartbeat)>VERIFY_ZOMBIE_THRESHOLD:
                    log = f'Daemon: Node {node_id} detected zombie process {pid} for {sess_id}'
                    do_log(exp_log_ref,log)
                    index_ref.set({sess_id:{
                        'status':'ZOMBIE',
                        'timestamp':str(time.time())
                    }},merge=True)
                    update_local_doc(evosys, sess_id, 'ZOMBIE', delete=True)
                    break
            else:
                if status == 'TRAINING':
                    # detect if cuda has any process running
                    is_training = False
                    for i in range(3):
                        gpu_info = torch.cuda.get_device_properties(0)
                        # BETA
                        if gpu_info.memory_used < 0.5 * 1024 ** 3: # 0.5GB, if less than 0.5GB, then it may not training actively
                            do_log(exp_log_ref,f'Daemon: Node {node_id} verification {sess_id} is training but no GPU memory used, checking times {i+1}...')
                            time.sleep(10)
                        else:
                            is_training = True
                            break
                    if not is_training:
                        do_log(exp_log_ref,f'Daemon: Node {node_id} verification {sess_id} is training but no GPU memory used, terminating...')
                        break
                else:
                    do_log(exp_log_ref,f'Daemon: Node {node_id} verification {sess_id} is running with status {status}')
                    update_local_doc(evosys, sess_id, status)
            time.sleep(60)  # Check every minute for active processes

        # Check if the process completed successfully
        try: 
            process=psutil.Process(pid)
            process.kill()
            update_local_doc(evosys, sess_id, 'TERMINATED', delete=True)
            do_log(exp_log_ref,f'Daemon: Node {node_id} forcefully killed verification process {pid} for {sess_id}')
        except Exception as e:
            update_local_doc(evosys, sess_id, 'TERMINATED', delete=True)
            do_log(exp_log_ref,f'Daemon: Node {node_id} failed to forcefully kill verification process {pid} for {sess_id}')
            print(f'Error killing process {pid}: {e}')

    except Exception as e:
        log = f'Daemon: Node {node_id} encountered an error during verification process {pid} on {design_id}_{scale}: {str(e)}'
        do_log(exp_log_ref,log)
        update_local_doc(evosys, sess_id, 'TERMINATED', delete=True)
    
    finally:
        # Ensure verification is marked as complete even if an exception occurred
        index_ref.set({sess_id:{
            'status':'TERMINATED',
            'timestamp':str(time.time())
        }},merge=True)
        update_local_doc(evosys, sess_id, 'TERMINATED', delete=True)

    return sess_id, pid




def get_system_info():
    cpu_info = {
        'Processor': platform.processor(),
        'Machine': platform.machine(),
        'Platform': platform.platform(),
        'System': platform.system(),
        'Version': platform.version(),
        'Release': platform.release(),
        'Python Version': platform.python_version(),
    }
    # GPU Information
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info['gpu_count'] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            gpu_info.update({
                'Device 0': torch.cuda.get_device_name(i),
                'Memory': f"{torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB",
                'Capability': torch.cuda.get_device_capability(i),
            })
            break
    else:
        gpu_info = {'GPU': 'Not Available'}
    # Memory Information
    mem_info = {
        'Total Memory': f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
        'Available Memory': f"{psutil.virtual_memory().available / (1024 ** 3):.2f} GB",
        'Used Memory': f"{psutil.virtual_memory().used / (1024 ** 3):.2f} GB",
        'Memory Percent': f"{psutil.virtual_memory().percent:.2f} %",
    }
    return cpu_info, gpu_info, mem_info


# Only one verification can be running at a time
def _run_verification(params, design_id, scale, resume, cli=False, prep_only=False, RANDOM_TESTING=False):
    params_str = shlex.quote(json.dumps(params))
    if not RANDOM_TESTING:
        CKPT_DIR = os.environ.get('CKPT_DIR')
        local_doc_dir = U.pjoin(CKPT_DIR, '.node.json')
        local_doc = U.load_json(local_doc_dir)
        local_doc['model_ready'] = False
        U.save_json(local_doc, local_doc_dir)
        cmd = f"python -m model_discovery.evolution --mode prep_model --params {params_str} --design_id {design_id} --scale {scale}"
        if cli:
            print('Preparing Model...')
            process = subprocess.Popen(cmd, shell=True)
        else:
            with st.spinner(f'Preparing Model...'):
                process = subprocess.run(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

        if prep_only:
            return process
        
        while not U.load_json(local_doc_dir)['model_ready']:
            time.sleep(1)

    # FIXME: should wait until the model is ready
    cmd = f"python -m model_discovery.evolution --mode verify --params {params_str} --design_id {design_id} --scale {scale}"
    if resume:
        cmd+=' --resume'
    if RANDOM_TESTING:
        cmd+=' --RANDOM_TESTING'
    if cli:
        print(f'Launching Verification with command:\n```{cmd}```')
        process = subprocess.Popen(cmd, shell=True)
        # process = subprocess.Popen('echo "Hello World"', shell=True)
    else:
        st.write(f'Launching Verification with command:\n```{cmd}```')
        with st.spinner('Running... Please check the console for verification progress.'):
            # process = subprocess.run(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
            process = subprocess.Popen(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    return process

def run_verification(params, design_id, scale, resume, cli=False, prep_only=False, RANDOM_TESTING=False):
    key = f"{design_id}_{scale}"
    # if key not in st.session_state['running_verifications']:
    if not cli: 
        polls=[p.poll() for p in st.session_state['running_verifications'].values()]
    if cli or (not None in polls):
        params = copy.deepcopy(params)
        process = _run_verification(params, design_id, scale, resume, cli=cli, prep_only=prep_only, RANDOM_TESTING=RANDOM_TESTING)
        if prep_only:
            return None,'Prepare only (for testing)'
        if not cli:
            st.session_state['running_verifications'][key] = process
            # st.session_state['output'][key] = []
            st.success(f"Verification process started for {design_id} on scale {scale}. Check console for output.")
        else:
            print(f"Success: Verification process started for {design_id} on scale {scale}. Check console for output.")
        return key,process.pid
    else:
        key=list(st.session_state['running_verifications'].keys())[0]
        design_id,scale=key.split('_')
        msg=f"A verification process for {design_id} on scale {scale} is already running."
        st.warning(msg)
        return None,msg


def stream_output(process, key):
    if hasattr(process, 'stdout'):
        for line in process.stdout:
            # st.session_state['output'][key].append(line)
            st.code(line)
        
    if hasattr(process, 'stderr'):
        for line in process.stderr:
            # st.session_state['output'][key].append(f"ERROR: {line}")
            st.code(f"ERROR: {line}")


def verify_engine(evosys,project_dir):
    
    st.title("Verification Engine")

    DISABLE_VERIFICATION=False
    if st.session_state.listening_mode and st.session_state.listener.accept_verify_job:
        st.warning("**WARNING:** You are running a listener with GPUs. Verification engine is taken over by the system.")
        DISABLE_VERIFICATION=True

    if st.session_state.evo_running:
        st.warning("**NOTE:** You are running as the master node. Verification engine is taken over by the system.")


    # evosys.ptree.load()
    
    # if 'output' not in st.session_state:
    #     st.session_state['output'] = {}


    with st.sidebar:
        with st.expander("View CPU stats"):
            cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
            if st.button("Refresh",key='refresh_btn_cpu'):
                cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
            cpu_df = pd.DataFrame(cpu_percentages, columns=["Usage (%)"], index=[f"Core {i}" for i in range(len(cpu_percentages))])
            st.dataframe(cpu_df)
                
        # Add a refresh button to manually update the page
        if st.button('🔄 Refresh',key='refresh_btn_engine',use_container_width=True):
            st.rerun()



    st.subheader("System Diagnostics")
    col1, col2, col3, col4 = st.columns(4)
    cpu_info, gpu_info, mem_info = get_system_info()
    with col1:
        with st.expander("Platform Info"):
            st.write(cpu_info)
    with col2:
        with st.expander("GPU Info"):
            st.write(gpu_info)
    with col3:
        with st.expander("Memory Info"):
            st.write(mem_info)
            if st.button("Refresh",key='refresh_btn_mem'):
                cpu_info, gpu_info, mem_info = get_system_info()
    with col4:
        with st.expander("Experiment Info"):
            st.write(f'Namespace: ```{evosys.evoname}```')
            st.write(evosys.params)

    Col1,Col2=st.columns([5,4])
    
    with Col1:
        with st.expander("NVIDIA-SMI Monitor"):
            def get_nvidia_smi_output():
                try:
                    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
                    return result.stdout.decode('utf-8')
                except Exception as e:
                    return f"Error: {e}"
            _output=get_nvidia_smi_output()
            if st.button("Refresh",key='refresh_btn'):
                _output=get_nvidia_smi_output()
            st.text(_output)


    with Col2:
        disable_diagnostic_panel=True
        with st.expander("Diagnostic Panel (For internal use)"):
            # st.subheader("Test loading datasets")
            config=GAMConfig_14M()
            st.subheader("Test dataset loader")
            col1,col2,col3=st.columns([3,2,2])
            with col1:
                dataset_name=st.selectbox("Choose a Dataset",options=SMOLLM_125_CORPUS,disabled=disable_diagnostic_panel)
            with col2:
                st.write('')
                st.write('')
                if st.button("Test Loading",use_container_width=True,disabled=disable_diagnostic_panel):
                    st.spinner('Loading... Please check the console for output')
                    config.training_data=[dataset_name]
                    load_datasets(config)
            with col3:
                st.write('')
                st.write('')
                if st.button("Load all",use_container_width=True,disabled=disable_diagnostic_panel):
                    st.spinner('Loading... Please check the console for output')
                    config.training_data=SMOLLM_125_CORPUS
                    load_datasets(config)




    ##################################################################################

    st.header("Verify Designs")
    col1,_,col2,_,col3,_,col4,col5,col6=st.columns([1,0.05,0.9,0.05,1.3,0.05,0.6,0.5,0.4])
    with col1:
        node_type=st.selectbox("Select Type",options=['Agent Designs (Implemented)','Human Baselines (Seed Tree)','Random Baselines'])
        if node_type=='Agent Designs (Implemented)':
            designs=evosys.ptree.filter_by_type(['DesignArtifactImplemented'])
        elif node_type=='Human Baselines (Seed Tree)':
            designs=evosys.ptree.filter_by_type(['ReferenceCoreWithTree','ReferenceCore'])
        else:
            designs=[]
        
        verified={}
        for design_id in designs:
            design=evosys.ptree.get_node(design_id)
            verifications=design.verifications
            verified[design_id]={}
            for scale,_ in verifications.items():
                verified[design_id][scale]=True

    with col2:
        selected_design=st.selectbox("Select a Design",options=verified.keys())
        if selected_design is not None:
            vss=list(verified[selected_design].keys())
            vsstr='*(Verified: '+', '.join(vss)+')*' if len(vss)>0 else ''
        # st.write(f':red[{vsstr}]')

    with col3:
        if selected_design is not None:
            color=AU.theme_aware_options(st,'orange','violet','orange')
            selected_scale=st.select_slider(f"Choose a Scale :{color}[{vsstr}]",options=evosys.target_scales)
        else:
            selected_scale=st.select_slider(f"Choose a Scale",options=evosys.target_scales,disabled=True)
            

    with col4:
        st.write('')
        st.write('')
        if selected_design is not None or node_type=='Random Baselines':
            if node_type=='Random Baselines':
                already_verified=False
            else:
                already_verified=selected_scale in verified[selected_design]
            txt='Run Verification'
            #  if not already_verified else 'Re-Run Verification'
            run_btn= st.button(txt,use_container_width=True,disabled=DISABLE_VERIFICATION or st.session_state.evo_running or already_verified)
        else:
            run_btn= st.button('Run Verification',use_container_width=True,disabled=True)
    
    with col5:
        st.write('')
        st.write('')
        if selected_design is not None:
            resume=st.checkbox("Resume",value=not already_verified)
        else:
            resume=st.checkbox("Resume",value=False,disabled=True)
        
    with col6:
        st.write('')
        st.write('')
        if selected_design is not None:
            check_tune_btn= st.button('*Check*',use_container_width=True,disabled=DISABLE_VERIFICATION or st.session_state.evo_running)
        else:
            check_tune_btn= st.button('*Check*',use_container_width=True,disabled=True)

    if len(verified)==0:
        st.info('No implemented designs found under this namespace.')

    unfinished_runs={}
    finished_runs={}
    if U.pexists(U.pjoin(evosys.evo_dir,'ve')):
        for run_name in os.listdir(U.pjoin(evosys.evo_dir,'ve')):
            scale=run_name.split('_')[-1]
            design_id=run_name[:-len(scale)-1]
            run_dir=U.pjoin(evosys.evo_dir,'ve',run_name)
            if not U.pexists(U.pjoin(run_dir,'report.json')):
                if design_id not in unfinished_runs:
                    unfinished_runs[design_id]={}
                wandb_ids=U.load_json(U.pjoin(run_dir,'wandb_ids.json'))
                unfinished_runs[design_id][scale]=wandb_ids
            else:
                if design_id not in finished_runs:
                    finished_runs[design_id]={}
                finished_runs[design_id][scale]=f'{design_id}_{scale}'

        with st.expander("Unfinished Verifications"):
            if len(unfinished_runs)==0:
                st.info("No unfinished verifications")
            else:
                for design_id in unfinished_runs:
                    for scale in unfinished_runs[design_id]:
                        wandb_ids=unfinished_runs[design_id][scale]
                        url=None
                        if 'pretrain' in wandb_ids:
                            wandb_id=wandb_ids['pretrain']['id']
                            wandb_name=wandb_ids['pretrain']['name']
                            project=wandb_ids['project']
                            entity=wandb_ids['entity']
                            url=f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                            
                        col1,col2,col3,col4,col5=st.columns([0.3,0.5,0.8,0.4,0.4])
                        with col1:
                            st.write(f"Run id: ```{wandb_id}```")
                        with col2:
                            st.write(f"Model name: **{design_id}**-*{scale}*")
                        with col3:
                            if url:
                                st.write(f"W&B run: [{wandb_name.replace(f'{evosys.evoname}_','')[:10]}]({url})")
                        with col4:
                            resume_btn = st.button(f'Resume',key=f'btn_{design_id}_{scale}',disabled=DISABLE_VERIFICATION or st.session_state.evo_running) #,use_container_width=True):
                        with col5:
                            restart_btn = st.button(f'Restart',key=f'btn_{design_id}_{scale}_restart',disabled=DISABLE_VERIFICATION or st.session_state.evo_running) #,use_container_width=True):
                        if resume_btn:
                            run_verification(evosys.params, design_id, scale, resume=True)
                        if restart_btn:
                            run_verification(evosys.params, design_id, scale, resume=False)

        with st.expander("Finished Verifications"):
            if len(finished_runs)==0:
                st.info("No finished verifications")
            else:
                for design_id in finished_runs:
                    for scale in finished_runs[design_id]:
                        id_scale=f'{design_id}_{scale}'
                        wandb_ids=U.load_json(U.pjoin(evosys.evo_dir,'ve',id_scale,'wandb_ids.json'))
                        if 'pretrain' in wandb_ids:
                            wandb_id=wandb_ids['pretrain']['id']
                            wandb_name=wandb_ids['pretrain']['name']
                            project=wandb_ids['project']
                            entity=wandb_ids['entity']
                            url=f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                        else:
                            wandb_name='N/A'
                            wandb_id='N/A'
                            url='N/A'
                        col1,col2,col3,col4,col5=st.columns([0.3,0.5,0.8,0.4,0.4])
                        with col1:
                            st.write(f"Run id: ```{wandb_id}```")
                        with col2:
                            st.write(f"Model: **{design_id}**-*{scale}*")
                        with col3:
                            st.write(f"W&B run: [{wandb_name.replace(f'{evosys.evoname}_','')}]({url})")
                        with col4:
                            report_path=U.pjoin(evosys.evo_dir,'ve',id_scale,'report.json')
                            st.download_button(
                                label="Download Report",
                                data=json.dumps(U.load_json(report_path),indent=4),
                                file_name=f"{design_id}_{scale}_report.json",
                                mime="text/json",
                                # use_container_width=True
                            )
                        with col5:
                            pretrain_path=U.pjoin(evosys.evo_dir,'ve',id_scale,'pretrained')
                            zip_path=U.pjoin(evosys.evo_dir,'ve',id_scale,'pretrained.zip')
                            if not U.pexists(zip_path):
                                with st.spinner(f"Zipping Pretrained Model for {design_id}-*{scale}*..."):
                                    U.zip_folder(pretrain_path,zip_path)
                            st.download_button(
                                label="Download Model",
                                data=U.load_zip_file(zip_path),
                                file_name=f"{design_id}_{scale}_pretrained.zip",
                                mime='application/zip'
                            )
    # else:
    #     st.info('No verification runs found under this namespace.')


    if run_btn:
        run_verification(evosys.params, selected_design, selected_scale, resume, RANDOM_TESTING=node_type=='Random Baselines')

    if check_tune_btn:
        run_verification(evosys.params, selected_design, selected_scale,resume,prep_only=True)


    ##################################################################################

    if None in [process.poll() for process in st.session_state['running_verifications'].values()]:
        st.subheader("🥏 *Running Verification*")

    running_process=None
    for key, process in st.session_state['running_verifications'].items():
        if process.poll() is None:
            running_process=process

    if running_process:
        with st.spinner(f"Verification process for {key} is running."):
            wandb_ids = U.load_json(U.pjoin(evosys.evo_dir, 've', key, 'wandb_ids.json'))
            if 'pretrain' in wandb_ids:
                wandb_id = wandb_ids['pretrain']['id']
                project = wandb_ids['project']
                entity = wandb_ids['entity']
                url = f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                st.info(f"Check console for output. View training run on [W&B]({url}).")
                if st.button(f"Terminate",key=f'btn_{key}_term',disabled=DISABLE_VERIFICATION or st.session_state.evo_running):
                    try:
                        parent = psutil.Process(process.pid)
                        children = parent.children(recursive=True)
                        for child in children:
                            child.terminate()
                        parent.terminate()
                        
                        gone, alive = psutil.wait_procs(children + [parent], timeout=5)
                        
                        for p in alive:
                            p.kill()
                        
                        st.success(f"Verification process for {key} terminated.")
                        del st.session_state['running_verifications'][key]
                    except psutil.NoSuchProcess:
                        st.info(f"Process for {key} has already ended.")
                    except Exception as e:
                        st.error(f"Error terminating process: {str(e)}")
            while process.poll() is None:
                time.sleep(1)




def linear_budget(L,roll=0):
    db=np.cumsum(np.ones(L))[::-1]
    db/=db.sum()
    if roll>=L: 
        db=np.zeros(L)
        db[-1]=1
    else:
        db=np.roll(db,roll)
        residual=db[:roll].sum()
        db[:roll]=0
        db[roll-1]+=residual
    
    db=np.zeros(L)
    db[-1]=1
    return db

def cost_estimate(costs,verify_budget,L=4,warmup=0,title=None,mode='H100'):
    if title:
        st.subheader(f'Cost Estimate for the Scale Climbing: {title}')

    def scale_time(scale,warmup,db,c500,c2k):
        b=verify_budget[scale]
        wb=np.floor(warmup*b)
        r=b-wb
        cost_weights=np.cumsum(np.ones(L)/L)
        avg=np.dot(db,cost_weights)
        rs=[]
        for i in range(L-1):
            rs.append(int(r*db[i]))
        rs.append(int(r-np.sum(rs)))
        st.write(f'Scale: {scale}, Budget: {b}')
        st.write(f'Warmup: {int(wb)} runs, Rest: {rs}, avg. {avg:.2f}')
        cost=[w*c2k for w in cost_weights]
        stime=wb*c500+np.dot(rs,cost)
        if mode=='H100':
            return stime*8/3.6/3600
        elif mode=='A6000x8':
            return stime/3600

    at=0
    for idx,scale in enumerate(verify_budget.keys()):
        # warmup=0.2 #warmups[s]
        db=linear_budget(L,roll=idx)
        st.write(db)
        stime=scale_time(scale,warmup,db,*costs[scale])
        at+=stime
        st.write(f'Total: {stime:.1f}, accumulated [{at:.1f}] GPUhrs ({mode})\n')


# A6000x8
COSTS_LOWER={
    14:[76,174],
    31:[190,437],
    70:[11.2*60,22.3*60],
    125:[80*60,146.3*60],
    350:[10*3600,17*3600],
    760:[30*3600,54.3*3600],
    1300:[0,137.5*3600],
}

COSTS_UPPER={
    14:[43,46],
    31:[112,123],
    70:[7.5*60,8.3*60],
    125:[50*60,54.3*60],
    350:[7.5*3600,7.85*3600],
    760:[30*3600,32.6*3600],
    1300:[0,92.5*3600],
}

# A6000
COSTS_2K_OPT={}
COSTS_2K_PAS={}
for scale in COSTS_LOWER:
    fast_2k=COSTS_UPPER[scale][1]
    med_2k=COSTS_LOWER[scale][1]
    slow_2k=med_2k*2-fast_2k
    COSTS_2K_OPT[scale]=fast_2k*8
    COSTS_2K_PAS[scale]=slow_2k*8

SPEEDUP = {
    'A6000': 1.0,
    'H100': 3.6
}

def verify_budget_tool(evosys,project_dir):
    
    st.header("Verify Budget Tool")
    st.caption("Help you estimate the run time and decide the budget for verification based on linear scaling assumption, the verification time is not included yet.")

    Col1,Col2 = st.columns(2)
    with Col1:
        st.write('#### GPU and Node Setup')
        cols = st.columns(4)
        with cols[0]:
            gpu_type=st.selectbox('GPU Type',options=['H100','A6000','Manual Input'],index=0,
                help='If you choose "Manual Input", you need to input the speedup manually.')
        with cols[1]:
            _speedup=SPEEDUP[gpu_type] if gpu_type!='Manual Input' else 1.0
            speedup=st.number_input('Speedup over A6000',value=_speedup,step=0.1,format='%.2f',disabled=gpu_type!='Manual Input')
        with cols[2]:
            n_gpus=st.number_input('GPUs per Node',value=1,min_value=1,step=1,help='The number of GPUs per node to use.')
        with cols[3]:
            n_nodes=st.number_input('Verification Nodes',value=1,min_value=1,step=1,help='The number of nodes to use.')
        
        optimistic_level=st.slider('Optimistic Level',min_value=0.0,max_value=1.0,value=0.5,step=0.01,help='The level of optimism for the cost estimate.')
        _use_manual_cost=st.checkbox('Use manual cost input below *(will overwrite the above)*')

        linear_cost = {}
        for scale in COSTS_2K_OPT:
            opt_cost = COSTS_2K_OPT[scale]
            pas_cost = COSTS_2K_PAS[scale]
            time_2k_a6000_single = optimistic_level*opt_cost + (1-optimistic_level)*pas_cost
            linear_cost[f'{scale}M'] = time_2k_a6000_single/speedup/n_gpus/n_nodes

        st.caption('Estimated Training Time for models with 2K token context and 20x tokens on each scale (in seconds):')
        linear_cost_df = pd.DataFrame(linear_cost,index=['Train (s)']).round(0)
        _cost_est_mat = st.data_editor(linear_cost_df,use_container_width=True)
        if _use_manual_cost:
            linear_cost = _cost_est_mat.to_dict(orient='records')[0]
    
    with Col2:
        st.write('#### Target Scale and Selection Ratio')
        subcol1, subcol2, subcol3= st.columns([1,1,0.25])
        with subcol1:
            target_scale=st.select_slider('Target Scale',options=TARGET_SCALES,value=evosys.params['scales'].split(',')[-1],
                help='The largest scale to train, will train `N Target` models at this scale.')
            scales=[]
            for s in TARGET_SCALES:
                if int(target_scale.replace('M',''))>=int(s.replace('M','')):
                    scales.append(s)
            scales=','.join(scales)
        with subcol2:
            selection_ratio=st.slider('Selection Ratio',min_value=0.0,max_value=1.0,value=evosys.params['selection_ratio'],
                help='The ratio of designs to keep from lower scale, e.g. targets 8 models on 70M with selection ratio 0.5 will train 16 models on 35M, 32 models on 14M.')
        with subcol3:
            n_target=st.number_input('N Target',value=evosys.params['n_target'],min_value=1,step=1)

        _manual_set_budget=st.checkbox('Use fine-grained verify budget below *(will overwrite the above)*')
        _verify_budget={i:0 for i in TARGET_SCALES}
        budget=n_target
        for scale in scales.split(',')[::-1]:
            _verify_budget[scale]=int(np.ceil(budget))
            budget/=selection_ratio
        verify_budget=_verify_budget.copy()
        _verify_budget_df = pd.DataFrame(_verify_budget,index=['Runs'])
        _verify_budget_df = st.data_editor(_verify_budget_df,hide_index=False,use_container_width=True)
        _verify_budget=_verify_budget_df.to_dict(orient='records')[0]
        _verify_budget={k:v for k,v in _verify_budget.items() if v!=0}
        if _manual_set_budget:
            verify_budget=_verify_budget

        st.caption('Training Token Multipliers for each Scale (Training tokens = #Params * Multiplier):')
        token_mults = copy.deepcopy(DEFAULT_TOKEN_MULTS)
        token_mults_df = pd.DataFrame(token_mults,index=['Mult'])
        token_mults_df = st.data_editor(token_mults_df,use_container_width=True)
        token_mults = token_mults_df.to_dict(orient='records')[0]

    st.info(f'**Note:** The budget estimated is based on simple linear scaling assumption, based on the data from training on 8 x A6000 GPUs, and the actual cost might be different.')

    st.write('#### Estimated Training Time')

    verify_costs = {s:linear_cost[s]*verify_budget[s]*token_mults[s]/20 for s in verify_budget}
    verify_time = {s:verify_costs[s]/3600 for s in verify_costs}
    verify_ghrs = {s:verify_costs[s]*n_gpus*n_nodes/3600 for s in verify_costs}

    def get_ghrs_df(ghrs):
        ghrs_df = pd.DataFrame(ghrs,index=['GPU Hrs'])
        ghrs_df['Total'] = ghrs_df.sum(axis=1)
        ghrs_df['Days'] = ghrs_df['Total']/24
        ghrs_df = ghrs_df.round(1)
        return ghrs_df

    st.write('GPU Hours (Total):')
    total_ghrs_df = get_ghrs_df(verify_ghrs)
    st.dataframe(total_ghrs_df,use_container_width=True)

    st.write(f'Running Time ({n_nodes} x {n_gpus} {gpu_type} nodes):')
    total_time_df = get_ghrs_df(verify_time)
    st.dataframe(total_time_df,use_container_width=True)



def dialog_cost_estimator(system,avg_input,avg_output,n_rounds,cost_dict,use_cache):
    input_token_price = cost_dict['input']
    output_token_price = cost_dict['output']
    if use_cache and 'cache_creation' in cost_dict and 'cache_read' in cost_dict:
        cache_write_price = cost_dict['cache_creation']
        cache_read_price = cost_dict['cache_read']
    old_tokens = 0
    new_tokens = system
    costs = {}
    tokens = {}
    aggregated_cost = {}
    aggregated_tokens = {}
    total_cost = 0
    for i in range(n_rounds):
        cost = {}
        token={}
        new_tokens += avg_input
        input_tokens = new_tokens + old_tokens
        if use_cache:
            cost['cache_write'] = cache_write_price*new_tokens
            cost['cache_read'] = cache_read_price*old_tokens
            token['cache_write'] = new_tokens
            token['cache_read'] = old_tokens
        else:
            cost['input'] = input_token_price*input_tokens
            token['input'] = input_tokens
        cost['output'] = output_token_price*avg_output
        token['output'] = avg_output
        old_tokens = input_tokens
        new_tokens += avg_output
        for k,v in cost.items():
            aggregated_cost[k] = aggregated_cost.get(k,0) + v
            total_cost += v 
        for k,v in token.items():
            aggregated_tokens[k] = aggregated_tokens.get(k,0) + v
        costs[i] = cost
        tokens[i] = token
    return costs, aggregated_cost, total_cost, aggregated_tokens, tokens



def design_budget_tool(evosys,project_dir):
    st.header("Design Budget Tools")

    if 'saved_estimations' not in st.session_state:
        st.session_state['saved_estimations'] = {}

    with st.sidebar:
        st.write('#### Saved Estimations')
        if len(st.session_state['saved_estimations']) > 0:
            total_costs = {k:v['total_cost'] for k,v in st.session_state['saved_estimations'].items()}
            ests = list(total_costs.items())
            # set agent as index
            est_df = pd.DataFrame(ests,columns=['Agent','Cost ($)']).set_index('Agent')
            st.dataframe(
                est_df.round(2), 
                use_container_width=True, 
                key="saved_estimations_editor",
            )
            total_cost = sum(total_costs.values())
            st.write(f'Total Cost: `${total_cost:.2f}`')
        else:
            st.info('No saved estimations.')
        
        cols = st.columns(2)
        with cols[0]:
            st.button('Refresh',use_container_width=True)
        with cols[1]:
            if st.button('Clear',use_container_width=True):
                st.session_state['saved_estimations'] = {}
                st.rerun()
        st.download_button(
            label='Export Raw JSON',
            data=json.dumps(st.session_state['saved_estimations'],indent=4),
            file_name='saved_estimations.json',
            mime='application/json',
            use_container_width=True
        )

    st.write('#### Dialog Cost Estimator')
    cols = st.columns(4)
    with cols[0]:
        system_prompt_token_length = st.slider('System Prompt Token Length',min_value=100,max_value=10000,value=2000,step=100)
    with cols[1]:
        avg_input_token_per_round = st.slider('Avg. Input Token Per Round',min_value=100,max_value=10000,value=1000,step=100)
    with cols[2]:
        avg_output_token_per_round = st.slider('Avg. Output Token Per Round',min_value=100,max_value=10000,value=1000,step=100)
    with cols[3]:
        n_rounds = st.slider('Number of Rounds',min_value=1,max_value=100,value=10,step=1)

    cols = st.columns([2,2,1,1,1])
    with cols[0]:   
        model_type = st.selectbox('Model Type',options=['OpenAI','Anthropic'],index=0)
    if model_type == 'OpenAI':
        cost_dicts = OPENAI_COSTS_DICT
    else:
        cost_dicts = ANTHROPIC_COSTS_DICT
    with cols[1]:
        model_name = st.selectbox('Model',options=cost_dicts.keys(),index=0)
        cost_dict = cost_dicts[model_name]
    with cols[2]:
        st.write('')
        st.write('')
        _use_preset = st.checkbox('Use Preset Prices',value=False)
    with cols[3]:
        agent_name = st.text_input('Agent Name',value='',
            help='The name of the agent for this estimation.')
    with cols[4]:
        st.write('')
        st.write('')
        _save_estimation = st.button('Save Estimation',
            help='Save the current estimation to the sidebar. Helpful to estimate multiple agents.')

    cols = st.columns([2,2,2,2,1])
    with cols[0]:
        _input_token_price = cost_dict['input']*1e6
        input_token_price = st.number_input('Input Token Price (USD/1M)',value=_input_token_price,step=0.01,disabled=_use_preset)
    with cols[1]:
        _output_token_price = cost_dict['output']*1e6
        output_token_price = st.number_input('Output Token Price (USD/1M)',value=_output_token_price,step=0.01,disabled=_use_preset)
    with cols[2]:
        _cache_write_price = cost_dict.get('cache_creation',0)*1e6
        cache_write_price = st.number_input('Cache Write Price (USD/1M)',value=_cache_write_price,step=0.01,disabled=_use_preset)
    with cols[3]:
        _cache_read_price = cost_dict.get('cache_read',0)*1e6
        cache_read_price = st.number_input('Cache Read Price (USD/1M)',value=_cache_read_price,step=0.01,disabled=_use_preset)
    with cols[4]:
        st.write('')
        st.write('')
        _disabled = False
        if _use_preset:
            _disabled = True
            if 'cache_creation' in cost_dict and 'cache_read' in cost_dict:
                _disabled = False
        use_cache = st.checkbox('Use Cache',value=False,disabled=_disabled)

    if _use_preset: 
        cost_dict = cost_dicts[model_name]
    else:
        cost_dict = {'input':input_token_price/1e6, 'output':output_token_price/1e6}
        if use_cache:
            cost_dict['cache_creation'] = cache_write_price/1e6
            cost_dict['cache_read'] = cache_read_price/1e6
    costs, aggregated_cost, total_cost, aggregated_tokens, tokens = dialog_cost_estimator(system_prompt_token_length, avg_input_token_per_round, avg_output_token_per_round, n_rounds, cost_dict, use_cache)

    if _save_estimation:
        agent_name = agent_name if agent_name else f'Est{len(st.session_state["saved_estimations"])+1}'
        st.session_state['saved_estimations'][agent_name] = {
            'total_cost':total_cost,
            'costs':costs,
            'aggregated_cost':aggregated_cost,
            'aggregated_tokens':aggregated_tokens,
            'tokens':tokens,
            'cost_dict':cost_dict,
            'use_cache':use_cache,
            'system_prompt_token_length':system_prompt_token_length,
            'avg_input_token_per_round':avg_input_token_per_round,
            'avg_output_token_per_round':avg_output_token_per_round,
            'n_rounds':n_rounds,
            'use_preset':_use_preset,
        }
        if _use_preset:
            st.session_state['saved_estimations'][agent_name]['model_name'] = model_name
            st.session_state['saved_estimations'][agent_name]['model_type'] = model_type
        st.rerun()

    col1,col2 = st.columns(2)
    with col1:
        st.write(f'#### Cost Estimation `${total_cost:.2f}`')
        st.write('Aggregated Cost:')
        st.write(aggregated_cost)
        st.write('Per Round Cost:')
        st.write(costs)
    with col2:
        st.write('#### Token Estimation')
        st.write('Aggregated Tokens:')
        st.write(aggregated_tokens)
        st.write('Per Round Tokens:')
        st.write(tokens)




    



def budget_tools(evosys,project_dir):
    # st.header("Budget Tools")
    with st.sidebar:
        choose_tool=st.selectbox("Choose Tool",options=['Verify Budget Tool','Design Budget Tool'])
   
    if choose_tool=='Verify Budget Tool':
        verify_budget_tool(evosys,project_dir)
    elif choose_tool=='Design Budget Tool':
        design_budget_tool(evosys,project_dir)



def verify(evosys,project_dir):

    with st.sidebar:
        AU.running_status(st,evosys)
        choose_mode=st.selectbox("Choose Mode",options=['Verification Engine','Budget Tools'])

    if choose_mode=='Verification Engine':
        verify_engine(evosys,project_dir)
    else:
        budget_tools(evosys,project_dir)


if __name__ == '__main__':
    import argparse
    from model_discovery.evolution import BuildEvolution

    AU.print_cli_title()

    setting=AU.get_setting()
    default_namespace=setting.get('default_namespace','test_evo_000')

    parser = argparse.ArgumentParser()
    parser.add_argument("--evoname", default=default_namespace, type=str) # the name of the whole evolution
    parser.add_argument("--design_id", default=None, type=str) # the name of the whole evolution
    parser.add_argument("--scale", default=None, type=str) # the name of the whole evolution
    parser.add_argument("--resume", action='store_true') # the name of the whole evolution
    parser.add_argument("--daemon", action='store_true')
    parser.add_argument("--prep_only", action='store_true')
    parser.add_argument("--sess_id", default=None, type=str)
    parser.add_argument("--node_id", default=None, type=str)
    parser.add_argument("--pid", default=None, type=int)
    parser.add_argument("--RANDOM_TESTING", action='store_true')

    args = parser.parse_args()

    args.design_id = None if args.design_id == 'None' else args.design_id
    args.scale = None if args.scale == 'None' else args.scale
    args.sess_id = None if args.sess_id == 'None' else args.sess_id
    args.node_id = None if args.node_id == 'None' else args.node_id
    args.pid = None if args.pid == 'None' else args.pid

    evokwargs = {}
    if args.daemon or args.prep_only:
        evokwargs={'db_only':True,'no_agent':True}
    evosys = BuildEvolution(
        params={'evoname':args.evoname,**evokwargs}, 
        do_cache=False,
    )

    if args.daemon:
        assert args.node_id is not None
        verify_daemon(args.evoname, evosys, args.sess_id, args.design_id, args.scale, args.node_id, args.pid)
    elif args.prep_only:
        # bash scripts/run_verify.sh --prep_only --design_id ghanet --scale 31M
        assert args.design_id is not None and args.scale is not None
        evosys._prep_model(args.design_id, args.scale)
    else:
        node_id= args.node_id if args.node_id else str(uuid.uuid4())[:8]
        verify_command(node_id,evosys,args.evoname,design_id=args.design_id,scale=args.scale,resume=args.resume,cli=True, RANDOM_TESTING=args.RANDOM_TESTING)


