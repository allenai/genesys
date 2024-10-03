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
import select
import pandas as pd
import multiprocessing
import random
from datetime import datetime, timedelta
import uuid
import subprocess
from google.cloud.firestore import DELETE_FIELD


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

from model_discovery.ve.data_loader import load_datasets



SMOLLM_125_CORPUS=['fineweb-edu-dedup']#,'cosmopedia-v2','python-edu','open-web-math','deepmind-math-small','stackoverflow-clean']


def verify_command(node_id, evoname, design_id=None, scale=None, resume=True, cli=False):
    evosys = BuildEvolution(params={'evoname': evoname}, do_cache=False)
    check_stale_verifications(evosys, evoname)
    sess_id, pid, _design_id, _scale = _verify_command(node_id, evosys, evoname, design_id, scale, resume, cli, ret_id=True)
    
    log_ref = evosys.remote_db.collection('experiment_logs').document(evoname)
    
    timestamp = datetime.now().strftime('%B %d, %Y at %I:%M:%S %p %Z')+'_'+str(uuid.uuid4())
    
    if sess_id:
        log = f'Node {node_id} running verification on {_design_id}_{_scale}'
        do_log(log_ref,timestamp,log)
        # Start the daemon in a separate process
        print('Starting Verify Daemon Process...')
        daemon_cmd = f"python -m bin.pages.verify --daemon --evoname {evoname} --sess_id {sess_id} --design_id {_design_id} --scale {_scale} --node_id {node_id} --pid {pid}"
        subprocess.Popen(daemon_cmd, shell=True)
    else:
        log = f'Node {node_id} failed to run verification on {design_id}_{scale} with error: {pid}'
        do_log(log_ref,timestamp,log)

    return sess_id, pid

# python -m bin.pages.verify --daemon --evoname {evoname} --sess_id {sess_id} --design_id {design_id} --scale {scale} --node_id {node_id} --pid {pid}

def _verify_command(node_id, evosys, evoname, design_id=None, scale=None, resume=True, cli=False,ret_id=False):
    if evosys.evoname != evoname:
        evosys.switch_ckpt(evoname)

    verify_ref = evosys.remote_db.collection('verifications').document(evoname)

    # Acquire lock
    while True:
        doc = verify_ref.get().to_dict()
        current_time = datetime.now()
        
        if doc is None or not doc.get('lock', {}).get('locked'):
            # No lock, we can acquire it
            verify_ref.set({
                'lock': {'locked': True, 'node_id': node_id, 'timestamp': current_time}
            }, merge=True)
            break
        else:
            lock_info = doc['lock']
            lock_time = lock_info['timestamp'].replace(tzinfo=None)  # Remove timezone info for comparison
            
            if (current_time - lock_time).total_seconds() > evosys.CM.zombie_threshold:
                # The lock is held by a zombie node, we can take it
                verify_ref.set({
                    'lock': {'locked': True, 'node_id': node_id, 'timestamp': current_time}
                }, merge=True)
                print(f"Took over zombie lock from node {lock_info['node_id']}")
                break
            
        time.sleep(1)  # Wait before trying again

    timestamp = datetime.now()
    try:
        verifying_dict = doc.get('verifying', {})
        unfinished_verifies = evosys.get_unfinished_verifies(evoname)
        available_verifies = [v for v in unfinished_verifies if v not in verifying_dict]
        if design_id is None or scale is None:
            doc = verify_ref.get().to_dict()

            if resume and available_verifies:
                exp = random.choice(available_verifies)
                scale = exp.split('_')[-1]
                design_id = exp[:-len(scale)-1]
            else:
                exclude_list = []
                for key in verifying_dict:
                    scale = key.split('_')[-1]
                    design_id = key[:-len(scale)-1]
                    exclude_list.append((design_id,scale))  
                design_id, scale = evosys.selector.select_verify(exclude_list=exclude_list)
                if design_id is None:
                    msg = "No unverified design found at any scale."
                    if not cli:
                        st.error(msg)
                    if ret_id:
                        return None, msg, None, None
                    else:
                        return None, msg
        else:
            if f'{design_id}_{scale}' not in available_verifies:
                msg = f"Verification on {design_id}_{scale} is already running or completed."
                if not cli:
                    st.error(msg)
                if ret_id:
                    return None, msg, None, None
                else:
                    return None, msg

            # Add to verifying list with timestamp
            verify_key = f"{design_id}_{scale}"
            verify_ref.update({
                f'verifying.{verify_key}': {
                    'node_id': node_id,
                    'timestamp': timestamp,
                    'pid': None,  # Will be updated later
                    'sess_id': None
                }
            })

    finally:
        # Release lock
        verify_ref.update({'lock': {'locked': False, 'node_id': None}})

    params = {'evoname': evoname}
    sess_id, pid = run_verification(params, design_id, scale, resume, cli=cli)
    if sess_id is not None:
        verify_key = f"{design_id}_{scale}"
        verify_ref.update({
            f'verifying.{verify_key}': {
                'node_id': node_id,
                'timestamp': timestamp,
                'pid': pid,
                'sess_id': sess_id
            }
        })
    if ret_id:
        return sess_id, pid, design_id, scale
    else:
        return sess_id, pid
    

def do_log(log_ref,timestamp,log):
    log_ref.set({timestamp: log}, merge=True)
    print(f'{timestamp.split("_")[0]}: {log}')

def verify_daemon(evoname, sess_id, design_id, scale, node_id, pid):
    evosys = BuildEvolution(
        params={'evoname': evoname,'tree_only':True,'no_agent':True}, 
        do_cache=False,
    )
    verify_ref = evosys.remote_db.collection('verifications').document(evoname)
    log_ref = evosys.remote_db.collection('experiment_logs').document(evoname)
    

    # Start heartbeat
    verify_ref.set({'heartbeats': {node_id: datetime.now()}}, merge=True)
    
    try:
        process = psutil.Process(pid)
        while True:
            try:
                # Update heartbeat
                verify_ref.set({'heartbeats': {node_id: datetime.now()}}, merge=True)
                
                if process.status() == psutil.STATUS_ZOMBIE:
                    log = f'Node {node_id} detected zombie process {pid} for {design_id}_{scale}'
                    do_log(log_ref,datetime.now().strftime('%B %d, %Y at %I:%M:%S %p %Z')+'_'+str(uuid.uuid4()),log)
                    break
                elif process.status() in [psutil.STATUS_DEAD, psutil.STATUS_STOPPED]:
                    break
                elif process.status() == psutil.STATUS_SLEEPING:
                    time.sleep(60)  # Wait for 1 minute before checking again
                else:
                    time.sleep(60)  # Check every minute for active processes
            except psutil.NoSuchProcess:
                log = f'Node {node_id} lost track of verification process {pid} for {design_id}_{scale}'
                do_log(log_ref,datetime.now().strftime('%B %d, %Y at %I:%M:%S %p %Z')+'_'+str(uuid.uuid4()),log)
                raise psutil.NoSuchProcess(pid)

        # Check if the process completed successfully
        try:
            exit_code = process.wait(timeout=1)
            if exit_code == 0:
                log = f'Node {node_id} completed verification process {pid} on {design_id}_{scale}'
            else:
                log = f'Node {node_id} failed verification process {pid} on {design_id}_{scale} with exit code {exit_code}'
        except psutil.TimeoutExpired:
            log = f'Node {node_id} failed to get exit code for verification process {pid} on {design_id}_{scale}'
        
        do_log(log_ref,datetime.now().strftime('%B %d, %Y at %I:%M:%S %p %Z')+'_'+str(uuid.uuid4()),log)

    except Exception as e:
        log = f'Node {node_id} encountered an error during verification process {pid} on {design_id}_{scale}: {str(e)}'
        do_log(log_ref,datetime.now().strftime('%B %d, %Y at %I:%M:%S %p %Z')+'_'+str(uuid.uuid4()),log)
    
    finally:
        # Ensure verification is marked as complete even if an exception occurred
        complete_verification(evosys, evoname, design_id, scale, node_id)
        

    return sess_id, pid


def complete_verification(evosys, evoname, design_id, scale, node_id):
    verify_ref = evosys.remote_db.collection('verifications').document(evoname)
    
    verify_key = f"{design_id}_{scale}"
    verify_ref.update({
        f'verifying.{verify_key}': DELETE_FIELD,
        f'heartbeats.{node_id}': DELETE_FIELD
    })

def check_stale_verifications(evosys, evoname, max_age=210):
    verify_ref = evosys.remote_db.collection('verifications').document(evoname)
    
    doc = verify_ref.get().to_dict()
    if doc is None:
        return
    verifying_dict = doc.get('verifying', {})
    heartbeats = doc.get('heartbeats', {})
    
    now = datetime.now(pytz.UTC)
    stale_verifications = []
    
    for verify_key, item in verifying_dict.items():
        node_id = item['node_id']
        last_heartbeat = heartbeats.get(node_id)
        
        if last_heartbeat is None:
            stale_verifications.append(verify_key)
        else:
            # Ensure last_heartbeat is timezone-aware
            if last_heartbeat.tzinfo is None:
                last_heartbeat = pytz.UTC.localize(last_heartbeat)
            
            if (now - last_heartbeat).total_seconds() > max_age:
                stale_verifications.append(verify_key)
        
        if U.pexists(U.pjoin(evosys.evo_dir,'ve',verify_key,'report.json')):
            stale_verifications.append(verify_key)
        else:
            scale=verify_key.split('_')[-1]
            design_id=verify_key[:-len(scale)-1]
            if evosys.ptree.FM.is_verified(design_id,scale):
                stale_verifications.append(verify_key)
    
    if stale_verifications:
        update_dict = {f'verifying.{key}': DELETE_FIELD for key in stale_verifications}
        update_dict.update({f'heartbeats.{verifying_dict[key]["node_id"]}': DELETE_FIELD for key in stale_verifications})
        verify_ref.update(update_dict)
        for key in stale_verifications:
            print(f"Removed stale verification: {key} from node {verifying_dict[key]['node_id']}")



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



def _run_verification(params, design_id, scale, resume, cli=False, prep_only=False):
    params_str = shlex.quote(json.dumps(params))
    cmd = f"python -m model_discovery.evolution --mode prep_model --params {params_str} --design_id {design_id} --scale {scale}"
    if cli:
        print('Preparing Model...')
        process = subprocess.Popen(cmd, shell=True)
    else:
        with st.spinner(f'Preparing Model...'):
            process = subprocess.run(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

    if prep_only:
        return process

    cmd = f"python -m model_discovery.evolution --mode verify --params {params_str} --design_id {design_id} --scale {scale}"
    if resume:
        cmd+=' --resume'
    if cli:
        print(f'Launching Verification with command:\n```{cmd}```')
        process = subprocess.Popen(cmd, shell=True)
    else:
        st.write(f'Launching Verification with command:\n```{cmd}```')
        with st.spinner('Running... Please check the console for verification progress.'):
            # process = subprocess.run(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
            process = subprocess.Popen(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    return process

def run_verification(params, design_id, scale, resume, cli=False, prep_only=False):
    key = f"{design_id}_{scale}"
    # if key not in st.session_state['running_verifications']:
    if not cli: 
        polls=[p.poll() for p in st.session_state['running_verifications'].values()]
    if cli or (not None in polls):
        params = copy.deepcopy(params)
        process = _run_verification(params, design_id, scale, resume, cli=cli, prep_only=prep_only)
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


def verify(evosys,project_dir):

    st.title("Verification Engine")

    if st.session_state.listening_mode:
        st.warning("**NOTE:** You are running in listening mode. Verification engine is taken over by the system.")

    if st.session_state.evo_running:
        st.warning("**NOTE:** You are running as the master node. Verification engine is taken over by the system.")

    # evosys.ptree.load()
    
    # if 'output' not in st.session_state:
    #     st.session_state['output'] = {}

    with st.sidebar:
        AU.running_status(st,evosys)
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
    col1,_,col2,_,col3,_,col4,col5,col6=st.columns([1,0.05,0.9,0.05,1.3,0.05,0.6,0.4,0.45])
    with col1:
        node_type=st.selectbox("Select Type",options=['Agent Designs (Implemented)','Human Baselines (Seed Tree)'])
        if node_type=='Agent Designs (Implemented)':
            designs=evosys.ptree.filter_by_type(['DesignArtifactImplemented'])
        else:
            designs=evosys.ptree.filter_by_type(['ReferenceCoreWithTree','ReferenceCore'])
        
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
            selected_scale=st.select_slider(f"Choose a Scale :orange[{vsstr}]",options=evosys.target_scales)
        else:
            selected_scale=st.select_slider(f"Choose a Scale",options=evosys.target_scales,disabled=True)
            

    with col4:
        st.write('')
        st.write('')
        if selected_design is not None:
            already_verified=selected_scale in verified[selected_design]
            txt='Run Verification'
            #  if not already_verified else 'Re-Run Verification'
            run_btn= st.button(txt,use_container_width=True,disabled=st.session_state.listening_mode or st.session_state.evo_running or already_verified)
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
            check_tune_btn= st.button('*Check & Tune*',use_container_width=True,disabled=st.session_state.listening_mode or st.session_state.evo_running)
        else:
            check_tune_btn= st.button('*Check & Tune*',use_container_width=True,disabled=True)

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
                            
                        col1,col2,col3,col4,col5=st.columns([0.5,0.5,1,0.3,0.3])
                        with col1:
                            st.write(f"Run id: ```{wandb_id}```")
                        with col2:
                            st.write(f"Model name: **{design_id}**-*{scale}*")
                        with col3:
                            if url:
                                st.write(f"W&B run: [{wandb_name}]({url})")
                        with col4:
                            resume_btn = st.button(f'Resume',key=f'btn_{design_id}_{scale}',disabled=st.session_state.listening_mode or st.session_state.evo_running) #,use_container_width=True):
                        with col5:
                            restart_btn = st.button(f'Restart',key=f'btn_{design_id}_{scale}_restart',disabled=st.session_state.listening_mode or st.session_state.evo_running) #,use_container_width=True):
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
                        col1,col2,col3,col4,col5=st.columns([0.5,0.5,1,0.4,0.4])
                        with col1:
                            st.write(f"Run id: ```{wandb_id}```")
                        with col2:
                            st.write(f"Model: **{design_id}**-*{scale}*")
                        with col3:
                            st.write(f"W&B run: [{wandb_name}]({url})")
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
        run_verification(evosys.params, selected_design, selected_scale, resume)

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
                if st.button(f"Terminate",key=f'btn_{key}_term',disabled=st.session_state.listening_mode or st.session_state.evo_running):
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
            


if __name__ == '__main__':
    import argparse
    from model_discovery.evolution import BuildEvolution

    parser = argparse.ArgumentParser()
    parser.add_argument("--evoname", default='test_evo_000', type=str) # the name of the whole evolution
    parser.add_argument("--design_id", default=None, type=str) # the name of the whole evolution
    parser.add_argument("--scale", default=None, type=str) # the name of the whole evolution
    parser.add_argument("--resume", action='store_true') # the name of the whole evolution
    parser.add_argument("--daemon", action='store_true')
    parser.add_argument("--prep_only", action='store_true')
    parser.add_argument("--sess_id", default=None, type=str)
    parser.add_argument("--node_id", default=None, type=str)
    parser.add_argument("--pid", default=None, type=int)

    args = parser.parse_args()

    args.design_id = None if args.design_id == 'None' else args.design_id
    args.scale = None if args.scale == 'None' else args.scale
    args.sess_id = None if args.sess_id == 'None' else args.sess_id
    args.node_id = None if args.node_id == 'None' else args.node_id
    args.pid = None if args.pid == 'None' else args.pid

    if args.daemon:
        assert args.node_id is not None
        verify_daemon(args.evoname, args.sess_id, args.design_id, args.scale, args.node_id, args.pid)
    elif args.prep_only:
        # bash scripts/run_verify.sh --prep_only --design_id ghanet --scale 31M
        assert args.design_id is not None and args.scale is not None
        from model_discovery.evolution import BuildEvolution
        evosys = BuildEvolution(
            params={'evoname':args.evoname,'db_only':True,'no_agent':True},do_cache=False)
        evosys._prep_model(args.design_id, args.scale)
    else:
        node_id= args.node_id if args.node_id else str(uuid.uuid4())[:8]
        verify_command(node_id,args.evoname,design_id=args.design_id,scale=args.scale,resume=args.resume,cli=True)


