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
import shlex
import select

sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

from model_discovery.ve.data_loader import load_datasets


TARGET_SCALES = ['14M','31M','70M','125M','350M','760M','1300M']

SMOLLM_125_CORPUS=['fineweb-edu-dedup']#,'cosmopedia-v2','python-edu','open-web-math','deepmind-math-small','stackoverflow-clean']



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


def nvidia_smi_monitor():
    with st.expander("NVIDIA-SMI Monitor"):
        def get_nvidia_smi_output():
            try:
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
                return result.stdout.decode('utf-8')
            except Exception as e:
                return f"Error: {e}"
        # while True:
        if st.button("Refresh",key='refresh_btn'):
            st.rerun()
        st.text(get_nvidia_smi_output())
        # time.sleep(0.1)


def _run_verification(params, design_id, scale, resume):
    params_str = shlex.quote(json.dumps(params))
    cmd = f"python -m model_discovery.evolution --mode verify --params {params_str} --design_id {design_id} --scale {scale}"
    if resume:
        cmd+=' --resume'
    st.write(f'Running Verification command:\n```{cmd}```')

    process = subprocess.run(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    return process

def run_verification(params, design_id, scale, resume):
    key = f"{design_id}_{scale}"
    if key not in st.session_state['running_verifications']:
        params = copy.deepcopy(params)
        process = _run_verification(params, design_id, scale, resume)
        st.session_state['running_verifications'][key] = process
        st.session_state['output'][key] = []
        st.success(f"Verification process started for {design_id} on scale {scale}.")
    else:
        st.warning(f"A verification process for {design_id} on scale {scale} is already running.")


def stream_output(process, key):
    for line in process.stdout:
        st.session_state['output'][key].append(line)
        
    for line in process.stderr:
        st.session_state['output'][key].append(f"ERROR: {line}")


def engine(evosys,project_dir):

    st.title("Verification Engine")

    # evosys.ptree.load()

    if 'running_verifications' not in st.session_state:
        st.session_state['running_verifications'] = {}
    
    if 'output' not in st.session_state:
        st.session_state['output'] = {}

    with st.sidebar:
        st.write(f'**Namespace: ```{evosys.evoname}```**')
        with st.expander("Running Sessions"):
            st.write(st.session_state['running_verifications'])


    st.subheader("System Diagnostics")
    col1, col2, col3, col4 = st.columns(4)
    cpu_info, gpu_info, mem_info = get_system_info()
    with col1:
        with st.expander("CPU Info"):
            st.write(cpu_info)
    with col2:
        with st.expander("GPU Info"):
            st.write(gpu_info)
    with col3:
        with st.expander("Memory Info"):
            st.write(mem_info)
    with col4:
        with st.expander("Experiment Info"):
            st.write(f'Namespace: ```{evosys.evoname}```')
            st.write(evosys.state)

    with st.expander("Diagnostic Panel (For internal use)"):
        # st.subheader("Test loading datasets")
        config=GAMConfig_14M()
        st.subheader("Test dataset loader")
        col1,_,col2,_,col3,_=st.columns([3,0.3,2,0.3,2,1])
        with col1:
            dataset_name=st.selectbox("Choose a Dataset",options=SMOLLM_125_CORPUS)
        with col2:
            st.write('')
            st.write('')
            if st.button("Test Loading "+dataset_name,use_container_width=True):
                st.spinner('Loading... Please check the console for output')
                config.training_data=[dataset_name]
                load_datasets(config)
        with col3:
            st.write('')
            st.write('')
            if st.button("Test Loading All Datasets",use_container_width=True):
                st.spinner('Loading... Please check the console for output')
                config.training_data=SMOLLM_125_CORPUS
                load_datasets(config)
    
    nvidia_smi_monitor()


    st.header("Verify Designs")
    col1,_,col2,_,col3,_,col4,col5=st.columns([1,0.05,0.9,0.05,1.3,0.05,0.6,0.4])
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
            for scale,verification in verifications.items():
                verified[design_id][scale]=verification
    with col2:
        selected_design=st.selectbox("Select a Design",options=verified.keys())
        vss=list(verified[selected_design].keys())
        vsstr='*(Verified on '+', '.join(vss)+')*' if len(vss)>0 else ''
        # st.write(f':red[{vsstr}]')

    with col3:
        scale=st.select_slider(f"Choose a Scale :red[{vsstr}]",options=TARGET_SCALES)

    with col4:
        st.write('')
        st.write('')
        already_verified=scale in verified[selected_design]
        txt='Run Verification' if not already_verified else 'Re-Run Verification'
        run_btn= st.button(txt,use_container_width=True)
    
    with col5:
        st.write('')
        st.write('')
        resume=st.checkbox("Resume",value=not already_verified)

    if len(verified)==0:
        st.warning('No implemented designs found in the experiment directory')

    running_runs={}
    for run_name in os.listdir(U.pjoin(evosys.evo_dir,'ve')):
        scale=run_name.split('_')[-1]
        design_id=run_name[:-len(scale)-1]
        run_dir=U.pjoin(evosys.evo_dir,'ve',run_name)
        if not U.pexists(U.pjoin(run_dir,'report.json')):
            if design_id not in running_runs:
                running_runs[design_id]={}
            wandb_ids=U.load_json(U.pjoin(run_dir,'wandb_ids.json'))
            running_runs[design_id][scale]=wandb_ids
    with st.expander("Unfinished Verifications"):
        if len(running_runs)==0:
            st.warning("No unfinished verifications")
        else:
            for design_id in running_runs:
                for scale in running_runs[design_id]:
                    wandb_ids=running_runs[design_id][scale]
                    col1,col2,col3=st.columns([1,0.1,0.1])
                    with col1:
                        wandb_id=wandb_ids['pretrain']
                        project=wandb_ids['project']
                        entity=wandb_ids['entity']
                        url=f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                        st.write(f'{design_id} on {scale} can be resumed [WANDB LINK]({url})')
                    with col2:
                        resume_btn = st.button(f'Resume',key=f'btn_{design_id}_{scale}') #,use_container_width=True):
                    with col3:
                        restart_btn = st.button(f'Restart',key=f'btn_{design_id}_{scale}_restart') #,use_container_width=True):
                    if resume_btn:
                        run_verification(evosys.params, design_id, scale, resume=True)
                    if restart_btn:
                        run_verification(evosys.params, design_id, scale, resume=False)
                        
    if run_btn:
        run_verification(evosys.params, selected_design, scale, resume)

    st.header("Running Verification")

    if not st.session_state['running_verifications']:
        st.warning("No running verification")
    else:
        for key, process in list(st.session_state['running_verifications'].items()):
            if hasattr(process, 'poll') and process.poll() is None:  # Process is still running
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())  # Print to console in real-time
                        st.session_state['output'][key].append(output.strip())
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"Verification process for {key} is running.")
                    wandb_ids = U.load_json(U.pjoin(evosys.evo_dir, 've', key, 'wandb_ids.json'))
                    wandb_id = wandb_ids['pretrain']
                    project = wandb_ids['project']
                    entity = wandb_ids['entity']
                    url = f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                    st.write(f'View on [WANDB LINK]({url})')
                with col2:
                    if st.button(f"Stop", key=f"stop_{key}"):
                        process.terminate()
                        st.success(f"Verification process for {key} stopped.")
                        del st.session_state['running_verifications'][key]
            else:  # Process has finished
                if process.returncode == 0:
                    st.success(f"Verification process for {key} completed successfully.")
                else:
                    st.error(f"Verification process for {key} encountered an error. Check the output for details.")
                del st.session_state['running_verifications'][key]

            # Display output in Streamlit
            with st.expander(f"Output for {key}"):
                st.text("\n".join(st.session_state['output'].get(key, [])))

    # Add a refresh button to manually update the page
    if st.button("Refresh"):
        st.rerun()


    