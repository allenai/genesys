import json
import time
import pathlib
import copy
import streamlit as st
import sys,os
import torch
import platform
import psutil
import multiprocessing
from multiprocessing import Process, Value, Queue
import ctypes
import signal

sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

from model_discovery.ve.data_loader import load_datasets
from model_discovery.evolution import BuildEvolution


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


def run_verification(params, design_id, scale, resume, stop_flag, completion_queue):
    def signal_handler(signum, frame):
        stop_flag.value = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Recreate the evosys object in this process
        _evosys = BuildEvolution(params=params, do_cache=False, stream=st)
        _evosys.verify(design_id, scale, resume=resume)
        completion_queue.put(("completed", None))
    except Exception as e:
        completion_queue.put(("error", str(e)))
    finally:
        stop_flag.value = True


def engine(evosys,project_dir):

    st.title("Verification Engine")

    # evosys.ptree.load()

    if 'running_verifications' not in st.session_state:
        st.session_state['running_verifications'] = {}



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
        ckpt_dir=U.pjoin(evosys.evo_dir,'ve',selected_design+'_'+scale)
        txt='Run Verification' if not already_verified else 'Re-Run Verification'
        if not already_verified and U.pexists(ckpt_dir):
            txt='Resume Verification'
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
                    col1,col2=st.columns([0.6,0.4])
                    with col1:
                        st.write(f'{design_id} on {scale} is running, wandb id: {wandb_ids}')
                    with col2:
                        st.write('')
                        st.write('')
                        if st.button(f'Stop/Resume'):#,use_container_width=True):
                            st.write('WIP')

    if run_btn:
        if f"{selected_design}_{scale}" not in st.session_state['running_verifications']:
            stop_flag = Value(ctypes.c_bool, False)
            completion_queue = Queue()
            params = copy.deepcopy(evosys.params)
            process = Process(target=run_verification, args=(params, selected_design, scale, resume, stop_flag, completion_queue))
            process.start()
            st.session_state['running_verifications'][f"{selected_design}_{scale}"] = {
                'process': process,
                'stop_flag': stop_flag,
                'completion_queue': completion_queue
            }
            st.success(f"Verification process started for {selected_design} on scale {scale}.")
        else:
            st.warning(f"A verification process for {selected_design} on scale {scale} is already running.")

        
    st.header("Running Verifications")

    if not st.session_state['running_verifications']:
        st.warning("No running verifications")
    else:
        for key, verification in list(st.session_state['running_verifications'].items()):
            process = verification['process']
            stop_flag = verification['stop_flag']
            completion_queue = verification['completion_queue']

            if not completion_queue.empty():
                status, message = completion_queue.get()
                if status == "completed":
                    st.success(f"Verification process for {key} completed successfully.")
                elif status == "error":
                    st.error(f"Verification process for {key} encountered an error: {message}")
                del st.session_state['running_verifications'][key]
            elif process.is_alive():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"Verification process for {key} is running.")
                with col2:
                    if st.button(f"Stop", key=f"stop_{key}"):
                        stop_flag.value = True
                        process.join(timeout=5)
                        if process.is_alive():
                            process.terminate()
                        del st.session_state['running_verifications'][key]
                        st.success(f"Verification process for {key} stopped.")
            else:
                del st.session_state['running_verifications'][key]





    