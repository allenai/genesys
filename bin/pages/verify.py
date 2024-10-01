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
import pandas as pd
import signal

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

def _run_verification(params, design_id, scale, resume, cli=False):
    params_str = shlex.quote(json.dumps(params))
    cmd = f"python -m model_discovery.evolution --mode prep_model --params {params_str} --design_id {design_id} --scale {scale}"
    if cli:
        process = subprocess.Popen(cmd, shell=True)
    else:
        with st.spinner(f'Preparing Model...'):
            process = subprocess.run(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

    cmd = f"python -m model_discovery.evolution --mode verify --params {params_str} --design_id {design_id} --scale {scale}"
    if resume:
        cmd+=' --resume'
    if cli:
        process = subprocess.Popen(cmd, shell=True)
    else:
        st.write(f'Launching Verification with command:\n```{cmd}```')
        with st.spinner('Running... Please check the console for verification progress.'):
            # process = subprocess.run(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
            process = subprocess.Popen(cmd, shell=True)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    return process

def run_verification(params, design_id, scale, resume, cli=False):
    key = f"{design_id}_{scale}"
    # if key not in st.session_state['running_verifications']:
    if not cli: 
        polls=[p.poll() for p in st.session_state['running_verifications'].values()]
    if cli or (not None in polls):
        params = copy.deepcopy(params)
        process = _run_verification(params, design_id, scale, resume, cli=cli)
        if not cli:
            st.session_state['running_verifications'][key] = process
            # st.session_state['output'][key] = []
            st.success(f"Verification process started for {design_id} on scale {scale}. Check console for output.")
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
        if st.button("Refresh",key='refresh_btn_engine'):
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
            st.write(evosys.state)

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
        if selected_design is not None:
            vss=list(verified[selected_design].keys())
            vsstr='*(Verified on '+', '.join(vss)+')*' if len(vss)>0 else ''
        # st.write(f':red[{vsstr}]')

    with col3:
        if selected_design is not None:
            scale=st.select_slider(f"Choose a Scale :red[{vsstr}]",options=TARGET_SCALES)
        else:
            scale=st.select_slider(f"Choose a Scale",options=TARGET_SCALES,disabled=True)
            

    with col4:
        st.write('')
        st.write('')
        if selected_design is not None:
            already_verified=scale in verified[selected_design]
            txt='Run Verification' if not already_verified else 'Re-Run Verification'
            run_btn= st.button(txt,use_container_width=True,disabled=st.session_state.listening_mode or st.session_state.evo_running)
        else:
            run_btn= st.button('Run Verification',use_container_width=True,disabled=True)
    
    with col5:
        st.write('')
        st.write('')
        if selected_design is not None:
            resume=st.checkbox("Resume",value=not already_verified)
        else:
            resume=st.checkbox("Resume",value=False,disabled=True)

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
        run_verification(evosys.params, selected_design, scale, resume)



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
            
