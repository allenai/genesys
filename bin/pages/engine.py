import json
import time
import pathlib
import streamlit as st
import sys,os
import torch
import platform
import psutil


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)

from model_discovery.ve.data_loader import load_datasets


TARGET_SCALES = ['14M','31M','70M','125M','350M','760M','1300M']

SMOLLM_125_CORPUS=['cosmopedia-v2','python-edu','fineweb-edu-dedup','open-web-math','deepmind-math-small','stackoverflow-clean']



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



def engine(evosys,project_dir):

    st.title("Verification Engine")

    # evosys.ptree.load()

    with st.sidebar:
        st.write(f'**Namespace: ```{evosys.evoname}```**')
        

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

    with st.expander("Diagnostic Panel"):
        # st.subheader("Test loading datasets")
        config=GAMConfig_14M()
        st.subheader("Test dataset loader")
        col1,_,col3,_,col4,_=st.columns([3,0.5,2,0.5,2,1])
        with col1:
            dataset_name=st.selectbox("Choose a Dataset",options=SMOLLM_125_CORPUS)
        with col3:
            st.write('')
            st.write('')
            if st.button("Test Loading "+dataset_name,use_container_width=True):
                config.training_data=[dataset_name]
                load_datasets(config)
        with col4:
            st.write('')
            st.write('')
            if st.button("Test Loading All Datasets",use_container_width=True):
                config.training_data=SMOLLM_125_CORPUS
                load_datasets(config)

    
    designed=evosys.ptree.filter_by_type(['DesignArtifactImplemented'])
    
    verified={}
    for design_id in designed:
        design=evosys.ptree.get_node(design_id)
        verifications=design.verifications
        verified[design_id]={}
        for scale,verification in verifications.items():
            verified[design_id][scale]=verification

    st.header("Verify Designs")
    if len(verified)==0:
        st.warning('No implemented designs found in the experiment directory')
    else:
        col1,col2,col3=st.columns(3)
        with col1:
            design_id=st.selectbox("Design",options=verified.keys())
            vss=list(verified[design_id].keys())
            vsstr='*Verified on '+', '.join(vss)+'*' if len(vss)>0 else ''
            st.write(vsstr)

        with col2:
            options=[]
            for scale in TARGET_SCALES:
                if scale not in verified[design_id]:
                    options.append(scale)
            tail=' (excluded verified scales)' if len(vss)>0 else ''
            scale=st.select_slider("Scale"+tail,options=options)
        with col3:
            st.write('')
            st.write('')
            if st.button("Run Verification"):
                st.write("Hello")

    st.header("Running Verifications")

    st.warning("No running verifications")
    





    