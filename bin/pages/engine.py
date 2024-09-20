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


TARGET_SCALES = ['14M','31M','70M','125M','350M','760M','1300M']



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

    # with st.sidebar:
        # wandb_api_key=st.text_input('Your Wandb API Key',type='password')
        # if wandb_api_key:
        #     os.environ['WANDB_API_KEY']=wandb_api_key

    st.header("System Info")
    col1, col2, col3 = st.columns(3)
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

    
    designed=evosys.ptree.filter_by_type(['DesignArtifactImplemented'])
    
    verified={}
    for design_id in designed:
        design=evosys.ptree.get_node(design_id)
        verifications=design.verifications
        verified[design_id]={}
        for scale,verification in verifications.items():
            verified[design_id][scale]=verification

    st.header("Verify Designs")
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
            st.balloons()



    