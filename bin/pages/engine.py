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
    evosys.ptree.reload()

    with st.sidebar:
        logo_png = AU.square_logo("VER", "ENG")
        st.image(logo_png, use_column_width=True)
    
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

    st.subheader("Designs")
    # if len(verified)==0:
    #     st.write("No designs have been verified yet.")
    # for design_id in verified:
    #     for scale,verification in verified[design_id].items():
    #         st.write(f"Design: {design_id}, Scale: {scale}, Verification: {verification}")

    # st.header("Unverified Designs")
    # if len(unverified)==0:
    #     st.write("There is no unverified designs.")
    # for design_id in unverified:
    #     st.write(f"Design: {design_id}")


    