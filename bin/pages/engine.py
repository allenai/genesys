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

    
    designed=os.listdir(U.pjoin(evosys.evo_dir,'db'))
    unverified=[]
    for design_id in designed:
        report_dir=U.pjoin(evosys.evo_dir,'ve',design_id,'report.json')
        if U.load_json(report_dir)=={}:
            unverified.append(design_id)

    st.header("Verified Designs")
    if len(designed)==0:
        st.write("No designs have been created yet.")
    for design_id in designed:
        report_dir=U.pjoin(evosys.evo_dir,'ve',design_id,'report.json')
        if U.load_json(report_dir)!={}:
            st.write(f"Design: {design_id}")
            st.write(U.load_json(report_dir))

    st.header("Unverified Designs")
    if len(unverified)==0:
        st.write("There is no unverified designs.")
    for design_id in unverified:
        st.write(f"Design: {design_id}")



    with st.sidebar:
        st.write("Empty sidebar")
    