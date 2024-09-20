import json
import time
import pathlib
import streamlit as st
import sys,os
import inspect
import pyflowchart as pfc
import streamlit.components.v1 as components
from subprocess import check_output

sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

TARGET_SCALES = ['14M','31M','70M','125M','350M','760M','1300M']



def config(evosys,project_dir):

    st.title("System Management")


    st.subheader("Environment Settings")

    # with st.expander("Environment Variables",expanded=True):
    #     with st.form("Environment Variables"):


    st.subheader("Evolution System Settings")

    SELECT_METHODS = ['random']
    
    config={}
    with st.expander("Evolution System Config",expanded=True):
        with st.form("Evolution System Config"):
            col1,col2=st.columns(2)
            with col1:
                params={}
                params['evoname']=st.text_input('Experiment Name',value=evosys.params['evoname'])
                target_scale=st.select_slider('Target Scale',options=TARGET_SCALES,value=evosys.params['scales'].split(',')[-1])
                scales=[]
                for s in TARGET_SCALES:
                    if int(target_scale.replace('M',''))>=int(s.replace('M','')):
                        scales.append(s)
                params['scales']=','.join(scales)
                params['selection_ratio']=st.slider('Selection Ratio',min_value=0.0,max_value=1.0,value=evosys.params['selection_ratio'])
                params['select_method']=st.selectbox('Seed Selection Method',options=SELECT_METHODS,index=SELECT_METHODS.index(evosys.params['select_method']))
                params['design_budget']=st.number_input('Design Budget ($)',value=evosys.params['design_budget'],min_value=0,step=100)
                config['params']=params
                st.session_state['EVOSYS_PARAMS']=params

            with col2:
                st.write("Current Settings:")
                settings={}
                settings['Experiment Directory']=evosys.evo_dir
                settings['Seed Selection Method']=evosys.select_method
                settings['Design Budget']=evosys.design_budget_limit
                settings['Verification Budges']=evosys.state['budgets']
                st.write(settings)

                submitted = st.form_submit_button("Apply")
                if submitted:
                    with st.spinner('Reloading...'):
                        evosys.reload(params)


    st.subheader("Existing Experiments")

    
    ckpts=U.listdir(evosys.ckpt_dir)
    


    
    with st.sidebar:

        st.download_button(
            label="Download your config",
            data=json.dumps(config),
            file_name="config.json",
            mime="text/json",
            use_container_width=True
        )

        uploaded_file = st.file_uploader(
            "Upload your config",
            type=['json'],
            accept_multiple_files=False,
            # use_container_width=True
        )

        if uploaded_file is not None:
            config = json.load(uploaded_file)
            with st.expander("Loaded Config",expanded=True):
                st.write(config)
