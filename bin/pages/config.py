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




def config(evosys,project_dir):

    st.title("System Management")

    with st.expander("Evolution System Config",expanded=True):
        with st.form("Evolution System Config"):
            col1,col2=st.columns(2)
            with col1:
                params={}
                params['evoname']=st.text_input('Experiment Name',value=evosys.params['evoname'])
                params['scales']=st.text_input('Scales',value=evosys.params['scales'])
                params['selection_ratio']=st.slider('Selection Ratio',min_value=0.0,max_value=1.0,value=evosys.params['selection_ratio'])
                params['select_method']=st.text_input('Select Method',value=evosys.params['select_method'])
                params['design_budget']=st.number_input('Design Budget ($)',value=evosys.params['design_budget'],min_value=0)
            with col2:
                st.write("Current Config:")
                st.write(evosys.params)

                submitted = st.form_submit_button("Reload")
                if submitted:
                    with st.spinner('Reloading...'):
                        evosys.reload(params)
    # with col2:
    #     with st.expander("Current Config",expanded=True):
    #         st.write(evosys.params)
    
    # ve_pages = {
    #     "Your account": [
    #         st.Page(mock1,title="Create your account"),
    #         st.Page(mock2,title="Manage your account"),
    #     ],
    #     "Resources": [
    #         st.Page(mock3,title="Learn about us"),
    #         st.Page(mock4,title="Try it out"),
    #     ],
    # }

    # pg = st.navigation(ve_pages)
    # pg.run()
    
    with st.sidebar:
        logo_png = AU.square_logo("SYS", "CFG")
        st.image(logo_png, use_column_width=True)
