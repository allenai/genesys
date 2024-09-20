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

    st.title("System Configurations")
    
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
