import json
import time
import streamlit as st
import sys,os

sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU



def tester(evosys,project_dir):
    

    ### Sidebar
    with st.sidebar:
        AU.running_status(st,evosys)
        