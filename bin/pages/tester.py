
import json
import time
import streamlit as st
import sys,os

sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU
from model_discovery.evolution import FirestoreManager






 

def tester(evosys,project_dir):
    
    ### Sidebar
    with st.sidebar:
        AU.running_status(st,evosys)

    fs=FirestoreManager(evosys.evoname,evosys.ptree.db_dir,evosys.ptree.remote_db)

    overwrite=st.checkbox('Overwrite',value=False)
    if st.button('Sync DB'):
        fs.sync_to_db(overwrite=overwrite)

    if st.button('Get Index'):
        si=fs.compress_index(fs.index)
        st.write(len(str(si)))
        st.write(si)
        di=fs.decompress_index(si)
        st.write(di==fs.index)
    
    selected_designs=st.selectbox('Select Designs',options=list(fs.index.keys()))
    if st.button('Load Design'):
        design=fs.get_design(selected_designs)
        st.write(design)

    if st.button('Show Cache'):
        st.write(fs.cache.keys())
