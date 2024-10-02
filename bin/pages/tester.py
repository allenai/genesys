
import json
import time
import streamlit as st
import sys,os
import random
sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.evolution import ConnectionManager



def tester(evosys,project_dir):
    
    ### Sidebar
    with st.sidebar:
        AU.running_status(st,evosys)

    st.title('Testing page (for internal use)')

    ### Connection manager
    CM = ConnectionManager(evosys.evoname, evosys.ptree.remote_db, st)


    ### Send command to all active connections
    col1,col2=st.columns([1,2])
    with col1:
        selected_conn=st.selectbox('Select connection',options=CM.get_active_connections())
    with col2:
        command_type=st.selectbox('Select command type',options=['design','verify'])
    
    
    if command_type == 'design':
        col1,col2=st.columns([1,3])
        with col1:
            send_btn = st.button('Send command to selected connection')
        with col2:
            resume = st.checkbox('Resume', value=True)
        if send_btn:
            CM.design_command(selected_conn,resume)
    elif command_type == 'verify':
        col1,col2,col3,col4=st.columns([1,1,1,1])
        with col1:
            budget = evosys.available_verify_budget
            scale = st.selectbox('Select scale',options=list(budget.keys()))
        unverified = evosys.ptree.get_unverified_designs(scale)
        with col2:
            design_id = st.selectbox('Select design',options=['random']+unverified)
            if design_id == 'random':
                design_id = random.choice(unverified)
        with col3:
            st.write('')
            st.write('')
            send_btn = st.button('Send command to selected connection')
        with col4:
            st.write('')
            st.write('')
            resume = st.checkbox('Resume', value=True)
        if send_btn:
            CM.verify_command(selected_conn, design_id, scale, resume)

    st.subheader('Connection Status')
    st.session_state['listener_connections']=CM.connections
    if CM.connections:
        st.write(CM.connections)
    else:
        st.info('No connections')


    st.write(evosys.ptree.get_node('sparsitron-x'))