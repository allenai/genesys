
import json
import time
import streamlit as st
import sys,os
import random
sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU

from google.cloud import firestore
from datetime import datetime, timedelta



class ConnectionManager:
    def __init__(self, evoname, remote_db):
        self.evoname = evoname
        self.collection = remote_db.collection(evoname + '_connections')
        self.zombie_threshold = 30  # seconds
        self.max_design_threads_per_node = 3

    def clear_zombie_connections(self):
        threshold_time = datetime.utcnow() - timedelta(seconds=self.zombie_threshold)
        zombie_connections = self.collection.where('last_heartbeat', '<', threshold_time).get()
        
        for doc in zombie_connections:
            print(f"Clearing zombie connection: {doc.id}")
            doc.reference.delete()
    
    def get_active_connections(self):
        self.clear_zombie_connections()
        connections = self.collection.where('status', '==', 'connected').get()
        self.connections = {c.id: c.to_dict() for c in connections}
        return list(self.connections.keys())

    def check_command_status(self,node_id):
        node_data = self.connections[node_id]
        if node_data and 'command_status' in node_data:
            return node_data['command_status']
        else:
            return None

    def design_command(self,node_id,resume=True):
        command_status = self.check_command_status(node_id)
        running_designs=[]
        if command_status:
            running_designs=[]
            for pid in command_status:
                command = command_status[pid]
                if command['command'].startswith('design') and command['status'] == 'running':
                    running_designs.append(pid)
        if len(running_designs) >= self.max_design_threads_per_node:
            st.toast(f"Max number of design threads reached ({self.max_design_threads_per_node}) for node {node_id}. Please wait for some threads to finish.",icon='🚨')
            return
        command = f'design,{self.evoname}'
        if resume:
            command += ',resume'
        self.send_command(node_id,command)
    
    def verify_command(self,node_id,design_id,scale,resume=True):
        command_status = self.check_command_status(node_id)
        running_verifies=[]
        if command_status:
            running_verifies=[]
            for pid in command_status:
                command = command_status[pid]
                if command['command'].startswith('verify') and command['status'] == 'running':
                    running_verifies.append(pid)
        if len(running_verifies) > 0:
            st.toast(f"There is already a verification running for node {node_id}. Please wait for it to finish.",icon='🚨')
            return
        command = f'verify,{self.evoname},{design_id},{scale}'
        if resume:
            command += ',resume'
        self.send_command(node_id,command)
    
    def send_command(self, node_id, command):
        self.get_active_connections()
        try:
            node_ref = self.collection.document(node_id)
            update_time = node_ref.update({
                'commands': firestore.ArrayUnion([command]),
                'last_command_sent': firestore.SERVER_TIMESTAMP
            })
            st.toast(f"Command sent to {node_id}: {command}")
            st.toast(f"Update time: {update_time}")
            return True
        except Exception as e:
            st.toast(f"Error sending command to {node_id}: {str(e)}",icon='🚨')
            return False

    def disconnect_node(self, node_id):
        node_ref = self.collection.document(node_id)
        node_ref.update({'status': 'disconnected'})

 

def tester(evosys,project_dir):
    
    ### Sidebar
    with st.sidebar:
        AU.running_status(st,evosys)

    st.title('Testing page (for internal use)')

    ### Connection manager
    CM = ConnectionManager(evosys.evoname, evosys.ptree.remote_db)


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
    st.write(CM.connections)
