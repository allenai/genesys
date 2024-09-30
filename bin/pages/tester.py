
import json
import time
import streamlit as st
import sys,os

sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU

from google.cloud import firestore
from datetime import datetime, timedelta



class ConnectionManager:
    def __init__(self, evoname, remote_db):
        self.evoname = evoname
        self.collection = remote_db.collection(evoname + '_connections')
        self.zombie_threshold = 90  # seconds

    def clear_zombie_connections(self):
        threshold_time = datetime.utcnow() - timedelta(seconds=self.zombie_threshold)
        zombie_connections = self.collection.where('last_heartbeat', '<', threshold_time).get()
        
        for doc in zombie_connections:
            print(f"Clearing zombie connection: {doc.id}")
            doc.reference.delete()
    
    def get_active_connections(self):
        return self.collection.where('status', '==', 'connected').get()
    
    def send_command(self, node_id, command):
        node_ref = self.collection.document(node_id)
        node_ref.update({
            'commands': firestore.ArrayUnion([command])
        })
    
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

    ### Get active connections
    if st.button('Get active connections'):
        active_connections = CM.get_active_connections()
        for connection in active_connections:
            st.write(connection.to_dict())

    ### Send command to all active connections
    command = st.text_input('Enter command to send to all active connections')
    if st.button('Send command to all active connections'):
        for connection in active_connections:
            CM.send_command(connection.id, command)

    ### Clear zombie connections
    if st.button('Clear zombie connections'):
        CM.clear_zombie_connections()

    