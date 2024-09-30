import json
import time
import streamlit as st
import sys, os
import uuid
import signal
import threading
sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU
from google.cloud import firestore

from bin.pages.design import run_design_thread
from bin.pages.verify import run_verification
from bin.pages.config import download_exp_from_db


# Comments
# 1. Load exp
# 2. Run design
# 3. Run verify


class Listener:
    def __init__(self, evoname, remote_db):
        self.evoname = evoname
        self.node_id = str(uuid.uuid4())[:8]
        self.collection = remote_db.collection(evoname + '_connections')
        self.doc_ref = self.collection.document(self.node_id)
        self.running = False
        
    def build_connection(self):
        self.doc_ref.set({
            'status': 'connected',
            'last_heartbeat': firestore.SERVER_TIMESTAMP,
            'commands': []
        })
        
    def listen_for_commands(self):
        self.running = True
        while self.running:
            doc = self.doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                if data['status'] == 'disconnected':
                    break
                
                commands = data.get('commands', [])
                if commands:
                    self.execute_commands(commands)
                    self.doc_ref.update({'commands': []})
                
                self.doc_ref.update({'last_heartbeat': firestore.SERVER_TIMESTAMP})
            
            time.sleep(5)  # Poll every 5 seconds
        self.cleanup()
    
    def execute_commands(self, commands):
        for command in commands:
            # Execute the command here
            print(f"Executing command: {command}")
            # You can add more complex command execution logic here

    def stop_listening(self):
        self.running = False

    def cleanup(self):
        print("Cleaning up and disconnecting...")
        self.doc_ref.delete()  # Delete the connection document


def start_listener_thread(listener):
    thread = threading.Thread(target=listener.listen_for_commands)
    thread.start()
    return thread

def listen(evosys, project_dir):
    ### Sidebar
    with st.sidebar:
        AU.running_status(st, evosys)

    st.title('Listener Control')

    # Initialize session state
    if 'listener' not in st.session_state:
        st.session_state.listener = None
    if 'listener_thread' not in st.session_state:
        st.session_state.listener_thread = None

    if not st.session_state.listening_mode:
        if st.button("Start Listening"):
            listener = Listener(evosys.evoname, evosys.ptree.remote_db)
            listener.build_connection()
            st.session_state.listener = listener
            st.session_state.listener_thread = start_listener_thread(listener)
            st.session_state.listening_mode = True
            st.success(f"Listening started. Node ID: {listener.node_id}")
            st.rerun()
    else:
        if st.button("Stop Listening"):
            if st.session_state.listener:
                st.session_state.listener.stop_listening()
                st.session_state.listener_thread.join()
                st.session_state.listener = None
                st.session_state.listener_thread = None
                st.session_state.listening_mode = False
                st.success("Listening stopped.")
                st.rerun()

    # Example of disabling features when in listening mode
    if st.session_state.listening_mode:
        st.warning("Some features are disabled while in listening mode.")
        st.text_input("Disabled input", value="This input is disabled", disabled=True)
    else:
        st.text_input("Enabled input", value="This input is enabled")

    # Other UI components can be added here
    st.write("This part of the UI is always accessible.")

# ... existing code ...