import json
import time
import streamlit as st
import sys, os
import uuid
import signal
import threading
import queue
import random
import psutil
sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU
from google.cloud import firestore
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime import get_instance

from bin.pages.design import run_design_thread
from bin.pages.verify import run_verification


# Comments
# 1. Run/stop design
# 2. Run/stop verify

def get_process(pid):
    try:
        return psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None


def verify_command(evosys, evoname, design_id, scale, resume=True):
    if evosys.evoname != evoname:
        evosys.switch_ckpt(evoname)
    if resume:
        unfinished_verifies = evosys.get_unfinished_verifies(evoname)
        if len(unfinished_verifies) > 0:
            exp = random.choice(unfinished_verifies)
            scale=exp.split('_')[-1]
            design_id = exp[:-len(scale)-1]
    params = {'evoname': evoname}
    sess_id,pid = run_verification(params, design_id, scale, resume)
    return sess_id,pid

def design_command(evosys, evoname, resume=True):
    sess_id = None
    params = {'evoname': evoname}
    if evosys.evoname != evoname:
        evosys.switch_ckpt(evoname)
    if resume:
        unfinished_designs = evosys.ptree.get_unfinished_designs()
        if len(unfinished_designs) > 0:
            sess_id = random.choice(unfinished_designs)
    sess_id,pid = run_design_thread(evosys, sess_id, params)
    return sess_id,pid


class Listener:
    def __init__(self, evosys, node_id=None):
        self.evosys = evosys
        remote_db = evosys.ptree.remote_db
        self.evoname = evosys.evoname
        self.node_id = node_id if node_id else str(uuid.uuid4())[:8]
        self.collection = remote_db.collection(evosys.evoname + '_connections')
        self.doc_ref = self.collection.document(self.node_id)
        self.running = False
        self.command_queue = queue.Queue()
        self.poll_freq = 5
        
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
                    for command in commands:
                        sess_id,pid = self.execute_command(command)
                        self.command_queue.put((command,sess_id,pid))
                    self.doc_ref.update({'commands': []})
                
                self.doc_ref.update({'last_heartbeat': firestore.SERVER_TIMESTAMP})
            
            time.sleep(self.poll_freq)  
        self.cleanup()
    
    def execute_command(self, command):
        st.write(f"Executing command: {command}")
        comps=command.split(',')
        if comps[0] == 'design':
            sess_id,pid = design_command(self.evosys, comps[1], resume='resume' in comps)
        elif comps[0] == 'verify':
            sess_id,pid = verify_command(self.evosys, comps[1], comps[2], comps[3], resume='resume' in comps)
        else:
            raise ValueError(f"Unknown command: {command}")
        return sess_id,pid
        
    def stop_listening(self):
        self.running = False

    def cleanup(self):
        st.info("Cleaning up and disconnecting...")
        self.doc_ref.delete()  # Delete the connection document


def start_listener_thread(listener):
    thread = threading.Thread(target=listener.listen_for_commands)
    add_script_run_ctx(thread)  # Add Streamlit context to the thread
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
    if 'exec_commands' not in st.session_state:
        st.session_state.exec_commands = {}

    col1,col2 = st.columns(2)

    with col1:
        node_id = st.text_input("Node ID (manual or random)", disabled=st.session_state.listening_mode)

    with col2:
        st.write('') 
        st.write('')    
        if not st.session_state.listening_mode:
            if st.button("Start Listening"):
                listener = Listener(evosys, node_id)
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
                    st.rerun() # Add this section to process commands in the main Streamlit thread

        if st.session_state.listener:
            try:
                while True:
                    st.toast("Listening for commands...")
                    command,sess_id,pid = st.session_state.listener.command_queue.get_nowait()
                    if not sess_id:
                        st.toast(f"Command {command} failed. {pid}")
                    else:
                        st.toast(f"Command {command} started. {pid}")
                        st.session_state.exec_commands[pid] = command, sess_id
                        ctype = command.split()[0]
                        if ctype == 'design':
                            st.session_state['design_threads'][sess_id] = get_process(pid)
                        elif ctype == 'verify':
                            st.session_state['running_verifications'][sess_id] = get_process(pid)
            except queue.Empty:
                pass

    st.subheader("Commands status")

    active_commands,inactive_commands = [],[]
    for pid,cmd_sess in st.session_state.exec_commands.items():
        command,sess_id = cmd_sess
        if get_process(pid).poll() is None:
            active_commands.append((command,sess_id))
        else:
            inactive_commands.append((command,sess_id))
    
    if len(active_commands) == 0 and len(inactive_commands) == 0:
        st.info("No commands running or finished.")

    with st.expander("Running Commands"):
        for command,sess_id in active_commands:
            st.write(f"{command} - {sess_id} - {pid}")

    with st.expander("Finished Commands"):
        for command,sess_id in inactive_commands:
            st.write(f"{command} - {sess_id} - {pid}")
