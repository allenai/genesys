import json
import time
import streamlit as st
import sys, os
import uuid
import signal
import threading
import queue
import random
from datetime import datetime
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


def verify_command(node_id, evosys, evoname, design_id=None, scale=None, resume=True, cli=False):
    if evosys.evoname != evoname:
        evosys.switch_ckpt(evoname)
    log_ref = evosys.remote_db.collection('experiment_logs').document(evoname)
    if design_id is None or scale is None:
        design_id,scale=evosys.select_verify()
        if design_id is None:
            msg = "No unverified design found at any scale."
            st.error(msg)
            return None,msg
    if resume:
        unfinished_verifies = evosys.get_unfinished_verifies(evoname)
        if len(unfinished_verifies) > 0:
            exp = random.choice(unfinished_verifies)
            scale=exp.split('_')[-1]
            design_id = exp[:-len(scale)-1]
    params = {'evoname': evoname}
    sess_id,pid = run_verification(params, design_id, scale, resume, cli=cli)
    timestamp=datetime.now().strftime('%B %d, %Y at %I:%M:%S %p %Z')+'_'+str(uuid.uuid4())
    if sess_id:
        log=f'Node {node_id} running verification on {design_id}_{scale}'
        log_ref.set({
            timestamp: log
        },merge=True)
    else:
        log=f'Node {node_id} failed to run verification on {design_id}_{scale} with error: {pid}'
        log_ref.set({
            timestamp: log
        },merge=True)
    print(f'{timestamp.split("_")[0]}: {log}')
    return sess_id,pid

def design_command(node_id, evosys, evoname, resume=True, cli=False):
    sess_id = None
    params = {'evoname': evoname}
    if evosys.evoname != evoname:
        evosys.switch_ckpt(evoname)
    log_ref = evosys.remote_db.collection('experiment_logs').document(evoname)
    if resume:
        unfinished_designs = evosys.ptree.get_unfinished_designs()
        if len(unfinished_designs) > 0:
            sess_id = random.choice(unfinished_designs)
    sess_id,pid = run_design_thread(evosys, sess_id, params, cli=cli)
    timestamp=datetime.now().strftime('%B %d, %Y at %I:%M:%S %p %Z')+'_'+str(uuid.uuid4())
    if sess_id:
        log=f'Node {node_id} running design thread with session id {sess_id}'
        log_ref.set({
            timestamp: log
        },merge=True)
    else:
        log=f'Node {node_id} failed to run design thread with error: {pid}'
        log_ref.set({
            timestamp: log
        },merge=True)
    print(f'{timestamp.split("_")[0]}: {log}')
    return sess_id,pid


class Listener:
    def __init__(self, evosys, node_id=None, cli=False):
        self.evosys = evosys
        remote_db = evosys.ptree.remote_db
        self.evoname = evosys.evoname
        self.node_id = node_id if node_id else str(uuid.uuid4())[:8]
        self.collection = remote_db.collection(evosys.evoname + '_connections')
        self.doc_ref = self.collection.document(self.node_id)
        self.running = False
        self.command_queue = queue.Queue()
        self.command_status = {}
        self.poll_freq = 5
        self.cli = cli
        self.active = True

    def build_connection(self):
        # check if the node_id is already in the collection
        doc = self.doc_ref.get()
        if doc.exists and doc.to_dict().get('status','n/a') == 'connected':
            self.active = False
        else:
            self.doc_ref.set({
                'status': 'connected',
                'last_heartbeat': firestore.SERVER_TIMESTAMP,
                'commands': []
            })
            self.active = True

    def listen_for_commands(self):
        self.running = True
        while self.running:
            if self.active:
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
                            if sess_id:
                                self.command_status[str(pid)] = {
                                    'command': str(command),
                                    'sess_id': str(sess_id),
                                    'status': 'running'
                                }
                        self.doc_ref.update({'commands': []})
                    
                    for pid in self.command_status:
                        process = get_process(int(pid))
                        if process and process.is_running():
                            self.command_status[str(pid)]['status'] = 'running'
                        else:
                            self.command_status[str(pid)]['status'] = 'finished'

                    self.doc_ref.update(
                        {
                            'last_heartbeat': firestore.SERVER_TIMESTAMP,
                            'command_status': self.command_status
                        })
            
            time.sleep(self.poll_freq)  
        self.cleanup()
    
    def execute_command(self, command):
        # st.write(f"Executing command: {command}")
        comps=command.split(',')
        if comps[0] == 'design':
            sess_id,pid = design_command(self.node_id, self.evosys, comps[1], resume='resume' in comps, cli=self.cli)
        elif comps[0] == 'verify':
            if len(comps) == 2 or (len(comps) == 3 and 'resume' in comps):
                sess_id,pid = verify_command(self.node_id, self.evosys, comps[1], resume='resume' in comps, cli=self.cli)
            else:
                sess_id,pid = verify_command(self.node_id, self.evosys, comps[1], comps[2], comps[3], resume='resume' in comps, cli=self.cli)
        else:
            raise ValueError(f"Unknown command: {command}")
        return sess_id,pid
        
    def stop_listening(self):
        self.running = False
        self.cleanup()

    def cleanup(self):
        # st.info("Cleaning up and disconnecting...")
        self.doc_ref.delete()  # Delete the connection document

def start_listener_thread(listener,add_ctx=True):
    thread = threading.Thread(target=listener.listen_for_commands)
    if add_ctx:
        add_script_run_ctx(thread)  # Add Streamlit context to the thread
    thread.start()
    return thread

def listen(evosys, project_dir):
    ### Sidebar
    with st.sidebar:
        AU.running_status(st, evosys)
    
    if st.session_state.evo_running:
        st.warning("**NOTE:** You are running as the master node. You cannot change the role of the node while the system is running.")


    st.title('Listening Mode')

    # Initialize session state
    if 'listener' not in st.session_state:
        st.session_state.listener = None
    if 'listener_thread' not in st.session_state:
        st.session_state.listener_thread = None
    if 'exec_commands' not in st.session_state:
        st.session_state.exec_commands = {}

    passive_mode=False  
    if st.session_state.listener and st.session_state.listener.running and not st.session_state.listener.active:
        passive_mode=True
        st.info('The listener is already running (background or still alive). You are in passive observation mode.')


    col1,_,col2,_,col3,_ = st.columns([3.5,0.1,1,0.1,1,1.5])

    with col1:
        node_id = st.text_input("Node ID (empty for random) :orange[*If a running node id in the same machine is provided, it will subscribe the status.*]", disabled=st.session_state.listening_mode)
        st.write(f'')

    with col2:
        st.write('') 
        st.write('')    
        if not st.session_state.listening_mode:
            if st.button("**Start Listening**", use_container_width=True, disabled=st.session_state.evo_running or passive_mode):
                listener = Listener(evosys, node_id)
                listener.build_connection()
                st.session_state.listener = listener
                st.session_state.listener_thread = start_listener_thread(listener)
                st.session_state.listening_mode = True
                st.success(f"Listening started. Node ID: {listener.node_id}")
                st.rerun()
        else:
            if st.button("**Stop Listening**", use_container_width=True, disabled=st.session_state.evo_running or passive_mode):
                if st.session_state.listener:
                    st.session_state.listener.stop_listening()
                    st.session_state.listener_thread.join()
                    st.session_state.listener = None
                    st.session_state.listener_thread = None
                    st.session_state.listening_mode = False
                    st.success("Listening stopped.")
                    st.rerun() # Add this section to process commands in the main Streamlit thread

        if st.session_state.listener:
            if st.session_state.listener.node_id not in st.session_state.exec_commands:
                st.session_state.exec_commands[st.session_state.listener.node_id] = {}
            if st.session_state.listener.active:
                try:
                    while True:
                        st.toast("Listening for commands...")
                        command,sess_id,pid = st.session_state.listener.command_queue.get_nowait()
                        if not sess_id:
                            st.toast(f"Command {command} failed. {pid}")
                        else:
                            st.toast(f"Command {command} started. {pid}")
                            st.session_state.exec_commands[st.session_state.listener.node_id][pid] = command, sess_id
                except queue.Empty:
                    pass
            else:
                doc = st.session_state.listener.doc_ref.get()
                command_status = doc.to_dict() if doc.exists else {}
                for pid,status in command_status.items():
                    st.session_state.exec_commands[st.session_state.listener.node_id][pid] = status['command'], status['sess_id']
            for pid in st.session_state.exec_commands[st.session_state.listener.node_id]:
                command,sess_id = st.session_state.exec_commands[st.session_state.listener.node_id][pid]
                ctype = command.split()[0]
                if ctype == 'design':
                    st.session_state['design_threads'][sess_id] = get_process(pid)
                elif ctype == 'verify':
                    st.session_state['running_verifications'][sess_id] = get_process(pid)

    with col3:
        st.write('')
        st.write('')
        st.button("*Refresh status*", use_container_width=True)

    st.subheader("Commands status")

    if len(st.session_state.exec_commands) > 0:
        
        col1,col2 = st.columns([2,1])
        with col1:
            all_nodes = list(st.session_state.exec_commands.keys())
            selected_node = st.selectbox("Select node ID", all_nodes,index=all_nodes.index(st.session_state.listener.node_id))
            st.write(':orange[*View details in **Design**, **Verify**, and **Viewer** tabs.*]')
        with col2:
            st.write('')
            st.write('')

        active_commands,inactive_commands = [],[]
        for pid in st.session_state.exec_commands[selected_node]:
            command,sess_id = st.session_state.exec_commands[selected_node][pid]
            process = get_process(pid)
            if process and process.is_running():
                active_commands.append((command,sess_id,pid))
            else:
                inactive_commands.append((command,sess_id,pid))
        
        if len(active_commands) == 0 and len(inactive_commands) == 0:
            st.info("No commands running or finished.")

        def show_commands(commands):
            for command,sess_id,pid in commands:
                cols=st.columns([0.6,0.05,1,0.05,2])
                ctype = command.split(',')[0]
                args = command.split(',')[1:]
                with cols[0]:
                    st.write(f'PID: ```{pid}```')
                with cols[2]:
                    if ctype == 'design':
                        st.write(f"Command: ```Design```")
                    elif ctype == 'verify':
                        st.write(f"Command: ```Verify```")
                with cols[4]:
                    if ctype == 'design':
                        st.write(f"Design Session: ```{sess_id}```")
                    elif ctype == 'verify':
                        st.write(f"Verified Model: ```{sess_id}```")

        col1,col2 = st.columns([1,1])
        with col1:
            with st.expander("Running Commands"):
                if len(active_commands) == 0:
                    st.info("No running commands.")
                else:
                    show_commands(active_commands)

        with col2:
            with st.expander("Finished Commands"):
                if len(inactive_commands) == 0:
                    st.info("No finished commands.")
                else:
                    show_commands(inactive_commands)

    else:
        st.info("No listener runned.")


if __name__ == "__main__":
    from model_discovery.evolution import BuildEvolution

    print("Running in CLI mode.")

    # run in CLI mode
    evosys = BuildEvolution(
        params={'evoname':'test_evo_000'}, # doesnt matter, will switch to the commanded evoname
        do_cache=False,
        # cache_type='diskcache',
    )

    listener = Listener(evosys, cli=True)
    listener.build_connection()
    listener_thread = start_listener_thread(listener,add_ctx=False)

    print("Listening started.")
    try:
        # Keep the main thread alive
        while listener.active:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        listener.stop_listening()
        listener_thread.join(timeout=10)  # Wait for the thread to finish, with a timeout

    print("Listener stopped.")



