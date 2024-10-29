import json
import time
import streamlit as st
import sys, os
import uuid
import signal
import threading
import queue
import random
from datetime import datetime, timedelta
import pytz
import psutil
import getmac
import torch
sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU
from google.cloud import firestore
from streamlit.runtime.scriptrunner import add_script_run_ctx

from bin.pages.design import design_command,DESIGN_TERMINAL_STATES,DESIGN_ACTIVE_STATES,DESIGN_ZOMBIE_THRESHOLD
from bin.pages.verify import verify_command,VERIFY_TERMINAL_STATES,VERIFY_ACTIVE_STATES,VERIFY_ZOMBIE_THRESHOLD


def get_process(pid):
    try:
        return psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None

def is_running(pid):
    process = get_process(pid)
    if process:
        if str(process.status()) in ['running','disk-sleep', 'waiting', 'waking']:
            return True
    return False


NODE_POLL_FREQ = 15 # seconds
NODE_ZOMBIE_THRESHOLD = AU.NODE_ZOMBIE_THRESHOLD
NODE_EXECUTION_DELAY = 2

class Listener:
    def __init__(self, evosys, node_id=None, group_id='default', max_design_threads=5, accept_verify_job=True, 
                 accept_baselines=False, cpu_only=False, silent=False, cli=False, free_verifier=False):
        self.evosys = evosys
        remote_db = evosys.ptree.remote_db
        self.evoname = evosys.evoname
        self.collection = remote_db.collection('working_nodes')
        self.running = False
        self.command_queue = queue.Queue()
        self.poll_freq = NODE_POLL_FREQ
        self.cli = cli
        self.active_mode = True
        self.zombie_threshold = NODE_ZOMBIE_THRESHOLD  # seconds
        self.execution_delay = NODE_EXECUTION_DELAY # seconds
        self.local_dir = U.pjoin(evosys.ckpt_dir,'.node.json')
        self.max_design_threads = max_design_threads
        self.accept_verify_job = accept_verify_job # XXX: if accepting verify jobs, may not allow design jobs to use GPUs?
        self.accept_baselines = accept_baselines
        self.cpu_only = cpu_only
        self.silent = silent
        self.group_id = group_id
        self.free_verifier = free_verifier
        self.initialize(node_id)

    def hanging(self):
        # assert not self.active_mode
        self.active_mode = False
        self.node_id = None
        self.doc_ref = None

    def wake_up(self,node_id):
        # assert not self.active_mode
        self.active_mode = False
        self.node_id = node_id
        self.doc_ref = self.collection.document(self.node_id)

    def initialize(self,node_id=None):
        local_doc = U.load_json(self.local_dir)
        running_node_id = AU._listener_running(self.evosys.ckpt_dir,self.zombie_threshold)
        if running_node_id:
            self.node_id = running_node_id
            self.group_id = local_doc['group_id']
            print(f'There is already a listener running in this machine/userspace, Node ID: {self.node_id}, will run in passive mode. Please check in GUI.')
            self.active_mode = False
        else:
            self.node_id = self._assign_node_id(node_id)
            if node_id and node_id != self.node_id:
                print(f'Node ID {node_id} is in use. Automatically assigned to {self.node_id} instead.')
            local_doc['node_id'] = self.node_id
            local_doc['group_id'] = self.group_id
            local_doc['max_design_threads'] = self.max_design_threads
            local_doc['accept_verify_job'] = self.accept_verify_job
            local_doc['cpu_only_checker'] = self.cpu_only
            U.save_json(local_doc,self.local_dir)
        self.doc_ref = self.collection.document(self.node_id)
    
    def _assign_node_id(self,node_id=None,autofix=False):
        if not node_id:
            node_id = str(uuid.uuid4())[:6]
        count=0
        while True:
            tail='_'+str(count) if count>0 else ''
            doc_ref=self.collection.document(node_id+tail)
            if not doc_ref.get().exists:
                return node_id+tail
            else:
                if autofix:
                    count+=1
                else:
                    raise ValueError(f'Node ID {node_id+tail} is already in use.')

    def build_connection(self):
        # check if the node_id is already in the collection
        doc = self.doc_ref.get()
        local_doc = U.load_json(self.local_dir)
        if doc.exists and doc.to_dict().get('status','n/a') == 'connected':
            self.active_mode = False
        else:
            self.reset_doc()
            self.active_mode = True
            local_doc['last_heartbeat'] = str(datetime.now(pytz.UTC))
            local_doc['status'] = 'running'
            U.save_json(local_doc,self.local_dir)

    
    def reset_doc(self, reset_commands=True):
        gpu_str = f'{torch.cuda.device_count()}x{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU'
        to_set = {
            'status': 'connected',
            'group_id': self.group_id,
            'mac_address': getmac.get_mac_address(),
            'last_heartbeat': firestore.SERVER_TIMESTAMP,
            'max_design_threads': self.max_design_threads,
            'accept_verify_job': self.accept_verify_job,
            'cpu_only_checker': self.cpu_only,
            'gpu': gpu_str
        }
        if reset_commands:
            to_set['commands'] = []
        self.doc_ref.set(to_set,merge=True)

    def listen_for_commands(self):
        self.running = True
        local_doc = U.load_json(self.local_dir)
        while self.running:
            to_sleep = self.poll_freq
            if self.active_mode:
                print(f' [{self.node_id}: {time.strftime("%Y-%m-%d %H:%M:%S")}] is listening for commands')
                self.reset_doc(reset_commands=False)
                if not self.node_id:
                    print(f' [{self.node_id}: {time.strftime("%Y-%m-%d %H:%M:%S")}] Hanging...')
                    time.sleep(self.poll_freq)
                    continue
                
                doc = self.doc_ref.get()
                if doc.exists:
                    data = doc.to_dict()
                    if 'status' in data and data['status'] in ['disconnected','stopped']:
                        break
                    
                    commands = data.get('commands', [])
                    if commands:
                        for command in commands:
                            print(f'[{self.node_id}: {time.strftime("%Y-%m-%d %H:%M:%S")}] Executing command: {command}')
                            evoname = command.split(',')[1]
                            self.evosys.CM.switch_ckpt(evoname)
                            sess_id,pid = self.execute_command(command)
                            if sess_id is not None:
                                time.sleep(self.execution_delay)
                                to_sleep -= self.execution_delay
                                self.command_queue.put((command,sess_id,pid))

                    local_doc['last_heartbeat'] = str(datetime.now(pytz.UTC))
                    U.save_json(local_doc,self.local_dir)

                self.reset_doc()

            if to_sleep>0:  
                time.sleep(to_sleep)  
        self.cleanup()

    def get_running_design_sessions(self,ret_raw=False):
        running_sessions = []
        raw={}
        design_workloads = self.evosys.CM.check_design_workload(self.node_id)
        for sess_id in design_workloads:
            item = design_workloads[sess_id]
            running_sessions.append(item['sess_id'])
            raw[item['sess_id']] = item
        if ret_raw:
            return running_sessions,raw
        else:
            return running_sessions
    
    def execute_command(self, command):
        # st.write(f"Executing command: {command}")
        comps=command.split(',')
        if comps[0] == 'design':
            running_sessions = self.get_running_design_sessions()
            if len(running_sessions) > self.max_design_threads:
                print(f"There are already {len(running_sessions)} design jobs running. Please wait for them to finish.")
                return None,None
            sess_id,pid = design_command(self.node_id, self.evosys, comps[1], resume='resume' in comps, cli=self.cli, 
                                         cpu_only=self.cpu_only, silent=self.silent, running_sessions=running_sessions)
        elif comps[0] == 'verify':
            verify_workloads = self.evosys.CM.check_verification_workload(self.node_id)
            if len(verify_workloads) > 0:
                for sess_id in verify_workloads:
                    if 'W&B Training Run' not in verify_workloads[sess_id]:
                        ve_dir = U.pjoin(self.evosys.evo_dir, 've', sess_id)
                        if os.path.exists(ve_dir):
                            wandb_ids = U.load_json(U.pjoin(ve_dir, 'wandb_ids.json'))
                            wandb_id=wandb_ids['pretrain']['id']
                            project=wandb_ids['project']
                            entity=wandb_ids['entity']
                            url=f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                            index_ref,_ = self.evosys.CM.get_verifications_index()
                            index_ref.set({sess_id:{'W&B Training Run':url}},merge=True)
                print(f"There is already a verification job running. Please wait for it to finish.")
                return None,None
            if len(comps) == 2 or (len(comps) == 3 and 'resume' in comps):
                sess_id,pid = verify_command(self.node_id, self.evosys, comps[1], resume='resume' in comps, cli=self.cli, accept_baselines=self.accept_baselines, free_verifier=self.free_verifier)
            else:
                sess_id,pid = verify_command(self.node_id, self.evosys, comps[1], comps[2], comps[3], resume='resume' in comps, cli=self.cli, accept_baselines=self.accept_baselines, free_verifier=self.free_verifier)
        else:
            raise ValueError(f"Unknown command: {command}")
        return sess_id,pid
        
    def stop_listening(self):
        self.running = False
        self.doc_ref.update({'status': 'stopped'})
        self.cleanup()

    def cleanup(self):
        # st.info("Cleaning up and disconnecting...")
        self.doc_ref.delete()  # Delete the connection document
        local_doc = U.load_json(self.local_dir)
        local_doc['status'] = 'stopped'
        U.save_json(local_doc,self.local_dir)

def start_listener_thread(listener,add_ctx=True):
    thread = threading.Thread(target=listener.listen_for_commands)
    if add_ctx:
        add_script_run_ctx(thread)  # Add Streamlit context to the thread
    thread.start()
    return thread


def launch_listener(evosys, node_id, group_id, max_design_threads, accept_verify_job, cpu_only_checker):
    listener = Listener(evosys, node_id, group_id, max_design_threads, accept_verify_job, cpu_only_checker)
    listener.build_connection()
    st.session_state.listener = listener
    st.session_state.listener_thread = start_listener_thread(listener)
    st.session_state.listening_mode = True
    st.success(f"Listening started. Node ID: {listener.node_id}. Group ID: {listener.group_id}. Max design threads: {listener.max_design_threads}. Accept verify job: {listener.accept_verify_job}.")
    st.rerun()

def stop_listening():
    if st.session_state.listener:
        st.session_state.listener.stop_listening()
        st.session_state.listener_thread.join()
        st.session_state.listener = None
        st.session_state.listener_thread = None
        st.session_state.listening_mode = False
        ckpt_dir = os.environ.get("CKPT_DIR")
        local_doc = U.load_json(U.pjoin(ckpt_dir,'.node.json'))
        local_doc['status'] = 'stopped'
        U.save_json(local_doc,U.pjoin(ckpt_dir,'.node.json'))
        st.success("Listening stopped.")
        st.rerun() # Add this section to process commands in the main Streamlit thread

def listen(evosys, project_dir):
        
    st.title('Listening Mode')

    if not st.session_state.listening_mode:
        st.warning('**NOTE:** It is recommended to run a node by `run_node.sh` for production use. Run node here for demonstration only.')


    _node_id = AU._listener_running(evosys.ckpt_dir)
    _local_doc = U.load_json(U.pjoin(evosys.ckpt_dir,'.node.json'))
    if not _node_id:
        if st.session_state.listener and _local_doc and _local_doc['status'] == 'stopped' and _local_doc['node_id'] == st.session_state.listener.node_id:
            stop_listening()
    
    passive_mode=False  
    if st.session_state.listener and st.session_state.listener.running and not st.session_state.listener.active_mode:
        passive_mode=True
        st.info(f'You are viewing the status of a running listener. Node ID: ```{st.session_state.listener.node_id}```. Group ID: ```{st.session_state.listener.group_id}```. Max design threads: ```{st.session_state.listener.max_design_threads}```. Accept verify job: ```{st.session_state.listener.accept_verify_job}```.')


    ### Sidebar
    with st.sidebar:
        AU.running_status(st, evosys)

    
    if st.session_state.evo_running:
        st.warning("**NOTE:** You are running as the master node. You cannot change the role of the node while the system is running.")



    col1,col2,col3,col4,col5,col6 = st.columns([1.2,1.2,1,0.9,0.85,1])


    with col1:
        input_node_id = st.text_input("Node ID (empty for random)", disabled=st.session_state.listening_mode)

    with col2:
        input_group_id = st.text_input("Group ID (empty for default)", disabled=st.session_state.listening_mode)
        input_group_id = input_group_id if input_group_id else 'default'
        
    with col3:
        input_max_design_threads = st.number_input("Max design threads", min_value=1, value=5, disabled=st.session_state.listening_mode)

    with col5:
        st.write('')    
        st.write('')
        input_accept_verify_job = st.checkbox("Accept verify", value=True, disabled=st.session_state.listening_mode)


    with col6:
        st.write('')    
        st.write('')
        input_cpu_only_checker = st.checkbox("CPU only checker", value=False, disabled=st.session_state.listening_mode)

    with col4:
        st.write('') 
        st.write('')    
        if not st.session_state.listening_mode:
            if st.button("**Start Listening**", use_container_width=True, disabled=st.session_state.evo_running or passive_mode):
                launch_listener(evosys, input_node_id, input_group_id, input_max_design_threads, input_accept_verify_job, input_cpu_only_checker)
        else:
            if st.button("**Stop Listening**", use_container_width=True, disabled=st.session_state.evo_running or passive_mode):
                stop_listening()

        if st.session_state.listener:
            if st.session_state.listener.node_id not in st.session_state.exec_commands:
                st.session_state.exec_commands[st.session_state.listener.node_id] = {}
            if st.session_state.listener.active_mode:
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
                # command_status = doc.to_dict()['command_status'] if doc.exists else {}
                # for pid,status in command_status.items():
                #     pid=int(pid)
                #     st.session_state.exec_commands[st.session_state.listener.node_id][pid] = status['command'], status['sess_id']
            for pid in st.session_state.exec_commands[st.session_state.listener.node_id]:
                command,sess_id = st.session_state.exec_commands[st.session_state.listener.node_id][pid]
                ctype = command.split()[0]
                if ctype == 'design':
                    st.session_state['design_threads'][sess_id] = get_process(pid)
                elif ctype == 'verify':
                    st.session_state['running_verifications'][sess_id] = get_process(pid)

        

    st.subheader("Commands status")

    with st.sidebar:
        if len(st.session_state.exec_commands) > 0 and st.session_state.listener:
            all_nodes = list(st.session_state.exec_commands.keys())
            selected_node = st.selectbox("Select node ID", all_nodes,index=all_nodes.index(st.session_state.listener.node_id))
            # st.caption(':orange[*View more details in **Design**, **Verify**, and **Viewer** tabs.*]')
            st.button("*Refresh status*",use_container_width=True)
        else:
            st.info("No listener running.")
            

    if len(st.session_state.exec_commands) > 0 and st.session_state.listener:
        
        active_commands,inactive_commands = [],[]
        for pid in st.session_state.exec_commands[selected_node]:
            pid=int(pid)
            command,sess_id = st.session_state.exec_commands[selected_node][pid]
            if is_running(pid):
                active_commands.append((command,sess_id,pid))
            else:
                inactive_commands.append((command,sess_id,pid))
        
        def show_commands(commands,max_lines=20):
            for command,sess_id,pid in commands[-max_lines:]:
                cols=st.columns([0.5,0.01,0.2,0.01,1.8])
                ctype = command.split(',')[0]
                args = command.split(',')[1:]
                process = get_process(pid)
                with cols[0]:
                    st.write(f'PID: ```{pid}``` ({process.status() if process else "N/A"})')
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
        st.info("No listener running.")


if __name__ == "__main__":
    from model_discovery.evolution import BuildEvolution
    import argparse

    AU.print_cli_title()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--node_id', type=str, default='None', help='Node ID (empty for random)')
    parser.add_argument('-V','--v_node', action='store_true', help='Fast launching as a verification-only node. It is recommended to separate the V and D nodes.')
    parser.add_argument('-D','--d_node', action='store_true', help='Fast launching as a design-only node. It is recommended to separate the V and D nodes.')
    parser.add_argument('-n','--no_verify', action='store_true', help='Do not use GPUs (will not accept verify jobs)')
    parser.add_argument('-b','--accept_baselines', action='store_true', help='Accept baseline verifications')
    parser.add_argument('-m','--max_design_threads', type=int, default=5, help='Max number of design threads can accept')
    parser.add_argument('-g','--group_id', type=str, default='default', help='Group ID, if you want to run multiple experiments')
    parser.add_argument('-c','--cpu_only', action='store_true', help='Run design threads in CPU only mode')
    parser.add_argument('-s','--silent', action='store_true', help='Run in silent mode')
    parser.add_argument('-f','--free_verifier', action='store_true', help='Free verifier can ignore the requirement of verifying all models in at least one scale.')
    args = parser.parse_args()

    node_id = None
    if args.node_id == 'None':
        if 'GENESYS_NODE_ID' in os.environ:
            node_id = os.environ.get("GENESYS_NODE_ID")
    else:
        node_id = args.node_id

    if args.v_node:
        args.max_design_threads = 0
        print('🐒 Running in verification-only mode. The node will not accept design jobs.')

    if args.d_node:
        args.no_verify = True
        assert not args.v_node, 'Cannot be both design-only and verification-only node.'
        if args.max_design_threads <= 0:
            print('⚠️ Design-only node must have at least 1 design thread. Set to 1.')
            args.max_design_threads = 1
        print('🐒 Running in design-only mode. The node will not accept verify jobs.')

    # run in CLI mode

        
    _node_id = AU._listener_running(os.environ.get("CKPT_DIR"))
    if _node_id:
        local_doc = U.load_json(U.pjoin(os.environ.get("CKPT_DIR"),'.node.json'))
        _group_id = local_doc['group_id']
        _max_design_threads = local_doc['max_design_threads']
        _accept_verify_job = local_doc['accept_verify_job'] 
        print(
            f'⚠️ Local running listener detected:\n'
            f'Node ID: {_node_id}.\n'
            f'Network Group ID: {_group_id}.\n'
            f'Max design threads: {_max_design_threads}.\n'
            f'Accept verify job: {_accept_verify_job}.\n'
            f'Accept baseline verifications: {args.accept_baselines}.\n'
            f'Free verifier: {args.free_verifier}.\n'
            f'Please view it in the GUI Listen tab.\n'
            f'❕ If you just stopped a listener, please wait for {NODE_ZOMBIE_THRESHOLD} seconds for it to cool down and cleanup.'
        )
    else:
        setting=AU.get_setting()
        default_namespace=setting.get('default_namespace','test_evo_000')
        evosys = BuildEvolution(
            params={'evoname':default_namespace}, # doesnt matter, will switch to the commanded evoname
            do_cache=False,
            # cache_type='diskcache',
        )
        listener = Listener(evosys, node_id, args.group_id, max_design_threads=args.max_design_threads, accept_baselines=args.accept_baselines,
                            accept_verify_job=not args.no_verify, cpu_only=args.cpu_only, silent=args.silent, cli=True, free_verifier=args.free_verifier)
        listener.build_connection()
        listener_thread = start_listener_thread(listener,add_ctx=False)

        print(f"🐵 Listener launched: \n"
              f"Node ID: {listener.node_id} \n"
              f"Group ID: {listener.group_id} \n"
              f"Max design threads: {listener.max_design_threads} \n"
              f"Accept verify job: {listener.accept_verify_job} \n"
              f"Accept baseline verifications: {listener.accept_baselines} \n"
              f"Free verifier: {listener.free_verifier}")
        if args.silent:
            print('🙊 Running in silent mode. The design threads will not print in screen, please check logs.')
        if args.no_verify:
            print('🙉 Running in no-verify mode. The node will not accept verify jobs.')
        if args.cpu_only:
            print('🙈 Running in CPU-only design mode. The design threads will not use GPUs for checking.')
        try:
            # Keep the main thread alive
            while listener.active_mode:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            listener.stop_listening()
            listener_thread.join(timeout=10)  # Wait for the thread to finish, with a timeout

        print("Listener stopped.")



