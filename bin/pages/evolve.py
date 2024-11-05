import json
import time
import pathlib
import streamlit as st
import sys,os
from enum import Enum
from subprocess import check_output
import pytz
import streamlit.components.v1 as components
from google.cloud import firestore
from datetime import datetime, timedelta
import threading
import pandas as pd
import psutil
import subprocess
from streamlit.runtime.scriptrunner import add_script_run_ctx


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU
from bin.pages.listen import DESIGN_ACTIVE_STATES,VERIFY_ACTIVE_STATES
from model_discovery.evolution import BENCH_MODE_OPTIONS
from bin.pages.viewer import export_leaderboards,leaderboard_filter,leaderboard_relative


CC_POLL_FREQ = 30 # seconds
CC_ZOMBIE_THRESHOLD = 60 # seconds
CC_COMMAND_DELAY = 2 # seconds


def _is_running(evosys):
    collection = evosys.remote_db.collection('experiment_connections')
    docs = collection.get()
    for doc in docs:
        if doc.to_dict().get('status','n/a') == 'connected':
            last_heartbeat = doc.to_dict().get('last_heartbeat')
            threshold_time = datetime.now(pytz.UTC) - timedelta(seconds=CC_ZOMBIE_THRESHOLD)
            is_zombie = last_heartbeat < threshold_time
            if is_zombie:
                doc.reference.update({'status':'disconnected'})
                continue
            evoname = doc.to_dict().get('evoname')
            group_id = doc.to_dict().get('group_id')
            if group_id==evosys.CM.group_id or evoname==evosys.evoname:
                return evoname, group_id
    return None, None

class CommandCenter:
    def __init__(self,evosys,max_designs_per_node,max_designs_total,stream,allow_resume=True):
        self.evosys=evosys
        self.evoname=evosys.evoname
        self.max_designs_per_node=max_designs_per_node
        self.max_designs_total=max_designs_total
        self.st=stream
        self.doc_ref = evosys.remote_db.collection('experiment_connections').document(self.evosys.evoname)
        self.running = False
        self.poll_freq=CC_POLL_FREQ
        self.zombie_threshold = CC_ZOMBIE_THRESHOLD  # seconds
        self.allow_resume = allow_resume

    def read_logs(self):
        return self.evosys.CM.get_log_ref().get().to_dict()

    def build_connection(self,active_mode=True):
        # check if the node_id is already in the collection
        evoname, _ = _is_running(self.evosys)
        self.active_mode = active_mode
        if evoname:
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Namespace {evoname} is already running. Connecting to the existing command center.')
            self.active_mode = False
            self.evoname = evoname
        else:
            if active_mode:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Namespace {self.evosys.evoname} is not running. Launching a new command center.')
                self.doc_ref.set({
                    'status': 'connected',
                    'last_heartbeat': firestore.SERVER_TIMESTAMP,
                    'evoname': self.evosys.evoname,
                    'group_id': self.evosys.CM.group_id,
                    'benchmark_mode': self.evosys.benchmark_mode,
                },merge=True)
                self.active_mode = True

    # def cleanup(self):
    #     # st.info("Cleaning up and disconnecting...")
    #     self.doc_ref.delete()  # Delete the connection document

    def assign_design_workloads(self,design_workloads,to_sleep):
        if self.evosys.remaining_verify_budget<=0:
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] No verify budget available, stopping design workload assignment')
            return to_sleep,design_workloads
        if self.evosys.max_samples>0 and self.evosys.finished_designs>=self.evosys.max_samples:
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Max design samples reached, stopping design workload assignment')
            return to_sleep,design_workloads
        
        if to_sleep<=0:
            return to_sleep,design_workloads
        # check if the design workloads are full
        if self.evosys.benchmark_mode: 
            if not self.evosys.ptree.acquire_design_lock():
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design is full, stopping design workload assignment')
                return to_sleep,design_workloads
            
        total_design_workloads = sum(design_workloads.values())
        if total_design_workloads == self.max_designs_total or self.max_designs_total==0:
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design workloads are full ({total_design_workloads}/{self.max_designs_total}), skipping design workload assignment')
        else:
            _design_availability = {k:self.evosys.CM.connections[k]['max_design_threads'] - v for k,v in design_workloads.items() if v<self.max_designs_per_node or self.max_designs_per_node==0}
            design_availability = {k:v for k,v in _design_availability.items() if v>0}
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design workloads: {design_workloads}, Design availability: {design_availability}')
            if sum(design_availability.values())>0:
                available_design_threads = self.max_designs_total-sum(design_workloads.values())
                for _ in range(max(0,available_design_threads)):
                    node_id = max(design_availability, key=design_availability.get)
                    if design_availability[node_id]>0:
                        if design_workloads[node_id]<self.max_designs_per_node or self.max_designs_per_node==0 :
                            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Assigning design workload to the most available node {node_id}: {design_availability[node_id]}')
                            self.evosys.CM.design_command(node_id)
                            design_availability[node_id] -= 1
                            design_workloads[node_id] += 1
                            time.sleep(CC_COMMAND_DELAY)
                            to_sleep-=CC_COMMAND_DELAY
                            if to_sleep<=0:
                                break
                    #     else:
                    #         print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design workload for the most available node {node_id} is full ({design_workloads[node_id]}/{self.max_designs_per_node})')
                    # else:
                    #     print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design availability for the most available node {node_id} is empty ({design_availability[node_id]})')
            else:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] No design workload available: {_design_availability}')
        return to_sleep,design_workloads

    def assign_verify_workloads(self,verify_workloads,to_sleep):
        # assigned_verify_workloads = False
        if to_sleep<=0:
            return to_sleep,verify_workloads
        if self.evosys.benchmark_mode:
            return to_sleep,verify_workloads    
        for node_id in verify_workloads:
            if verify_workloads[node_id] == 0 and self.evosys.CM.accept_verify_job[node_id]:
                if self.evosys.CM.verify_command(node_id,resume=self.allow_resume):
                    # assigned_verify_workloads = True
                    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Assigning verify workload to node {node_id}')
                    time.sleep(CC_COMMAND_DELAY)
                    to_sleep -= CC_COMMAND_DELAY
                    verify_workloads[node_id] += 1
                    if to_sleep<=0:
                        break
        return to_sleep,verify_workloads
    
    def send_user_commands(self,design_workloads,verify_workloads,to_sleep):
        if to_sleep<=0:
            return to_sleep,design_workloads,verify_workloads
        user_commands = self.doc_ref.get().to_dict().get('user_command_stack',[])
        if len(user_commands)==0:
            return to_sleep,design_workloads,verify_workloads
        processed_command_idx = []
        for idx,command in enumerate(user_commands):
            cmds = command.strip().split()
            cmd = cmds[0]
            args = cmds[1:]
            if cmd == 'verify':
                if len(args)==0:
                    to_sleep,verify_workloads = self.assign_verify_workloads(verify_workloads,to_sleep)
                elif len(args)==2:
                    design_id, scale = args
                    for node_id in verify_workloads:
                        if verify_workloads[node_id] == 0 and self.evosys.CM.accept_verify_job[node_id]:
                            if self.evosys.CM.verify_command(node_id,design_id,scale.upper(),resume=self.allow_resume):
                                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Assigning verify workload to node {node_id}')
                                time.sleep(CC_COMMAND_DELAY)
                                to_sleep -= CC_COMMAND_DELAY
                                processed_command_idx.append(idx)
                                verify_workloads[node_id] += 1
                                break
            elif cmd == 'design':
                to_sleep,design_workloads = self.assign_design_workloads(design_workloads,to_sleep)
        user_commands = [user_commands[i] for i in range(len(user_commands)) if i not in processed_command_idx]
        self.doc_ref.set({'user_command_stack': user_commands},merge=True)
        return to_sleep,verify_workloads,design_workloads

    def run(self):
        self.evosys.sync_to_db() # sync the config to the db
        self.evosys.CM.start_log()
        self.running = True
        mode = 'active mode' if self.active_mode else 'passive mode'
        _type = 'Benchmark' if self.evosys.benchmark_mode else 'Evolution'
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {_type} launched for {self.evosys.evoname} in {mode}')
        while self.running:
            to_sleep = self.poll_freq
            if self.active_mode:
                # check if the status is still stopped
                if self.doc_ref.get().to_dict().get('status','n/a') == 'stopped':
                    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Command center for {self.evosys.evoname} in {mode} is stopped.')
                    break

                if self.evosys.should_stop():
                    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Done, stopping the command center.')
                    self.evosys.conclude()
                    break

                # self.evosys.CM.get_active_connections() # refresh the connection status
                design_workloads, verify_workloads = self.evosys.CM.get_all_workloads() # will refresh the connection status
                design_workloads = {k:len(v) for k,v in design_workloads.items() if k in self.evosys.CM.connections}
                verify_workloads = {k:len(v) for k,v in verify_workloads.items() if k in self.evosys.CM.connections}
                
                if not self.evosys.benchmark_mode:
                    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design workloads: {design_workloads}, Verify workloads: {verify_workloads}')
                else:
                    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design workloads: {design_workloads}')

                to_sleep,design_workloads,verify_workloads = self.send_user_commands(design_workloads,verify_workloads,to_sleep)
                to_sleep,_ = self.assign_design_workloads(design_workloads,to_sleep)
                to_sleep,_ = self.assign_verify_workloads(verify_workloads,to_sleep)

                self.doc_ref.update({'last_heartbeat': firestore.SERVER_TIMESTAMP})
            
            if to_sleep>0:
                time.sleep(to_sleep)  
        # self.cleanup()

        self.stop()


    def stop(self):
        self.running = False
        if self.doc_ref.get().exists:
            self.doc_ref.update({'status': 'stopped'})
        # self.cleanup()

    def process_user_command(self,command):
        _command = command.lower().strip().split()
        if len(_command)==0:
            return
        cmd = _command[0]
        if cmd == 'verify':
            if len(_command) not in [1,3]:
                st.toast(f'Invalid command format: {command}. Please use `verify` followed by a design name or design id.',icon='🚨')
                return
            elif len(_command)==3:
                design_id, scale = _command[1:]
                models = self.evosys.ptree.filter_by_type(['DesignArtifactImplemented','ReferenceCoreWithTree'])
                if design_id not in models:
                    st.toast(f'Design {design_id} not found.',icon='🚨')
                    return
                if scale.upper() not in self.evosys.target_scales:
                    st.toast(f'Scale {scale} not in target scales: {self.evosys.target_scales}.',icon='🚨')
                    return
        elif cmd == 'design':
            if len(_command) != 1:
                st.toast(f'Invalid command format: {command}. Please use `design` without any arguments.',icon='🚨')
                return
        else:
            st.toast(f'Unknown command: {command}',icon='🚨')
            return
        user_commands = self.doc_ref.get().to_dict().get('user_command_stack',[])
        user_commands.append(command)
        self.doc_ref.set({'user_command_stack': user_commands},merge=True)
        st.toast(f'Command {command} sent. Please refresh the page to see the latest command stack.',icon='✅')

    def clear_user_command_stack(self):
        self.doc_ref.set({'user_command_stack': []},merge=True)

    def show_user_command_stack(self):
        user_commands = self.doc_ref.get().to_dict().get('user_command_stack',[])
        if len(user_commands)==0:
            st.info('No command stacks available at the moment.')
        else:
            st.write(user_commands)

def x_evolve_passive(command_center,cli=False): # extereme evolution 
    thread = threading.Thread(target=command_center.run)
    if not cli:
        add_script_run_ctx(thread)  # Add Streamlit context to the thread
    thread.start()
    doc = command_center.doc_ref.get()
    pid = doc.to_dict().get('pid') if doc.exists else None
    return thread, pid

def x_evolve(command_center):
    cmd = (
        f'python -m bin.pages.evolve --evoname {command_center.evosys.evoname} '
        f'--max_designs_per_node {command_center.max_designs_per_node} '
        f'--max_designs_total {command_center.max_designs_total} '
        f'--group_id {command_center.evosys.CM.group_id}'
    )
    if not command_center.allow_resume:
        cmd += ' --no_resume'

    # Use subprocess.DETACH_PROCESS on Windows, or os.setsid on Unix-like systems
    if os.name == 'nt':  # For Windows
        process = subprocess.Popen(
            cmd,
            shell=True,
            creationflags=subprocess.DETACH_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:  # For Unix-like systems (Linux, macOS)
        process = subprocess.Popen(
            cmd,
            shell=True,
            # preexec_fn=os.setsid,
            start_new_session=True
        )
    command_center.doc_ref.set({'pid': process.pid},merge=True)
    return process.pid


def launch_evo(evosys,max_designs_per_node,max_designs_total,active_mode=True,allow_resume=True):
    if len(evosys.CM.get_active_connections())==0 and active_mode:
        st.toast('No nodes connected. Please remember to launch nodes.',icon='🚨')
    command_center = CommandCenter(evosys,max_designs_per_node,max_designs_total,st,allow_resume=allow_resume) # launch a passive command center first
    if active_mode:
        st.session_state.evo_process_pid = x_evolve(command_center)
        time.sleep(5)
    command_center.build_connection(active_mode=False) # launch a passive 
    thread, pid = x_evolve_passive(command_center)
    st.session_state.evo_process_pid = pid
    st.session_state.evo_passive_thread = thread
    st.session_state.command_center = command_center
    st.session_state.evo_running = True
    evoname = st.session_state.command_center.evoname
    _mode = 'active mode' if active_mode else 'passive mode'
    _type = 'Benchmark' if evosys.benchmark_mode else 'Evolution'
    _icon = '🪑' if evosys.benchmark_mode else '🚀'
    st.toast(f"{_type} launched for ```{evoname}``` in {_mode}.",icon=_icon)
    time.sleep(1)
    st.rerun()

def stop_evo(evosys):
    if st.session_state.command_center:
        print('Stopping evolution...')
        evoname = st.session_state.command_center.evoname
        if st.session_state.evo_process_pid:
            print(f'Stopping process {st.session_state.evo_process_pid}')
            try:
                process = psutil.Process(st.session_state.evo_process_pid)
                process.terminate()
                process.wait(timeout=10)
                print(f'Process {st.session_state.evo_process_pid} terminated')
            except psutil.NoSuchProcess:
                print(f'Process {st.session_state.evo_process_pid} not found')
        if st.session_state.evo_passive_thread:
            print(f'Stopping thread {st.session_state.evo_passive_thread}')
            st.session_state.command_center.stop()
            st.session_state.evo_passive_thread.join(timeout=5)
            if st.session_state.evo_passive_thread.is_alive():
                print("Thread did not terminate, forcing exit")
            st.session_state.evo_passive_thread = None
        print('Stopping command center...')
        _type = 'Benchmark' if evosys.benchmark_mode else 'Evolution'
        st.session_state.command_center.stop()
        st.session_state.command_center = None
        st.session_state.evo_process_pid = None
        st.session_state.evo_running = False
        time.sleep(1)
        st.toast(f"{_type} stopped for ```{evoname}```.",icon='🛑')
        time.sleep(1)
        st.rerun()


def network_status(evosys):
    group_id = evosys.CM.group_id
    benchmark_mode = evosys.benchmark_mode
    st.write(f'#### *Network Group ```{group_id}``` Status*')

    with st.expander('Nodes Running Status',expanded=True):
        nodes = evosys.CM.get_active_connections()
        if not nodes or len(nodes)==0:
            st.info('No active working nodes connected')
        else:
            st.write(f'##### Connected Nodes ```{len(nodes)}```')
            _nodes = {}
            design_workloads, verify_workloads = evosys.CM.get_all_workloads()
            for node_id in nodes:
                node_data = evosys.CM.collection.document(node_id).get().to_dict()
                accept_verify_job = node_data['accept_verify_job']
                design_load = design_workloads.get(node_id,[])
                verify_load = verify_workloads.get(node_id,[])
                node_max_design_threads = node_data['max_design_threads']
                _nodes[node_id] = {
                    'Design Workload': f'{len(design_load)}/{node_max_design_threads}' if node_max_design_threads>0 else 'N/A',
                    'Verify Workload': f'{len(verify_load)}/1' if accept_verify_job else 'N/A',
                    'Accept Verify Job': accept_verify_job,
                    'Use GPU Checker': not node_data['cpu_only_checker'],
                    'Last Heartbeat': node_data['last_heartbeat'].strftime('%Y-%m-%d %H:%M:%S %Z'),
                    'Status': node_data['status'],
                    'MAC Address': node_data['mac_address'],
                    'GPU': node_data.get('gpu',None)
                }
            nodes_df = pd.DataFrame(_nodes).T
            st.dataframe(nodes_df,use_container_width=True)
            
        CC = st.session_state.command_center
        active_design_sessions = evosys.CM.get_active_design_sessions()
        # if CC is None:
        st.write(f'##### Active Design Sessions ```{len(active_design_sessions)}```')
        # else:
        #     max_design_threads = '♾️'
        #     if CC.max_designs_total>0:
        #         if CC.max_designs_per_node>0:

        #             max_design_threads = CC.max_designs_total
        #     st.write(f'##### Active Design Sessions ```{len(active_design_sessions)}/{max_design_threads}```')
        if len(active_design_sessions)>0:
            active_design_sessions_df = pd.DataFrame(active_design_sessions).T
            active_design_sessions_df.rename(columns={'latest_log':'started_at'},inplace=True)
            if 'pid' in active_design_sessions_df.columns:
                active_design_sessions_df['pid'] = active_design_sessions_df['pid'].astype(str)
            if 'started_at' in active_design_sessions_df.columns:
                # active_design_sessions_df['started_at'] = pd.to_datetime(active_design_sessions_df['started_at'],unit='s').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                del active_design_sessions_df['started_at']
            if 'heartbeat' in active_design_sessions_df.columns:
                active_design_sessions_df['heartbeat'] = pd.to_datetime(active_design_sessions_df['heartbeat'],unit='s').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            if 'timestamp' in active_design_sessions_df.columns:
                del active_design_sessions_df['timestamp']
            # if 'progress' in active_design_sessions_df.columns:
            #     active_design_sessions_df['progress'] = active_design_sessions_df['progress'].apply(lambda x: x.split(', {')[0])
            active_design_sessions_df.rename(columns={'mode':'Mode'},inplace=True)
            st.dataframe(active_design_sessions_df,use_container_width=True)
        else:
            st.info('No active design sessions')

        if not benchmark_mode:
            running_verifications = evosys.CM.get_running_verifications()
            running_verifications = {v:running_verifications[v] for v in running_verifications if running_verifications[v]['heartbeat'] is not None}
            st.write(f'##### Running Verifications ```{len(running_verifications)}```')
            if len(running_verifications)>0:
                # for sess_id in running_verifications:
                running_verifications_df = pd.DataFrame(running_verifications).T
                if 'latest_log' in running_verifications_df.columns:
                    del running_verifications_df['latest_log']
                if 'timestamp' in running_verifications_df.columns:
                    del running_verifications_df['timestamp']
                if 'W&B Training Run' in running_verifications_df.columns: # ad hoc for some errors in log
                    del running_verifications_df['W&B Training Run']
                if 'wandb_url' in running_verifications_df.columns:
                    running_verifications_df.rename(columns={'wandb_url':'W&B Training Run'},inplace=True)
                if 'W&B Training Run' not in running_verifications_df.columns:
                    wandb_urls = []
                    for sess_id in running_verifications_df.index:
                        ve_dir = U.pjoin(evosys.evo_dir, 've', sess_id)
                        if os.path.exists(ve_dir):
                            wandb_ids = U.load_json(U.pjoin(ve_dir, 'wandb_ids.json'))
                            wandb_id=wandb_ids['pretrain']['id']
                            project=wandb_ids['project']
                            entity=wandb_ids['entity']
                            url=f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                            wandb_urls.append(url)
                        else:
                            wandb_urls.append(None)
                    running_verifications_df['W&B Training Run'] = wandb_urls
                if 'pid' in running_verifications_df.columns:
                    running_verifications_df['pid'] = running_verifications_df['pid'].astype(str)
                if 'heartbeat' in running_verifications_df.columns:
                    running_verifications_df['heartbeat'] = pd.to_datetime(running_verifications_df['heartbeat'],unit='s').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                running_verifications_df.rename(columns={'status':'Status','pid':'PID','heartbeat':'Last Heartbeat','node_id':'Node ID'},inplace=True)
                st.dataframe(running_verifications_df,use_container_width=True,column_config={'W&B Training Run': st.column_config.LinkColumn('W&B Training Run')})
            else:
                st.info('No running verifications')

# def system_config(evosys):
#     with st.expander(f"System Config for ```{evosys.evoname}```",expanded=False,icon='🔍'):
#         col1,col2=st.columns(2)
#         with col1:
#             st.write('Design Config',evosys.design_cfg)
#             st.write('Search Config',evosys.search_cfg)
#         with col2:
#             st.write('Select Config',evosys.select_cfg)
#             st.write('Verify Engine Config',evosys.ve_cfg)



def evolution_launch_pad(evosys):
    passive_mode=False
    if st.session_state.command_center and st.session_state.command_center.running and not st.session_state.command_center.active_mode:
        passive_mode=True
        # _evoname = st.session_state.command_center.evoname
        # st.info(f'The command center is already running for namespace ```{_evoname}```. You are in passive observation mode.')

    
    st.header("Launch Pad")
    col1, col2, col3, col4,col5,col6 = st.columns([1,1,1,1,1,0.9],gap='small')
    with col1:
        input_max_designs_per_node=st.number_input("Max Designs Per Node",min_value=0,value=0,disabled=st.session_state.evo_running,
            help='Global control of the maximum number of design threads to run on each node in addition to the local settings on each node. 0 is unlimited.'
        )
    with col2:
        input_max_designs_total=st.number_input("Max Designs in Parallel",min_value=0,value=10,disabled=st.session_state.evo_running,
            help='The maximum number of design threads run across all nodes at the same time. 0 is unlimited (which means only bound by the per-node settings).'
        )
    with col3:
        node_schedule=st.selectbox("Network Scheduling",['load balancing'],disabled=True, #st.session_state.evo_running,
            help='Overall network task scheduling strategy. Currently, only load balancing is supported, which always assigns the jobs to the highest available nodes.'
        )
    with col4:
        # always use extreme mode, use as much gpus as possible
        verify_schedule=st.selectbox("Verification Scheduling",['full utilization'],disabled=True, #st.session_state.evo_running,
            help='The strategy to schedule verification jobs across nodes. Currently, only full utilization is supported, which means scheduling verify jobs immediately when a node is free.'
        )
    with col6:
        st.write('')
        st.write('')
        input_allow_resume=st.checkbox("Allow Resume",value=True,disabled=True,
            help='Whether allow resume training.'
        )

    with col5:
        st.write('')
        st.write('')
        # distributed='Distributed ' if evosys.remote_db else ''
        if not st.session_state.evo_running:
            title1 = f":rainbow[***Launch Evolution***] :rainbow[🚀]"
            title2 = f"***Launch Evolution*** 🚀"
            run_evo_btn = st.button(
                AU.theme_aware_options(st,title1,title2,title1),
                disabled=not evosys.remote_db or passive_mode or evosys.benchmark_mode,
                use_container_width=True
            ) 
        else:
            stop_evo_btn = st.button(
                "***Stop Evolution*** 🛑",
                disabled=not evosys.remote_db,
                use_container_width=True
            )
    
    if not st.session_state.evo_running:
        if run_evo_btn:       
            with st.spinner('Launching... The evolution will be launched in the background. You may leave the gui and can always go back here to check.'):  
                launch_evo(evosys,input_max_designs_per_node,input_max_designs_total,True,allow_resume=input_allow_resume)
    else:
        if stop_evo_btn:
            if st.session_state.command_center:
                with st.spinner('Stopping... Note, the nodes will keep working on the unfinished jobs'):
                    stop_evo(evosys)

    st.subheader("Command Center Panel")
    cols = st.columns([3,1,1,2.5])
    with cols[0]:
        user_command = st.text_input('Input a command',value='',disabled=not st.session_state.evo_running,help='''
You can send command to the command center. Format example:
```
verify # verify a design by select 
verify gpt 14M # verify a given design
design # design a new model
```
''')
    with cols[1]:
        st.write('')
        st.write('')
        send_command_btn = st.button('Send Command',use_container_width=True,
                disabled=not st.session_state.evo_running)
    with cols[2]:
        st.write('')
        st.write('')
        clear_command_btn = st.button('Clear Command',use_container_width=True,
                disabled=not st.session_state.evo_running)
    with cols[3]:
        st.write('')
        st.write('')
        with st.expander('User command stacks',expanded=False):
            if st.session_state.command_center:
                st.session_state.command_center.show_user_command_stack()
            else:
                st.info('No command center running.')

    if send_command_btn:
        st.session_state.command_center.process_user_command(user_command)
    if clear_command_btn:
        st.session_state.command_center.clear_user_command_stack()



def benchmark_launch_pad(evosys):
    
    passive_mode=False
    if st.session_state.command_center and st.session_state.command_center.running and not st.session_state.command_center.active_mode:
        passive_mode=True
        # _evoname = st.session_state.command_center.evoname
        # st.info(f'The command center is already running for namespace ```{_evoname}```. You are in passive observation mode.')

    
    st.header("Launch Pad")

    _color = AU.theme_aware_options(st,'orange','violet','violet')
    st.markdown(f'###### Convinient Benchmark Settings :{_color}[*(Please remember to configure other settings in the ```Config``` tab)*]')
    cols = st.columns([1,1,1.3,2.5,1])
    with cols[0]:
        n_trials_cur = evosys.benchmark_settings.get('n_trials',100)
        _n_trials = st.number_input('Number of Trials',min_value=1,value=n_trials_cur,disabled=st.session_state.evo_running,
            help='The number of design sessions to run.')
    with cols[1]:
        max_retries_cur = evosys.benchmark_settings.get('max_retries',3)
        if max_retries_cur is None:
            max_retries_cur = evosys.ptree.challenging_threshold
        _max_retries = st.number_input('Max Impl. Retries',min_value=0,value=max_retries_cur,disabled=st.session_state.evo_running, 
            help='The maximum number of *implementation retries* of one session. If set to 0, it will be no retry.')
    with cols[2]:
        _MODE_OPTIONS = BENCH_MODE_OPTIONS.copy()
        _MODE_OPTIONS[3] = 'Mixed (Edit right or Config tab)'
        design_mode_cur = evosys.benchmark_settings.get('design_mode','Mutation-only')
        design_mode_cur = _MODE_OPTIONS.index(design_mode_cur)
        _design_mode = st.selectbox('Design Mode',options=_MODE_OPTIONS,index=design_mode_cur,disabled=st.session_state.evo_running,
            help='If you choose Mixed mode, the number of seeds will follow the distribution settings, otherwise, it will always sample 1 (Mutation-only), 2 (Crossover-only), 0 (Scratch-only) seeds respectively.'
        )
        if _design_mode == 'Mixed (Edit right or Config tab)':
            _design_mode = 'Mixed'
    with cols[3]:
        default_n_seeds_dist = {'0': 0.1, '1': 0.8, '2': 0.1, '3': 0, '4': 0, '5': 0}
        n_seeds_dist_cur = evosys.benchmark_settings.get('n_seeds_dist',default_n_seeds_dist)
        _n_seeds_dist_df = pd.DataFrame(n_seeds_dist_cur,index=['Weights'])
        _n_seeds_dist_df = st.data_editor(_n_seeds_dist_df,use_container_width=True,disabled=st.session_state.evo_running or 'Mixed' not in _design_mode)
        _n_seeds_dist = _n_seeds_dist_df.to_dict(orient='records')[0]
        _n_seeds_dist = {k:v for k,v in _n_seeds_dist.items()}
    with cols[4]:
        st.write('')
        st.write('')
        overwrite_config_cur = evosys.benchmark_settings.get('overwrite_config',True)
        _overwrite_config = st.checkbox('Overwrite',value=overwrite_config_cur,disabled=st.session_state.evo_running or 'Mixed' not in _design_mode,
            help='If checked, will apply the n seeds distribution on the left instead of the one in config in Mixed mode. Notice that if use this one, there will be no warmup.')


    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1.2],gap='small')
    with col1:
        input_max_designs_per_node=st.number_input("Max Designs Per Node",min_value=0,value=0,disabled=st.session_state.evo_running,
            help='Global control of the maximum number of design threads to run on each node in addition to the local settings on each node. 0 is unlimited.'
        )
    with col2:
        input_max_designs_total=st.number_input("**Max Designs in Parallel**",min_value=0,value=10,disabled=st.session_state.evo_running,
            help='The maximum number of design threads run across all nodes at the same time. 0 is unlimited (which means only bound by the per-node settings). Please keep it consistent for fairer comparison in benchmarks.'
        )
    with col3:
        node_schedule=st.selectbox("Network Scheduling",['load balancing'],disabled=True, #st.session_state.evo_running,
            help='Overall network task scheduling strategy. Currently, only load balancing is supported, which always assigns the jobs to the highest available nodes.'
        )

    with col5:
        st.write('')
        st.write('')
        allow_tree_cur = evosys.benchmark_settings.get('allow_tree',True)
        _allow_tree=st.checkbox("Allow Sampling from Tree",value=allow_tree_cur,disabled=st.session_state.evo_running,
            help='Whether allow sampling from the phylogenetic tree or only sampling from the seed references.'
        )

        
    benchmark_settings = {
        'n_trials': _n_trials,
        'max_retries': _max_retries,
        'design_mode': _design_mode,
        'n_seeds_dist': _n_seeds_dist,
        'overwrite_config': _overwrite_config,
        'allow_tree': _allow_tree,
    }


    with col4:
        st.write('')
        st.write('')
        # distributed='Distributed ' if evosys.remote_db else ''
        if not st.session_state.evo_running:
            run_bench_btn = st.button(
                "***Launch Benchmark*** 🪑",
                disabled=not evosys.remote_db or passive_mode or not evosys.benchmark_mode,
                use_container_width=True
            ) 
        else: 
            stop_bench_btn = st.button(
                "***Stop Benchmark*** 🛑",
                disabled=not evosys.remote_db,
                use_container_width=True
            )

    if not st.session_state.evo_running:
        if run_bench_btn:       
            with st.spinner('Launching... The benchmark will be launched in the background. You may leave the gui and can always go back here to check.'):  
                evosys.set_benchmark_mode(benchmark_settings)
                launch_evo(evosys,input_max_designs_per_node,input_max_designs_total)
    else:
        if stop_bench_btn:
            if st.session_state.command_center:
                with st.spinner('Stopping... Note, the nodes will keep working on the unfinished jobs'):
                    stop_evo(evosys)

    
def session_statistics(evosys):
    st.subheader("Session Statistics Monitor")
    with st.expander(f"Design Session Statistics for ```{evosys.evoname}```",expanded=True):#,icon='📊'):
        st.info('No design session statistics available at the moment.')






def _evolve(evosys,mode):
    
    running_evoname,running_group_id = _is_running(evosys)
    if running_evoname:
        if not st.session_state.evo_running:
            if evosys.CM.group_id == running_group_id or evosys.evoname == running_evoname:
                st.toast(f'Network group ```{running_group_id}``` is already running for evolution ```{running_evoname}```. Launching a command center in passive mode.')
            else:
                st.toast(f'Evolution ```{running_evoname}``` is already running in network group ```{running_group_id}```. Launching a command center in passive mode.')
            launch_evo(evosys,0,0,active_mode=False)
    # else:
    #     if st.session_state.evo_running:
    #         st.toast(f'The command center is not active anymore. You may stop listing to command center.')
    #         # stop_evo(evosys)



    if mode == EvoModes.BENCH:
        if evosys.benchmark_mode:
            benchmark_launch_pad(evosys)
        else:
            st.warning(f'The namespace ```{evosys.evoname}``` is not set to benchmark mode. Please set it to benchmark mode to launch the benchmark.')
    elif mode == EvoModes.EVOLVE:
        if evosys.benchmark_mode:
            st.warning(f'The namespace ```{evosys.evoname}``` is set to benchmark mode. Please do not run evolution in this namespace.')
        else:
            evolution_launch_pad(evosys)


    network_status(evosys)

    view_latest_K=30
    if st.session_state.evo_running:
        with st.expander(f"📝 **Running Logs** *(Latest {view_latest_K} logs)*",expanded=True):
            evo_log=st.session_state.command_center.read_logs()
            if evo_log:
                log_df = pd.DataFrame(evo_log.items(),columns=['timestamp','log']).set_index('timestamp')
                # convert time.time() to datetime
                log_df.index = pd.to_datetime(log_df.index,unit='s')
                log_df = log_df.sort_index(ascending=False)
                st.dataframe(log_df.head(view_latest_K),use_container_width=True)
            else:
                st.info("No logs available at the moment.")


    session_statistics(evosys)
    

    st.subheader("Phylogenetic Tree Monitor")

    col1, col2, col3 = st.columns([6,0.1,2])
    

    if 'ptree_max_nodes' not in st.session_state:
        st.session_state.ptree_max_nodes=100

    n_implemented = len(evosys.ptree.filter_by_type('DesignArtifactImplemented'))
    n_designs = n_implemented + len(evosys.ptree.filter_by_type('DesignArtifact'))

    _bg_color=AU.theme_aware_options(st,"#fafafa","#f0f0f0","#fafafa")
    
    export_height = 900
    evosys.ptree.export(max_nodes=st.session_state.ptree_max_nodes,height=f'{export_height}px',bgcolor=_bg_color)
    ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_{st.session_state.ptree_max_nodes}.html')

    with col1:
        _max_nodes=st.slider('Max Nodes to Display',min_value=0,max_value=len(evosys.ptree.G.nodes),value=st.session_state.ptree_max_nodes)

    # check this: https://github.com/napoles-uach/streamlit_network 
    with col3:
        st.write('')
        st.write('')
        if st.button(f'Refresh & Sync Tree'):#,use_container_width=True):
            evosys.ptree.update_design_tree()
            evosys.ptree.export(max_nodes=_max_nodes,height=f'{export_height}px',bgcolor=_bg_color,
                legend_font_size=12,legend_width_constraint=100,legend_x=-2400,legend_y=-200,legend_step=100)
            ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_{_max_nodes}.html')
            st.session_state.ptree_max_nodes=_max_nodes
    
    st.write(f'**First {st.session_state.ptree_max_nodes} nodes under the namespace ```{evosys.evoname}```**, :red[{n_designs}] designs, :blue[{n_implemented}] implemented. *(Node Size by # of citations or children)*.')
            # 'Legend: :red[Seed Designs (*Displayed Pink*)] | :blue[Design Artifacts] | :orange[Reference w/ Code] | :violet[Reference w/o Code] *(Size by # of citations)*')

    HtmlFile = open(ptree_dir_small, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = export_height)



def _is_eureka(node,baseline,threshold=0.05):
    # given population, node, decide if it is an eureka node
    # 1. fixed baseline, whether a node has any highlight (any scale any metric) over the baseline
    # 2. dynamic baseline, the baseline is not fixed, but relative to the last population

    pass


BASIC_BASELINES = [
    'gpt2 (baseline)',
    'rwkv6 (baseline)',
    'mamba2 (baseline)',
    'ttt (baseline)',
    'retnet (baseline)',
    'random',
]

def _eureka(evosys):
    with st.status('Loading latest data...'):
        design_vectors = evosys.ptree.get_design_vectors()
        baseline_vectors = evosys.ptree.get_baseline_vectors()
        leaderboards_normed,_,_,baselines=export_leaderboards(evosys,design_vectors,baseline_vectors)
        leaderboards_normed.pop('all')

    default_baseline = 'random'
    cols = st.columns(4)
    with cols[0]:
        baseline = st.selectbox('Baseline',options=BASIC_BASELINES,index=0)
    with cols[1]:
        eureka_threshold_single = st.number_input('Eureka Threshold (Single)',min_value=0,max_value=100,value=15,step=1)
    with cols[2]:
        eureka_threshold_overall = st.number_input('Eureka Threshold (Overall)',min_value=0,max_value=100,value=1,step=1)
    with cols[3]:
        st.write('')
        st.write('')
        absolute = st.checkbox('Absolute',value=True)

    combined_eureka = pd.DataFrame()

    st.subheader('Raw Data for Computing Fixed-baseline Eureka')
    for scale in leaderboards_normed:
        relative = baseline if baseline in baselines[scale] else default_baseline
        _leaderboards_normed = leaderboard_filter(leaderboards_normed[scale])
        cols = st.columns(2)
        with cols[0]:
            with st.expander(f'{scale} Normed metrics (0-1, higher is better)',expanded=False):
                leaderboards_normed_ = _leaderboards_normed.copy()
                leaderboards_normed_['avg.'] = leaderboards_normed_.mean(axis=1)
                leaderboards_normed_['max.'] = leaderboards_normed_.max(axis=1)
                st.dataframe(leaderboards_normed_)
        with cols[1]:
            with st.expander(f'{scale} Relative to ```{relative}``` (Normed metrics, %, {"relative" if absolute else "absolute"})',expanded=False):
                leaderboards_relative = leaderboard_relative(_leaderboards_normed,relative=relative,absolute=absolute)
                leaderboards_relative['avg.'] = leaderboards_relative.mean(axis=1)
                leaderboards_relative['max.'] = leaderboards_relative.max(axis=1)
                leaderboards_relative['eureka'] = (
                    (leaderboards_relative['avg.'] >= eureka_threshold_overall) 
                    | (leaderboards_relative['max.'] >= eureka_threshold_single)
                )
                st.dataframe(leaderboards_relative)
                combined_eureka[scale] = leaderboards_relative['eureka']

    st.subheader('Combined Eureka Moments')
    # fill the missing rows with False
    combined_eureka = combined_eureka.fillna(False)
    combined_eureka = combined_eureka.drop(BASIC_BASELINES, errors='ignore')
    combined_eureka['eureka'] = combined_eureka.any(axis=1)
    # add time stamp column by 
    def get_timestamp(x):
        return evosys.ptree.get_node(x).timestamp
    combined_eureka['timestamp'] = combined_eureka.index.map(get_timestamp)
    st.dataframe(combined_eureka)

    
    




class EvoModes(Enum):
    EVOLVE = 'Evolution System'
    BENCH = 'Agent Benchmark'
    EUREKA = 'Eureka Moments'


def evolve(evosys,project_dir):
    
    if 'command_center' not in st.session_state:
        st.session_state.command_center = None
    if 'evo_process_pid' not in st.session_state:
        st.session_state.evo_process_pid = None
    if 'evo_passive_thread' not in st.session_state:
        st.session_state.evo_passive_thread = None


    with st.sidebar:
        AU.running_status(st,evosys)

        # benchmark_mode = st.checkbox("***Agent Benchmark 🪑***",
        #     value=evosys.benchmark_mode,disabled=st.session_state.evo_running)
        mode = st.selectbox("Mode",options=[i.value for i in EvoModes],index=0,
            help='Choose the mode to view the evolution system or the agent benchmark.'
        )
        mode = EvoModes(mode)

        st.button('🔄 Refresh',use_container_width=True)
        
        if st.session_state.command_center:
            st.download_button(
                label="📩 Download Logs",
                data=json.dumps(st.session_state.command_center.read_logs(),indent=4),
                file_name=f"{evosys.evoname}_logs.json",
                mime="text/json",
                use_container_width=True
            )


        
    if mode == EvoModes.EVOLVE:
        st.title("Evolution System")
        if not evosys.benchmark_mode:
            st.warning(f'The namespace ```{evosys.evoname}``` is set to evolution mode. Please do not run benchmark in this namespace.')
    elif mode == EvoModes.BENCH:
        st.title("Agent Benchmark")
        if evosys.benchmark_mode:
            st.warning(f'The namespace ```{evosys.evoname}``` is set to benchmark mode. Please do not run evolution in this namespace.')
    elif mode == EvoModes.EUREKA:
        st.title("Eureka Moments")


    assert evosys.remote_db, "You must connect to a remote database to run the evolution."

    if mode in [EvoModes.EVOLVE,EvoModes.BENCH]:
        _evolve(evosys,mode)
    elif mode == EvoModes.EUREKA:
        _eureka(evosys)



if __name__ == '__main__':
    from model_discovery.evolution import BuildEvolution
    import argparse

    AU.print_cli_title()

    setting=AU.get_setting()
    default_namespace=setting.get('default_namespace','test_evo_000')   

    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--evoname', default=default_namespace, type=str) # the name of the whole evolution
    parser.add_argument('-dn','--max_designs_per_node', type=int, default=0) # the max number of threads to use
    parser.add_argument('-dt','--max_designs_total', type=int, default=10) # the group id of the evolution
    parser.add_argument('-g','--group_id', default='default', type=str) # the group id of the evolution
    parser.add_argument('-nr','--no_resume', action='store_true')
    args = parser.parse_args()
        
    evosys = BuildEvolution(
        params={'evoname':args.evoname, 'group_id':args.group_id}, 
        do_cache=False,
        # cache_type='diskcache',
    )

    
    if evosys.benchmark_mode:
        print(f'🪑 Launching benchmark for namespace {args.evoname} with group id {args.group_id}')
    else:
        print(f'🚀 Launching evolution for namespace {args.evoname} with group id {args.group_id}')

    command_center = CommandCenter(evosys,args.max_designs_per_node,args.max_designs_total,st,allow_resume=not args.no_resume)
    command_center.build_connection()
    command_center.run()


    # command_center_thread=x_evolve(command_center,cli=True)
    
    # print("Evolution launched!")
    # try:
    #     # Keep the main thread alive
    #     while command_center_thread.active:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("Shutting down...")
    # finally:
    #     command_center_thread.stop_listening()
    #     command_center_thread.join(timeout=10)  # Wait for the thread to finish, with a timeout

    # print("Evolution stopped.")


