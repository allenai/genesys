import json
import time
import pathlib
import ast
import streamlit as st
import sys,os
from enum import Enum
from subprocess import check_output
import pytz
import random
import altair as alt
import streamlit.components.v1 as components
from google.cloud import firestore
from datetime import datetime, timedelta
import threading
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import subprocess
from streamlit.runtime.scriptrunner import add_script_run_ctx
from wordcloud import WordCloud

from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import wandb
import re

sys.path.append('.')
import model_discovery.utils as U
from model_discovery.evolution import BENCHMARK_DIR
import bin.app_utils as AU
from bin.pages.listen import DESIGN_ACTIVE_STATES,VERIFY_ACTIVE_STATES
from bin.pages.viewer import export_leaderboards,leaderboard_filter,leaderboard_relative


CC_POLL_FREQ = 30 # seconds
CC_ZOMBIE_THRESHOLD = 60 # seconds
CC_COMMAND_DELAY = 2 # seconds


def _is_running(evosys):
    if st.session_state.is_demo:
        return None, None
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
        if not st.session_state.is_demo:
            self.doc_ref = evosys.remote_db.collection('experiment_connections').document(self.evosys.evoname)
        else:
            self.doc_ref = None
        self.running = False
        self.poll_freq=CC_POLL_FREQ
        self.zombie_threshold = CC_ZOMBIE_THRESHOLD  # seconds
        self.allow_resume = allow_resume

    def read_logs(self):
        if st.session_state.is_demo:
            return {}
        return self.evosys.CM.get_log_ref().get().to_dict()

    def build_connection(self,active_mode=True):
        # check if the node_id is already in the collection
        if st.session_state.is_demo:
            return
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
        if not st.session_state.is_demo:
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
                    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Done, stopping the command center. &&&&&&&')
                    # self.evosys.conclude()
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
        if st.session_state.is_demo:
            return
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
                if st.session_state.use_cache:
                    models = st.session_state.filter_by_types['DesignArtifactImplemented'] + st.session_state.filter_by_types['ReferenceCoreWithTree']
                else:
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
        if st.session_state.is_demo:
            return
        self.doc_ref.set({'user_command_stack': []},merge=True)

    def show_user_command_stack(self):
        if st.session_state.is_demo:
            user_commands = []
        else:
            user_commands = self.doc_ref.get().to_dict().get('user_command_stack',[])
        if len(user_commands)==0:
            st.info('No command stacks available at the moment.')
        else:
            st.write(user_commands)

def x_evolve_passive(command_center,cli=False): # extereme evolution 
    thread = threading.Thread(target=command_center.run)
    if st.session_state.is_demo:
        return thread, None
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
    if st.session_state.is_demo:
        st.toast('Evolution is not really running in demo mode.',icon='🚨')
    else:
        if active_mode and len(evosys.CM.get_active_connections())==0:
            st.toast('No nodes connected. Please remember to launch nodes.',icon='🚨')
    command_center = CommandCenter(evosys,max_designs_per_node,max_designs_total,st,allow_resume=allow_resume) # launch a passive command center first
    if active_mode and not st.session_state.is_demo:
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
            try:
                st.session_state.evo_passive_thread.join(timeout=5)
            except Exception as e:
                pass
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


def network_status(evosys,benchmark_mode=False):
    group_id = 'null' if st.session_state.is_demo else evosys.CM.group_id
    st.write(f'#### *Network Group ```{group_id}``` Status*')

    with st.expander('Nodes Running Status',expanded=True):
        nodes = None if st.session_state.is_demo else evosys.CM.get_active_connections()
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
        active_design_sessions = [] if st.session_state.is_demo else evosys.CM.get_active_design_sessions()
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

        running_verifications = {} if st.session_state.is_demo else evosys.CM.get_running_verifications()
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
            if not benchmark_mode:
                st.info('No running verifications')
            else:
                st.info('No running verifications ***(verification is not needed in agent benchmark)***')

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

    if st.session_state.is_demo:
        st.warning('***Demo mode:** No real evolution will be launched, but it will show how the evolution system works.*')

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
                disabled=passive_mode or evosys.benchmark_mode,
                use_container_width=True
            ) 
        else:
            stop_evo_btn = st.button(
                "***Stop Evolution*** 🛑",
                # disabled=not evosys.remote_db,
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
                disabled=not st.session_state.evo_running or st.session_state.is_demo)
    with cols[2]:
        st.write('')
        st.write('')
        clear_command_btn = st.button('Clear Command',use_container_width=True,
                disabled=not st.session_state.evo_running or st.session_state.is_demo)
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

    col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,0.9],gap='small')
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

    
    with col4:
        # always use extreme mode, use as much gpus as possible
        verify_schedule=st.selectbox("Verification Scheduling",['full utilization'],disabled=True, #st.session_state.evo_running,
            help='The strategy to schedule verification jobs across nodes. Currently, only full utilization is supported, which means scheduling verify jobs immediately when a node is free. (***Verification is not needed in agent benchmark***)'
        )


    with col6:
        st.write('')
        st.write('')
        input_allow_resume=st.checkbox("Allow Resume",value=True,disabled=True,
            help='Whether allow resume training. Not useful in benchmark mode.'
        )

    with col5:
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
                evosys.set_benchmark_mode()
                launch_evo(evosys,input_max_designs_per_node,input_max_designs_total,True,allow_resume=input_allow_resume)
    else:
        if stop_bench_btn:
            if st.session_state.command_center:
                with st.spinner('Stopping... Note, the nodes will keep working on the unfinished jobs'):
                    stop_evo(evosys)
        

def running_logs(evosys):
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


def ptree_monitor(evosys):
    st.subheader("Phylogenetic Tree Monitor")

    col1, col2, col3, col4 = st.columns([6,0.1,1.2,1])
    num_nodes = len(evosys.ptree.G.nodes)
    
    default_max_nodes = min(300,num_nodes) #296 
    export_height = 900 # 900
    default_evo_only = num_nodes>default_max_nodes
    size_mult = 1

    if 'ptree_max_nodes' not in st.session_state:
        st.session_state.ptree_max_nodes=default_max_nodes
    
    _bg_color=AU.theme_aware_options(st,"#fafafa","#f0f0f0","#fafafa")

    _bg_color='#ffffff'
    

    with col1:
        _max_nodes=st.slider('Max Nodes to Display',min_value=0,max_value=len(evosys.ptree.G.nodes),value=st.session_state.ptree_max_nodes)

    with col4:
        st.write('')
        st.write('')
        input_evo_only=st.checkbox("Evo Only",value=default_evo_only,help='Only show the evolution nodes.')

    # check this: https://github.com/napoles-uach/streamlit_network 
    with col3:
        st.write('')
        st.write('')
        reload_btn = st.button(f'Reload',use_container_width=True)
            
    if reload_btn:
        if not st.session_state.evo_running:
            evosys.ptree.update_design_tree()
        ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_{_max_nodes}.html')
        if not (st.session_state.use_cache and os.path.exists(ptree_dir_small)):
            with st.status(f'Exporting ptree for {st.session_state.ptree_max_nodes} nodes...',expanded=False):
                evosys.ptree.export(max_nodes=_max_nodes,height=f'{export_height}px',bgcolor=_bg_color,
                    legend_font_size=12,legend_width_constraint=100,legend_x=-2400,legend_y=-200,legend_step=100,
                    evo_only=input_evo_only,size_mult=size_mult)
        st.session_state.ptree_max_nodes=_max_nodes
    else:
        ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_{st.session_state.ptree_max_nodes}.html')
        if not (st.session_state.use_cache and os.path.exists(ptree_dir_small)):
            with st.status(f'Exporting ptree for {st.session_state.ptree_max_nodes} nodes...',expanded=False):
                evosys.ptree.export(max_nodes=st.session_state.ptree_max_nodes,height=f'{export_height}px',
                    bgcolor=_bg_color,evo_only=default_evo_only,size_mult=size_mult)
                
    if st.session_state.use_cache:
        n_implemented = len(st.session_state.filter_by_types['DesignArtifactImplemented'])
        n_designs = n_implemented + len(st.session_state.filter_by_types['DesignArtifact'])
    else:
        n_implemented = len(evosys.ptree.filter_by_type('DesignArtifactImplemented'))
        n_designs = n_implemented + len(evosys.ptree.filter_by_type('DesignArtifact'))
    
    st.write(f'**First {st.session_state.ptree_max_nodes} nodes under the namespace ```{evosys.evoname}```**, :red[{n_designs}] designs, :blue[{n_implemented}] implemented. *(Node Size by # of citations or children)*.')
            # 'Legend: :red[Seed Designs (*Displayed Pink*)] | :blue[Design Artifacts] | :orange[Reference w/ Code] | :violet[Reference w/o Code] *(Size by # of citations)*')

    HtmlFile = open(ptree_dir_small, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = export_height)


def _evolve(evosys):
    # else:
    #     if st.session_state.evo_running:
    #         st.toast(f'The command center is not active anymore. You may stop listing to command center.')
    #         # stop_evo(evosys)

    st.title("Evolution System")
    if evosys.benchmark_mode:# and not st.session_state.evo_running:
        st.error(f'The namespace ```{evosys.evoname}``` is set to benchmark mode. Please do not run evolution in this namespace.')
    else:
        evolution_launch_pad(evosys)

    network_status(evosys)
    running_logs(evosys)
    ptree_monitor(evosys)



def count_class_loc(source_code, base_class_name):
    """Finds classes inherited from a base class and counts their lines of code."""
    # Parse the source code into an AST
    source_code = re.sub(r'"""[\s\S]*?"""', '', source_code)
    source_code = re.sub(r"'''[\s\S]*?'''", '', source_code)
    tree = ast.parse(source_code)

    # Find classes that inherit from the specified base class
    target_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):  # If the node is a class definition
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == base_class_name:
                    target_classes.append(node)

    # Count lines of code for the target classes
    total_lines = 0
    for target_class in target_classes:
        class_start = target_class.lineno
        class_end = max(getattr(node, 'lineno', class_start) for node in ast.walk(target_class))

        # Extract and process the lines of the class
        class_code_lines = source_code.splitlines()[class_start - 1:class_end]
        for line in class_code_lines:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith("#"):  # Exclude comments and blank lines
                total_lines += 1

    return total_lines

def count_fn_loc(source_code):
    """Counts all function and method code lines, excluding comments and blank lines."""
    # Parse the source code into an AST
    source_code = re.sub(r'"""[\s\S]*?"""', '', source_code)
    source_code = re.sub(r"'''[\s\S]*?'''", '', source_code)
    tree = ast.parse(source_code)

    # Find all function definitions
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    # Count lines of code for each function
    total_lines = 0
    for func in functions:
        func_start = func.lineno
        func_end = max(getattr(node, 'lineno', func_start) for node in ast.walk(func))

        # Extract and process the lines of the function
        func_code_lines = source_code.splitlines()[func_start - 1:func_end]
        for line in func_code_lines:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith("#"):  # Exclude comments and blank lines
                total_lines += 1

    return total_lines

def _count_fn_loc(impl):
    if isinstance(impl,str):
        return count_fn_loc(impl)
    else:
        return sum([count_fn_loc(impl.units[i].code) for i in impl.units])

def _count_gab_loc(source_code):
    return count_class_loc(source_code,'GABBase')+count_class_loc(source_code,'Module')

def _count_gau_loc(source_code):
    return count_class_loc(source_code,'GAUBase')

def _count_ga_loc(impl):
    if isinstance(impl,str):
        return _count_gab_loc(impl)
    else:
        return sum([_count_gau_loc(impl.units[i].code) for i in impl.units])


def bench_summary(evosys):
    st.subheader("🏆 Benchmark Status")
    with st.expander(f"**Benchmark Summary for ```{evosys.evoname}```**",expanded=True):
        bench_designs = os.listdir(BENCHMARK_DIR)
        nodes = []
        for d in bench_designs:
            node = evosys.ptree.get_node(d)
            if node is None:
                print(f'Benchmark design {d} not found in the phylogenetic tree.')
            else:
                nodes.append(node)
        status = {}
        rounds = {}
        costs = {}
        raw_states = {}
        no_fcheckers = False
        succeeded = []
        failed = []
        unfinished = []
        codes = {}
        for node in nodes:
            if node.implementation:
                state,n_tries = node.state.split(':')
                codes[node.acronym] = node.implementation.implementation
                raw_states[node.acronym] = state
                threshold = 5
                if ('implemented' in state or 'succeeded' in state) and int(n_tries)<=threshold:
                    status[node.acronym] = 'succeeded'
                elif int(n_tries)<threshold:
                    status[node.acronym] = 'unfinished'
                else:
                    status[node.acronym] = 'failed'
                if 'invalid' in state and status[node.acronym]!='unfinished':
                    status[node.acronym] += ' (invalid)'
                    no_fcheckers = True
                elif 'valid' in state and status[node.acronym]!='unfinished':
                    status[node.acronym] += ' (valid)'
                    no_fcheckers = True
                if status[node.acronym] in ['succeeded','succeeded (valid)']:
                    succeeded.append(node.acronym)
                elif status[node.acronym] in ['failed','failed (invalid)','failed (valid)','succeeded (invalid)']:
                    failed.append(node.acronym)
                elif status[node.acronym]=='unfinished':
                    unfinished.append(node.acronym)
                rounds[node.acronym] = min(int(n_tries),threshold)
                costs[node.acronym] = sum(node.implementation.get_cost().values())
            else:
                status[node.acronym] = 'unfinished'
                unfinished.append(node.acronym)
        freqs = _data_to_freq(status)
        raw_freqs = _data_to_freq(raw_states)
        # st.write(raw_freqs)
        avg_rounds=np.mean(list(rounds.values())) if rounds else 0
        avg_costs = np.mean(list(costs.values())) if costs else 0
        std_costs = np.std(list(costs.values())) if costs else 0
        std_rounds = np.std(list(rounds.values())) if rounds else 0
        if not no_fcheckers:
            if 'succeeded' not in freqs:
                freqs['succeeded']=0
            if 'failed' not in freqs:
                freqs['failed']=0
            freqs['unfinished']=len(nodes)-freqs['succeeded']-freqs['failed']
            success_rate = freqs['succeeded']/len(nodes)
            st.write(
                f'{len(nodes)/len(bench_designs):.2%} of benchmark nodes loaded. ',
            )
            st.write(
                f':green[{success_rate:.2%}] succeeded, ',
                f':red[{freqs["failed"]/len(nodes):.2%}] failed, ',
                f':grey[{freqs["unfinished"]/len(nodes):.2%}] unfinished. ',
            )
            st.write(
                f'Avg. attempts: :blue[{avg_rounds:.2f}] (std: :red[{std_rounds:.2f}]), ',
                f'Avg. cost: :blue[{avg_costs:.2f}] (std: :red[{std_costs:.2f}]), ',
                f'Adjusted cost: :blue[{avg_costs/success_rate:.2f}] (std: :red[{std_costs/success_rate:.2f}]). '
            )
            locs_fn = [_count_fn_loc(codes[d]) for d in succeeded]
            locs_ga = [_count_ga_loc(codes[d]) for d in succeeded]
            st.write(
                f'Avg. LoC (by function): :blue[{np.mean(locs_fn):.2f}] (std: :red[{np.std(locs_fn):.2f}]), ',
                f'Avg. LoC (by class): :blue[{np.mean(locs_ga):.2f}] (std: :red[{np.std(locs_ga):.2f}]). '
            )
        else:
            if 'succeeded (valid)' not in freqs:
                freqs['succeeded (valid)']=0
            if 'failed (invalid)' not in freqs:
                freqs['failed (invalid)']=0
            if 'succeeded (invalid)' not in freqs:
                freqs['succeeded (invalid)']=0
            if 'failed (valid)' not in freqs: # theoretically impossible
                freqs['failed (valid)']=0
            freqs['Succeded & Valid']=freqs['succeeded (valid)']+freqs['failed (valid)']
            freqs['Failed / Invalid']=freqs['failed (invalid)']+freqs['succeeded (invalid)']+freqs.get('failed',0)
            freqs['unfinished']=len(nodes)-freqs['Succeded & Valid']-freqs['Failed / Invalid']
            success_rate = freqs['Succeded & Valid']/len(nodes)
            st.write(
                f'{len(nodes)/len(bench_designs):.2%} of benchmark nodes loaded. ',
            )
            st.write(
                f':green[{freqs["Succeded & Valid"]/len(nodes):.2%}] succeded & valid, ',
                f':red[{freqs["Failed / Invalid"]/len(nodes):.2%}] failed or invalid, ',
                f':grey[{freqs["unfinished"]/len(nodes):.2%}] unfinished. ',
            )   
            st.write(
                f'Avg. attempts: :blue[{avg_rounds:.2f}] (std: :red[{std_rounds:.2f}]), ',
                f'Avg. cost: :blue[{avg_costs:.2f}] (std: :red[{std_costs:.2f}]), ',
                f'Adjusted cost: :blue[{avg_costs/success_rate:.2f}] (std: :red[{std_costs/success_rate:.2f}]). '
            )
            locs_fn = [_count_fn_loc(codes[d]) for d in succeeded]
            locs_ga = [_count_ga_loc(codes[d]) for d in succeeded]
            st.write(   
                f'Avg. LoC (by function): :blue[{np.mean(locs_fn):.2f}] (std: :red[{np.std(locs_fn):.2f}]), ',
                f'Avg. LoC (by class): :blue[{np.mean(locs_ga):.2f}] (std: :red[{np.std(locs_ga):.2f}]). '
            )
        
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander(f'Succeeded Designs: `{len(succeeded)}`'):
            _data = {
                'Succeeded':succeeded,
                'Attempts':[rounds[d] for d in succeeded],
                'Cost':[costs[d] for d in succeeded]
            }
            st.dataframe(_data,use_container_width=True)
    with col2:
        with st.expander(f'Failed Designs: `{len(failed)}`'):
            _data = {
                'Failed':failed,
                'Attempts':[rounds[d] for d in failed],
                'Cost':[costs[d] for d in failed]
            }
            st.dataframe(_data,use_container_width=True)
    with col3:
        with st.expander(f'Unfinished Designs: `{len(unfinished)}`'):
            _data = {
                'Unfinished':unfinished,
                'Attempts':[rounds.get(d, None) for d in unfinished],
                'Cost':[costs.get(d, None) for d in unfinished]
            }
            st.dataframe(_data,use_container_width=True)



def _bench(evosys):
    st.title("Agent Benchmark")
    if not evosys.benchmark_mode:# and not st.session_state.evo_running:
        st.error(f'The namespace ```{evosys.evoname}``` is not set to benchmark mode. Please set it to benchmark mode to launch the benchmark.')
    else:
        benchmark_launch_pad(evosys)

    network_status(evosys,benchmark_mode=True)
    running_logs(evosys)
    bench_summary(evosys)
    ptree_monitor(evosys)



BASIC_BASELINES = [
    'gpt2 (baseline)',
    'rwkv6 (baseline)',
    'mamba2 (baseline)',
    'ttt (baseline)',
    'retnet (baseline)',
    'random',
]

def _eureka(evosys):
    st.title("Eureka Moments (Experimental)")
    default_baseline = 'random'
    cols = st.columns(5)
    with cols[0]:
        baseline = st.selectbox('Baseline',options=BASIC_BASELINES,index=2)
    with cols[1]:
        eureka_threshold_single = st.number_input('Eureka Threshold (Single)',min_value=0,max_value=100,value=10,step=1)
    # with cols[2]:
    #     eureka_threshold_overall = st.number_input('Eureka Threshold (Overall)',min_value=0,max_value=100,value=1,step=1)
    with cols[2]:
        first_N = st.number_input('First N Designs',min_value=0,max_value=100,value=0,step=50)
        first_N = None if first_N==0 else first_N
    with cols[3]:
        filter_threshold = st.number_input('Filter Threshold (%)',min_value=0,max_value=100,value=5,step=1,
            help='Leave the metrics where there is at least one design with relative rating higher than this threshold'
        )
    with cols[4]:
        st.write('')
        st.write('')
        absolute = st.checkbox('Absolute Diff.',value=True,help='Whether to use absolute values or relative difference compared to the baseline.')

        
    with st.status('Loading latest data...'):
        design_vectors = evosys.ptree.get_design_vectors(first_N=first_N)
        baseline_vectors = evosys.ptree.get_baseline_vectors()
        leaderboards_normed,_,_,baselines=export_leaderboards(evosys,design_vectors,baseline_vectors)
        leaderboards_normed.pop('all')


    combined_eureka = pd.DataFrame()
    _scales = list(leaderboards_normed.keys())
    sorted_scales = sorted(_scales,key=lambda x: int(x.replace('M','')))
    st.subheader('Raw Data for Computing Fixed-baseline Eureka')
    for scale in sorted_scales:
        relative = baseline if baseline in baselines[scale] else default_baseline
        _leaderboards_normed = leaderboard_filter(leaderboards_normed[scale])
        leaderboards_relative = leaderboard_relative(_leaderboards_normed,relative=relative,absolute=absolute,filter_threshold=filter_threshold)
        leaderboards_relative['avg.'] = leaderboards_relative.mean(axis=1)

        leaderboards_normed_combined = _leaderboards_normed.copy()
        leaderboards_normed_combined = leaderboards_normed_combined.loc[:,leaderboards_relative.columns]

        # Drop NA values from both DataFrames
        leaderboards_normed_combined = leaderboards_normed_combined.dropna()
        leaderboards_relative = leaderboards_relative.dropna()
        
        # Ensure both DataFrames have the same index after dropping NA
        common_index = leaderboards_normed_combined.index.intersection(leaderboards_relative.index)
        leaderboards_normed_combined = leaderboards_normed_combined.loc[common_index]
        leaderboards_relative = leaderboards_relative.loc[common_index]
        # recompute avg for both, remove the old avg
        leaderboards_normed_combined = leaderboards_normed_combined.drop(columns=['avg.'])
        leaderboards_relative = leaderboards_relative.drop(columns=['avg.'])
        leaderboards_normed_combined['avg.'] = leaderboards_normed_combined.mean(axis=1)
        # _relative = f'{relative} (baseline)' if relative != 'random' else 'random'
        relative_avg = leaderboards_normed_combined.loc[relative,'avg.']
        leaderboards_relative['avg.'] = 100*(leaderboards_normed_combined['avg.'] - relative_avg)/relative_avg
        
        # Combine the values of the two leaderboards as normed (relative) e.g., 3.2 (4.5%)
        def combine_values(normed, relative):
            return normed.applymap(lambda x: f'{x:.4f}') + ' (' + relative.applymap(lambda x: f'{x:.2f}%') + ')'

        leaderboards_normed_combined = combine_values(leaderboards_normed_combined, leaderboards_relative)


        highlight_color = 'violet'
        with st.expander(f'Combined leaderboard for ```{scale}``` (with relative (%) to ```{relative}```, max highlighted in :{highlight_color}[{highlight_color}])',expanded=True):
            baseline_rows = leaderboards_normed_combined[leaderboards_normed_combined.index.str.contains('(baseline)')]
            random_row = leaderboards_normed_combined.loc['random']
            remaining_rows = leaderboards_normed_combined[~leaderboards_normed_combined.index.isin(baseline_rows.index)]
            remaining_rows = remaining_rows.drop(index='random')
            remaining_rows = remaining_rows.sort_values(by='avg.',ascending=True)
            leaderboards_normed_combined = pd.concat([random_row.to_frame().T,baseline_rows,remaining_rows])
            st.dataframe(leaderboards_normed_combined.style.highlight_max(axis=0,color=highlight_color),use_container_width=True)
        
        leaderboards_relative_remaining = leaderboards_relative.loc[remaining_rows.index]
        # mark eureka designs (rows) by the number of columns that are >= eureka_threshold_single
        # num_highlights = leaderboards_relative_remaining.apply(lambda x: (x >= eureka_threshold_single).sum(), axis=1)
        _highlights = leaderboards_relative_remaining.apply(lambda x: x[x >= eureka_threshold_single].sum(), axis=1)
        _disappointments = leaderboards_relative_remaining.apply(lambda x: x[x < -eureka_threshold_single].sum(), axis=1)
        combined_eureka[scale] = _highlights+_disappointments



    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Combined Eureka Moments')
        # combined_eureka = combined_eureka.fillna()
        def _reweight(x):
            # mean over non-None values
            return x.mean()
        combined_eureka['eureka'] = combined_eureka.apply(lambda x: _reweight(x),axis=1)
        combined_eureka = combined_eureka.drop(BASIC_BASELINES, errors='ignore')
        # add time stamp column by 
        def get_timestamp(x):
            return evosys.ptree.get_node(x).timestamp
        combined_eureka['timestamp'] = combined_eureka.index.map(get_timestamp)
        combined_eureka = combined_eureka.sort_values(by='timestamp',ascending=True)
        st.dataframe(combined_eureka)

    with col2:
        st.subheader('Population Eureka over time')
        population_size = 50
        step_size = 30
        _col1, _col2 = st.columns(2)
        with _col1:
            population_size = st.slider('Population size',min_value=10,max_value=100,value=50,step=10)
        with _col2:
            step_size = st.slider('Step size',min_value=10,max_value=100,value=30,step=10)
        generations = []
        gen_sizes = []
        eureka_means = []
        eureka_stds = []
        eureka_maxs = []
        eureka_mins = []
        for i in range(0,len(combined_eureka),step_size):
            _designs = list(combined_eureka.index)[i:i+population_size]
            if len(_designs)<population_size*0.7:
                continue
            gen_sizes.append(len(_designs))
            generations.append(int(i/step_size))
            eureka_means.append(np.mean(combined_eureka.loc[_designs]['eureka']))
            eureka_stds.append(np.std(combined_eureka.loc[_designs]['eureka']))
            eureka_maxs.append(np.max(combined_eureka.loc[_designs]['eureka']))
            eureka_mins.append(np.min(combined_eureka.loc[_designs]['eureka']))
        chart_data = pd.DataFrame({
            'generation':generations,
            'size':gen_sizes,
            'mean':eureka_means,
            'std':eureka_stds,
            'max':eureka_maxs,
            'min':eureka_mins
        })
        chart_data['std_upper'] = chart_data['mean'] + chart_data['std']
        chart_data['std_lower'] = chart_data['mean'] - chart_data['std']

        line = alt.Chart(chart_data).mark_line(color='blue').encode(x='generation', y='mean')
        line_max = alt.Chart(chart_data).mark_line(color='red', strokeDash=[4,4]).encode(x='generation', y='max')
        line_min = alt.Chart(chart_data).mark_line(color='green', strokeDash=[4,4]).encode(x='generation', y='min')
        band = alt.Chart(chart_data).mark_area(opacity=0.2).encode(x='generation', y='std_lower', y2='std_upper')
        chart = band + line + line_max + line_min
        st.altair_chart(chart.interactive(), use_container_width=True)




def _draw_pie(data,startangle=90):
    labels = [f'{k}: {v}' for k,v in data.items()]
    sizes = list(data.values())
    explode = tuple([0]*len(labels))
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
        shadow=True, startangle=startangle)
    ax.axis('equal')  
    return fig

def _draw_pie_alt(data, startangle=90):
    # Convert the dictionary to a DataFrame
    data_df = pd.DataFrame(list(data.items()), columns=['Category', 'Value'])
    
    # Calculate the percentage for each slice
    total = sum(data.values())
    data_df['Percentage'] = data_df['Value'] / total * 100
    total_value = data_df['Value'].sum()
    data_df['Category'] = data_df.apply(lambda row: f"{row['Category']} ({row['Value']/total_value:.1%})", axis=1)  # Label with category and value
    data_df['Label'] = data_df['Percentage'].map('{:.1f}%'.format)  # Format percentage labels
    
    
    # Create the pie chart
    pie_chart = alt.Chart(data_df).mark_arc(innerRadius=0).encode(
        theta=alt.Theta('Value', type='quantitative'),
        color=alt.Color('Category', type='nominal'),
        tooltip=[alt.Tooltip('Category', title='Category'), 
                 alt.Tooltip('Value', title='Value'),
                 alt.Tooltip('Percentage', format='.2f', title='Percentage (%)')]
    ).configure_view(
        strokeWidth=0
    ).configure_arc(
        startAngle=startangle
    )# .properties(
    #     title="Pie Chart"
    # )
    
    return pie_chart

def _data_to_freq(data,unit=None,outlier=None,precision=2):
    freq = {}
    outliers = 0
    if isinstance(data,dict):
        values = list(data.values())
    else:
        values = data
    for v in values:
        if unit:
            v = (v//unit)*unit
            v = round(v,precision)
        if outlier:
            if v>outlier:
                outliers += 1
                continue
        freq[v] = freq.get(v,0) + 1
    if outlier:
        return freq,outliers
    return freq

def _data_filter(data,filter_func):
    return {k:v for k,v in data.items() if filter_func(v)}

def _score_stratify(_scores):
    return np.array([
        _score_stratify_single(x) for x in _scores
    ])

def _score_stratify_single(score):
    return '<0.6' if score < 0.6 else '0.6-0.65' if score < 0.65 else '0.65-0.7' if score < 0.7 else '>0.7'

def _stratify_to_weight(stratify):
    if stratify=='<0.6':
        return 0.25
    elif stratify=='0.6-0.65':
        return 0.5
    elif stratify=='0.65-0.7':
        return 0.75
    else:
        return 1
    
def _stratify_to_mean(stratify):
    if stratify=='<0.6':
        return 0.55
    elif stratify=='0.6-0.65':
        return 0.625
    elif stratify=='0.65-0.7':
        return 0.675
    else:
        return 0.75

def session_stats(evosys,design_nodes,implemented_nodes):
    EXPORT_DIR = U.pjoin(evosys.ckpt_dir,'EXPORT')
    U.mkdir(EXPORT_DIR)
    st.subheader("Session Statistics Monitor")
    with st.expander(f"Design Session Statistics for ```{evosys.evoname}```",expanded=True):#,icon='📊'):
        # evosys.ptree.update_design_tree()

        sessions = evosys.ptree.design_sessions

        states = {}
        costs = {}
        rounds = {}
        impl_costs = {}
        verified = {}
        proposal_costs = {}
        avg_score = {}
        ratings = {}
        attempts = {}
        timestamps = {}
        cfg_implementations = {}
        cfg_proposals = {}
        for node in design_nodes+implemented_nodes:
            design = node.acronym
            state = node.state
            if ':' in state:
                state, attempt = state.split(':')
            else:
                attempt = 0
            states[design] = state
            costs[design] = sum(node.get_cost(with_history=True).values())
            proposal_costs[design] = sum(node.proposal.costs.values())
            if node.implementation:
                impl_costs[design] = sum(node.implementation.get_cost(with_history=True).values())
            try:
                timestamps[design] = node.timestamp
            except:
                timestamps[design] = None
            avg_score[design] = np.mean(list(node.get_scores().values())) #node.get_score(scale='14M')
            ratings[design] = node.proposal.rating
            cfg_proposals[design] = node.proposal.design_cfg
            verified[design] = len(node.verifications)>0
            if node.implementation and node.implementation.history:
                cfg_implementations[design] = node.implementation.history[-1].design_cfg
                rounds[design] = []
                for h in node.implementation.history:
                    rounds[design].append(len(h.rounds))
                attempts[design] = attempt #sum(rounds[design]) #attempt


        if len(design_nodes)+len(implemented_nodes)+len(sessions) == 0:
            st.info('No design session statistics available at the moment.')
        else:
            # _angle=st.slider('Pie chart start angle',min_value=0,max_value=360,value=90,step=10)
            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader('Design state distribution')  
                state_counts = _data_to_freq(states)
                unfinished_states = ['unfinished (initial_pass)',"proposed (unimplemented)","unfinished (unfinished)"]
                # state_counts['implementing'] = sum([state_counts[k] for k in to_merge])
                for k in unfinished_states:
                    state_counts.pop(k)
                rename_map = {
                    "implementing":'Implementing',
                    "unfinished (failed)":'Failed (Implementation)',
                    "failed":'Failed (Proposal)',
                    "implemented (unverified)":'Failed (Verification)',
                    "implemented (verified)":'Implemented & Verified',
                }
                renamed_state_counts = {rename_map[k]:v for k,v in state_counts.items()}
                st.altair_chart(
                    _draw_pie_alt(renamed_state_counts,startangle=100).properties(
                        width=400,
                        height=300
                    ).configure_legend(
                        orient='right',  # Place legend on the right
                        labelLimit=300,  # Increase label width limit
                        padding=10,      # Add padding around legend
                        offset=5,         # Offset from the chart
                        labelColor='white',
                        titleColor='white',
                    ).interactive()
                )
            with col2:
                st.subheader('Implementation attempt distribution')
                # st.pyplot(_draw_pie(attempt_counts,startangle=30))
                attempt_counts = _data_to_freq(attempts)
                attempt_counts = {int(k):v for k,v in attempt_counts.items()}
                chart_data = pd.DataFrame(list(attempt_counts.items()),columns=['Implementation attempts','Frequency'])
                st.bar_chart(chart_data,x='Implementation attempts',y='Frequency')

            st.subheader('Design cost distribution')
            finished_deigns = [k for k in costs if states[k] not in unfinished_states]
            finished_costs = {k:costs[k] for k in finished_deigns}
            mean_cost = np.mean(list(finished_costs.values()))
            std_cost = np.std(list(finished_costs.values()))
            outlier=100
            costs_counts,outliers = _data_to_freq(costs,unit=1,outlier=outlier,precision=1)
            median_cost = np.median(list(finished_costs.values()))
            st.write(f'Mean: ```{mean_cost:.2f}```, Median: ```{median_cost:.2f}```, Std: ```{std_cost:.2f}```, Outliers (```{outlier}+```): ```{outliers}```')
            chart_data = pd.DataFrame(list(costs_counts.items()),columns=['Total cost','Frequency'])
            st.bar_chart(chart_data,x='Total cost',y='Frequency')

            col1, col2 = st.columns([1,1])
            with col1:
                finished_proposal_costs = {k:proposal_costs[k] for k in finished_deigns}
                mean_proposal_cost = np.mean(list(finished_proposal_costs.values()))
                std_proposal_cost = np.std(list(finished_proposal_costs.values()))
                outlier=50
                proposal_costs_counts,outliers = _data_to_freq(finished_proposal_costs,unit=1,outlier=outlier,precision=1)
                median_proposal_cost = np.median(list(finished_proposal_costs.values()))
                st.write(f'Mean: ```{mean_proposal_cost:.2f}```, Std: ```{std_proposal_cost:.2f}```, Median: ```{median_proposal_cost:.2f}```, Outliers (```{outlier}+```): ```{outliers}```')
                proposal_costs_counts.pop(0)
                chart_data = pd.DataFrame(list(proposal_costs_counts.items()),columns=['Proposal cost','Frequency'])
                st.bar_chart(chart_data,x='Proposal cost',y='Frequency')

            with col2:
                finished_impl_costs = {k:impl_costs.get(k,0) for k in finished_deigns}
                mean_impl_cost = np.mean(list(finished_impl_costs.values()))
                std_impl_cost = np.std(list(finished_impl_costs.values()))
                outlier=50
                impl_costs_counts,outliers = _data_to_freq(finished_impl_costs,unit=1,outlier=outlier,precision=1)
                median_impl_cost = np.median(list(finished_impl_costs.values()))
                st.write(f'Mean: ```{mean_impl_cost:.2f}```, Std: ```{std_impl_cost:.2f}```, Median: ```{median_impl_cost:.2f}```, Outliers (```{outlier}+```): ```{outliers}```')
                impl_costs_counts.pop(0)
                chart_data = pd.DataFrame(list(impl_costs_counts.items()),columns=['Implementation cost','Frequency'])
                st.bar_chart(chart_data,x='Implementation cost',y='Frequency')


            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader('Avg. fitness Distribution')
                scores_filtered = _data_filter(avg_score,lambda x: x is not None and 0<x<1)
                _scores = np.array(list(scores_filtered.values())).astype(float)
                mean_score = np.mean(_scores)
                median_score = np.median(_scores)
                std_score = np.std(_scores)
                scores_counts = _data_to_freq(_scores,unit=0.01)    
                st.write(f'Mean: ```{mean_score:.2f}```, Median: ```{median_score:.2f}```, Std: ```{std_score:.2f}```, Total: ```{len(scores_filtered)}```')
                chart_data = pd.DataFrame(list(scores_counts.items()),columns=['Fitness','Frequency'])
                st.bar_chart(chart_data,x='Fitness',y='Frequency')

            with col2:
                st.subheader('Fitness-Cost Correlation')
                center=0.58
                cost_filtered = {k:float(v) for k,v in costs.items() if k in scores_filtered}
                _costs = np.array(list(cost_filtered.values())).astype(float)
                _data = np.array([_costs,_scores-center,_scores-center]).T
                y_label = 'Fitness (centered)'
                y_tag = 'fitness (centered)'
                chart_data = pd.DataFrame(
                    _data, columns=["Total cost", y_label, y_tag]
                )
                chart_data["fitness_stratification"] = _score_stratify(_scores)
                corr_coef = np.corrcoef(_data.T)
                st.write(f'Correlation coefficient: ```{corr_coef[0,1]:.2f}```')
                st.scatter_chart(
                    chart_data,
                    x="Total cost",
                    y=y_label,
                    color="fitness_stratification",
                    size=y_tag,
                )
            
            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader('Proposal rating Distribution')
                proposal_ratings = {}
                proposal_ratings['all'] = list(ratings.values()) 
                reviewer_models = {k:v['_agent_types']['PROPOSAL_REVIEWER'] for k,v in cfg_proposals.items()}
                for d in reviewer_models:
                    model = reviewer_models[d]
                    if model not in proposal_ratings:
                        proposal_ratings[model] = []
                    proposal_ratings[model].append(ratings[d])

                def draw_rating_chart(_ratings):
                    mean_rating = np.mean(_ratings)
                    median_rating = np.median(_ratings)
                    std_rating = np.std(_ratings)
                    ratings_counts = _data_to_freq(_ratings,unit=0.1)
                    st.write(f'Mean: ```{mean_rating:.2f}```, Median: ```{median_rating:.2f}```, Std: ```{std_rating:.2f}```')
                    chart_data = pd.DataFrame(list(ratings_counts.items()),columns=['Proposal rating','Frequency'])
                    st.bar_chart(chart_data,x='Proposal rating',y='Frequency')

                _model = st.selectbox('Model',list(proposal_ratings.keys()))
                _ratings = proposal_ratings[_model]
                draw_rating_chart(_ratings)
                
            with col2:
                st.subheader('Proposal rating-Fitness Correlation')
                ratings_filtered =  {k:float(v) for k,v in ratings.items() if k in scores_filtered}
                _ratings = np.array(list(ratings_filtered.values())).astype(float)
                _data = np.array([_ratings-4.0,_scores-center,_scores-center]).T
                rating_label = 'rating (centered)'
                chart_data = pd.DataFrame(
                    _data, columns=[rating_label, y_label, y_tag]
                )
                chart_data["fitness_stratification"] = _score_stratify(_scores)
                corr_coef = np.corrcoef(_data.T)
                st.write(f'Correlation coefficient: ```{corr_coef[0,1]:.2f}```')
                st.scatter_chart(
                    chart_data,
                    x=rating_label,
                    y=y_label,
                    color="fitness_stratification",
                    size=y_tag,
                )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Fitness-Time Correlation')
                timestamp_filtered = {k:v for k,v in timestamps.items() if k in scores_filtered}
                _timestamps = np.array(list(timestamp_filtered.values()))#.astype(float)
                _data = np.array([_timestamps,_scores-0.5,_scores]).T
                y_label = 'fitness (centered)'
                y_tag = 'Fitness'
                chart_data = pd.DataFrame(
                    _data, columns=["timestamp", y_label, y_tag]
                )
                chart_data["fitness_stratification"] = _score_stratify(_scores)
                
                st.scatter_chart(
                    chart_data,
                    x="timestamp",
                    y=y_label,
                    color="fitness_stratification",
                    size=y_tag,
                )
            with col2:
                st.subheader('Fitness-Population over time')
                # population_size = 30
                # step_size = 20
                _col1, _col2 = st.columns(2)
                with _col1:
                    population_size = st.slider('Population size',min_value=10,max_value=100,value=50,step=10)
                with _col2:
                    step_size = st.slider('Step size',min_value=10,max_value=100,value=30,step=10)
                timestamp_filtered = {k:v for k,v in timestamps.items() if k in scores_filtered}
                timestamp_filtered = dict(sorted(timestamp_filtered.items(), key=lambda x: x[1]))
                generations = []
                score_means = []
                score_stds = []
                score_mins = []
                score_maxs = []
                center=0.5
                for i in range(0,len(timestamp_filtered),step_size):
                    _designs = list(timestamp_filtered.keys())[i:i+population_size]
                    if len(_designs)<population_size*0.7:
                        continue
                    generations.append(int(i/step_size))
                    score_means.append(np.mean([scores_filtered[d] for d in _designs])-center)
                    score_stds.append(np.std([scores_filtered[d] for d in _designs]))
                    score_mins.append(np.min([scores_filtered[d] for d in _designs])-center)
                    score_maxs.append(np.max([scores_filtered[d] for d in _designs])-center)
                chart_data = pd.DataFrame({
                    'generation':generations,
                    'mean':score_means,
                    'std':score_stds,
                    'max':score_maxs,
                    'min':score_mins
                })
                
                # Calculate upper and lower bounds for the shaded area
                chart_data['std_upper'] = chart_data['mean'] + chart_data['std']
                chart_data['std_lower'] = chart_data['mean'] - chart_data['std']

                # Create the line and shaded area chart with Altair
                line = alt.Chart(chart_data).mark_line(color='blue').encode(
                    x='generation', y='mean'
                )
                # Line for max
                line_max = alt.Chart(chart_data).mark_line(color='red', strokeDash=[4,4]).encode(
                    x='generation', y='max'
                )
                # Line for min
                line_min = alt.Chart(chart_data).mark_line(color='green', strokeDash=[4,4]).encode(
                    x='generation', y='min'
                )
                # Shaded region for standard deviation
                band = alt.Chart(chart_data).mark_area(opacity=0.2).encode(
                    x='generation',
                    y='std_lower',
                    y2='std_upper'
                    # y='min',
                    # y2='max'
                )

                # Combine the line and the shaded area
                chart = band + line + line_min + line_max

                # Display in Streamlit
                st.altair_chart(chart.interactive(), use_container_width=True)
                # st.area_chart(chart_data,x='generation',y=['score_mean','score_std'])
            
    with st.expander(f"Agent Analysis for ```{evosys.evoname}```",expanded=True):#,icon='📊'):

        if len(design_nodes)+len(implemented_nodes)+len(sessions) == 0:
            st.info('No agent analysis available at the moment.')
        else:
            model_abbr = {
                'claude3.5_sonnet':'C35',
                'gpt4o_0806':'G4O',
                'o1_mini':'O1M',
                'o1_preview':'O1P',
                'gpt-4.1-nano':'G41N',
            }
            impl_agents = {i:model_abbr[v['_agent_types']['IMPLEMENTATION_CODER']] for i,v in cfg_implementations.items()}
            proposal_agents = {i:model_abbr[v['_agent_types']['DESIGN_PROPOSER']] for i,v in cfg_proposals.items()}
            reviewer_agents = {i:model_abbr[v['_agent_types']['PROPOSAL_REVIEWER']] for i,v in cfg_proposals.items()}
            obs_agents = {i:model_abbr[v['_agent_types']['IMPLEMENTATION_OBSERVER']] for i,v in cfg_proposals.items()}
            planner_agents = {i:model_abbr[v['_agent_types']['IMPLEMENTATION_PLANNER']] for i,v in cfg_proposals.items()}
            impl_agent_scores = {}
            impl_agent_costs = {}
            impl_agent_effective = {}
            proposal_agent_scores = {}
            proposal_agent_costs = {}
            proposal_agent_effective = {}
            pair_cost = {}
            pair_score = {}
            pair_effective = {}
            double_score = {}
            double_effective = {}
            double_cost = {}
            trio_score = {}
            trio_effective = {}
            trio_valid = {}
            trio_cost = {}
            for design in impl_costs:
                agent = impl_agents[design]
                pagent = proposal_agents[design]
                ragent = reviewer_agents[design]
                oagent = obs_agents[design]
                plagent = planner_agents[design]
                score = avg_score[design]
                cost = impl_costs[design]
                _verified = verified[design]
                pcost = proposal_costs[design]
                if not 0<score<1:
                    continue
                if agent not in impl_agent_scores:
                    impl_agent_scores[agent] = []
                impl_agent_scores[agent].append(score)
                if agent not in impl_agent_effective:
                    impl_agent_effective[agent] = []
                impl_agent_effective[agent].append(100*score/cost)
                if pagent not in proposal_agent_scores:
                    proposal_agent_scores[pagent] = []
                proposal_agent_scores[pagent].append(score)
                if pagent not in proposal_agent_effective:
                    proposal_agent_effective[pagent] = []
                proposal_agent_effective[pagent].append(100*score/pcost)
                if agent not in impl_agent_costs:
                    impl_agent_costs[agent] = []
                impl_agent_costs[agent].append(cost)
                if pagent not in proposal_agent_costs:
                    proposal_agent_costs[pagent] = []
                proposal_agent_costs[pagent].append(pcost)
                pair = f'{agent}-{pagent}'
                double = f'{pagent}-{ragent}'
                trio = f'{plagent}-{agent}-{oagent}'
                if pair not in pair_score:
                    pair_score[pair] = []
                pair_score[pair].append(score)
                if double not in double_score:
                    double_score[double] = []
                double_score[double].append(score)
                if double not in double_effective:
                    double_effective[double] = []
                double_effective[double].append(100*score/(cost+pcost))
                if double not in double_cost:
                    double_cost[double] = []
                double_cost[double].append(cost+pcost)
                if pair not in pair_cost:
                    pair_cost[pair] = []
                pair_cost[pair].append(cost+pcost)
                if pair not in pair_effective:
                    pair_effective[pair] = []
                pair_effective[pair].append(100*score/(cost+pcost))
                if trio not in trio_score:
                    trio_score[trio] = []
                trio_score[trio].append(score)
                if trio not in trio_effective:
                    trio_effective[trio] = []
                trio_effective[trio].append(100*score/(cost))
                if trio not in trio_cost:
                    trio_cost[trio] = []
                trio_cost[trio].append(cost)
                if trio not in trio_valid:
                    trio_valid[trio] = []
                trio_valid[trio].append(_verified)
            
            trio_valid = {k:np.sum(v)/len(v) for k,v in trio_valid.items()}
            for k,v in trio_effective.items():
                trio_effective[k] = [i/trio_valid[k] for i in v]

            def draw_chart_bar(scores,x_label='agent',y_label='mean',center=0):
                # scores_mean = {f'{k} ({len(v)})':np.mean(v) for k,v in scores.items()}
                # scores_std = {f'{k} ({len(v)})':np.std(v) for k,v in scores.items()}
                scores_mean = {f'{k}':np.mean(v) for k,v in scores.items()}
                scores_std = {f'{k}':np.std(v) for k,v in scores.items()}

                # Create DataFrame with both mean and std
                if center!=0:
                    y_label = f'{y_label} (centered)'
                    scores_mean = {k:v-center for k,v in scores_mean.items()}
                chart_data = pd.DataFrame({
                    x_label: list(scores_mean.keys()),
                    y_label: list(scores_mean.values()),
                    'std': list(scores_std.values())
                })
                # Calculate error bar endpoints
                chart_data['upper'] = chart_data[y_label] + chart_data['std']
                chart_data['lower'] = chart_data[y_label] - chart_data['std']
                # Create Altair chart with error bars
                bars = alt.Chart(chart_data).mark_bar().encode(
                    x=x_label,
                    y=y_label,
                    tooltip=[x_label, y_label, 'std']
                )
                # Add error bars
                error_bars = alt.Chart(chart_data).mark_errorbar().encode(
                    x=x_label,
                    y=y_label,
                    # y2='lower',
                    # y1='upper'
                )
                # Combine bars and error bars
                chart = (bars + error_bars).properties(
                    width=600,
                    # height=250
                )
                st.altair_chart(chart, use_container_width=True)

            fitness_center = 0.58 #min(avg_score.values())

            # COL1, COL2 = st.columns([1,1])
            # with COL1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader('Prop. agent-fitness')
                draw_chart_bar(proposal_agent_scores,x_label='Proposal agent',y_label='Fitness',center=fitness_center)
            with col2:
                st.subheader('Prop. agent-cost')
                draw_chart_bar(proposal_agent_costs,x_label='Proposal agent',y_label='Proposal cost')
            with col3:
                st.subheader('Prop. cost-effect')
                draw_chart_bar(proposal_agent_effective,x_label='Proposal agent',y_label='Cost effectiveness')

            # with COL2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader('Impl. agent-fitness')
                draw_chart_bar(impl_agent_scores,x_label='Impl agent',y_label='Fitness',center=fitness_center)
            with col2:
                st.subheader('Impl. agent-cost')
                draw_chart_bar(impl_agent_costs,x_label='Impl agent',y_label='Implementation cost')
            with col3:
                st.subheader('Impl. agent-cost-effect')
                draw_chart_bar(impl_agent_effective,x_label='Impl agent',y_label='Cost effectiveness')


            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader('Agent-pair-fitness')
                draw_chart_bar(pair_score,x_label='Proposal-Impl agent combination',y_label='Fitness',center=fitness_center)

            with col2:
                st.subheader('Agent-pair-cost')
                draw_chart_bar(pair_cost,x_label='Proposal-Impl agent combination',y_label='Cost')
            with col3:
                st.subheader('Cost-effectiveness')
                draw_chart_bar(pair_effective,x_label='Proposal-Impl agent combination',y_label='Cost effectiveness')
            

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader('Proposal-Reviewer fitness')
                draw_chart_bar(double_score,x_label='Proposal-Review agent combination',y_label='Fitness',center=fitness_center)

            with col2:
                st.subheader('Proposal-Reviewer cost')
                draw_chart_bar(double_cost,x_label='Proposal-Review agent combination',y_label='Cost')
            
            with col3:
                st.subheader('Proposal-Reviewer CE')
                draw_chart_bar(double_effective,x_label='Proposal-Review agent combination',y_label='Cost effectiveness')

            st.subheader('Agent-trio-fitness')
            draw_chart_bar(trio_score,x_label='Planner-Impl-Observer agent trio',y_label='Fitness',center=fitness_center)

            st.subheader('Agent-trio-cost')
            draw_chart_bar(trio_cost,x_label='Planner-Impl-Observer agent trio',y_label='Cost')

            # st.subheader('Agent-trio-validity')
            # chart_data = pd.DataFrame(list(trio_valid.items()),columns=['trio','validity'])
            # st.bar_chart(chart_data,x='trio',y='validity')

            st.subheader('Agent-trio-cost-effectiveness')
            draw_chart_bar(trio_effective,x_label='Planner-Impl-Observer agent combination',y_label='Cost effectiveness')


    # st.subheader('Verification Analysis')

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.subheader('Score ranking')
    #     scores_filtered = {k:v for k,v in avg_score.items() if k in scores_filtered}
    #     sorted_socres = sorted(scores_filtered.items(),key=lambda x: x[1],reverse=True)
    #     chart_data = pd.DataFrame(sorted_socres,columns=['design','score'])
    #     st.dataframe(chart_data,use_container_width=True)

    # with col2:
    #     st.subheader('Benchmark contribution')





def _stat_func_check(func_check):
    errors=[]
    if 'The code is not parsable' in func_check:
        errors.append('syntax: unparsable')
    if 'The code is not executable' in func_check:
        errors.append('syntax: unexecutable')
    if 'The class "GAB" is not defined in the provided code' in func_check:
        errors.append('format: GAB_not_defined')
    if 'The method "_forward" is not defined in the class "GAB" or "GAB" is not defined.' in func_check:
        errors.append('format: _forward_not_defined')
    if 'The method "__init__" is not defined in the class "GAB" or "GAB" is not defined.' in func_check:
        errors.append('format: __init___not_defined')
    if 'The dictionary "gab_config" is not defined.' in func_check:
        errors.append('format: gab_config_not_defined')
    if '"gab_config" should be a dictionary.' in func_check:
        errors.append('format: gab_config_not_dict')
    if 'The class "GAB" is not defined in the provided code, cannot validate "gab_config".' in func_check:
        errors.append('format: GAB_not_defined')
    if 'The dictionary "gab_config" is missing the following required arguments: ' in func_check:
        errors.append('format: gab_config_missing_args')

    if 'An error occurred while executing the unit test' in func_check:
        errors.append('unit_test')
    if 'Error: Model initialization failed' in func_check:
        if 'An exception occurred during the forward pass' in func_check:
            if 'RuntimeError: shape' in func_check or 'shapes cannot be' in func_check or 'does not match the number of dimensions' in func_check or 'input tensor must have' in func_check\
                or 'the number of sizes provided' in func_check or 'got shape' in func_check or 'must be a sequence of shape' in func_check or 'must be a 3D tensor' in func_check\
                or 'must match the size of tensor' in func_check or 'IndexError' in func_check or 'Sizes of tensors must match' in func_check:
                errors.append('model_init: shape_error')
            elif 'type must be' in func_check or 'expected scalar type' in func_check or 'must have the same dtype' in func_check:
                errors.append('model_init: dtype_error')
            else:
                errors.append('model_init: forward_error')
        else:
            errors.append('model_init: other')

    if 'Error: Forward pass failed with error' in func_check:
        errors.append('forward_pass')
    if 'Error: The output shape of GAB should be the same as the input.' in func_check:
        errors.append('output_shape')

    if 'Error: Causality test failed' in func_check:
        errors.append('causality')
    if 'Differentiability test failed' in func_check:
        errors.append('differentiability')

    if 'The model is not training correctly. The loss is not decreasing.' in func_check:
        errors.append('diverging: loss_not_decreasing')
    if 'The model is diverging. The loss is' in func_check:
        errors.append('diverging: loss')
    if 'The model is diverging. The gradient norm is NaN.' in func_check:
        errors.append('diverging: grad_norm')
    if 'The model is not efficient. The training time is overly long.' in func_check:
        errors.append('inefficient: training_time')
    if 'The model is not efficient. The memory usage is overly high.' in func_check:
        errors.append('inefficient: memory_usage')
    if 'The model is not efficient. The FLOPs is overly high.' in func_check:
        errors.append('inefficient: flops')
    if 'The model is not effective. The training loss is overly high.' in func_check:
        errors.append('ineffective: training_loss')

    if 'Format check failed with fetal errors' in func_check:
        errors.append('format: fetal_errors')

    return errors

def _stat_format_check(format_check):
    errors=[]
    for line in format_check:
        if 'Fetal Error: parsing the code' in line or 'Fetal Error: Failed to parse the reformatted code' in line:
            errors.append('syntax: unparsable')
        if 'Fetal Error: No GAUBase class found.' in line or 'Fetal Error: Cannot detect the GAU class.' in line:
            errors.append('gau: GAUBase_not_found')
        if 'Fetal Error: Multiple GAUBase classes found' in line or 'Error: Multiple classes inheriting from GAUBase found' in line:
            errors.append('gau: multiple_gau')

        if 'Fetal Error when trying to execute the code by exec(reformatted_code)' in line:
            errors.append('syntax: unexecutable')
        
        if 'Fetal Error: Failed to parse CHILDREN_DECLARATIONS' in line:
            errors.append('gau: CHILDREN_DECLARATIONS_not_parseable')
        if 'are declared as children but never used' in line:
            errors.append('gau: unused_children')

        if 'Fetal Error: No __init__ method found in the GAU.' in line:
            errors.append('gau: __init___not_found')
        if 'Fetal Error: No _forward method found in the GAU.' in line:
            errors.append('gau: _forward_not_found')


        if 'a GAU can only inherit from GAUBase directly, not multiple classes' in line:
            errors.append('gau: GAUBase_inheritance')
        if 'you are not allowed to define any nn.Module, whenever you need a submodule, declare a child GAU instead.' in line:
            errors.append('gau: nn.Module_not_allowed')
        if 'Error: The forward method in GAUBase should not be overridden.' in line:
            errors.append('gau: forward_override')
        if 'Error: nn.Sequential is not supported in GAU. You may use ModuleList, ModuleDict.' in line:
            errors.append('gau: nn.Sequential_not_supported')
        if 'Error: in-place assignment to a gau instance is never encouraged' in line:
            errors.append('gau: inplace_assignment')
        if 'Error: GAU call must have the sequence as the first argument and the **Z.' in line:
            errors.append('gau: gau_call_args')
        if 'Error: GAU call always returns a tuple of two variables, the first is a sequence and the second must be the updated Z. If you need to return other variables, you can include them in Z.' in line:
            errors.append('gau: gau_call_return')

    return errors

def impl_analysis(evosys,design_nodes,implemented_nodes):
    st.subheader('Implementation Analysis')
    rounds=[]
    for node in design_nodes:
        impl = node.implementation
        if impl is None:
            continue
        for attempt in impl.history:
            rounds+=attempt.rounds
    with st.expander(f'**Analysis over ```{len(rounds)}``` rounds**',expanded=True):
        failed_format_checks=[]
        failed_func_checks=[]
        total_format_checks=0
        total_func_checks=0
        for round in rounds:
            if 'unit_design_traces' in round:
                for r in round['unit_design_traces']:
                    func_check = r['func_checks']
                    format_check = r['format_checks']
                    for unit in format_check:
                        _check=format_check[unit]['format_errors']
                        total_format_checks+=1
                        if len(_check)>0:
                            failed_format_checks.append(_check)
                    total_func_checks+=1
                    if not func_check['checkpass']:
                        failed_func_checks.append(func_check['check_report'])
            
        func_errors = {}
        gau_errors = {}
        for func_check in failed_func_checks:
            for error in _stat_func_check(func_check):
                if error not in func_errors:
                    func_errors[error] = 0
                func_errors[error]+=1
        for format_check in failed_format_checks:
            for error in _stat_format_check(format_check):
                if error not in gau_errors:
                    gau_errors[error] = 0
                gau_errors[error]+=1
        n_failed_func = len(failed_func_checks)
        n_failed_format = len(failed_format_checks)
        if n_failed_func==0 or n_failed_format==0:
            SAVE_DIR = U.pjoin(evosys.ckpt_dir,evosys.evoname)
            errors = U.load_json(U.pjoin(SAVE_DIR,f'impl_errors_{evosys.evoname}.json'))
            func_errors = errors.get('func',{})
            gau_errors = errors.get('gau',{})
            n_failed_func = errors.get('n_failed_func',0)
            n_failed_format = errors.get('n_failed_format',0)
            total_func_checks = errors['total_func']
            total_format_checks = errors['total_format']

        col1, col2 = st.columns(2)
        with col1:
            st.write(f'**Failed Func Checks: ```{100*n_failed_func/total_func_checks:.1f}%```**')
            # st.write(func_errors.keys())
            rename_map = {
                "diverging: loss_not_decreasing":"Loss not decreasing",
                "format: fetal_errors": "Fetal format errors",
                'diverging: loss': 'Loss diverging',
                'diverging: grad_norm': 'Gradnorm diverged',
                'inefficient: training_time': 'Training too slow',
                'inefficient: flops': 'FLOPs too high',
                'model_init: forward_error': 'Forward pass error',
                'model_init: shape_error': 'Shape error',
                'model_init: dtype_error': 'Dtype error',
                'model_init: other': 'Model init error',
                # 'forward_pass': 'Forward pass error',
                'causality': 'Not causal',
                'differentiability': 'Not differentiable',
                'syntax: unparsable': 'Unparsable',
                'syntax: unexecutable': 'Unexecutable',
            }
            _func_errors = {}
            for i in rename_map:
                if i in func_errors:
                    _func_errors[rename_map[i]] = func_errors[i]
            func_errors = _func_errors
            # func_errors = {rename_map[k] if k in rename_map else k.capitalize().replace('_',' '):func_errors[k] for k in func_errors}
            st.bar_chart(pd.DataFrame(list(func_errors.items()),columns=['Functional error type','Frequency']),x='Functional error type',y='Frequency')
        with col2:
            st.write(f'**Failed Format Checks: ```{100*n_failed_format/total_format_checks:.1f}%```**')
            rename_map = {
                "gau: unused_children":"Unused children",
                "gau: multiple_gau":"Multiple GAU found",
                "gau: forward_override":"Forward override",
                "gau: nn.Module_not_allowed":"Found nn.Module",
                "syntax: unexecutable":"Unexecutable",
                "gau: gau_call_args":"Wrong GAU args",
                "gau: _forward_not_found":"_forward not found",
                "gau: gau_call_return":"Wrong GAU return",
                "gau: __init___not_found":"__init__ not found",
                # "gau: inplace_assignment":"GAU overwrite",
            }
            _gau_errors = {}
            for i in rename_map:
                if i in gau_errors:
                    _gau_errors[rename_map[i]] = gau_errors[i]
            gau_errors = _gau_errors
            st.bar_chart(pd.DataFrame(list(gau_errors.items()),columns=['Format error type','Frequency']),x='Format error type',y='Frequency')
        errors = {'func':func_errors,'gau':gau_errors,'total_func':total_func_checks,'total_format':total_format_checks,
                  'n_failed_func':n_failed_func,'n_failed_format':n_failed_format,'n_rounds':len(rounds)}
        st.download_button(label='Download Impl. Error Stats',data=json.dumps(errors),file_name=f'impl_errors_{evosys.evoname}.json')


def unit_analyzer(evosys,design_nodes,implemented_nodes):
    if evosys.benchmark_mode:
        return
    st.subheader("Unit Analyzer")
    with st.expander(f"Unit Analysis for ```{evosys.evoname}```",expanded=True):#,icon='📊'):
        st.subheader('Block-Unit Tree')

        if 'utree_max_nodes' not in st.session_state:
            st.session_state.utree_max_nodes=50

        _bg_color=AU.theme_aware_options(st,"#fafafa","#f0f0f0","#fafafa")
        
        col1, col2, col3, col4, col5 = st.columns([5,0.1,1.5,0.8,0.8])
        with col1:
            _max_nodes=st.slider('Max Units to Display',min_value=0,max_value=len(evosys.ptree.GD.terms),value=st.session_state.utree_max_nodes)

        with col4:
            st.write('')
            st.write('')
            _no_root = st.checkbox('No Root',value=False)

        with col5:
            st.write('')
            st.write('')
            _no_units = st.checkbox('No Units',value=True)

        export_height = 750
        _bg_color = '#ffffff'
        physics=True
        
        evosys.ptree.GD.export(max_nodes=st.session_state.utree_max_nodes,height=f'{export_height}px',bgcolor=_bg_color,
                               no_root=_no_root,no_units=_no_units,physics=physics)
        utree_dir_small=U.pjoin(evosys.evo_dir,f'UTree_{st.session_state.utree_max_nodes}.html')

        # check this: https://github.com/napoles-uach/streamlit_network 
        with col3:
            st.write('')
            st.write('')
            if st.button(f'Refresh & Sync Tree'):#,use_container_width=True):
                evosys.ptree.GD.export(max_nodes=_max_nodes,height=f'{export_height}px',
                                    bgcolor=_bg_color,no_root=_no_root,no_units=_no_units,physics=physics)
                utree_dir_small=U.pjoin(evosys.evo_dir,f'UTree_{_max_nodes}.html')
                st.session_state.utree_max_nodes=_max_nodes
                
        st.write(f'**First {st.session_state.utree_max_nodes} nodes under the namespace ```{evosys.evoname}```** Legend: :red[Unit Nodes/Reuse Edges] | :blue[Design Nodes/Variant Edges] | :orange[Root Units]')

        HtmlFile = open(utree_dir_small, 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height = export_height)


        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Naive Bayes Analysis of Unit Bag-of-Words')

            avg_score = {}
            bows = {}
            word_freq = {}
            for node in implemented_nodes:
                _score = np.mean(list(node.get_scores().values())) #node.get_score(scale='14M')
                design = node.acronym
                if _score>0:
                    avg_score[design] = _score_stratify_single(_score)
                    bows[design] = node.get_bow()
                    for bow in bows[design]:
                        if bow not in word_freq:
                            word_freq[bow] = 0
                        word_freq[bow] += _stratify_to_weight(avg_score[design])

            data = [(bows[design],avg_score[design]) for design in bows]
            bow_texts = [' '.join(bow) for bow, _ in data]  # Join words in each BOW
            labels = [label for _, label in data]

            # Transform BOWs into features
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(bow_texts)
            y = labels

            # Train Naive Bayes classifier
            nb = MultinomialNB()
            nb.fit(X, y)

            words = st.multiselect('Select words',vectorizer.get_feature_names_out())
            X_words = vectorizer.transform([' '.join(words)])
            prediction = nb.predict(X_words)[0]
            probabilities = nb.predict_proba(X_words)[0]
            _preds = pd.DataFrame({
                'Category': nb.classes_,
                'Probability': probabilities
            })
            # .sort_values('Probability', ascending=False)
            weight = sum(probabilities[i] * _stratify_to_mean(nb.classes_[i]) for i in range(len(nb.classes_)))
            st.write(f'**Weighted Score: ```{weight:.4f}```**')
            st.bar_chart(_preds,x='Category',y='Probability')

        with col2:
            st.subheader('Fitness-weighted Wordcloud of Units')
            exclude_words = ['RMSNorm','Conv','RotaryPositionalEmbeddings','RotaryEmbedding','SwiGluMLP','GatedMLP']
            for word in exclude_words:
                if word in word_freq:
                    del word_freq[word]
            SIZE=1
            wordcloud = WordCloud(width=600*SIZE, height=400*SIZE, background_color='white').generate_from_frequencies(word_freq)
            st.image(wordcloud.to_image())

        col1, col2, col3,col4 = st.columns([1,1,1,1])
        with col1:
            st.subheader('Num of Units')
            if st.session_state.use_cache:
                designs = {design:evosys.ptree.get_node(design) for design in st.session_state.filter_by_types['DesignArtifactImplemented']}
            else:
                designs = {design:evosys.ptree.get_node(design) for design in evosys.ptree.filter_by_type(['DesignArtifactImplemented'])}
            n_units = {}
            reused_units = {}
            reused_units_ratio = {}
            for design,node in designs.items():
                impl = node.implementation.implementation
                n_units[design] = len(impl.descendants(impl.root))+1
                units = list(impl.descendants(impl.root)) + [impl.root]
                num_reuses = 0
                for unit in units:
                    _unit = impl.units[unit]
                    if _unit.reuse_from:
                        num_reuses += 1
                reused_units[design] = num_reuses
                reused_units_ratio[design] = round(num_reuses/len(units),2)
            mean_units = np.mean(list(n_units.values()))
            std_units = np.std(list(n_units.values()))
            median_units = np.median(list(n_units.values()))
            min_units = np.min(list(n_units.values()))
            max_units = np.max(list(n_units.values()))
            st.caption(f'**Mean: ```{mean_units:.2f}``` | Std: ```{std_units:.2f}``` | Median: ```{median_units:.2f}``` | Min: ```{min_units:.2f}``` | Max: ```{max_units:.2f}```**')
            unit_freq = _data_to_freq(n_units)
            x_name = 'Number of Units'
            st.bar_chart(pd.DataFrame(list(unit_freq.items()),columns=[x_name,'Frequency']),x=x_name,y='Frequency')

    with col2:
        st.subheader('Unit Reuse')
        GDT = evosys.ptree.GD.terms
        reuses = {}
        reuse_designs = {}
        for unit in GDT:
            term = GDT[unit]
            if term.is_root: continue
            variants = term.variants
            reuse_designs[unit] = []
            for variant in variants:
                if variant not in reuse_designs[unit]:
                    reuse_designs[unit].append(variant)
                node = variants[variant][0]
                if node.reuse_from:  # seems wrong?
                    _design,reuse_from = node.reuse_from.split('.')
                    if reuse_from not in GDT:
                        print(f'{unit} -> {reuse_from} not found')
                        continue
                    if reuse_from not in reuses:
                        reuses[reuse_from] = []
                    if unit not in reuses[reuse_from]:
                        reuses[reuse_from].append(unit)

        n_reuses = {unit:len(reuses[unit]) for unit in reuses}
        n_reuse_designs = {unit:len(reuse_designs[unit]) for unit in reuse_designs}

        n_reuse_freq = _data_to_freq(n_reuses)
        max_reuse = max(n_reuses.values())
        mean_reuse = np.mean(list(n_reuses.values()))
        median_reuse = np.median(list(n_reuses.values()))
        min_reuse = np.min(list(n_reuses.values()))
        std_reuse = np.std(list(n_reuses.values()))
        st.caption(f'**Max: ```{max_reuse:.2f}``` | Mean: ```{mean_reuse:.2f}``` | Median: ```{median_reuse:.2f}``` | Min: ```{min_reuse:.2f}``` | Std: ```{std_reuse:.2f}```**')
        x_name = 'Reuses'
        st.bar_chart(pd.DataFrame(list(n_reuse_freq.items()),columns=[x_name,'Frequency']),x=x_name,y='Frequency')

    with col3:
        st.subheader('Unit Usage')
        n_reuse_designs_freq = _data_to_freq(n_reuse_designs)
        max_reuse_designs = max(n_reuse_designs.values())
        mean_reuse_designs = np.mean(list(n_reuse_designs.values()))
        median_reuse_designs = np.median(list(n_reuse_designs.values()))
        min_reuse_designs = np.min(list(n_reuse_designs.values()))
        std_reuse_designs = np.std(list(n_reuse_designs.values()))
        st.caption(f'**Max: ```{max_reuse_designs:.2f}``` | Mean: ```{mean_reuse_designs:.2f}``` | Median: ```{median_reuse_designs:.2f}``` | Min: ```{min_reuse_designs:.2f}``` | Std: ```{std_reuse_designs:.2f}```**')
        x_name = 'Usage'
        st.bar_chart(pd.DataFrame(list(n_reuse_designs_freq.items()),columns=[x_name,'Frequency']),x=x_name,y='Frequency')

    with col4:
        st.subheader('Num of Reuse')
        reused_units = reused_units_ratio
        reused_units_freq = _data_to_freq(reused_units)
        max_reused_units = max(reused_units.values())
        mean_reused_units = np.mean(list(reused_units.values()))
        median_reused_units = np.median(list(reused_units.values()))
        min_reused_units = np.min(list(reused_units.values()))
        std_reused_units = np.std(list(reused_units.values()))
        st.caption(f'**Max: ```{max_reused_units:.2f}``` | Mean: ```{mean_reused_units:.2f}``` | Median: ```{median_reused_units:.2f}``` | Min: ```{min_reused_units:.2f}``` | Std: ```{std_reused_units:.2f}```**')
        x_name = 'Ratio of Reused Units'
        st.bar_chart(pd.DataFrame(list(reused_units_freq.items()),columns=[x_name,'Frequency']),x=x_name,y='Frequency')


def network_analysis(evosys,design_nodes,implemented_nodes):

    st.subheader('Network Analysis')

    with st.expander(f'Network Analysis for ```{evosys.evoname}```',expanded=True):
        root_degree_dist = {7: 0.04040404040404041, 10: 0.03367003367003367, 2: 0.1750841750841751, 3: 0.15151515151515152, 0: 0.06397306397306397, 1: 0.18518518518518517, 8: 0.03367003367003367, 6: 0.06734006734006734, 9: 0.02356902356902357, 4: 0.12121212121212122, 12: 0.006734006734006734, 5: 0.06397306397306397, 11: 0.003367003367003367, 18: 0.003367003367003367, 22: 0.003367003367003367, 15: 0.006734006734006734, 14: 0.006734006734006734, 21: 0.003367003367003367, 17: 0.003367003367003367, 13: 0.003367003367003367, 23: 0}
        full_degree_dist = {17: 0.0020632737276478678, 1: 0.6953232462173315, 2: 0.21664374140302614, 0: 0.052269601100412656, 21: 0.0020632737276478678, 13: 0.001375515818431912, 129: 0.000687757909215956, 86: 0.000687757909215956, 9: 0.0020632737276478678, 4: 0.0041265474552957355, 3: 0.0034387895460797797, 57: 0.000687757909215956, 24: 0.001375515818431912, 30: 0.000687757909215956, 19: 0.001375515818431912, 56: 0.001375515818431912, 40: 0.000687757909215956, 25: 0.001375515818431912, 59: 0.000687757909215956, 5: 0.000687757909215956, 38: 0.000687757909215956, 7: 0.001375515818431912, 103: 0.000687757909215956, 78: 0.000687757909215956, 39: 0.000687757909215956, 23: 0.001375515818431912, 317: 0.000687757909215956, 12: 0.001375515818431912, 75: 0.000687757909215956, 70: 0.000687757909215956, 15: 0.000687757909215956, 20: 0.000687757909215956, 318: 0}
        rand_degree_dist = {1: 0.4634433962264151, 5: 0.04363207547169811, 3: 0.10849056603773585, 2: 0.22287735849056603, 6: 0.02240566037735849, 4: 0.07075471698113207, 13: 0.0011792452830188679, 11: 0.0011792452830188679, 0: 0.03537735849056604, 7: 0.012971698113207548, 12: 0.0011792452830188679, 9: 0.008254716981132075, 10: 0.0047169811320754715, 8: 0.003537735849056604, 14: 0}
                
        def _process_dist(dist, cutoff=9):
            _dist = {}
            for i in dist:
                if i >= cutoff:
                    if cutoff not in _dist:
                        _dist[cutoff] = 0
                    _dist[cutoff] += dist[i]
                else:
                    _dist[i] = dist[i]
            
            # Convert to list of dicts format for Altair
            return [{'Degree': f'{k}+' if k == cutoff else str(k), 
                    'DegreeSort': float('inf') if k == cutoff else k,  # Make cutoff sort to end
                    'Ratio': v} for k, v in _dist.items()]

        # Process the distributions
        library_data = _process_dist(root_degree_dist)
        full_data = _process_dist(full_degree_dist)
        random_data = _process_dist(rand_degree_dist)

        # Add source identifier to each data point
        for d in library_data:
            d['Source'] = 'Library'
        for d in full_data:
            d['Source'] = 'Full'
        for d in random_data:
            d['Source'] = 'Random'

        # Combine all data
        combined_data = pd.DataFrame(library_data + full_data + random_data)

        col, _ = st.columns([2.5,1])
        with col:
            st.bar_chart(combined_data.sort_values('DegreeSort'), x='Degree', y='Ratio', color='Source')

def scaling_analysis(evosys,design_nodes,implemented_nodes):
    if evosys.benchmark_mode:
        return
    st.subheader('Scaling Analysis')
    api = None 
    token_mults = evosys.ptree.token_mults
    with st.expander(f"Scaling Analysis for ```{evosys.evoname}```",expanded=True):
        bows = {}
        scores = {}
        n_params = {}
        losses = {}
        for node in implemented_nodes:
            if len(node.verifications)>0:
                scores[node.acronym] = node.get_scores()
                bows[node.acronym] = node.get_bow()
                n_params[node.acronym] = {}
                losses[node.acronym] = {}
                for scale in node.verifications:
                    report = node.verifications[scale].verification_report
                    wandb_ids = report['wandb_ids.json']
                    trainer_state = report.get('trainer_state.json',{})
                    if not trainer_state or 'loss' not in trainer_state:
                        if st.session_state.is_demo:
                            continue
                        if api is None:
                            api = wandb.Api()
                        run = api.run(f"{wandb_ids['entity']}/{wandb_ids['project']}/{wandb_ids['pretrain']['id']}")
                        metrics = run.summary  # Summary includes the latest metrics logged, e.g., accuracy, loss
                        trainer_state['loss'] = metrics.get('train/loss')
                        node.verifications[scale].verification_report['trainer_state.json'] = trainer_state
                        evosys.ptree.FM.upload_verification(node.acronym,node.verifications[scale].to_dict(),scale,overwrite=True,protect_keys=['wandb_ids.json','eval_results.json'])
                        verification_path=U.pjoin(evosys.evo_dir,'db','designs',node.acronym,'verifications',scale+'.json')
                        U.save_json(node.verifications[scale].to_dict(),verification_path)
                        print(f'Uploaded verification for scale {scale} in design {node.acronym}, loss: {trainer_state["loss"]}')
                    losses[node.acronym][scale] = node.verifications[scale].verification_report['trainer_state.json']['loss']
                    n_params[node.acronym][scale] = report['eval_results.json']['config']['model_num_parameters']

        all_params = []
        all_tokens = []
        all_scores = []
        all_losses = []
        all_scales = []
        design_losses = {}
        design_params = {}
        design_scales = {}
        design_scores = {}
        for design in losses:
            design_losses[design] = []
            design_params[design] = []
            design_scales[design] = []
            design_scores[design] = []
            for scale in losses[design]:
                _loss = losses[design][scale]
                _score = scores[design][scale]
                if _loss == 0 or _score == 0:
                    continue
                _params = n_params[design][scale]
                _tokens = token_mults[scale]*_params
                all_tokens.append(_tokens)
                all_params.append(_params)
                all_scores.append(_score)
                all_losses.append(_loss)
                all_scales.append(int(scale.replace('M','')))
                design_losses[design].append(_loss)
                design_params[design].append(_params)
                design_scales[design].append(int(scale.replace('M','')))
                design_scores[design].append(_score)

        st.subheader('Training Tokens vs Loss')
        tokens_loss = pd.DataFrame({
            'Tokens':all_tokens,
            'Loss':all_losses,
            'Scale':all_scales,
            'Score':all_scores,
        })
        st.scatter_chart(tokens_loss,x='Tokens',y='Loss',color='Score',size='Scale')

        col1, col2 = st.columns([4,1])
        with col1:
            st.subheader('Design Params vs Loss')
            combined_data = []
            for design in design_params:
                for i in range(len(design_params[design])):
                    combined_data.append({
                        'Params': design_params[design][i],
                        'Loss': design_losses[design][i],
                        'Scale': design_scales[design][i],
                        'Design': design
                    })
                
            combined_df = pd.DataFrame(combined_data)

            # Create line chart using Altair
            chart = alt.Chart(combined_df).mark_line(point=True,opacity=0.1).encode(
                x=alt.X('Params', scale=alt.Scale(type='log')),
                y='Loss',
                color='Design',
                tooltip=['Design', 'Scale', 'Params', 'Loss']
            ).interactive()
        
            st.altair_chart(chart, use_container_width=True)


        
        st.subheader('Design Params vs Score')
        combined_data = []
        for design in design_params:
            for i in range(len(design_params[design])):
                combined_data.append({
                    'Params': design_params[design][i],
                    'Score': design_scores[design][i],
                    'Scale': design_scales[design][i],
                    'Design': design
                })
        
        combined_df = pd.DataFrame(combined_data)
        
        # Create line chart using Altair
        chart = alt.Chart(combined_df).mark_line(point=True,opacity=0.1).encode(
            x=alt.X('Params', scale=alt.Scale(type='log')),
            y='Score',
            color='Design',
            tooltip=['Design', 'Scale', 'Params', 'Score']
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)



def _stats(evosys):
    st.title("Evolution Statistics")
    if st.session_state.use_cache:
        design_nodes = [evosys.ptree.get_node(i) for i in st.session_state.filter_by_types['DesignArtifact']]
        implemented_nodes = [evosys.ptree.get_node(i) for i in st.session_state.filter_by_types['DesignArtifactImplemented']]
    else:
        design_nodes = [evosys.ptree.get_node(i) for i in evosys.ptree.filter_by_type('DesignArtifact')]
        implemented_nodes = [evosys.ptree.get_node(i) for i in evosys.ptree.filter_by_type('DesignArtifactImplemented')]
    session_stats(evosys,design_nodes,implemented_nodes)
    impl_analysis(evosys,design_nodes,implemented_nodes)
    unit_analyzer(evosys,design_nodes,implemented_nodes)
    scaling_analysis(evosys,design_nodes,implemented_nodes)
    network_analysis(evosys,design_nodes,implemented_nodes)

class EvoModes(Enum):
    EVOLVE = 'Evolution System'
    BENCH = 'Agent Benchmark'
    STATS = 'Evolution Statistics'
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

        _index = 1 if evosys.benchmark_mode else 0
        if st.session_state.is_demo:
            options = [i.value for i in EvoModes if i not in [EvoModes.EUREKA,EvoModes.BENCH]]
        else:
            options = [i.value for i in EvoModes]
        mode = st.selectbox("Sub-tabs",options=options,index=_index,
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
    
    if not st.session_state.is_demo:    
        assert evosys.remote_db, "You must connect to a remote database to run the evolution."

    if mode in [EvoModes.EVOLVE,EvoModes.BENCH]:
        running_evoname,running_group_id = _is_running(evosys)
        if running_evoname:
            if not st.session_state.evo_running:
                if evosys.CM.group_id == running_group_id or evosys.evoname == running_evoname:
                    st.toast(f'Network group ```{running_group_id}``` is already running for evolution ```{running_evoname}```. Launching a command center in passive mode.')
                else:
                    st.toast(f'Evolution ```{running_evoname}``` is already running in network group ```{running_group_id}```. Launching a command center in passive mode.')
                launch_evo(evosys,0,0,active_mode=False)

    if mode == EvoModes.EVOLVE:
        _evolve(evosys)
    elif mode == EvoModes.BENCH:
        _bench(evosys)
    elif mode == EvoModes.EUREKA:
        _eureka(evosys)
    elif mode == EvoModes.STATS:
        _stats(evosys)



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


