import json
import time
import pathlib
import streamlit as st
import sys,os
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


CC_POLL_FREQ = 30 # seconds
CC_ZOMBIE_THRESHOLD = 60 # seconds
CC_COMMAND_DELAY = 2 # seconds


def _is_running(evosys):
    docs = evosys.remote_db.collection('experiment_connections').get()
    for doc in docs:
        if doc.to_dict().get('status','n/a') == 'connected':
            last_heartbeat = doc.to_dict().get('last_heartbeat')
            threshold_time = datetime.now(pytz.UTC) - timedelta(seconds=CC_ZOMBIE_THRESHOLD)
            is_zombie = last_heartbeat < threshold_time
            evoname = doc.to_dict().get('evoname')
            group_id = doc.to_dict().get('group_id')
            if not is_zombie and (group_id==evosys.CM.group_id or evoname==evosys.evoname):
                return evoname, group_id
            else:
                return None, None
    return None, None

class CommandCenter:
    def __init__(self,evosys,max_designs_per_node,max_designs_total,stream):
        self.evosys=evosys
        self.evoname=evosys.evoname
        self.max_designs_per_node=max_designs_per_node
        self.max_designs_total=max_designs_total
        self.st=stream
        self.doc_ref = evosys.remote_db.collection('experiment_connections').document(self.evosys.evoname)
        self.running = False
        self.poll_freq=CC_POLL_FREQ
        self.zombie_threshold = CC_ZOMBIE_THRESHOLD  # seconds

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
                },merge=True)
                self.active_mode = True

    # def cleanup(self):
    #     # st.info("Cleaning up and disconnecting...")
    #     self.doc_ref.delete()  # Delete the connection document

    def run_evolution(self):
        self.evosys.sync_to_db() # sync the config to the db
        self.evosys.CM.start_log()
        self.running = True
        mode = 'active mode' if self.active_mode else 'passive mode'
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Evolution launched for {self.evosys.evoname} in {mode}')
        while self.running:
            if self.active_mode:
                to_sleep = self.poll_freq
                # check if the status is still stopped
                if self.doc_ref.get().to_dict().get('status','n/a') == 'stopped':
                    break
                # self.evosys.CM.get_active_connections() # refresh the connection status
                design_workloads, verify_workloads = self.evosys.CM.get_all_workloads() # will refresh the connection status
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design workloads: {design_workloads}, Verify workloads: {verify_workloads}')

                # check if the design workloads are full
                total_design_workloads = sum(design_workloads.values())
                if total_design_workloads == self.max_designs_total or self.max_designs_total==0:
                    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design workloads are full ({total_design_workloads}/{self.max_designs_total}), skipping design workload assignment')
                else:
                    design_availability = {k:self.evosys.CM.max_design_threads[k] - v for k,v in design_workloads.items()}
                    if sum(design_availability.values())>0:
                        available_design_threads = self.max_designs_total-sum(design_workloads.values())
                        for _ in range(max(0,available_design_threads)):
                            node_id = max(design_availability, key=design_availability.get)
                            if design_availability[node_id]>0:
                                if design_workloads[node_id]<self.max_designs_per_node or self.max_designs_per_node==0:
                                    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Assigning design workload to the most available node {node_id}')
                                    self.evosys.CM.design_command(node_id)
                                    design_availability[node_id] -= 1
                                    design_workloads[node_id] += 1
                                    time.sleep(CC_COMMAND_DELAY)
                                    to_sleep = self.poll_freq
                                    if to_sleep<=0:
                                        break
                            #     else:
                            #         print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design workload for the most available node {node_id} is full ({design_workloads[node_id]}/{self.max_designs_per_node})')
                            # else:
                            #     print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Design availability for the most available node {node_id} is empty ({design_availability[node_id]})')
                    else:
                        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] No design workload available: {design_availability}')

                # assigned_verify_workloads = False
                for node_id in verify_workloads:
                    if verify_workloads[node_id] == 0 and self.evosys.CM.accept_verify_job[node_id]:
                        if self.evosys.CM.verify_command(node_id):
                            # assigned_verify_workloads = True
                            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Assigning verify workload to node {node_id}')
                            time.sleep(CC_COMMAND_DELAY)
                            to_sleep = self.poll_freq
                            if to_sleep<=0:
                                break

                self.doc_ref.update({'last_heartbeat': firestore.SERVER_TIMESTAMP})
            
            if to_sleep>0:
                time.sleep(to_sleep)  
        # self.cleanup()

    def stop_evolution(self):
        self.running = False
        if self.doc_ref.get().exists:
            self.doc_ref.update({'status': 'stopped'})
        # self.cleanup()
        

def x_evolve_passive(command_center,cli=False): # extereme evolution 
    thread = threading.Thread(target=command_center.run_evolution)
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
    process = subprocess.Popen(cmd,shell=True)
    command_center.doc_ref.set({'pid': process.pid},merge=True)
    return process.pid


def launch_evo(evosys,max_designs_per_node,max_designs_total,active_mode=True):
    if len(evosys.CM.get_active_connections())==0 and active_mode:
        st.toast('No nodes connected. Please remember to launch nodes.',icon='🚨')
    command_center = CommandCenter(evosys,max_designs_per_node,max_designs_total,st) # launch a passive command center first
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
    st.toast(f"Evolution launched for ```{evoname}``` in {_mode}.",icon='🚀')
    time.sleep(1)
    st.rerun()

def stop_evo():
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
            st.session_state.command_center.stop_evolution()
            st.session_state.evo_passive_thread.join(timeout=5)
            if st.session_state.evo_passive_thread.is_alive():
                print("Thread did not terminate, forcing exit")
            st.session_state.evo_passive_thread = None
        print('Stopping command center...')
        st.session_state.command_center.stop_evolution()
        st.session_state.command_center = None
        st.session_state.evo_process_pid = None
        st.session_state.evo_running = False
        time.sleep(1)
        st.toast(f"Evolution stopped for ```{evoname}```.",icon='🛑')
        time.sleep(1)
        st.rerun()


def network_status(evosys):
    group_id = evosys.CM.group_id
    st.write(f'#### *Network Group ```{group_id}``` Status*')

    with st.expander('Nodes Running Status',expanded=True):
        nodes = evosys.CM.get_active_connections()
        if len(nodes)==0:
            st.info('No active working nodes connected')
        else:
            st.write(f'##### Connected Nodes ```{len(nodes)}```')
            _nodes = {}
            design_workloads, verify_workloads = evosys.CM.get_workloads()
            for node_id in nodes:
                node_data = evosys.CM.collection.document(node_id).get().to_dict()
                accept_verify_job = node_data['accept_verify_job']
                design_load = design_workloads.get(node_id,[])
                verify_load = verify_workloads.get(node_id,[])
                _nodes[node_id] = {
                    'Design Workload': f'{len(design_load)}/{node_data["max_design_threads"]}',
                    'Verify Workload': f'{len(verify_load)}/1' if accept_verify_job else 'N/A',
                    'Accept Verify Job': accept_verify_job,
                    'Use GPU Checker': not node_data['cpu_only_checker'],
                    'Last Heartbeat': node_data['last_heartbeat'].strftime('%Y-%m-%d %H:%M:%S %Z'),
                    'Status': node_data['status'],
                    'MAC Address': node_data['mac_address'],
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
                active_design_sessions_df['started_at'] = pd.to_datetime(active_design_sessions_df['started_at'],unit='s').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            if 'heartbeat' in active_design_sessions_df.columns:
                active_design_sessions_df['heartbeat'] = pd.to_datetime(active_design_sessions_df['heartbeat'],unit='s').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            st.dataframe(active_design_sessions_df,use_container_width=True)
        else:
            st.info('No active design sessions')

        running_verifications = evosys.CM.get_running_verifications()
        st.write(f'##### Running Verifications ```{len(running_verifications)}```')
        if len(running_verifications)>0:
            # for sess_id in running_verifications:
            running_verifications_df = pd.DataFrame(running_verifications).T
            running_verifications_df.rename(columns={'latest_log':'started_at'},inplace=True)
            if 'pid' in running_verifications_df.columns:
                running_verifications_df['pid'] = running_verifications_df['pid'].astype(str)
            if 'started_at' in running_verifications_df.columns:
                running_verifications_df['started_at'] = pd.to_datetime(running_verifications_df['started_at'],unit='s').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            if 'heartbeat' in running_verifications_df.columns:
                running_verifications_df['heartbeat'] = pd.to_datetime(running_verifications_df['heartbeat'],unit='s').dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            st.dataframe(running_verifications_df,use_container_width=True)
        else:
            st.info('No running verifications')




def system_status(evosys):
    with st.expander(f"System Status for ```{evosys.evoname}```",expanded=False,icon='💻'):
        settings={}
        settings['Experiment Directory']=evosys.evo_dir
        if evosys.design_budget_limit>0:
            text=f'Design Budget Usage: {evosys.ptree.design_cost:.2f}/{evosys.design_budget_limit:.2f}'
            st.progress(evosys.ptree.design_cost/evosys.design_budget_limit,text=text)
        else:
            text=f'Design Budget Usage: {evosys.ptree.design_cost:.2f}/♾️'
            st.progress(1.0,text=text)
        for scale,num in evosys.selector._verify_budget.items():
            remaining = evosys.selector.verify_budget[scale] 
            text=f'Verification Budget Usage ({scale}): {remaining}/{num}'
            st.progress(remaining/num,text=text)
        st.write(f'Budget Type: ```{evosys.params["budget_type"]}```')

def system_config(evosys):
    with st.expander(f"System Config for ```{evosys.evoname}```",expanded=False,icon='🔍'):
        col1,col2=st.columns(2)
        with col1:
            st.write('Design Config',evosys.design_cfg)
            st.write('Search Config',evosys.search_cfg)
        with col2:
            st.write('Select Config',evosys.select_cfg)
            st.write('Verify Engine Config',evosys.ve_cfg)

def evolve(evosys,project_dir):

    st.title("Evolution System")

    if 'command_center' not in st.session_state:
        st.session_state.command_center = None
    if 'evo_process_pid' not in st.session_state:
        st.session_state.evo_process_pid = None
    if 'evo_passive_thread' not in st.session_state:
        st.session_state.evo_passive_thread = None
     
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
    #         # stop_evo()


    passive_mode=False
    if st.session_state.command_center and st.session_state.command_center.running and not st.session_state.command_center.active_mode:
        passive_mode=True
        # _evoname = st.session_state.command_center.evoname
        # st.info(f'The command center is already running for namespace ```{_evoname}```. You are in passive observation mode.')


    col1, col2 = st.columns([2,4])
    with col1:
        system_status(evosys)
    with col2:
        system_config(evosys)

    
    st.header("Launch Pad")
    col1, col2, col3, col4,col5 = st.columns([1,1,1,1,1],gap='small')
    with col1:
        input_max_designs_per_node=st.number_input("Max Design Threads (Per Node)",min_value=0,value=4,disabled=st.session_state.evo_running,
            help='Global control of the maximum number of design threads to run on each node in addition to the local settings on each node. 0 is unlimited.'
        )
    with col2:
        input_max_designs_total=st.number_input("Max Design Threads (Total)",min_value=0,value=10,disabled=st.session_state.evo_running,
            help='The maximum number of total design threads run across all nodes at the same time. 0 is unlimited (which means only bound by the per-node settings).'
        )
    with col3:
        # always use extreme mode, use as much gpus as possible
        verify_schedule=st.selectbox("Verification Scheduling",['full utilization'],disabled=True, #st.session_state.evo_running,
            help='The strategy to schedule verification jobs across nodes. Currently, only full utilization is supported, which means scheduling verify jobs immediately when a node is free.'
        )
    with col4:
        node_schedule=st.selectbox("Network Scheduling",['load balancing'],disabled=True, #st.session_state.evo_running,
            help='Overall network task scheduling strategy. Currently, only load balancing is supported, which always assigns the jobs to the highest available nodes.'
        )
    with col5:
        st.write('')
        st.write('')
        # distributed='Distributed ' if evosys.remote_db else ''
        if not st.session_state.evo_running:
            run_evo_btn = st.button(
                f":rainbow[***Launch Evolution***] :rainbow[🚀]",
                disabled=not evosys.remote_db or passive_mode,
                # use_container_width=True
            ) 
        else:
            stop_evo_btn = st.button(
                f"***Stop Evolution*** 🛑",
                disabled=not evosys.remote_db,
                # use_container_width=True
            )
    
    if not st.session_state.evo_running:
        if run_evo_btn:       
            with st.spinner('Launching...'):  
                launch_evo(evosys,input_max_designs_per_node,input_max_designs_total)
    else:
        if stop_evo_btn:
            if st.session_state.command_center:
                with st.spinner('Stopping... Note, the nodes will keep working on the unfinished jobs'):
                    stop_evo()

    if not evosys.remote_db:
        st.warning("Now only support distributed mode, all working nodes should run in listening mode.")

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

    st.subheader("Phylogenetic Tree Monitor")

    col1, col2, col3 = st.columns([6,0.1,2])
    

    if 'ptree_max_nodes' not in st.session_state:
        st.session_state.ptree_max_nodes=100

    _bg_color=AU.theme_aware_options(st,"#fafafa","#f0f0f0","#fafafa")
    
    evosys.ptree.export(max_nodes=st.session_state.ptree_max_nodes,height='800px',bgcolor=_bg_color)
    ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_{st.session_state.ptree_max_nodes}.html')

    with col1:
        _max_nodes=st.slider('Max Nodes to Display',min_value=0,max_value=len(evosys.ptree.G.nodes),value=st.session_state.ptree_max_nodes)

    # check this: https://github.com/napoles-uach/streamlit_network 
    with col3:
        st.write('')
        st.write('')
        if st.button(f'Refresh & Sync Tree'):#,use_container_width=True):
            evosys.ptree.update_design_tree()
            evosys.ptree.export(max_nodes=_max_nodes,height='800px',bgcolor=_bg_color)
            ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_{_max_nodes}.html')
            st.session_state.ptree_max_nodes=_max_nodes
    
    st.write(f'**First {st.session_state.ptree_max_nodes} nodes under the namespace ```{evosys.evoname}```**. '
            'Legend: :red[Seed Designs (*Displayed Pink*)] | :blue[Design Artifacts] | :orange[Reference w/ Code] | :violet[Reference w/o Code] *(Size by # of citations)*')

    HtmlFile = open(ptree_dir_small, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 800)


    with st.sidebar:
        AU.running_status(st,evosys)

        st.button('🔄 Refresh',use_container_width=True)

        if st.session_state.command_center:
            st.download_button(
                label="📩 Download Logs",
                data=json.dumps(st.session_state.command_center.read_logs(),indent=4),
                file_name=f"{evosys.evoname}_logs.json",
                mime="text/json",
                use_container_width=True
            )


        # logo = AU.square_logo("μLM")
        # logo_path = U.pjoin(pathlib.Path(__file__).parent,'..','assets','storm_logo.svg')
        # logo=AU.svg_to_image(logo_path)
        # st.image(logo, use_column_width=True)


if __name__ == '__main__':
    from model_discovery.evolution import BuildEvolution
    import argparse

    AU.print_cli_title()

    setting=AU.get_setting()
    default_namespace=setting.get('default_namespace','test_evo_000')   

    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--evoname', default=default_namespace, type=str) # the name of the whole evolution
    parser.add_argument('-dn','--max_designs_per_node', type=int, default=3) # the max number of threads to use
    parser.add_argument('-dt','--max_designs_total', type=int, default=10) # the group id of the evolution
    parser.add_argument('-g','--group_id', default='default', type=str) # the group id of the evolution
    args = parser.parse_args()
        
    print(f'Launching evolution for namespace {args.evoname} with group id {args.group_id}')
    evosys = BuildEvolution(
        params={'evoname':args.evoname, 'group_id':args.group_id}, 
        do_cache=False,
        # cache_type='diskcache',
    )
    command_center = CommandCenter(evosys,args.max_designs_per_node,args.max_designs_total,st)
    command_center.build_connection()
    command_center.run_evolution()


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


