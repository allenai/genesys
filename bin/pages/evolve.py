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
from streamlit.runtime.scriptrunner import add_script_run_ctx


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU


def _is_running(evosys,zombie_threshold=30):
    docs = evosys.remote_db.collection('experiment_connections').get()
    for doc in docs:
        if doc.to_dict().get('status','n/a') == 'connected':
            last_heartbeat = doc.to_dict().get('last_heartbeat')
            threshold_time = datetime.now(pytz.UTC) - timedelta(seconds=zombie_threshold)
            is_zombie = last_heartbeat < threshold_time
            evoname = doc.to_dict().get('evoname')
            group_id = doc.to_dict().get('group_id')
            if not is_zombie and (group_id==evosys.CM.group_id or evoname==evosys.evoname):
                return evoname, group_id
            else:
                return None, None
    return None, None

class CommandCenter:
    def __init__(self,evosys,design_to_verify_ratio,stream,cli=False):
        self.evosys=evosys
        self.evoname=evosys.evoname
        self.design_to_verify_ratio=design_to_verify_ratio
        self.st=stream
        self.doc_ref = evosys.remote_db.collection('experiment_connections').document(self.evosys.evoname)
        self.log_ref = evosys.remote_db.collection('experiment_logs').document(self.evosys.evoname)
        self.running = False
        self.cli=cli
        self.poll_freq=20
        self.zombie_threshold = 30  # seconds

    @property
    def max_design_threads_total(self):
        verify_availability = sum([self.evosys.CM.accept_verify_job.values()])
        return self.design_to_verify_ratio * verify_availability

    def read_logs(self):
        return self.log_ref.get().to_dict()

    def build_connection(self):
        # check if the node_id is already in the collection
        evoname, _ = _is_running(self.evosys,self.zombie_threshold)
        if evoname:
            self.active = False
            self.evoname = evoname
        else:
            self.doc_ref.set({
                'status': 'connected',
                'last_heartbeat': firestore.SERVER_TIMESTAMP,
                'evoname': self.evosys.evoname,
                'group_id': self.evosys.CM.group_id,
            })
            self.active = True

    def _assign_design_workload(self,design_availability):
        # find the node with largest availability
        node_id = max(design_availability, key=design_availability.get)
        self.evosys.CM.design_command(node_id)
        design_availability[node_id] -= 1
        return design_availability
        
    def cleanup(self):
        # st.info("Cleaning up and disconnecting...")
        self.doc_ref.delete()  # Delete the connection document

    def run_evolution(self):
        self.evosys.sync_to_db() # sync the config to the db
        self.running = True
        while self.running:
            if self.active:
                self.evosys.CM.get_active_connections() # refresh the connection status
                design_workloads, verify_workloads = self.evosys.CM.get_all_workloads()

                design_availability = {k:self.evosys.CM.max_design_threads[k] - v for k,v in design_workloads.items()}
                if sum(design_availability.values())>0:
                    for _ in range(self.max_design_threads_total-sum(design_workloads.values())):
                        design_availability = self._assign_design_workload(design_availability)
                    
                for node_id in verify_workloads:
                    if verify_workloads[node_id] == 0 and self.evosys.CM.accept_verify_job[node_id]:
                        self.evosys.CM.verify_command(node_id)
                
                self.doc_ref.update({'last_heartbeat': firestore.SERVER_TIMESTAMP})
                
            time.sleep(self.poll_freq)  
        self.cleanup()

    def stop_evolution(self):
        self.running = False
        self.cleanup()
        


def x_evolve(command_center,cli=False): # extereme evolution 
    thread = threading.Thread(target=command_center.run_evolution)
    if not cli:
        add_script_run_ctx(thread)  # Add Streamlit context to the thread
    thread.start()
    return thread

def launch_evo(evosys,design_to_verify_ratio=4):
    if len(evosys.CM.get_active_connections())==0:
        st.toast('No listener connected, evolution will not start. Please launch listeners first.',icon='🚨')
    else:
        command_center = CommandCenter(evosys,design_to_verify_ratio,st)
        command_center.build_connection()
        st.session_state.command_center = command_center
        st.session_state.command_center_thread = x_evolve(command_center)
        st.session_state.evo_running = True
        time.sleep(1)
        evoname = st.session_state.command_center.evoname
        st.toast(f"Evolution launched for ```{evoname}```.",icon='🚀')
        time.sleep(1)
        st.rerun()

def stop_evo():
    if st.session_state.command_center:
        evoname = st.session_state.command_center.evoname
        st.session_state.command_center.stop_evolution()
        st.session_state.command_center_thread.join()
        st.session_state.command_center = None
        st.session_state.command_center_thread = None
        st.session_state.evo_running = False
        time.sleep(1)
        st.toast(f"Evolution stopped for ```{evoname}```.",icon='🛑')
        time.sleep(1)
        st.rerun()

def evolve(evosys,project_dir):

    st.title("Evolution System")

    if 'command_center' not in st.session_state:
        st.session_state.command_center = None
    if 'command_center_thread' not in st.session_state:
        st.session_state.command_center_thread = None
     
    running_evoname,running_group_id = _is_running(evosys,zombie_threshold=30)
    if running_evoname:
        if not st.session_state.evo_running:
            if evosys.CM.group_id == running_group_id or evosys.evoname == running_evoname:
                st.toast(f'Network group ```{running_group_id}``` is already running for evolution ```{running_evoname}```. Launching a command center in passive mode.')
            else:
                st.toast(f'Evolution ```{running_evoname}``` is already running in network group ```{running_group_id}```. Launching a command center in passive mode.')
            launch_evo(evosys)
    else:
        if st.session_state.evo_running and not st.session_state.command_center.active:
            st.toast(f'The command center is not running for namespace {running_evoname} anymore. Stopping the command center.')
            stop_evo()


    passive_mode=False
    if st.session_state.command_center and st.session_state.command_center.running and not st.session_state.command_center.active:
        passive_mode=True
        _evoname = st.session_state.command_center.evoname
        st.info(f'The command center is already running for namespace ```{_evoname}```. You are in passive observation mode.')


    st.subheader("System Check")
    col1, col2 = st.columns([1,2])
    with col1:
        with st.expander("System Status", icon='💻'):
            st.write(evosys.get_evo_state())
    with col2:
        with st.expander("Full Raw Configuration", icon='🔍'):
            st.write(evosys._config)

    st.header("Launch Pad")
    col1, col2, col3, col4 = st.columns([1,1,1,1],gap='small')
    with col1:
        input_design_to_verify_ratio=st.number_input("Design to Verify Ratio",min_value=1,value=4,disabled=st.session_state.evo_running)
    with col2:
        # always use extreme mode, use as much gpus as possible
        verify_schedule=st.selectbox("Node Scheduling",['maximal utilization'],disabled=st.session_state.evo_running)
    with col3:
        node_schedule=st.selectbox("Network Scheduling",['load balancing'],disabled=st.session_state.evo_running)
    with col4:
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
                disabled=not evosys.remote_db or passive_mode,
                # use_container_width=True
            )
    
    if not st.session_state.evo_running:
        if run_evo_btn:       
            with st.spinner('Launching...'):  
                 
                launch_evo(evosys,input_design_to_verify_ratio)
    else:
        if stop_evo_btn:
            if st.session_state.command_center:
                with st.spinner('Stopping... Note, the nodes will keep working on the unfinished jobs'):
                    stop_evo()

    if not evosys.remote_db:
        st.warning("Now only support distributed mode, all working nodes should run in listening mode.")


    view_latest_K=10
    if st.session_state.evo_running:
        with st.expander(f"📝 **Running Logs** *(Latest {view_latest_K} logs)*",expanded=True):
            evo_log=st.session_state.command_center.read_logs()
            if evo_log:
                evo_log=sorted(evo_log.items(),key=lambda x:datetime.strptime(x[0].split("_")[0].strip(),'%B %d, %Y at %I:%M:%S %p'),reverse=True)
                for timestamp,log in evo_log[-view_latest_K:]:
                    _node_id=log.split(" ")[1]
                    _log=" ".join(log.split(" ")[2:])
                    st.write(f'**[{timestamp.split("_")[0]}]** Node ```{_node_id}``` {_log}')
                st.write(f'**......** *({len(evo_log)-view_latest_K} more logs are hidden, {len(evo_log)} total logs)*')
            else:
                st.info("No logs available at the moment.")

    st.subheader("Phylogenetic Tree Monitor")

    col1, col2, col3 = st.columns([6,0.1,2])
    
    max_nodes=100
    
    evosys.ptree.export(max_nodes=100,height='800px')
    ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_100.html')

    with col1:
        _max_nodes=st.slider('Max Nodes to Display',min_value=0,max_value=len(evosys.ptree.G.nodes),value=100)

    # check this: https://github.com/napoles-uach/streamlit_network 
    with col3:
        st.write('')
        st.write('')
        if st.button(f'Refresh & Sync Tree'):#,use_container_width=True):
            evosys.ptree.update_design_tree()
            evosys.ptree.export(max_nodes=_max_nodes,height='800px')
            ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_{_max_nodes}.html')
            max_nodes=_max_nodes
    
    
    st.write(f'**First {max_nodes} nodes under the namespace ```{evosys.evoname}```**. '
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

    AU.print_title()

    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--evoname', default='test_evo_000', type=str) # the name of the whole evolution
    parser.add_argument('-r','--design_to_verify_ratio', type=int, default=4) # the max number of threads to use
    parser.add_argument('-g','--group_id', default='default', type=str) # the group id of the evolution
    args = parser.parse_args()
        
    evosys = BuildEvolution(
        params={'evoname':args.evoname, 'group_id':args.group_id}, 
        do_cache=False,
        # cache_type='diskcache',
    )
    command_center = CommandCenter(evosys,args.design_to_verify_ratio,st,cli=True)
    command_center.build_connection()
    command_center_thread=x_evolve(command_center,cli=True)
    
    print("Evolution launched!")
    try:
        # Keep the main thread alive
        while command_center_thread.active:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        command_center_thread.stop_listening()
        command_center_thread.join(timeout=10)  # Wait for the thread to finish, with a timeout

    print("Evolution stopped.")


