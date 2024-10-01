import json
import time
import pathlib
import streamlit as st
import sys,os
from subprocess import check_output
import graphviz
import streamlit.components.v1 as components


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU


class DistributedCommandCenter:
    def __init__(self,evosys,max_design_threads_total,stream,cli=False):
        self.evosys=evosys
        self.max_design_threads_total=max_design_threads_total
        self.st=stream
        self.cli=cli

    def run_evolution(self):
    


def x_evolve(evosys,max_design_threads_total,cli=False): # extereme evolution 
    if not cli:
        st.session_state.evo_running=True
    raise NotImplementedError


def x_stop_evo(evosys,cli=False):
    if not cli:
        st.session_state.evo_running=False
    raise NotImplementedError

def get_evo_state(evosys):
    evo_state={}
    evo_state.update(evosys.state)
    evo_state.pop('budgets')
    evo_state.pop('design_budget')
    if 'action_strategy' in evo_state:
        evo_state.pop('action_strategy')
    evo_state['target_scales']=evosys.target_scales
    evo_state.pop('scales')
    evo_state['remaining_verify_budget']=evosys.verify_budget
    evo_state['remaining_design_budget']=evosys.design_budget
    evo_state['design_cost']=evosys.ptree.design_cost
    return evo_state

def evolve(evosys,project_dir):

    st.title("Evolution System")

    if st.session_state.listening_mode:
        st.warning("**NOTE:** You are running in listening mode. You cannot control the evolution system by yourself.")
    
    st.header("System Check")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("System Status"):
            st.write(get_evo_state(evosys))
    with col2:
        with st.expander("Configuration"):
            st.write(evosys._config)

    st.header("Launch Pad")
    col1, _, col2, _, col3, _, col4 = st.columns([1.2,0.05,1,0.05,1,0.05,2],gap='small')
    with col1:
        max_design_threads_total=st.number_input("Max Design Threads (bounded by API rate)",min_value=1,value=4)
    with col2:
        # always use extreme mode, use as much gpus as possible
        verify_schedule=st.selectbox("Verification Scheduling",['extremely'])
    with col3:
        node_schedule=st.selectbox("Workload Scheduling",['load balanced'])
    with col4:
        st.write('')
        st.write('')
        distributed='Distributed ' if evosys.remote_db else ''
        if not st.session_state.evo_running:
            st.button(
                f":rainbow[***Launch {distributed}Evolution***] :rainbow[🚀]",
                on_click=x_evolve,args=(evosys,max_design_threads_total),
                disabled=st.session_state.listening_mode or not evosys.remote_db
            )
        else:
            st.button(
                f"***Stop {distributed}Evolution*** 🛑",
                disabled=st.session_state.listening_mode or not evosys.remote_db,
                on_click=x_stop_evo
            )
    
    if not evosys.remote_db:
        st.warning("Now only support distributed mode, all working nodes should run in listening mode.")


    st.header("Phylogenetic Tree Monitor")

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
        if st.button(f'Refresh Tree with First {_max_nodes} Nodes'):#,use_container_width=True):
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

        # logo = AU.square_logo("μLM")
        # logo_path = U.pjoin(pathlib.Path(__file__).parent,'..','assets','storm_logo.svg')
        # logo=AU.svg_to_image(logo_path)
        # st.image(logo, use_column_width=True)

    