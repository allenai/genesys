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



def run_evolve(evosys,project_dir):
    raise NotImplementedError

def run_sample(evosys,project_dir,step=False):
    while evosys.evolve():
        if step:
            break

def evolve(evosys,project_dir):

    st.title("Evolution Engine")
    
    evo_state={}
    evo_state['evoname']=evosys.evoname
    evo_state['evo_dir']=evosys.evo_dir
    evo_state['data_dir']=os.environ.get("DATA_DIR")
    evo_state['selection_method']=evosys.select_method
    evo_state.update(evosys.load_state())
    st.header("System Info")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Status"):
            st.write(evo_state)
    with col2:
        with st.expander("Configuration"):
            st.write(evosys._config)

    st.header("Evolution Controls")
    col1, col2, col3 = st.columns(3,gap='small')
    with col1:
        st.button("Run Pure Sampling",on_click=run_sample,args=(evosys,project_dir))
    with col2:
        st.button("Step One Sampling",on_click=run_sample,args=(evosys,project_dir,True))


    st.header("Phylogenetic Tree Viewer")
    ptree_dir_full=U.pjoin(evosys.evo_dir,f'PTree.html')
    if st.button('Click here to view the Full Phylogenetic Tree'):
        check_output("start " + ptree_dir_full, shell=True)

    # check this: https://github.com/napoles-uach/streamlit_network 

    max_nodes=200
    evosys.ptree.export(max_nodes=max_nodes,height='800px')
    ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_{max_nodes}.html')
    st.write(f'Only showing the first {max_nodes} nodes.')
    HtmlFile = open(ptree_dir_small, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 800)


