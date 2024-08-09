import json
import time
import pathlib
import streamlit as st
import sys,os

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


    st.header("Tree Viewer")
    ptree_dir=U.pjoin(evosys.ckpt_dir,'PTree.html')
    ptree=U.read_file(ptree_dir)
    st.html(ptree)
