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




def run_evolve(evosys,project_dir):
    raise NotImplementedError

def run_sample(evosys,project_dir,step=False):
    while evosys.evolve():
        if step:
            break


def evolve(evosys,project_dir):

    st.title("Evolution System")
    
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
    # ptree_dir_full=U.pjoin(evosys.evo_dir,f'PTree.html')

    col1, col2, col3 = st.columns([6,0.1,2])
    
    max_nodes=100
    
    evosys.ptree.export(max_nodes=100,height='800px')
    ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_100.html')

    with col1:
        _max_nodes=st.slider('Max Nodes',min_value=0,max_value=len(evosys.ptree.G.nodes),value=100)

    # check this: https://github.com/napoles-uach/streamlit_network 
    with col3:
        st.write('')
        st.write('')
        if st.button(f'Refresh Tree with First {_max_nodes} Nodes'):#,use_container_width=True):
            evosys.ptree.export(max_nodes=_max_nodes,height='800px')
            ptree_dir_small=U.pjoin(evosys.evo_dir,f'PTree_{_max_nodes}.html')
            max_nodes=_max_nodes
    
    # with col3:
    #     st.write('')
    #     st.write('')
    #     if st.button('View Full Phylogenetic Tree',use_container_width=True):
    #         if not U.pexists(ptree_dir_full):
    #             with st.spinner('Loading...'):
    #                 evosys.ptree.export()
    #         check_output("xdg-open " + ptree_dir_full, shell=True)
            
    st.write(f'**First {max_nodes} nodes under the namespace ```{evosys.evoname}```**. '
            'Legend: :rainbow[Seed Designs (*Pink*)] | :blue[Design Artifacts] | :orange[Reference w/ Code] | :violet[Reference w/o Code] *(Size by # of citations)*')

    HtmlFile = open(ptree_dir_small, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 800)

    with st.sidebar:
        AU.running_status(st,evosys)

        # logo = AU.square_logo("μLM")
        # logo_path = U.pjoin(pathlib.Path(__file__).parent,'..','assets','storm_logo.svg')
        # logo=AU.svg_to_image(logo_path)
        # st.image(logo, use_column_width=True)

    