import json
import time
import pathlib
import streamlit as st
import sys,os
import inspect
import pyflowchart as pfc
import streamlit.components.v1 as components
from subprocess import check_output

sys.path.append('.')
import model_discovery.utils as U

from model_discovery.evolution import EvolutionSystem

from model_discovery.agents.flow.gu_flow import GUFlow

def mock1():
    st.title("Mock Page 1")

def mock2():
    st.title("Mock Page 2")

def mock3():
    st.title("Mock Page 3")

def mock4():
    st.title("Mock Page 4")


def prompt(evosys:EvolutionSystem,project_dir):

    st.title("Prompt Lab")
    
    # ve_pages = {
    #     "Your account": [
    #         st.Page(mock1,title="Create your account"),
    #         st.Page(mock2,title="Manage your account"),
    #     ],
    #     "Resources": [
    #         st.Page(mock3,title="Learn about us"),
    #         st.Page(mock4,title="Try it out"),
    #     ],
    # }

    # pg = st.navigation(ve_pages)
    # pg.run()
    st.markdown('## ALang Design Flow Viewer')

    system=evosys.rnd_agent
    evo_dir=evosys.evo_dir
    
    simple_mode = 'VIEW_FLOWCHART_SIMPLE' 
    if simple_mode not in st.session_state:
        st.session_state[simple_mode] = True

    with st.sidebar:
        if st.button('View Simplified Flowchart'):
            st.session_state[simple_mode] = True
            st.rerun()
        if st.button('View Full Flowchart'):
            st.session_state[simple_mode] = False
            st.rerun()


    simple_mode = st.session_state[simple_mode] # True

    # flow=system.design_flow
    # script=system.DESIGN_ALANG_reformatted

    gu_flow = GUFlow(system,None,None)
    flow = gu_flow.flow
    script = gu_flow.script

    if simple_mode:
        col1, col2 = st.columns([2,1])
        with col1:
            selected_id = flow.export(800,simplify=simple_mode)

        with col2:
            nodes=flow.nodes
            if selected_id:
                node_id=int(selected_id)
                if node_id in nodes:
                    node=nodes[node_id]
                    if node:
                        source=node.inspect()
                        st.markdown(f'### Selected Node ID {node_id}: {node.alias}')
                        st.code(source)
                    else:
                        st.markdown(f'### Node ID {node_id} does not have a source.')
                else:
                    st.markdown('### Select a code and view source here.')
            else:
                st.markdown('### Select a code and view source here.')
    else:
        flow.export(800)


    st.markdown('## ALang Design Flow Source')
    st.write('Automatically reformatted by compiler')
    st.code(script,line_numbers=True,language='bash')

    st.markdown('## Naive Control Flow Viewer')

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Click here to view the Flow Chart of a Naive Design Flow'):
            fc_dir = system.design_flow_naive.export(evo_dir)
            check_output("start " + fc_dir, shell=True)
        st.code(inspect.getsource(system.design_flow_naive.prog))

    with col2:
        if st.button('Click here to view the Flow Chart of a Naive Review Flow'):
            fc_dir = system.review_flow.export(evo_dir)
            check_output("start " + fc_dir, shell=True)
        st.code(inspect.getsource(system.review_flow.prog))

    # components.html(open(fc_dir).read(),height=800,scrolling=True)
