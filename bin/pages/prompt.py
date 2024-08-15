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

def mock1():
    st.title("Mock Page 1")

def mock2():
    st.title("Mock Page 2")

def mock3():
    st.title("Mock Page 3")

def mock4():
    st.title("Mock Page 4")


def prompt(evosys,project_dir):

    st.title("Prompt Lab")
    
    ve_pages = {
        "Your account": [
            st.Page(mock1,title="Create your account"),
            st.Page(mock2,title="Manage your account"),
        ],
        "Resources": [
            st.Page(mock3,title="Learn about us"),
            st.Page(mock4,title="Try it out"),
        ],
    }

    pg = st.navigation(ve_pages)
    # pg.run()
    st.markdown('## ALang Design Flow Viewer')

    system=evosys.rnd_agent
    evo_dir=evosys.evo_dir
    
    # simple_mode = 'VIEW_FLOWCHART_SIMPLE' 
    # if simple_mode not in st.session_state:
    #     st.session_state[simple_mode] = True

    # with st.sidebar:
    #     if st.button('View Simplified Flowchart'):
    #         st.session_state[simple_mode] = True
    #     if st.button('View Full Flowchart'):
    #         st.session_state[simple_mode] = False

    simple_mode = True

    selected_id = system.design_flow_test.export(700,simplify=simple_mode)
    nodes=system.design_flow_test.nodes

    if simple_mode:
        if selected_id:
            node_id=int(selected_id)
            if node_id in nodes:
                node=nodes[node_id]
                if node:
                    source=inspect.getsource(node._call)
                    source=U.remove_leading_indent(source)
                    st.markdown(f'### Selected Node ID {node_id}: {node.alias}')
                    st.code(source)
                else:
                    st.markdown(f'### Node ID {node_id} does not have a source.')

            else:
                st.markdown('### Select a code and view source here.')
        else:
            st.markdown('### Select a code and view source here.')


    design_code = inspect.getsource(system._design)
    design_code=U.remove_leading_indent(design_code)
    design_fc=pfc.Flowchart.from_code(design_code)
    
    fc_dir=U.pjoin(evo_dir,'design_flowchart_raw.html')
    pfc.output_html(fc_dir,'Design Flowchart',design_fc.flowchart())

    if st.button('Click here to view the Control Flow Graph of the Python based Design Flow'):
        check_output("start " + fc_dir, shell=True)

    # st.code(design_fc.flowchart())

    # components.html(open(fc_dir).read(),height=800,scrolling=True)



