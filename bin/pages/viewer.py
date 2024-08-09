import json
import time
import pathlib
import streamlit as st
import sys,os

from streamlit_markmap import markmap
from streamlit_timeline import timeline

sys.path.append('.')
from model_discovery.system import DialogTreeViewer
import model_discovery.utils as U




def viewer(evosys,project_dir):
    
    ### build the system 
    st.title("Agent Viewer")

    log_dir = U.pjoin(evosys.evo_dir, 'log')
    dialogs = {}
    for d in os.listdir(log_dir):
        dialogs[d] = DialogTreeViewer(U.pjoin(log_dir, d))

    if not dialogs:
        st.warning("No dialogs found in the log directory")
    else:
        selected_dialog = st.selectbox("Select a dialog", list(dialogs.keys()))
        dialog = dialogs[selected_dialog]
        markmap(dialog.to_markmap(),height=300)
        selected_thread = st.selectbox("Select a thread", list(dialog.threads.keys()))
        thread = dialog.threads[selected_thread]
        timeline(thread.to_timeline(),height=600)

    

    ### side bar 
    # st.sidebar.button("reset")


        
    # filler = "Find me a new model"

    # instruction = st.text_input(label = "Add any additional instructions (optional)" )
    # submit = st.button(label="Design model")

    
    # if submit or instruction:
    #     instruction = str(None) if not instruction else instruction 
        
    #     with st.spinner(text="running discovery loop"):
    #         evosys(instruction,frontend=True,stream=st)
        
    