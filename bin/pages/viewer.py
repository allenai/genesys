import json
import time
import pathlib
import streamlit as st
import sys,os

from streamlit_markmap import markmap
from streamlit_timeline import timeline

sys.path.append('.')
from model_discovery.agents.flow.alang import DialogTreeViewer
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
        timeline(thread.to_timeline(),height=800)

    with st.sidebar:
        st.write("Empty sidebar")
    
