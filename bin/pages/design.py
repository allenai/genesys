import json
import time
import pathlib
import streamlit as st
import sys,os


sys.path.append('.')
import model_discovery.utils as U




def design(evosys,project_dir):

    system = evosys.rnd_agent

    ### build the system 
    st.title("Model Design Engine")

    ### side bar 
    st.sidebar.button("reset design query")
        
    filler = "Find me a new model"

    instruction = st.text_input(label = "Add any additional instructions (optional)" )
    submit = st.button(label="Design model")

    
    if submit or instruction:
        instruction = str(None) if not instruction else instruction 
        
        with st.spinner(text="running discovery loop"):
            system(instruction,frontend=True,stream=st)