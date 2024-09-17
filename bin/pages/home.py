import json
import time
import pathlib
import streamlit as st
import sys,os


sys.path.append('.')
import model_discovery.utils as U




def home(evosys,project_dir):
    
    readme = U.read_file(U.pjoin(project_dir,'README.md'))
    st.markdown(readme)

    # st.balloons()

    with st.sidebar:
        st.write("*Welcome to the Model Discovery System!*")
       
        st.image('https://images.unsplash.com/photo-1722691694088-b3b2ab29be31?q=80&w=2535&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')
