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

