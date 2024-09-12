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
    