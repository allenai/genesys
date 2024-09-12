import json
import time
import pathlib
import streamlit as st
import sys,os
import torch
import platform
import psutil


sys.path.append('.')
import model_discovery.utils as U
from model_discovery.agents.search_utils import SuperScholarSearcher




def search(evosys,project_dir):

    st.title("Super Scholar Search Engine")
    







