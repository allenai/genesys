
import json
import time
import streamlit as st
import sys,os
import random
import shutil
sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.evolution import ConnectionManager

from model_discovery.agents.roles.selector import *
import datetime

from model_discovery.model.composer import GAUTree,GAUNode,UnitSpec
from model_discovery.agents.flow.gau_flows import GAU_TEMPLATE

import pandas as pd

from model_discovery.agents.roles.selector import scale_weight_results

import tiktoken
import anthropic

from model_discovery.agents.agent_utils import *
import model_discovery.utils as U




def tester(evosys,project_dir):
    
    ### Sidebar
    with st.sidebar:
        AU.running_status(st,evosys)

    st.title('Testing page (for debugging)')




