
import json
import time
import streamlit as st
import sys,os
import random
sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.evolution import ConnectionManager

from model_discovery.agents.roles.selector import *
import datetime

from model_discovery.model.composer import GAUTree,GAUNode,UnitSpec
from model_discovery.agents.flow.gau_flows import GAU_TEMPLATE





def tester(evosys,project_dir):
    
    ### Sidebar
    with st.sidebar:
        AU.running_status(st,evosys)

    st.title('Testing page (for internal use)')

    sess_id='2024-09-17-16-43-41-6f5f87'
    tail = sess_id.split('-')[-1]
    timestr=sess_id[:-len(tail)-1]
    st.write(timestr)

    sess_id = '2024-10-11-20-49-17-555a28'

    tree = evosys.ptree.get_gau_tree('gpt2')
