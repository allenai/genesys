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

    cfg={}
    cfg['result_limits']={}
    cols=st.columns(3)
    with cols[0]:
        cfg['result_limits']['s2']=st.number_input("S2 Search Result Limit",value=10,min_value=1,step=1)
    with cols[1]:
        cfg['result_limits']['arxiv']=st.number_input("Arxiv Search Result Limit",value=5,min_value=1,step=1)
    with cols[2]:
        cfg['result_limits']['pwc']=st.number_input("Papers With Code Search Result Limit",value=5,min_value=1,step=1)

    sss=SuperScholarSearcher(evosys.ptree,stream=st,cfg=cfg)

    cols=st.columns([9,1])
    with cols[0]:
        query=st.text_input("Search Query")
    with cols[1]:
        st.write("")
        st.write("")
        search_btn=st.button("Search",use_container_width=True)
    if query or search_btn:
        with st.spinner('Searching...'):    
            _,prt=sss.search_external(query,pure=False)
            st.markdown(prt)
    







