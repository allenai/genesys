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

    st.title("Paper Search Engine")

    with st.expander("Search Configurations",expanded=True):
        search_cfg={}
        search_cfg['result_limits']={}
        search_cfg['perplexity_settings']={}

        cols=st.columns(4)
        with cols[0]:
            search_cfg['result_limits']['lib']=st.number_input("Library: Primary Search Result Limit",value=5,min_value=0,step=1)
        with cols[1]:
            search_cfg['result_limits']['lib2']=st.number_input("Library: Secondary Search Result Limit",value=0,min_value=0,step=1,disabled=True)
        with cols[2]:
            search_cfg['result_limits']['libp']=st.number_input("Library: Plus Search Result Limit",value=0,min_value=0,step=1,disabled=True)
        with cols[3]:
            search_cfg['rerank_ratio']=st.slider("Rerank Scale Ratio (0 means no rerank)",min_value=0.0,max_value=1.0,value=0.2,step=0.01)

        cols=st.columns([2,2,2,2,2,1])
        with cols[0]:
            search_cfg['result_limits']['s2']=st.number_input("S2 Search Result Limit",value=5,min_value=0,step=1)
        with cols[1]:
            search_cfg['result_limits']['arxiv']=st.number_input("Arxiv Search Result Limit",value=3,min_value=0,step=1)
        with cols[2]:
            search_cfg['result_limits']['pwc']=st.number_input("Papers With Code Search Result Limit",value=3,min_value=0,step=1)
        with cols[3]:
            search_cfg['perplexity_settings']['model_size']=st.selectbox("Perplexity Model Size",options=['none','small','large','huge'],index=2)
        with cols[4]:
            search_cfg['perplexity_settings']['max_tokens']=st.number_input("Perplexity Max Tokens",value=2000,min_value=500,step=100,disabled=search_cfg['perplexity_settings']['model_size']=='none')
        with cols[5]:
            st.write("")
            st.write("")
            prompting=st.checkbox("Prompting",value=False)

    sss=SuperScholarSearcher(evosys.ptree,stream=st,cfg=search_cfg)

    
    details=st.text_area("Search Content with Detailed Query (for vector store search)",placeholder='I want to ask about ...',height=100)

    cols=st.columns([9,1])
    with cols[0]:
        query=st.text_input("Search Title and Abstract")
    with cols[1]:
        st.write("")
        st.write("")
        search_btn=st.button("Search",use_container_width=True)
    if search_btn:
        with st.spinner('Searching...'):    
            prt=sss(query,details,prompt=prompting)
            st.markdown(prt,unsafe_allow_html=True)
    







