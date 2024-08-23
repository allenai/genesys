import json
import time
import pathlib
import datetime
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
        
    col1, col2 = st.columns([3, 1])
    with col1:
        instruction = st.text_input(label = "Add any additional instructions (optional)" )
    with col2:
        K = st.number_input(label="Number of seeds to sample",min_value=1,value=2)

    submit = st.button(label="Design model")

    if submit or instruction:
        instruction = str(None) if not instruction else instruction 
        
        with st.spinner(text="running discovery loop"):
            session_id=f'sample_test_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
            log_dir=U.pjoin(evosys.evo_dir,'log',session_id)
            
            instruct,seeds=evosys.select(K) # use the seed_ids to record the phylogenetic tree
            instruct+=f'\n\n## Additional Instructions from the user\n\n{instruction}'
            system(instruct,frontend=True,stream=st,log_dir=log_dir)


            