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

    col1, col2 = st.columns([1, 3])
    with col1:
        mode = st.selectbox(label="Design Mode",options=['Design from scratch','Design from existing design'])
    with col2:
        instruction = st.text_input(label = "Add any additional instructions (optional)" )


    sources = ['ReferenceCoreWithTree', 'DesignArtifact', 'ReferenceCore', 'ReferenceWithCode', 'Reference', 'Reference1hop']
    sources={i:len(evosys.ptree.filter_by_type(i)) for i in sources}
    n_sources = {}

    st.subheader("Configure the number of seeds to sample from each source")
    cols = st.columns(len(sources))
    for i,source in enumerate(sources):
        with cols[i]:
            if mode=='Design from existing design' and source=='ReferenceCoreWithTree':
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=1,value=1,max_value=1,disabled=True)
            else:
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=0,max_value=sources[source])

    submit = st.button(label="Design model")

    if submit:
        if sum(n_sources.values())==0:
            st.write("You selected no seed, the agent will randomly generate a design for you.")
        with st.spinner(text="running model design loop"):
            session_id=f'sample_test_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
            log_dir=U.pjoin(evosys.evo_dir,'log',session_id)
            
            instruct,seeds=evosys.select(n_sources) # use the seed_ids to record the phylogenetic tree
        if instruction:
            instruct+=f'\n\n## Additional Instructions from the user\n\n{instruction}'
        system(instruct,frontend=True,stream=st,log_dir=log_dir)

    
