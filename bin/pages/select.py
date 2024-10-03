import copy
import json
import time
import pathlib
import streamlit as st
import sys,os

from model_discovery.agents.flow.gau_flows import DesignModes
from model_discovery.agents.roles.selector import DEFAULT_SEED_DIST,SCHEDULER_OPTIONS
from model_discovery.evolution import DEFAULT_N_SOURCES


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU




def design_selector(evosys,project_dir):
    st.header('Design Selector')

    seed_dist = copy.deepcopy(DEFAULT_SEED_DIST)

    _col1,_col2=st.columns([2,3])
    with _col1:
        st.write('###### Configure Selector')
        col1, col2 = st.columns(2)
        with col1:
            # st.markdown("#### Configure design mode")
            mode = st.selectbox(label="Design Mode",options=[i.value for i in DesignModes])
        with col2:
            select_method = st.selectbox(label="Selection Method",options=['random'])
    with _col2:
        st.write('###### Configure *Seed* Selection Distribution')
        cols = st.columns(3)
        with cols[0]:
            seed_dist['scheduler'] = st.selectbox('Scheduler',options=SCHEDULER_OPTIONS,index=SCHEDULER_OPTIONS.index(seed_dist['scheduler']))
        with cols[1]:
            seed_dist['restart_prob'] = st.slider('Restart Probability',min_value=0.0,max_value=1.0,step=0.01,value=seed_dist['restart_prob'])
        with cols[2]:
            seed_dist['warmup_rounds'] = st.number_input('Warmup Rounds',min_value=0,value=seed_dist['warmup_rounds'])

    n_sources = DEFAULT_N_SOURCES

    sources={i:len(evosys.ptree.filter_by_type(i)) for i in DEFAULT_N_SOURCES}
    st.markdown("###### Configure the number of *references* from each source")
    cols = st.columns(len(sources))
    for i,source in enumerate(sources):
        with cols[i]:
            if source in ['DesignArtifact','DesignArtifactImplemented']:
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=n_sources[source])#,disabled=True)
            else:
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=n_sources[source],max_value=sources[source])#,disabled=True)
    
    if st.button('Select'):
        selector_args = {
            'n_sources': n_sources,
        }
        
        instruct,seeds,refs=evosys.selector.select_design(selector_args,DesignModes(mode),select_method)

        st.subheader(f'**Instructions from the selector:**')
        if instruct:
            st.write(instruct)
        else:
            st.warning('No instructions from the selector.')

        st.subheader(f'**{len(seeds)} seeds selected:**')
        for seed in seeds:
            with st.expander(f'**{seed.acronym}** *({seed.type})*'):
                st.write(seed.to_prompt())

        st.subheader(f'**{len(refs)} references selected:**')
        for ref in refs:
            with st.expander(f'**{ref.acronym}** *({ref.type})*'):
                st.write(ref.to_prompt())
    else:
        st.info(f'**NOTE:** All settings here will only be applied to this playground. The playground will directly work on the selected running namespace ```{evosys.evoname}```.')



def verify_selector(evosys,project_dir):
    st.header('Verify Selector')

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        verify_strategy = st.selectbox(label="Verify Strategy",options=['random'])
    with col2:
        st.write('')
        with st.expander('Remaining Budget'):
            st.write(evosys.selector.verify_budget)
    with col3:
        scale = st.selectbox(label="Scale",options=evosys.target_scales)
    with col4:
        st.write('')
        with st.expander(f'Unverified Designs under ***{scale}***'):
            unverified = evosys.ptree.get_unverified_designs(scale)
            st.write(unverified)

    if verify_strategy == 'random':
        st.subheader('Random Strategy')
        st.write('*Random strategy will use up smaller scale budgets first.*')

    verify_selected = st.button('Select')

    if verify_selected:
        design_id,scale=evosys.selector.select_verify(verify_strategy=verify_strategy)
        if design_id is None:
            st.warning('No design to verify.')
        else:
            st.write(f'Selected {design_id} at scale {scale} to be verified.')


def select(evosys,project_dir):
    
    st.title('Selector Playground')

    if 'view_selector' not in st.session_state:
        st.session_state.view_selector = 'design'

    with st.sidebar:
        AU.running_status(st,evosys)

        st.write('Choose the selector to view')
        if st.button('**Design Selector**' if st.session_state.view_selector == 'design' else 'Design Selector',use_container_width=True):
            st.session_state.view_selector = 'design'
        if st.button('**Verify Selector**' if st.session_state.view_selector == 'verify' else 'Verify Selector',use_container_width=True):
            st.session_state.view_selector = 'verify'


    if st.session_state.view_selector == 'design':
        design_selector(evosys,project_dir)
    elif st.session_state.view_selector == 'verify':
        verify_selector(evosys,project_dir)

