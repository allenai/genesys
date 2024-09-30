import json
import time
import pathlib
import streamlit as st
import sys,os
from model_discovery.agents.flow.gau_flows import DesignModes


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU




def design_selector(evosys,project_dir):
    st.header('Design Selector')
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        # st.markdown("#### Configure design mode")
        mode = st.selectbox(label="Design Mode",options=[i.value for i in DesignModes])
    with col2:
        selection_mode = st.selectbox(label="Selection Mode",options=['random'])


    st.subheader('Random Select')
    n_sources = {}

    sources = ['ReferenceCoreWithTree', 'DesignArtifactImplemented', 'DesignArtifact', 'ReferenceCore', 'ReferenceWithCode', 'Reference']
    sources={i:len(evosys.ptree.filter_by_type(i)) for i in sources}
    st.markdown("##### Configure the number of seeds to sample from each source")
    cols = st.columns(len(sources))
    for i,source in enumerate(sources):
        with cols[i]:
            if mode==DesignModes.MUTATION.value and source=='ReferenceCoreWithTree':
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=1)#,max_value=1,disabled=True)
            else:
                init_value=0 if source in ['DesignArtifact','ReferenceCore'] else min(2,sources[source])
                if source == 'DesignArtifactImplemented':
                    init_value = min(1,sources[source])
                # disabled=True if source == 'DesignArtifact' else False
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=init_value,max_value=sources[source])#,disabled=disabled)
    
    if st.button('Select'):
        instruct,seeds,refs=evosys.select_design(n_sources,mode=DesignModes(mode))

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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selection_mode = st.selectbox(label="Selection Mode",options=['random'])
    with col2:
        st.write('')
        with st.expander('Remaining Budget'):
            st.write(evosys.verify_budget)
    with col3:
        scale = st.selectbox(label="Scale",options=evosys.target_scales)
    with col4:
        st.write('')
        with st.expander(f'Unverified Designs under :red[**{scale}**]'):
            unverified = evosys.ptree.get_unverified_designs(scale)
            st.write(unverified)



    verify_selected = st.button('Select')

    if verify_selected:
        design_id,scale=evosys.select_verify()
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

