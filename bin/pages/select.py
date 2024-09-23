import json
import time
import pathlib
import streamlit as st
import sys,os
from model_discovery.agents.flow.gau_flows import DesignModes


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU


def select(evosys,project_dir):
    
    st.title('Seed Selector')

    with st.sidebar:
        st.write(f'**Namespace: ```{evosys.evoname}```**')



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
        instruct,seeds,refs=evosys.select(n_sources,mode=DesignModes(mode))

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


