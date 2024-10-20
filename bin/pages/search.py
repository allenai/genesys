import json
import time
import pathlib
import streamlit as st
import sys,os
import torch
import platform
import psutil
import copy

from model_discovery.agents.search_utils import EmbeddingDistance,OPENAI_EMBEDDING_MODELS,TOGETHER_EMBEDDING_MODELS,COHERE_EMBEDDING_MODELS
from model_discovery.evolution import LIBRARY_DIR

sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU


def paper_search(evosys,project_dir):

    st.subheader("Paper Search Engine")

    with st.expander("Search Configurations",expanded=True):
        search_cfg={}
        search_cfg['result_limits']={}
        search_cfg['perplexity_settings']={}
        search_cfg['proposal_search_cfg']={}

        cols=st.columns([2,2,2,3,2,3])
        with cols[0]:
            search_cfg['result_limits']['lib']=st.number_input("Library Primary",value=5,min_value=0,step=1,
                help='The core library with 300+ state-of-the-art language model architecture related papers.')
        with cols[1]:
            search_cfg['result_limits']['lib2']=st.number_input("Library Secondary",value=0,min_value=0,step=1,disabled=True,
                help='The secondary library of the papers that are cited by the primary library.')
        with cols[2]:
            search_cfg['result_limits']['libp']=st.number_input("Library Plus",value=0,min_value=0,step=1,disabled=True,
                help='The library of the papers that are recommended by Semantic Scholar for core library papers.')
        with cols[3]:
            search_cfg['rerank_ratio']=st.slider("Rerank Scale Ratio (0 means disable)",min_value=0.0,max_value=1.0,value=0.2,step=0.01)
        with cols[4]:
            search_cfg['proposal_search_cfg']['top_k']=st.number_input("Proposal Top K",value=3,min_value=0,step=1)
        with cols[5]:
            search_cfg['proposal_search_cfg']['cutoff']=st.slider("Proposal Search Cutoff",min_value=0.0,max_value=1.0,value=0.5,step=0.01)

        cols=st.columns([2,2,2,2,2,1])
        with cols[0]:
            search_cfg['result_limits']['s2']=st.number_input("S2 Result Limit",value=5,min_value=0,step=1)
        with cols[1]:
            search_cfg['result_limits']['arxiv']=st.number_input("Arxiv Result Limit",value=3,min_value=0,step=1)
        with cols[2]:
            search_cfg['result_limits']['pwc']=st.number_input("Papers w/ Code Result Limit",value=3,min_value=0,step=1)
        with cols[3]:
            search_cfg['perplexity_settings']['model_size']=st.selectbox("Perplexity Model Size",options=['none','small','large','huge'],index=2)
        with cols[4]:
            search_cfg['perplexity_settings']['max_tokens']=st.number_input("Perplexity Max Tokens",value=2000,min_value=500,step=100,disabled=search_cfg['perplexity_settings']['model_size']=='none')
        with cols[5]:
            st.write("")
            st.write("")
            prompting=st.checkbox("Prompt",value=False)

        analysis=st.text_area("Instructs to the Search Agent",placeholder='Please finds me information about ...',height=100)

    sss=evosys.agents.sss
    sss.reconfig(search_cfg,st)

    with st.sidebar:
        def display_status(name,status):
            marks={True:'✅',False:'❌'}
            st.write(f'{marks[status]} *{name}*')
        with st.expander("Service Connection",expanded=True):
            display_status('Cohere',sss.co is not None)
            display_status('Pinecone',sss.pc is not None)
            display_status('Perplexity',sss.ppl_key_set)
            display_status('Semantic Scholar',sss.s2_key_set)
    
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
            prt=sss(query,details,analysis,prompt=prompting)
            st.markdown(prt,unsafe_allow_html=True)
    else:
        st.info(f'**NOTE:** All settings here will only be applied to this playground. The playground will directly work on the selected running namespace ```{evosys.evoname}```.')
    

def library_explorer(evosys,project_dir):
    st.subheader("Library Explorer")
    
    primary_lib_index = U.read_file(U.pjoin(LIBRARY_DIR,'INDEX.md'))
    with st.expander("Primary Library Index (may look messy)",expanded=False):
        st.markdown(primary_lib_index,unsafe_allow_html=True)
    primary_lib_dir = U.pjoin(LIBRARY_DIR,'tree')
    code_dir = U.pjoin(LIBRARY_DIR,'base')
    secondary_lib_dir = U.pjoin(LIBRARY_DIR,'tree_ext','secondary')
    lib_plus_dir = U.pjoin(LIBRARY_DIR,'tree_ext','plus')

    choose_source = st.selectbox("Choose a Source",options=['Primary Library','Secondary Library','Library Plus'])
    if choose_source == 'Primary Library':
        lib_dir = primary_lib_dir
        files = []
        for file_name in os.listdir(lib_dir):
            file_name = file_name.split('.')[0]
            _code_dir = U.pjoin(code_dir,file_name,f'{file_name}_edu.py')
            if os.path.exists(_code_dir):
                files.append(file_name+' (with Code)')
            else:
                files.append(file_name)

    elif choose_source == 'Secondary Library':
        lib_dir = secondary_lib_dir
        files = os.listdir(lib_dir)
    else:
        lib_dir = lib_plus_dir
        files = os.listdir(lib_dir)
        

    choose_file = st.selectbox("Choose a File",options=files)

    if choose_source == 'Primary Library':
        choose_file = choose_file.split(' (')[0]
        file=U.load_json(U.pjoin(lib_dir,choose_file+'.json'))
        _code_dir = U.pjoin(code_dir,choose_file,f'{choose_file}_edu.py')
        if os.path.exists(_code_dir):
            with st.expander('Reference Code',expanded=False):
                st.code(file.pop('code'),language='python')
    else:
        file=U.load_json(U.pjoin(lib_dir,choose_file))
        if 'code' in file:
            file.pop('code')
    with st.expander('Metadata',expanded=True):
        st.write(file)


def unit_explorer(evosys,project_dir):
    
    st.subheader("Unit Dictionary Explorer")

    GD = evosys.ptree.GD

    cols=st.columns(2)
    with cols[0]:
        _options = []
        for i in list(GD.terms.keys()):
            if GD.terms[i].is_root:
                _options.append(f'{i} (root)')
            else:
                _options.append(i)
        choose_unit=st.selectbox("Choose a Unit",options=_options).replace(' (root)','').strip()
    with cols[1]:
        _variants=GD.terms[choose_unit].variants if choose_unit is not None else []
        choose_variant=st.selectbox("Choose a Variant",options=['None (random)']+list(_variants.keys())) if len(_variants)>0 else None
        choose_variant = None if choose_variant == 'None (random)' else choose_variant

    if choose_unit is not None:
        unit,tree_name,decl=GD.get_unit(choose_unit,choose_variant)
        st.write(f'Tree: ```{tree_name}```')
        st.write(f'Decl: ```{decl}```')
        cols=st.columns([1,1])
        with cols[0]:
            with st.expander("Code",expanded=False):
                st.code(unit.code,line_numbers=True)
        with cols[1]:
            with st.expander("Document",expanded=False):
                st.markdown(unit.spec.document)
        with st.expander("Raw data",expanded=False):
            st.write(unit)

    designs=GD.ptree.filter_by_type('DesignArtifactImplemented')
    cols = st.columns(3)
    with cols[0]:
        choose_design=st.selectbox("Choose a Design",options=list(designs))
    with cols[1]:
        if choose_design is not None:
            artifact=GD.ptree.get_node(choose_design)
            impl_history=artifact.implementation.history
        else:
            impl_history=[]
        select_attempt=st.selectbox("Choose an Attempt",options=list(range(len(impl_history))))
    with cols[2]:
        rounds=impl_history[select_attempt].rounds if len(impl_history)>0 else []
        select_round=st.selectbox("Choose a Round",options=list(range(len(rounds))))

    if select_round is not None:
        with st.expander("Round Details",expanded=False):
            st.write(impl_history[select_attempt].rounds[select_round])



def proposal_explorer(evosys,project_dir):
    
    sss=evosys.agents.sss
    st.subheader("Design Proposal Explorer")
    select_design=st.selectbox("Choose a Design Proposal available for search",options=['None']+list(sss.design_proposals.keys()))
    select_design = None if select_design == 'None' else select_design
    
    if select_design is not None:
        with st.expander("Proposal Details",expanded=True):
            st.markdown(sss.design_proposals[select_design],unsafe_allow_html=True)



def explorers(evosys,project_dir):

    with st.sidebar:
        mode=st.selectbox("Choose an Explorer",options=['Unit Explorer','Proposal Explorer','Library Explorer'],index=0)

    if mode=='Library Explorer':
        library_explorer(evosys,project_dir)
    elif mode=='Unit Explorer':
        unit_explorer(evosys,project_dir)
    else:
        proposal_explorer(evosys,project_dir)
    

_embeddding_models = {
    'OpenAI':OPENAI_EMBEDDING_MODELS,
    'Cohere':COHERE_EMBEDDING_MODELS,
    'Together':TOGETHER_EMBEDDING_MODELS,
}

def units_search(evosys,project_dir):
    sss=evosys.agents.sss

    with st.sidebar:
        st.success(f'Number of units: ```{len(sss.unit_codes)}```')

    st.subheader("Units Search Engine")
    cfg_backup = copy.deepcopy(sss.cfg)

    _cfg = sss.cfg
    _unit_search_cfg = sss.unit_search_cfg
    _unit_embedding_model = sss.embedding_models['unitcode']
    _unit_embedding_distance = sss.embedding_distances['unitcode']

    with st.expander("Search Configurations",expanded=True):
        cols = st.columns([1,1.5,1,1.5,1])
        with cols[0]:
            _unit_search_cfg['top_k']=st.number_input("Top K",value=_unit_search_cfg['top_k'],min_value=1,step=1)
        with cols[1]:
            _unit_search_cfg['cutoff']=st.slider("Cutoff",min_value=0.0,max_value=1.0,value=_unit_search_cfg['cutoff'],step=0.01)
        with cols[2]:
            if _unit_embedding_model in _embeddding_models['OpenAI']:
                embedding_model_type = 'OpenAI'
            elif _unit_embedding_model in _embeddding_models['Cohere']:
                embedding_model_type = 'Cohere'
            elif _unit_embedding_model in _embeddding_models['Together']:
                embedding_model_type = 'Together'
            _model_types = list(_embeddding_models.keys())
            embedding_model_type = st.selectbox("Embedding Model Type",options=_model_types,index=_model_types.index(embedding_model_type))
        with cols[3]:
            _index=_embeddding_models[embedding_model_type].index(_unit_embedding_model) if _unit_embedding_model in _embeddding_models[embedding_model_type] else 0
            _unit_embedding_model=st.selectbox("Embedding Model",options=_embeddding_models[embedding_model_type],index=_index)
        with cols[4]:
            embedding_distances = [i.value for i in EmbeddingDistance]
            _unit_embedding_distance=st.selectbox("Embedding Distance",options=embedding_distances,index=embedding_distances.index(_unit_embedding_distance))
    _cfg['unit_search'] = _unit_search_cfg
    _cfg['embedding_models']['unitcode'] = _unit_embedding_model
    _cfg['embedding_distances']['unitcode'] = _unit_embedding_distance

    query=st.text_area("Unit code query (you can paste here)")
    search_proposal_btn=st.button("Search Unit Code")

    if search_proposal_btn:
        with st.spinner('Searching...'):
            sss.reconfig(_cfg,st)
            _,prt=sss.query_units_by_code(query)
            sss.reconfig(cfg_backup,st)
            st.markdown(prt,unsafe_allow_html=True)
    else:
        st.info(f'**NOTE:** All settings here will only be applied to this playground. The playground will directly work on the selected running namespace ```{evosys.evoname}```.')


def proposal_search(evosys,project_dir):
    sss=evosys.agents.sss
    cfg_backup = copy.deepcopy(sss.cfg)

    with st.sidebar:
        st.success(f'Number of proposals: ```{len(sss.design_proposals)}```')

    _cfg = sss.cfg
    _proposal_search_cfg = sss.proposal_search_cfg
    _proposal_embedding_model = sss.embedding_models['proposal']
    _proposal_embedding_distance = sss.embedding_distances['proposal']

    with st.expander("Search Configurations",expanded=True):
        cols = st.columns([1,1,1.5,1,1.3,1])
        with cols[0]:
            _proposal_search_cfg['top_k']=st.number_input("Top K",value=_proposal_search_cfg['top_k'],min_value=0,step=1)
        with cols[1]:
            _proposal_search_cfg['sibling']=st.number_input("Sibling Top K",value=_proposal_search_cfg['sibling'],min_value=0,step=1)
        with cols[2]:
            _proposal_search_cfg['cutoff']=st.slider("Cutoff",min_value=0.0,max_value=1.0,value=_proposal_search_cfg['cutoff'],step=0.01)
        with cols[3]:
            if _proposal_embedding_model in _embeddding_models['OpenAI']:
                embedding_model_type = 'OpenAI'
            elif _proposal_embedding_model in _embeddding_models['Cohere']:
                embedding_model_type = 'Cohere'
            elif _proposal_embedding_model in _embeddding_models['Together']:
                embedding_model_type = 'Together'
            _model_types = list(_embeddding_models.keys())
            embedding_model_type = st.selectbox("Embedding Model Type",options=_model_types,index=_model_types.index(embedding_model_type))
        with cols[4]:
            _index=_embeddding_models[embedding_model_type].index(_proposal_embedding_model) if _proposal_embedding_model in _embeddding_models[embedding_model_type] else 0
            _proposal_embedding_model=st.selectbox("Embedding Model",options=_embeddding_models[embedding_model_type],index=_index)
        with cols[5]:
            embedding_distances = [i.value for i in EmbeddingDistance]
            _proposal_embedding_distance=st.selectbox("Embedding Distance",options=embedding_distances,index=embedding_distances.index(_proposal_embedding_distance))
    _cfg['proposal_search'] = _proposal_search_cfg
    _cfg['embedding_models']['proposal'] = _proposal_embedding_model
    _cfg['embedding_distances']['proposal'] = _proposal_embedding_distance

    st.subheader("Proposal Search Engine")
    query=st.text_area("Proposal query (you can paste here)")
    search_proposal_btn=st.button("Search Proposal")

    if search_proposal_btn:
        with st.spinner('Searching...'):
            sss.reconfig(_cfg,st)
            _,prt=sss._search_designs(query)
            sss.reconfig(cfg_backup,st)
            st.markdown(prt,unsafe_allow_html=True)
    else:
        st.info(f'**NOTE:** All settings here will only be applied to this playground. The playground will directly work on the selected running namespace ```{evosys.evoname}```.')


def search(evosys,project_dir):

    st.header('Search Engine Playground')

    with st.sidebar:
        AU.running_status(st,evosys)

        # mode=st.radio('Playground Options',options=['Paper','Units','Proposal','Explorer'],index=0)
        mode=st.selectbox("Playground Options",options=['Paper Search','Units Search','Proposal Search','Explorers'])


    if mode=='Paper Search':
        paper_search(evosys,project_dir)
    elif mode=='Units Search':
        units_search(evosys,project_dir)
    elif mode=='Proposal Search':
        proposal_search(evosys,project_dir)
    else:
        explorers(evosys,project_dir)
        

