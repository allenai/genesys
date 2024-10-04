import json
import time
import pathlib
import streamlit as st
import sys,os
import inspect
import pyflowchart as pfc
import uuid
import copy
import streamlit.components.v1 as components
import shutil
import functools as ft

sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU
import numpy as np

from model_discovery.agents.flow.gau_flows import DesignModes,RunningModes
from model_discovery.evolution import DEFAULT_PARAMS,DEFAULT_N_SOURCES
from model_discovery.agents.roles.selector import DEFAULT_SEED_DIST,SCHEDULER_OPTIONS
from model_discovery.system import DEFAULT_AGENTS,DEFAULT_MAX_ATTEMPTS,DEFAULT_TERMINATION,\
    DEFAULT_THRESHOLD,DEFAULT_SEARCH_SETTINGS,DEFAULT_NUM_SAMPLES,DEFAULT_MODE,DEFAULT_UNITTEST_PASS_REQUIRED
from model_discovery.agents.search_utils import DEFAULT_SEARCH_LIMITS,DEFAULT_RERANK_RATIO,\
    DEFAULT_PERPLEXITY_SETTINGS,DEFAULT_PROPOSAL_SEARCH_CFG

TARGET_SCALES = ['14M','31M','70M','125M','350M','760M','1300M']
SELECT_METHODS = ['random']
VERIFY_STRATEGY = ['random']


    

def apply_config(evosys,config):
    if config['params']['evoname']!=evosys.params['evoname']:
        evosys.switch_ckpt(config['params']['evoname'],load_params=False)
    evosys.reconfig(design_cfg=config['design_cfg'],select_cfg=config['select_cfg'],search_cfg=config['search_cfg'])
    evosys.reload(config['params'])
    U.save_json(config,U.pjoin(evosys.evo_dir,'config.json'))
    st.toast(f"Applied and saved config in {evosys.evo_dir}")


def apply_env_vars(evosys,env_vars):
    changed=False
    for k,v in env_vars.items():
        if v:
            if k in os.environ and os.environ[k]==v:
                continue
            st.toast(f"Applied change to environment variable {k}")
            os.environ[k]=v
            changed=True
    if changed:
        st.toast("Reloading...") # neede to manually reload the evosys
    else:
        st.toast("No changes to environment variables")

    return changed

def apply_select_config(evosys,select_cfg):
    with st.spinner('Applying and saving select config...'):
        evosys.reconfig(select_cfg=select_cfg)
        st.toast("Applied and saved select config")

def apply_design_config(evosys,design_cfg):
    with st.spinner('Applying and saving design config...'):
        evosys.reconfig(design_cfg=design_cfg)
        st.toast("Applied and saved design config")

def apply_search_config(evosys,search_cfg):
    with st.spinner('Applying and saving search config...'):    
        evosys.reconfig(search_cfg=search_cfg)
        st.toast("Applied and saved search config")

def design_config(evosys):

    st.subheader("Model Design Engine Settings")

    design_cfg=copy.deepcopy(evosys.design_cfg)
    select_cfg=copy.deepcopy(evosys.select_cfg)
    search_cfg=copy.deepcopy(evosys.search_cfg)

    select_method=select_cfg.get('select_method','random')
    verify_strategy=select_cfg.get('verify_strategy','random')
    n_sources=select_cfg.get('n_sources',DEFAULT_N_SOURCES)
    seed_dist=select_cfg.get('seed_dist',DEFAULT_SEED_DIST)


    design_cfg['max_attemps']=U.safe_get_cfg_dict(design_cfg,'max_attemps',DEFAULT_MAX_ATTEMPTS)
    design_cfg['agent_types']=U.safe_get_cfg_dict(design_cfg,'agent_types',DEFAULT_AGENTS)
    design_cfg['termination']=U.safe_get_cfg_dict(design_cfg,'termination',DEFAULT_TERMINATION)
    design_cfg['threshold']=U.safe_get_cfg_dict(design_cfg,'threshold',DEFAULT_THRESHOLD)
    design_cfg['search_settings']=U.safe_get_cfg_dict(design_cfg,'search_settings',DEFAULT_SEARCH_SETTINGS)
    design_cfg['running_mode']=RunningModes(design_cfg.get('running_mode',DEFAULT_MODE))
    design_cfg['num_samples']=U.safe_get_cfg_dict(design_cfg,'num_samples',DEFAULT_NUM_SAMPLES)
    design_cfg['unittest_pass_required']=design_cfg.get('unittest_pass_required',DEFAULT_UNITTEST_PASS_REQUIRED)
    
    search_cfg['result_limits']=U.safe_get_cfg_dict(search_cfg,'result_limits',DEFAULT_SEARCH_LIMITS)
    search_cfg['rerank_ratio']=search_cfg.get('rerank_ratio',DEFAULT_RERANK_RATIO)
    search_cfg['perplexity_settings']=U.safe_get_cfg_dict(search_cfg,'perplexity_settings',DEFAULT_PERPLEXITY_SETTINGS)
    search_cfg['proposal_search_cfg']=U.safe_get_cfg_dict(search_cfg,'proposal_search',DEFAULT_PROPOSAL_SEARCH_CFG)
    
    #### Configure design
    
    with st.expander(f"Node Selector Configurations for ```{evosys.evoname}```",expanded=False,icon='🌱'):
        
        _col1,_col2=st.columns([2,3])
        with _col1:
            st.write('###### Configure Selector')
            cols=st.columns(2)
            with cols[0]:
                select_cfg['select_method']=st.selectbox('Select Method',options=SELECT_METHODS,index=SELECT_METHODS.index(select_method))
            with cols[1]:
                select_cfg['verify_strategy']=st.selectbox('Verify Strategy',options=VERIFY_STRATEGY,index=VERIFY_STRATEGY.index(verify_strategy))
        with _col2:
            st.write('###### Configure *Seed* Selection Distribution')
            cols = st.columns(3)
            with cols[0]:
                seed_dist['scheduler'] = st.selectbox('Scheduler',options=SCHEDULER_OPTIONS,index=SCHEDULER_OPTIONS.index(seed_dist['scheduler']))
            with cols[1]:
                seed_dist['restart_prob'] = st.slider('Restart Probability',min_value=0.0,max_value=1.0,step=0.01,value=DEFAULT_SEED_DIST['restart_prob'])
            with cols[2]:
                seed_dist['warmup_rounds'] = st.number_input('Warmup Rounds',min_value=0,value=seed_dist['warmup_rounds'])

        sources={i:len(evosys.ptree.filter_by_type(i)) for i in DEFAULT_N_SOURCES}
        
        st.markdown("###### Configure the number of *references* from each source")
        cols = st.columns(len(sources))
        mode=evosys.design_cfg.get('mode',DesignModes.MUTATION.value)
        for i,source in enumerate(sources):
            with cols[i]:
                if source in ['DesignArtifact','DesignArtifactImplemented']:
                    n_sources[source] = st.number_input(label=f'{source}',min_value=0,value=n_sources[source])#,disabled=True)
                else:
                    n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=n_sources[source],max_value=sources[source])#,disabled=True)
        st.write('***Note:** 1. :red[ReferenceCoreWithTree] and :blue[DesignArtifactImplemented] are seed types. 2. Only considering **Mutation Mode** for now.*') # as mutation mode has the highest marginal effect and the basis of all modes 
        select_cfg['n_sources']=n_sources
        select_cfg['seed_dist']=seed_dist

        st.button("Save and Apply",key='save_select_config',on_click=apply_select_config,args=(evosys,select_cfg),disabled=st.session_state.listening_mode or st.session_state.evo_running)   



    with st.expander(f"Design Agent Configurations for ```{evosys.evoname}```",expanded=False,icon='🎨'):

        col1, col2 = st.columns([1, 5])
        with col1:
            mode = st.selectbox(label="Design Mode",options=[i.value for i in DesignModes],disabled=True)
        with col2:
            # st.markdown("#### Configure the base models for each agent")
            AGENT_TYPES = ['claude3.5_sonnet','gpt4o_0806','gpt4o_mini']
            agent_type_labels = {
                'DESIGN_PROPOSER':'Proposal Agent',
                'PROPOSAL_REVIEWER':'Proposal Reviewer',
                'IMPLEMENTATION_PLANNER':'Implementation Planner',
                'IMPLEMENTATION_CODER':'Implementation Coder',
                'IMPLEMENTATION_OBSERVER':'Implementation Observer',
                'SEARCH_ASSISTANT': '*Separate Search Assistant*'
            }
            agent_types = {}
            cols = st.columns(len(agent_type_labels))
            for i,agent in enumerate(agent_type_labels):
                with cols[i]:
                    index=0 
                    options=AGENT_TYPES
                    if agent in ['SEARCH_ASSISTANT']:
                        options=AGENT_TYPES+['None']
                        index=len(options)-1
                    elif agent in ['IMPLEMENTATION_OBSERVER']:
                        options=AGENT_TYPES+['o1_preview','o1_mini','None']
                        index=len(options)-2
                    elif agent in ['IMPLEMENTATION_CODER']:
                        options=['o1_preview','o1_mini']
                        index=len(options)-1
                    elif agent in ['DESIGN_PROPOSER','PROPOSAL_REVIEWER','IMPLEMENTATION_PLANNER']: 
                        options=AGENT_TYPES+['o1_preview','o1_mini']
                        if agent in ['IMPLEMENTATION_CODER']:
                            index=len(options)-1
                        else:
                            index=len(options)-1
                    agent_types[agent] = st.selectbox(label=agent_type_labels[agent],options=options,index=index,disabled=agent=='SEARCH_ASSISTANT')
            design_cfg['agent_types'] = agent_types

        if mode!=DesignModes.MUTATION.value:
            st.toast("WARNING!!!: Only mutation mode is supported now. Other modes are not stable or unimplemented.")

        col1,col2=st.columns([3,2])
        termination={}
        threshold={}
        max_attempts = {}
        with col1:
            st.markdown("##### Configure termination conditions and budgets")
            cols=st.columns(4)
            with cols[0]:
                termination['max_failed_rounds'] = st.number_input(label="Max failed rounds (0 is no limit)",min_value=1,value=3)
            with cols[1]:
                termination['max_total_budget'] = st.number_input(label="Max total budget (0 is no limit)",min_value=0,value=0)
            with cols[2]:
                termination['max_debug_budget'] = st.number_input(label="Max debug budget (0 is no limit)",min_value=0,value=0)
            with cols[3]:
                max_attempts['max_search_rounds'] = st.number_input(label="Max search rounds",min_value=0,value=4)
        with col2:
            st.markdown("##### Configure the threshold for rating the design")
            cols=st.columns(2)
            with cols[0]:
                threshold['proposal_rating'] = st.slider(label="Proposal rating",min_value=0,max_value=5,value=4)
            with cols[1]:
                threshold['implementation_rating'] = st.slider(label="Implementation rating",min_value=0,max_value=5,value=3)
        design_cfg['termination'] = termination
        design_cfg['threshold'] = threshold 


        col1,col2,col3=st.columns([4,5,2])
        with col1:
            st.markdown("##### Configure max number of attempts")
            cols=st.columns(3)
            with cols[0]:
                max_attempts['design_proposal'] = st.number_input(label="Max proposal attempts",min_value=3,value=5)
            with cols[1]:
                max_attempts['implementation_debug'] = st.number_input(label="Max debug attempts",min_value=3,value=5)
            with cols[2]:
                max_attempts['post_refinement'] = st.number_input(label="Max post refinements",min_value=0,value=0)
        design_cfg['max_attempts'] = max_attempts
        with col2:
            num_samples={}
            st.markdown("##### Configure number of samples")
            cols=st.columns(3)
            with cols[0]:
                num_samples['proposal']=st.number_input(label="Proposal Samples",min_value=1,value=1)
            with cols[1]:
                num_samples['implementation']=st.number_input(label="Implementation Samples",min_value=1,value=1)
            with cols[2]:
                rerank_methods=['random','rating']
                num_samples['rerank_method']=st.selectbox(label="Rerank Method",options=rerank_methods,index=rerank_methods.index('rating'),disabled=True)
        design_cfg['num_samples']=num_samples
        with col3:
            st.markdown("##### Configure unittests")
            st.write('')
            design_cfg['unittest_pass_required']=st.checkbox('Require passing unittests',value=design_cfg['unittest_pass_required'])

        st.button("Save and Apply",key='save_design_config',on_click=apply_design_config,args=(evosys,design_cfg),disabled=st.session_state.listening_mode or st.session_state.evo_running)   



    with st.expander(f"Search Engine Configurations for ```{evosys.evoname}```",expanded=False,icon='🔎'):
        search_cfg={}
        search_cfg['result_limits']={}
        search_cfg['perplexity_settings']={}
        search_cfg['proposal_search_cfg']={}

        cols=st.columns([2,2,2,3,2,3])
        with cols[0]:
            search_cfg['result_limits']['lib']=st.number_input("Library Primary",value=5,min_value=0,step=1)
        with cols[1]:
            search_cfg['result_limits']['lib2']=st.number_input("Library Secondary",value=0,min_value=0,step=1,disabled=True)
        with cols[2]:
            search_cfg['result_limits']['libp']=st.number_input("Library Plus",value=0,min_value=0,step=1,disabled=True)
        with cols[3]:
            search_cfg['rerank_ratio']=st.slider("Rerank Scale Ratio (0 means no rerank)",min_value=0.0,max_value=1.0,value=0.2,step=0.01)
        with cols[4]:
            search_cfg['proposal_search_cfg']['top_k']=st.number_input("Proposal Top K",value=3,min_value=0,step=1)
        with cols[5]:
            search_cfg['proposal_search_cfg']['cutoff']=st.slider("Proposal Search Cutoff",min_value=0.0,max_value=1.0,value=0.5,step=0.01)

        cols=st.columns([2,2,2,2,2])
        with cols[0]:
            search_cfg['result_limits']['s2']=st.number_input("S2 Search Result Limit",value=5,min_value=0,step=1)
        with cols[1]:
            search_cfg['result_limits']['arxiv']=st.number_input("Arxiv Search Result Limit",value=3,min_value=0,step=1)
        with cols[2]:
            search_cfg['result_limits']['pwc']=st.number_input("Papers With Code Search Result Limit",value=3,min_value=0,step=1)
        with cols[3]:
            search_cfg['perplexity_settings']['model_size']=st.selectbox("Perplexity Model Size",options=['none','small','large','huge'],index=2)
        with cols[4]:
            search_cfg['perplexity_settings']['max_tokens']=st.number_input("Perplexity Max Tokens",value=4000,min_value=500,step=100,disabled=search_cfg['perplexity_settings']['model_size']=='none')
       
        st.button("Save and Apply",key='save_search_config',on_click=apply_search_config,args=(evosys,search_cfg),disabled=st.session_state.listening_mode or st.session_state.evo_running)


    with st.expander(f"Check Configurations for ```{evosys.evoname}``` (Missing parts will apply default)",expanded=False,icon='🔧'):
        col1,col2,col3=st.columns(3)
        with col1:
            st.write("**Check Select Config:**")
            st.write(evosys.select_cfg)
        with col2:
            st.write("**Check Design Config:**")
            st.write(evosys.design_cfg)
        with col3:
            st.write("**Check Search Config:**")
            st.write(evosys.search_cfg)


def upload_exp_to_db(evosys,exp,config=None):
    collection=evosys.ptree.remote_db.collection('experiments')
    to_set={}
    if config:
        to_set['config']=config
    if len(to_set)>0:
        collection.document(exp).set(to_set,merge=True)

def delete_exp_from_db(evosys,exp):
    collection=evosys.ptree.remote_db.collection('experiments')
    collection.document(exp).delete()
    st.toast(f"Deleted experiment from remote DB: {exp}")

def sync_exps_to_db(evosys):
    for exp in os.listdir(evosys.ckpt_dir):
        config=U.load_json(U.pjoin(evosys.ckpt_dir,exp,'config.json'))
        upload_exp_to_db(evosys,exp,config)
    st.toast("Synced all experiments to remote DB")


def download_exp_from_db(evosys,exp):
    collection=evosys.ptree.remote_db.collection('experiments')
    doc=collection.document(exp).get()
    if doc.exists:
        doc_id=doc.id
        doc=doc.to_dict()
        config=doc.get('config',{})
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'ve'))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'db','sessions'))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'db','designs'))
        if config:
            U.save_json(config,U.pjoin(evosys.ckpt_dir,doc_id,'config.json'))

def sync_exps_from_db(evosys):
    collection=evosys.ptree.remote_db.collection('experiments')
    docs=collection.get()
    for doc in docs:
        doc_id=doc.id
        doc=doc.to_dict()
        config=doc.get('config',{})
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'ve'))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'db','sessions'))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'db','designs'))
        if config:
            U.save_json(config,U.pjoin(evosys.ckpt_dir,doc_id,'config.json'))
    st.toast("Synced all experiments from remote DB")
    st.rerun()


def evosys_config(evosys):

    st.subheader("Evolution System Settings")

    config=U.load_json(U.pjoin(evosys.evo_dir,'config.json'))

    with st.expander("Evolution Settings",expanded=False,icon='🧬'):
        with st.form("Evolution System Config"):
            _params={}
            col1,col2=st.columns(2)
            with col1:
                _params['evoname']=st.text_input('Experiment Namespace',value=evosys.params['evoname'])
                target_scale=st.select_slider('Target Scale',options=TARGET_SCALES,value=evosys.params['scales'].split(',')[-1])
                scales=[]
                for s in TARGET_SCALES:
                    if int(target_scale.replace('M',''))>=int(s.replace('M','')):
                        scales.append(s)
                _params['scales']=','.join(scales)
                _params['selection_ratio']=st.slider('Selection Ratio',min_value=0.0,max_value=1.0,value=evosys.params['selection_ratio'])
                _params['design_budget']=st.number_input('Design Budget ($)',value=evosys.params['design_budget'],min_value=0,step=100)
                _params['use_remote_db']=st.checkbox('Use Remote DB (Required for distributed evolution)',value=evosys.params['use_remote_db'])

            with col2:
                st.write("Current Settings:")
                settings={}
                settings['Experiment Directory']=evosys.evo_dir
                # settings['Seed Selection Method']=evosys.select_method
                # settings['Verification Strategy']=evosys.verify_strategy
                settings['Design Budget']=evosys.design_budget_limit if evosys.design_budget_limit>0 else '♾️'
                settings['Verification Budges']=evosys.selector.verify_budget
                settings['Use Remote DB']=evosys.params['use_remote_db']
                st.write(settings)

            if st.form_submit_button("Apply and Save"):
                with st.spinner("Applying and saving..."):
                    config['params']=_params
                    if config['params']['evoname']!=evosys.params['evoname']:
                        evosys.switch_ckpt(config['params']['evoname'],load_params=False)
                    evosys.reload(config['params'])
                    U.save_json(config,U.pjoin(evosys.evo_dir,'config.json'))
                    st.toast(f"Applied and saved params in {evosys.evo_dir}")

    with st.expander("Verification Engine Settings",expanded=False,icon='⚙️'):
        st.write("**TODO**")

    

def config(evosys,project_dir):

    st.title("Experiment Management")

    if st.session_state.listening_mode:
        st.warning("**NOTE:** You are running in listening mode. You cannot modify the system configuration.")

    if st.session_state.evo_running:
        st.warning("**NOTE:** Evolution system is running. You cannot modify the system configuration while the system is running.")


    st.subheader("Environment Settings")

    env_vars={}
    with st.expander("Environment Variables (Leave blank to use default)",icon='🔑'):
        with st.form("Environment Variables"):
            col1,col2,col3,col4=st.columns(4)
            with col1:
                env_vars['CKPT_DIR']=st.text_input('CKPT_DIR (No need to change)',value=os.environ.get("CKPT_DIR"))
                env_vars['DATA_DIR']=st.text_input('DATA_DIR (No need to change)',value=os.environ.get("DATA_DIR"))
                env_vars['S2_API_KEY']=st.text_input('S2_API_KEY',type='password')
            with col2:
                env_vars['WANDB_API_KEY']=st.text_input('WANDB_API_KEY (Required for Training)',type='password')
                env_vars['PINECONE_API_KEY']=st.text_input('PINECONE_API_KEY',type='password')
                env_vars['HF_KEY']=st.text_input('HUGGINGFACE_API_KEY',type='password')
            with col3:
                env_vars['MY_OPENAI_KEY']=st.text_input('OPENAI_API_KEY',type='password')
                env_vars['ANTHROPIC_API_KEY']=st.text_input('ANTHROPIC_API_KEY',type='password')
                env_vars['COHERE_API_KEY']=st.text_input('COHERE_API_KEY',type='password')
            with col4:
                env_vars['PERPLEXITY_API_KEY']=st.text_input('PERPLEXITY_API_KEY',type='password')
                env_vars['MATHPIX_API_ID']=st.text_input('MATHPIX_API_ID (Optional)',type='password')
                st.write("")
                st.write("")
                if st.form_submit_button("Apply"):
                    changed=apply_env_vars(evosys,env_vars)
                    if changed:
                        evosys.reload()
    
    evosys_config(evosys)
    design_config(evosys)
    

    col1,col2,col3,_=st.columns([1.2,1,1,3])
    with col1:
        st.subheader("Existing Experiments")
    with col2:
        if st.button("*Upload to Remote DB*",use_container_width=True,disabled=evosys.ptree.remote_db is None or st.session_state.listening_mode or st.session_state.evo_running):
            sync_exps_to_db(evosys)
    with col3:
        if st.button("*Download from Remote DB*",use_container_width=True,disabled=evosys.ptree.remote_db is None or st.session_state.listening_mode or st.session_state.evo_running):
            sync_exps_from_db(evosys)
    
    def delete_exp(dir,evoname):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        st.toast(f"Deleted directory: {dir}")
        if evosys.ptree.remote_db:
            delete_exp_from_db(evosys,evoname)
        # st.rerun()
    
    def switch_dir(evoname):
        evosys.switch_ckpt(evoname)
        st.toast(f"Switched to {evoname}")


    experiments={}
    for ckpt in os.listdir(evosys.ckpt_dir):
        exp_dir=U.pjoin(evosys.ckpt_dir,ckpt)
        experiment={}
        experiment['namespace']=ckpt
        ckpt_config=U.load_json(U.pjoin(exp_dir,'config.json'))
        if not ckpt_config: continue
        if 'params' not in ckpt_config: ckpt_config['params']={}
        ckpt_config['params']=U.init_dict(ckpt_config['params'],DEFAULT_PARAMS)
        experiment['selection_ratio']=ckpt_config['params']['selection_ratio']
        verify_budget={}
        budget=1
        for scale in ckpt_config['params']['scales'].split(',')[::-1]:
            verify_budget[scale]=int(np.ceil(budget))
            budget/=ckpt_config['params']['selection_ratio']
        experiment['remaining_budget']=verify_budget
        U.mkdir(U.pjoin(exp_dir,'db','sessions'))
        U.mkdir(U.pjoin(exp_dir,'db','designs'))
        experiment['created_sessions']=len(os.listdir(U.pjoin(exp_dir,'db','sessions')))
        experiment['sampled_designs']=len(os.listdir(U.pjoin(exp_dir,'db','designs')))
        experiment['use_remote_db']=ckpt_config.get('params',{}).get('use_remote_db',False)
        if exp_dir==evosys.evo_dir:
            experiment['ICON']='🏠'
            experiment['BUTTON']=[('Current Directory',None,True)]
            experiments[ckpt+' (Current)']=experiment
        else:
            experiment['BUTTON']=[
                ('Delete',ft.partial(delete_exp,exp_dir,ckpt),st.session_state.listening_mode or st.session_state.evo_running),
                ('Switch',ft.partial(switch_dir,ckpt),st.session_state.listening_mode or st.session_state.evo_running)
            ]
            experiments[ckpt]=experiment

    if len(experiments)>0:
        AU.grid_view(st,experiments,per_row=3,spacing=0.05)
    else:
        st.info("No experiments found in the current directory. You may download from remote DB")



    ################### Side bar ###################

    with st.sidebar:

        AU.running_status(st,evosys)

        def dump_config(_evosys):
            _config=U.load_json(U.pjoin(_evosys.evo_dir,'config.json'))
            _config['select_cfg']=_evosys.select_cfg
            _config['design_cfg']=_evosys.design_cfg
            if 'running_mode' in _config['design_cfg']:
                if not isinstance(_config['design_cfg']['running_mode'],str):
                    _config['design_cfg']['running_mode'] = _config['design_cfg']['running_mode'].value
            _config['search_cfg']=_evosys.search_cfg
            return json.dumps(_config,indent=4)

        st.download_button(
            label="Download your config",
            data=dump_config(evosys),
            file_name="config.json",
            mime="text/json",
            use_container_width=True
        )

        uploaded_file = st.file_uploader(
            "Upload your config",
            type=['json'],
            accept_multiple_files=False,
            # use_container_width=True
        )

        if uploaded_file is not None:
            uploaded_config = json.load(uploaded_file)
            with st.expander("Loaded Config",expanded=False):
                st.write(uploaded_config)
            st.button("Apply Uplaoded Config",on_click=apply_config,args=(evosys,uploaded_config,),disabled=st.session_state.listening_mode or st.session_state.evo_running)


# if __name__ == "__main__":



