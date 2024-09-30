import json
import time
import pathlib
import streamlit as st
import sys,os
import inspect
import pyflowchart as pfc
import uuid
import streamlit.components.v1 as components
import shutil
import functools as ft

sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.agents.flow.gau_flows import DesignModes,RunningModes

TARGET_SCALES = ['14M','31M','70M','125M','350M','760M','1300M']


def apply_config(evosys,config,params_only=False):
    apply_env_vars(evosys,config['env_vars'])
    if config['params']['evoname']!=evosys.params['evoname']:
        evosys.switch_ckpt(config['params']['evoname'])
    if not params_only:
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
    evosys.reconfig(select_cfg=select_cfg)

def apply_design_config(evosys,design_cfg):
    evosys.reconfig(design_cfg=design_cfg)

def apply_search_config(evosys,search_cfg):
    evosys.reconfig(search_cfg=search_cfg)

def design_config(evosys):

    st.subheader("Model Design Engine Settings")

    design_cfg=evosys.design_cfg
    select_cfg=evosys.select_cfg
    search_cfg=evosys.search_cfg

    n_sources=select_cfg.get('n_sources',{})
    
    #### Configure design
    
    with st.expander(f"Node Selector Configurations for ```{evosys.evoname}```",expanded=False,icon='🌱'):

        sources = ['ReferenceCoreWithTree', 'DesignArtifactImplemented', 'DesignArtifact', 'ReferenceCore', 'ReferenceWithCode', 'Reference']
        sources={i:len(evosys.ptree.filter_by_type(i)) for i in sources}
        
        st.markdown("##### Configure the number of seeds to sample from each source")
        cols = st.columns(len(sources))
        mode=evosys.design_cfg.get('mode',DesignModes.MUTATION.value)
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
        if mode==DesignModes.MUTATION.value:
            st.write('**ReferenceCoreWithTree and DesignArtifactImplemented are seed types in MUTATION mode. Will randomly sample one from samples from them as seed.*')
        select_cfg['n_sources']=n_sources

        st.button("Save and Apply",key='save_select_config',on_click=apply_select_config,args=(evosys,select_cfg))   



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
                    elif agent in ['IMPLEMENTATION_CODER','DESIGN_PROPOSER','PROPOSAL_REVIEWER','IMPLEMENTATION_PLANNER']: 
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


        col1,col2=st.columns([4,5])
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

        st.button("Save and Apply",key='save_design_config',on_click=apply_design_config,args=(evosys,design_cfg,select_cfg))   



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
       
        st.button("Save and Apply",key='save_search_config',on_click=apply_search_config,args=(evosys,search_cfg))


    with st.expander(f"Check Configurations for ```{evosys.evoname}``` (Empty means using default)",expanded=False,icon='🔧'):
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


def upload_exp_to_db(evosys,exp,config=None,state=None):
    collection=evosys.ptree.remote_db.collection('experiments')
    to_set={}
    if config:
        to_set['config']=config
    if state:
        to_set['state']=state
    if len(to_set)>0:
        collection.document(exp).set(to_set,merge=True)

def delete_exp_from_db(evosys,exp):
    collection=evosys.ptree.remote_db.collection('experiments')
    collection.document(exp).delete()

def sync_exps_to_db(evosys):
    for exp in os.listdir(evosys.ckpt_dir):
        config=U.load_json(U.pjoin(evosys.ckpt_dir,exp,'config.json'))
        state=U.load_json(U.pjoin(evosys.ckpt_dir,exp,'state.json'))
        if not state: continue
        upload_exp_to_db(evosys,exp,config,state)
    st.toast("Synced all experiments to remote DB")


def download_exp_from_db(evosys,exp):
    collection=evosys.ptree.remote_db.collection('experiments')
    doc=collection.document(exp).get()
    if doc.exists:
        doc_id=doc.id
        doc=doc.to_dict()
        config=doc.get('config',{})
        state=doc.get('state')
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'db','sessions'))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'db','designs'))
        if config:
            U.save_json(config,U.pjoin(evosys.ckpt_dir,doc_id,'config.json'))
        if state:
            U.save_json(state,U.pjoin(evosys.ckpt_dir,doc_id,'state.json'))

def sync_exps_from_db(evosys):
    collection=evosys.ptree.remote_db.collection('experiments')
    docs=collection.get()
    for doc in docs:
        doc_id=doc.id
        doc=doc.to_dict()
        config=doc.get('config',{})
        state=doc.get('state')
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'db','sessions'))
        U.mkdir(U.pjoin(evosys.ckpt_dir,doc_id,'db','designs'))
        if config:
            U.save_json(config,U.pjoin(evosys.ckpt_dir,doc_id,'config.json'))
        if state:
            U.save_json(state,U.pjoin(evosys.ckpt_dir,doc_id,'state.json'))
    st.toast("Synced all experiments from remote DB")
    st.rerun()


def config(evosys,project_dir):

    st.title("Experiment Management")


    config={}

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
    
    config['env_vars']=env_vars


    st.subheader("Evolution System Settings")

    SELECT_METHODS = ['random']
    

    
    with st.expander("Evolution Settings",expanded=False,icon='🧬'):
        with st.form("Evolution System Config"):
            params={}
            col1,col2=st.columns(2)
            with col1:
                params['evoname']=st.text_input('Experiment Namespace',value=evosys.params['evoname'])
                target_scale=st.select_slider('Target Scale',options=TARGET_SCALES,value=evosys.params['scales'].split(',')[-1])
                scales=[]
                for s in TARGET_SCALES:
                    if int(target_scale.replace('M',''))>=int(s.replace('M','')):
                        scales.append(s)
                params['scales']=','.join(scales)
                params['selection_ratio']=st.slider('Selection Ratio',min_value=0.0,max_value=1.0,value=evosys.params['selection_ratio'])
                params['select_method']=st.selectbox('Seed Selection Method',options=SELECT_METHODS,index=SELECT_METHODS.index(evosys.params['select_method']))
                params['design_budget']=st.number_input('Design Budget ($)',value=evosys.params['design_budget'],min_value=0,step=100)
                params['use_remote_db']=st.checkbox('Use Remote DB (Required for distributed evolution)',value=evosys.params['use_remote_db'])
            config['params']=params

            with col2:
                st.write("Current Settings:")
                settings={}
                settings['Experiment Directory']=evosys.evo_dir
                settings['Seed Selection Method']=evosys.select_method
                settings['Design Budget']=evosys.design_budget_limit if evosys.design_budget_limit>0 else '♾️'
                settings['Verification Budges']=evosys.state['budgets']
                settings['Use Remote DB']=evosys.params['use_remote_db']
                st.write(settings)


            st.form_submit_button("Apply and Save",on_click=apply_config,args=(evosys,config,True))

    with st.expander("Verification Engine Settings",expanded=False,icon='⚙️'):
        st.write("**TODO**")

    

    design_config(evosys)
    

    col1,col2,col3,_=st.columns([1.2,1,1,3])
    with col1:
        st.subheader("Existing Experiments")
    with col2:
        if st.button("*Upload to Remote DB*",use_container_width=True,disabled=evosys.ptree.remote_db is None):
            sync_exps_to_db(evosys)
    with col3:
        if st.button("*Download from Remote DB*",use_container_width=True,disabled=evosys.ptree.remote_db is None):
            sync_exps_from_db(evosys)
    
    def delete_dir(dir):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        st.toast(f"Deleted directory: {dir}")
        if evosys.ptree.remote_db:
            delete_exp_from_db(evosys,dir)
        # st.rerun()
    
    def switch_dir(dir):
        evosys.switch_ckpt(dir)
        st.toast(f"Switched to {dir}")

    experiments={}
    for ckpt in os.listdir(evosys.ckpt_dir):
        exp_dir=U.pjoin(evosys.ckpt_dir,ckpt)
        experiment={}
        experiment['namespace']=ckpt
        if not U.pexists(U.pjoin(exp_dir,'state.json')):
            continue
        state=U.load_json(U.pjoin(exp_dir,'state.json'))
        config=U.load_json(U.pjoin(exp_dir,'config.json'))
        experiment['selection_ratio']=state['selection_ratio']
        experiment['remaining_budget']=state['budgets']
        experiment['created_sessions']=len(os.listdir(U.pjoin(exp_dir,'db','sessions')))
        experiment['sampled_designs']=len(os.listdir(U.pjoin(exp_dir,'db','designs')))
        experiment['use_remote_db']=config.get('params',{}).get('use_remote_db',False)
        if exp_dir==evosys.evo_dir:
            experiment['ICON']='🏠'
            experiment['BUTTON']=[('Current Directory',None,True)]
            experiments[ckpt+' (Current)']=experiment
        else:
            experiment['BUTTON']=[
                ('Delete',ft.partial(delete_dir,exp_dir),False),
                ('Switch',ft.partial(switch_dir,ckpt),False)
            ]
            experiments[ckpt]=experiment

    if len(experiments)>0:
        AU.grid_view(st,experiments,per_row=3,spacing=0.05)
    else:
        st.write("No experiments found.")



    ################### Side bar ###################

    with st.sidebar:

        AU.running_status(st,evosys)
        _config=U.load_json(U.pjoin(evosys.evo_dir,'config.json'))
        _config['select_cfg']=evosys.select_cfg
        _config['design_cfg']=evosys.design_cfg
        _config['search_cfg']=evosys.search_cfg
        st.download_button(
            label="Download your config",
            data=json.dumps(_config,indent=4),
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
            st.button("Apply Uplaoded Config",on_click=apply_config,args=(evosys,uploaded_config,))