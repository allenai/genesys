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

TARGET_SCALES = ['14M','31M','70M','125M','350M','760M','1300M']


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
            

def config(evosys,project_dir):

    st.title("Experiment Management")


    config={}

    def apply_config(config):
        apply_env_vars(evosys,config['env_vars'])
        evosys.reload(config['params'])
        st.toast(f"Applied config:\n{config}")

    st.subheader("Environment Settings")

    env_vars={}
    with st.expander("Environment Variables (Leave blank to use default)"):
        with st.form("Environment Variables"):
            col1,col2,col3,col4=st.columns(4)
            with col1:
                env_vars['CKPT_DIR']=st.text_input('CKPT_DIR',value=os.environ.get("CKPT_DIR"))
                env_vars['DATA_DIR']=st.text_input('DATA_DIR',value=os.environ.get("DATA_DIR"))
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
                if st.form_submit_button("Apply"):
                    changed=apply_env_vars(evosys,env_vars)
                    if changed:
                        evosys.reload()
    
    config['env_vars']=env_vars


    st.subheader("Evolution System Settings")

    SELECT_METHODS = ['random']
    

    
    with st.expander("Experiment Settings",expanded=True):
        with st.form("Evolution System Config"):
            col1,col2=st.columns(2)
            with col1:
                params={}
                params['evoname']=st.text_input('Experiment Name',value=evosys.params['evoname'])
                target_scale=st.select_slider('Target Scale',options=TARGET_SCALES,value=evosys.params['scales'].split(',')[-1])
                scales=[]
                for s in TARGET_SCALES:
                    if int(target_scale.replace('M',''))>=int(s.replace('M','')):
                        scales.append(s)
                params['scales']=','.join(scales)
                params['selection_ratio']=st.slider('Selection Ratio',min_value=0.0,max_value=1.0,value=evosys.params['selection_ratio'])
                params['select_method']=st.selectbox('Seed Selection Method',options=SELECT_METHODS,index=SELECT_METHODS.index(evosys.params['select_method']))
                params['design_budget']=st.number_input('Design Budget ($)',value=evosys.params['design_budget'],min_value=0,step=100)
                config['params']=params
                st.session_state['EVOSYS_PARAMS']=params

            with col2:
                st.write("Current Settings:")
                settings={}
                settings['Experiment Directory']=evosys.evo_dir
                settings['Seed Selection Method']=evosys.select_method
                settings['Design Budget']=evosys.design_budget_limit if evosys.design_budget_limit>0 else '♾️'
                settings['Verification Budges']=evosys.state['budgets']
                st.write(settings)

            st.form_submit_button("Apply",on_click=apply_config,args=(config,))


    st.subheader("Existing Experiments")
    
    def delete_dir(dir):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        st.toast(f"Deleted directory: {dir}")
        # st.rerun()
    
    def switch_dir(dir):
        evosys.switch_ckpt(dir)
        st.toast(f"Switched to {dir}")

    experiments={}
    for ckpt in os.listdir(evosys.ckpt_dir):
        exp_dir=U.pjoin(evosys.ckpt_dir,ckpt)
        experiment={}
        experiment['directory']=exp_dir
        state=U.load_json(U.pjoin(exp_dir,'state.json'))
        experiment['selection_ratio']=state['selection_ratio']
        experiment['remaining_budget']=state['budgets']
        experiment['created_sessions']=len(os.listdir(U.pjoin(exp_dir,'db','sessions')))
        experiment['sampled_designs']=len(os.listdir(U.pjoin(exp_dir,'db','designs')))
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


    
    with st.sidebar:
        st.download_button(
            label="Download your config",
            data=json.dumps(config,indent=4),
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
            st.button("Apply Uplaoded Config",on_click=apply_config,args=(uploaded_config,))
