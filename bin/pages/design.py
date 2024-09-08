import json
import time
import pathlib
import datetime
import streamlit as st
import sys,os


sys.path.append('.')
import model_discovery.utils as U





def show_log(log):
    # four types: enter, exit, write, markdown
    in_status = False
    code = []
    for dt, text, type in log:
        line = []
        if in_status:
            line.append('\t')
        
        # Use repr() to get a string literal representation
        text_repr = repr(text)
        
        if type == 'enter':
            in_status = True
            line.append(f"with st.status({text_repr}):")
        elif type == 'exit':
            in_status = False
        elif type == 'write':
            line.append(f"st.write({text_repr}, unsafe_allow_html=True)")
        elif type == 'markdown':
            line.append(f"st.markdown({text_repr}, unsafe_allow_html=True)")
        
        code.append(''.join(line))
    
    final_code = '\n'.join(code)
    exec(final_code)





def design(evosys,project_dir):

    system = evosys.rnd_agent

    ### build the system 
    st.title("Model Design Engine")

    ### side bar 
    st.sidebar.button("reset design query")

    col1, col2 = st.columns([1, 3])
    with col1:
        # st.markdown("#### Configure design mode")
        mode = st.selectbox(label="Design Mode",options=['Design from existing design','Design from scratch (unstable)'])
    with col2:
        # st.markdown("#### Configure the base models for each agent")
        AGENT_TYPES = ['claude3.5_sonnet','gpt4o_0806','gpt4o_mini']
        agent_type_labels = {
            'DESIGN_PROPOSER':'Proposal Agent',
            'PROPOSAL_REVIEWER':'Proposal Reviewer',
            'DESIGN_IMPLEMENTER':'Implementation Agent',
            'IMPLEMENTATION_REVIEWER':'Implementation Reviewer',
        }
        agent_types = {}
        cols = st.columns(4)
        for i,agent in enumerate(agent_type_labels):
            with cols[i]:
                agent_types[agent] = st.selectbox(label=agent_type_labels[agent],options=AGENT_TYPES)
        design_cfg = {'agent_types':agent_types}


    if mode!='Design from existing design':
        st.header("WARNING!!!: Design from scratch has not been updated, it may not work as expected.")


    sources = ['ReferenceCoreWithTree', 'DesignArtifact', 'ReferenceCore', 'ReferenceWithCode', 'Reference', 'Reference1hop']
    sources={i:len(evosys.ptree.filter_by_type(i)) for i in sources}
    n_sources = {}
    
    st.markdown("##### Configure the number of seeds to sample from each source")
    cols = st.columns(len(sources))
    for i,source in enumerate(sources):
        with cols[i]:
            if mode=='Design from existing design' and source=='ReferenceCoreWithTree':
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=1,value=1,max_value=1,disabled=True)
            else:
                init_value=min(1,sources[source])
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=init_value,max_value=sources[source])


    col1,col2=st.columns([4,5])
    max_attempts = {}
    with col1:
        st.markdown("##### Configure max number of attempts")
        cols=st.columns(3)
        with cols[0]:
            max_attempts['design_proposal'] = st.number_input(label="Max proposal attempts",min_value=3,value=10)
        with cols[1]:
            max_attempts['implementation_debug'] = st.number_input(label="Max debug attempts",min_value=3,value=7)
        with cols[2]:
            max_attempts['post_refinement'] = st.number_input(label="Max post refinements",min_value=0,value=5)
    with col2:
        st.markdown("##### Configure experiment settings")
        cols=st.columns([1,2,1])
        with cols[0]:
            save_folder_name = st.text_input(label="Experiment folder name",value='default')
            save_folder=U.pjoin(evosys.evo_dir,'log',save_folder_name)
        with cols[1]:
            designs=[]
            if os.path.exists(save_folder):
                designs = os.listdir(save_folder)
            designs.insert(0,'')
            selected_design = st.selectbox(label=f"View previous runs in folder *{save_folder_name}*",options=designs)
        with cols[2]:
            st.text('')
            st.text('')
            view_stat = st.button(label="View folder stats", use_container_width=True)

    cols = st.columns([5,2,2])
    with cols[0]:
        instruction = st.text_input(label = "Add any additional instructions (optional)" )
    with cols[1]:
        EXPERIMENT_RUNS = st.number_input(label="Number of design runs",min_value=1,value=1)
    with cols[2]:
        st.text('')
        st.text('')
        submit = st.button(label="*Run designs*", use_container_width=True)


    design_cfg['max_attempts'] = max_attempts

    if submit:
        for i in range(EXPERIMENT_RUNS):
            with st.empty():
                st.subheader(f'Running {EXPERIMENT_RUNS} designs, save to *{save_folder_name}*...')

            if sum(n_sources.values())==0:
                st.write("You selected no seed, the agent will randomly generate a design for you.")

            if EXPERIMENT_RUNS==1:
                spinner_text = "running model design loop"
            else:
                spinner_text = f"running model design loop ({i+1}/{EXPERIMENT_RUNS})"
            with st.spinner(text=spinner_text):
                type_abbr = {'claude3.5_sonnet':'C','gpt4o_0806':'G','gpt4o_mini':'M'}
                type_abbr = ''.join([type_abbr[i] for i in agent_types.values()])
                session_id=f'{type_abbr}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
                log_dir=U.pjoin(save_folder,session_id)
                
                _mode = 'existing' if mode=='Design from existing design' else 'scratch'
                instruct,metadata=evosys.select(n_sources,mode=_mode) # use the seed_ids to record the phylogenetic tree
            if instruction:
                instruct+=f'\n\n## Additional Instructions from the user\n\n{instruction}'
            system(instruct,frontend=True,stream=st,log_dir=log_dir,metadata=metadata,design_cfg=design_cfg)
    
    elif view_stat:
        with st.empty():
            st.subheader(f'Viewing statistics for experiment folder: *{save_folder_name}*')
    
    elif selected_design:
        with st.empty():
            st.markdown(f'### Viewing design log for session: *{selected_design}*...')
        log=eval(U.read_file(U.pjoin(save_folder,selected_design,'stream.log')))
        show_log(log)
    


