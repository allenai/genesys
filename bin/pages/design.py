import json
import time
import pathlib
import datetime
import streamlit as st
import sys,os


sys.path.append('.')
import model_discovery.utils as U
from model_discovery.agents.flow.gau_flows import EndReasons,END_REASONS_LABELS



def stat_logs(logs):
    end_reasons = {}
    end_labels = {}
    for reason in EndReasons:
        end_reasons[reason] = 0
    for _,label in END_REASONS_LABELS.items():
        end_labels[label] = 0
    for log in logs:
        unfinished = True
        for _,text,type in log:
            if type=='end':
                unfinished=False
                end_reasons[text] += 1
        if unfinished:
            end_reasons[EndReasons.UNFINISHED] += 1
    
    end_ratios = {i:num/sum(end_reasons.values()) for i,num in end_reasons.items()}
    for i in end_reasons:
        end_labels[END_REASONS_LABELS[i]] += end_reasons[i]
    end_label_ratios = {i:num/sum(end_labels.values()) for i,num in end_labels.items()}


    col1,col2=st.columns([1,1])
    
    with col1:
        st.write('###### Total number of runs: ',sum(end_reasons.values()))
        st.write('###### Success rate')
        for reason in end_label_ratios:
            count = end_labels[reason]
            ratio = end_label_ratios[reason]
            st.write(f"{reason}: {count} ({ratio*100:.2f}%)")
            
    with col2:
        st.write('###### End reasons')
        for reason in end_ratios:
            count = end_reasons[reason]
            ratio = end_ratios[reason]
            st.write(f"{reason.value}: {count} ({ratio*100:.2f}%)")
    


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
    design_cfg = {}
    n_sources = {}

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
        design_cfg['agent_types'] = agent_types


    if mode!='Design from existing design':
        st.header("WARNING!!!: Design from scratch has not been updated, it may not work as expected.")


    sources = ['ReferenceCoreWithTree', 'DesignArtifact', 'ReferenceCore', 'ReferenceWithCode', 'Reference', 'Reference1hop']
    sources={i:len(evosys.ptree.filter_by_type(i)) for i in sources}
    
    st.markdown("##### Configure the number of seeds to sample from each source")
    cols = st.columns(len(sources))
    for i,source in enumerate(sources):
        with cols[i]:
            if mode=='Design from existing design' and source=='ReferenceCoreWithTree':
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=1,value=1,max_value=1,disabled=True)
            else:
                init_value=min(1,sources[source])
                n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=init_value,max_value=sources[source])


    col1,col2=st.columns([3,2])
    termination={}
    threshold={}
    with col1:
        st.markdown("##### Configure termination conditions (0 means no limit)")
        cols=st.columns(3)
        with cols[0]:
            termination['max_failed_rounds'] = st.number_input(label="Max failed rounds",min_value=1,value=3)
        with cols[1]:
            termination['max_total_budget'] = st.number_input(label="Max total budget",min_value=0,value=0)
        with cols[2]:
            termination['max_debug_budget'] = st.number_input(label="Max debug budget",min_value=0,value=2)
    with col2:
        st.markdown("##### Configure the threshold for rating the design")
        cols=st.columns(2)
        with cols[0]:
            threshold['proposal_rating'] = st.slider(label="Proposal rating",min_value=0,max_value=5,value=4)
        with cols[1]:
            threshold['implementation_rating'] = st.slider(label="Implementation rating",min_value=0,max_value=5,value=3)
    design_cfg['termination'] = termination
    design_cfg['threshold'] = threshold 


    col1,col2=st.columns([3,5])
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
        cols=st.columns([2,2,3])
        with cols[0]:
            save_folder_name = st.text_input(label="Experiment folder name",value='default')
            save_folder=U.pjoin(evosys.evo_dir,'log',save_folder_name)
        with cols[1]:
            folders=['']
            for i in os.listdir(U.pjoin(evosys.evo_dir,'log')):
                if os.path.isdir(U.pjoin(evosys.evo_dir,'log',i)):
                    folders.append(i)
            selected_folder = st.selectbox(label="**View folder stats**",options=folders)
            selected_folder_dir = U.pjoin(evosys.evo_dir,'log',selected_folder)
        with cols[2]:
            designs=['']
            if selected_folder and os.path.exists(selected_folder_dir):
                designs += os.listdir(selected_folder_dir)
            folder_name = selected_folder if selected_folder else 'No folder selected'
            selected_design = st.selectbox(label=f"***View runs in selected folder: {folder_name}***",options=designs)
    design_cfg['max_attempts'] = max_attempts


    cols = st.columns([7,3,2])
    with cols[0]:
        instruction = st.text_input(label = "Add any additional instructions (optional)" )
    with cols[1]:
        EXPERIMENT_RUNS = st.number_input(label="Number of design runs",min_value=1,value=1)
    with cols[2]:
        st.text('')
        st.text('')
        submit = st.button(label="*Run designs*", use_container_width=True)



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
    
    elif selected_design:
        with st.empty():
            st.markdown(f'### Viewing design log for session: *{selected_design}*...')
        log=eval(U.read_file(U.pjoin(save_folder,selected_design,'stream.log')))
        show_log(log)
    
    elif selected_folder:
        with st.empty():
            st.subheader(f'Viewing statistics for experiment folder: *{selected_folder}*')
        if not U.pexists(selected_folder_dir):
            st.write('No runs in this folder.')
        else:
            logs=[]
            for session in os.listdir(selected_folder_dir):
                log=eval(U.read_file(U.pjoin(selected_folder_dir,session,'stream.log')))
                logs.append(log)
            stat_logs(logs)
