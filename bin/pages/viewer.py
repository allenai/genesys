import json
import time
import pathlib
import streamlit as st
import sys,os
import inspect
import pyflowchart as pfc
import streamlit.components.v1 as components
from subprocess import check_output
from streamlit_markmap import markmap
from streamlit_timeline import timeline
from enum import Enum
import pandas as pd

sys.path.append('.')
from model_discovery.agents.flow.alang import DialogTreeViewer
import model_discovery.utils as U


from model_discovery.evolution import EvolutionSystem

from model_discovery.agents.flow._legacy_gau_flows import GUFlowScratch
from model_discovery.agents.flow._legacy_naive_flows import design_flow_definition,review_naive,design_naive
from model_discovery.agents.flow.gau_flows import GUFlowMutation
from model_discovery.agents.flow.alang import AgentDialogFlowNaive,ALangCompiler
from model_discovery.model.library.tester import check_tune
from model_discovery.evolution import DesignArtifact
import bin.app_utils as AU

class ViewModes(Enum):
    DESIGNS = 'Design Artifacts'
    EXPERIMENTS = 'Experiments'
    DIALOGS = 'Agent Dialogs (Experimental)'
    FLOW = 'Agent Flows (Experimental)'



def load_db(db_dir):
    artifacts={}
    if not U.pexists(U.pjoin(db_dir,'designs')):
        return artifacts
    for id in os.listdir(U.pjoin(db_dir,'designs')):
        artifact = DesignArtifact.load(U.pjoin(db_dir,'designs',id))
        artifacts[artifact.acronym]=artifact
    return artifacts





def _view_designs(evosys,design_artifacts,selected_design):

    st.title('Design Artifact Viewer')

    # st.header('14M Training Results')
    csv_res_dir=U.pjoin(evosys.evo_dir,'..','..','notebooks','all_acc_14M.csv')
    csv_res_norm_dir=U.pjoin(evosys.evo_dir,'..','..','notebooks','all_acc_14M_norm.csv')
    df=pd.read_csv(csv_res_dir)
    df_norm=pd.read_csv(csv_res_norm_dir)

    col1,col2=st.columns(2)
    with col1:
        st.markdown('### Raw Results on 14M')
        st.dataframe(df)
    with col2:
        st.markdown('### Relative to Random (%)')
        st.dataframe(df_norm)

    if design_artifacts:
        design=design_artifacts[selected_design]
        st.subheader(f'Proposal for {selected_design}')
        with st.expander('View Proposal'):
            st.markdown(design.proposal.proposal)
        with st.expander('View Review'):
            st.markdown(design.proposal.review)
            st.write('#### Rating: ',design.proposal.rating,'out of 5')
        if design.implementation:
            st.subheader(f'GAU Tree for {selected_design}')
            with st.expander('Click to expand'):
                itree=design.implementation.implementation
                st.write(itree.view()[0],unsafe_allow_html=True)
            gab_code=check_tune('14M',design.acronym,code=itree.compose(),skip_tune=True,reformat_only=True)
            st.subheader('Exported GAB Code')
            with st.expander('Click to expand'):
                st.code(gab_code,language='python')
        else:
            st.warning('The design has not been implemented yet.')
    else:
        st.warning('No design artifacts found in the experiment directory')


def _view_experiments(evosys):
    st.title('Experiment Viewer')
    st.write(evosys.evo_dir)


def _view_dialogs(evosys):
    st.title('ALang Dialog Viewer (Experimental)')
    sess_dir = U.pjoin(evosys.evo_dir, 'db', 'sessions')

    if not os.path.exists(sess_dir):
        st.warning("No dialogs found in the log directory")
    else:
        dialogs = {}
        for d in os.listdir(sess_dir):
            dialogs[d] = DialogTreeViewer(U.pjoin(sess_dir, d,'log'))

        if not dialogs:
            st.warning("No dialogs found in the log directory")
        else:
            selected_dialog = st.selectbox("Select a dialog", list(dialogs.keys()))
            dialog = dialogs[selected_dialog]
            markmap(dialog.to_markmap(),height=300)
            selected_thread = st.selectbox("Select a thread", list(dialog.threads.keys()))
            thread = dialog.threads[selected_thread]
            timeline(thread.to_timeline(),height=800)



def _view_flows(evosys,selected_flow,flow):

            
    st.title('ALang Design Flow Viewer (Experimental)')
    st.subheader(f'Viewing: {selected_flow}')


    simple_mode = 'VIEW_FLOWCHART_SIMPLE' 
    if simple_mode not in st.session_state:
        st.session_state[simple_mode] = True

    with st.sidebar:
        st.write('Viewer Setting')
        if st.button('View Simplified Chart',use_container_width=True):
            st.session_state[simple_mode] = True
            st.rerun()
        if st.button('View Full Chart',use_container_width=True):
            st.session_state[simple_mode] = False
            st.rerun()


    simple_mode = st.session_state[simple_mode] # True

    flow,script = flow

    if simple_mode:
        col1, col2 = st.columns([2,1])
        with col1:
            selected_id = flow.export(800,simplify=simple_mode)

        with col2:
            nodes=flow.nodes
            if selected_id:
                node_id=int(selected_id)
                if node_id in nodes:
                    node=nodes[node_id]
                    if node:
                        source=node.inspect()
                        st.markdown(f'### Selected Node ID {node_id}: {node.alias}')
                        st.code(source)
                    else:
                        st.markdown(f'### Node ID {node_id} does not have a source.')
                else:
                    st.markdown('### Select a code and view source here.')
            else:
                st.markdown('### Select a code and view source here.')
    else:
        flow.export(800)


    st.markdown('## ALang Design Flow Source')
    st.write('Automatically reformatted by compiler')
    st.code(script,line_numbers=True,language='bash')

    # st.markdown('## Naive Control Flow Viewer')

    # col1, col2 = st.columns(2)

    # with col1:
    #     if st.button('Click here to view the Flow Chart of a Naive Design Flow'):
    #         fc_dir = naive.export(evo_dir)
    #         check_output("start " + fc_dir, shell=True)
    #     st.code(inspect.getsource(naive.prog))

    # with col2:
    #     if st.button('Click here to view the Flow Chart of a Naive Review Flow'):
    #         fc_dir = system.review_flow.export(evo_dir)
    #         check_output("start " + fc_dir, shell=True)
    #     st.code(inspect.getsource(review_flow.prog))

    # components.html(open(fc_dir).read(),height=800,scrolling=True)




def viewer(evosys,project_dir):
    
    

    ### Sidebar
    with st.sidebar:
        view_mode = st.selectbox("View Mode", list([i.value for i in ViewModes]))
        view_mode = ViewModes(view_mode)
        if view_mode == ViewModes.FLOW:
            # Lagacy flows for development
            DESIGN_ALANG =design_flow_definition()
            design_flow,DESIGN_ALANG_reformatted=ALangCompiler().compile(DESIGN_ALANG,design_flow_definition,reformat=True)
            
            design_flow_naive=AgentDialogFlowNaive('Model Design Flow',design_naive)
            review_flow_naive=AgentDialogFlowNaive('Model Review Flow',review_naive)
            # gu_flow_scratch = GUFlowScratch(system,None,None)
            # gu_flow_mutation = GUFlowMutation(system,None,None,'',{})

            flows={
                # 'GU Flow (Scratch) (Legacy)':gu_flow_scratch,
                # 'GU Flow (Mutation)':gu_flow_mutation,
                'Naive Design Flow (Legacy)':(design_flow,DESIGN_ALANG_reformatted),
                # 'Naive Review Flow':(review_flow_naive,''),
            }
            selected_flow = st.selectbox("Select a flow", list(flows.keys()))
            flow = flows[selected_flow]
        elif view_mode == ViewModes.DESIGNS:
            ckpts=os.listdir(evosys.ckpt_dir)
            selected_ckpt=st.selectbox('Select a experiment directory',ckpts)
            db_dir=U.pjoin(evosys.ckpt_dir,selected_ckpt,'db')
            design_artifacts = load_db(db_dir)
            selected_design = st.selectbox("Select a design", list(design_artifacts.keys()))
        elif view_mode == ViewModes.DIALOGS:
            pass


    if view_mode == ViewModes.DESIGNS:
        _view_designs(evosys,design_artifacts,selected_design)

    elif view_mode == ViewModes.EXPERIMENTS:
        _view_experiments(evosys)

    elif view_mode == ViewModes.DIALOGS:
        _view_dialogs(evosys)
            
    elif view_mode == ViewModes.FLOW:
        _view_flows(evosys,selected_flow,flow)