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
import bin.app_utils as AU

class ViewModes(Enum):
    DESIGNS = 'Design Artifacts'
    DIALOGS = 'Agent Dialogs'
    FLOW = 'Agent Flows'



def viewer(evosys,project_dir):
    
    system=evosys.rnd_agent
    ptree=evosys.ptree

    ### build the system 
    # st.title("Viewers")

    # with st.sidebar:
        # logo_png = AU.square_logo("DES", "ART")
        # st.image(logo_png, use_column_width=True)

    
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
        'Naive Design Flow':design_flow_naive,
        'Naive Review Flow':review_flow_naive,
    }


    ### Sidebar
    with st.sidebar:
        view_mode = st.selectbox("View Mode", list([i.value for i in ViewModes]))
        view_mode = ViewModes(view_mode)
        if view_mode == ViewModes.FLOW:
            selected_flow = st.selectbox("Select a flow", list(flows.keys()))
            flow = flows[selected_flow]
        elif view_mode == ViewModes.DESIGNS:
            design_artifacts = evosys.ptree.filter_by_type(['DesignArtifact','DesignArtifactImplemented'])
            selected_design = st.selectbox("Select a design", design_artifacts)
        elif view_mode == ViewModes.DIALOGS:
            log_dir = U.pjoin(evosys.evo_dir, 'log')
            dialogs = {}
            for d in os.listdir(log_dir):
                dialogs[d] = DialogTreeViewer(U.pjoin(log_dir, d))

    if view_mode == ViewModes.DESIGNS:

        
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

        design=ptree.get_node(selected_design)

        st.subheader(f'Proposal for {selected_design}')
        with st.expander('View Proposal'):
            st.markdown(design.proposal.proposal)
        with st.expander('View Review'):
            st.markdown(design.proposal.review)
            st.write('#### Rating: ',design.proposal.rating,'out of 5')
        st.subheader(f'GAU Tree for {selected_design}')
        with st.expander('Click to expand'):
            itree=design.implementation.implementation
            st.write(itree.view()[0],unsafe_allow_html=True)
        gab_code=check_tune('14M',design.acronym,code=itree.compose(),skip_tune=True,reformat_only=True)
        st.subheader('Exported GAB Code')
        with st.expander('Click to expand'):
            st.code(gab_code,language='python')


    elif view_mode == ViewModes.DIALOGS:
        st.title('ALang Dialog Viewer')
        sess_dir = U.pjoin(evosys.evo_dir, 'db', 'sessions')
        dialogs = {}
        for d in os.listdir(sess_dir):
            dialogs[d] = DialogTreeViewer(U.pjoin(sess_dir, d))

        if not dialogs:
            st.warning("No dialogs found in the log directory")
        else:
            selected_dialog = st.selectbox("Select a dialog", list(dialogs.keys()))
            dialog = dialogs[selected_dialog]
            markmap(dialog.to_markmap(),height=300)
            selected_thread = st.selectbox("Select a thread", list(dialog.threads.keys()))
            thread = dialog.threads[selected_thread]
            timeline(thread.to_timeline(),height=800)

        with st.sidebar:
            st.write("Empty sidebar")
            

    elif view_mode == ViewModes.FLOW:
            
        st.title('ALang Design Flow Viewer')

        system=evosys.rnd_agent
        evo_dir=evosys.evo_dir

        simple_mode = 'VIEW_FLOWCHART_SIMPLE' 
        if simple_mode not in st.session_state:
            st.session_state[simple_mode] = True

        with st.sidebar:
            if st.button('View Simplified Flowchart'):
                st.session_state[simple_mode] = True
                st.rerun()
            if st.button('View Full Flowchart'):
                st.session_state[simple_mode] = False
                st.rerun()


        simple_mode = st.session_state[simple_mode] # True

        # flow=system.design_flow
        # script=system.DESIGN_ALANG_reformatted

        # flow = flow.flow
        script = flow.script

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

        st.markdown('## Naive Control Flow Viewer')

        col1, col2 = st.columns(2)

        with col1:
            if st.button('Click here to view the Flow Chart of a Naive Design Flow'):
                fc_dir = system.design_flow_naive.export(evo_dir)
                check_output("start " + fc_dir, shell=True)
            st.code(inspect.getsource(system.design_flow_naive.prog))

        with col2:
            if st.button('Click here to view the Flow Chart of a Naive Review Flow'):
                fc_dir = system.review_flow.export(evo_dir)
                check_output("start " + fc_dir, shell=True)
            st.code(inspect.getsource(system.review_flow.prog))

        # components.html(open(fc_dir).read(),height=800,scrolling=True)

