import json
import time
import pathlib
import streamlit as st
import sys,os
import inspect
import pyflowchart as pfc
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from subprocess import check_output
from streamlit_markmap import markmap
from streamlit_timeline import timeline
from enum import Enum
import pandas as pd
from streamlit_theme import st_theme

sys.path.append('.')
from model_discovery.agents.flow.alang import DialogTreeViewer
import model_discovery.utils as U


from model_discovery.agents.flow._legacy_gau_flows import GUFlowScratch
from model_discovery.agents.flow._legacy_naive_flows import design_flow_definition,review_naive,design_naive
from model_discovery.agents.flow.gau_flows import GUFlow
from model_discovery.agents.flow.alang import AgentDialogFlowNaive,ALangCompiler
from model_discovery.model.library.tester import check_tune
import bin.app_utils as AU


from model_discovery.agents.roles.selector import *
from model_discovery.agents.roles.selector import _group_results,_flat_weighted_metrics,\
    _combine_design_quadrant,_rank_combined_quadrant,_get_ranking_quadrant,_01_normed_soft_filter

from bin.pages.design import show_log,load_log


class ViewModes(Enum):
    METRICS = 'Experiment Visulizer'
    DESIGNS = 'Design Artifacts'
    SESSIONS = 'Local Session Logs'
    DIALOGS = 'Local Agent Dialogs'
    # FLOW = 'Agent Flows (Experimental)'



def _view_tree(tree):
    with st.expander('View detailed tree'):
        st.write(tree.view()[0],unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1.5])
    with col1:
        updated_state = tree.export(800,light_mode=False)

    with col2:
        selected_id=updated_state.selected_id 
        if selected_id:
            source=tree.units[selected_id].code
            st.markdown(f'### Selected Unit: {selected_id}')
            with st.container(height=750):
                st.code(source,line_numbers=True)
        else:
            st.markdown('### Select a node to view the source of the node here.')



def _view_designs(evosys):
    st.title('Design Artifact Viewer')

    
    with st.status('Loading design artifacts...'):
        design_artifacts = evosys.ptree.filter_by_type(['DesignArtifactImplemented','DesignArtifact'])
        acronyms= evosys.ptree.filter_by_type(['ReferenceCoreWithTree'])
        # corerefs_with_tree = {acronym:evosys.ptree.get_node(acronym) for acronym in acronyms}


    with st.sidebar:
        category = st.selectbox('Select a category',['Design Artifacts','Seed Trees'])
        if category == 'Design Artifacts':
            # sort by alphabetical order
            designs = sorted(list(design_artifacts))
            selected_design = st.selectbox('Select a design',designs)
        else:
            selected_coreref = st.selectbox('Select a core reference',list(acronyms))

    # # st.header('14M Training Results')
    # csv_res_dir=U.pjoin(evosys.evo_dir,'..','..','notebooks','all_acc_14M.csv')
    # csv_res_norm_dir=U.pjoin(evosys.evo_dir,'..','..','notebooks','all_acc_14M_norm.csv')
    # df=pd.read_csv(csv_res_dir)
    # df_norm=pd.read_csv(csv_res_norm_dir)

    # col1,col2=st.columns(2)
    # with col1:
    #     st.markdown('### Raw Results on 14M')
    #     st.dataframe(df)
    # with col2:
    #     st.markdown('### Relative to Random (%)')
    #     st.dataframe(df_norm)

    if category == 'Design Artifacts':
        if design_artifacts:
            design=evosys.ptree.get_node(selected_design)
            sessdata = design.sess_snapshot
            with st.expander(f'Meta Data for {selected_design}',expanded=False):
                metadata = {
                    'Design Session ID': design.sess_id,
                    'Seed IDs': design.seed_ids,
                    'References': sessdata['ref_ids'],
                    'Agents': design.proposal.design_cfg['_agent_types'],
                }
                st.json(metadata)
            st.subheader(f'Proposal for {selected_design}')
            with st.expander('View Proposal'):
                st.download_button('Download Proposal',design.proposal.proposal,file_name=f'{selected_design}_proposal.md')
                st.markdown(design.proposal.proposal)
            with st.expander('View Review'):
                st.download_button('Download Review',design.proposal.review,file_name=f'{selected_design}_review.md')
                st.markdown(design.proposal.review)
                st.write('#### Rating: ',design.proposal.rating,'out of 5')
            if design.implementation:
                st.subheader(f'GAU Tree for {selected_design}')
                itree=design.implementation.implementation
                _view_tree(itree)
                gab_code=check_tune('14M',design.acronym,code=itree.compose(),skip_tune=True,reformat_only=True)
                st.subheader('Exported GAB Code')
                with st.expander('Click to expand'):
                    st.download_button('Download GAB Code',gab_code,file_name=f'{selected_design}_gab.py')
                    st.code(gab_code,language='python')
            else:
                st.warning('The design has not been implemented yet.')
            if design.verifications:
                st.subheader('Verification Results')
                for scale in design.verifications:
                    with st.expander(f'{scale} Verification Results',expanded=True):
                        reports = design.verifications[scale].verification_report
                        if 'wandb_ids.json' in reports:
                            wandb_ids = reports['wandb_ids.json']
                            if 'pretrain' in wandb_ids:
                                wandb_id=wandb_ids['pretrain']['id']
                                wandb_name=wandb_ids['pretrain']['name']
                                project=wandb_ids['project']
                                entity=wandb_ids['entity']
                                url=f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                                st.write(f'WANDB URL: {url}')
            else:
                st.warning('No verification results found for this design.')
        else:
            st.warning('No design artifacts found in the experiment directory')
    
    elif acronyms and category == 'Seed Trees':
        coreref = evosys.ptree.get_node(selected_coreref)
        with st.expander(f'Meta Data for {selected_coreref}',expanded=False):
            metadata = {
                'Reference ID': coreref.acronym,
                'S2 ID': coreref.s2id,
                'Abstract': coreref.abstract,
                'Authors': coreref.authors,
                'Year': coreref.year,
                'Venue': coreref.venue,
                'TLDR': coreref.tldr,
                'Citation Count': coreref.citationCount,
                'Influential Citation Count': coreref.influentialCitationCount,
            }
            st.json(metadata)
        st.subheader(f'GAU Tree for {selected_coreref}')
        _view_tree(coreref.tree)
        if coreref.verifications:
            st.subheader('Verification Results')
            for scale in coreref.verifications:
                with st.expander(f'{scale} Verification Results',expanded=True):
                    for _mult in coreref.verifications[scale]:
                        reports = coreref.verifications[scale][_mult].verification_report
                        if 'wandb_ids.json' in reports:
                            wandb_ids = reports['wandb_ids.json']
                            if 'pretrain' in wandb_ids:
                                wandb_id=wandb_ids['pretrain']['id']
                                wandb_name=wandb_ids['pretrain']['name']
                                project=wandb_ids['project']
                                entity=wandb_ids['entity']
                                url=f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                                st.write(f'WANDB URL: {url}')
        else:
            st.warning('No verification results found for this core reference.')
    


def _view_dialogs(evosys):
    st.title('Agent Dialog Viewer')

    with st.sidebar:
        folders = [i for i in list(os.listdir(U.pjoin(evosys.ckpt_dir))) if not i.endswith('.json')]
        evoname = st.selectbox("Select a folder", folders)

    sess_dir = U.pjoin(evosys.ckpt_dir, evoname, 'db', 'sessions')

    if not os.path.exists(sess_dir):
        st.warning("No dialogs found in the log directory")
    else:
        dialogs = {}
        for d in os.listdir(sess_dir):
            try:
                dialogs[d] = DialogTreeViewer(U.pjoin(sess_dir, d,'log'))
            except Exception as e:
                dialogs[d+' (Failed to load)'] = str(e)

        if not dialogs:
            st.warning("No dialogs found in the log directory")
        else:
            selected_dialog = st.selectbox("Select a dialog", list(dialogs.keys()))
            dialog = dialogs[selected_dialog]
            if isinstance(dialog,str):
                st.warning('Failed to load the dialog: '+dialog)
            else:
                markmap(dialog.to_markmap(),height=300)
                selected_thread = st.selectbox("Select a thread", list(dialog.threads.keys()))
                thread = dialog.threads[selected_thread]
                timeline(thread.to_timeline(),height=800)


def _view_sessions(evosys):
    st.title('Local Session Logs')

    with st.sidebar:
        _folders = os.listdir(U.pjoin(evosys.ckpt_dir))
        folders = []
        for folder in _folders:
            if os.path.exists(U.pjoin(evosys.ckpt_dir,folder,'config.json')):
                folders.append(folder)
        evoname = st.selectbox("Select a folder", folders)

    sess_dir = U.pjoin(evosys.ckpt_dir, evoname, 'db', 'sessions')

    if not os.path.exists(sess_dir):
        st.warning("No session logs found in the log directory")
    else:
        cols = st.columns([3,3,1])
        with cols[0]:
            sessions = os.listdir(sess_dir)
            selected_session = st.selectbox("Select a session", sessions)
        with cols[1]:
            if selected_session:
                logs = os.listdir(U.pjoin(sess_dir,selected_session,'log'))
                logs = [i for i in logs if i.endswith('.log')]
                selected_log = st.selectbox("Select a log", logs)
        with cols[2]:
            st.write('')
            st.write('')
            view_btn = st.button("View",use_container_width=True,disabled=not (selected_session and selected_log))

        if view_btn:
            log = load_log(U.pjoin(sess_dir,selected_session,'log',selected_log))
            show_log(log)
    



def _view_flows(evosys,selected_flow,flow):

            
    st.title('ALang Design Flow Viewer (Experimental)')
    st.subheader(f'Viewing: {selected_flow}')


    # simple_mode = 'VIEW_FLOWCHART_SIMPLE' 
    # if simple_mode not in st.session_state:
    #     st.session_state[simple_mode] = True

    with st.sidebar:
        # st.write('Viewer Setting')
        # if st.button('View Simplified Chart',use_container_width=True):
        #     st.session_state[simple_mode] = True
        #     st.rerun()
        # if st.button('View Full Chart',use_container_width=True):
        #     st.session_state[simple_mode] = False
        #     st.rerun()
        choose_mode=st.selectbox("Choose a Mode",options=['Simplified Chart','Full Chart'],index=0)
        simple_mode = choose_mode=='Simplified Chart'


    # simple_mode = st.session_state[simple_mode] # True

    flow,script = flow

    if simple_mode:
        col1, col2 = st.columns([2,1])
        with col1:
            light_mode = True
            selected_id = flow.export(800,simplify=simple_mode,light_mode=light_mode)

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
                    st.markdown('### Select a node to view the source of the node here.')
            else:
                st.markdown('### Select a node to view the source of the node here.')
    else:
        light_mode = True # st_theme()['base']=='light'
        flow.export(800,light_mode=light_mode)


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



#############################################################################



############################################################



def dict_sub(d1,d2):
    if not d1 or not d2:
        return d1
    d1 = copy.deepcopy(d1)
    for k in d1:
        if k not in d2:
            continue
        if isinstance(d1[k],dict):
            d1[k] = dict_sub(d1[k],d2[k])
        else:
            d1[k] -= d2[k]
    return d1

def get_relative_metrics(evosys,design_vector,relative_vector):
    proposal_rating,training_metrics,eval_metrics,eval_stderrs = get_raw_metrics(design_vector)
    if not relative_vector:
        return proposal_rating,training_metrics,eval_metrics,eval_stderrs
    if isinstance(relative_vector,str):
        if relative_vector == 'random':
            _proposal_rating,_training_metrics,r_eval_metrics,r_eval_stderrs = get_random_metrics(evosys)
            _eval_metrics = {scale:r_eval_metrics for scale in eval_metrics}
            _eval_stderrs = {scale:r_eval_stderrs for scale in eval_stderrs}
        elif relative_vector == 'none':
            return proposal_rating,training_metrics,eval_metrics,eval_stderrs
        else:
            raise ValueError(f'Unknown relative vector: {relative_vector}')
    else:
        _proposal_rating,_training_metrics,_eval_metrics,_eval_stderrs = get_raw_metrics(relative_vector)
    if proposal_rating and _proposal_rating:
        proposal_rating -= _proposal_rating
    training_metrics = dict_sub(training_metrics,_training_metrics)
    eval_metrics = dict_sub(eval_metrics,_eval_metrics)
    eval_stderrs = dict_sub(eval_stderrs,_eval_stderrs)
    return proposal_rating,training_metrics,eval_metrics,eval_stderrs


def random_normalized_eval_metrics(evosys,design_vector,threshold=None):
    if threshold is None:
        threshold = evosys.selector.select_cfg.get('random_eval_threshold',DEFAULT_RANDOM_EVAL_THRESHOLD)
    _,_,_eval_metrics,_ = get_relative_metrics(evosys,design_vector,'random')
    _,_,r_eval_metrics,_ = get_random_metrics(evosys)
    normalized_eval_metrics = {}
    for scale in _eval_metrics:
        normalized_eval_metrics[scale] = {}
        for metric in _eval_metrics[scale]:
            if metric not in r_eval_metrics:
                continue
            normalized_eval_metrics[scale][metric] = {}
            for task in _eval_metrics[scale][metric]:
                if task not in r_eval_metrics[metric]:
                    continue
                _result = _eval_metrics[scale][metric][task]
                r_result = r_eval_metrics[metric][task]
                if metric.endswith(',H'):
                    if _result > r_result and (r_result == 0 or (_result - r_result)/r_result > threshold):
                        normalized_eval_metrics[scale][metric][task] = _result
                elif metric.endswith(',L'):
                    if _result < r_result and (r_result == 0 or (r_result - _result)/_result > threshold):
                        normalized_eval_metrics[scale][metric][task] = _result
    return normalized_eval_metrics
   

def _mean_eval_metrics(eval_metrics):
    mean_metrics = {}
    for scale in eval_metrics:
        mean_metrics[scale] = {}
        for metric in eval_metrics[scale]:
            result = eval_metrics[scale][metric]
            mean_metrics[scale][metric] = np.mean([result[i] for i in result])
    return mean_metrics


def design_utility_simple(select_cfg,design_vector):
    proposal_rating,_,eval_metrics,_ = get_raw_metrics(design_vector)
    scale_weighted_grouped_metrics = scale_weight_results(select_cfg,group_results(eval_metrics))
    _01_normed_metrics,_ = _flat_weighted_metrics(scale_weighted_grouped_metrics)
    return proposal_rating,np.mean(list(_01_normed_metrics.values()))

def get_random_metrics(evosys):
    eval_metrics,eval_stderrs = eval_results_getter(evosys.ptree.random_baseline)
    return None,None,eval_metrics,eval_stderrs


def show_design(evosys,design_vector,relative=None,threshold=None):
    select_cfg = evosys.selector.select_cfg
    confidence,total_points = design_confidence_simple(select_cfg,design_vector)
    confidence_percentage = confidence / total_points * 100
    proposal_rating,weighted_acc = design_utility_simple(select_cfg,design_vector)
    proposal_rating,training_metrics,eval_metrics,eval_stderrs=get_raw_metrics(design_vector)
    grouped_metrics = group_results(eval_metrics)
    scale_weighted_grouped_metrics = scale_weight_results(select_cfg,grouped_metrics)
    _01_normed_metrics,_unnormed_metrics = _flat_weighted_metrics(scale_weighted_grouped_metrics)
    
    ranking_metrics,ranking_metrics_unnormed = get_ranking_metrics(select_cfg,design_vector,False)
    cols=st.columns(2)
    with cols[0]:
        st.write(f'###### Design confidence (simple count points): ```{confidence_percentage:.2f}%``` ({confidence}/{total_points})')
        st.write(f'###### Proposal rating: ```{proposal_rating}/5.0```') if proposal_rating else st.write('Proposal rating: ```N/A```')
        st.write(f'###### Scale-Weighted mean 0-1 normed metrics: ```{weighted_acc}```')
    
    with cols[1]:
        with st.expander('**Ranking metrics (0-1 normed)**',expanded=False):
            st.json(ranking_metrics)
        with st.expander('***Ranking metrics (Unnormed)***',expanded=False):
            st.json(ranking_metrics_unnormed)

    cols=st.columns(2)
    with cols[0]:
        st.write('##### Raw metrics')
        RAW_METRICS = [
            'Training metrics',
            'Evaluation metrics',
            'Evaluation stderrs',
            'Grouped evaluation metrics',
            'Scale Weighted Grouped Evaluation Metrics',
        ]
        selected_raw_metrics = st.selectbox('Select raw metrics to show',options=RAW_METRICS)
        with st.expander(selected_raw_metrics,expanded=True):
            if selected_raw_metrics == 'Training metrics':
                st.json(training_metrics)
            elif selected_raw_metrics == 'Evaluation metrics':
                st.json(eval_metrics)
            elif selected_raw_metrics == 'Evaluation stderrs':
                st.json(eval_stderrs)
            elif selected_raw_metrics == 'Grouped evaluation metrics':
                st.json(grouped_metrics)
            elif selected_raw_metrics == 'Scale Weighted Grouped Evaluation Metrics':
                st.json(scale_weighted_grouped_metrics)

    with cols[1]:
        st.write('##### Processed metrics')
        PROCESSED_METRICS = [
            'Mean grouped evaluation metrics',
            'Random normalized evaluation metrics',
            'Relative evaluation metrics',
            'Scale Weighted Grouped 0-1 Normed Metrics',
            'Scale Weighted Grouped Unnormed Metrics',
        ]
        selected_processed_metrics = st.selectbox('Select processed metrics to show',options=PROCESSED_METRICS)
        with st.expander(selected_processed_metrics,expanded=True):
            if selected_processed_metrics == 'Mean grouped evaluation metrics':
                st.json(_mean_eval_metrics(group_results(eval_metrics)))
            elif selected_processed_metrics == 'Random normalized evaluation metrics':
                st.json(random_normalized_eval_metrics(evosys,design_vector,threshold))
            elif selected_processed_metrics == 'Relative evaluation metrics':
                if relative and relative != 'none':
                    st.json(get_relative_metrics(evosys,design_vector,relative)[2])
                else:
                    st.info('No relative metrics to show.')
            elif selected_processed_metrics == 'Scale Weighted Grouped 0-1 Normed Metrics':
                st.json(_01_normed_metrics)
            elif selected_processed_metrics == 'Scale Weighted Grouped Unnormed Metrics':
                st.json(_unnormed_metrics)




##################### Visualization #####################

def visualize_ranking_quadrant(quadrants):
    design_upper = quadrants['design_upper']
    design_lower = quadrants['design_lower']
    confidence_upper = quadrants['confidence_upper']
    confidence_lower = quadrants['confidence_lower']

    # Combine design and confidence data
    design_data = pd.concat([design_upper, design_lower])
    confidence_data = pd.concat([confidence_upper, confidence_lower])
    
    # Merge design and confidence data on 'name'
    combined_data = pd.merge(design_data, confidence_data, on='name', suffixes=('_design', '_confidence'))
    combined_data = combined_data.drop_duplicates(subset=['name'])

    fig, ax = plt.subplots(figsize=(12, 12))

    # Calculate the midpoints for x and y axes
    x_midpoint = (confidence_upper['confidence'].min() + confidence_lower['confidence'].max()) / 2
    y_midpoint = (design_upper['rating'].min() + design_lower['rating'].max()) / 2

    # Adjust the data points relative to the midpoints
    combined_data['confidence_adjusted'] = combined_data['confidence'] - x_midpoint
    combined_data['rating_adjusted'] = combined_data['rating'] - y_midpoint

    # Plot the points
    scatter = ax.scatter(combined_data['confidence_adjusted'], combined_data['rating_adjusted'], 
                         c=combined_data['rank_design'], cmap='viridis', alpha=0.7)

    # Add labels and title
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Design Rating')
    ax.set_title('Confidence vs Design Rating Quadrant')

    # Add colorbar
    plt.colorbar(scatter, label='Design Rank')

    # Add gridlines
    ax.grid(True)

    # Add quadrant labels
    ax.text(0.95, 0.95, 'Good Design\nHigh Confidence', ha='right', va='top', transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.95, 'Good Design\nLow Confidence', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    ax.text(0.95, 0.05, 'Poor Design\nHigh Confidence', ha='right', va='bottom', transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.05, 'Poor Design\nLow Confidence', ha='left', va='bottom', transform=ax.transAxes, fontsize=12)

    # Add axes at the midpoint
    ax.axhline(y=0, color='k', linestyle='--')
    ax.axvline(x=0, color='k', linestyle='--')

    # Add name labels to points
    for idx, row in combined_data.iterrows():
        ax.annotate(row['name'], (row['confidence_adjusted'], row['rating_adjusted']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    return fig
    

##################### Leaderboard #####################

LEADERBOARD_1 = [
    'blimp',
    'inverse_scaling',
    'glue',
    'qa4mre',
    'mathqa',
    'wsc273'
]

def post_process_leaderboard(leaderboard):
    # transpose the dataframe
    leaderboard = leaderboard.transpose()
    leaderboard['avg.']=leaderboard.mean(axis=1)
    return leaderboard

def export_leaderboards(evosys,design_vectors, baseline_vectors):
    _,_,eval_metrics,_ = get_random_metrics(evosys)
    random_eval_metrics = _group_results(eval_metrics)
    design_eval_metrics = {}
    baseline_eval_metrics = {}
    for mode in ['design','baseline']:
        vectors = design_vectors if mode == 'design' else baseline_vectors
        eval_metrics = design_eval_metrics if mode == 'design' else baseline_eval_metrics
        for acronym in vectors:
            _eval_metrics = get_raw_metrics(vectors[acronym])[2]
            if len(_eval_metrics) > 0:
                _eval_metrics = group_results(_eval_metrics)
                for scale in _eval_metrics:
                    if scale not in eval_metrics:
                        eval_metrics[scale] = {}
                    eval_metrics[scale][acronym] = _eval_metrics[scale]
    random_01_normed_metrics,random_unnormed_metrics=_flat_weighted_metrics(random_eval_metrics)
    leaderboards_normed={}
    leaderboards_unnormed_h={}
    leaderboards_unnormed_l={}
    for scale in design_eval_metrics:
        leaderboards_normed[scale] = pd.DataFrame(list(random_01_normed_metrics.items()), columns=['metrics', 'random']).set_index('metrics')
        leaderboards_unnormed_h[scale] = pd.DataFrame(list(random_unnormed_metrics['higher_is_better'].items()), columns=['metrics', 'random']).set_index('metrics')
        leaderboards_unnormed_l[scale] = pd.DataFrame(list(random_unnormed_metrics['lower_is_better'].items()), columns=['metrics', 'random']).set_index('metrics')
        _design_eval_metrics = design_eval_metrics[scale]
        _baseline_eval_metrics = baseline_eval_metrics.get(scale,{})
        for mode in ['baseline','design']:
            _eval_metrics = _design_eval_metrics if mode == 'design' else _baseline_eval_metrics
            for acronym in _eval_metrics:
                colname = acronym if mode == 'design' else acronym+' (baseline)'
                _01_normed_metrics,_unnormed_metrics=_flat_weighted_metrics(_eval_metrics[acronym])
                _leaderboards_normed = pd.DataFrame(list(_01_normed_metrics.items()),columns=['metrics',colname]).set_index('metrics')
                leaderboards_normed[scale] = pd.concat([leaderboards_normed[scale],_leaderboards_normed],axis=1)
                _leaderboards_unnormed_h = pd.DataFrame(list(_unnormed_metrics['higher_is_better'].items()),columns=['metrics',colname]).set_index('metrics')
                leaderboards_unnormed_h[scale] = pd.concat([leaderboards_unnormed_h[scale],_leaderboards_unnormed_h],axis=1)
                _leaderboards_unnormed_l = pd.DataFrame(list(_unnormed_metrics['lower_is_better'].items()),columns=['metrics',colname]).set_index('metrics')
                leaderboards_unnormed_l[scale] = pd.concat([leaderboards_unnormed_l[scale],_leaderboards_unnormed_l],axis=1)
    for scale in leaderboards_normed:
        leaderboards_normed[scale] = post_process_leaderboard(leaderboards_normed[scale])
        leaderboards_unnormed_h[scale] = post_process_leaderboard(leaderboards_unnormed_h[scale])
        leaderboards_unnormed_l[scale] = post_process_leaderboard(leaderboards_unnormed_l[scale])
    return leaderboards_normed,leaderboards_unnormed_h,leaderboards_unnormed_l

def leaderboard_relative(leaderboard,relative='random',filter_threshold=0):
    base_row = leaderboard.loc[relative]
    # remove the columns where the base_row is 0
    base_row = base_row[base_row != 0]
    leaderboard = leaderboard[base_row.index]
    leaderboard = (leaderboard - base_row) / base_row
    leaderboard = leaderboard.dropna(axis=1)
    leaderboard = leaderboard*100
    # filter out the columns where the max value is larger than the filter threshold
    leaderboard = leaderboard.loc[:, leaderboard.max(axis=0) >= filter_threshold]
    # add base_row back with all 0s
    # _row = pd.DataFrame(0,index=[base_row.name],columns=leaderboard.columns)
    # leaderboard = pd.concat([_row,leaderboard])
    return leaderboard.round(2)


def leaderboard_filter(leaderboard,task_filter=[]):
    if not task_filter:
        return leaderboard
    filtered_columns = [col for col in leaderboard.columns if any(task in col for task in task_filter)]
    filtered_leaderboard = leaderboard[filtered_columns]
    filtered_leaderboard = filtered_leaderboard.dropna(axis=1)
    filtered_leaderboard['avg.'] = filtered_leaderboard.mean(axis=1)
    return filtered_leaderboard

def selector_lab(evosys,project_dir):
    st.title('*Experiment Visualizer*')

    with st.status('Loading latest data...'):
        design_vectors = evosys.ptree.get_design_vectors()
        baseline_vectors = evosys.ptree.get_baseline_vectors()

    st.subheader('Real-time Leaderboard')
    with st.status('Generating leaderboard...',expanded=True):
        leaderboards_normed,leaderboards_unnormed_h,leaderboards_unnormed_l=export_leaderboards(evosys,design_vectors,baseline_vectors)
        cols = st.columns([1,1,3,1])
        with cols[0]:
            scale = st.selectbox('Select scale',options=list(sorted(leaderboards_normed.keys(),key=lambda x:U.letternum2num(x))))
        with cols[1]:
            _options = ['random']+list(baseline_vectors.keys())
            relative = st.selectbox('Relative to',options=_options)
        with cols[2]:
            input_task_filter = st.text_input('Task filter list (keywords matching, comma separated)',value=','.join(LEADERBOARD_1))
            input_task_filter=[i.strip() for i in input_task_filter.split(',')]
        with cols[3]:
            filter_threshold = st.number_input('Filter threshold (%)',min_value=0,max_value=100,step=1,value=0,
                help='Leave the metrics where there is at least one design with relative rating higher than this threshold')

    cols = st.columns(2)
    with cols[0]:
        with st.expander('Normed metrics (0-1, higher is better)',expanded=True):
            if scale is not None:
                _leaderboards_normed = leaderboard_filter(leaderboards_normed[scale],input_task_filter)
                leaderboards_normed = _leaderboards_normed.copy()
                leaderboards_normed['avg.'] = leaderboards_normed.mean(axis=1)
                st.dataframe(leaderboards_normed)
            else:
                st.info('No results available at this moment.')
    with cols[1]:
        with st.expander(f'Relative to ```{relative}``` (Normed metrics, %)',expanded=True):
            if scale is not None:
                _relative = f'{relative} (baseline)' if relative != 'random' else 'random'
                leaderboards_relative = leaderboard_relative(_leaderboards_normed,relative=_relative,filter_threshold=filter_threshold)
                leaderboards_relative['avg.'] = leaderboards_relative.mean(axis=1)
                st.dataframe(leaderboards_relative)
            else:
                st.info('No results available at this moment.')

    # filter rows
    filter_rows = st.text_input('Filter designs (exact match, comma separated)')
    if filter_rows:
        filter_rows = [i.strip() for i in filter_rows.split(',')]
        for _baseline in baseline_vectors:
            filter_rows.append(f'{_baseline} (baseline)')
        filter_rows.append('random')
    else:
        filter_rows = None
    
    highlight_color = 'violet'
    with st.expander(f'Combined leaderboard for ```{scale}``` (with relative (%) to ```{relative}```, max highlighted in :{highlight_color}[{highlight_color}])',expanded=True):
        leaderboards_normed_combined = leaderboards_normed.copy()
        leaderboards_normed_combined = leaderboards_normed_combined.loc[:,leaderboards_relative.columns]
        
        # Drop NA values from both DataFrames
        leaderboards_normed_combined = leaderboards_normed_combined.dropna()
        leaderboards_relative = leaderboards_relative.dropna()
        
        # Ensure both DataFrames have the same index after dropping NA
        common_index = leaderboards_normed_combined.index.intersection(leaderboards_relative.index)
        leaderboards_normed_combined = leaderboards_normed_combined.loc[common_index]
        leaderboards_relative = leaderboards_relative.loc[common_index]
        # recompute avg for both, remove the old avg
        leaderboards_normed_combined = leaderboards_normed_combined.drop(columns=['avg.'])
        leaderboards_relative = leaderboards_relative.drop(columns=['avg.'])
        leaderboards_normed_combined['avg.'] = leaderboards_normed_combined.mean(axis=1)
        _relative = f'{relative} (baseline)' if relative != 'random' else 'random'
        relative_avg = leaderboards_normed_combined.loc[_relative,'avg.']
        leaderboards_relative['avg.'] = 100*(leaderboards_normed_combined['avg.'] - relative_avg)/relative_avg
        
        # Combine the values of the two leaderboards as normed (relative) e.g., 3.2 (4.5%)
        def combine_values(normed, relative):
            return normed.applymap(lambda x: f'{x:.4f}') + ' (' + relative.applymap(lambda x: f'{x:.2f}%') + ')'

        leaderboards_normed_combined = combine_values(leaderboards_normed_combined, leaderboards_relative)
        if filter_rows:
            # filter out the rows not in df from filter_rows first
            filter_rows = [i for i in filter_rows if i in leaderboards_normed_combined.index]
            leaderboards_normed_combined = leaderboards_normed_combined.loc[filter_rows]


        baseline_rows = leaderboards_normed_combined[leaderboards_normed_combined.index.str.contains('(baseline)')]
        random_row = leaderboards_normed_combined.loc['random']
        remaining_rows = leaderboards_normed_combined[~leaderboards_normed_combined.index.isin(baseline_rows.index)]
        remaining_rows = remaining_rows.drop(index='random')
        remaining_rows = remaining_rows.sort_values(by='avg.',ascending=True)
        leaderboards_normed_combined = pd.concat([random_row.to_frame().T,baseline_rows,remaining_rows])
        st.dataframe(leaderboards_normed_combined.style.highlight_max(axis=0,color=highlight_color),use_container_width=True)

        # 14M
        # blimp,inverse_scaling,mathqa,qa4mre,mrpc,cola (5%)
        # adaptivespectralgau,memory_augmented_multi_head_atte,adaptivesparselinearattention,hierarchicaladaptivesparserwkv,adarmsnorm

        # 31M
        # wnli,qa4mre_2011,mrpc,sst2,rte,mathqa,inverse
        # sparsesink,ssmaugmentedmha,firelight_gpt_1,densehierarchicalrwkv_6,gpt_flash_ahsaqe,sparsesinkssmattn,sma_gatedmlp



    cols = st.columns(2)
    with cols[0]:
        with st.expander('Unnormed metrics (Higher is better)'):
            if scale is not None:
                _leaderboards_unnormed_h = leaderboard_filter(leaderboards_unnormed_h[scale],input_task_filter)
                if _leaderboards_unnormed_h.empty:
                    st.info('No data to show')
                else:
                    st.dataframe(_leaderboards_unnormed_h)
            else:
                st.info('No results available at this moment.')
    with cols[1]:
        with st.expander('Unnormed metrics (Lower is better)'):
            if scale is not None:
                _leaderboards_unnormed_l = leaderboard_filter(leaderboards_unnormed_l[scale],input_task_filter)
                if _leaderboards_unnormed_l.empty:
                    st.info('No data to show')
                else:
                    st.dataframe(_leaderboards_unnormed_l)
            else:
                st.info('No results available at this moment.')



    st.subheader('Metrics Explorer')

    # metrics_btn = st.button('Compute and show metrics and rankings',use_container_width=True)
    # if not metrics_btn:
    #     return

    cols = st.columns(3)
    with cols[0]:
        show_mode = st.selectbox('Select a category',options=['design','baseline'])

    with cols[1]:
        if show_mode == 'design':
            vectors = design_vectors
        elif show_mode == 'baseline':
            vectors = baseline_vectors
        options = []
        for design in vectors:
            if len(vectors[design]["verifications"]) > 0:
                options.append(f'{design} ({len(vectors[design]["verifications"])} verified)')
        selected_design = st.selectbox('Select a verified design',options=options)
        

    with cols[2]:
        relative = st.selectbox('Relative to',options=['none','random']+list(baseline_vectors.keys()))
        if relative == 'none':
            relative = None
        elif relative == 'random':
            relative = 'random'
        else:
            relative = baseline_vectors[relative]

    cols=st.columns(3)
    with cols[0]:
        with st.expander('Confidence points'):
            confidence_points = U.safe_get_cfg_dict(evosys.selector.select_cfg,'confidence_points',DEFAULT_CONFIDENCE_POINTS)
            st.json(confidence_points)
    with cols[1]:
        with st.expander('Random eval norm threshold'):
            threshold = st.number_input('Threshold',min_value=0.0,max_value=1.0,step=0.01,value=evosys.selector.select_cfg.get('random_eval_threshold',DEFAULT_RANDOM_EVAL_THRESHOLD))
    with cols[2]:
        with st.expander('Scale weights'):
            scale_weights = U.safe_get_cfg_dict(evosys.selector.select_cfg,'scale_weights',DEFAULT_SCALE_WEIGHTS)
            st.json(scale_weights)

    if selected_design is not None:
        selected_design = selected_design.split(' ')[0]
        design_vector=vectors[selected_design]
        show_design(evosys,design_vector,relative,threshold)
    else:
        st.info('No results available at this moment.')


    st.subheader('**Selector Ranking**',help='How the selector ranks the designs and make decisions.')

    ranking_args = U.safe_get_cfg_dict(evosys.selector.select_cfg,'ranking_args',DEFAULT_RANKING_ARGS)
    cols = st.columns([5,1,0.8,0.8])
    with cols[0]:
        _cols=st.columns([2,1])
        with _cols[0]:
            _value = ranking_args['ranking_method']
            if isinstance(_value,str):
                _value = [_value]
            ranking_args['ranking_method'] = st.multiselect('Ranking method (Required)',options=RANKING_METHODS,default=_value,
                help='Ranking method to use, if muliple methods are provided, will be aggregated by the "multi-rank merge" method')
        with _cols[1]:
            ranking_args['multi_rank_merge'] = st.selectbox('Multi-rank merge',options=MERGE_METHODS)
    with cols[1]:
        st.write('')
        normed_only = st.checkbox('Normed only',value=ranking_args['normed_only'])
    with cols[2]:
        st.write('')
        drop_zero = st.checkbox('Drop 0',value=ranking_args['drop_zero'],
            help='If set, will drop all-zero columns')
    with cols[3]:
        st.write('')
        drop_na = st.checkbox('Drop N/A',value=ranking_args['drop_na'])
    # with cols[4]:
    #     st.write('')
    #     rank_design_btn = st.button('Rank',use_container_width=True)

    with st.expander('Ranking method arguments',expanded=True):
        cols = st.columns([2,2,2,2])
        with cols[0]:
            ranking_args['draw_margin'] = st.number_input('Draw margin',min_value=0.0,max_value=1.0,step=0.001,value=ranking_args['draw_margin'], format="%0.3f",
                help='Margin for draw (tie)')
        with cols[1]:
            ranking_args['convergence_threshold'] = st.number_input('Convergence threshold',min_value=0.0,max_value=1.0,step=0.001,value=ranking_args['convergence_threshold'], format="%0.5f",
            help='Convergence threshold for iterations in methods like Markov chain')
        with cols[2]:
            ranking_args['markov_restart'] = st.number_input('Markov restart',min_value=0.0,max_value=1.0,step=0.001,value=ranking_args['markov_restart'], format="%0.3f")
        with cols[3]:
            ranking_args['metric_wise_merge'] = st.selectbox('Metric-wise merge',options=['None']+MERGE_METHODS,
                help='If set, will rank for each metric separately and then aggregate by the "metric-wise merge" method, not available for markov method')
        
        cols = st.columns([5,1,1])
        with cols[0]:
            ranking_args['soft_filter_threshold'] = st.slider('Filtering Threshold',min_value=-1.0,max_value=1.0,step=0.001,value=float(ranking_args['soft_filter_threshold']), format="%0.3f",
                help='If set, will filter out metrics with the highest difference in rating compared to a random metric lower than this, -1 (i.e. -100%) means no filtering')
        with cols[1]:
            st.write('')
            ranking_args['absolute_value_threshold'] = st.checkbox('Absolute',value=ranking_args['absolute_value_threshold'],
                help='If set, will use absolute difference instead of relative difference `difference/random` for filtering')
        with cols[2]:
            st.write('')
            ranking_args['normed_difference'] = st.checkbox('Norm Diff.',value=ranking_args['normed_difference'],
                help='If set, will use normed difference `|x-random|` instead of direct difference `x-random` for filtering')

    if len(design_vectors) == 0:
        st.info('No verified designs available at this moment.')
        return

    assert ranking_args['ranking_method'], 'Ranking method is required'
    with st.status('Generating ranking matrix...',expanded=False):
        relative_to_01_normed,_ = evosys.selector._get_random_metrics()
        ranking_matrix = get_ranking_matrix(evosys.selector.select_cfg,design_vectors,normed_only,
            drop_na=drop_na,drop_zero=drop_zero,soft_filter_threshold=ranking_args['soft_filter_threshold'],
            relative_to_01_normed=relative_to_01_normed,absolute_value_threshold=ranking_args['absolute_value_threshold'],
            normed_difference=ranking_args['normed_difference'])
        # compute average at the end of each row
        ranking_matrix = _01_normed_soft_filter(ranking_matrix,relative_to_01_normed,
            ranking_args['soft_filter_threshold'],ranking_args['absolute_value_threshold'],
            ranking_args['normed_difference'],st=st)
        _ranking_matrix = ranking_matrix.copy()
        _ranking_matrix['avg.'] = _ranking_matrix.mean(axis=1)
        # concat with random row, preserve intersection of columns in random
        relative_to_01_normed = pd.DataFrame(relative_to_01_normed,index=['random'])
        relative_to_01_normed = relative_to_01_normed[ranking_matrix.columns.intersection(relative_to_01_normed.columns)]
        _ranking_matrix = pd.concat([relative_to_01_normed,_ranking_matrix])
    
    # check if only one row (random)
    if _ranking_matrix.shape[0] == 1:
        st.info('No verified designs to rank')
        return


    with st.expander('Final ranking matrix (avg. and random will not used in ranking)',expanded=True):
        st.dataframe(_ranking_matrix)
    
    with st.status('Ranking designs...',expanded=True):
        design_rank,subranks,subsubranks = rank_designs(ranking_matrix,ranking_args)
        cols=st.columns(3)
        with cols[0]:
            st.subheader('Design Ranking')
            st.write(design_rank)
        with cols[1]:
            select_subrank = st.selectbox('Select subrank',options=subranks.keys())
            st.write(subranks[select_subrank])
        with cols[2]:
            select_subsubrank = st.selectbox('Select subsubrank',options=subsubranks[select_subrank].keys())
            if select_subsubrank:
                st.write(subsubranks[select_subrank][select_subsubrank])
            else:
                st.info('No subsubrank to show')

    st.subheader('Ranking Quadrant',help='The quadrants how selector make exploit & exploration trade-off decision for design and verify selections.')
    if design_rank.empty:
        st.info('No available designs to rank')
        return

    cols = st.columns([1,2])

    with cols[0]:
        with st.status('Ranking confidence filtered by design rank...',expanded=True):
            st.subheader('Confidence Ranking')
            confidence_rank = rank_confidences(evosys.selector.select_cfg,design_vectors,design_rank['name'].tolist())
            st.dataframe(confidence_rank)

    with cols[1]:
        with st.status('Generating ranking quadrant...',expanded=True):
            ranking_quadrants = _get_ranking_quadrant(evosys.selector.select_cfg,design_rank,confidence_rank)
            combined_quadrant = _combine_design_quadrant(ranking_quadrants)
            fig = visualize_ranking_quadrant(ranking_quadrants)
            st.pyplot(fig)

    with cols[0]:
        with st.expander('Ranked Ranking Quadrants',expanded=True):
            _rerank_method = st.selectbox('Rerank merge method',options=MERGE_METHODS)
            ranked_quadrants = _rank_combined_quadrant(combined_quadrant,_rerank_method,rename=False)
            _category = st.selectbox('Select a category',options=ranked_quadrants.keys())
            st.dataframe(ranked_quadrants[_category])
    
    # with st.expander('Exploration settings'):
    #     design_explore_args = U.safe_get_cfg_dict(evosys.selector.select_cfg,'explore_args',DEFAULT_DESIGN_EXPLORE_ARGS)
    #     verify_explore_args = U.safe_get_cfg_dict(evosys.selector.select_cfg,'verify_explore_args',DEFAULT_VERIFY_EXPLORE_ARGS)
    #     seed_dist = U.safe_get_cfg_dict(evosys.selector.select_cfg,'seed_dist',DEFAULT_SEED_DIST)

    # st.subheader('Uniform Performance improvement over time')
    # TODO




def viewer(evosys,project_dir):
    
    

    ### Sidebar
    with st.sidebar:
        AU.running_status(st,evosys)
        # if st.session_state.is_deploy:
        #     modes = [ViewModes.METRICS.value,ViewModes.DESIGNS.value,ViewModes.SESSIONS.value]
        # else:
        modes = list([i.value for i in ViewModes])
        view_mode = st.selectbox("Choose a view", modes)
        view_mode = ViewModes(view_mode)
        # if view_mode == ViewModes.FLOW:
        #     # Lagacy flows for development
        #     DESIGN_ALANG =design_flow_definition()
        #     design_flow,DESIGN_ALANG_reformatted=ALangCompiler().compile(DESIGN_ALANG,design_flow_definition,reformat=True)
            
        #     design_flow_naive=AgentDialogFlowNaive('Model Design Flow',design_naive)
        #     review_flow_naive=AgentDialogFlowNaive('Model Review Flow',review_naive)
        #     # gu_flow_scratch = GUFlowScratch(system,None,None)
        #     # gu_flow_mutation = GUFlowMutation(system,None,None,'',{})

        #     flows={
        #         # 'GU Flow (Scratch) (Legacy)':gu_flow_scratch,
        #         # 'GU Flow (Mutation)':gu_flow_mutation,
        #         'Naive Design Flow (Legacy)':(design_flow,DESIGN_ALANG_reformatted),
        #         # 'Naive Review Flow':(review_flow_naive,''),
        #     }
        #     selected_flow = st.selectbox("Select a flow", list(flows.keys()))
        #     flow = flows[selected_flow]


    if view_mode == ViewModes.DESIGNS:
        _view_designs(evosys)

    elif view_mode == ViewModes.DIALOGS:
        _view_dialogs(evosys)

    elif view_mode == ViewModes.SESSIONS:
        _view_sessions(evosys)
            
    # elif view_mode == ViewModes.FLOW:
    #     _view_flows(evosys,selected_flow,flow)

    elif view_mode == ViewModes.METRICS:
        selector_lab(evosys,project_dir)
