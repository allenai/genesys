import copy
import streamlit as st
import sys,os
import numpy as np

from model_discovery.evolution import DEFAULT_N_SOURCES

sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.agents.roles.selector import *






def design_selector(evosys,project_dir):
    st.title('Design Selector')

    seed_dist = copy.deepcopy(DEFAULT_SEED_DIST)

    with st.sidebar:
        designs = evosys.ptree.filter_by_type('DesignArtifactImplemented')
        st.success(f'Number of designs: ```{len(designs)}```')
        # select_design = st.selectbox('Select a design',options=['None']+[d.acronym for d in designs])
        # with st.expander('Selected Design',expanded=True):
        #     if select_design is not None:
        #         node=evosys.ptree.get_node(select_design)
        #         st.write(node.proposal.ratings)
        #     else:
        #         st.info('No design selected.')

    with st.expander('Design Selector basic settings',expanded=True):
        _col1,_col2=st.columns([2,3])
        with _col1:
            st.write('###### Configure Selector')
            col1, col2 = st.columns(2)
            with col1:
                n_seeds = st.number_input(label="Number of seeds",min_value=0,value=1,
                    help='Number of seed designs, it decides the mode of design, design from scratch: 0 seed, mutation: 1 seed, crossover: >=2 seeds')
            with col2:
                select_method = st.selectbox(label="Selection Method",options=SELECT_METHODS)
        with _col2:
            st.write('###### Configure *Seed* Selection Distribution')
            cols = st.columns(3)
            with cols[0]:
                seed_dist['scheduler'] = st.selectbox('Annealing Scheduler',options=SCHEDULER_OPTIONS,index=SCHEDULER_OPTIONS.index(seed_dist['scheduler']))
            with cols[1]:
                seed_dist['restart_prob'] = st.slider('Restart Probability',min_value=0.0,max_value=1.0,step=0.01,value=seed_dist['restart_prob'])
            with cols[2]:
                seed_dist['warmup_rounds'] = st.number_input('Warmup Rounds',min_value=0,value=0)

        n_sources = DEFAULT_N_SOURCES

        sources={i:len(evosys.ptree.filter_by_type(i)) for i in DEFAULT_N_SOURCES}
        st.markdown("###### Configure the number of *references* from each source")
        cols = st.columns(len(sources))
        for i,source in enumerate(sources):
            with cols[i]:
                if source in ['DesignArtifact','DesignArtifactImplemented']:
                    _label=source if source=='DesignArtifact' else 'ImplementedDesign'
                    n_sources[source] = st.number_input(label=f'{_label} ({sources[source]})',min_value=0,value=n_sources[source])#,disabled=True)
                else:
                    n_sources[source] = st.number_input(label=f'{source} ({sources[source]})',min_value=0,value=n_sources[source],max_value=sources[source])#,disabled=True)


    selector_args = {
        'n_sources': n_sources,
    }

    with st.expander(f"Selector Ranking and Exploration Settings",expanded=True):
        
        select_cfg=copy.deepcopy(evosys.selector.select_cfg)
        select_cfg['seed_dist']=seed_dist
        ranking_args = U.safe_get_cfg_dict(select_cfg,'ranking_args',DEFAULT_RANKING_ARGS)
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
            ranking_args['normed_only'] = st.checkbox('Normed only',value=ranking_args['normed_only'])
        with cols[2]:
            st.write('')
            ranking_args['drop_zero'] = st.checkbox('Drop 0',value=ranking_args['drop_zero'],
                                                    help='If set, will drop all-zero columns')
        with cols[3]:
            st.write('')
            ranking_args['drop_na'] = st.checkbox('Drop N/A',value=ranking_args['drop_na'])

        cols = st.columns(4)
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


        col1,col2=st.columns(2)
        with col1:
            st.write("##### Quadrant settings")
            cols=st.columns(3)
            quadrant_args=U.safe_get_cfg_dict(select_cfg,'quadrant_args',DEFAULT_QUADRANT_ARGS)
            with cols[0]:
                ranking_args['quadrant_merge']=st.selectbox('Quadrant Merge',options=MERGE_METHODS,index=MERGE_METHODS.index(ranking_args.get('quadrant_merge','average')))
            with cols[1]:
                quadrant_args['design_quantile']=st.number_input('Design Quantile',min_value=0.0,max_value=1.0,step=0.01,value=quadrant_args['design_quantile'])
            with cols[2]:
                quadrant_args['confidence_quantile']=st.number_input('Confidence Quantile',min_value=0.0,max_value=1.0,step=0.01,value=quadrant_args['confidence_quantile'])

        with col2:
            st.write("##### Design Exploration settings")
            cols=st.columns(3)
            design_explore_args=U.safe_get_cfg_dict(select_cfg,'design_explore_args',DEFAULT_DESIGN_EXPLORE_ARGS)
            with cols[0]:
                design_explore_args['explore_prob']=st.number_input('Design Explore Prob',min_value=0.0,max_value=1.0,step=0.01,value=design_explore_args['explore_prob'])
            with cols[1]:
                design_explore_args['scheduler']=st.selectbox('Design Scheduler',options=SCHEDULER_OPTIONS,index=SCHEDULER_OPTIONS.index(design_explore_args['scheduler']))
            with cols[2]:
                design_explore_args['background_noise']=st.number_input('Design Background Noise',min_value=0.0,max_value=1.0,step=0.01,value=design_explore_args['background_noise'])
            
        select_cfg['ranking_args']=ranking_args
        select_cfg['quadrant_args']=quadrant_args
        select_cfg['design_explore_args']=design_explore_args

    if st.button('Select'):
        
        with st.status('Selecting seeds...'):
            instruct,seeds,refs=evosys.selector.select_design(selector_args,n_seeds,select_method,select_cfg)

        st.subheader(f'**{len(seeds)} seeds selected:**')
        for seed in seeds:
            with st.expander(f'**{seed.acronym}** *({seed.type})*'):
                st.write(seed.to_prompt())

        st.subheader(f'**{len(refs)} references selected:**')
        for ref in refs:
            with st.expander(f'**{ref.acronym}** *({ref.type})*'):
                st.write(ref.to_prompt())
                
        st.subheader(f'**Instructions from the selector:**')
        if instruct:
            st.write(instruct)
        else:
            st.info('No instructions from the selector.')
    else:
        st.info(f'**NOTE:** All settings here will only be applied to this playground. The playground will directly work on the selected running namespace ```{evosys.evoname}```.')


def verify_selector(evosys,project_dir):
    st.title('Verify Selector')

    with st.sidebar:
        unverified = evosys.ptree.get_unverified_scales()
        _select_design = st.selectbox('Select a design',options=list(unverified.keys()))
        with st.expander('Unverified scales',expanded=True):
            if _select_design is not None:
                st.write(unverified[_select_design])
            else:
                st.info('No design available.')

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        verify_strategy = st.selectbox(label="Verify Strategy",options=VERIFY_STRATEGIES)
    with col2:
        st.write('')
        with st.expander('Remaining Budget'):
            st.write(evosys.selector.verify_budget)
    with col3:
        scale = st.selectbox(label="Select a scale",options=evosys.target_scales)
    with col4:
        st.write('')
        with st.expander(f'Unverified Designs under ***{scale}***'):
            if scale is not None:
                unverified = evosys.ptree.get_unverified_designs(scale)
                st.write(unverified)
            else:
                st.info('No scale available.')

    with st.expander(f"Selector Ranking and Exploration Settings",expanded=True):
        
        select_cfg=copy.deepcopy(evosys.selector.select_cfg)
        ranking_args = U.safe_get_cfg_dict(select_cfg,'ranking_args',DEFAULT_RANKING_ARGS)
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
            ranking_args['normed_only'] = st.checkbox('Normed only',value=ranking_args['normed_only'])
        with cols[2]:
            st.write('')
            ranking_args['drop_zero'] = st.checkbox('Drop 0',value=ranking_args['drop_zero'],
                                                    help='If set, will drop all-zero columns')
        with cols[3]:
            st.write('')
            ranking_args['drop_na'] = st.checkbox('Drop N/A',value=ranking_args['drop_na'])

        cols = st.columns(4)
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


        col1,col2=st.columns(2)
        with col1:
            st.write("##### Quadrant settings")
            cols=st.columns(3)
            quadrant_args=U.safe_get_cfg_dict(select_cfg,'quadrant_args',DEFAULT_QUADRANT_ARGS)
            with cols[0]:
                ranking_args['quadrant_merge']=st.selectbox('Quadrant Merge',options=MERGE_METHODS,index=MERGE_METHODS.index(ranking_args.get('quadrant_merge','average')))
            with cols[1]:
                quadrant_args['design_quantile']=st.number_input('Design Quantile',min_value=0.0,max_value=1.0,step=0.01,value=quadrant_args['design_quantile'])
            with cols[2]:
                quadrant_args['confidence_quantile']=st.number_input('Confidence Quantile',min_value=0.0,max_value=1.0,step=0.01,value=quadrant_args['confidence_quantile'])

        with col2:
            st.write("##### Verify Exploration settings")
            cols=st.columns(3)
            verify_explore_args=U.safe_get_cfg_dict(select_cfg,'verify_explore_args',DEFAULT_VERIFY_EXPLORE_ARGS)
            with cols[0]:
                verify_explore_args['explore_prob']=st.number_input('Verify Explore Prob',min_value=0.0,max_value=1.0,step=0.01,value=verify_explore_args['explore_prob'])
            with cols[1]:
                verify_explore_args['scheduler']=st.selectbox('Verify Scheduler',options=SCHEDULER_OPTIONS,index=SCHEDULER_OPTIONS.index(verify_explore_args['scheduler']))
            with cols[2]:
                verify_explore_args['background_noise']=st.number_input('Verify Background Noise',min_value=0.0,max_value=1.0,step=0.01,value=verify_explore_args['background_noise'])

        select_cfg['ranking_args']=ranking_args
        select_cfg['quadrant_args']=quadrant_args
        select_cfg['verify_explore_args']=verify_explore_args

    verify_selected = st.button('Select')

    select_cfg['verify_all']=False

    if verify_selected:
        with st.status('Selecting design to verify...'):
            design_id,scale=evosys.selector.select_verify(verify_strategy=verify_strategy,select_cfg=select_cfg)
        if design_id is None:
            st.warning('No design to verify.')
        else:
            st.write(f'Selected {design_id} at scale {scale} to be verified.')



def select(evosys,project_dir):
    
    with st.sidebar:
        # AU.running_status(st,evosys)

        mode=st.selectbox('Choose a Selector',options=['Design Selector','Verify Selector'],index=0)


    if mode=='Design Selector':
        design_selector(evosys,project_dir)
    elif mode=='Verify Selector':
        verify_selector(evosys,project_dir)


