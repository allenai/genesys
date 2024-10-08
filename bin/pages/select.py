import copy
import json
import time
import pathlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys,os
import numpy as np
from model_discovery.agents.flow.gau_flows import DesignModes
from model_discovery.agents.roles.selector import DEFAULT_SEED_DIST,DEFAULT_RANKING_ARGS,\
    SCHEDULER_OPTIONS,DEFAULT_RANDOM_EVAL_THRESHOLD,DEFAULT_CONFIDENCE_POINTS,\
        DEFAULT_SCALE_WEIGHTS,RANKING_METHODS,MERGE_METHODS,DEFAULT_QUADRANT_ARGS
from model_discovery.evolution import DEFAULT_N_SOURCES

from rankit.Table import Table
import rankit.Ranker as Ranker
from rankit.Merge import borda_count_merge,average_ranking_merge


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU


SELECTOR_NOTES = '''
#### *Selector as the Planner & Orchestrator*

The target for the selector is to find the **global optimal design** by exploring
the evolution tree. The utility of a node is decided by its *potential* to be
the optimal language model design in unforeseen computational scales. For
example, the computational budget only allows us to train a 130B model once, but
more budgets for smaller models, so we have to make all design choices under
smaller scales and hope we can succeed in one shot. This is a common scenario
for not only language modeling but also many scientific exploration problems. 

The key to tackling this problem is to **1. forecast the performance of a design
on larger scales; 2. increase the certainty of the forecast**. These two
sub-problems are taken care by the design selection and verification selection
respectively. Along with the maximization of utility, the selector also utilize
some random or bandit-based exploration to keep the search space sufficiently
large. 


##### *The relations between the selector and other parts of the system*

Selector is one of the three core pillars of the evolution system, the other two
pillars are:
 * *the model design agents which focus on how to generate the best and local
   designs based on the provided seeds, references, and instructions, usually
   the evolution
algorithm assumes massive sampling which is not practical for some real-life
problems like language modeling, thus the quality plays a key role;*
 * *and the evolution engine which focuses on how to honestly, and efficiently
   evaluate a design under a scale, the highest throughput, and distributed
   robustness. The
reason avoid massive sampling is because of the high cost of both design and
verification, which makes it impractical for some real-life problems like
language modeling, thus how to improve the efficiency and evaluate the models
with lower cost is important.*

While the other two pillars focus on the quality and quantity of the sampling,
which are the fundamental aspects of the search process, the selector plays the
role of orchestrator and planner of the search.


##### *The jobs of the selector*

The selector determines the direction of evolution & search by the following two
kinds of decisions (design and verify selections as described above):

1. **Select seed and references for a design job**: design selection produce the
   following outputs for the design agent to launch a design job:
    - *Seed*: The base design that the agent will be directly working on. A new
      design will be evolved from this seed by mutating one unit. 
    - *Reference*: The references that indicates the agent with potential
     directions about how to improve the seed.
   - *Instruct*: *Optional* hints from the selector agent or potentially the
     user about how to make the mutation.

2. **Select which design to verify and which scale**: there are two products of
   a verify selection:
    - *Design*: The design to be verified.
    - *Scale*: The scale at which the design will be verified.

   *(**NOTE:** Here we only consider the mutation mode which only need one
   single seed to work on, other modes can essentially be derived from this,
   e.g. crossover essentially reuse units from references to derive the new
   design, design scratch essentially rewrite the root node.)*


#### *Selector's Decision Making*

Overall, the selector always exploits the known good designs to mutate and explore 
unknown spaces, we can use the following information to value a design:

1. **Design artifacts**: Including design proposal, implementation, GAU tree, etc.
2. **Verification results**: Including the training and evaluation metrics under different scales.

Correspondingly, the confidence of a valuation is determined by:

1. **The number of information that is available.** For example, a newly produced design has only design artifacts, 
the confidence of its valuation is low, and we can increase the confidence of its valuation by verifying it at 
different scales. The more verification results we have, the higher confidence we can have on the design's utility. 

2. **The knowledge of the model architecture.** For example, we have more knowledge on the scaling characteristics of 
the Transformer family which has been investigated by previous literatures, similarly, the selector may also
learn the scaling characteristics of other model architectures during the evolution.

##### *Selection general strategies*

Four situations and design selection strategies:

1. **The design has high utility and high confidence.** This may indicate a good design to exploit. The selector may tend to select as the seeds with a high priority. ***(Design selector exploit)***
2. **The design has high utility but low confidence.** This may indicate a good design to explore. The selector may choose to verify it to make it more confident with a high priority. ***(Verify selector exploit)***
3. **The design has low utility but high confidence.** This may indicate a bad design. However, it may still have the potential to be improved, the selector may choose to mutate it with a low priority. ***(Design selector explore)***
4. **The design has low utility and low confidence.** This may indicate a bad design. However, it may be powerful in higher scales, the selector may choose to verify it at different scales to find its potential with a low priority. ***(Verify selector explore)***

Once the design is selected, the design selector needs to consider the following two aspects, which provides assistance from global view as compared to the search engine which actually provides similar information:

1. **The reference selection** The selector may need to select the reference that is most likely to lead to a promising mutation. At present, we use a random strategy to select the reference. It may be improved by some graph or embedding based strategy in the future.
2. **The instruction generation** The selector may need to generate the instruction for the promising mutation. We do not consider it right now, it can be done by a planner agent later.

The verification selector mainly focus on the following two aspects:

1. **The scale of the design to be verified**, given the budget constraint, the basic principle is to put more 
resources when exploiting and less resources when exploring.
2. ***Assist the exploration of scaling characteristics*** While the verification does not only provide the information
for design selection it also help the selector to explore the scaling characteristics of the design. So for a new 
unfamiliar family of design, the selector may put more resources for exploration. (maybe future work)

##### *Valuation of a design*

* **Utility** The utility is a scale-normalized metric that indicates the potential of a design to perform better at 
larger scales. One model is the AUC of the scaling curve, however the problem is to **forcast the scaling curve** of different
designs which can be a *online learning problem* as described above. The evaluate of a model at a given scale can be
obtained by benchmarks (despite the benchmarking of LMs is also an open problem today), we follow the common protocol of 
the model architecture research community.

* **Confidence** The confidence of a design is determined by the number of information that is available as described above.
Confidence can be regarded as a alternative to the scaling curve to make claim about the potential of a design. 
However, the scaling curve and also the knowledge of the model architecture can still improve the confidence.


##### *Exploration and exploitation*

* **For design selection:** we use a random exploration with annealing scheduler. And may incorporate bandit-based exploration strategy, maybe in the future work. 

* **For verify selection:** when selecting the design, it is similar to the design selection; 
while deciding the scale, we use a preservation strategy that start from the lower scale and 
gradually verify the design at larger scales. 

#### :red[*Most important improvement directions*]

1. **RL/tuning of design agents** The design agents can use the utility as a signal to improve the design over the evolution process.
2. **Online learning of scaling curve for selector** The selector can learn the scaling curve to guide the exploration better. 

'''



def design_selector(evosys,project_dir):
    st.header('Design Selector')

    seed_dist = copy.deepcopy(DEFAULT_SEED_DIST)

    _col1,_col2=st.columns([2,3])
    with _col1:
        st.write('###### Configure Selector')
        col1, col2 = st.columns(2)
        with col1:
            # st.markdown("#### Configure design mode")
            mode = st.selectbox(label="Design Mode",options=[i.value for i in DesignModes])
        with col2:
            select_method = st.selectbox(label="Selection Method",options=['random'])
    with _col2:
        st.write('###### Configure *Seed* Selection Distribution')
        cols = st.columns(3)
        with cols[0]:
            seed_dist['scheduler'] = st.selectbox('Annealing Scheduler',options=SCHEDULER_OPTIONS,index=SCHEDULER_OPTIONS.index(seed_dist['scheduler']))
        with cols[1]:
            seed_dist['restart_prob'] = st.slider('Restart Probability',min_value=0.0,max_value=1.0,step=0.01,value=seed_dist['restart_prob'])
        with cols[2]:
            seed_dist['warmup_rounds'] = st.number_input('Warmup Rounds',min_value=0,value=seed_dist['warmup_rounds'])

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
    
    if st.button('Select'):
        selector_args = {
            'n_sources': n_sources,
        }
        
        instruct,seeds,refs=evosys.selector.select_design(selector_args,DesignModes(mode),select_method)

        st.subheader(f'**Instructions from the selector:**')
        if instruct:
            st.write(instruct)
        else:
            st.warning('No instructions from the selector.')

        st.subheader(f'**{len(seeds)} seeds selected:**')
        for seed in seeds:
            with st.expander(f'**{seed.acronym}** *({seed.type})*'):
                st.write(seed.to_prompt())

        st.subheader(f'**{len(refs)} references selected:**')
        for ref in refs:
            with st.expander(f'**{ref.acronym}** *({ref.type})*'):
                st.write(ref.to_prompt())
    else:
        st.info(f'**NOTE:** All settings here will only be applied to this playground. The playground will directly work on the selected running namespace ```{evosys.evoname}```.')



def verify_selector(evosys,project_dir):
    st.header('Verify Selector')

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        verify_strategy = st.selectbox(label="Verify Strategy",options=['random'])
    with col2:
        st.write('')
        with st.expander('Remaining Budget'):
            st.write(evosys.selector.verify_budget)
    with col3:
        scale = st.selectbox(label="Scale",options=evosys.target_scales)
    with col4:
        st.write('')
        with st.expander(f'Unverified Designs under ***{scale}***'):
            unverified = evosys.ptree.get_unverified_designs(scale)
            st.write(unverified)

    if verify_strategy == 'random':
        st.subheader('Random Strategy')
        st.write('*Random strategy will use up smaller scale budgets first.*')

    verify_selected = st.button('Select')

    if verify_selected:
        design_id,scale=evosys.selector.select_verify(verify_strategy=verify_strategy)
        if design_id is None:
            st.warning('No design to verify.')
        else:
            st.write(f'Selected {design_id} at scale {scale} to be verified.')


############################################################

def trainer_state_getter(trainer_state):
    if 'log_history' not in trainer_state:
        return {}
    return {
        'flops': trainer_state['log_history'][-1]['total_flops'], # not sure if resuming influence this
        'loss': trainer_state['log_history'][-1]['train_loss'],
    }

def eval_results_getter(eval_results):
    metrics = {}
    stderrs = {}
    for task_name in eval_results['results']:
        data = eval_results['results'][task_name]
        alias = data['alias']
        for _metric_name in data:
            if _metric_name == 'alias':
                continue
            metric_type = _metric_name.split(',')[0].replace('_stderr','')
            metric = _metric_name.replace('_stderr','')
            if 'smollm125' in task_name:
                continue # still buggy
            higher_is_better = eval_results['higher_is_better'][task_name][metric_type]
            metric+=',H' if higher_is_better else ',L'
            if metric not in metrics:
                metrics[metric] = {}
                stderrs[metric] = {}
            if data[_metric_name] == 'N/A':
                continue
            if 'stderr' in _metric_name:
                stderrs[metric][alias] = data[_metric_name]
            else:
                metrics[metric][alias] = data[_metric_name]
    return metrics,stderrs



def group_results(eval_results):
    grouped_results = {}
    for scale in eval_results:
        if not eval_results[scale]:
            continue
        grouped_results[scale] = _group_results(eval_results[scale])
    return grouped_results


_GROUP_KEYWORDS = ['blimp','inverse_scaling']

def _group_results(results):
    grouped_results = {}

    def _in_group(alias):
        for g in _GROUP_KEYWORDS:
            if g in alias:
                return g
        return False

    for metric in results:
        metric_results = results[metric]
        grouped={}
        for alias in metric_results:
            group = _in_group(alias)
            result = metric_results[alias]
            if group:
                if group not in grouped:
                    grouped[group] = []
                grouped[group].append(result)
            else:
                grouped[alias] = [result]
        for group in grouped:
            grouped[group] = np.mean(grouped[group])
        grouped_results[metric] = grouped
    return grouped_results

def scale_weight_results(evosys,eval_results): # WILL REMOVE SCALES
    weights = U.safe_get_cfg_dict(evosys.selector.select_cfg,'scale_weights',DEFAULT_SCALE_WEIGHTS)
    total_weights = 0
    weighted_results = {}
    for scale in eval_results:
        total_weights+=weights[scale]
        for metric in eval_results[scale]:
            if metric not in weighted_results:
                weighted_results[metric] = {}
            results = eval_results[scale][metric]
            for task in results:
                if task not in weighted_results[metric]:
                    weighted_results[metric][task] = 0
                weighted_results[metric][task]+=results[task]*weights[scale]
    for metric in weighted_results:
        for task in weighted_results[metric]:
            weighted_results[metric][task]/=total_weights
    return weighted_results



_01_NORMED_METRIC_KEYWORDS = [
    'acc',
    'mcc',
    'contains',
    'exact_match'
]

def _is_01_normed(metric):
    for i in _01_NORMED_METRIC_KEYWORDS:
        if i in metric:
            return True
    return False

def _get_acc_norm(scale_free_eval_metrics,mean=False,acc_norm_exclude=['mmlu']): # if no acc_norm, use acc
    accs = {}
    acc = scale_free_eval_metrics['acc,none,H']
    acc_norm = scale_free_eval_metrics['acc_norm,none,H']
    for task in acc:
        if task in acc_norm and task not in acc_norm_exclude:
            accs[task+'/acc_norm'] = acc_norm[task]
        else:
            accs[task+'/acc'] = acc[task]
    return accs

def _flat_weighted_metrics(scale_weighted_eval_metrics,acc_norm_exclude=['mmlu']):
    # https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/106 
    _01_normed_metrics = {} # from 0 to 1, the higher the better
    _unnormed_metrics_h = {} # unknown unnormalized metrics, higher is better
    _unnormed_metrics_l = {} # unknown unnormalized metrics, lower is better
    if len(scale_weighted_eval_metrics) == 0:
        return _01_normed_metrics,{'higher_is_better':{},'lower_is_better':{}}
    accs = _get_acc_norm(scale_weighted_eval_metrics,acc_norm_exclude=acc_norm_exclude)
    scale_weighted_eval_metrics.pop('acc,none,H')
    scale_weighted_eval_metrics.pop('acc_norm,none,H')
    _01_normed_metrics.update(accs)

    for _metric in scale_weighted_eval_metrics:
        if _is_01_normed(_metric):
            for task in scale_weighted_eval_metrics[_metric]:
                result = scale_weighted_eval_metrics[_metric][task]
                if 'mcc' in _metric:
                    result = (result+1)/2
                if _metric.endswith(',L'): # although I think people already did this
                    result = 1-result
                metric = _metric.replace(',L','').replace(',H','')
                metric,tail=metric.split(',')
                if tail!='none':
                    metric+=f'({tail})'
                _01_normed_metrics[task+'/'+metric] = result
        else:
            for task in scale_weighted_eval_metrics[_metric]:
                result = scale_weighted_eval_metrics[_metric][task]
                metric,tail,LH = _metric.split(',')
                if tail!='none':
                    metric+=f'({tail})'
                if LH == 'H':
                    _unnormed_metrics_h[task+'/'+metric] = result
                else:
                    _unnormed_metrics_l[task+'/'+metric] = result
    _unnormed_metrics = {
        'higher_is_better':_unnormed_metrics_h,
        'lower_is_better':_unnormed_metrics_l,
    }
    return _01_normed_metrics,_unnormed_metrics

def get_ranking_metrics(evosys,design_vector,normed_only=True):
    proposal_rating,training_metrics,eval_metrics,_ = get_raw_metrics(evosys,design_vector)
    scale_weighted_grouped_metrics = scale_weight_results(evosys,group_results(eval_metrics))
    _01_normed_metrics,_unnormed_metrics = _flat_weighted_metrics(scale_weighted_grouped_metrics)
    
    ranking_metrics = _01_normed_metrics
    scale_weighted_training_metrics = scale_weight_results(evosys,training_metrics)
    ranking_metrics.update(scale_weighted_training_metrics)
    ranking_metrics['proposal_rating'] = proposal_rating/5.0
    if normed_only:
        return ranking_metrics,None
    ranking_metrics_unnormed=_unnormed_metrics['higher_is_better']
    for i in _unnormed_metrics['lower_is_better']:
        ranking_metrics_unnormed[i.replace(' - ','')] = -_unnormed_metrics['lower_is_better'][i]
    return ranking_metrics,ranking_metrics_unnormed


def get_raw_metrics(evosys,design_vector):
    proposal_rating = design_vector['proposal_rating'] if 'proposal_rating' in design_vector else None
    training_metrics = {}
    eval_metrics = {}
    eval_stderrs = {}
    for scale in design_vector['verifications']:
        verification_report = design_vector['verifications'][scale]
        if 'trainer_state.json' in verification_report:
            training_record = verification_report['trainer_state.json']
            training_metrics[scale] = trainer_state_getter(training_record)
        else:
            training_metrics[scale] = {}
        if 'eval_results.json' in verification_report:
            eval_results = verification_report['eval_results.json']
            eval_metrics[scale],eval_stderrs[scale] = eval_results_getter(eval_results)
        else:
            eval_metrics[scale] = {}
            eval_stderrs[scale] = {}
    return proposal_rating,training_metrics,eval_metrics,eval_stderrs


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
    proposal_rating,training_metrics,eval_metrics,eval_stderrs = get_raw_metrics(evosys,design_vector)
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
        _proposal_rating,_training_metrics,_eval_metrics,_eval_stderrs = get_raw_metrics(evosys,relative_vector)
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




def design_utility_simple(evosys,design_vector):
    proposal_rating,_,eval_metrics,_ = get_raw_metrics(evosys,design_vector)
    scale_weighted_grouped_metrics = scale_weight_results(evosys,group_results(eval_metrics))
    _01_normed_metrics,_ = _flat_weighted_metrics(scale_weighted_grouped_metrics)
    return proposal_rating,np.mean(list(_01_normed_metrics.values()))

def get_random_metrics(evosys):
    eval_metrics,eval_stderrs = eval_results_getter(evosys.ptree.random_baseline)
    return None,None,eval_metrics,eval_stderrs

def design_confidence_simple(evosys,design_vector):
    verifications = design_vector['verifications']
    verified_scales = verifications.keys()
    confidence_points = U.safe_get_cfg_dict(evosys.selector.select_cfg,'confidence_points',DEFAULT_CONFIDENCE_POINTS)
    confidence = sum([confidence_points[i] for i in verified_scales]) + confidence_points['proposed'] + confidence_points['implemented']
    total_points = sum(confidence_points.values())
    return confidence,total_points


def show_design(evosys,design_vector,relative=None,threshold=None):
    confidence,total_points = design_confidence_simple(evosys,design_vector)
    confidence_percentage = confidence / total_points * 100
    proposal_rating,weighted_acc = design_utility_simple(evosys,design_vector)
    proposal_rating,training_metrics,eval_metrics,eval_stderrs=get_raw_metrics(evosys,design_vector)
    grouped_metrics = group_results(eval_metrics)
    scale_weighted_grouped_metrics = scale_weight_results(evosys,grouped_metrics)
    _01_normed_metrics,_unnormed_metrics = _flat_weighted_metrics(scale_weighted_grouped_metrics)
    
    ranking_metrics,ranking_metrics_unnormed = get_ranking_metrics(evosys,design_vector,False)
    cols=st.columns(2)
    with cols[0]:
        st.write(f'###### Design confidence (simple count points): ```{confidence_percentage:.2f}%``` ({confidence}/{total_points})')
        st.write(f'###### Proposal rating: ```{proposal_rating}/5.0```') if proposal_rating else st.write('Proposal rating: ```N/A```')
        st.write(f'###### Scale-Weighted mean 01 normed metrics: ```{weighted_acc}```')
    
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
            'Tie sensitivity normalized metrics'
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

def around_ranking_matrix(ranking_matrix,draw_margin=0.01):
    ranking_matrix = ranking_matrix.apply(lambda x: np.around(x/draw_margin)*draw_margin)
    return ranking_matrix

def get_ranking_matrix(evosys,design_vectors,normed_only=True,verified_only=True,
        drop_na=True,drop_zero=True):
    ranking_matrix = pd.DataFrame()
    for acronym in design_vectors:
        design_vector = design_vectors[acronym]
        ranking_metrics,ranking_metrics_unnormed = get_ranking_metrics(evosys,design_vector,normed_only)
        if verified_only and len(ranking_metrics)<=1: # only proposal rating
            continue
        ranking_matrix = pd.concat([ranking_matrix,pd.DataFrame(ranking_metrics,index=[acronym])],axis=0)
        if not normed_only:
            ranking_matrix = pd.concat([ranking_matrix,pd.DataFrame(ranking_metrics_unnormed,index=[acronym])],axis=0)
    # drop cols with any N/A, drop all 0 cols
    if drop_na:
        ranking_matrix=ranking_matrix.dropna(axis=1)
    if drop_zero:
        ranking_matrix = ranking_matrix.loc[:, (ranking_matrix != 0).any(axis=0)]
    return ranking_matrix


def column_to_pairs(column):
    pairs = []
    for i in range(len(column)):
        for j in range(i+1,len(column)):
            obj1 = column.index[i]
            obj2 = column.index[j]
            v1 = column[obj1]
            v2 = column[obj2]
            if v1 > v2:
                pairs.append((obj1,obj2,v1,v2))
            else:
                pairs.append((obj2,obj1,v2,v1))
    cols=['design1','design2','metric1','metric2']
    return pd.DataFrame(pairs,columns=cols)


def ranking_matrix_to_pairs(ranking_matrix:pd.DataFrame):
    # convert to pairs of winner_name,loser_name,winner_score,loser_score
    columns = []
    for metric in ranking_matrix.columns:
        column = ranking_matrix[metric]
        pairs = column_to_pairs(column)
        columns.append(pairs)
    merged_columns = pd.concat(columns,axis=0)
    return merged_columns

def __rank_designs(ranking_matrix,ranking_args,ranking_method):
    cols=['design1','design2','metric1','metric2']
    data = Table(ranking_matrix, col = cols)
    if ranking_method == 'massey':
        ranker = Ranker.MasseyRanker(drawMargin = ranking_args['draw_margin'])
    elif ranking_method == 'colley':
        ranker = Ranker.ColleyRanker(drawMargin = ranking_args['draw_margin'])
    elif ranking_method == 'markov':
        ranker = Ranker.MarkovRanker(restart = ranking_args['markov_restart'], 
            threshold = ranking_args['convergence_threshold'])
    else:
        raise ValueError(f'Unknown ranking method: {ranking_method}')
    return ranker.rank(data)

def _merge_ranks(ranks,merge_method):
    _ranks=[ranks[i] for i in ranks]
    if merge_method == 'borda':
        rank = borda_count_merge(_ranks)
        rank.rename(columns={'BordaCount':'rating'},inplace=True)
        return rank
    elif merge_method == 'average':
        return average_ranking_merge(_ranks)
    else:
        raise ValueError(f'Unknown merge method: {merge_method}')
    

def _rank_designs(ranking_matrix,ranking_args,ranking_method):
    metric_wise_merge = ranking_args['metric_wise_merge']
    metric_wise_merge = None if metric_wise_merge in ['None','none'] else metric_wise_merge
    if ranking_method not in ['colley','massey']:
        ranking_matrix = around_ranking_matrix(ranking_matrix,ranking_args['draw_margin']) 
    ranks = {}
    if metric_wise_merge and ranking_method != 'markov':
        # separte each column 
        for metric in ranking_matrix.columns:
            column = ranking_matrix[metric]
            if ranking_method == 'average':
                # directly convert the to the format name, rating, rank
                ranki = column.sort_values(ascending=False)
                ranki = ranki.to_frame(name='rating')
                ranki.reset_index(inplace=True)
                ranki.rename(columns={'index':'name'},inplace=True)
                ranki['rank'] = ranki['rating'].rank(method='min', ascending=False).astype(int)
                ranks[metric] = ranki
            else:
                pairs = column_to_pairs(column)
                ranks[metric] = __rank_designs(pairs,ranking_args,ranking_method)
        rank = _merge_ranks(ranks,metric_wise_merge)
    else:
        # merge all columns, concate the index
        if ranking_method == 'average':
            rank = ranking_matrix.rank(method='average')
            cols = rank.columns
            rank['rating'] = rank.mean(axis=1)
            rank = rank.drop(cols,axis=1)
            rank = rank.reset_index()
            rank = rank.rename(columns={'index':'name'})
            rank['rank'] = rank['rating'].rank(method='min', ascending=False).astype(int)
        else:   
            merged_columns = ranking_matrix_to_pairs(ranking_matrix)
            rank = __rank_designs(merged_columns,ranking_args,ranking_method)
    return rank,ranks


def rank_designs(ranking_matrix,ranking_args):
    ranking_methods = ranking_args['ranking_method']
    if isinstance(ranking_methods,str):
        ranking_methods = [ranking_methods]
    subranks = {}
    subsubranks = {}
    for ranking_method in ranking_methods:
        _subrank,_subsubranks = _rank_designs(ranking_matrix,ranking_args,ranking_method)
        subranks[ranking_method] = _subrank
        subsubranks[ranking_method] = _subsubranks
    if len(ranking_methods) > 1:
        merge_method = ranking_args['multi_rank_merge']
        rank = _merge_ranks(subranks,merge_method)
    else:
        rank = subranks[ranking_methods[0]]
    return rank,subranks,subsubranks

def rank_confidences(evosys,design_vectors,filter=[]):
    rank={}
    for acronym in design_vectors:
        if filter and acronym not in filter:
            continue
        design_vector = design_vectors[acronym]
        confidence,total_points = design_confidence_simple(evosys,design_vector)
        rank[acronym] = confidence
    # to df of Nx1, each row is a confidence, indexed by design
    rank = pd.DataFrame(list(rank.items()), columns=['name', 'confidence'])
    rank = rank.sort_values(by='confidence',ascending=False)
    rank.reset_index(inplace=True)
    rank.drop(columns=['index'],inplace=True)
    rank['rank'] = rank['confidence'].rank(method='min', ascending=False).astype(int)
    return rank

# def _shuffle_within_rank(df):
#     return df.groupby('rank').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)

def _divide_rank(df,quantile):
    # divide into upper quantile and lower quantile
    upper_quantile = df['rank'].quantile(quantile)
    lower_quantile = df['rank'].quantile(1-quantile)
    upper_df = df[df['rank'] > upper_quantile]
    lower_df = df[df['rank'] < lower_quantile]
    return upper_df,lower_df

def get_ranking_quadrant(evosys,design_rank,confidence_rank,ranking_args):
    quadrant_args = U.safe_get_cfg_dict(evosys.selector.select_cfg,'quadrant_args',DEFAULT_QUADRANT_ARGS)
    # confidence_rank = _shuffle_within_rank(confidence_rank)
    # design_rank = _shuffle_within_rank(design_rank)
    design_upper,design_lower = _divide_rank(design_rank,quadrant_args['design_quantile'])
    confidence_upper,confidence_lower = _divide_rank(confidence_rank,quadrant_args['confidence_quantile'])
    return {
        'design_upper':design_upper,
        'design_lower':design_lower,
        'confidence_upper':confidence_upper,
        'confidence_lower':confidence_lower
    }

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
    

LEADERBOARD_1 = [
    'blimp',
    'inverse_scaling',
    'glue',
    'qa4mre',
    'mathqa',
    'wsc273'
]

def export_leaderboard(evosys,design_vectors, baseline_vectors, scale):
    raise NotImplementedError
    leaderboard = {}
    leaderboard_relative = {}
    leaderboard['baseline'] = {}
    leaderboard['design'] = {}
    leaderboard['random'] = {}
    leaderboard_relative['random'] = {}
    leaderboard_relative['baseline'] = {}
    leaderboard_relative['design'] = {}
    _,_,eval_metrics,_ = get_random_metrics(evosys)
    grouped_metrics = group_results(eval_metrics)
    for task in LEADERBOARD_1:
        leaderboard['random'][task] = grouped_metrics[scale][task]
        leaderboard_relative['random'][task] = 0
    for mode in ['baseline','design']:
        vectors = baseline_vectors if mode == 'baseline' else design_vectors
        for acronym in vectors:
            leaderboard[mode][acronym] = {}
            _,_,eval_metrics,_ = get_raw_metrics(evosys,baseline_vectors[acronym])
            grouped_metrics = group_results(eval_metrics)
            for metric in LEADERBOARD_1:
                result = grouped_metrics[scale][metric]
                if metric not in leaderboard['random'] or metric not in leaderboard[mode][acronym]:
                    leaderboard[mode][acronym][metric] = 'N/A'
                    leaderboard_relative[mode][acronym][metric] = 'N/A'
                else:
                    leaderboard[mode][acronym][metric] = result
                    leaderboard_relative[mode][acronym][metric] = result / leaderboard['random'][metric] - 1
    return leaderboard,leaderboard_relative





def selector_lab(evosys,project_dir):
    st.header('*Selector Lab*')
    
    with st.expander('Notes for Selector and the Evolution System (For internal reference)',icon='🎼'):
        st.markdown(SELECTOR_NOTES)

    st.subheader('Design metrics explorer')

    with st.status('Loading data...'):
        design_vectors = evosys.ptree.get_design_vectors()
        baseline_vectors = evosys.ptree.get_baseline_vectors()

    cols = st.columns(3)
    with cols[0]:
        show_mode = st.selectbox('Show mode',options=['design','baseline'])

    with cols[1]:
        if show_mode == 'design':
            vectors = design_vectors
        elif show_mode == 'baseline':
            vectors = baseline_vectors
        options = []
        for design in vectors:
            options.append(f'{design} ({len(vectors[design]["verifications"])} verified)')
        selected_design = st.selectbox('Select a design',options=options)
        selected_design = selected_design.split(' ')[0]
        design_vector=vectors[selected_design]

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

    show_design(evosys,design_vector,relative,threshold)


    st.subheader('**Design Ranking**')
    ranking_args = U.safe_get_cfg_dict(evosys.selector.select_cfg,'ranking_args',DEFAULT_RANKING_ARGS)
    cols = st.columns([5,1.5,0.8,0.8])
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
        drop_zero = st.checkbox('Drop All 0',value=ranking_args['drop_zero'])
    with cols[3]:
        st.write('')
        drop_na = st.checkbox('Drop N/A',value=ranking_args['drop_na'])
    # with cols[4]:
    #     st.write('')
    #     rank_design_btn = st.button('Rank',use_container_width=True)

    with st.expander('Ranking method arguments',expanded=True):
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

    # if rank_design_btn:
    assert ranking_args['ranking_method'], 'Ranking method is required'
    with st.status('Generating ranking matrix...',expanded=True):
        ranking_matrix = get_ranking_matrix(evosys,design_vectors,normed_only,
            drop_na=drop_na,drop_zero=drop_zero)
        # compute average at the end of each row
        _ranking_matrix = ranking_matrix.copy()
        _ranking_matrix['avg.'] = _ranking_matrix.mean(axis=1)
        st.dataframe(_ranking_matrix)
    
    with st.status('Ranking designs...',expanded=True):
        design_rank,subranks,subsubranks = rank_designs(ranking_matrix,ranking_args)
    
    with st.expander('Design Ranking',expanded=True):
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

    st.subheader('Ranking Quadrant')
    cols = st.columns([1,2])
    with cols[0]:
        with st.status('Ranking confidence filter by design rank...',expanded=True):
            st.subheader('Confidence Ranking')
            confidence_rank = rank_confidences(evosys,design_vectors,design_rank['name'].tolist())
            st.dataframe(confidence_rank)

    with cols[1]:
        with st.status('Generating ranking quadrant...',expanded=True):
            ranking_quadrants = get_ranking_quadrant(evosys,design_rank,confidence_rank,ranking_args)
            fig = visualize_ranking_quadrant(ranking_quadrants)
            st.pyplot(fig)




def select(evosys,project_dir):
    
    st.title('Selector Playground')

    if 'select_view_mode' not in st.session_state:
        st.session_state.select_view_mode = 'design'

    with st.sidebar:
        AU.running_status(st,evosys)

        st.write('Choose Mode')
        if st.button('**Design Selector**' if st.session_state.select_view_mode == 'design' else 'Design Selector',use_container_width=True):
            st.session_state.select_view_mode = 'design'
        if st.button('**Verify Selector**' if st.session_state.select_view_mode == 'verify' else 'Verify Selector',use_container_width=True):
            st.session_state.select_view_mode = 'verify'
        if st.button('***Selector Lab***' if st.session_state.select_view_mode == 'lab' else '*Selector Lab*',use_container_width=True):
            st.session_state.select_view_mode = 'lab'


    if st.session_state.select_view_mode == 'design':
        design_selector(evosys,project_dir)
    elif st.session_state.select_view_mode == 'verify':
        verify_selector(evosys,project_dir)
    elif st.session_state.select_view_mode == 'lab':
        selector_lab(evosys,project_dir)

