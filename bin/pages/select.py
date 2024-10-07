import copy
import json
import time
import pathlib
import streamlit as st
import sys,os
import numpy as np
from model_discovery.agents.flow.gau_flows import DesignModes
from model_discovery.agents.roles.selector import DEFAULT_SEED_DIST,\
    SCHEDULER_OPTIONS,DEFAULT_RANDOM_EVAL_THRESHOLD,DEFAULT_CONFIDENCE_POINTS,DEFAULT_SCALE_WEIGHTS
from model_discovery.evolution import DEFAULT_N_SOURCES


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

Correspondingly, the confidance of a valuation is determined by:

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

def _group_results(results):
    groups = ['blimp','inverse_scaling']
    grouped_results = {}

    def _in_group(alias):
        for g in groups:
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

def _get_acc_norm(eval_metrics,mean=False): # if no acc_norm, use acc
    accs = {}
    for scale in eval_metrics:
        accs[scale] = {}
        acc = eval_metrics[scale]['acc,none,H']
        acc_norm = eval_metrics[scale]['acc_norm,none,H']
        for task in acc:
            if task in acc_norm:
                accs[scale][task] = acc_norm[task]
            else:
                accs[scale][task] = acc[task]
    if mean:
        for scale in accs:
            accs[scale] = np.mean(list(accs[scale].values()))
    return accs




def design_utility_simple(evosys,design_vector):
    weights = U.safe_get_cfg_dict(evosys.selector.select_cfg,'scale_weights',DEFAULT_SCALE_WEIGHTS)
    proposal_rating,_,eval_metrics,_ = get_raw_metrics(evosys,design_vector)
    mean_grouped_accs = _get_acc_norm(group_results(eval_metrics),mean=True)
    weighted_acc = 0
    for scale in mean_grouped_accs:
        weighted_acc+=mean_grouped_accs[scale]*weights[scale]/len(mean_grouped_accs)
    return proposal_rating,weighted_acc

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

def design_ranks(evosys,design_vectors):
    proposal_ratings = {}
    weighted_accs = {}
    confidence_points = {}
    for design in design_vectors:
        proposal_ratings[design],weighted_accs[design] = design_utility_simple(evosys,design_vectors[design])
        confidence_points[design],_ = design_confidence_simple(evosys,design_vectors[design])
    
    def _sort_dict(d,tiering=True):
        d = sorted(d.items(),key=lambda x:float(x[1]),reverse=True)
        if tiering:
            tiers = {}
            for acronym,rating in d:
                if rating not in tiers:
                    tiers[rating] = []
                tiers[rating].append(acronym)
            # tiers = {k:v for k,v in sorted(tiers.items(),key=lambda x:float(x[0]),reverse=True)}
            return tiers
        else:
            return {acronym:rank for acronym,rank in d}
    proposal_ranks = _sort_dict(proposal_ratings)
    weighted_ranks = _sort_dict(weighted_accs)
    confidence_ranks = _sort_dict(confidence_points)
    return proposal_ranks,weighted_ranks,confidence_ranks



def show_design(evosys,design_vector,relative=None,threshold=None):
    confidence,total_points = design_confidence_simple(evosys,design_vector)
    confidence_percentage = confidence / total_points * 100
    st.write(f'###### Design confidence (simple count points): ```{confidence_percentage:.2f}%``` ({confidence}/{total_points})')
    proposal_rating,weighted_acc = design_utility_simple(evosys,design_vector)
    st.write(f'###### Proposal rating: ```{proposal_rating}/5.0```') if proposal_rating else st.write('Proposal rating: ```N/A```')
    st.write(f'###### Weighed mean grouped norm-prioritized accuracy: ```{weighted_acc}```')

    proposal_rating,training_metrics,eval_metrics,eval_stderrs=get_raw_metrics(evosys,design_vector)

    cols=st.columns(2)
    with cols[0]:
        st.write('##### Raw metrics')
        with st.expander('Training metrics'):
            st.json(training_metrics)
        with st.expander('Evaluation metrics'):
            st.json(eval_metrics)
        with st.expander('Evaluation stderrs'):
            st.json(eval_stderrs)
        with st.expander('Group evaluation metrics'):
            st.json(group_results(eval_metrics))
        with st.expander('Accuracy (Norm prioritized)'):
            st.json(_get_acc_norm(eval_metrics))

    with cols[1]:
        st.write('##### Processed metrics')
        with st.expander('Mean grouped evaluation metrics'):
            st.json(_mean_eval_metrics(group_results(eval_metrics)))
        with st.expander('Random normalized evaluation metrics'):
            st.json(random_normalized_eval_metrics(evosys,design_vector,threshold))
        with st.expander('Relative evaluation metrics'):
            if relative and relative != 'none':
                st.json(get_relative_metrics(evosys,design_vector,relative)[2])
            else:
                st.info('No relative metrics to show.')
        with st.expander('Grouped Accuracy (Norm prioritized)'):
            st.json(_get_acc_norm(group_results(eval_metrics)))
        with st.expander('Mean Grouped Accuracy (Norm prioritized)'):
            st.json(_get_acc_norm(group_results(eval_metrics),mean=True))



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

    st.subheader('Design scores')

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


    st.subheader('Design ranks')
    proposal_ranks,weighted_ranks,confidence_ranks = design_ranks(evosys,design_vectors)
    cols = st.columns(3)
    with cols[0]:
        with st.expander('Proposal rating',expanded=True):
            st.json(proposal_ranks)
    with cols[1]:
        with st.expander('Weighted accuracy',expanded=True):
            st.json(weighted_ranks)
    with cols[2]:
        with st.expander('Confidence points',expanded=True):
            st.json(confidence_ranks)






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

