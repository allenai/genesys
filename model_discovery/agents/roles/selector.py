from typing import List, Tuple, Dict
import random
import copy
import numpy as np
import pandas as pd
import model_discovery.utils as U
from model_discovery.system import DesignModes

from rankit.Table import Table
import rankit.Ranker as Ranker
from rankit.Merge import borda_count_merge,average_ranking_merge



SEED_TYPES = ['DesignArtifactImplemented','ReferenceCoreWithTree']



# https://github.com/wattlebird/ranking
RANKING_METHODS = [
    'average',
    'massey',
    'colley',
    'markov'
]

MERGE_METHODS = ['borda','average']

SCHEDULER_OPTIONS = ['constant']

SELECT_METHODS = ['quadrant','random']
VERIFY_STRATEGIES = ['quadrant','random']



DEFAULT_SELECT_METHOD = 'quadrant'
DEFAULT_VERIFY_STRATEGY = 'quadrant'


# https://huggingface.co/docs/timm/en/reference/schedulers Similar to lr schedulers
DEFAULT_SEED_DIST = {
    'scheduler':'constant', # linear or cosine
    'restart_prob':0.05, # the chance of sampling from the seeds again
    'warmup_rounds':10, # the number of verified designs to warmup the scheduler
}


DEFAULT_N_SEEDS_SETTINGS = {
    'warmup_rounds_crossover':10, # the number of implemented designs to warmup the scheduler, before that, only do mutation
    'warmup_rounds_scratch':20, # the number of implemented designs to warmup the scheduler, before that, only do mutation
}

DEFAULT_N_SEEDS_DIST = {
    '0': 0.01,
    '1': 0.9,
    '2': 0.09,
    '3': 0,
    '4': 0,
    '5': 0,
}



DEFAULT_CONFIDENCE_POINTS = {
    'proposed': 1,
    'implemented': 1,
    '14M': 1,
    '31M': 2,
    '70M': 3,
    '125M': 4,
    '350M': 5,
    '760M': 6,
    '1300M': 7,
    '2700M': 8,
    '6700M': 9,
    '13B': 10,
    '175B': 11,
    '1T': 12,
    'units_placeholder': 20,
}


DEFAULT_SCALE_WEIGHTS = {
    '14M': 1,
    '31M': 1,
    '70M': 1,
    '125M': 1,
    '350M': 1,
    '760M': 1,
    '1300M': 1,
    '2700M': 1,
    '6700M': 1,
    '13B': 1,
    '175B': 1,
    '1T': 1,
}


DEFAULT_RANDOM_EVAL_THRESHOLD = 0.02 # decide if a comparison is significant


DEFAULT_RANKING_ARGS = {
    'ranking_method':'massey', # or a list of ranking methods, and aggregated by borda
    'normed_only':True,
    'draw_margin':0.01,
    'drop_na':True,
    'drop_zero':True,
    'metric_wise_merge':None, # if Set, it will rank for each metric and then aggregate by borda
    'multi_rank_merge': 'borda', # 'borda', 'average'
    'markov_restart': 0.3,
    'convergence_threshold': 1e-4,
    'quadrant_merge': 'average', # 'borda', 'average'
    'soft_filter_threshold': 0.02, # filter out metrics with the highest relative rating lower than this (-1, i.e. -100% means no filtering)
    'absolute_value_threshold': True,
    'normed_difference': False, # if True, it will use |difference| instead of difference
}

DEFAULT_QUADRANT_ARGS = {
    'design_quantile':0.25, # upper quantile regarded as good
    'confidence_quantile':0.25, # upper quantile regarded as good
}

DEFAULT_DESIGN_EXPLORE_ARGS = {
    'explore_prob':0.2, # chance go to explore quadrant
    'scheduler':'constant',
    'background_noise':0.05 # noise when selecting from ranked designs
}

DEFAULT_VERIFY_EXPLORE_ARGS = {
    'explore_prob':0.2,
    'scheduler':'constant',
    'background_noise':0.05
}



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

def scale_weight_results(select_cfg,eval_results): # WILL REMOVE SCALES
    weights = U.safe_get_cfg_dict(select_cfg,'scale_weights',DEFAULT_SCALE_WEIGHTS)
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


# def _shuffle_within_rank(df):
#     return df.groupby('rank').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)

def _divide_rank(df,quantile):
    # divide into upper quantile and lower quantile
    df = df.groupby('rank').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)
    df = df.sort_values(by='rank',ascending=True)
    upper_quantile = round(len(df)*quantile)
    upper_df = df.iloc[:upper_quantile]
    lower_df = df.iloc[upper_quantile:]
    return upper_df,lower_df

def design_confidence_simple(select_cfg,design_vector):
    verifications = design_vector['verifications']
    verified_scales = verifications.keys()
    confidence_points = U.safe_get_cfg_dict(select_cfg,'confidence_points',DEFAULT_CONFIDENCE_POINTS)
    confidence = sum([confidence_points[i] for i in verified_scales]) + confidence_points['proposed'] + confidence_points['implemented']
    total_points = sum(confidence_points.values())
    return confidence,total_points


def _get_ranking_quadrant(select_cfg,design_rank,confidence_rank):
    quadrant_args = U.safe_get_cfg_dict(select_cfg,'quadrant_args',DEFAULT_QUADRANT_ARGS)
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

def rank_confidences(search_cfg,design_vectors,filter=[]):
    rank={}
    for acronym in design_vectors:
        if filter and acronym not in filter:
            continue
        design_vector = design_vectors[acronym]
        confidence,total_points = design_confidence_simple(search_cfg,design_vector)
        rank[acronym] = confidence
    # to df of Nx1, each row is a confidence, indexed by design
    rank = pd.DataFrame(list(rank.items()), columns=['name', 'confidence'])
    rank = rank.sort_values(by='confidence',ascending=False)
    rank.reset_index(inplace=True)
    rank.drop(columns=['index'],inplace=True)
    rank['rank'] = rank['confidence'].rank(method='min', ascending=False).astype(int)
    return rank


def _merge_ranks(ranks,merge_method,rename=True):
    _ranks=[ranks[i] for i in ranks]
    if merge_method == 'borda':
        rank = borda_count_merge(_ranks)
        if rename:
            rank.rename(columns={'BordaCount':'rating'},inplace=True)
        return rank
    elif merge_method == 'average':
        rank = average_ranking_merge(_ranks)
        if rename:
            rank.rename(columns={'AverageRank':'rating'},inplace=True)
        return rank
    else:
        raise ValueError(f'Unknown merge method: {merge_method}')

def around_ranking_matrix(ranking_matrix,draw_margin=0.01):
    ranking_matrix = ranking_matrix.apply(lambda x: np.around(x/draw_margin)*draw_margin)
    return ranking_matrix



def trainer_state_getter(trainer_state):
    if 'log_history' not in trainer_state:
        return {}
    _trainer_state = {}
    if 'total_flops' in trainer_state['log_history'][-1]:
        _trainer_state['flops'] = {'total_flops':trainer_state['log_history'][-1]['total_flops']}
    if 'train_loss' in trainer_state['log_history'][-1]:
        _trainer_state['loss'] = {'train_loss':trainer_state['log_history'][-1]['train_loss']}
    return _trainer_state


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
            accs[task.replace(' - ','')+'/acc_norm'] = acc_norm[task]
        else:
            accs[task.replace(' - ','')+'/acc'] = acc[task]
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
                task = task.replace(' - ','')
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
                task = task.replace(' - ','')
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


def get_raw_metrics(design_vector):
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


def get_ranking_metrics(select_cfg,design_vector,normed_only=True):
    proposal_rating,training_metrics,eval_metrics,_ = get_raw_metrics(design_vector)
    scale_weighted_grouped_metrics = scale_weight_results(select_cfg,group_results(eval_metrics))
    _01_normed_metrics,_unnormed_metrics = _flat_weighted_metrics(scale_weighted_grouped_metrics)
    ranking_metrics = _01_normed_metrics
    scale_weighted_training_metrics = scale_weight_results(select_cfg,training_metrics)
    ranking_metrics.update(scale_weighted_training_metrics)
    ranking_metrics['proposal_rating'] = proposal_rating/5.0
    if normed_only:
        return ranking_metrics,None
    ranking_metrics_unnormed=_unnormed_metrics['higher_is_better']
    for i in _unnormed_metrics['lower_is_better']:
        ranking_metrics_unnormed[i.replace(' - ','')] = -_unnormed_metrics['lower_is_better'][i]
    return ranking_metrics,ranking_metrics_unnormed



def _01_normed_soft_filter(ranking_matrix, relative_to_01_normed, soft_filter_threshold,
    absolute_value_threshold=False,normed_difference=False,st=None):
    if st:
        st.write('###### Original ranking matrix')
        st.write(ranking_matrix)
    
    # Convert relative_to_01_normed to a DataFrame if it's not already
    pd_random_metrics = pd.DataFrame(relative_to_01_normed, index=['random'])

    if st:
        st.write('###### Random metrics')
        st.write(pd_random_metrics)

    # Identify common columns
    common_columns = ranking_matrix.columns.intersection(pd_random_metrics.columns)
    non_common_columns = ranking_matrix.columns.difference(common_columns)

    # Calculate relative performance for common columns
    _ranking_matrix_relative = ranking_matrix[common_columns].sub(pd_random_metrics[common_columns].iloc[0])
    if not absolute_value_threshold:
        _ranking_matrix_relative = _ranking_matrix_relative.div(pd_random_metrics[common_columns].iloc[0].replace(0, np.nan))
    if normed_difference:
        _ranking_matrix_relative = _ranking_matrix_relative.abs()

    if st:
        if absolute_value_threshold:
            st.write('###### Absolute difference to random')
            st.write(_ranking_matrix_relative) 
        else:
            st.write('###### Relative difference to random (%)')
            st.write(_ranking_matrix_relative*100) 

    # Find columns where the maximum relative improvement is greater than or equal to the threshold
    columns_to_keep = _ranking_matrix_relative.max() >= soft_filter_threshold

    if st:
        st.write('###### Columns to keep in relative difference matrix')
        if absolute_value_threshold:
            st.write(_ranking_matrix_relative[common_columns.intersection(columns_to_keep[columns_to_keep].index)])
        else:
            st.write(_ranking_matrix_relative[common_columns.intersection(columns_to_keep[columns_to_keep].index)]*100)

    # Filter the common columns based on the threshold
    filtered_common = ranking_matrix[common_columns.intersection(columns_to_keep[columns_to_keep].index)]
    if st:
        st.write('###### Filtered common columns')
        st.write(filtered_common)

    # Combine filtered common columns with non-common columns
    filtered_matrix = pd.concat([filtered_common, ranking_matrix[non_common_columns]], axis=1)
    return filtered_matrix
    

def get_ranking_matrix(select_cfg,design_vectors,normed_only=True,verified_only=True,
        drop_na=True,drop_zero=True,soft_filter_threshold=-1,relative_to_01_normed=None,
        absolute_value_threshold=True,normed_difference=False):
    ranking_matrix = pd.DataFrame()
    for acronym in design_vectors:
        design_vector = design_vectors[acronym]
        ranking_metrics,ranking_metrics_unnormed = get_ranking_metrics(select_cfg,design_vector,normed_only)
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
    if relative_to_01_normed:
        ranking_matrix = _01_normed_soft_filter(ranking_matrix,relative_to_01_normed,
            soft_filter_threshold,absolute_value_threshold,normed_difference)
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


def _combine_design_quadrant(quadrants):
    design_upper = quadrants['design_upper']
    design_lower = quadrants['design_lower']
    confidence_upper = quadrants['confidence_upper']
    confidence_lower = quadrants['confidence_lower']

    # Combine design and confidence data
    design_data = pd.concat([design_upper, design_lower])
    confidence_data = pd.concat([confidence_upper, confidence_lower])
    
    # Merge design and confidence data on 'name'
    all_data = pd.merge(design_data, confidence_data, on='name', suffixes=('_design', '_confidence'))
    all_data = all_data.drop_duplicates(subset=['name'])

    # Calculate midpoints
    design_midpoint = (design_upper['rating'].min() + design_lower['rating'].max()) / 2
    confidence_midpoint = (confidence_upper['confidence'].min() + confidence_lower['confidence'].max()) / 2

    # Divide into quadrants based on midpoints
    quadrants = {
        'good_design_high_confidence': all_data[(all_data['rating'] > design_midpoint) & (all_data['confidence'] > confidence_midpoint)],
        'good_design_low_confidence': all_data[(all_data['rating'] > design_midpoint) & (all_data['confidence'] <= confidence_midpoint)],
        'poor_design_high_confidence': all_data[(all_data['rating'] <= design_midpoint) & (all_data['confidence'] > confidence_midpoint)],
        'poor_design_low_confidence': all_data[(all_data['rating'] <= design_midpoint) & (all_data['confidence'] <= confidence_midpoint)]
    }
    return quadrants


def _rank_combined_quadrant(quadrants,merge_method='average',rename=True):
    ranked_quadrants = {}
    for quadrant_name, quadrant_data in quadrants.items():
        design_data = quadrant_data.loc[:,['name','rating','rank_design']]
        design_data.rename(columns={'rank_design':'rank'},inplace=True)
        confidence_data = quadrant_data.loc[:,['name','confidence','rank_confidence']]
        confidence_data.rename(columns={'rank_confidence':'rank'},inplace=True)
        ranked_quadrants[quadrant_name] = _merge_ranks({'design':design_data,'confidence':confidence_data},merge_method,rename=rename)
    return ranked_quadrants


def get_ranked_quadrant(select_cfg,design_rank,confidence_rank):
    quadrants = _get_ranking_quadrant(select_cfg,design_rank,confidence_rank)
    combined_quadrant = _combine_design_quadrant(quadrants)
    ranking_args = U.safe_get_cfg_dict(select_cfg,'ranking_args',DEFAULT_RANKING_ARGS)
    ranked_quadrants = _rank_combined_quadrant(combined_quadrant,ranking_args['multi_rank_merge'])
    return ranked_quadrants


class Selector:
    def __init__(self,ptree,select_cfg,_verify_budget,stream,
            design_budget_limit,budget_type,token_mults,target_scales):
        self.ptree=ptree
        self.select_cfg=select_cfg
        self.budget_type=budget_type
        self._verify_budget=_verify_budget
        self.stream=stream
        self.design_budget_limit=design_budget_limit
        self.token_mults=token_mults
        self.target_scales=target_scales

    @property
    def design_budget(self):
        if self.design_budget_limit>0:
            return self.design_budget_limit - self.ptree.design_cost
        else:
            if self.budget_type=='design_bound':
                print('WARNING: The evolution will run forever since design budget is not limited while in design bound mode.')
            return float('inf')

    def _get_ranked_quadrants(self,select_cfg=None):
        select_cfg = self.select_cfg if select_cfg is None else select_cfg
        self.design_vectors = self.ptree.get_design_vectors() # cache it
        design_rank = self._rank_designs(self.design_vectors)
        if design_rank is None:
            return None
        confidence_rank = rank_confidences(select_cfg,self.design_vectors,design_rank['name'].tolist())
        if design_rank.empty:
            return None
        ranked_quadrants = get_ranked_quadrant(select_cfg,design_rank,confidence_rank)
        return ranked_quadrants

    def _rank_designs(self,design_vectors,select_cfg=None):
        select_cfg = self.select_cfg if select_cfg is None else select_cfg
        ranking_args = U.safe_get_cfg_dict(select_cfg,'ranking_args',DEFAULT_RANKING_ARGS)
        ranking_matrix = get_ranking_matrix(
            select_cfg,design_vectors,ranking_args['normed_only'],
            drop_na=ranking_args['drop_na'],drop_zero=ranking_args['drop_zero'],
            soft_filter_threshold=ranking_args['soft_filter_threshold'],
            relative_to_01_normed=self._get_random_metrics(select_cfg)[0],
            absolute_value_threshold=ranking_args['absolute_value_threshold'],
            normed_difference=ranking_args['normed_difference'])
        if len(ranking_matrix)==0:
            return None
        design_rank,_,_ = rank_designs(ranking_matrix,ranking_args)
        return design_rank

    def _get_random_metrics(self,select_cfg=None):
        select_cfg = self.select_cfg if select_cfg is None else select_cfg
        eval_metrics,_ = eval_results_getter(self.ptree.random_baseline)
        _grouped_metrics = _group_results(eval_metrics)
        _01_normed_metrics,_unnormed_metrics = _flat_weighted_metrics(_grouped_metrics)
        return _01_normed_metrics,_unnormed_metrics


    #########################  Select Design  #########################

    def _get_n_seeds(self,select_cfg=None):
        select_cfg = self.select_cfg if select_cfg is None else select_cfg
        n_seeds_settings = U.safe_get_cfg_dict(select_cfg,'n_seeds_settings',DEFAULT_N_SEEDS_SETTINGS)
        n_seeds_dist = U.safe_get_cfg_dict(select_cfg,'n_seeds_dist',DEFAULT_N_SEEDS_DIST)
        n_implemented = len(self.ptree.filter_by_type(['DesignArtifactImplemented']))
        if n_implemented < n_seeds_settings['warmup_rounds_scratch']:
            n_seeds_dist['0'] = 0
        if n_implemented < n_seeds_settings['warmup_rounds_crossover']:
            for k in n_seeds_dist:
                if k not in ['0','1']:
                    n_seeds_dist[k] = 0
        n_seeds_dist_sum = sum(n_seeds_dist.values())
        if n_seeds_dist_sum == 0:
            n_seeds_dist['1'] = 1
        n_seeds = {k:v/n_seeds_dist_sum for k,v in n_seeds_dist.items()}
        n_seeds = np.random.choice(list(n_seeds.keys()),size=1,p=list(n_seeds.values()))[0]
        n_seeds = int(n_seeds)
        return n_seeds
    

    def select_design(self,selector_args,n_seeds=None,select_method=None,select_cfg=None):
        '''
        Return:
            seeds: List[NodeObject]
            instruct: str, the prompt generated from the selector and seeds
        '''
        select_cfg = self.select_cfg if select_cfg is None else select_cfg
        if self.budget_type == 'design_bound':
            if self.design_budget<=0:
                print('No design budget available, stop evolution')
                exit(0)
        if select_method is None:
            select_method = select_cfg.get('select_method',DEFAULT_SELECT_METHOD)
            select_method = 'quadrant' if select_method not in SELECT_METHODS else select_method
        if n_seeds is None:
            n_seeds = self._get_n_seeds(select_cfg)
        if select_method=='quadrant':
            instruct,seeds,refs=self._quadrant_select_design(n_seeds,select_cfg=select_cfg,**selector_args)
        elif select_method=='random':
            instruct,seeds,refs=self._random_select_design(n_seeds,select_cfg=select_cfg,**selector_args)
        else:
            raise ValueError(f"Invalid select method: {select_method}")

        if len(seeds)>n_seeds: # FIXME: not sure why this happens, keep it for now
            seeds = seeds[:n_seeds]

        return instruct,seeds,refs
        
    def nodes2data(self,nodes): # convert the nodes to data: NodeObject
        return [self.ptree.G.nodes[node]['data'] for node in nodes]
    
    def _random_select_design(self,n_seeds,n_sources,select_cfg=None,allow_tree=True):
        refs=self._sample_from_sources(n_sources)
        if allow_tree:
            pool = self.ptree.filter_by_type(['ReferenceCoreWithTree','DesignArtifactImplemented'])
        else:
            pool = self.ptree.filter_by_type(['DesignArtifactImplemented'])
        seeds = self._sample_k_pool(pool,n_seeds,1,topk=False)
        seeds = [self.ptree.get_node(i) for i in seeds]
        return '',seeds,refs

    def _quadrant_select_design(self,n_seeds, n_sources: Dict[str,int],select_cfg=None):
        select_cfg = self.select_cfg if select_cfg is None else select_cfg
        instruct=''
        refs=self._sample_from_sources(n_sources)

        seed_dist = U.safe_get_cfg_dict(select_cfg,'seed_dist',DEFAULT_SEED_DIST)
        restart_prob = self._get_restart_prob(seed_dist)
        seeds = []
        if n_seeds>0:
            init_seeds = self.ptree.filter_by_type(['ReferenceCoreWithTree'])
            seed_ids = []
            if self.ptree.n_verified<=seed_dist['warmup_rounds']:
                seed_ids += random.choices(init_seeds,k=1)
            while n_seeds-len(seed_ids)>0:
                _init_seeds = list(set(init_seeds)-set(seed_ids))
                if len(_init_seeds)>0 and random.random()<restart_prob:
                    seed_ids.append(random.choice(_init_seeds))
                else:
                    _seed_ids=self._quadrant_design_seeds(n_seeds,select_cfg)
                    if _seed_ids is None:
                        _all_designs = set(self.ptree.filter_by_type(['DesignArtifactImplemented']))-set(seed_ids)
                        if len(_all_designs)>0:
                            _seed_ids = self._sample_k_pool(_all_designs,n_seeds,1,topk=False)
                        else:
                            _seed_ids = self._sample_k_pool(_init_seeds,n_seeds,1,topk=False)
                    seed_ids+=_seed_ids
                    break
            
            seed_ids = list(set(seed_ids))
            left = n_seeds - len(seed_ids)
            if left>0: # not likely to happen, but for safety
                _all_designs = set(init_seeds+self.ptree.filter_by_type(['DesignArtifactImplemented']))-set(seed_ids)
                _seed_ids = self._sample_k_pool(_all_designs,left,1,topk=False)
                seed_ids+=_seed_ids
            seeds = [self.ptree.get_node(i) for i in seed_ids]
            refs=[ref for ref in refs if ref.acronym not in seed_ids]
        return instruct,seeds,refs

    def _sample_k_pool(self,pool,k,background_noise,topk=True):
        candidates = list(pool)
        k=min(k,len(candidates))
        if k==0:
            return []
        probs = np.zeros(len(candidates))
        if topk:
            for i in range(k):
                probs[i] = k-i
        probs = probs + background_noise/len(candidates)
        probs = probs / probs.sum()
        return list(np.random.choice(candidates, size=k, replace=False, p=probs))
    
    def _random_explore_exploit(self,n_seeds,exploit_pool,explore_pool,explore_args,exclude=[]):
        explore_prob = explore_args['explore_prob']
        background_noise = explore_args['background_noise']

        if len(exclude)>0:
            exploit_pool = exploit_pool.copy()
            exploit_pool = exploit_pool[~exploit_pool['name'].isin(exclude)]
            explore_pool = explore_pool.copy()
            explore_pool = explore_pool[~explore_pool['name'].isin(exclude)]
        if len(exploit_pool)==0 and len(explore_pool)==0:
            return None,n_seeds
        if len(exploit_pool)==0:
            mode = 'explore'
        elif len(explore_pool)==0:
            mode = 'exploit'
        else:
            if random.random()<explore_prob:
                mode = 'explore'
            else:
                mode = 'exploit'

        if mode=='explore':
            pool = explore_pool['name'].tolist()
            another_pool = exploit_pool['name'].tolist()
        else:
            pool = exploit_pool['name'].tolist()    
            another_pool = explore_pool['name'].tolist()
        pool_samples = min(n_seeds,len(pool))
        another_pool_samples = min(n_seeds - pool_samples, len(another_pool))
        remaining = n_seeds - pool_samples - another_pool_samples
        
        acronyms = self._sample_k_pool(pool,pool_samples,background_noise) 
        acronyms += self._sample_k_pool(another_pool,another_pool_samples,background_noise)
        return acronyms, remaining


    def _quadrant_design_seeds(self,n_seeds,select_cfg=None):
        select_cfg = self.select_cfg if select_cfg is None else select_cfg
        ranked_quadrants = self._get_ranked_quadrants()
        if ranked_quadrants is None:
            return None
        good_design_high_confidence = ranked_quadrants['good_design_high_confidence']
        poor_design_high_confidence = ranked_quadrants['poor_design_high_confidence']

        explore_args = U.safe_get_cfg_dict(select_cfg,'design_explore_args',DEFAULT_DESIGN_EXPLORE_ARGS) 
        acronyms,remaining = self._random_explore_exploit(n_seeds,good_design_high_confidence,poor_design_high_confidence,explore_args)
        if acronyms is None:
            pool = self.ptree.filter_by_type(['ReferenceCoreWithTree','DesignArtifactImplemented'])
            return self._sample_k_pool(pool,n_seeds,1,topk=False)
        if remaining>0:
            init_seeds = self.ptree.filter_by_type(['ReferenceCoreWithTree'])
            acronyms += self._sample_k_pool(init_seeds,remaining,1,topk=False)
        return acronyms

    def _get_restart_prob(self,seed_dist):
        scheduler = seed_dist['scheduler']
        if scheduler=='constant':
            return seed_dist['restart_prob']    
        else:
            raise ValueError(f"Invalid scheduler: {scheduler}")


    def _sample_from_sources(self,n_sources: Dict[str,int],with_type=False):
        nodes={} if with_type else []
        for source_type,num in n_sources.items():
            pool = self.ptree.filter_by_type(source_type)
            num=min(num,len(pool))
            if num<=0: continue
            topk=self._sample_k_pool(pool,num,1,topk=False)
            if with_type:
                nodes[source_type]=self.nodes2data(topk)
            else:
                nodes.extend(self.nodes2data(topk))
        return nodes


    #########################  Select Verify  #########################

    def _get_exclude(self,exclude_list):
        exclude={}
        for design_scale in exclude_list: # list of (design_id,scale) being verified by other nodes
            design_id,scale = design_scale
            if scale not in exclude:
                exclude[scale]=[]
            exclude[scale].append(design_id)
        return exclude

    def _get_exclude_inv(self,exclude_list):
        exclude_inv={}
        for design_scale in exclude_list: # list of (design_id,scale) being verified by other nodes
            design_id,scale = design_scale
            if design_id not in exclude_inv:
                exclude_inv[design_id]=[]
            exclude_inv[design_id].append(scale)
        return exclude_inv

    def _get_unverified_scale_designs(self,exclude_list):
        exclude=self._get_exclude(exclude_list)
        unverified_scale_designs=self.ptree.get_unverified_designs(exclude=exclude)
        return unverified_scale_designs

    def _get_unverified_design_scales(self,exclude_list):
        exclude_inv=self._get_exclude_inv(exclude_list)
        unverified_design_scales=self.ptree.get_unverified_scales(exclude_inv=exclude_inv)
        return unverified_design_scales

    def select_verify(self,verify_strategy=None,exclude_list=[],select_cfg=None):
        design,scale=self._verify_baselines(exclude_list)
        if design is not None:
            print(f'Select unverified baseline design: {design}_{scale}')
            return design,scale
        select_cfg = self.select_cfg if select_cfg is None else select_cfg
        if verify_strategy is None:
            verify_strategy = select_cfg.get('verify_strategy',DEFAULT_VERIFY_STRATEGY)
            verify_strategy = 'quadrant' if verify_strategy not in VERIFY_STRATEGIES else verify_strategy
        available_verify_budget=self.available_verify_budget
        if len(available_verify_budget)<=0:
            if self.budget_type=='verify_bound':
                self.stream.write(f"No available verify budget found.")
                return None,None
            elif self.budget_type=='design_bound':
                available_verify_budget=self.request_temporal_budget()
            else:
                raise ValueError(f"Invalid budget type: {self.budget_type}")
        if verify_strategy=='quadrant':
            return self._quadrant_select_verify(available_verify_budget,exclude_list,select_cfg)
        elif verify_strategy=='random':
            return self._random_select_verify(available_verify_budget,exclude_list,select_cfg)  
        else:
            raise ValueError(f"Invalid verify strategy: {verify_strategy}")
        
    def _verify_baselines(self,exclude_list=[]):
        self.ptree.update_baselines()
        baselines = self.ptree.filter_by_type(['ReferenceCore','ReferenceCoreWithTree'])
        for scale in self.target_scales:
            mult=self.token_mults[scale]
            for acronym in baselines:
                if (acronym,scale) in exclude_list:
                    continue
                node = self.ptree.get_node(acronym)
                if not node.verifications:
                    return acronym,scale
                if scale not in node.verifications:
                    return acronym,scale
                if mult not in node.verifications[scale]:
                    return acronym,scale
        return None,None
            

    def _random_select_verify(self,available_verify_budget,exclude_list=[],select_cfg=None):
        unverified_by_scale=self._get_unverified_scale_designs(exclude_list) # indexed by scale

        unverified_by_scale={k:v for k,v in unverified_by_scale.items() if len(v)>0}
        n_unverified=sum([len(v) for v in unverified_by_scale.values()])
        if n_unverified==0:
            return None,None
        scales=list(unverified_by_scale.keys())
        lowest_scale = sorted(scales,key=lambda x:int(x.replace('M','')))[0]
        design_id = random.choice(unverified_by_scale[lowest_scale])
        return design_id,lowest_scale


    def _quadrant_select_verify(self,available_verify_budget,exclude_list=[],select_cfg=None): # exclude_list is a list of (design_id,scale) being verified by other nodes        
        unverified_by_scale=self._get_unverified_scale_designs(exclude_list) # indexed by scale
        unverified_by_scale={k:v for k,v in unverified_by_scale.items() if len(v)>0}
        n_unverified=sum([len(v) for v in unverified_by_scale.values()])
        if n_unverified==0:
            return None,None
        verify_all = select_cfg.get('verify_all',True)
        if verify_all:
            unverified_14M=unverified_by_scale.get('14M',[])
            if len(unverified_14M)>0:
                design_id=random.choice(unverified_14M)
                return design_id,'14M'
        # Now all the designs are at least verified at 14M
        unverified_by_design=self._get_unverified_design_scales(exclude_list) # indexed by design_id
        ranked_quadrants = self._get_ranked_quadrants()

        unverified_scales=[i for i in unverified_by_scale.keys() if i in available_verify_budget]
        unverified_scales.sort(key=lambda x:int(x.replace('M','')))
        lowest_scale=unverified_scales[0]
        lowest_pool = unverified_by_scale[lowest_scale]

        if ranked_quadrants is None:
            acronym = random.choice(lowest_pool)
            return acronym,lowest_scale

        good_design_low_confidence = ranked_quadrants['good_design_low_confidence']
        poor_design_low_confidence = ranked_quadrants['poor_design_low_confidence']

        explore_args = U.safe_get_cfg_dict(self.select_cfg,'verify_explore_args',DEFAULT_VERIFY_EXPLORE_ARGS) 
        exclude_design=[]
        while True:
            acronyms,_ = self._random_explore_exploit(1,good_design_low_confidence,poor_design_low_confidence,explore_args,exclude_design)
            if acronyms is None: # randomly select a design from the lowest scale
                break
            acronym = acronyms[0]
            scales = unverified_by_design[acronym]
            scale = sorted(scales,key=lambda x:int(x.replace('M','')))[0]
            if scale in available_verify_budget:
                return acronym,scale
            else:
                exclude_design.append(acronym)
        # rank unverified lowest scale designs
        vectors = {i:self.design_vectors[i] for i in lowest_pool}
        design_rank = self._rank_designs(vectors)
        if design_rank is None or design_rank.eZmpty or len(design_rank)==0: # unlikely happen, but for safety, randomly select a design from the lowest scale
            acronym = random.choice(lowest_pool)
            self.stream.write(f"No design ranks available now, randomly select a design from the lowest scale: {lowest_scale}")
            return acronym,lowest_scale
        acronym = design_rank.iloc[0]['name']
        return acronym,lowest_scale


    @property
    def verify_budget(self):
        vb = self.ptree.budget_status(self._verify_budget)
        vb=sorted(vb.items(),key=lambda x:int(x[0].replace('M','')))
        vb = {k:v for k,v in vb}
        return vb

    @property
    def available_verify_budget(self):
        budget=self.verify_budget
        return {k:v for k,v in budget.items() if v>0}

    def request_temporal_budget(self): # keep selection ratios
        _,used=self.ptree.budget_status(self._verify_budget,ret_verified=True)
        exceeded={}
        for scale in used:
            exceeded[scale]=used[scale]-self._verify_budget[scale]
        scales=list(exceeded.keys())
        scales.sort(key=lambda x:int(x.replace('M','')))
        selection_ratios = {}
        # budget current scale / budget previous scale, lowest scale is 1
        for i in range(len(scales)):
            if i==0:
                selection_ratios[scales[i]]=1
            else:
                selection_ratios[scales[i]]=self._verify_budget[scales[i]]/self._verify_budget[scales[i-1]]

        def dict_geq(d1,d2):
            for k in d1:
                if k not in d2:
                    continue
                if d1[k]<d2[k]:
                    return False
            return True
        
        def dict_add(d1,d2):
            return {k:d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}

        def dict_sub(d1,d2):
            return {k:d1.get(k,0)-d2.get(k,0) for k in set(d1)|set(d2)}
        
        def scale_to_int(scale):
            return int(scale.replace('M',''))
        
        def get_lower_half(scale,scales):
            lower=[s for s in scales if scale_to_int(s)<=scale_to_int(scale)]
            lower.sort(key=scale_to_int)
            return lower

        def try_assign(scale,scales):
            assign={s:0 for s in scales}
            lower=get_lower_half(scale,scales)
            budget=1
            for s in lower[::-1]:
                assign[s]=budget
                budget=int(budget/selection_ratios[s])
            return assign

        assign={}
        found=False
        while not found:
            # scan from small scale to large scale
            for scale in scales:
                _assign=copy.deepcopy(assign)
                _assign=dict_add(_assign,try_assign(scale,scales))
                if dict_geq(_assign,exceeded):
                    found=True
                    break
            assign=_assign # next round with full budget or found

        remaining = dict_sub(assign,exceeded)
        available_budget = {k:v for k,v in remaining.items() if v>0}
        return available_budget
            
        
