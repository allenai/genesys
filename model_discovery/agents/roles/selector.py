from typing import List, Tuple, Dict
import random
import copy
import numpy as np

import model_discovery.utils as U
from model_discovery.system import DesignModes

SEED_TYPES = ['DesignArtifactImplemented','ReferenceCoreWithTree']

DEFAULT_SELECT_METHOD = 'random'
DEFAULT_VERIFY_STRATEGY = 'random'


# https://huggingface.co/docs/timm/en/reference/schedulers Similar to lr schedulers
DEFAULT_SEED_DIST = {
    'scheduler':'constant', # linear or cosine
    'restart_prob':0.2, # the chance of sampling from the seeds again
    'warmup_rounds':10, # the number of rounds to warmup the scheduler
}
SCHEDULER_OPTIONS = ['constant']

DEFAULT_JUMP_PROB = {
    '14M':[0.1,0.05,0.025], # 10% to 35, 5% to 70, 2.5% to 130, 82.5% to stay
    '35M':[0.1,0.05], # 10% to 70, 5% to 130M, 85% to stay
    '70M':[0.1], # 10% to 130M, 90% to stay
}


class Selector:
    def __init__(self,ptree,select_cfg,_verify_budget,selection_ratio,stream,allow_temporal_budget=True):
        self.ptree=ptree
        self.select_cfg=select_cfg
        self._verify_budget=_verify_budget
        self.selection_ratio=selection_ratio
        self.allow_temporal_budget=allow_temporal_budget
        self.stream=stream

    #########################  Select Design  #########################

    def select_design(self,selector_args,mode=None,select_method=None):
        '''
        Return:
            seeds: List[NodeObject]
            instruct: str, the prompt generated from the selector and seeds
        '''
        if mode is None:
            mode=DesignModes.MUTATION
        if select_method is None:
            select_method = self.select_cfg.get('select_method',DEFAULT_SELECT_METHOD)
        if mode==DesignModes.MUTATION:
            if select_method=='random':
                instruct,seeds,refs=self._random_select_mutate(**selector_args)
            else:
                raise ValueError(f"Invalid select method: {select_method}")
        elif mode==DesignModes.SCRATCH:
            raise NotImplementedError('Scratch mode is unstable, do not use it')
        elif mode==DesignModes.CROSSOVER:
            raise NotImplementedError('Crossover mode is not implemented')
        else:
            raise ValueError(f'Invalid design mode: {mode}')

        return instruct,seeds,refs
        
    def nodes2data(self,nodes): # convert the nodes to data: NodeObject
        return [self.ptree.G.nodes[node]['data'] for node in nodes]

    def _random_select_mutate(self,n_sources: Dict[str,int]):
        instruct=''
        refs=self._sample_from_sources(n_sources)

        seed_dist = U.safe_get_cfg_dict(self.select_cfg,'seed_dist',DEFAULT_SEED_DIST)
        _seeds=self._sample_from_sources({i:1 for i in SEED_TYPES},with_type=True)
        
        restart_prob = self._get_restart_prob(seed_dist)
        if random.random()<restart_prob: # use ReferenceCoreWithTree as seeds
            seeds=_seeds['ReferenceCoreWithTree']
        else:
            seeds=_seeds['DesignArtifactImplemented']
        seed_ids=[i.acronym for i in seeds]
        refs=[ref for ref in refs if ref.acronym not in seed_ids]
        return instruct,seeds,refs

    def _get_restart_prob(self,seed_dist):
        scheduler = seed_dist['scheduler']
        if scheduler=='constant':
            designs=self.ptree.filter_by_type('DesignArtifactImplemented')
            if len(designs)<seed_dist['warmup_rounds']:
                return 1.0
            else:
                return seed_dist['restart_prob']    
        else:
            raise ValueError(f"Invalid scheduler: {scheduler}")


    def _sample_from_sources(self,n_sources: Dict[str,int],with_type=False):
        nodes={} if with_type else []
        for source_type,num in n_sources.items():
            num=min(num,len(self.ptree.filter_by_type(source_type)))
            if num<=0: continue
            topk=random.sample(self.ptree.filter_by_type(source_type),num)
            if with_type:
                nodes[source_type]=self.nodes2data(topk)
            else:
                nodes.extend(self.nodes2data(topk))
        return nodes

    def report_to_metrics(report):
        metrics={}
        trainer_state=report['trainer_state.json']
        eval_results=report['eval_results.json']
        
        metrics['perf']={}
        metrics['perf']['total_train_flos']=trainer_state['total_flos']
        metrics['perf']['total_eval_time']=float(eval_results['total_evaluation_time_seconds'])
        metrics['perf']['num_params']=eval_results['config']['model_num_parameters']
        metrics['eval']={}
        for task in eval_results['results']:
            res=eval_results['results'][task]
            task_alias=res['alias']
            if 'perplexity,none' in res:
                metrics['eval'][f'{task_alias}_ppl']=-np.log10(res['perplexity,none']) # the lower the better
            elif 'acc_norm,none' in res: 
                metrics['eval'][f'{task_alias}_acc_norm']=res['acc_norm,none']
            elif 'acc,none' in res:
                metrics['eval'][f'{task_alias}_acc']=res['acc,none']
        
        return metrics

    def _get_design_scores(self):
        design_vectors = self.ptree.get_design_vectors()
        raise NotImplementedError('Need to implement the design score function')


    #########################  Select Verify  #########################

    def _get_exclude(self,exclude_list):
        exclude={}
        for design_id,scale in exclude_list: # list of (design_id,scale) being verified by other nodes
            if scale not in exclude:
                exclude[scale]=[]
            exclude[scale].append(design_id)
        return exclude

    def _get_exclude_inv(self,exclude_list):
        exclude_inv={}
        for design_id,scale in exclude_list: # list of (design_id,scale) being verified by other nodes
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

    def select_verify(self,verify_strategy=None,exclude_list=[]):
        if verify_strategy is None:
            verify_strategy = self.select_cfg.get('verify_strategy',DEFAULT_SELECT_METHOD)
        exclude=self._get_exclude(exclude_list)
        if verify_strategy=='random':
            return self.random_select_verify(exclude)
        else:
            raise ValueError(f"Invalid verify strategy: {verify_strategy}")

    def random_select_verify(self,exclude_list=[]): # exclude_list is a list of (design_id,scale) being verified by other nodes
        available_verify_budget=self.available_verify_budget
        if len(available_verify_budget)==0:
            self.stream.write(f"No available verify budget found.")
            return None,None
        # unverified=self._get_unverified_scale_designs(exclude_list) # indexed by scale
        unverified=self._get_unverified_design_scales(exclude_list) # indexed by design_id
        if len(unverified)==0:
            self.stream.write(f"No unverified design found at any scale.")
            return None,None
        else:
            # select a random design
            if 'jump_prob' not in self.select_cfg:
                self.select_cfg['jump_prob']=DEFAULT_JUMP_PROB
            for scale in DEFAULT_JUMP_PROB:
                if scale not in self.select_cfg['jump_prob']:
                    self.select_cfg['jump_prob'][scale]=DEFAULT_JUMP_PROB[scale]
            design_id=random.choice(list(unverified.keys()))

            available_scales=[]
            for scale in unverified[design_id]:
                if scale in available_verify_budget:
                    available_scales.append(scale)
            if len(available_scales)==0:
                self.stream.write(f"No available verify scale found for design {design_id}.")
                return None,None
            scale=random.choice(available_scales)
            return design_id,scale


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


    def request_temporal_budget(self): # keep selection ratio
        _,used=self.ptree.budget_status(self._verify_budget,ret_verified=True)
        exceeded={}
        for scale in used:
            exceeded[scale]=used[scale]-self._verify_budget[scale]
        scales=list(exceeded.keys())
        scales.sort(key=lambda x:int(x.replace('M','')))

        def dict_geq(d1,d2):
            for k in d1:
                if k not in d2:
                    continue
                if d1[k]<d2[k]:
                    return False
            return True
        
        def dict_add(d1,d2):
            return {k:d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}
        
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
                budget=int(budget/self.selection_ratio)
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

        return assign
            
        
