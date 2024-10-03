from typing import List, Tuple, Dict
import random

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


class Selector:
    def __init__(self,ptree,select_cfg,_verify_budget,stream):
        self.ptree=ptree
        self.select_cfg=select_cfg
        self._verify_budget=_verify_budget
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


    def _get_design_scores(self):
        design_vectors = self.ptree.get_design_vectors()
        raise NotImplementedError('Need to implement the design score function')


    #########################  Select Verify  #########################

    def select_verify(self,verify_strategy=None,exclude_list=[]):
        if verify_strategy is None:
            verify_strategy = self.select_cfg.get('verify_strategy',DEFAULT_SELECT_METHOD)
        exclude={}
        for design_id,scale in exclude_list: # list of (design_id,scale) being verified by other nodes
            if scale not in exclude:
                exclude[scale]=[]
            exclude[scale].append(design_id)
        if verify_strategy=='random':
            for scale in self.available_verify_budget:
                unverified=self.ptree.get_unverified_designs(scale)
                unverified=[i for i in unverified if i not in exclude.get(scale,[])]
                if len(unverified)==0:
                    self.stream.write(f"No unverified design at scale {scale}.")
                else:
                    design_id=random.choice(unverified)
                    return design_id,scale
            self.stream.write(f"No unverified design found at any scale.")
            return None,None
        else:
            raise ValueError(f"Invalid verify strategy: {verify_strategy}")
    
    
    @property
    def verify_budget(self):
        vb = self.ptree.remaining_budget(self._verify_budget)
        vb=sorted(vb.items(),key=lambda x:int(x[0].replace('M','')))
        vb = {k:v for k,v in vb}
        return vb

    @property
    def available_verify_budget(self):
        budget=self.verify_budget
        return {k:v for k,v in budget.items() if v>0}
