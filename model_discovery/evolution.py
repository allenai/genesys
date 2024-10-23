''' System of R & D Agents and Selector for Scale Climbing Evolution '''

from __future__ import annotations

import os
import sys
try: # a stupid patch for windows
    from .secrets import *
    os.environ['MY_OPENAI_KEY']=MY_OPENAI_KEY
    os.environ['ANTHROPIC_API_KEY']=ANTHROPIC_API_KEY
    os.environ['HF_KEY']=HF_KEY
    os.environ['HF_HUB_KEY']=HF_HUB_KEY
    os.environ['GITHUB_TOKEN']=GITHUB_TOKEN
    os.environ['WANDB_API_KEY']=WANDB_API_KEY
    os.environ['S2_API_KEY']=S2_API_KEY
    os.environ['AWS_SECRET_ACCESS_KEY']=AWS_SECRET_ACCESS_KEY
    os.environ['AWS_ACCESS_KEY_ID']=AWS_ACCESS_KEY_ID
    os.environ['MATHPIX_API_ID']=MATHPIX_API_ID
    os.environ['UNSTRUCTURED_API_ID']=UNSTRUCTURED_API_ID
    os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
    os.environ['COHERE_API_KEY']=COHERE_API_KEY
    os.environ['PERPLEXITY_API_KEY']=PERPLEXITY_API_KEY
    os.environ['DATA_DIR']=DATA_DIR
    os.environ['CKPT_DIR']=CKPT_DIR
    os.environ['DB_KEY_PATH']=DB_KEY_PATH
    os.environ['HF_DATASETS_TRUST_REMOTE_CODE']='1'    
except:
    pass

import sys
import re
import exec_utils
import pathlib
from datetime import datetime, timedelta, timezone
import json
import copy
from enum import Enum
import time
import tempfile
from dataclasses import dataclass, field, asdict
import networkx as nx
import pandas as pd
import numpy as np
from io import StringIO
import hashlib
import random
# from networkx.drawing.nx_pydot import to_pydot,warnings
from pyvis.network import Network
import math
import multiprocessing
import shutil
from google.cloud import firestore


try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


from types import ModuleType
from typing import (
    Type,
    List,
    Dict,
    Any,
    TypeVar,
    Tuple,
    Optional,
    Union
)
from .system import BuildSystem,PrintSystem,DesignModes,RunningModes,DESIGN_TERMINAL_STATES,DESIGN_ACTIVE_STATES,DESIGN_ZOMBIE_THRESHOLD
from exec_utils.factory import _check_config
from exec_utils import BuildSystem as NativeBuild
from exec_utils.aliases import ConfigType
from google.cloud.firestore import DELETE_FIELD

from model_discovery.agents.roles.selector import Selector,DEFAULT_SELECT_METHOD,DEFAULT_VERIFY_STRATEGY

from model_discovery.model.composer import GAUTree,GAUDict,UnitSpec,UnitDecl
from model_discovery import utils as U
from .configs.gam_config import ( 
    GAMConfig,GAMConfig_14M,GAMConfig_31M,GAMConfig_70M,GAMConfig_125M,GAMConfig_350M,GAMConfig_760M,
    GAMConfig_1300M,GAMConfig_2700M,GAMConfig_6700M,GAMConfig_13B,GAMConfig_175B,GAMConfig_1T,GAMConfig_debug
)
from .configs.const import *
from .ve.run import main as ve_main
from .ve.run import parser as ve_parser
from .ve.run import get_history_report

from model_discovery.agents.flow.gau_flows import GAU_TEMPLATE

__all__ = [
    "EvolutionSystem",
    "BuildEvolution",
]




class FirestoreManager:
    def __init__(self, ptree, evoname, db_dir, remote_db):
        self.ptree = ptree
        self.db_dir = db_dir
        self.evoname = evoname
        self.remote_db = remote_db
        self.collection=remote_db.collection(evoname)
        self.baseline_collection=remote_db.collection('baselines')
        self.key_dict={
            'metadata':'m',
            'proposal':'p',
            'implementation':'i',
            'proposal_traces':'pt',
            'implementation_history':'ih',
            'verifications':'v',
            'verification_report':'vr',
            'eval_results.json':'er',
            'trainer_state.json':'ts',
            'wandb_ids.json':'wi',
            'rounds':'r'
        }
        for i in range(100): # should definitely be enough
            self.key_dict[f'trace_{i}']=f't{i}'
            self.key_dict[f'{i}_rounds']=f'{i}r'
        self.key_dict_inv={v: k for k, v in self.key_dict.items()}
        self.cache={}
        self.index=None
        self.log_doc_ref = self.remote_db.collection('experiment_logs').document(self.evoname)
        self.sync_to_db(verbose=False) # see if there is anything to upload
        self.updated_terms=[]

    def compress_index(self,data):
        return U.translate_dict_keys(data,self.key_dict,allow_missing=True)

    def decompress_index(self,data):
        return U.translate_dict_keys(data,self.key_dict_inv,allow_missing=True)
    
    def fix_index(self):
        for id in self.index:
            if 'verifications' in self.index[id]:
                error_scales = []
                for scale in list(self.index[id]['verifications'].keys()):
                    if ('wandb_ids.json' not in self.index[id]['verifications'][scale] or
                        'trainer_state.json' not in self.index[id]['verifications'][scale] or
                        'eval_results.json' not in self.index[id]['verifications'][scale]):
                        error_scales.append(scale)
                
                if error_scales:
                    # index_updates = {}
                    for scale in error_scales:
                        # Remove the scale from the local index
                        self.index[id]['verifications'].pop(scale)
                        print(f'Deleted error verification data for scale {scale} in design {id}')
                        
                        # # Prepare to delete the scale from the Firestore index
                        # index_updates[f'verifications.{scale}'] = DELETE_FIELD
                        
                        # Delete the verification data from the database
                        self.collection.document(id).collection('verifications').document(scale).delete()
                        
                        # Delete the verification reports from local
                        report_path = U.pjoin(self.db_dir, 'designs', id, 'verifications', f'{scale}.json')
                        if U.pexists(report_path):
                            os.remove(report_path)
                    
                    # # Update the index document for this design
                    # if index_updates: # WILL TRIGGER NESTED DELETE ERROR
                    #     self.collection.document('index').update({id: index_updates})

    def upload_experiment(self,exp,config=None):
        collection=self.remote_db.collection('experiments')
        to_set={}
        if config:
            to_set['config']=config
        if len(to_set)>0:
            collection.document(exp).set(to_set,merge=True)

    def download_experiment(self,ckpt_dir,exp):
        collection=self.remote_db.collection('experiments')
        doc=collection.document(exp).get()
        if doc.exists:
            doc_id=doc.id
            doc=doc.to_dict()
            config=doc.get('config',{})
            U.mkdir(U.pjoin(ckpt_dir,doc_id))
            U.mkdir(U.pjoin(ckpt_dir,doc_id,'ve'))
            U.mkdir(U.pjoin(ckpt_dir,doc_id,'db','sessions'))
            U.mkdir(U.pjoin(ckpt_dir,doc_id,'db','designs'))
            if config:
                U.save_json(config,U.pjoin(ckpt_dir,doc_id,'config.json'))

    def get_index(self,is_baseline=False):
        collection = self.baseline_collection if is_baseline else self.collection
        index=collection.document('index').get().to_dict()
        if index is None:
            index={}
        _index=self.decompress_index(index)
        if is_baseline:
            self.baseline_index=_index
        else:
            if self.index is None:  
                self.index=_index
            else:
                for id in _index:
                    if id not in self.updated_terms:
                        self.updated_terms.append(id)
                    else:
                        _term=_index[id]
                        term=self.index[id]
                        if U.dict_diff(term,_term):
                            self.updated_terms.append(id)
                self.index=_index
            self.fix_index()
        return self.index
        
    def get_baseline_index(self):
        self.get_index(is_baseline=True)
        return self.baseline_index

    def update_index(self,merge=True,is_baseline=False):
        collection = self.baseline_collection if is_baseline else self.collection
        Index = self.baseline_index if is_baseline else self.index
        self.safe_upload(collection.document('index'),self.compress_index(Index),merge=merge)

    def doc_to_design(self,doc):
        design_data = doc.to_dict()
        proposal_traces = self.get_subcollection(doc.reference, 'proposal_traces')
        if proposal_traces:
            design_data['proposal_traces'] = proposal_traces
        verifications = self.get_subcollection(doc.reference, 'verifications')
        if verifications:
            design_data['verifications'] = verifications
            for scale in verifications:
                verification_reports = self.get_subcollection(doc.reference, 'verifications',scale, 'verification_report')
                if verification_reports:
                    design_data['verifications'][scale]['verification_report'] = verification_reports
        if 'implementation' in design_data:
            implementation_history = self.get_subcollection(doc.reference, 'implementation_history')
            design_data['implementation']['history'] = [implementation_history[i] for i in sorted(implementation_history.keys())]
        codes = self.get_subcollection(doc.reference, 'codes')
        if codes:
            design_data['codes'] = codes
        return design_data

    def is_verified(self, design_id, scale):
        self.get_index()
        if design_id in self.index:
            if 'verifications' in self.index[design_id]:
                if scale in self.index[design_id]['verifications']:
                    return True
        return False
    
    def get_design(self, design_id, use_cache=True):
        if use_cache and design_id in self.cache:
            design_data=self.cache[design_id]
        else:
            doc=self.collection.document(design_id).get()
            design_data=self.doc_to_design(doc)
            self.cache[design_id]=design_data # update cache
        metadata=design_data['metadata']
        proposal = Proposal.from_dict(design_data['proposal']) if 'proposal' in design_data else None
        implementation = Implementation.from_dict(design_data['implementation']) if 'implementation' in design_data else None
        verifications = {scale: Verification.from_dict(design_data['verifications'][scale]) for scale in design_data['verifications']} if 'verifications' in design_data else {}
        codes = {scale: design_data['codes'][scale] for scale in design_data['codes']} if 'codes' in design_data else {}
        return DesignArtifact(proposal=proposal, implementation=implementation, verifications=verifications, codes=codes, **metadata)

    def get_subcollection(self, doc_ref, subcollection_name):
        subcollection = doc_ref.collection(subcollection_name).get()
        return {doc.id: doc.to_dict() for doc in subcollection}

    def safe_upload(self, ref, data, merge=True):
        """
        Safely upload data to Firestore.
        
        :param ref: Firestore document reference
        :param data: Data to upload
        :param overwrite: Whether to overwrite existing data
        :return: True if upload was successful, False otherwise
        """
        try:
            ref.set(data, merge=merge)
            return True
        except Exception as e:
            print(f"Error uploading data: {e}")
            return False

    def upload_key_data(self, design_id, key, data, overwrite=False, verbose=False):
        key=str(key)
        design_ref = self.collection.document(design_id)
        if design_id not in self.index:
            self.index[design_id]={}
        upload=True
        if key in self.index[design_id] and not overwrite:
            upload=False
            if verbose:
                print(f'Key "{key}" already exists in design "{design_id}"')
        if upload:
            if self.safe_upload(design_ref, {key: data}):
                self.index[design_id][key]=1
                print(f'Uploaded "{key}" for design "{design_id}" successfully')
            else:
                print(f'Failed to upload "{key}" for design "{design_id}"')

    def upload_collection_key_data(self, design_id, collection_name, key, data, overwrite=False, verbose=False,collection=None):
        key=str(key)
        collection = collection if collection is not None else self.collection
        design_ref = collection.document(design_id)
        data_ref = design_ref.collection(collection_name).document(str(key))
        if design_id not in self.index:
            self.index[design_id]={}
        if collection_name not in self.index[design_id]:
            self.index[design_id][collection_name]={}
        upload=True
        if key in self.index[design_id][collection_name] and not overwrite:
            upload=False
            if verbose:
                print(f'Key "{key}" already exists in design "{design_id}" collection "{collection_name}"')
        if upload:
            if self.safe_upload(data_ref, {key: data}):
                self.index[design_id][collection_name][key]=1
                print(f'Uploaded "{key}" for design "{design_id}" collection "{collection_name}" successfully')
            else:
                print(f'Failed to upload "{key}" for design "{design_id}" collection "{collection_name}"')
            
    def upload_implementation(self, design_id, implementation, overwrite=False,verbose=False):
        history=implementation.pop('history')
        self.upload_key_data(design_id,'implementation',implementation,overwrite,verbose=verbose)
        for idx,step in enumerate(history):
            if 'rounds' not in step:
                step['rounds']=[]
            rounds=step.pop('rounds')
            self.upload_collection_key_data(design_id,f'implementation_history',idx,step,overwrite,verbose=verbose)
            step_rounds_collection = self.collection.document(design_id).collection('implementation_history').document(str(idx)).collection('rounds')
            rounds_idx = f'{idx}_rounds'
            if rounds_idx not in self.index[design_id]['implementation_history']:
                self.index[design_id]['implementation_history'][rounds_idx]={}
            for round_idx,round_data in enumerate(rounds):
                upload=True
                if str(round_idx) in self.index[design_id]['implementation_history'][rounds_idx] and not overwrite:
                    upload=False
                    if verbose:
                        print(f'Round "{round_idx}" already exists in design "{design_id}" implementation history "{idx}"')
                if upload:
                    if self.safe_upload(step_rounds_collection.document(str(round_idx)),round_data):
                        self.index[design_id]['implementation_history'][rounds_idx][str(round_idx)]=1
                        print(f'Uploaded round "{round_idx}" for design "{design_id}" implementation history "{idx}"')
                    else:
                        print(f'Failed to upload round "{round_idx}" for design "{design_id}" implementation history "{idx}"')

    def upload_verification(self, design_id, verification, scale, overwrite=False, verbose=False,is_baseline=False):
        collection = self.baseline_collection if is_baseline else self.collection
        Index = self.baseline_index if is_baseline else self.index  
        reports=verification.pop('verification_report')
        if 'eval_results.json' not in reports:
            return
        if 'trainer_state.json' not in reports:
            return
        if design_id not in Index:
            Index[design_id]={}
        if 'verifications' not in Index[design_id]:
            Index[design_id]['verifications']={}
        upload=True
        if scale in Index[design_id]['verifications'] and not overwrite:   
            upload=False
            if verbose:
                print(f'Verification for scale "{scale}" already exists in design "{design_id}"')
        if upload:
            Index[design_id]['verifications'][scale]={}        
            if self.safe_upload(collection.document(design_id).collection('verifications').document(scale),verification):
                print(f'Uploaded verification metadata for scale "{scale}" in design "{design_id}"')
            else:
                print(f'Failed to upload verification metadata for scale "{scale}" in design "{design_id}"')
        for key,report in reports.items():
            if key in ['training_record.csv','system_metrics.csv']:
                continue # XXX: skip these files for now
            upload=True        
            if key in Index[design_id]['verifications'][scale] and not overwrite:
                if key in ['training_record.csv','system_metrics.csv']: continue
                upload=False
                if verbose:
                    print(f'Verification report for scale "{scale}" and key "{key}" already exists in design "{design_id}"')
            if upload:
                if self.safe_upload(collection.document(design_id).collection('verifications').document(scale).collection('verification_report').document(key),report):
                    Index[design_id]['verifications'][scale][key]=1
                    print(f'Uploaded verification report for scale "{scale}" and key "{key}" in design "{design_id}"')
                else:
                    print(f'Failed to upload verification report for scale "{scale}" and key "{key}" in design "{design_id}"')
        if is_baseline:
            self.baseline_index=Index
        else:
            self.index=Index

    def load_design_local(self, design_id):
        design_path = os.path.join(self.db_dir, 'designs', design_id)
        if not U.pexists(design_path):
            print(f"Design '{design_id}' not found in local database.")
            return None
        design={}
        
        metadata_path = os.path.join(design_path, 'metadata.json')
        if os.path.exists(metadata_path):
            design['metadata']=U.load_json(metadata_path)

        proposal_path = os.path.join(design_path, 'proposal.json')
        if os.path.exists(proposal_path):
            design['proposal']=U.load_json(proposal_path)
            design['proposal_traces']={}
            traces_path = os.path.join(design_path, 'proposal_traces')
            if U.pexists(traces_path):
                for trace in os.listdir(traces_path):
                    if trace.endswith('.json'):
                        trace_data=U.load_json(U.pjoin(traces_path,trace))
                        trace_id=trace.split('.')[0]
                        design['proposal_traces'][trace_id]=trace_data

        implementation_path = os.path.join(design_path, 'implementation.json')
        if os.path.exists(implementation_path):
            design['implementation']=U.load_json(implementation_path)

        verifications_path = os.path.join(design_path, 'verifications')
        if U.pexists(verifications_path):
            design['verifications']={}
            for verification in os.listdir(verifications_path):
                if verification.endswith('.json'):
                    verification_data=U.load_json(U.pjoin(verifications_path,verification))
                    if 'verification_report' not in verification_data:
                        continue
                    verification_report = verification_data['verification_report']
                    if 'wandb_ids.json' not in verification_report:
                        continue
                    if 'trainer_state.json' not in verification_report:
                        continue
                    if 'eval_results.json' not in verification_report:
                        continue
                    scale=verification.split('.')[0]
                    design['verifications'][scale]=verification_data

        codes_path=U.pjoin(design_path,'codes.json')
        if U.pexists(codes_path):
            design['codes']=U.load_json(codes_path)

        return design

    def upload_metadata(self,design_id,metadata,overwrite=False,verbose=False):
        self.upload_key_data(design_id,'metadata',metadata,overwrite,verbose=verbose)

    def upload_proposal(self,design_id,proposal,overwrite=False,verbose=False):
        self.upload_key_data(design_id,'proposal',proposal,overwrite,verbose=verbose)
    
    def upload_proposal_traces(self,design_id,proposal_traces,overwrite=False,verbose=False):
        for trace_id,trace in proposal_traces.items():
            self.upload_collection_key_data(design_id,'proposal_traces',trace_id,trace,overwrite,verbose=verbose)

    def upload_design(self, design_id, design, overwrite=False, upload_index=True, verbose=False):
        """
        Upload a single design to Firestore.
        
        :param design_id: The acronym of the design
        :param overwrite: Whether to overwrite existing data
        """
        # Upload metadata.json
        if 'metadata' in design:
            self.upload_metadata(design_id,design['metadata'],overwrite,verbose=verbose)

        # Upload proposal.json
        if 'proposal' in design:
            self.upload_proposal(design_id,design['proposal'],overwrite,verbose=verbose)
            self.upload_proposal_traces(design_id,design['proposal_traces'],overwrite,verbose=verbose)

        # Upload implementation.json
        if 'implementation' in design:
            self.upload_implementation(design_id,design['implementation'],overwrite,verbose=verbose)

        # Upload verifications
        if 'verifications' in design:
            for scale,verification in design['verifications'].items():
                self.upload_verification(design_id,verification,scale,overwrite,verbose=verbose)

        # Upload codes.json
        if 'codes' in design:
            for scale,code in design['codes'].items():  
                self.upload_collection_key_data(design_id,'codes',scale,code,overwrite,verbose=verbose)

        if verbose:
            print(f"Upload of design '{design_id}' completed.")
        if upload_index:
            self.update_index()
    
    def to_session_progress(self,sessdata):
        proposed=sessdata['proposed']
        reranked=sessdata['reranked']
        return f'{proposed}, {reranked}'

    def upload_design_session(self,sess_id,sessdata,overwrite=False,verbose=False):
        log_collection=self.log_doc_ref.collection('design_sessions')
        log_ref = log_collection.document(sess_id)
        log_ref.set(sessdata,merge=True)
        index_ref = log_collection.document('index')
        index_ref.set({sess_id:{'progress':self.to_session_progress(sessdata)}},merge=True)
        print(f'Uploaded session {sess_id} to DB')

    def get_design_session(self,sess_id):
        log_collection=self.log_doc_ref.collection('design_sessions')
        log_ref = log_collection.document(sess_id).get()
        if not log_ref.exists:
            return None
        return log_ref.to_dict()
    
    def upload_baselines(self,overwrite=False,verbose=False):
        corerefs=self.ptree.filter_by_type(['ReferenceCore','ReferenceCoreWithTree'])
        baseline_index=self.get_baseline_index()
        for coreref in corerefs:
            node=self.ptree.get_node(coreref)
            index_term=baseline_index.get(coreref,{})
            if node.verifications:
                for scale,verifications in node.verifications.items():
                    for mult,verification in verifications.items():
                        scale = f'{scale}_{mult}'
                        if 'verifications' in index_term and scale in index_term['verifications']:
                            continue
                        self.upload_verification(coreref,verification.to_dict(),scale,overwrite=overwrite,verbose=verbose,is_baseline=True)
                        print(f'Uploaded verification for scale {scale} with mult {mult} in baseline {coreref}')
        self.update_index(is_baseline=True)

    def download_baselines(self,overwrite=False,verbose=False):
        self.get_baseline_index()
        print(self.baseline_index)
        for acronym in self.baseline_index:
            index_term=self.baseline_index[acronym]
            if 'verifications' in index_term:
                for scale_mult in index_term['verifications']:
                    scale,mult=scale_mult.split('_')
                    coreref_dir=self.ptree.coreref_dir(acronym,mult)
                    verification_path=U.pjoin(coreref_dir,'verifications',scale+'.json')
                    if not U.pexists(verification_path) or overwrite:
                        U.mkdir(U.pjoin(coreref_dir,'verifications'))
                        verification=self.baseline_collection.document(acronym).collection('verifications').document(scale_mult).get().to_dict()
                        if not verification:
                            continue
                        verification['verification_report']={}
                        for key in index_term['verifications'][scale_mult]:
                            report=self.baseline_collection.document(acronym).collection('verifications').document(scale_mult).collection('verification_report').document(key).get().to_dict()
                            verification['verification_report'][key]=report
                        U.save_json(verification,verification_path)
                        print(f'Downloaded verification for scale {scale} and mult {mult} in baseline {acronym}')
            
    
    def sync_sessions_to_db(self,overwrite=False,verbose=False):
        log_collection=self.log_doc_ref.collection('design_sessions')
        index_ref = log_collection.document('index')
        index = index_ref.get().to_dict()
        if index is None:
            return
        sessions_dir=U.pjoin(self.db_dir,'sessions')
        sessions=os.listdir(sessions_dir)
        for sess_id in sessions:
            sessdata = U.load_json(U.pjoin(sessions_dir,sess_id, 'metadata.json'))
            if sessdata:
                if sess_id in index:
                    if index[sess_id].get('progress','')==self.to_session_progress(sessdata):
                        continue
                self.upload_design_session(sess_id,sessdata,overwrite=overwrite,verbose=verbose)

    def sync_to_db(self,overwrite=False,verbose=False): # upload all local designs to db
        self.get_index()
        designs=os.listdir(U.pjoin(self.db_dir,'designs'))
        for design_id in designs:
            design=self.load_design_local(design_id)
            self.upload_design(design_id,design,overwrite=overwrite,upload_index=False,verbose=verbose)
        self.sync_sessions_to_db(overwrite=overwrite,verbose=verbose)
        self.update_index()
        print('Local designs synced to remote DB')
    
    def download_design(self,design_id,overwrite=False): # download a single design from db
        design_dir=U.pjoin(self.db_dir,'designs',design_id)
        if not U.pexists(design_dir):
            U.mkdir(design_dir)
        index_term=self.index[design_id]

        # check metadata
        Doc=None
        if 'metadata' in index_term:
            metadata_path=U.pjoin(design_dir,'metadata.json')
            if not U.pexists(metadata_path) or overwrite:
                print(f'Downloading metadata for design {design_id}')
                Doc=self.collection.document(design_id).get().to_dict()
                metadata=Doc['metadata']
                U.save_json(metadata,metadata_path)
                print(f'Downloaded metadata for design {design_id}')
        
        # check proposal
        if 'proposal' in index_term:    
            proposal_path=U.pjoin(design_dir,'proposal.json')
            if not U.pexists(proposal_path) or overwrite:
                if Doc is None:
                    Doc=self.collection.document(design_id).get().to_dict()
                proposal=Doc['proposal']
                U.save_json(proposal,proposal_path)
                print(f'Downloaded proposal for design {design_id}')

        # check proposal traces     
        if 'proposal_traces' in index_term:
            for trace_id in index_term['proposal_traces']:
                trace_path=U.pjoin(design_dir,'proposal_traces',trace_id+'.json')
                if not U.pexists(trace_path) or overwrite:
                    U.mkdir(U.pjoin(design_dir,'proposal_traces'))
                    trace=self.collection.document(design_id).collection('proposal_traces').document(trace_id).get().to_dict()[trace_id]
                    U.save_json(trace,trace_path)
                    print(f'Downloaded proposal trace {trace_id} for design {design_id}')

        # check implementation
        if 'implementation' in index_term:
            implementation_path=U.pjoin(design_dir,'implementation.json')
            if not U.pexists(implementation_path) or overwrite:
                if Doc is None:
                    Doc=self.collection.document(design_id).get().to_dict()
                implementation=Doc['implementation']
                implementation['history']=[]
                for idx in index_term['implementation_history']:
                    step_ref = self.collection.document(design_id).collection('implementation_history').document(str(idx))
                    if step_ref.get().exists:
                        step = step_ref.get().to_dict()[str(idx)]
                        implementation['history'].append(step)
                U.save_json(implementation,implementation_path)
                print(f'Downloaded implementation for design {design_id}')
            
            elif U.pexists(implementation_path):
                _implementation=U.load_json(implementation_path)
                index_ih = [i for i in index_term['implementation_history'] if 'rounds' not in i]
                if len(_implementation['history'])<len(index_ih):
                    if Doc is None:
                        Doc=self.collection.document(design_id).get().to_dict()
                    implementation=Doc['implementation']
                    implementation['history']=_implementation['history']
                    for idx in range(len(_implementation['history']),len(index_ih)):
                        step=self.collection.document(design_id).collection('implementation_history').document(str(idx)).get().to_dict()[str(idx)]
                        if 'rounds' not in step:
                            step['rounds']=[]
                        index_step_rounds = index_term['implementation_history'][f'{idx}_rounds']
                        for round_idx in range(len(step['rounds']),len(index_step_rounds)):
                            rounds_data = self.collection.document(design_id).collection('implementation_history').document(str(idx)).collection('rounds').document(str(round_idx)).get().to_dict()
                            step['rounds'].append(rounds_data)
                        implementation['history'].append(step)
                    U.save_json(implementation,implementation_path)
                    print(f'Downloaded implementation for design {design_id}')

        # check verifications
        if 'verifications' in index_term:
            for scale in index_term['verifications']:
                verification_path=U.pjoin(design_dir,'verifications',scale+'.json')
                if not U.pexists(verification_path) or overwrite:
                    U.mkdir(U.pjoin(design_dir,'verifications'))
                    verification=self.collection.document(design_id).collection('verifications').document(scale).get().to_dict()
                    if not verification:
                        continue
                    verification['verification_report']={}
                    for key in index_term['verifications'][scale]:
                        report=self.collection.document(design_id).collection('verifications').document(scale).collection('verification_report').document(key).get().to_dict()
                        verification['verification_report'][key]=report
                    U.save_json(verification,verification_path)
                    print(f'Downloaded verification for scale {scale} in design {design_id}')
        
        # check codes
        if 'codes' in index_term:
            codes_path=U.pjoin(design_dir,'codes.json')
            if not U.pexists(codes_path) or overwrite:
                codes={}
                for scale in index_term['codes']:
                    code=self.collection.document(design_id).collection('codes').document(scale).get().to_dict()
                    codes[scale]=code
                U.save_json(codes,codes_path)
                print(f'Downloaded codes for design {design_id}')

    def sync_from_db(self,overwrite=False): # download all designs from db if out of date
        self.get_index()
        self.get_baseline_index()
        for design_id in self.index:
            self.download_design(design_id,overwrite=overwrite)
        self.FM.download_baselines()
        print('Local designs synced from remote DB')

    def delete_design(self,design_id):
        # should be used by human admin to fix error, not the agent
        # 1. delete the design 
        # 2. delete the index
        # 3. delete from GAUDict
        raise NotImplementedError("Not implemented yet")




############################################

LIBRARY_DIR = U.pjoin(os.path.dirname(__file__),'model','library')


FAILED_COLOR='#856d72'
DESIGN_COLOR='#5698c3'
DESIGN_IMPLEMENTED_COLOR='#1177b0'

NODE_COLOR_MAP={
    '14M':'#8B81C3',
    '31M':'#70649A',
    '70M':'#9B90C2',
    '125M':'#8A6BBE',
    '350M':'#6A4C9C',
    '760M':'#77428D',
    '1300M':'#4A225D',
}

ROOT_COLOR='#9eccab'

FAILED_SIZE=12
DESIGN_SIZE=15
DESIGN_IMPLEMENTED_SIZE=20

NODE_SIZE_MAP={
    '14M':22,
    '31M':24,
    '70M':26,
    '125M':28,
    '350M':30,
    '760M':32,
    '1300M':34,
}

CORE_COLOR = '#f0a1a8' # core reference
REFERENCE_COLOR = '#AF47D2'
RWC_COLOR = '#FB773C' # reference with code
EXT_COLOR_1HOC = '#ed556a' # extended 1-hop reference

# # from low to high
# TARGET_SCALES = ['14M','31M','70M','125M','350M']#,'760M','1300M']


@dataclass
class NodeObject:
    acronym: str # acronym is the unique identifier for the node
    title: str
    seed_ids: List[str]

    @property
    def type(self) -> str:
        return self.__class__.__name__
    
    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, dict: Dict):
        return cls(**dict)
    
    @classmethod
    def load(cls, save_dir: str, acronym:str):
        with open(U.pjoin(save_dir,acronym+'.json'),'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def save(self,save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(U.pjoin(save_dir,self.acronym+'.json'),'w', encoding='utf-8') as f:
            json.dump(self.to_dict(),f,indent=4)

    def to_desc(self) -> str:
        raise NotImplementedError
    
    def to_prompt(self) -> str:
        raise NotImplementedError


######### Library Reference

@dataclass
class LibraryReference(NodeObject):
    s2id: str = None
    abstract: str = None
    authors: List[str] = None
    venue: str = None
    year: int = None
    tldr: str = None
    # embedding: list
    citationCount: int = None
    influentialCitationCount: int = None
    code: str = None
    description: str = None
    url: str = None
    tree: GAUTree = None
    verifications: Dict[str, Verification] = field(default_factory=dict)

    def load_verifications(self,verification_dir,mult=20):
        if U.pexists(verification_dir):
            for scale in os.listdir(verification_dir):
                report_dir=U.pjoin(verification_dir,scale)
                report=U.load_json(report_dir)
                scale=scale.split('.')[0]
                if 'verification_report' in report:
                    _verification=Verification.from_dict(report['verification_report'])
                else:
                    _verification=Verification(scale=scale,verification_report=report)
                reports=_verification.verification_report
                error=False
                # if 'wandb_ids.json' not in reports:
                #     error=True
                if 'eval_results.json' not in reports:
                    error=True
                if 'trainer_state.json' not in reports:
                    error=True
                if not error:
                    if scale not in self.verifications:
                        self.verifications[scale]={}
                    self.verifications[scale][f'{mult}']=_verification

    def reload_verifications(self):
        core_dir=U.pjoin(LIBRARY_DIR,'core',self.acronym)
        DATA_DIR=os.environ.get('DATA_DIR')
        if U.pexists(core_dir):
            # verification_dir=U.pjoin(core_dir,'verifications')
            # U.mkdir(verification_dir)
            # self.load_verifications(verification_dir,mult=20)
            token_mults_dir=U.pjoin(DATA_DIR,'corerefs',self.acronym,'token_mults')
            U.mkdir(token_mults_dir)
            for mult in os.listdir(token_mults_dir):
                self.load_verifications(U.pjoin(token_mults_dir,mult,'verifications'),mult=int(mult))

    def __post_init__(self):
        py_dir=U.pjoin(LIBRARY_DIR,'base',self.acronym,self.acronym+'_edu.py')
        go_dir=U.pjoin(LIBRARY_DIR,'base',self.acronym,self.acronym+'_edu.go')
        core_dir=U.pjoin(LIBRARY_DIR,'core',self.acronym)
        if U.pexists(py_dir):
            self.code=f'# {self.acronym}_edu.py\n\n'+open(py_dir,'r', encoding='utf-8').read()
        elif U.pexists(go_dir):
            self.code=f'// {self.acronym}_edu.go\n\n'+open(go_dir,'r', encoding='utf-8').read()
        elif U.pexists(core_dir):
            code_dir=U.pjoin(core_dir,self.acronym+'_edu.py')
            tree_dir=U.pjoin(core_dir,'units')
            if U.pexists(code_dir):    
                self.code=f'# {self.acronym}_edu.py\n\n'+open(code_dir,'r', encoding='utf-8').read()
            if U.pexists(tree_dir):
                self.tree=GAUTree.load_from_base(tree_dir)
            if self.tree is not None:
                print(f'{self.acronym} tree loaded')
            self.reload_verifications()
                    
        else:
            self.code=None

        

    @property
    def type(self) -> str:
        core_dir=U.pjoin(LIBRARY_DIR,'core',self.acronym)
        if self.tree is not None: # as long as there is a tree, it is a core reference
            return 'ReferenceCoreWithTree'
        elif self.code is not None: # if there is code, it is a reference with code
            if U.pexists(core_dir): # if there is a core dir, it is a core reference
                return 'ReferenceCore'
            else: # otherwise, it is a reference with code
                return 'ReferenceWithCode'
        else: # if there is no code, it is a reference
            return 'Reference'

    def to_desc(self, reformat=True) -> str:
        mdtext = f'## {self.title}\n'
        
        if self.s2id:
            mdtext += f'\n**S2 ID:** {self.s2id}'
        
        if self.authors:
            authors = ', '.join(self.authors)
            mdtext += f'\n**Authors:** {U.break_sentence(authors, 100)}'
        
        if self.tldr:
            mdtext += f'\n\n**TL;DR:** \n{U.break_sentence(self.tldr, 100)}'
        
        if self.abstract:
            abstract = U.break_sentence(self.abstract, 100) if reformat else self.abstract
            mdtext += f'\n\n**Abstract:**\n{abstract}'
        
        if self.venue:
            mdtext += f'\n\n**Published at:** *{self.venue}* in {self.year}'
        
        if self.citationCount:
            mdtext += f'\n\n**Cited:** {self.citationCount} times'
        
        if self.influentialCitationCount:
            mdtext += f'\n\n**Impactful Citations:** {self.influentialCitationCount}'
        
        if self.description:
            description = U.break_sentence(self.description, 100) if reformat else self.description
            mdtext += f'\n\n### Description:\n{description}'
        
        if self.url:
            mdtext += f'\n\n**[Link to Paper]({self.url})**'

        if reformat:
            return mdtext.replace(':', ' ').replace('e.\ng.\n', 'e.g.').replace('i.\ne.\n', 'i.e.')
        
        return mdtext

    def to_prompt(self, full_tree=True) -> str:
        prompt = self.to_desc(reformat=False)
        
        if self.tree:
            if full_tree:
                prompt += f'\n\n{self.tree.to_prompt()}\n\n'
            else:
                prompt += f'\n\n## Reference Document\n\n{self.tree.root_node.spec.document}\n\n'
        elif self.code:
            if self.type == 'ReferenceCore':
                prompt += (
                    f'\n\n## GAB Implementation\n'
                    '<details><summary>Click to expand</summary>\n\n'
                    '```python\n'
                    f'{self.code}\n'
                    '```\n'
                    '</details>\n\n'
                )
            else:
                prompt += (
                    f'\n\n## Reference Code\n'
                    '<details><summary>Click to expand</summary>\n\n'
                    '```python\n'
                    f'{self.code}\n'
                    '```\n'
                    '</details>\n\n'
                )

        return prompt


##### 1-hop reference, low SNR, not used now, most are just about how to use those models to do applications
# @dataclass
# class LibraryReference1hop(LibraryReference):

#     @property
#     def type(self) -> str:
#         # if self.code is not None:
#         #     return 'Reference1hopWithCode'
#         # else:
#         return 'Reference1hop'



######## Design Artifacts


@dataclass
class Proposal:
    selection:str
    modelname:str
    variantname:str
    proposal:str
    review:str
    rating:int
    passed:bool
    suggestions:str
    reflection:str
    changes:str
    ideation:str
    instructions:str
    search_report:str
    search_references:str
    # traces: List[Dict]
    costs: Dict[str, float]
    design_cfg: Dict[str, Any]
    user_input: str
    abstract:str = None
    search_stack: List[str] = None
    review_search_stack: List[str] = None

    def save(self, design_dir: str, name='proposal.json'):
        U.save_json(self.to_dict(), U.pjoin(design_dir, name))
    
    def to_dict(self):
        dict=asdict(self)
        dict['design_cfg']['running_mode']=self.design_cfg['running_mode'].value
        return dict

    @classmethod
    def from_dict(cls, dict: Dict):
        dict['design_cfg']['running_mode']=RunningModes(dict['design_cfg']['running_mode'])
        return cls(**dict)

    @classmethod
    def load(cls, design_dir: str, name='proposal.json'):
        if not U.pexists(U.pjoin(design_dir, name)):
            return None
        dict=U.load_json(U.pjoin(design_dir, name))
        dict['design_cfg']['running_mode']=RunningModes(dict['design_cfg']['running_mode'])
        return cls.from_dict(dict)


@dataclass
class ImplementationAttempt:
    status: str
    rounds: int
    costs: Dict[str, float]
    tree: GAUTree
    design_cfg: Dict[str, Any]
    user_input: str

    def to_dict(self):
        dict=asdict(self)
        dict['design_cfg']['running_mode']=self.design_cfg['running_mode'].value
        dict['tree']=self.tree.to_dict()
        return dict

    @classmethod
    def from_dict(cls, dict: Dict):
        dict['design_cfg']['running_mode']=RunningModes(dict['design_cfg']['running_mode'])
        dict['tree']=GAUTree.from_dict(dict['tree'])
        return cls(**dict)


# def _patch__try_fix_history(history): # FIXME: figure out why it happens
#     if isinstance(history,dict):
#         if '0' in history:
#             return [history[k] for k in history]
#         else:
#             return [history]
#     new_history=[]
#     if len(history)>0:
#         if '0' in history[0]: # not sure why like that, but do it for now
#             for _history in history:
#                 for _,v in _history.items():
#                     new_history.append(v)
#         else:
#             new_history=history
#     return new_history


@dataclass
class Implementation:
    status: str # implemented, failed, or unfinished
    implementation: GAUTree
    history: List[ImplementationAttempt]
    # TODO:consider gaudict management, but how to represent a unit snippet?

    def save(self, design_dir: str):
        U.save_json(self.to_dict(), U.pjoin(design_dir, f'implementation.json'))

    def to_dict(self):
        dict=asdict(self)
        dict['implementation']=self.implementation.to_dict()
        dict['history']=[attempt.to_dict() for attempt in self.history]
        return dict

    @classmethod
    def from_dict(cls, _dict: Dict):
        # _dict['history']=_patch__try_fix_history(_dict['history'])
        for i in range(len(_dict['history'])):
            if 'design_cfg' in _dict['history'][i]:
                _dict['history'][i]['design_cfg']['running_mode']=RunningModes(_dict['history'][i]['design_cfg']['running_mode'])
        _dict['history'] = []
        for attempt in _dict['history']:
            if 'rounds' not in attempt:
                attempt['rounds']=[]
            _dict['history'].append(ImplementationAttempt.from_dict(attempt))
        _dict['implementation']=GAUTree.from_dict(_dict['implementation'])
        return cls(**_dict)

    @classmethod
    def load(cls, design_dir: str):
        if not U.pexists(U.pjoin(design_dir, 'implementation.json')):
            return None
        dict=U.load_json(U.pjoin(design_dir, 'implementation.json'))
        return cls.from_dict(dict)
    
    @property
    def cost(self):
        return self.get_cost()
    
    def get_cost(self):
        costs={}
        for attempt in self.history:
            for k,v in attempt.costs.items():
                if k not in costs:
                    costs[k]=0
                costs[k]+=v
        return costs

@dataclass
class Verification:
    scale: str
    verification_report: Dict

    def save(self, design_dir: str):
        U.mkdir(U.pjoin(design_dir, 'verifications'))
        U.save_json(self.to_dict(), U.pjoin(design_dir, 'verifications',f'{self.scale}.json'))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict: Dict):
        return cls(**dict)
    
    @classmethod
    def load(cls, dir: str):
        return cls.from_dict(U.load_json(dir))

@dataclass
class DesignArtifact(NodeObject):
    sess_id: str # design session id, time info can be parsed from sess_id
    proposal: Proposal
    implementation: Implementation = None # find by modelname/id
    verifications: Dict[str, Verification] = field(default_factory=dict) # find by modelname/id
    codes: Dict[str, str] = field(default_factory=dict) # find by modelname/id
    sess_snapshot: Dict[str, str] = field(default_factory=dict) 

    @property
    def type(self) -> str:
        if self.is_implemented():
            return 'DesignArtifactImplemented'
        else:
            return 'DesignArtifact'
    
    @property
    def timestamp(self) -> datetime:
        tail = self.sess_id.split('-')[-1]
        timestr=self.sess_id[:-len(tail)-1]
        timeformat='%Y-%m-%d-%H-%M-%S'
        return datetime.strptime(timestr, timeformat)
    
    def is_finished(self,challenging_threshold):
        if self.implementation and self.implementation.status=='implemented':
            return True
        if not self.proposal.passed:
            return True
        if self.implementation and len(self.implementation.history)>=challenging_threshold:
            return True
        return False
        
        
    @classmethod
    def load(cls, design_dir: str):
        metadata = U.load_json(U.pjoin(design_dir, 'metadata.json'))
        proposal = Proposal.load(design_dir)
        if proposal is None:
            return None
        implementation = Implementation.load(design_dir)
        verifications = {}
        ver_dir=U.pjoin(design_dir,'verifications')
        if U.pexists(ver_dir):
            for scale in os.listdir(ver_dir):
                scale=scale.split('.')[0]
                dir = U.pjoin(ver_dir,f'{scale}.json')
                if U.pexists(dir):
                    _verification = Verification.load(dir)
                    error=False
                    if 'wandb_ids.json' not in _verification.verification_report:
                        error=True
                    if 'eval_results.json' not in _verification.verification_report:
                        error=True
                    if 'trainer_state.json' not in _verification.verification_report:
                        error=True
                    if not error:
                        verifications[scale] = _verification
        codes = U.load_json(U.pjoin(design_dir, 'codes.json'))
        return cls(proposal=proposal, implementation=implementation, verifications=verifications, codes=codes, **metadata)

    def to_prompt(self, full_tree=True):
        prompt=f"""
# Proposal: {self.proposal.modelname}

{self.proposal.proposal}

## Review

{self.proposal.review}

### Rating: {self.proposal.rating} out of 5

### Reviewer Suggestions

{self.proposal.suggestions}
"""
        if self.is_implemented():
            if full_tree:
                prompt+=f"""
# Implementation

{self.implementation.implementation.to_prompt()}
            """
            else:
                prompt+=f"""
# Reference Document

{self.implementation.implementation.root_node.spec.document}
            """
        for scale in self.verifications:
            pass # TODO

        return prompt


    def to_desc(self):
        mdtext = f'## {self.proposal.modelname}'
        mdtext += f'\n**Model:** {self.proposal.modelname}'
        if self.proposal.selection:
            mdtext += f'\n**Selection:** {self.proposal.selection} ({self.proposal.variantname})'
        if self.proposal.abstract:
            mdtext += f'\n**Abstract:**\n{U.break_sentence(self.proposal.abstract, 100)}'
        mdtext += f'\n**Rating:** {self.proposal.rating}/5'
        mdtext += f'\n**Passed:** {self.proposal.passed}'
        if self.sess_snapshot:
            mdtext += f'\n**Mode:** {self.sess_snapshot["mode"]}'
            mdtext += f'\n**Seed IDs:** {self.sess_snapshot["seed_ids"]}'
            mdtext += f'\n**Ref IDs:** {self.sess_snapshot["ref_ids"]}'
            if self.sess_snapshot['instruct']:
                mdtext += f'\n**Instruct:** {self.sess_snapshot["instruct"]}'
        else:
            mdtext += f'\n**Seed IDs:** {self.seed_ids}'
        if self.implementation:
            mdtext += f'\n**Implementated:** {self.implementation.status}'
            mdtext += f'\n**Units:** {list(self.implementation.implementation.units.keys())}'
        else:
            mdtext += f'\n**Implementated:** False'
        mdtext += f'\n\n**Verification:**\n'
        if len(self.verifications)>0:
            for scale in self.verifications:
                eval_results = self.verifications[scale].verification_report['eval_results.json']['results']
                avg_acc = 0
                cnt = 0
                for k,v in eval_results.items():
                    if 'acc,none' in v:
                        avg_acc += v['acc,none']
                        cnt += 1
                avg_acc /= cnt
                wandb_ids = self.verifications[scale].verification_report['wandb_ids.json']
                wandb_url = f"https://wandb.ai/{wandb_ids['entity']}/{wandb_ids['project']}/runs/{wandb_ids['pretrain']['id']}"
                mdtext += f'{scale} avg. acc: {avg_acc:.4f} [wandb]({wandb_url})\n'
        else:
            mdtext += 'No verification results\n'
        return mdtext.replace(':', ' ').replace('e.\ng.\n', 'e.g.').replace('i.\ne.\n', 'i.e.')

    def is_implemented(self):
        return self.implementation is not None and self.implementation.status=='implemented'

    @property
    def costs(self):
        return self.get_cost()
    
    def get_cost(self):
        costs=self.proposal.costs
        if self.implementation:
            icosts=self.implementation.get_cost()
            costs={k:v+icosts[k] for k,v in costs.items() if k in icosts}
        # TODO: maybe cost of rerank, selection, etc. also considered, now only agents 
        return costs


# def write_dot(G, path):
#     """Write NetworkX graph G to Graphviz dot format on path.

#     Path can be a string or a file handle.
#     """
#     msg = (
#         "nx.nx_pydot.write_dot depends on the pydot package, which has"
#         "known issues and is not actively maintained. Consider using"
#         "nx.nx_agraph.write_dot instead.\n\n"
#         "See https://github.com/networkx/networkx/issues/5723"
#     )
#     warnings.warn(msg, PendingDeprecationWarning, stacklevel=2)
#     P = to_pydot(G)
#     with open(path, "w", encoding='utf-8') as f:
#         f.write(P.to_string())
#     return



class PhylogeneticTree:
    # Read from a design base and construct a phylogenetic tree
    """
    Physical structure:
    evoname
    ├── designs
    │   ├── acronym
    |   |   ├── metadata.json
    │   │   ├── proposal.json
    │   │   ├── proposal_traces
    │   │   │   ├── trace1.json
    │   │   │   └── ...
    │   │   ├── implementation.json
    │   │   ├── verifications
    │   │   │   ├── scale1.json
    │   │   │   └── ...
    │   └── ...
    ├── sessions
    │   ├── sess_id
    │   │   ├── metadata.json
    │   │   ├── log
    │   │   │   ├── log1.log
    │   │   │   └── ...
    │   └── ...
    | ... # units, etc.
    """
    def __init__(self, evoname, target_scales, db_dir: str, db_only=False, remote_db=None, use_remote_db=True,
                 challenging_threshold=3,CM=None,token_mults=None): 
        self.evoname = evoname
        self.target_scales = target_scales
        self.db_dir = db_dir
        self.lib_dir = U.pjoin(LIBRARY_DIR,'tree')
        self.lib_ext_dir = U.pjoin(LIBRARY_DIR,'tree_ext')
        self.design_sessions = {}
        U.mkdir(db_dir)
        U.mkdir(U.pjoin(db_dir,'designs'))
        U.mkdir(U.pjoin(db_dir,'sessions'))
        self.challenging_threshold=challenging_threshold
        self.db_only=db_only
        self.FM = None
        self.GD = None
        self.CM = CM
        self.use_remote_db=use_remote_db
        self.remote_db = remote_db
        self.token_mults = token_mults
        if use_remote_db and self.remote_db is not None:
            self.FM = FirestoreManager(self,evoname,db_dir,self.remote_db)
            self.FM.sync_from_db()
        self.load()

        random_baseline = self.remote_db.collection('random_baseline').document('eval_results.json').get()
        self.random_baseline = random_baseline.to_dict() if random_baseline.exists else {}
        assert self.random_baseline, 'No random baseline eval results found, please run `bash scripts/run_verify.sh --RANDOM_TESTING` first'
        if self.FM:
            self.FM.upload_baselines()
        
    # new design: proposal -> implement -> verify

    # def reload(self): # why do we need this at all??
    #     self.load()

    @property
    def n_verified(self):
        n=0
        for acronym in self.filter_by_type('DesignArtifactImplemented'):
            design=self.get_node(acronym)
            if design.verifications:
                n+=1
        return n

    def get_nodes(self,acronyms):
        if isinstance(acronyms,str):
            acronyms=[acronyms]
        return [self.get_node(acronym) for acronym in acronyms]
    
    def get_abstracts(self,acronyms):
        abstracts=[]
        reviews=[]
        ratings=[]
        for acronym in acronyms:
            proposal=self.get_node(acronym).proposal
            if proposal.abstract:
                abstracts.append(proposal.abstract)
            else:
                abstracts.append(proposal.proposal[:1600]+'...') # ~400 tokens
            reviews.append(proposal.review)
            ratings.append(proposal.rating)
        return abstracts,reviews,ratings
    
    def find_sibling_designs(self,parents):
        if isinstance(parents,str):
            parents=[parents]
        siblings=[]
        for acronym in self.filter_by_type(['DesignArtifact','DesignArtifactImplemented']):
            design=self.get_node(acronym)
            if design.seed_ids == parents:
                siblings.append(acronym)
        return siblings
    
    def get_design_session(self,sess_id:str):
        if sess_id not in self.design_sessions and self.FM:
            sessdata = self.FM.get_design_session(sess_id)
            if sessdata:
                sessdata['mode']=DesignModes(sessdata['mode'])
                self.design_sessions[sess_id] = sessdata
                self.save_session(sess_id)
        if sess_id not in self.design_sessions:
            return None
        return self.design_sessions[sess_id]

    def load_design_sessions(self):
        self.design_sessions={}
        to_delete=[]
        for sess_id in os.listdir(U.pjoin(self.db_dir,'sessions')):
            if self.FM:
                metadata = self.FM.get_design_session(sess_id)
                if not metadata:
                    metadata = U.load_json(U.pjoin(self.session_dir(sess_id), 'metadata.json'))
            else:
                metadata = U.load_json(U.pjoin(self.session_dir(sess_id), 'metadata.json'))
            if 'mode' in metadata:
                metadata['mode']=DesignModes(metadata['mode'])
            if not metadata:
                to_delete.append(sess_id) # delete empty sessions
                continue
            self.design_sessions[sess_id] = metadata
        for sess_id in to_delete:
            shutil.rmtree(self.session_dir(sess_id))

    @property
    def design_cost(self):
        costs=0
        designs=self.filter_by_type(['DesignArtifact','DesignArtifactImplemented'])
        for design in designs:
            costs+=sum(self.get_node(design).costs.values())
        return costs

    def budget_status(self,budgets,ret_verified=False):
        budgets=copy.deepcopy(budgets)
        verified={}
        designs=self.filter_by_type(['DesignArtifactImplemented'])
        for design in designs:
            for scale in self.get_node(design).verifications:
                if scale in budgets:
                    budgets[scale]-=1
                if scale not in verified:
                    verified[scale]=0
                verified[scale]+=1
        if ret_verified:
            return budgets,verified
        else:
            return budgets
        
    def update_baselines(self):
        self.FM.download_baselines()
        designs=self.filter_by_type(['ReferenceCore','ReferenceCoreWithTree'])
        for design in designs:
            self.G.nodes[design]['data'].reload_verifications()

    def get_design_vectors(self,is_baseline=False): # a more numerical representation of a design, selector to use
        if is_baseline:
            self.FM.download_baselines()
            designs=self.filter_by_type(['ReferenceCore','ReferenceCoreWithTree'])
        else:
            self.update_design_tree()
            designs=self.filter_by_type('DesignArtifactImplemented')
        design_vectors = {}
        for design in designs:
            design_vectors[design] = self.get_design_vector(design,is_baseline)
        return design_vectors

    def get_design_vector(self,acronym,is_baseline=False):
        vector = {}
        node = self.get_node(acronym)
        if not is_baseline:
            vector['proposal_rating'] = node.proposal.rating
            vector['units'] = {}
            for unit_name, unit in node.implementation.implementation.units.items():
                vector['units'][unit_name] = unit.rating
        vector['verifications'] = {}
        for scale in node.verifications:
            vs = node.verifications[scale]
            if is_baseline:
                token_mult = 20 if self.token_mults is None else self.token_mults[scale]
                if token_mult not in vs:
                    continue
                verification_report = vs[token_mult].verification_report
            else:
                verification_report = vs.verification_report
            if 'training_record.csv' not in verification_report or 'system_metrics.csv' not in verification_report:
                if 'wandb_ids.json' in verification_report:
                    verification_report.update(get_history_report(verification_report['wandb_ids.json']))
            vector['verifications'][scale] = verification_report
        return vector

    def get_baseline_vectors(self):
        return self.get_design_vectors(is_baseline=True)

    def get_design_artifacts(self):
        self.update_design_tree()
        acronyms= self.filter_by_type(['DesignArtifact','DesignArtifactImplemented'])
        return {acronym:self.get_node(acronym) for acronym in acronyms}

    # How to handle variants? i.e., in GPT, there are optional pre-conv and post-conv, maybe just all of them to the tree, let selector to choose

    def new_design(self, seed_ids, ref_ids, instruct, num_samples, sess_id=None): # new design session, a session explore the steps from a selected node
        if len(seed_ids)==0:
            mode=DesignModes.SCRATCH
        elif len(seed_ids)==1:
            mode=DesignModes.MUTATION
        else:
            mode=DesignModes.CROSSOVER
            
        hash_tail=hashlib.sha256(f"{sorted(ref_ids)}{sorted(seed_ids)}{instruct}{mode.value}".encode()).hexdigest()
        if sess_id is None:
            sess_id = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{hash_tail[-6:]}"
        sessdata = {
            'seed_ids': seed_ids,
            'ref_ids': ref_ids,
            'instruct': instruct,
            'mode': mode,
            'proposed': [],
            'reranked': {},
            'num_samples': num_samples,
        }
        self.design_sessions[sess_id] = sessdata
        sess_dir=self.session_dir(sess_id)
        U.mkdir(sess_dir)
        self.save_session(sess_id)
        U.mkdir(U.pjoin(sess_dir, 'log'))
        return sess_id
    
    def get_unverified_designs(self,scale=None,exclude={}): # exclude is a dict: {scale: [design_id, ...], ...}
        unverified=[] if scale else {s:[] for s in self.target_scales}
        for acronym in self.filter_by_type('DesignArtifactImplemented'):
            if acronym in exclude.get(scale,[]):
                continue
            design=self.get_node(acronym)
            if scale:
                if scale not in design.verifications:
                    unverified.append(acronym)
            else:
                for _scale in self.target_scales:
                    if _scale not in design.verifications:
                        unverified[_scale].append(acronym)
        return unverified

    def get_unverified_scales(self,acronym=None,exclude_inv={}): # exclude_inv is a dict: {design_id: [scale, ...], ...}
        if acronym:
            return self._get_unverified_scales(acronym,exclude_inv.get(acronym,[]))
        else:
            unverified={}
            for acronym in self.filter_by_type('DesignArtifactImplemented'):
                unverified[acronym]=self._get_unverified_scales(acronym,exclude_inv.get(acronym,[]))
            return unverified

    def _get_unverified_scales(self,acronym,exclude_scales=[]): # from low to high
        unverified=[]
        design=self.get_node(acronym)
        for scale in self.target_scales:
            if scale not in design.verifications and scale not in exclude_scales:
                unverified.append(scale)
        unverified.sort(key=lambda x: int(x.replace('M','')))
        return unverified

    def get_gau_tree(self,acronym:str):
        node=self.get_node(acronym)
        if node.type=='ReferenceCoreWithTree':
            tree=node.tree
            return tree
        elif node.type=='DesignArtifactImplemented':
            tree=node.implementation.implementation
            return tree
        else:
            return None 
        
    def new_gau_tree(self): # get an empty unimplemented tree
        new_tree = GAUTree('NEW_TREE',proposal='',review='',rating=0,suggestions='')
        root_spec = UnitSpec(unitname='root',document='',inputs=['X'],outputs=['Y'])
        root_decl = UnitDecl(unitname='root',requirements='',inputs=['X'],outputs=['Y'])
        new_tree.add_unit(spec=root_spec,code=GAU_TEMPLATE,args='',desc='',review='',rating=None,children=[],gautests={},suggestions='')
        new_tree.declares['root']=root_decl
        return new_tree
    
    def get_session_input(self,sess_id:str):
        sessdata=self.design_sessions[sess_id]
        seeds=[self.get_node(seed_id) for seed_id in sessdata['seed_ids']]
        refs=[self.get_node(ref_id) for ref_id in sessdata['ref_ids']]
        mode=sessdata['mode']
        if isinstance(mode,str): # just in case
            mode=DesignModes(mode)
        return seeds,refs,sessdata['instruct'],mode
    
    def session_dir(self, sess_id: str):
        sess_dir=U.pjoin(self.db_dir, 'sessions', sess_id)
        U.mkdir(sess_dir)
        return sess_dir
    
    def session_get(self,sess_id:str,key:str):
        return self.design_sessions[sess_id].get(key)

    def session_set(self,sess_id:str,key:str,value):
        self.design_sessions[sess_id][key]=value
        self.save_session(sess_id)
    
    def session_append(self,sess_id:str,key:str,value):
        if key not in self.design_sessions[sess_id]:
            self.design_sessions[sess_id][key]=[]
        self.design_sessions[sess_id][key].append(value)
        self.save_session(sess_id)
    
    def design_dir(self, acronym: str):
        design_dir=U.pjoin(self.db_dir, 'designs', acronym)
        U.mkdir(design_dir)
        return design_dir
    
    def coreref_dir(self, acronym: str, token_mult=20):
        # token_mult = int(token_mult)
        # if token_mult==20:
        #     coreref_dir=U.pjoin(LIBRARY_DIR,'core',acronym)
        # else:
        DATA_DIR=os.environ.get("DATA_DIR")
        coreref_dir=U.pjoin(DATA_DIR,'corerefs',acronym,'token_mults',f'{token_mult}')
        U.mkdir(coreref_dir)
        return coreref_dir
    
    def _get_node(self, acronym: str):
        if acronym not in self.G.nodes:
            if self.FM:
                self.FM.get_index()
                if acronym not in self.FM.index:
                    return None
                self.FM.download_design(acronym)
                artifact=DesignArtifact.load(self.design_dir(acronym))
                self.G.add_node(acronym, data=artifact)
        if acronym in self.G.nodes:
            return self.G.nodes[acronym]['data']
        else:
            return None

    def get_node(self, acronym: str): # sometimes the agent output with "" or ''
        node = self._get_node(acronym)
        if node is None:
            node = self._get_node(f'"{acronym}"')
        if node is None:
            node = self._get_node(f"'{acronym}'")
        return node
    
    def get_finished_designs(self):
        self.update_design_tree()
        designs = self.filter_by_type(['DesignArtifactImplemented','DesignArtifact'])
        finished = [design for design in designs if self.get_node(design).is_finished(self.challenging_threshold)]
        return finished
    
    def acquire_design_lock(self,sess_id=None): # no need to really lock, as CC is still sequential
        if not self.CM.benchmark_mode:
            return True
        active_sessions = self.CM.get_active_design_sessions()
        active_sessions=list(active_sessions.keys())
        finished_designs = self.get_finished_designs()
        if sess_id and sess_id not in active_sessions:
            active_sessions.append(sess_id)
        if len(active_sessions)+len(finished_designs)<=self.CM.max_designs:
            return True
        else:
            return False
    
    def get_session_state(self,sess_id:str):
        passed,_ = self.session_proposals(sess_id,passed_only=True)
        implemented,unfinished = self.session_implementations(sess_id)
        unfinished_impls=self.get_implementations(unfinished)
        challenging=[]
        for acronym in unfinished_impls:
            impl=unfinished_impls[acronym]
            if len(impl.history)>self.challenging_threshold:
                challenging.append(acronym)
        return passed,implemented,challenging,unfinished

    def get_challenging_designs(self,sess_id:str):
        sessdata=self.design_sessions[sess_id]
        challenging={}
        for acronym in sessdata['proposed']:
            impl=self.get_node(acronym).implementation  
            if impl:
                if impl.status!='implemented':
                    if len(impl.history)>self.challenging_threshold:
                        challenging[acronym]=len(impl.history)
        return challenging

    def add_design_artifact(self,id,artifact):
        self.G.add_node(id, data=artifact)
        edges_to_add = []
        for seed_id in artifact.seed_ids:
            edges_to_add.append((seed_id, id))
        for seed_id, product_id in edges_to_add:
            if seed_id not in self.G.nodes or product_id not in self.G.nodes:
                continue
            if seed_id == product_id or nx.has_path(self.G, product_id, seed_id):
                continue
            self.G.add_edge(seed_id, product_id)

    def update_design_tree(self):
        if self.FM:
            self.FM.sync_from_db()
            updated_terms=self.FM.updated_terms
        edges_to_add = []
        for id in os.listdir(U.pjoin(self.db_dir,'designs')):
            if id not in self.G.nodes:
                artifact = DesignArtifact.load(self.design_dir(id))
                self.G.add_node(id, data=artifact)
                for seed_id in artifact.seed_ids:
                    edges_to_add.append((seed_id, id))
            elif id in updated_terms:
                self.G.nodes[id]['data']=DesignArtifact.load(self.design_dir(id))

        for seed_id, product_id in edges_to_add:
            if seed_id not in self.G.nodes or product_id not in self.G.nodes:
                continue
            if seed_id == product_id or nx.has_path(self.G, product_id, seed_id):
                continue
            self.G.add_edge(seed_id, product_id)
        
        if self.FM:
            self.FM.updated_terms=[]

    def get_unfinished_designs(self,return_finished=False):
        self.load_design_sessions()
        self.update_design_tree()
        unfinished_designs = []
        finished_designs = []
        for sess_id in self.design_sessions:
            sessdata=self.design_sessions[sess_id]
            num_samples=sessdata['num_samples']
            passed,implemented,challenging,_=self.get_session_state(sess_id)
            if len(passed)<num_samples['proposal']:
                unfinished_designs.append(sess_id)
            elif len(implemented)+len(challenging)<num_samples['implementation']:
                unfinished_designs.append(sess_id)
            else:
                finished_designs.append(sess_id)
        if return_finished:
            return unfinished_designs,finished_designs
        else:
            return unfinished_designs

    def is_challenging(self,acronym:str):
        design=self.get_node(acronym)
        if design.implementation:
            impl=design.implementation
            if len(impl.history)>self.challenging_threshold:
                return True
        return False

    def get_implementation_checkpoint(self,acronym:str):
        design=self.get_node(acronym)
        if design.implementation:
            impl=design.implementation
            return impl.implementation,impl.status
        else:
            return None,None
    
    def session_proposals(self,sess_id:str,passed_only=False):
        sessdata=self.design_sessions[sess_id]
        acronyms=[]
        proposals=[]
        for acronym in sessdata['proposed']:
            design=self.get_node(acronym)
            if passed_only and not design.proposal.passed:
                continue
            proposals.append(design.proposal)
            acronyms.append(acronym)
        return proposals,acronyms
    
    def session_implementations(self,sess_id:str):
        sessdata=self.design_sessions[sess_id]
        implemented=[]
        unfinished=[]
        for acronym in sessdata['proposed']:
            design=self.get_node(acronym)
            if design.implementation:
                if design.implementation.status=='implemented':
                    implemented.append(acronym) 
                else:
                    unfinished.append(acronym)
        return implemented,unfinished
    
    def get_implementations(self,acronyms:list):
        implementations={}
        for acronym in acronyms:
            design=self.get_node(acronym)
            implementations[acronym]=design.implementation
        return implementations

    def propose(self, sess_id: str, proposal,proposal_traces,costs,design_cfg,user_input): # create a new design artifact
        sessdata=copy.deepcopy(self.design_sessions[sess_id])
        seeds=sessdata['seed_ids']
        proposal['costs']=costs
        proposal['design_cfg']=design_cfg
        proposal['user_input']=user_input
        proposal = Proposal(**proposal)
        title = proposal.modelname
        for line in proposal.proposal.split("\n"):
            if line.startswith("# "):
                title = line[2:]
                break
        acronym = self.unique_acronym(proposal.modelname)
        proposal.modelname = acronym
        sessdata['mode']=sessdata['mode'].value
        metadata = {'sess_id': sess_id, 'acronym': acronym, 'seed_ids': seeds, 'title': title, 'sess_snapshot': sessdata} # sess_snapshot is the status of the session when this sampling happens
        U.save_json(metadata, U.pjoin(self.design_dir(acronym), 'metadata.json'))
        proposal.save(self.design_dir(acronym))
        traces_dir=U.pjoin(self.design_dir(acronym),'proposal_traces')
        _proposal_traces={}
        for idx,trace in enumerate(proposal_traces):
            U.mkdir(traces_dir)
            trace['costs']=costs # show the full cost directly
            trace['design_cfg']=design_cfg
            trace['user_input']=user_input
            proposal_trace=Proposal(**trace)
            proposal_trace.save(traces_dir,f'trace_{idx}.json')
            _proposal_traces[f'trace_{idx}']=proposal_trace.to_dict()
        design_artifact = DesignArtifact(sess_id=sess_id, acronym=acronym, seed_ids=seeds, title=title, proposal=proposal)
        self.G.add_node(acronym, data=design_artifact)
        self.session_append(sess_id,'proposed',acronym)
        self.FM.upload_metadata(acronym,metadata,overwrite=True)
        self.FM.upload_proposal(acronym,proposal.to_dict(),overwrite=True)
        self.FM.upload_proposal_traces(acronym,_proposal_traces,overwrite=True)
        self.FM.update_index()

    def save_session(self,sess_id: str):
        sessdata=self.design_sessions[sess_id]        
        try:
            sessdata['mode']=DesignModes(sessdata['mode'])
        except:
            pass
        sessdata['mode']=sessdata['mode'].value
        U.save_json(sessdata, U.pjoin(self.session_dir(sess_id), 'metadata.json'))
        if self.FM: # keep firestore always up to date
            self.FM.upload_design_session(sess_id,sessdata)
        sessdata['mode']=DesignModes(sessdata['mode'])

    def implement(self, acronym: str, tree,ROUNDS,status,costs,design_cfg,user_input): # update a proposal node with implementation
        from model_discovery.model.library.tester import check_tune
        
        design_artifact=self.get_node(acronym)
        acronym=design_artifact.acronym
        implementation=design_artifact.implementation
        attempt=ImplementationAttempt(status=status, rounds=ROUNDS, costs=costs, tree=tree, design_cfg=design_cfg, user_input=user_input)
        if implementation is None:
            implementation=Implementation(status=status, implementation=tree, history=[attempt])
        else:
            implementation.status=status
            implementation.implementation=tree
            implementation.history.append(attempt)
        implementation.save(self.design_dir(acronym))
        design_artifact.implementation=implementation
        self.G.nodes[acronym]['data']=design_artifact
        self.GD.new_term(acronym,tree)
        # Tune in all target scales
        if status=='implemented':
            codes = {}
            _code = tree.compose()
            for scale in self.target_scales:
                codes[scale] = check_tune(scale,acronym, code=_code,check_only=True,cpu_only=True,reformat_only=True)
            U.save_json(codes, U.pjoin(self.design_dir(acronym), 'codes.json'))
        self.FM.upload_implementation(acronym,implementation.to_dict(),overwrite=True)
        self.FM.update_index()

    def verify(self, acronym: str, scale: str, verification_report, RANDOM_TESTING=False, token_mult=20): # attach a verification report under a scale to an implemented node
        if 'eval_results.json' not in verification_report:
            return
        if RANDOM_TESTING:
            eval_results = verification_report['eval_results.json']
            self.remote_db.collection('random_baseline').document('eval_results.json').set(eval_results)
            return
        # if 'trainer_state.json' not in verification_report:
        #     return
        design_artifact=self.get_node(acronym)
        acronym=design_artifact.acronym
        verification=Verification(scale=scale, verification_report=verification_report)
        design_artifact.verifications[scale]=verification
        self.G.nodes[acronym]['data']=design_artifact
        if design_artifact.type=='DesignArtifactImplemented':
            verification.save(self.design_dir(acronym))
            self.FM.upload_verification(acronym,verification.to_dict(),scale,overwrite=True)
            self.FM.update_index()
        else:
            # for baselines, it should be saved in repo already, can be synced by github
            verification.save(self.coreref_dir(acronym,token_mult))
            self.FM.upload_baselines(overwrite=True,verbose=True)
            self.FM.update_index(is_baseline=True)

    def unique_acronym(self, acronym: str, max_length=32) -> str:
        acronym = acronym.lower()
        acronym = re.sub(r'[^a-z0-9_]', '_', acronym)
        acronym = re.sub(r'_+', '_', acronym)
        acronym = acronym.strip('_')
        acronym = acronym[:max_length]

        existing_acronyms = set(self.G.nodes)
        if self.FM:
            self.FM.get_index()
            existing_acronyms.update(self.FM.index.keys())
        if acronym not in existing_acronyms:
            return acronym
        i = 1
        while f"{acronym}_{i}" in existing_acronyms:
            i += 1
        return f"{acronym}_{i}"

    def filter_by_type(self,types):
        if isinstance(types, str):
            types=[types]
        nodes=[]
        for node in self.G.nodes:
            if self.G.nodes[node]['data'].type in types:
                nodes.append(node)
        return nodes
    
    def remove_redundant_edges(self,G):
        topological_order = list(nx.topological_sort(G))
        redundant_edges = []

        for node in topological_order:
            # Get all successors of the current node
            successors = list(G.successors(node))
            for succ in successors:
                # Temporarily remove the edge to check for other paths
                G.remove_edge(node, succ)
                if nx.has_path(G, node, succ):
                    redundant_edges.append((node, succ))
                # Re-add the edge
                G.add_edge(node, succ)

        # Remove all redundant edges
        for u, v in redundant_edges:
            G.remove_edge(u, v)
        
        return G

    def load(self):
        self.G=self.load_graph()
        if not self.db_only:
            self.load_design_sessions()
            self.GD = GAUDict.from_ptree(self)

    def load_graph(self,max_nodes=None):
        edges_to_add = []
        count=0
        G=nx.DiGraph()

        # Load designs
        implemented_designs=[]
        other_designs=[]
        for id in os.listdir(U.pjoin(self.db_dir,'designs')):
            artifact = DesignArtifact.load(self.design_dir(id))
            if artifact.type=='DesignArtifactImplemented':
                implemented_designs.append(artifact)
            else:
                other_designs.append(artifact)
        
        # Load primary library
        core_refs=[]
        other_refs=[]
        for id in os.listdir(self.lib_dir):
            ref = LibraryReference.load(self.lib_dir, id.split('.')[0])
            if ref.type in ['ReferenceCore','ReferenceCoreWithTree']:
                core_refs.append(ref)
            else:
                other_refs.append(ref)

        nodes_to_add=implemented_designs+other_designs+core_refs+other_refs
        for data in nodes_to_add:
            if max_nodes and count>max_nodes:
                break
            G.add_node(data.acronym, data=data)
            count+=1
            for seed_id in data.seed_ids:
                edges_to_add.append((seed_id, data.acronym))
         
        if self.db_only:
            return G
            
        # Add edges
        for seed_id, product_id in edges_to_add:
            if seed_id not in G.nodes or product_id not in G.nodes:
                continue
            if seed_id == product_id or nx.has_path(G, product_id, seed_id):
                continue
            G.add_edge(seed_id, product_id)
        
        G=self.remove_redundant_edges(G)

        return G
    
    def add_legend(self,G,legend_x=-300,legend_y=-250,legend_step=100,legend_font_size=20,legend_width_constraint=50):
        num_actual_nodes = len(self.G.nodes)
        legend_sizes = {
            'Reference\n(Size by cites)': 25,
            'Core Reference\n(Size by cites)': 25,
            'Reference with Code\n(Size by cites)': 25,
            'Proposed\n(Unverified)': DESIGN_SIZE,
            'Implemented\n(Unverified)': DESIGN_IMPLEMENTED_SIZE,
            'Failed Proposals': FAILED_SIZE,
            'Scratch Design\n(Unverified)': 15,
        }
        legend_colors = {
            'Core Reference\n(Size by cites)': CORE_COLOR,
            'Reference\n(Size by cites)': REFERENCE_COLOR,
            'Reference with Code\n(Size by cites)': RWC_COLOR,
            'Proposed\n(Unverified)': DESIGN_COLOR,
            'Implemented\n(Unverified)': DESIGN_IMPLEMENTED_COLOR,
            'Failed Proposals': FAILED_COLOR,
            'Scratch Design\n(Unverified)': ROOT_COLOR,
        }
        legend_labels=list(legend_colors.keys())
        for scale in NODE_COLOR_MAP:
            legend_labels.append('Verified '+scale)
            legend_colors[f'Verified {scale}'] = NODE_COLOR_MAP[scale]
            legend_sizes[f'Verified {scale}'] = NODE_SIZE_MAP[scale]
        num_legend_nodes = len(legend_labels)
        legend_nodes = [
            (
                num_actual_nodes, 
                {
                    'group': 'legend',
                    'label': 'Legend',
                    'size': 20,
                    'color': '#000000',
                    # 'fixed': True, # So that we can move the legend nodes around to arrange them better
                    'physics': False, 
                    'x': legend_x, 
                    'y': f'{legend_y}px',
                    'shape': 'box', 
                    'widthConstraint': 100, 
                    'font': {'size': legend_font_size}
                }
            )
        ]
        G.add_nodes_from(legend_nodes)

        legend_nodes = []
        for legend_node in range(num_legend_nodes):
            label = legend_labels[legend_node]
            size = legend_sizes[label]
            color = legend_colors[label]
            legend_nodes.append(
                (
                    num_actual_nodes +1 + legend_node, 
                    {
                        'group': legend_node, 
                        'label': label,
                        'size': size,
                        'color': color,
                        # 'fixed': True, # So that we can move the legend nodes around to arrange them better
                        'physics': False, 
                        'x': legend_x, 
                        'y': f'{legend_y + (1+legend_node)*legend_step}px',
                        # 'shape': 'box', 
                        # 'widthConstraint': legend_width_constraint, 
                        'font': {'size': legend_font_size}
                    }
                )
            )
        G.add_nodes_from(legend_nodes)
        return G

    def viz(self,G,height=5000,width="100%",layout=False,max_nodes=None,bgcolor="#fafafa",
            legend_x=-300,legend_y=-250,legend_step=100,legend_font_size=20,legend_width_constraint=50): # larger canvas may be needed for large trees
        # G=self.add_legend(G,legend_x,legend_y,legend_step,legend_font_size,legend_width_constraint)
        nt=Network(
            directed=True,height=height,width=width,
            layout=layout, 
            bgcolor=bgcolor, #font_color="#ffffff",
            #select_menu=True, # filter_menu=True,
            # heading=f'Phylogenetic Tree for {self.db_dir.split("/")[-2]}'
        )
        nt.prep_notebook(True)#,'./etc/ptree_template.html')
        nt.from_nx(G)
        fname='PTree' if not layout else 'PTree_layout'
        if max_nodes: fname+=f'_{max_nodes}'
        nt.show(U.pjoin(self.db_dir, '..', fname+'.html'))


    def export(self,max_nodes=None,height=5000,layout=False,bgcolor="#eeeeee",
               legend_x=-300,legend_y=-250,legend_step=100,legend_font_size=20,legend_width_constraint=100): #,with_ext=False
        G=nx.DiGraph()
        if not max_nodes or max_nodes==0 or max_nodes>=len(self.G.nodes):
            _G=self.G.copy()
        else:
            _G=self.load_graph(max_nodes)
        for idx,node in enumerate(_G.nodes):
            if max_nodes and idx>max_nodes:
                break
            data=_G.nodes[node]['data']
            if data.type in ['DesignArtifact','DesignArtifactImplemented']:
                color=DESIGN_COLOR if data.type=='DesignArtifact' else DESIGN_IMPLEMENTED_COLOR
                if data.seed_ids == []:
                    color=ROOT_COLOR
                size=DESIGN_SIZE if data.type=='DesignArtifact' else DESIGN_IMPLEMENTED_SIZE
                if not data.proposal.passed:
                    color=FAILED_COLOR
                    size=FAILED_SIZE
                for scale in list(NODE_COLOR_MAP.keys())[::-1]:
                    if scale in data.verifications:
                        color=NODE_COLOR_MAP[scale]
                        size=NODE_SIZE_MAP[scale]
            elif data.type=='Reference':
                color=REFERENCE_COLOR
                citations=data.citationCount
                size=5*max(0,int(math.log(citations,3)))+10 if citations else 10
            elif data.type=='ReferenceWithCode':
                color=RWC_COLOR
                citations=data.citationCount
                size=5*max(0,int(math.log(citations,3)))+10 if citations else 10
            elif data.type in ['ReferenceCore', 'ReferenceCoreWithTree']:
                color=CORE_COLOR
                citations=data.citationCount
                size=5*max(0,int(math.log(citations,3)))+10 if citations else 10
            # else: # VERY SLOW TO LOAD
            #     if not with_ext: continue
            #     color=EXT_COLOR_1HOC
            #     citations=data.citationCount
            #     size=5*max(0,int(math.log(citations,3)))+10 if citations else 10
            #     # continue # skip the 1hop reference, too much
            G.add_node(
                node,
                title=data.to_desc(),
                size=size,
                color=color,
                # scale=scale,
                # rating=data.rating
            )
        for edge in _G.edges:
            if edge[0] in G.nodes and edge[1] in G.nodes:
                G.add_edge(edge[0],edge[1])
        fname='phylogenetic_tree'
        if max_nodes: fname+=f'_{max_nodes}'
        # write_dot(G, U.pjoin(self.db_dir, '..', fname+".dot"))
        self.viz(G,max_nodes=max_nodes,height=height,layout=layout,bgcolor=bgcolor,
                 legend_x=legend_x,legend_y=legend_y,legend_step=legend_step,
                 legend_font_size=legend_font_size,legend_width_constraint=legend_width_constraint)




class ConnectionManager:
    def __init__(self, evoname, group_id, remote_db, stream):
        self.evoname = evoname
        self.group_id = group_id
        self.remote_db = remote_db
        self.collection = remote_db.collection('working_nodes')
        self.log_doc_ref = remote_db.collection('experiment_logs').document(evoname)
        self.zombie_threshold = NODE_ZOMBIE_THRESHOLD  # seconds
        self.st = stream
        self.max_design_threads={}
        self.accept_verify_job={}
        self.last_refresh = 0
        self.benchmark_mode = False

    def set_benchmark_mode(self,max_designs):
        self.benchmark_mode = True
        self.max_designs = max_designs
    
    def unset_benchmark_mode(self):
        self.benchmark_mode = False
        self.max_designs = None

    def switch_ckpt(self,evoname):
        self.evoname = evoname
        self.log_doc_ref = self.remote_db.collection('experiment_logs').document(evoname)

    def get_log_ref(self):
        latest_log = self.log_doc_ref.get().to_dict().get('latest_log',None)
        if latest_log:
            return self.log_doc_ref.collection('logs').document(latest_log)
        return None

    def start_log(self):
        timestamp = str(time.time())
        self.log_doc_ref.set({'latest_log':timestamp},merge=True)
        self.log_ref = self.log_doc_ref.collection('logs').document(timestamp)
    
    def set_group_id(self, group_id):
        self.group_id = group_id

    def clear_zombie_connections(self):
        threshold_time = datetime.now(timezone.utc) - timedelta(seconds=self.zombie_threshold)
        zombie_connections = self.collection.where('last_heartbeat', '<', threshold_time).get()
        
        for doc in zombie_connections:
            print(f"Clearing zombie connection: {doc.id}")
            doc.reference.delete()
    
    def get_active_connections(self):
        if time.time()-self.last_refresh < 1:
            return list(self.connections.keys())
        self.clear_zombie_connections()
        self.last_refresh = time.time()
        query = firestore.And([
            firestore.FieldFilter("status", "==", "connected"),
            firestore.FieldFilter("group_id", "==", self.group_id)
        ])
        connections = self.collection.where(filter=query).get()
        self.connections = {c.id: c.to_dict() for c in connections}
        for node_id in self.connections:
            self.max_design_threads[node_id] = self.connections[node_id]['max_design_threads']
            self.accept_verify_job[node_id] = self.connections[node_id]['accept_verify_job']
        return list(self.connections.keys())
    
    def _get_log(self,sess_id,log_collection_name,zombie_threshold=None):
        log_collection = self.log_doc_ref.collection(log_collection_name)
        index_term = log_collection.document('index').get()
        if not index_term.exists:
            return None,None,None
        index_term = index_term.to_dict()
        if sess_id not in index_term:
            return None,None,None
        log_df = None
        status = index_term[sess_id]['status']
        heartbeat = index_term[sess_id]['timestamp']
        latest_log = index_term[sess_id]['latest_log'] 
        if not latest_log:
            return None,None,None
        log_ref = log_collection.document(sess_id).collection('logs').document(latest_log)
        log = log_ref.get()
        if not log.exists:
            return None,None,None
        log = log.to_dict()
        log_df = pd.DataFrame(log).T
        log_df = log_df.sort_index(ascending=False)
        status = log_df.iloc[0]['status']
        heartbeat = log_df.index[0]
        if zombie_threshold and time.time()-float(heartbeat)>zombie_threshold:
            print(f'Detected zombie session: {sess_id}')
            index_ref = self.log_doc_ref.collection(log_collection_name).document('index')
            index_ref.set({sess_id:{
                'status':'ZOMBIE',
                'timestamp':str(time.time())
            }},merge=True)
            status = 'ZOMBIE'
        return log_df,status,heartbeat

    def get_session_log(self,sess_id):
        return self._get_log(sess_id,'design_sessions',DESIGN_ZOMBIE_THRESHOLD)
    
    def get_verification_log(self,sess_id):
        return self._get_log(sess_id,'verifications',VERIFY_ZOMBIE_THRESHOLD)
    
    def get_active_design_sessions(self):
        active_design_sessions = {}
        log_collection = self.log_doc_ref.collection('design_sessions')
        index_term = log_collection.document('index').get()
        index_ref = self.log_doc_ref.collection('design_sessions').document('index')
        if not index_term.exists:
            return {}
        index_term = index_term.to_dict()
        for sess_id in index_term:
            index_item = index_term[sess_id]
            if index_item['status'] in DESIGN_ACTIVE_STATES:
                # check if it is zombie, if it is, update the status and skip
                _,status,heartbeat = self.get_session_log(sess_id)
                if status != 'ZOMBIE':
                    index_item['status'] = status
                    index_item['heartbeat'] = heartbeat
                    active_design_sessions[sess_id] = index_item
        return active_design_sessions
    
    def get_running_verifications(self):
        running_verifications = {}
        log_collection = self.log_doc_ref.collection('verifications')
        index_term = log_collection.document('index').get()
        if not index_term.exists:
            return {}
        index_term = index_term.to_dict()
        for sess_id in index_term:
            index_item = index_term[sess_id]
            if index_item['status'] in VERIFY_ACTIVE_STATES:
                _,status,heartbeat = self.get_verification_log(sess_id)
                if status != 'ZOMBIE':
                    if status:
                        index_item['status'] = status 
                    index_item['heartbeat'] = heartbeat
                    running_verifications[sess_id] = index_item
        return running_verifications
    
    def _get_workloads(self,sess_data):
        workloads = {}
        for sess_id in sess_data:
            index_item = sess_data[sess_id]
            if 'node_id' not in index_item:
                continue
            node_id = index_item['node_id']
            index_item['sess_id'] = sess_id
            if node_id not in workloads:
                workloads[node_id] = []
            workloads[node_id].append(index_item)
        self.get_active_connections()
        for node_id in self.connections:
            if node_id not in workloads:
                workloads[node_id] = []
        return workloads
    

    def get_design_workloads(self):
        active_design_sessions = self.get_active_design_sessions()
        return self._get_workloads(active_design_sessions)
    
    def get_verification_workloads(self):
        running_verifications = self.get_running_verifications()
        return self._get_workloads(running_verifications)

    def get_all_workloads(self):
        design_workload = self.get_design_workloads()
        verify_workload = self.get_verification_workloads()
        return design_workload, verify_workload
    
    def check_design_workload(self,node_id):
        design_workload = self.get_design_workloads()
        return design_workload.get(node_id,[])
    
    def check_verification_workload(self,node_id):
        verify_workload = self.get_verification_workloads()
        return verify_workload.get(node_id,[])
    
    def check_workload(self,node_id):
        design_workload = self.check_design_workload(node_id)
        verify_workload = self.check_verification_workload(node_id)
        return design_workload, verify_workload

    def design_command(self,node_id,resume=True):
        self.get_active_connections() # refresh the connection status
        running_designs = self.check_design_workload(node_id)
        if len(running_designs) >= self.max_design_threads[node_id]:
            self.toast(f"Max number of design threads reached ({self.max_design_threads[node_id]}) for node {node_id}. Please wait for some threads to finish.",icon='🚨')
            return False
        command = f'design,{self.evoname}'
        if resume:
            command += ',resume'
        self.send_command(node_id,command)
        return True
    
    def verify_command(self,node_id,design_id=None,scale=None,resume=True):
        self.get_active_connections() # refresh the connection status
        if not self.accept_verify_job[node_id]:
            self.toast(f"Node {node_id} is not accepting verify jobs.",icon='🚨')
            return False
        running_verifies = self.check_verification_workload(node_id)
        if len(running_verifies) > 0:
            self.toast(f"There is already a verification running for node {node_id}. Please wait for it to finish.",icon='🚨')
            return False
        command = f'verify,{self.evoname}'
        if design_id:
            assert scale is not None, "Scale is required for verify command if design_id is specified"
            command += f',{design_id},{scale}'
        if resume:
            command += ',resume'
        self.send_command(node_id,command)
        return True

    def toast(self,message,icon=None,verbose=True):
        if self.st:
            self.st.toast(message,icon=icon)
        if verbose:
            print(message)
    
    def send_command(self, node_id, command):
        self.get_active_connections()
        try:
            node_ref = self.collection.document(node_id)
            update_time = node_ref.update({
                'commands': firestore.ArrayUnion([command]),
                'last_command_sent': firestore.SERVER_TIMESTAMP
            })
            self.toast(f"Command sent to {node_id}: {command}")
            self.toast(f"Update time: {update_time}",verbose=False)
            return True
        except Exception as e:
            self.toast(f"Error sending command to {node_id}: {str(e)}",icon='🚨')
            return False

    def disconnect_node(self, node_id):
        node_ref = self.collection.document(node_id)
        node_ref.update({'status': 'disconnected'})



DEFAULT_PARAMS = {
    'action_policy': 'random',
    'design_budget': 0,
    'scales': '14M,31M,70M,125M,350M',
    'selection_ratio': 0.25,
    'no_agent': False,
    'db_only': False,
    'use_remote_db': True,
    'group_id': 'default',
    'budget_type': 'design_bound',
    'n_target': 1,
    'challenging_threshold': 3,
    'benchmark_mode': False,
}


DEFAULT_N_SOURCES={ 
    'DesignArtifact':2,
    'DesignArtifactImplemented':0,
    'ReferenceCore':0,
    'ReferenceCoreWithTree':0,
    'Reference':2,
    'ReferenceWithCode':2,
}


BUDGET_TYPES = ['design_bound','verify_bound']

BENCH_MODE_OPTIONS = ['Mutation-only','Crossover-only','Scratch-only','Mixed']

DEFAULT_BENCHMARK_SETTINGS = {
    'n_trials': 100,
    'max_retries': None,
    'design_mode': 'Mutation-only',
    'n_seeds_dist': {'0': 0.1,'1': 0.8,'2': 0.1,'3': 0,'4': 0,'5': 0},
    'overwrite_config': True,
    'allow_tree': True,
}

# @exec_utils.Registry("config","evolution")
# class CustomParams(exec_utils.ModuleParams):
#     strparams: str = exec_utils.ParamField(
#         default='',
#         metadata={
#             "help"         : '";" separated parameters, e.g. "param1=val1;param2=val2", just use it for now',
#             "exclude_hash" : True,
#         }
#     )

@exec_utils.Registry(
    resource_type="system_type",
    name="evolution",
    #cache="query_system",
)
class EvolutionSystem(exec_utils.System):
    def __init__(self,agent_system,config,silent=False,**kwargs):
        self.agents = agent_system
        self._config = config
        self.params=config.params
        self.stream = None # PrintSystem(config,silent=silent)
        self.design_cfg = {}
        self.search_cfg = {}
        self.select_cfg = {}
        self.ve_cfg = {}
        self.benchmark_mode = False
        self.benchmark_settings = {}
        self.load(**kwargs)

    def load(self,**kwargs):
        self.remote_db = None
        self.CM = None
        db_key_path = os.environ.get("DB_KEY_PATH",None)
        if U.pexists(db_key_path):
            self.remote_db = firestore.Client.from_service_account_json(db_key_path)
            self.CM = ConnectionManager(self.params['evoname'], 'default', self.remote_db, self.stream)
            print(f'Remote db connected.')
        else:
            raise ValueError(f'No db key found at {db_key_path}. Only distributed evolution is supported now.')
            print(f'No db key found at {db_key_path}, using local db only. Will sync when remote db key is available.')

        # set the name and save dir
        assert 'evoname' in self.params, "evoname is required"
        self.evoname=self.params['evoname'] # Provide the name for the whole run including evolutions of all scales, all designs, all agents
        self.ckpt_dir=os.environ.get("CKPT_DIR")
        self.evo_dir=U.pjoin(self.ckpt_dir,self.evoname)
        U.mkdir(self.evo_dir)
        U.mkdir(U.pjoin(self.evo_dir,'ve'))
        
        self.load_config()

        self.params=U.init_dict(self.params,DEFAULT_PARAMS)

        if self.params['benchmark_mode']:
            self.set_benchmark_mode(self.benchmark_settings)
        else:
            self.unset_benchmark_mode()

        if self.CM:
            print(f"Connecting to group id: {self.params['group_id']}")
            self.CM.set_group_id(self.params['group_id'])
        
        self.action_policy=self.params['action_policy']
        self.design_budget_limit=self.params['design_budget']

        self._verify_budget=self.params.get('verify_budget',{})
        if self._verify_budget == {}:
            budget=self.params['n_target']
            for scale in self.params['scales'].split(',')[::-1]:
                self._verify_budget[scale]=int(np.ceil(budget))
                budget/=self.params['selection_ratio']
        self.target_scales=list(self._verify_budget.keys())
        self.target_scales.sort(key=lambda x:int(x.replace('M','')))

        self.max_samples = self.params.get('max_samples',0)

        self.scales=[eval(f'GAMConfig_{scale}()') for scale in self.target_scales]

        if self.stream:
            self.stream.write(f"Evolution system initialized with scales: {self.target_scales}")
            self.stream.write(f"Budget type: {self.params['budget_type']}")
            if self.max_samples>0:
                self.stream.write(f'Max samples: {self.max_samples}')
            self.stream.write(f'Design budget: {self.design_budget_limit}')
            self.stream.write(f"Verify budgets: {self._verify_budget}")
            self.stream.write(f"Checkpoint directory: {self.evo_dir}")

        self.ptree=PhylogeneticTree(self.evoname,self.target_scales,U.pjoin(self.evo_dir,'db'),self.params['db_only'],
                self.remote_db,self.params['use_remote_db'],challenging_threshold=self.params['challenging_threshold'],
                CM=self.CM,token_mults=self.ve_cfg.get('training_token_multipliers',DEFAULT_TOKEN_MULTS))
        print(f"Phylogenetic tree loaded with {len(self.ptree.G.nodes)} nodes and {len(self.ptree.design_sessions)} design sessions from {self.ptree.db_dir}.")
        
        # Scan VE for missing verifications
        ve_dir=U.pjoin(self.evo_dir,'ve')
        for design_scale in os.listdir(ve_dir):
            scale=design_scale.split('_')[-1]
            design_id=design_scale[:-len(scale)-1]
            node=self.ptree.get_node(design_id)
            if scale not in node.verifications:
                report_dir=U.pjoin(ve_dir,design_scale,'report.json')
                report=U.load_json(report_dir)
                self.ptree.verify(node.acronym,scale,report)

        self.selector = Selector(self.ptree,self.select_cfg,self._verify_budget,
            self.stream,self.design_budget_limit,self.params['budget_type'],
            token_mults=self.ve_cfg.get('training_token_multipliers',DEFAULT_TOKEN_MULTS),
            target_scales=self.target_scales)

        if self.params['no_agent']:
            self.agents = None
        else:
            self.agents = BuildSystem(
                debug_steps=False, # True for debugging, but very long
                # cache_type="diskcache", #<-- agent caching method 
                temperature=0.1,
                jupyter=False,
                # cache_id=919,
                #from_json='/path/to/config'
                **kwargs
            )
            self.agents.bind_ptree(self.ptree,self.stream)
            # self.ptree.export()

    def get_verify_budget(self,full=False):
        if full:
            _verify_budget = copy.deepcopy(self._verify_budget)
            _TARGET_SCALES = sorted(TARGET_SCALES,key=lambda x:int(x.replace('M','')))
            for scale in _TARGET_SCALES:
                _verify_budget[scale] = _verify_budget.get(scale,0)
            return _verify_budget
        else:
            return self._verify_budget
        
    def set_benchmark_mode(self,benchmark_settings={}):
        self.benchmark_mode = True
        self.benchmark_settings = U.init_dict(benchmark_settings,DEFAULT_BENCHMARK_SETTINGS)
        self.CM.set_benchmark_mode(benchmark_settings['n_trials'])
    
    def unset_benchmark_mode(self):
        self.benchmark_mode = False
        self.CM.unset_benchmark_mode()

    def should_stop(self):
        if self.CM.benchmark_mode:
            if len(self.ptree.get_finished_designs())>=self.CM.max_designs:
                return True
        else:
            if self.selector.budget_type=='design_bound':
                if self.selector.design_budget<=0:
                    return True
                elif self.max_samples>0 and len(self.ptree.get_finished_designs())>=self.max_samples:
                    return True
            elif self.selector.budget_type=='verify_bound':
                if sum(self.selector.verify_budget.values())<=0:
                    return True
        return False
    
    def conclude(self):
        # conclude results and report to db
        pass

    def get_evo_state(self):
        evo_state={}
        evo_state['Seed Selection Method']=self.design_cfg.get('select_method',DEFAULT_SELECT_METHOD)
        evo_state['Verification Strategy']=self.design_cfg.get('verify_strategy',DEFAULT_VERIFY_STRATEGY)
        evo_state['target_scales']=self.target_scales
        evo_state['Remaining Verify Budget']=self.selector.verify_budget
        evo_state['Remaining Design Budget']=self.selector.design_budget
        evo_state['Design Cost Spent']=self.ptree.design_cost
        evo_state['Use Remote DB']=self.ptree.use_remote_db
        if not self.remote_db:
            evo_state['Action Choice Policy']=self.action_policy
        else:
            evo_state['Network Group ID']=self.CM.group_id
        return evo_state

    def link_stream(self,stream):
        self.stream=stream
        self.agents.sss.stream=stream
        self.selector.stream=stream
        if self.CM is not None:
            self.CM.st = stream

    def reconfig(self,design_cfg=None,search_cfg=None,select_cfg=None,ve_cfg=None,benchmark_settings=None):
        if design_cfg is not None:
            self.design_cfg = design_cfg
        if search_cfg is not None:
            self.search_cfg = search_cfg
        if select_cfg is not None:
            self.select_cfg = select_cfg
        if ve_cfg is not None:
            self.ve_cfg = ve_cfg
        if benchmark_settings is not None:
            self.set_benchmark_mode(benchmark_settings)
        if design_cfg is not None or search_cfg is not None or select_cfg is not None or ve_cfg is not None or benchmark_settings is not None:
            self.save_config()

    def reload(self,params=None):
        if params:
            self.params = params
            self._config.params = params
        self.load()

    def switch_ckpt(self,ckpt_name,load_params=True):
        self.design_cfg = {}
        self.search_cfg = {}
        self.select_cfg = {}
        _params = {'evoname':ckpt_name}
        if self.remote_db:
            self.sync_from_db(ckpt_name)
        if load_params:
            config_path = U.pjoin(self.ckpt_dir,ckpt_name,'config.json')
            params = U.load_json(config_path).get('params',{})
            _params.update(params)
        self.reload(_params)

    def get_unfinished_verifies(self,evoname=None):
        if evoname is None:
            evoname = self.evoname
        ve_dir=U.pjoin(self.ckpt_dir,evoname,'ve')
        unfinished_verifies = []
        for v in os.listdir(ve_dir):
            if not U.pexists(U.pjoin(ve_dir,v,'report.json')):
                unfinished_verifies.append(v)
        return unfinished_verifies

    def query_system(self,
        query: Optional[str] = '',
        stream: Optional[ModuleType] = None,
        frontend: Optional[bool] = False,
        status: Optional[bool] = True,
        **kwargs
    ) -> list:
        """ Talk to the selector agent """
        if self.stream:
            self.stream.write("Hello from the evolution system")

    def sync_to_db(self):
        collection=self.remote_db.collection('experiments')
        config=U.load_json(U.pjoin(self.evo_dir,'config.json'))
        design_cfg = copy.deepcopy(self.design_cfg)
        if 'running_mode' in design_cfg:
            if not isinstance(design_cfg['running_mode'],str):
                design_cfg['running_mode'] = design_cfg['running_mode'].value
        config.update({
            'params': self.params,
            'design_cfg': design_cfg,
            'search_cfg': self.search_cfg,
            'select_cfg': self.select_cfg,
        })
        collection.document(self.evoname).set({'config': config})
        
    def sync_from_db(self,evoname=None):
        if evoname is None:
            evoname=self.evoname
        collection=self.remote_db.collection('experiments')
        doc=collection.document(evoname).get()
        if doc.exists:
            doc=doc.to_dict()
            config=doc.get('config',{})
            evo_dir=U.pjoin(self.ckpt_dir,evoname)
            U.mkdir(evo_dir)
            U.save_json(config,U.pjoin(evo_dir,'config.json'))

    def load_config(self):
        if self.remote_db:
            self.sync_from_db()
        config = U.load_json(U.pjoin(self.evo_dir,'config.json'))
        self.design_cfg = config.get('design_cfg',{})
        if 'running_mode' in self.design_cfg:
            self.design_cfg['running_mode'] = RunningModes(self.design_cfg['running_mode'])
        self.search_cfg = config.get('search_cfg',{})
        self.select_cfg = config.get('select_cfg',{})
        self.ve_cfg = config.get('ve_cfg',{})
        self.benchmark_settings = config.get('benchmark_settings',{})
        params = config.get('params',{})
        params.update(self.params) # to directly use config, provide only evoname in config
        self.params = params # logic is that, if new params provided, use new params, otherwise, use config, otherwise, use default
        return config
    
    def save_config(self):
        config = U.load_json(U.pjoin(self.evo_dir,'config.json'))
        config['design_cfg'] = self.design_cfg
        if 'running_mode' in config['design_cfg']:
            if not isinstance(config['design_cfg']['running_mode'],str):
                config['design_cfg']['running_mode'] = config['design_cfg']['running_mode'].value
        config['search_cfg'] = self.search_cfg
        config['select_cfg'] = self.select_cfg
        config['ve_cfg'] = self.ve_cfg
        config['benchmark_settings'] = self.benchmark_settings
        U.save_json(config,U.pjoin(self.evo_dir,'config.json'))
        if self.ptree.FM:
            self.ptree.FM.upload_experiment(self.evoname,config)
            self.stream.toast(f"Saved experiment configs to remote DB: {self.evoname}")

    def check_budget(self,action):
        if action=='design': # check the design budget
            if self.design_budget<=0:
                return False
        elif action=='verify':
            if sum(self.selector.verify_budget.values())<=0:
                return False
        else:
            raise ValueError(f"Invalid action: {action}")
        return True
        
    def evolve_step(self): # each time do one step, agent choose what to do, use shell to run it continuously 
        raise NotImplementedError("Evolve step is still working in progress")
        # NOTE: sequential evolve is not considered for now, as it involves action policy which is complicated
        if not self.check_budget('design') and not self.check_budget('verify'):
            self.stream.write(f"No budget for design and verify, evolution stops, system sleeps, please terminate manually")
            time.sleep(1e9)
            return
        act=self.choose()
        if not self.check_budget(act):
            self.stream.write(f"No budget for {act}, will do another action")
            act='design' if act=='verify' else 'verify'

        act='verify' # XXX: for testing

        if act=='design':
            self.design() # FIXME: it needs to be updated with actual selector and design cfg
        elif act=='verify':
            self.verify()
            
    def _process_manual_input(self,manual):
        if manual is None:
            return None
        if isinstance(manual,str):
            return [self.ptree.get_node(manual)]
        elif isinstance(manual,NodeObject):
            return [manual]
        elif isinstance(manual,list):
            _nodes = []
            for i in manual:
                if isinstance(i,str):
                    _nodes.append(self.ptree.get_node(i))
                elif isinstance(i,NodeObject):
                    _nodes.append(i)
                else:
                    raise ValueError(f"Invalid manual input: {i}")
            return _nodes
        else:
            raise ValueError(f"Invalid manual input: {manual}")

    def design(self,select_cfg=None,design_cfg=None,search_cfg=None,user_input='',sess_id=None,n_seeds=None,
        resume=True,in_process=False,manual_seed=None,manual_refs=None,silent=False,cpu_only=False
    ): 
        # user_input and design_cfg maybe changed by the user, so we need to pass them in
        # self.ptree.reload() # WHY WE NEED THIS???
        
        if select_cfg is None:
            select_cfg = self.select_cfg
        if design_cfg is None:
            design_cfg = self.design_cfg
        else:
            _design_cfg = self.design_cfg.copy()
            _design_cfg.update(design_cfg)
            design_cfg = _design_cfg
        if search_cfg is None:
            search_cfg = self.search_cfg

        if self.benchmark_mode:
            if self.benchmark_settings['design_mode']=='Mutation-only':
                select_cfg['n_seeds_dist'] = {'0': 0, '1': 1.0, '2': 0, '3': 0, '4': 0, '5': 0}
            elif self.benchmark_settings['design_mode']=='Crossover-only':
                select_cfg['n_seeds_dist'] = {'0': 0, '1': 0, '2': 1.0, '3': 0, '4': 0, '5': 0}
            elif self.benchmark_settings['design_mode']=='Scratch-only':
                select_cfg['n_seeds_dist'] = {'0': 1.0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            else:
                if self.benchmark_settings['overwrite_config']:
                    select_cfg['n_seeds_dist'] = self.benchmark_settings['n_seeds_dist']
                    select_cfg['n_seeds_settings'] = {'warmup_rounds_scratch': 0,'warmup_rounds_crossover': 0}
            if self.benchmark_settings['max_retries'] is not None:
                self.ptree.challenging_threshold = self.benchmark_settings['max_retries']


        unfinished_designs = self.ptree.get_unfinished_designs()
        if self.stream:
            self.stream.write(f"Found {len(unfinished_designs)} unfinished designs, allow resume: {resume}")

        selector_args={}
        selector_args['n_sources']=select_cfg.get('n_sources',DEFAULT_N_SOURCES)

        manual_seed = self._process_manual_input(manual_seed)
        manual_refs = self._process_manual_input(manual_refs)

        def _new_sample(selector_args,sess_id=None,_silent=False,_cpu_only=False):
            if self.benchmark_mode:
                select_cfg['select_method'] = 'random'
                selector_args['allow_tree'] = self.benchmark_settings['allow_tree']
                instruct,seeds,refs=self.selector.select_design(selector_args,n_seeds=n_seeds,select_cfg=select_cfg) # use the seed_ids to record the phylogenetic tree
            else:
                instruct,seeds,refs=self.selector.select_design(selector_args,n_seeds=n_seeds,select_cfg=select_cfg) # use the seed_ids to record the phylogenetic tree
            seeds = manual_seed if manual_seed is not None else seeds
            refs = manual_refs if manual_refs is not None else refs
            self.sample(instruct,seeds,refs,sess_id=sess_id,user_input=user_input,
                design_cfg=design_cfg,search_cfg=search_cfg,silent=_silent,cpu_only=_cpu_only)

        if sess_id is None:
            if len(unfinished_designs)==0 or not resume:
                print('No unfinished designs, will start a new design session')
                _new_sample(selector_args,_silent=silent,_cpu_only=cpu_only) # use the seed_ids to record the phylogenetic tree
            else:
                sess_id = random.choice(unfinished_designs)
                print(f'Found {len(unfinished_designs)} unfinished designs, will restore a random one: {sess_id}')
                passed,implemented,challenging,unfinished=self.ptree.get_session_state(sess_id)
                mode=DesignModes(self.ptree.session_get(sess_id,'mode'))
                if self.stream:
                    self.stream.write(f"Restoring a session {sess_id}, mode: {mode}. {len(passed)} proposals passed, {len(implemented)} implemented, {len(unfinished)} are unfinished where {len(challenging)} are challenging.")
                self.sample(sess_id=sess_id,user_input=user_input,design_cfg=design_cfg,search_cfg=search_cfg,silent=silent,cpu_only=cpu_only) # should not change the design_cfg
        else:
            sessdata = self.ptree.get_design_session(sess_id)
            if sessdata is not None:
                print(f'Design id provided and exists, will restore session {sess_id}')
                mode=DesignModes(self.ptree.session_get(sess_id,'mode'))
                if self.stream:
                    self.stream.write(f"Design id provided, will restore session {sess_id}, mode: {mode}")
                self.sample(sess_id=sess_id,user_input=user_input,design_cfg=design_cfg,search_cfg=search_cfg,silent=silent,cpu_only=cpu_only)
            else: # create a new design session using an external id
                print(f'Create a new design session using an external id: {sess_id}')
                _new_sample(selector_args,sess_id=sess_id,_silent=silent,_cpu_only=cpu_only) # use the seed_ids to record the phylogenetic tree
        if in_process: # exit the process after sampling
            sys.exit(0)

    def sample(self,instruct=None,seeds:List[NodeObject]=None,refs:List[NodeObject]=None,sess_id=None,
        user_input='',design_cfg={},search_cfg={},silent=False,cpu_only=False
    ):
        """ 
        Sample a design at a given scale and verify it 
        
        Input optional selector instruct and metadata, return a design artifact
        DB should be fully managed by the agent system, likewise, VE should be fully managed by the verification engine
        agent system should only have access to DB, and VE should only have access to VE

        Selector choose which seeds to use, and budget for verification
        Given the seeds which direct the global direction, the agent system should be fully responsible for the best local move
        """

        log_collection = self.CM.log_doc_ref.collection('design_sessions') if self.CM else None

        self.agents(
            user_input,
            instruct=instruct,
            seeds=seeds,
            refs=refs,
            sess_id=sess_id,
            stream=self.stream,
            design_cfg=design_cfg,
            search_cfg=search_cfg,
            silent=silent,
            cpu_only=cpu_only,
            log_collection=log_collection
        )

    def choose(self): # For sequential evolution, not needed for parallel
        """ Choose a move, select a design to verify """
        if self.action_policy=='random':
            return random.choice(['design','verify'])
        else:
            raise ValueError(f"Invalid action policy: {self.action_policy}")

    def _prep_ve_args(self,args,design_id,scale):
        training_token_multipliers = self.ve_cfg.get('training_token_multipliers',DEFAULT_TOKEN_MULTS)
        args.training_token_multiplier=training_token_multipliers[scale]
        args.wandb_project=self.ve_cfg.get('wandb_project',DEFAULT_WANDB_PROJECT)
        args.wandb_entity=self.ve_cfg.get('wandb_entity',DEFAULT_WANDB_ENTITY)
        args.eval_tasks=self.ve_cfg.get('eval_tasks',','.join(DEFAULT_EVAL_TASKS))
        args.training_data=self.ve_cfg.get('training_data',','.join(DEFAULT_TRAINING_DATA))
        args.tokenizer=self.ve_cfg.get('tokenizer',DEFAULT_TOKENIZER)
        args.context_length=self.ve_cfg.get('context_length',DEFAULT_CONTEXT_LENGTH)
        args.optim=self.ve_cfg.get('optim',DEFAULT_OPTIM)
        args.seed=self.ve_cfg.get('seed',DEFAULT_RANDOM_SEED)
        args.save_steps=self.ve_cfg.get('save_steps',DEFAULT_SAVE_STEPS)
        args.logging_steps=self.ve_cfg.get('logging_steps',DEFAULT_LOG_STEPS)
            
        args.evoname=self.evoname
        args.design_id=design_id+f'_{scale}'
        args.ckpt_dir=os.environ.get("CKPT_DIR")
        args.data_dir=os.environ.get("DATA_DIR")
        return args

    def verify(self,args,design_id=None,scale=None,in_process=False): # choose then verify
        if design_id is None or scale is None:
            self.ptree.update_design_tree()
            design_id,scale=self.selector.select_verify()
        if design_id is None or scale is None: # no available design to verify
            return None
        if self.stream:
            self.stream.write(f"Verifying design {design_id} at scale {scale}...")
        else:
            print(f"Verifying design {design_id} at scale {scale}...")
        args = self._prep_ve_args(args,design_id,scale)

        log_fn = None
        if self.CM:
            log_collection = self.CM.log_doc_ref.collection('verifications')
            latest_log = str(time.time())
            sess_id = f'{design_id}_{scale}'
            log_ref = log_collection.document(sess_id).collection('logs').document(latest_log)
            index_ref = log_collection.document('index')
            def log_fn(msg,status='RUNNING'):
                ve_dir = U.pjoin(self.evo_dir, 've', sess_id)
                url='N/A'
                if os.path.exists(ve_dir):
                    wandb_ids = U.load_json(U.pjoin(ve_dir, 'wandb_ids.json'))
                    if 'pretrain' in wandb_ids:
                        wandb_id=wandb_ids['pretrain']['id']
                        project=wandb_ids['project']
                        entity=wandb_ids['entity']
                        url=f'https://wandb.ai/{entity}/{project}/runs/{wandb_id}'
                timestamp = str(time.time())
                log_ref.set({
                    timestamp:{
                        'status':status,
                        'message':msg
                    }
                },merge=True)
                # if status in DESIGN_TERMINAL_STATES+['BEGIN']: # only update the index at begining and ends
                index_ref.set({
                    sess_id:{
                        'timestamp':timestamp,
                        'status':status,
                        'latest_log':latest_log,
                        'wandb_url':url
                    }
                },merge=True)
                    
        ve_main(args,log_fn) # verify the design until it's done
        if args.RANDOM_TESTING:
            report_dir=U.pjoin(self.ckpt_dir,'random','ve','random','report.json')
        else:
            report_dir=U.pjoin(self.evo_dir,'ve',design_id+f'_{scale}','report.json')
        report=U.load_json(report_dir)
        token_mult = self.ve_cfg.get('training_token_multipliers',DEFAULT_TOKEN_MULTS)[scale]
        self.ptree.verify(design_id,scale,report,RANDOM_TESTING=args.RANDOM_TESTING,token_mult=token_mult)
        if in_process:
            sys.exit(0)
        if report!={}: 
            return 'SUCCESS'
        return 'FAILED'

    def _prep_model(self,design_id,scale):
        from model_discovery.model.library.tester import check_tune
        
        design=self.ptree.get_node(design_id) # need to ensure this design has not been verified under scale
        design_id=design.acronym
        ### XXX need manully check then comment it, need to fix, TUNE cause the problem
        if design.type=='DesignArtifactImplemented':
            _code = design.implementation.implementation.compose()
        else:
            code_dir=U.pjoin(LIBRARY_DIR,'core',design_id,design_id+'.py')
            if U.pexists(code_dir):
                _code = U.read_file(code_dir)
            else:
                raise FileNotFoundError(f"Code file not found for design {design_id}")
        code = check_tune(scale,design_id, code=_code,check_only=True,cpu_only=False,reformat_only=True)
        
        with open('./model_discovery/model/gab.py','w', encoding='utf-8') as f:
            f.write(code)

        CKPT_DIR = os.environ.get('CKPT_DIR')
        local_doc_dir = U.pjoin(CKPT_DIR, '.node.json')
        local_doc = U.load_json(local_doc_dir)
        local_doc['model_ready'] = True
        U.save_json(local_doc, local_doc_dir)

    @classmethod
    def from_config(cls,config,silent=False,**kwargs):
        """Loads all the evolution components from configuration 

        :param config:
            The global configuration spec. 

        """
        config.system_type = "model_discovery_system"
        agent = BuildSystem(
            config,
            **kwargs
        )
        return cls(agent,config,silent=silent) 

def BuildEvolution(
        config: Optional[ConfigType] = None,
        stream: Optional[ModuleType] = None,
        silent: Optional[bool] = False,
        **kwargs
    ) -> EvolutionSystem:
    """Factory for loading evolution system 

    :param config: 
        Configuration object (optional) 

    """
    kwargs["system_type"] = "evolution"
    evolution = NativeBuild(config,silent=silent,**kwargs)
    if stream:
        evolution.link_stream(stream)
    return evolution


############################################################################################################

def test_evolve(params,step=False):
    evolution_system = BuildEvolution(
        params=params,
        do_cache=True,
        cache_type='diskcache',
    )
    while evolution_system.evolve_step():
        if step:
            break


if __name__ == '__main__':
    args = ve_parser.parse_args()
    # print('*'*20)
    # print('Parsed args:',args)
    # print('*'*20)
    if args.mode=='test':
        params={
            'evoname':'temp',
            'scales':'14M,31M,70M',
            'selection_ratio':0.25,
            'design_budget':0,
        }
        test_evolve(params,step=True)
    else:
        if args.params=='':
            raise ValueError("Params is required")
        params=json.loads(args.params)
        args.evoname=params['evoname']
        
        if args.mode=='prep_model':
            params['no_agent']=True
            params['db_only']=True
        evolution_system = BuildEvolution(
            params=params,
            do_cache=False,
            silent=args.silent,
            # cache_type='diskcache',
        )
        
        if args.mode=='prep_model':
            evolution_system._prep_model(args.design_id, args.scale)
        elif args.mode=='verify':
            design_id = None if args.design_id=='None' else args.design_id
            scale = None if args.scale=='None' else args.scale
            evolution_system.verify(args,design_id=design_id,scale=scale,in_process=True)
        elif args.mode=='design':
            sess_id=None if args.sess_id=='' else args.sess_id
            evolution_system.design(sess_id=sess_id,in_process=True,silent=args.silent,cpu_only=args.cpu_only)
        elif args.mode=='evolve_step': # Sequential evolve one step
            evolution_system.evolve_step()
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
