
import json
import time
import streamlit as st
import sys,os
import random
import shutil
sys.path.append('.')

import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.evolution import ConnectionManager

from model_discovery.agents.roles.selector import *
import datetime

from model_discovery.model.composer import GAUTree,GAUNode,UnitSpec
from model_discovery.agents.flow.gau_flows import GAU_TEMPLATE

import pandas as pd

from model_discovery.agents.roles.selector import scale_weight_results

import tiktoken
import anthropic

from model_discovery.agents.agent_utils import *
import model_discovery.utils as U




def tester(evosys,project_dir):
    
    ### Sidebar
    with st.sidebar:
        AU.running_status(st,evosys)

    st.title('Testing page (for internal use)')

#     sess_id='2024-09-17-16-43-41-6f5f87'
#     tail = sess_id.split('-')[-1]
#     timestr=sess_id[:-len(tail)-1]
#     st.write(timestr)

#     sess_id = '2024-10-11-20-49-17-555a28'


#     example = """
# Many words map to one token, but some don't: indivisible.

# Unicode characters like emojis may be split into many tokens containing the underlying bytes: 🤚🏾

# Sequences of characters commonly found next to each other may be grouped together: 1234567890
# """

        
#     OPENAI_MODELS=list(OPENAI_TOKEN_LIMITS.keys())
#     ANTHROPIC_MODELS=list(ANTHROPIC_TOKEN_LIMITS.keys())

    # example=example*200
    # history = [
    #     (example*20, 'user'),
    #     ('hi'+example*20, 'assistant'),
    #     ('how are you?'+example*20, 'user'),
    #     ('I am fine, thank you!'+example*20, 'assistant'),
    # ]

    # for model_name in ANTHROPIC_MODELS:
    #     # truncated_text = truncate_text(example,40,model_name,buffer=10)
    #     st.write(model_name)
    #     # st.write(count_tokens(truncated_text,model_name))
    #     # st.write(truncated_text)

    #     history = context_safe_guard(history,model_name)
    #     st.write(count_tokens(str(history),model_name))


    # sss = evosys.agents.sss
    # embeddings = sss.embedding_proposal
    # evaluator = sss.emb_evaluator_proposal


    upload_csv = st.file_uploader('Upload CSV',type='csv')
    if upload_csv:
        df = pd.read_csv(upload_csv)
        df.rename(columns={'Unnamed: 0':'design'},inplace=True)
        df.reset_index(drop=True,inplace=True)
        df.set_index('design',inplace=True)
        # st.dataframe(df,use_container_width=True)
        # fetch rows with (baseline)
        baseline_rows = df[df.index.str.contains('(baseline)')]
        # fetch 'random' row
        random_row = df.loc['random']
        # remaining rows
        remaining_rows = df[~df.index.isin(baseline_rows.index)]
        remaining_rows = remaining_rows.drop(index='random')
        # sort remaining rows by 'avg.' column
        remaining_rows = remaining_rows.sort_values(by='avg.',ascending=True)
        # recombine
        df = pd.concat([random_row.to_frame().T,baseline_rows,remaining_rows])
        # bolden the highest row each column
        st.subheader(f'Evaluation scores under `14M`')
        st.dataframe(df.style.highlight_max(axis=0,color='violet'),use_container_width=True)



    # st.write(list(evosys.ptree.design_sessions.items())[0])

    # st.write(len(evosys.ptree.filter_by_type(['DesignArtifact'])))
    # st.write(len(os.listdir(U.pjoin(evosys.evo_dir,'db','designs'))))

    # da = evosys.ptree.filter_by_type(['DesignArtifactImplemented'])
    # ds = evosys.ptree.filter_by_type(['DesignArtifact'])
    # random.seed(42)
    # da_100 = random.sample(da,100)

    # benchmark_data_dir = U.pjoin(project_dir,'model_discovery','agents','bench_data')
    # st.write(len(os.listdir(benchmark_data_dir)))
    
    # if not U.pexists(benchmark_data_dir):
    #     U.mkdir(benchmark_data_dir)
    # for design in da_100:
    #     ndir = U.pjoin(evosys.evo_dir,'db','designs',design)
    #     ddir = U.pjoin(benchmark_data_dir,design)
    #     if not U.pexists(ddir):
    #         U.mkdir(ddir)
    #     if not U.pexists(U.pjoin(ddir,'proposal.json')):
    #         shutil.copy(U.pjoin(ndir,'proposal.json'),U.pjoin(ddir,'proposal.json'))
    #     if not U.pexists(U.pjoin(ddir,'metadata.json')):
    #         shutil.copy(U.pjoin(ndir,'metadata.json'),U.pjoin(ddir,'metadata.json'))
    #     metadata = U.load_json(U.pjoin(ddir,'metadata.json'))
    #     sess_snapshot = metadata['sess_snapshot']
    #     if not U.pexists(U.pjoin(ddir,'sess_snapshot.json')):
    #         sess_snapshot['proposed'] = [design]
    #         sess_snapshot['reranked'] = {'rank':[design]}
    #         U.save_json(sess_snapshot,U.pjoin(ddir,'sess_snapshot.json'))
    #     seed_ids = sess_snapshot['seed_ids']
    #     ref_ids = sess_snapshot['ref_ids']
    #     seeds_dir = U.pjoin(ddir,'seeds')
    #     for seed_id in seed_ids:
    #         if seed_id in da+ds:
    #             if not U.pexists(U.pjoin(seeds_dir,seed_id)):
    #                 shutil.copytree(U.pjoin(evosys.evo_dir,'db','designs',seed_id),U.pjoin(seeds_dir,seed_id))
    #     ref_dir = U.pjoin(ddir,'refs')
    #     for ref_id in ref_ids:
    #         if ref_id in da+ds:
    #             if not U.pexists(U.pjoin(ref_dir,ref_id)):
    #                 shutil.copytree(U.pjoin(evosys.evo_dir,'db','designs',ref_id),U.pjoin(ref_dir,ref_id))
        

    # da = evosys.ptree.filter_by_type('DesignArtifact')
    # st.write(da)
    # node = evosys.ptree.get_node('hybridstatespacetransformer')
    # st.write(node)

    # unfinished = evosys.ptree.get_unfinished_designs()
    # st.write(unfinished)


    # failed = []
    # designs = evosys.ptree.filter_by_type(['DesignArtifact'])
    # for design in designs:
    #     node = evosys.ptree.get_node(design)
    #     if node.state == 'failed':
    #         failed.append(design)
    # st.write(failed)


    # d = random.choice(evosys.ptree.filter_by_type(['DesignArtifactImplemented']))
    # node = evosys.ptree.get_node(d)
    # st.write(node.implementation.history[-1])


    # evosys.ptree.FM.sync_to_db()




    ##################### Debugging C Guard #####################

    # from model_discovery.agents.agent_utils import context_safe_guard,count_msg

    # def count_message(message,model_name):
    #     return sum([count_msg(msg,model_name) for msg in message])
    
    # CKPT_DIR = os.environ['CKPT_DIR']
    # debug_dir = os.path.join(CKPT_DIR,'ctx_error_logs')
    # if not os.path.exists(debug_dir):
    #     st.write('No debug records found')
    # else:
    #     debug_records = []
    #     for file in os.listdir(debug_dir):
    #         debug_records.append(U.load_json(os.path.join(debug_dir,file)))
    #     st.write(len(debug_records))
    #     message = debug_records[1]
    #     model_name = 'o1-preview'
    #     st.write(count_message(message,model_name))
    #     st.write(len(message))
    #     message=context_safe_guard(message,model_name)
    #     st.write(count_message(message,model_name))
    #     st.write(len(message))
    #     # for msg in message:
    #     #     st.code(msg['content'][-100:])





                    
        
    # for debug_record in debug_records:
    #     st.write('---')
    #     history = debug_record['history']
    #     prompt = debug_record['prompt']
    #     system = debug_record['system']  
    #     system_tokens = count_tokens(system, model_name)
    #     prompt_tokens = count_tokens(prompt, model_name)
    #     history_tokens = [count_tokens(content, model_name) for content, _ in history]
    #     total_tokens = system_tokens + prompt_tokens + sum(history_tokens)
    #     st.write('before',total_tokens,system_tokens,prompt_tokens,sum(history_tokens),len(history))
    #     history, prompt, system = truncate_history(history,prompt,system)
    #     system_tokens = count_tokens(system, model_name)
    #     prompt_tokens = count_tokens(prompt, model_name)
    #     history_tokens = [count_tokens(content, model_name) for content, _ in history]
    #     total_tokens = system_tokens + prompt_tokens + sum(history_tokens)
    #     st.write('after',total_tokens,system_tokens,prompt_tokens,sum(history_tokens),len(history))
    #     # break
