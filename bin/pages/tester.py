
import json
import time
import streamlit as st
import sys,os
import random
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


    # da = evosys.ptree.filter_by_type('DesignArtifact')
    # st.write(da)
    # node = evosys.ptree.get_node('hybridstatespacetransformer')
    # st.write(node)

    unfinished = evosys.ptree.get_unfinished_designs()
    st.write(unfinished)

    