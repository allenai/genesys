import copy
import json
import time
import pathlib
import streamlit as st
import sys,os
import numpy as np
from model_discovery.agents.flow.gau_flows import DesignModes

from model_discovery.evolution import DEFAULT_N_SOURCES

sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

from model_discovery.agents.roles.selector import DEFAULT_SEED_DIST,SCHEDULER_OPTIONS



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

#### :red[*TODO Next*]

1. **RL/tuning of design agents** The design agents can use the utility as a signal to improve the design over the evolution process.
2. **Online learning of scaling curve for selector** The selector can learn the scaling curve to guide the exploration better. 

'''



def design_selector(evosys,project_dir):
    st.title('Design Selector')

    seed_dist = copy.deepcopy(DEFAULT_SEED_DIST)

    with st.expander('Design Selector basic settings',expanded=True):
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

    with st.expander('Ranking and Design Exploration settings'):
        st.write('**TODO** refer to viewer for details')

    if st.button('Select'):
        selector_args = {
            'n_sources': n_sources,
        }
        
        with st.status('Selecting seeds...'):
            instruct,seeds,refs=evosys.selector.select_design(selector_args,DesignModes(mode),select_method)


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

    # if verify_strategy == 'random':
    #     st.subheader('Random Strategy')
    #     st.write('*Random strategy will use up smaller scale budgets first.*')

    with st.expander('Ranking and Verify Exploration settings'):
        st.write('**TODO** refer to viewer for details')

    verify_selected = st.button('Select')

    if verify_selected:
        with st.status('Selecting design to verify...'):
            design_id,scale=evosys.selector.select_verify(verify_strategy=verify_strategy)
        if design_id is None:
            st.warning('No design to verify.')
        else:
            st.write(f'Selected {design_id} at scale {scale} to be verified.')



def select(evosys,project_dir):
    

    if 'select_view_mode' not in st.session_state:
        st.session_state.select_view_mode = 'design'

    with st.sidebar:
        AU.running_status(st,evosys)

        st.write('Choose Mode')
        if st.button('**Design Selector**' if st.session_state.select_view_mode == 'design' else 'Design Selector',use_container_width=True):
            st.session_state.select_view_mode = 'design'
        if st.button('**Verify Selector**' if st.session_state.select_view_mode == 'verify' else 'Verify Selector',use_container_width=True):
            st.session_state.select_view_mode = 'verify'
        # if st.button('***Selector Lab***' if st.session_state.select_view_mode == 'lab' else '*Selector Lab*',use_container_width=True):
        #     st.session_state.select_view_mode = 'lab'


    if st.session_state.select_view_mode == 'design':
        design_selector(evosys,project_dir)
    elif st.session_state.select_view_mode == 'verify':
        verify_selector(evosys,project_dir)
    # elif st.session_state.select_view_mode == 'lab':
    #     selector_lab(evosys,project_dir)


