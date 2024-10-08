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

The selector's goal is to find the **global optimal design** by exploring the
evolution tree. Each node's utility is determined by its *potential* to be the
optimal language model design at larger, unforeseen computational scales. For
instance, the computational budget may only allow us to train a 130B model once
but smaller models more frequently. Therefore, we must make design choices at
smaller scales, hoping they succeed when applied to larger scales. This
challenge is common not only in language modeling but also in other scientific
exploration problems.

To address this, the selector must: 
 1. **Forecast the performance of a design at larger scales.** 
 2. **Increase the certainty of this forecast.**


These are achieved via **design selection** and **verification selection**.
Along with maximizing utility, the selector also need to consider the
exploration and exploitation trade-off to ensure a sufficiently large search
space.

##### *The relations between the selector and other parts of the system*

The selector is one of three core pillars in the evolution system, it guides the
search process by orchestrating the other two:

- **Model design agents**: Generates new designs based on the seeds, references,
  and potential instructions the selector provided in design selection. It aims
  high-quality *local* sampling to deal with the high sampling cost happening in
  language modeling.

- **Distributed V/D engine**: Call design threads and evaluate designs/scales
  selected in verification selection efficiently, parallelly, and robustly. It
  prioritizes throughput and distributed robustness to improve sampling
  efficiency which is crucial for evolution.

##### *Selector's Responsibilities*

The selector guides the evolution and search process by making two types of
decisions: design and verification selections.

1. **Design selection**: This process generates the following outputs for the
   design agent:
    - *Seed*: The base design from which a new design is evolved by mutating one
      unit.
    - *Reference*: Recommended papers or codes to *cold-start* ideation process.
    - *Instruction*: *Optional* hints for guiding the mutation, either from the
      selector or the user.

2. **Verification selection**: This determines:
    - *Design*: The design to be verified.
    - *Scale*: The scale at which the design will be verified.

   *(**Note**: While we focus on mutation (which operates on a single seed),
   other modes can be built based on mutation like crossover by reusing units
   from references, and scratch designs by rewriting root nodes.)*



#### *Design Valuation*

- **Utility**: A scale-normalized metric that indicates the potential for a
  design to perform better at larger scales. 
Ideally, the precise measure is the AUC of the scaling curve. However, the main
challenge is **forecasting the scaling curve**, which may require online
learning. The utility can be estimated based on:
    1. **Design artifacts**: Including proposals, implementation details, and GAU
        tree structure.
    2. **Verification results**: Training and evaluation metrics at various scales.
  
- **Confidence**: Confidence is based on the amount of available information, it
  indicates how we can for sure say that a design is good, it serves as a proxy
  for the scaling curve. The confidence comes from:
    1. **Available information**: Newly created designs only have design artifacts,
        making their confidence low. This can be improved by verifying them at
        different scales.
    2. ***Prior knowledge (future work)***: For instance, we have significant
        knowledge of Transformer scaling characteristics from previous research.
        Similarly, the selector can learn the scaling behavior of other architectures
        over time.

#### *Selection framework*

The selector divides designs into four quadrants by utility and confidence :

1. **High utility, high confidence**: Indicates a strong design to further
   exploit. The selector prioritizes using it as a seed. :blue[*(Design selector
   exploit)*]
2. **High utility, low confidence**: A promising design worth investigating
   further by verifying it in more scales. :red[*(Verify selector exploit)*]
3. **Low utility, high confidence**: A weak design, but with potential for
   improvement. The selector may choose to mutate it with lower priority.
   :blue[*(Design selector explore)*]
4. **Low utility, low confidence**: An uncertain design that might perform well
   at larger scales. The selector may verify it at different scales with lower
   priority. :red[*(Verify selector explore)*]

Besides on seed selection, the design selector also needs to consider two
additional aspects:

1. **Reference selection**: The selector chooses references that could lead to a
   promising mutation. Currently, this is done randomly but could be enhanced
   with graph or embedding-based strategies.
2. **Instruction generation**: The selector may generate instructions to guide
   promising mutations, though this is not implemented yet. A planner agent
   could handle this in the future.


For verification, the selector focuses on:

1. **Scale selection**: Given budget constraint, more resources are allocated to
   exploitation and fewer to exploration. 
2. **Scaling characteristic exploration**: Verification not only supports design
   selection but also helps explore the scaling behavior of new or unfamiliar
   design families, which may require more resources for exploration. (future
   work)


##### *Random Exploration*

- **For design selection**: A random exploration strategy with an annealing
  scheduler is used. Bandit-based exploration strategies may be added in future
  work.
  
- **For verification selection**: Scale selection begins at lower scales and
  gradually increases. This preservation strategy ensures efficient resource
  allocation. And all models are guaranteed to be verified at least at the
  smallest scale.


#### :violet[*TODO Next*]

1. **RL/Tuning of design agents**: Design agents can use utility as a signal to
   improve designs over the evolution process.
2. **Online learning of scaling curves**: The selector can learn the scaling
   curve over time, improving exploration efficiency.

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
        st.write("**TODO:** Refer to viewer page for more details")

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
        st.write("**TODO:** Refer to viewer page for more details")

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


