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


#### :orange[*TODO Next*]

1. **RL/Tuning of design agents**: Design agents can use utility as a signal to
   improve designs over the evolution process.
2. **Online learning of selector/scaling curves**: The selector can learn the scaling
   curve over time, improving exploration efficiency.
