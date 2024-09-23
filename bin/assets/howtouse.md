## How to use

1. Configure the experiment settings in the **:blue[Config]** tab. 
- Or selecting an existing namespace to resume the experiment.
- Or create a new namespace to start a new experiment.
- Or directly use the default "text_evo_000" namespace (if you just want to try).

2. Run the evolution loop in the **:blue[Evolve]** tab. 
- Or play with the design engine in the **:blue[Design]** tab.
- Or play with the knowledge base in the **:blue[Search]** tab.
- Or tunning the selector in the **:blue[Select]** tab.
- Or train a design in the verification **:blue[Engine]** tab.

3. View the system states and results in the **:blue[Viewer]** tab.

## Components

1. **Select**: The selector sample seed node(s) from the tree for the next round of evolution.
2. **Design**: The designer will sample a new design based on the seed(s).
3. **Search**: The designer agent can search the knowledge base during the design process.
4. **Engine**: The verification engine can be used to train a **chosen design** (not necessarily the new design and not necessarily take turns with the design step) in a given scale and evaluate the performance.
5. **Evolve**: The evolution loop will repeat the above processes. 

