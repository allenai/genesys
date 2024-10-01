import json
import time
import pathlib
import streamlit as st
import sys,os
from datetime import datetime
import asyncio
import python_weather


sys.path.append('.')
import model_discovery.utils as U
import bin.app_utils as AU

current_dir = pathlib.Path(__file__).parent
logo_path = U.pjoin(current_dir,'..','assets','storm_logo.svg')

logo=AU.svg_to_image(logo_path)


async def _getweather(city):
  # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
  async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
    weather = await client.get(city)
    return weather.temperature
  
def home(evosys,project_dir):

  
    with st.sidebar:
      # st.write("*Welcome to the Model Discovery System!*")
      # AU.running_status(st,evosys)
      # st.image(logo,use_column_width=True)
      st.image('https://images.unsplash.com/photo-1722691694088-b3b2ab29be31?q=80&w=2535&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')
      city = 'Seattle' 
      temprature = asyncio.run(_getweather(city))
      st.write(f"*Today is {datetime.now().strftime('%b %d, %Y, %a')}. The temperature in {city} right now is :blue[{temprature}]°F.*")


    # col1,col2=st.columns([1.45,1])
    # with col1: 
    st.markdown('# Welcome to the Model Discovery System')
    # with col2:
    #   st.write('')
    #   st.image(logo,width=80)
    
    st.markdown('''
The Model Discovery System is a system aimed at using large language model
agents to discover novel and human-level autoregressive language model designs.
This is a graphical user interface (GUI) for the Model Discovery System. You can
use it to run experiments, view results, and tune the agents. We use this GUI as
the main interface to the system instead of the common command-line interface
(CLI).

* ***Why not a CLI?*** CLI is *insufficient* to provide necessary
  **observability** and **manageability** with such a complicated system. In a
  CLI, it is hard to manage the experiment states (e.g., launching multiple
  design threads, running verifications on multiple nodes), and there is no way
  to observe the agent behaviours and design artifacts (e.g., long markdown or
  python outputs from multiple agents).

* ***Can I use a CLI instead?*** Yes, you can always use the CLI to run the
  system. The GUI essentially integrated experiment monitors and runners to
  create subprocess to run the low-level CLI instructions. 
''')

    st.markdown('''
## :rainbow[How to launch the evolution?] 

Only distributed evolution is supported now. As sequantial evolution is complicated which need to consider the action policy. See details below. To launch the distributed evolution:
 1. Configure the experiment settings in the **Config** tab. Save and upload to the cloud. (*CLI mode is working in progress*)
 2. Launch the nodes in the **Listen** tab. Or directly launch by `bash scripts/run_node.sh`. Remember to run it in GPU-available sessions. You can also launch a node in the master.
 3. Run the evolution in the **Evolve** tab. Or directly launch by `bash scripts/run_evo.sh`. 

**NOTE:** do not run multiple nodes in the same user space as it will cause the file access conflict. Just use all the GPUs in each node.
''')

    # welcome = U.read_file(U.pjoin(project_dir,'bin','assets','howtouse.md'))
    # st.markdown(welcome)

    col1,col2=st.columns(2)
    with col1:
      st.markdown('''
## How to use this GUI?

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
- Or check the distributed design base in the Firestore.
''')

    with col2:
      st.markdown('''
## Components

1. **Select**: The selector sample seed node(s) from the tree for the next round of evolution.
2. **Design**: The designer will sample a new design based on the seed(s).
3. **Search**: The designer agent can search the knowledge base during the design process.
4. **Verify**: The verification engine can be used to train a **chosen design** 
(not necessarily the new design and not necessarily take turns with the design step) 
in a given scale and evaluate the performance by the customed LM-Eval.
5. **Evolve**: The evolution loop will repeat the above processes asynchronously. 

## Icon Guideline

 * **Namespace**: The experiment name, 📶 is online with remote DB connected, 📴 means offline.
 * **Listening**: When the system is running in the listening mode or connected to listeners as master node, the status will show with 👂, 🐎 means running designs 🥏 means running verifies. 
''')


    st.markdown('''
## Distributed Evolution

The evolutionary system continously run two threads asynchronously in multiple nodes until the budget is exhausted:
1. **Design Threads**: Continously sample new designs on selected nodes. It is driven by the *Model Design Engine* in the **Design** tab.
2. **Verify Threads**: Continously run verifications on the selected design and scale. It is driven by the *Verification Engine* in the **Verify** tab.

The two threads can be runned in multiple nodes besides the master node.
The network is managed in the **Firestore**. To add a node to the network,
simply run `python -m model_discovery.listen` or `bash script/run_node.sh`
on the node. Or you can also run it in `Listen` tab in the GUI, so that you
can view the design process and verification results in real-time in the 
***Design***, ***Verify*** and ***Viewer*** tabs.
You can also run the system with CLI, then you can always check the status in GUI (on the same machine) afterwards
by input the same node id in the **Listen** tab. 


''')

    # st.balloons()


    ################################################


    st.subheader('Tabs')

    tabs=st.tabs(['Evolve','Design','Verify','Search','Select','Viewer','Config','Listen'])

    with tabs[0]:
      st.markdown('''
The Evolve tab is the main interface for the Model Discovery System. It allows
you to run the full model discovery evolution loop. Before you start, make sure
to configure the experiment settings in the Config tab. By default, the system
will always run under the "text_evo_000" **namespace**. You can create new
namespaces to save your experiments.
''')

    with tabs[1]:
      st.markdown('''
Sampler for sampling single design based on the seed.
''')

    with tabs[2]:
      st.markdown('''
The verification engine for the training a design under a scale.
''')

    with tabs[3]:
      st.markdown('''
The knowledge base for the agents to use in RAG.
''')

    with tabs[4]:
      st.markdown('''
Select the next seeds for the next round of evolution. Place for tunning and
testing the selector.
''')

    with tabs[5]:
      st.markdown('''
Multiple views of the system states and results.
''')

    with tabs[6]:
      st.markdown('''
Configure the experiment settings. Recommended to set it up before running.
''')  

    with tabs[7]:
      st.markdown('''
Listening mode, accepting commands from master node. Can also run it in the CLI using `python -m model_discovery.listen` or `bash script/run_node.sh`.
''')  
