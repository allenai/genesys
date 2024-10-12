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
  

def tabs():
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
Listening mode, accepting commands from master node. Can also run it in the CLI using `genesys listen [args]`.
''')  



def howtouse():


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

* ***Why not just CLI?*** CLI is *insufficient* to provide necessary
  **observability** and **manageability** with such a complicated system. In a
  CLI, it is hard to manage the experiment states (e.g., launching multiple
  design threads, running verifications on multiple nodes), and there is no way
  to observe the agent behaviors and design artifacts (e.g., long markdown or
  python outputs from multiple agents).

* ***Can I use a CLI instead?*** Yes, you can always use the CLI to run the
  system. The GUI essentially integrated experiment monitors and runners to
  create a subprocess to run the low-level CLI instructions. You can install the
  CLI by `pip install -e .` and run it by `genesys <command> [args]`. See details below.
''')

    title_color=AU.theme_aware_options(st,"rainbow","violet","violet")
      
    st.markdown(f'''
## :{title_color}[How to launch the evolution?] 

The evolution is distributed, asynchronous and parallel (see details below). To set it up:

1. Configure the experiment settings:
   - Use the **Config** tab in the UI to set up your experiment. Save and upload to the cloud.
   - Alternatively, you can manually edit the experiment folders.
   - For command-line configuration, run:
     ```
     genesys cfg [-u] [-d] [-h]
     ```
     - `-u, --upload`: Upload all local configs to the remote DB
     - `-d, --download`: Download all configs from the remote DB
   - **Always remember to upload the local configs, otherwise it will be overwritten by remote DB.**

2. Launch nodes:
   ```
   genesys node [-i <node_id>] [-g <group_id>] [-m <max_design_threads>] [-n] [-h]
   ```
   - `-i, --node_id`: Optional. If not specified, a random node id will be assigned.
   - `-g, --group_id`: Default is 'default'. Set it if you need to run multiple experiments simultaneously.
   - `-m, --max_design_threads`: Default is 5. Maximum number of design threads on this node.
   - `-n, --no_gpus`: If specified, the node will not accept verification jobs.

   You can also use the **Listen** tab in the UI to manage nodes.

3. Run the evolution:
   - Use the **Evolve** tab in the UI to start and manage the evolution.
   - Or launch directly from the command line:
     ```
     genesys evo [-e <evoname>] [-g <group_id>] [-r <design_to_verify_ratio>] [-h]
     ```
     - `-e, --evoname`: Name of the evolution (default: 'test_evo_000', you can change it in `Config` tab, or directly edit `CKPT_DIR/.setting.json: default_namespace`)
     - `-g, --group_id`: Default is 'default'. Should match the one used for the nodes.
     - `-r, --design_to_verify_ratio`: Ratio of design threads to verification nodes (default: 4).

**NOTE:** Avoid running multiple nodes in the same user space to prevent file access conflicts. Instead, utilize all available GPUs in each node.
''')

    # welcome = U.read_file(U.pjoin(project_dir,'bin','assets','howtouse.md'))
    # st.markdown(welcome)

    st.markdown('''
### Tips and Checklists for running the evolution

- [ ] Configure the experiment settings in the **Config** tab before running.
  
  - [ ] Use the tools in **Verify - Budget Tools** tab to have a very rough
    estimate of the design and verification costs in order to set the budget
    properly.

  - [ ] Tune your configs in each playgrounds (**Design**, **Verify**,
    **Search**, **Select**) before running large-scale evolution.

  - [ ] Upload the config to the remote DB, so that the nodes will be able to
    access it.

- [ ] Launch the nodes with the same group ID (keep default is fine) by `genesys
  node`. View the node status in the **Listen** tab.

- [ ] Run the evolution by `genesys evo` or use the **Evolve** tab in the GUI.

''')

    col1,col2=st.columns(2)
    with col1:
      st.markdown('''
## How can I use this GUI?

1. Configure the experiment settings in the **:blue[Config]** tab. 
- Or selecting an existing namespace to resume the experiment.
- Or create a new namespace to start a new experiment.
- Or directly use the default "text_evo_000" namespace (if you just want to try).

2. Run the evolution loop in the **:blue[Evolve]** tab. (See details above)
- Or play with the design engine in the **:blue[Design]** tab.
- Or play with the knowledge base in the **:blue[Search]** tab.
- Or tunning the selector in the **:blue[Select]** tab.
- Or train a design in the verification **:blue[Engine]** tab.

3. View the system states and results in the **:blue[Viewer]** tab. 
- Or check the remote design base in the Firestore.
''')

    with col2:
      st.markdown('''
## Core Components

1. **Select**: The selector sample seed node(s) from the tree for the next round of evolution.
2. **Design**: The designer will sample a new design based on the seed(s).
3. **Search**: The designer agent can search the knowledge base during the design process.
4. **Verify**: The verification engine can be used to train a **chosen design** 
(not necessarily the new design and not necessarily taking turns with the design step) 
on a given scale and evaluate the performance using the customed LM-Eval.
5. **Evolve**: The evolution loop will repeat the above processes asynchronously. 

## Icon Guideline

 * **Namespace**: The experiment name, 📶 is online with remote DB connected, 📴 means offline.
 * **Listening**: To view your running node status, just launch the GUI and go to the **Listen** tab.
''')


    st.markdown('''
## Evolution Process

The evolutionary system continuously runs two threads asynchronously
orchestrated by the selector in multiple nodes until the budget is exhausted: 
 1. **Design Threads**: Sample new designs on selected nodes. It is driven by
    the *Model Design Engine* in the **Design** tab. You can also run it in CLI
    by `genesys design [args]`. 

 2. **Verify Threads**: Run verifications on the selected design and scale. It
    is driven by the *Verification Engine* in the **Verify** tab. You can also
    run it in CLI by `genesys verify [args]`.

The network of worker nodes are orchestrated by a master node through the
**Firebase**. To add a node to the network, check run node above. It is
recommended to run it in GPU-available sessions. The machine that runs the
master node can also run a worker node at the same time.
''')


    with st.expander('Notes about Selector design and the Evolution Process',icon='🎼'):
      with open('bin/assets/selector_notes.md','r') as f:
        st.markdown(f.read())
  


def home(evosys,project_dir):
  
    with st.sidebar:
      # st.write("*Welcome to the Model Discovery System!*")
      # AU.running_status(st,evosys)
      # st.image(logo,use_column_width=True)
      # img_path='https://plus.unsplash.com/premium_photo-1676035055997-8a0b479d6e7e?q=80&w=2264&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
      img_path=U.pjoin(current_dir,'..','assets','gene.PNG')
      st.image(img_path)
      # city = 'Seattle' 
      # try:
      #   temprature = asyncio.run(_getweather(city)) # unreliable sometimes
      # except Exception as e:
      #   temprature = 'N/A'
      # st.caption(f":blue[{datetime.now().strftime('%b %d, %Y, %A')}]. {city} temperature :blue[{temprature} °F].")

      st.caption(f"Today is :blue[{datetime.now().strftime('%b %d, %Y, %A')}].")


    howtouse()
    tabs()
