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


async def _getweather(city):
  # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
  async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
    weather = await client.get(city)
    return weather.temperature
  
def home(evosys,project_dir):

    st.markdown('''
# Welcome to the Model Discovery System

The Model Discovery System is a system aimed at using large language model agents to discover novel and human-level autoregressive language model designs. 
This is a frontend for the Model Discovery System.


''')
    
    # welcome = U.read_file(U.pjoin(project_dir,'bin','assets','howtouse.md'))
    # st.markdown(welcome)

    col1,col2=st.columns(2)
    with col1:
      st.markdown('''
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
''')

    with col2:
      st.markdown('''
## Components

1. **Select**: The selector sample seed node(s) from the tree for the next round of evolution.
2. **Design**: The designer will sample a new design based on the seed(s).
3. **Search**: The designer agent can search the knowledge base during the design process.
4. **Engine**: The verification engine can be used to train a **chosen design** (not necessarily the new design and not necessarily take turns with the design step) in a given scale and evaluate the performance.
5. **Evolve**: The evolution loop will repeat the above processes. 

## UI Info

 * **Namespace**: The experiment name, 📶 means connected to the remote DB, 📴 means local only.
 * **Running Sessions**: Design and scale that is being verified and running design sessions.

''')



    # st.balloons()


    ################################################


    st.subheader('Tabs')

    tabs=st.tabs(['Evolve','Design','Search','Engine','Select','Viewer','Config'])

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


    with st.sidebar:
        # st.write("*Welcome to the Model Discovery System!*")
       
        st.image('https://images.unsplash.com/photo-1722691694088-b3b2ab29be31?q=80&w=2535&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')
        city = 'Seattle' 
        temprature = asyncio.run(_getweather(city))
        st.write(f"*Today is {datetime.now().strftime('%b %d, %Y, %a')}. The temperature in {city} right now is :blue[{temprature}]°F.*")
