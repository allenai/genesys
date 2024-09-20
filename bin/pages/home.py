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
    
    readme = U.read_file(U.pjoin(project_dir,'README.md'))
    st.markdown(readme)

    # st.balloons()

    with st.sidebar:
        st.write("*Welcome to the Model Discovery System!*")
       
        st.image('https://images.unsplash.com/photo-1722691694088-b3b2ab29be31?q=80&w=2535&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')
        city = 'Seattle'
        temprature = asyncio.run(_getweather(city))
        st.write(f"*Today is {datetime.now().strftime('%b %d, %Y, %a')}. The temperature in {city} right now is :blue[{temprature}]°F.*")


