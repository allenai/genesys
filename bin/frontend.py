import sys,os
import time
import pathlib
import functools as ft
import streamlit as st
import importlib

st.set_page_config(page_title="AlphaGPT", layout="wide")


sys.path.append('.')
from model_discovery import BuildEvolution
import model_discovery.utils as U

from streamlit_navigation_bar import st_navbar

# Import full modules
import bin.pages.home
import bin.pages.viewer
import bin.pages.design
import bin.pages.evolve
import bin.pages.engine
import bin.pages.prompt

# importlib.reload(bin.pages.home)
# importlib.reload(bin.pages.viewer)
# importlib.reload(bin.pages.evolve)
# importlib.reload(bin.pages.design)
# importlib.reload(bin.pages.engine)
# importlib.reload(bin.pages.prompt)

home = bin.pages.home.home
viewer = bin.pages.viewer.viewer
design = bin.pages.design.design
evolve = bin.pages.evolve.evolve
engine = bin.pages.engine.engine
prompt = bin.pages.prompt.prompt



# Setup the evo system

@st.cache_resource()
def build_evo_system(name='test_evo_004'):
    strparams=[
        f"evoname={name}",
        "scales=14M,31M,70M",
        "selection_ratio=0.25",
        "select_method=random",
    ]
    evo_system = BuildEvolution(
        strparams=';'.join(strparams),
        do_cache=False,
        # cache_type='diskcache',
    )
    return evo_system

evosys = build_evo_system()


# Setup the streamlit pages

current_dir = pathlib.Path(__file__).parent
project_dir = current_dir.parent
logo_path = U.pjoin(current_dir,'assets','storm.svg')

styles = {
    "nav": {
        # "background-color": "royalblue",
        # "justify-content": "left",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        # "color": "white",
        "padding": "14px",
    },
    # "active": {
    #     "background-color": "white",
    #     "color": "var(--text-color)",
    #     "font-weight": "normal",
    #     "padding": "14px",
    # }
}

urls = {"GitHub": "https://github.com/allenai/model_discovery"}

pages = {
    "Viewer": viewer,
    'Evolve': evolve,
    'Design': design,
    'Engine': engine,
    'Prompt': prompt,
}
titles=list(pages.keys())+['GitHub']
pg = st_navbar(
    titles,
    logo_path=logo_path,
    styles=styles,
    urls=urls
)
pages['Home'] = home

if pg == 'Home':
    importlib.reload(bin.pages.home)
elif pg == 'Viewer':
    importlib.reload(bin.pages.viewer)
elif pg == 'Evolve':
    importlib.reload(bin.pages.evolve)
elif pg == 'Design':
    importlib.reload(bin.pages.design)
elif pg == 'Engine':
    importlib.reload(bin.pages.engine)
elif pg == 'Prompt':
    importlib.reload(bin.pages.prompt)

pages[pg](evosys,project_dir)
