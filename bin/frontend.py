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


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the parent module first
import bin.pages

# Function to dynamically import and reload modules
def import_and_reload(module_name):
    full_module_name = f'bin.pages.{module_name}'
    if full_module_name in sys.modules:
        return importlib.reload(sys.modules[full_module_name])
    return importlib.import_module(full_module_name)

# Import and reload modules
home = import_and_reload('home').home
viewer = import_and_reload('viewer').viewer
design = import_and_reload('design').design
evolve = import_and_reload('evolve').evolve
engine = import_and_reload('engine').engine
prompt = import_and_reload('prompt').prompt
search = import_and_reload('search').search



# from bin.pages import home,viewer,design,evolve,engine,prompt,search





# Setup the evo system

@st.cache_resource()
def build_evo_system(name='test_evo_000'):
    strparams=[
        f"evoname={name}",
        "scales=14M,31M,70M",
        "selection_ratio=0.25",
        "select_method=random",
        "design_budget=0",
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
    'Search': search,
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

pages[pg](evosys,project_dir)
