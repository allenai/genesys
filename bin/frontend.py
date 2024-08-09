import sys,os
import time
import pathlib
import functools as ft
import streamlit as st

st.set_page_config(page_title="Storm", layout="wide")


sys.path.append('.')
from model_discovery import BuildEvolution
import model_discovery.utils as U

from streamlit_navigation_bar import st_navbar

from bin.pages.home import home
from bin.pages.viewer import viewer
from bin.pages.design import design
from bin.pages.evolve import evolve
from bin.pages.engine import engine
from bin.pages.prompt import prompt


# Setup the evo system

@st.cache_resource()
def build_evo_system(name='test_evo_003'):
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
page = st_navbar(
    titles,
    logo_path=logo_path,
    styles=styles,
    urls=urls
)

pages['Home'] = home


pages[page](evosys,project_dir)

