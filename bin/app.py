import sys,os
sys.path.append('.')

import time
import pathlib
import functools as ft
import streamlit as st
import importlib
import multiprocessing
import model_discovery.utils as U
from PIL import Image
import bin.app_utils as AU



custom_args = sys.argv[1:]

DEPLOY_MODE = 'deploy' in custom_args or '--deploy' in custom_args or '-d' in custom_args


current_dir = pathlib.Path(__file__).parent
logo_path = U.pjoin(current_dir,'assets','storm_logo.svg')

logo=AU.svg_to_image(logo_path)
st.set_page_config(page_title="μLM", layout="wide",page_icon=logo)


from model_discovery import BuildEvolution

from streamlit_navigation_bar import st_navbar


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the parent module first
if not DEPLOY_MODE:
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
    verify = import_and_reload('verify').verify
    config = import_and_reload('config').config
    search = import_and_reload('search').search
    select = import_and_reload('select').select
    listen = import_and_reload('listen').listen
    tester = import_and_reload('tester').tester

else:
    from bin.pages import home,viewer,design,evolve,verify,config,search,select,listen





# Setup the evo system

@st.cache_resource()
def build_evo_system(name='test_evo_000'):
    params={
        'evoname':name,
        'scales':'14M,31M,70M',
        'selection_ratio':0.25,
        'select_method':'random',
    }
    evo_system = BuildEvolution(
        params=params,
        do_cache=False,
        stream=st,
        # cache_type='diskcache',
    )
    return evo_system

evosys = build_evo_system()


# Setup the streamlit session state

if 'listening_mode' not in st.session_state:
    st.session_state.listening_mode = False

if 'evo_running' not in st.session_state:
    st.session_state.evo_running = False

if 'design_threads' not in st.session_state:
    st.session_state['design_threads'] = {}  

if 'max_design_threads' not in st.session_state:
    st.session_state['max_design_threads'] = 5

if 'running_verifications' not in st.session_state:
    st.session_state['running_verifications'] = {}

if 'listener_connections' not in st.session_state:
    st.session_state['listener_connections'] = {}


# Setup the streamlit pages


project_dir = current_dir.parent

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
    'Evolve': evolve,
    'Design': design,
    'Verify': verify,
    'Search': search,
    'Select': select,
    'Viewer': viewer,
    'Config': config,
    'Listen': listen,
}
if not DEPLOY_MODE:
    pages['Tester'] = tester

titles=list(pages.keys())+['GitHub']
pg = st_navbar(
    titles,
    logo_path=U.pjoin(current_dir,'assets','storm.svg'),
    styles=styles,
    urls=urls
)
pages['Home'] = home

pages[pg](evosys,project_dir)
