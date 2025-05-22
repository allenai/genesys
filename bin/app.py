import sys,os
sys.path.append('.')

try: # a stupid patch 
    from model_discovery.secrets import *
except:
    pass
    
import time
import pathlib
import functools as ft
import streamlit as st
import importlib
import multiprocessing
import model_discovery.utils as U
from PIL import Image
from streamlit_theme import st_theme

import bin.app_utils as AU



custom_args = sys.argv[1:]

DEPLOY_MODE = 'deploy' in custom_args or '--deploy' in custom_args or '-d' in custom_args or 'd' in custom_args
DEMO_MODE = 'demo' in custom_args or '--demo' in custom_args or '-m' in custom_args or 'm' in custom_args

DEMO_MODE = True

# if DEMO_MODE:
#     DEPLOY_MODE = True

current_dir = pathlib.Path(__file__).parent
logo_path = U.pjoin(current_dir,'assets','storm_logo.svg')

logo=AU.svg_to_image(logo_path)
st.set_page_config(page_title="Genesys", layout="wide",page_icon=logo)




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
    launch_listener = import_and_reload('listen').launch_listener
    export_leaderboards = import_and_reload('viewer').export_leaderboards
else:
    from bin.pages import home,viewer,design,evolve,verify,config,search,select,listen
    from bin.pages.listen import launch_listener
    from bin.pages.viewer import export_leaderboards



if 'use_cache' not in st.session_state:
    st.session_state.use_cache = False # as it needs update online sometimes

st.session_state.daily_usage_limit = 5

# Setup the evo system

if 'system_built' not in st.session_state:
    st.session_state.system_built = False


@st.cache_resource()
def build_evo_system(name):
    params={
        'evoname':name,
    }
    t0 = time.time()
    evo_system = BuildEvolution(
        params=params,
        do_cache=False,
        stream=st,
        demo_mode=DEMO_MODE,
        # cache_type='diskcache',
    )
    if DEMO_MODE:
        print('Setting up for demo mode')
        evo_system.set_demo_mode()
        # dump the data so no need to load from firebase WIP
        # import json
        # def load_results(evosys,fname):
        #     ckptdir=os.environ['CKPT_DIR']
        #     dir=os.path.join(ckptdir,'RESULTS')
        #     with open(os.path.join(dir,f'{fname}_results.json'),'r') as f:
        #         all_results = json.load(f)
        #     print(f'{fname}: {len(all_results)}/{len(evosys.ptree.G.nodes)} loaded')
        #     for d in all_results:
        #         if d not in evosys.ptree.G.nodes:
        #             print(f'{d} not in evosys.ptree.G.nodes')
        #             continue
        #         for scale in evosys.ptree.G.nodes[d]['data'].verifications:
        #             results = all_results[d][scale]
        #             evosys.ptree.G.nodes[d]['data'].verifications[scale].verification_report['eval_results.json']['results'] = results
        #     return evosys
        # evo_system = load_results(evo_system,'full_aug')
        st.session_state.use_cache = True

    t1 = time.time()
    print(f'Time taken to setup evo system: {t1-t0:.2f} seconds')

    if st.session_state.use_cache:
        st.session_state.pre_filter = []
        st.session_state.filter_by_types = {}
        sources = ['ReferenceCoreWithTree', 'DesignArtifactImplemented', 'DesignArtifact', 'ReferenceCore', 'ReferenceWithCode', 'Reference']
        for source in sources:
            st.session_state.filter_by_types[source] = evo_system.ptree.filter_by_type(source,ret_data=True)
        st.session_state.implemented_designs = st.session_state.filter_by_types['DesignArtifactImplemented']
        st.session_state.unimplemented_designs = st.session_state.filter_by_types['DesignArtifact']
        st.session_state.design_artifacts = {**st.session_state.implemented_designs,**st.session_state.unimplemented_designs}
        st.session_state.acronyms = st.session_state.filter_by_types['ReferenceCoreWithTree']
        st.session_state.design_vectors = evo_system.ptree.get_design_vectors(pre_filter=[])
        st.session_state.baseline_vectors = evo_system.ptree.get_baseline_vectors()
        st.session_state.custom_vectors = evo_system.ptree.get_custom_vectors('CUSTOM_HOLD')
        st.session_state.design_vectors.update(st.session_state.custom_vectors)
        st.session_state.leaderboards_normed,st.session_state.leaderboards_unnormed_h,st.session_state.leaderboards_unnormed_l,st.session_state.baselines=export_leaderboards(evo_system,st.session_state.design_vectors,st.session_state.baseline_vectors)
    t2 = time.time()
    print(f'Time taken to pre-load design artifacts and leaderboard: {t2-t1:.2f} seconds')
    st.session_state.system_built = True
    return evo_system



setting=AU.get_setting()
default_namespace=setting.get('default_namespace','test_evo_000')
if DEMO_MODE:
    default_namespace = 'evo_exp_full_a'

if not st.session_state.system_built:
    st.toast('Welcome to Genesys! System is preparing, please wait...',icon='👋')
    st.toast(f'Start building Genesys and loading data...',icon='🔥')

evosys = build_evo_system(default_namespace)


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


st.session_state.is_deploy = DEPLOY_MODE
st.session_state.is_demo = DEMO_MODE
st.session_state.current_theme = st_theme()


# Initialize session state
if 'listener' not in st.session_state:
    st.session_state.listener = None
if 'listener_thread' not in st.session_state:
    st.session_state.listener_thread = None
if 'exec_commands' not in st.session_state:
    st.session_state.exec_commands = {}

def _check_local_listener(st):
    _node_id = AU._listener_running()
    _local_doc = U.read_local_doc() 
    if _node_id:
        _group_id = _local_doc['group_id']
        _max_design_threads = _local_doc['max_design_threads']
        _accept_verify_job = _local_doc['accept_verify_job']
        _cpu_only_checker = _local_doc.get('cpu_only_checker',False)
        if not st.session_state.listening_mode:
            st.toast(f'Local running listener detected. Node ID: {_node_id}. Group ID: {_group_id}. Launching a listener in passive mode.')
            with st.spinner('Launching listener...'):
                launch_listener(evosys, _node_id, _group_id, _max_design_threads, _accept_verify_job,_cpu_only_checker)

_check_local_listener(st)

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
    'Select': select,
    'Search': search,
    'Viewer': viewer,
    'Config': config,
}
if not (DEPLOY_MODE or DEMO_MODE):
    pages['Listen'] = listen
    pages['Tester'] = tester



titles=list(pages.keys())+['GitHub']

_logo=AU.theme_aware_options(st,"storm.svg","storm_logo.svg","storm_logo.svg")

pg = st_navbar(
    titles,
    logo_path=U.pjoin(current_dir,'assets',_logo),
    styles=styles,
    urls=urls
)
pages['Home'] = home

pages[pg](evosys,project_dir)
