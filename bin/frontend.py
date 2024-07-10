import json
import time
import pathlib
import streamlit as st
import sys

sys.path.append('.')
from model_discovery import BuildSystem

@st.cache_resource()
def build_system():
    system = BuildSystem(
        cache_type="diskcache",
        temperature=0.1,
    )
    return system

def main(argv):
    
    ### build the system 
    system = build_system()

    st.title("Model discovery engine")

    ### side bar 
    st.sidebar.button("reset design query")
        
    filler = "Find me a new model"

    instruction = st.text_input(label = "Add any additional instructions (optional)" )
    submit = st.button(label="Design model")

    
    if submit or instruction:
        instruction = str(None) if not instruction else instruction 
        
        with st.spinner(text="running discovery loop"):
            system(instruction,frontend=True,stream=st)
        
    
if __name__ == "__main__":
    main(sys.argv[1:]) 
