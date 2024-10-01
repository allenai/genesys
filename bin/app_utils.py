import cairosvg
from PIL import Image
import io
import numpy as np
import uuid


SQUARE_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">
    <path d="M50,50 L250,50 L250,250 L50,250 Z" fill="none" stroke="#{COLOR}" stroke-width="4" />
    <text x="150" y="140" font-family="Arial, sans-serif" font-size="60" font-weight="bold" fill="#{COLOR}" text-anchor="middle">{UPPER_TEXT}</text>
    <text x="150" y="210" font-family="Arial, sans-serif" font-size="60" font-weight="bold" fill="#{COLOR}" text-anchor="middle">{LOWER_TEXT}</text>
</svg>
"""

SQUARE_CENTER_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">
    <path d="M50,50 L250,50 L250,250 L50,250 Z" fill="none" stroke="#{COLOR}" stroke-width="4" />
    <text x="150" y="165" font-family="Arial, sans-serif" font-size="60" font-weight="bold" fill="#{COLOR}" text-anchor="middle">{UPPER_TEXT}</text>
</svg>
"""
# font-style="italic" 

def square_logo(upper_text, lower_text=None, color='000000'):
    if lower_text is None:
        svg_code = SQUARE_CENTER_LOGO_SVG.format(UPPER_TEXT=upper_text, COLOR=color)
    else:
        svg_code = SQUARE_LOGO_SVG.format(UPPER_TEXT=upper_text, LOWER_TEXT=lower_text, COLOR=color)
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data))

def svg_to_image(svg_path):
    svg_data = open(svg_path, 'r').read()
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    return Image.open(io.BytesIO(png_data))


def grid_view(st,item_dict:dict,per_row=3,spacing=0.05):
    n_items=len(item_dict)
    n_rows = int(np.ceil(len(item_dict)/per_row))
    for i in range(n_rows):
        col_widths=np.ones(per_row*2-1)
        col_widths[1::2]=spacing
        cols=st.columns(col_widths)
        n_cols=min(per_row,n_items-i*per_row)
        for j in list(range(n_cols)):
          with cols[j*2]:
            index=i*per_row+j
            title,value=list(item_dict.items())[index]
            icon=value.pop('ICON',None)
            with st.expander(title,expanded=True,icon=icon):
              if 'BUTTON' in value:
                buttons=value.pop('BUTTON')
                st.write(value)
                if len(buttons)==1:
                  text,fn,disabled=buttons[0]
                  st.button(text,on_click=fn,disabled=disabled,key=str(uuid.uuid4()))
                else:
                  btn_cols=st.columns(max(4,len(buttons)))
                  for j in range(len(buttons)):
                    text,fn,disabled=buttons[j]
                    with btn_cols[j]:
                      st.button(text,on_click=fn,disabled=disabled,key=str(uuid.uuid4()),use_container_width=True)
              else:
                st.write(value)

def running_status(st,evosys):
  db_status = '📶' if evosys.ptree.remote_db else '📴'
  st.write(f'🏠 **Namespace:\n```{evosys.evoname}``` {db_status}**')
  if st.session_state.listening_mode:
    st.write(f'👂```{st.session_state.listener.node_id}``` *is listening*\n')
  else:
    if st.session_state.evo_running:
      st.write('🚀 ***Evolution System is running***')
    if evosys.CM is not None:
      with st.expander("👂 Connections",expanded=False):
        for conn in evosys.CM.get_active_connections():
          _running_designs, _running_verifies = evosys.CM.check_workload(conn)
          _max_designs = evosys.CM.max_design_threads_per_node
          st.write(f'```{conn}```: {len(_running_designs)}/{_max_designs} 🐎 {len(_running_verifies)}/1 🥏')
          
  running_verifications=[key for key,process in st.session_state.get('running_verifications',{}).items() if process.poll() is None]
  if len(running_verifications)!=0:
    with st.expander("🥏 Running Verifies",expanded=False):
      for idx,key in enumerate(running_verifications):
        st.write(f'```{key}```')
  running_designs=[key for key,process in st.session_state.get('design_threads',{}).items() if process.poll() is None]
  if len(running_designs)!=0:
    with st.expander("🐎 Running Designs",expanded=False):
      for idx,key in enumerate(running_designs):
        st.write(f'```{key}```')
  
  if not st.session_state.evo_running:
    if len(running_verifications)==0 and len(running_designs)==0:
      st.write('')
      st.write(':gray[*No workloads running*] 💤')
  

  st.divider()
    
            
          
  