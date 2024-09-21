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
# font-style="italic" 

def square_logo(upper_text, lower_text, color='000000'):
    svg_code = SQUARE_LOGO_SVG.format(UPPER_TEXT=upper_text, LOWER_TEXT=lower_text, COLOR=color)
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
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



            
          
  