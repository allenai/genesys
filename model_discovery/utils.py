import os
import functools as ft

pjoin=os.path.join
pexists=os.path.exists
mkdir=ft.partial(os.makedirs, exist_ok=True)

def strmodelsize(model):
    size = sum(p.numel() for p in model.parameters())
    return strscale(size)

def strscale(size):
    if size>1e9: return f"{size/1e9:.2f}B"
    elif size>1e6: return f"{size/1e6:.2f}M"
    elif size>1e3: return f"{size/1e3:.2f}K"
    else: return f"{int(size)}"
