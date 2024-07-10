import os
import json
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
        
def load_json(file,default={}):
    if not pexists(file):
        return default
    with open(file) as f:
        return json.load(f)
    
def save_json(data,file): 
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def get_last_checkpoint(output_dir: str):
    """Gets the last checkpoint 

    :param output_dir: 
        The output directory containing the last checkpoint
    """
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [pjoin(output_dir, d) for d in os.listdir(output_dir) 
                   if pexists(pjoin(output_dir, d, "pytorch_model.bin")) and d.startswith("checkpoint")]
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return checkpoints[-1] # dir of the last checkpoint