import os
import json
import functools as ft
import time

pjoin=os.path.join
pexists=os.path.exists
mkdir=ft.partial(os.makedirs, exist_ok=True)

def pdebug(*value):
    print(f'[**Debug info**]',*value)

getmodelsize = lambda model: sum(p.numel() for p in model.parameters())
strmodelsize = lambda model: strscale(getmodelsize(model))

def strscale(size):
    if size>1e9: return f"{size/1e9:.2f}B"
    elif size>1e6: return f"{size/1e6:.2f}M"
    elif size>1e3: return f"{size/1e3:.2f}K"
    else: return f"{int(size)}"

def letternum2num(s):
    """
    Convert a letter-number string to a numeric value for sorting.
    'M' is treated as 1,000,000
    'B' is treated as 1,000,000,000
    """
    num = float(s[:-1])
    unit = s[-1]
    
    if unit == 'M':
        return num * 1_000_000
    elif unit == 'B':
        return num * 1_000_000_000
    else:
        raise ValueError("Unknown unit in string: " + s)
        
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


class CodeTimer:
    def __init__(self, label="code block"):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print(f"[Elapsed time for {self.label}: {self.elapsed:.3f} seconds]")
