import os
import json
import functools as ft
import time
import re
import keyword
import textwrap
import torch
import yaml
import zipfile
from contextlib import contextmanager

pjoin=os.path.join
psplit=os.path.split
pexists=os.path.exists
mkdir=ft.partial(os.makedirs, exist_ok=True)

def pdebug(*value):
    print(f'[**Debug info**]',*value)

getmodelsize = lambda model: sum(p.numel() for p in model.parameters())
strmodelsize = lambda model: strscale(getmodelsize(model))

def strscale(size):
    if size>1e12: return f"{size/1e12:.2f}T"
    elif size>1e9: return f"{size/1e9:.2f}B"
    elif size>1e6: return f"{size/1e6:.2f}M"
    elif size>1e3: return f"{size/1e3:.2f}K"
    else: return f"{int(size)}"


def dict_eq(dict1,dict2):
    equal=True
    for k,v in dict1.items():
        if k not in dict2:
            equal=False
        else:
            if isinstance(v,dict) and isinstance(dict2[k],dict):
                equal=dict_eq(v,dict2[k])
            elif v!=dict2[k]:
                equal=False
    return equal

def letternum2num(s):
    """
    Convert a letter-number string to a numeric value for sorting.
    'M' is treated as 1,000,000
    'B' is treated as 1,000,000,000
    """
    if isinstance(s,float):
        return s
    if isinstance(s,int):
        return s    
    if s.isdigit():
        return int(s)
    num = float(s[:-1])
    unit = s[-1]
    
    if unit=='K':
        return num * 1_000
    elif unit == 'M':
        return num * 1_000_000
    elif unit == 'B':
        return num * 1_000_000_000
    elif unit == 'T':
        return num * 1_000_000_000_000
    else:
        raise ValueError("Unknown unit in string: " + s)
        
def load_json(file,default={}):
    if not pexists(file):
        return default
    with open(file, encoding='utf-8') as f:
        return json.load(f)
    
def save_json(data,file,indent=4): 
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def safe_save_json(data,file,indent=4,max_try=3):
    # save, and read back to check if it is error and same as original
    with file_lock(f'{file}.save_lock'):
        try:
            _safe_save_json(data,file,indent,max_try,check_eq=True)
        except Exception as e:
            pdebug(f"Failed to save json to {file} after {max_try} tries: {e}")
        _safe_save_json(data,file,indent,max_try,check_eq=False) # try with no check


def _safe_save_json(data,file,indent=4,max_try=3,check_eq=True):
    # save, and read back to check if it is error and same as original
    for _ in range(max_try):
        save_json(data,file,indent)
        try:
            data_new=load_json(file)
            if not check_eq:
                return
            if dict_eq(data_new,data): 
                return
            else:
                raise Exception(f"Saved json is not the same as original: {data_new} and {data}")
        except Exception as e:
            pdebug(f"Failed to save json to {file} after {_} tries: {e}")
    raise Exception(f"Failed to save json to {file} after {max_try} tries")

def acquire_lock(name,tts=20):
    lock_dir=pjoin(os.environ.get('CKPT_DIR','.lock'))
    mkdir(lock_dir)
    lock_file=pjoin(lock_dir,f'{name}.lock')
    if pexists(lock_file):
        lock_time,tts=read_file(lock_file).split('\n')
        if time.time()-float(lock_time)<=float(tts):
            return False
    with open(lock_file,'w') as f:
        f.write(f'{time.time()}\n{tts}')
    return True

def release_lock(name):
    lock_dir=pjoin(os.environ.get('CKPT_DIR','.lock'))
    lock_file=pjoin(lock_dir,f'{name}.lock')
    if pexists(lock_file):
        os.remove(lock_file)

def read_file(file,lines=False):
    if not pexists(file):
        return None
    with open(file, encoding='utf-8') as f:
        if lines:
            return f.readlines()
        return f.read()

def load_zip_file(zip_file_path):
    if not pexists(zip_file_path):
        return None
    with open(zip_file_path, "rb") as file:
        zip_data = file.read()
    return zip_data
    
def write_file(file,data):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(data)

def append_file(file,data):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(data)

def load_yaml(file): # load yaml file as a dictionary
    if not pexists(file):
        return {}
    with open(file, encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


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


def remove_leading_indent(source_code):
    # Split source code into lines
    lines = source_code.splitlines()

    # Find the first line that starts with "def", "class", or another relevant keyword
    indent_level = None
    for line in lines:
        stripped_line = line.lstrip()
        if not indent_level:
            indent_level = len(line) - len(stripped_line) # use first line indent by default
        if stripped_line.startswith(("def ", "class ")):
            # Calculate the number of leading spaces (indentation level)
            indent_level = len(line) - len(stripped_line)
            break

    # Remove the leading indent based on the detected indent_level
    normalized_lines = [line[indent_level:] if len(line) > indent_level else line for line in lines]

    return '\n'.join(normalized_lines)

def replace_from_second(text, old, new):
    if len(text.split(old)) == 1:
        return text
    first_part, remaining = text.split(old, 1)
    remaining = remaining.replace(old, new)
    return first_part + old + remaining

def to_camel_case_gab_class_name(name):
    # Replace non-word characters with spaces to isolate words
    words = re.sub(r'\W|^(?=\d)', ' ', name).split()
    
    # Capitalize the first letter of each word and join them together
    camel_case_name = ''.join(word.capitalize() for word in words)
    
    # Ensure the variable name doesn't start with a digit
    if camel_case_name and camel_case_name[0].isdigit():
        camel_case_name = 'GAB' + camel_case_name
    
    # Check if the name is a Python keyword
    if keyword.iskeyword(camel_case_name):
        camel_case_name += 'GAB'
    
    # Fallback if the name is empty
    if not camel_case_name:
        raise ValueError("The name is empty after converting to camel case")
    
    return camel_case_name

def add_line_num(code):  # line i: code
    lines = code.split('\n')
    return '\n'.join([f'line {i+1}: {line}' for i, line in enumerate(lines)])

def get_factory_kwargs(cpu_only=False):
    if cpu_only:
        device='cpu'
        dtype=torch.float16
    else:
        if torch.cuda.is_available():
            device='cuda'
            dtype=torch.bfloat16
        else:
            device='cpu'
            dtype=torch.float16
    return {"device": device, "dtype": dtype}


def safe_get_cfg_dict(cfg,key,default):
    _dict=cfg.get(key,default) 
    _cfg_dict={}
    for k,v in default.items():
        _cfg_dict[k]=_dict.get(k,v)
    return _cfg_dict

def init_dict(cfg,default):
    for k,v in default.items():
        if k not in cfg:
            cfg[k]=v
    return cfg

def zip_folder(folder,zip_file):
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder))


def translate_dict_keys(data,key_dict,allow_missing=True):
    new_data={}
    for k,v in data.items():
        if isinstance(v,dict):
            v=translate_dict_keys(v,key_dict,allow_missing)
        k=key_dict.get(k,k) if allow_missing else key_dict[k]
        new_data[k]=v
    return new_data


def parse_verify_id(verify_id):
    scale = verify_id.split('_')[-1]
    design_id = verify_id[:-len(scale)-1]
    return design_id, scale

def get_local_doc():
    CKPT_DIR=os.environ.get('CKPT_DIR',None)
    if CKPT_DIR is None:
        return None
    return load_json(pjoin(CKPT_DIR,'.node.json'))

def save_local_doc(data):
    CKPT_DIR=os.environ.get('CKPT_DIR',None)
    if CKPT_DIR is None:
        return
    save_json(data,pjoin(CKPT_DIR,'.node.json'))


def break_sentence(text, max_length=100):
    # insert \n every max_length
    return '\n'.join([text[i:i+max_length] for i in range(0, len(text), max_length)])


def sort_dict_by_scale(dict,ascending=True):
    sorted_keys = sorted(dict.keys(), key=lambda x: letternum2num(x),reverse=not ascending)
    return {k: dict[k] for k in sorted_keys}


def acquire_local_lock(tts=20):
    while not acquire_lock('.node',tts):
        time.sleep(1)
    return True

def release_local_lock():
    release_lock('.node')



@contextmanager
def file_lock(file,tts=20):
    try:
        acquire_lock(file,tts)
        yield
    finally:
        release_lock(file)

@contextmanager
def local_lock(tts=20):
    try:
        acquire_local_lock(tts)
        yield
    finally:
        release_local_lock()

def read_local_doc(file='.node'):
    CKPT_DIR = os.environ.get("CKPT_DIR")
    local_doc_path = f"{CKPT_DIR}/{file}.json"
    local_doc = load_json(local_doc_path)
    return local_doc

def write_local_doc(local_doc,file='.node'):
    CKPT_DIR = os.environ.get("CKPT_DIR")
    local_doc_path = f"{CKPT_DIR}/{file}.json"
    safe_save_json(local_doc,local_doc_path)

def log_error_model(design_id,scale):
    local_doc = read_local_doc()
    local_doc['error_models'] = local_doc.get('error_models',{})
    local_doc['error_models'][design_id] = scale
    write_local_doc(local_doc)

def log_slow_model(design_id,time_elapsed,time_lower):
    local_doc = read_local_doc()
    if 'too_slow' not in local_doc:
        local_doc['too_slow'] = {}
    local_doc['too_slow'][f'{design_id}'] = (time_elapsed,time_lower)
    write_local_doc(local_doc)
    
def check_error_model(design_id):
    local_doc = read_local_doc()
    return design_id in local_doc.get('error_models',{})

