import os
import json
import functools as ft
import time
import re
import keyword
import textwrap

pjoin=os.path.join
psplit=os.path.split
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
    with open(file, encoding='utf-8') as f:
        return json.load(f)
    
def save_json(data,file,indent=4): 
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def read_file(file,lines=False):
    if not pexists(file):
        return None
    with open(file, encoding='utf-8') as f:
        if lines:
            return f.readlines()
        return f.read()

def write_file(file,data):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(data)

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