# from .gab import *
from .gab_random import *
import os

GAB_PATH = os.path.join(os.environ.get('CKPT_DIR'),'gab.py')
GAB_PATH = os.path.expanduser(GAB_PATH)

if not os.path.exists(GAB_PATH):
    with open(GAB_PATH,'w', encoding='utf-8') as f:
        f.write('')

with open(GAB_PATH) as f:
    code = compile(f.read(), GAB_PATH, 'exec')
    exec(code, globals())

