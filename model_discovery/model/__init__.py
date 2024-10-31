# from .gab import *
from .gab_random import *

import os
import importlib

# import * from GAB_PATH
GAB_PATH = os.environ.get('GAB_PATH')

gab_module = importlib.import_module(GAB_PATH)
globals().update(vars(gab_module))
