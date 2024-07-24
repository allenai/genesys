import inspect
from .ttt import ttt
from .mamba2 import mamba2


MODEL2CODE = {
    "ttt": inspect.getsource(ttt),
    "mamba2": inspect.getsource(mamba2),
}


