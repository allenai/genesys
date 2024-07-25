import inspect
from .ttt import ttt
from .mamba2 import mamba2
from .rwkv6 import rwkv6
from .retnet import retnet


MODEL2CODE = {
    "ttt": inspect.getsource(ttt),
    "mamba2": inspect.getsource(mamba2),
    "rwkv6": inspect.getsource(rwkv6),
    "retnet": inspect.getsource(retnet),
}


