import inspect
from .obj.ttt import ttt
from .obj.mamba2 import mamba2
from .obj.rwkv6 import rwkv6
from .obj.retnet import retnet


MODEL2CODE = {
    "ttt": inspect.getsource(ttt),
    "mamba2": inspect.getsource(mamba2),
    "rwkv6": inspect.getsource(rwkv6),
    "retnet": inspect.getsource(retnet),
}


