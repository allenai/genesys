import inspect
from .core.ttt import ttt
from .core.mamba2 import mamba2
from .core.rwkv6 import rwkv6
from .core.retnet import retnet
from .core.gpt2 import gpt2


MODEL2CODE = {
    "ttt": inspect.getsource(ttt),
    "mamba2": inspect.getsource(mamba2),
    "rwkv6": inspect.getsource(rwkv6),
    "retnet": inspect.getsource(retnet),
    'gpt2': inspect.getsource(gpt2)
}


