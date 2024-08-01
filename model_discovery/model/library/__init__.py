import inspect
from .base.ttt import ttt
from .base.mamba2 import mamba2
from .base.rwkv6 import rwkv6
from .base.retnet import retnet
from .base.gpt2 import gpt2


MODEL2CODE = {
    "ttt": inspect.getsource(ttt),
    "mamba2": inspect.getsource(mamba2),
    "rwkv6": inspect.getsource(rwkv6),
    "retnet": inspect.getsource(retnet),
    'gpt2': inspect.getsource(gpt2)
}


