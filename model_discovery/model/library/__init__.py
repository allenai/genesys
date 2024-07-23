import inspect
from .ttt import ttt


MODEL2CLASS = {
    "ttt": ttt.GAB,
}


MODEL2CODE = {
    "ttt": inspect.getsource(ttt),
}


