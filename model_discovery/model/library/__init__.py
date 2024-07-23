import inspect
from .TTT import ttt


MODEL2CLASS = {
    "ttt": ttt.GAB,
}


MODEL2CODE = {
    "ttt": inspect.getsource(ttt),
}


