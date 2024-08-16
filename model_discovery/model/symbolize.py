''' Symbolic Representation and Operations of GAB '''

import inspect

from .block_registry import BlockRegister

from .library import *

from torch.fx import symbolic_trace

class Symbolizer:
    def __init__(self, gab_name: str):
        gab_class,gab_config = BlockRegister.load_block(gab_name)
        # print(gab_class)
        # print(gab_config)

        gab=gab_class(128,1) # can we do some static analysis?

        symbolic_traced=symbolic_trace(gab)

        # source_code = inspect.getsource(gab_class)
        # inf_code= inspect.getsource(gab._forward)

        print(symbolic_traced.code)
        print(symbolic_traced.graph)







if __name__ == '__main__':

    block=ttt

    sym=Symbolizer('mamba_simple')


    