''' Symbolic Representation and Operations of GAB '''

import inspect

from .block_registry import BlockRegister


class Symbolizer:
    def __init__(self, gab_name: str):
        gab_class,gab_config = BlockRegister.load_block(gab_name)
        print(gab_class)
        print(gab_config)

        gab=gab_class(10,1) # can we do some static analysis?

        source_code = inspect.getsource(gab_class)
        inf_code= inspect.getsource(gab._forward)

        print(inf_code)







if __name__ == '__main__':

    sym=Symbolizer('default')


    