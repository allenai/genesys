# gau.py   # DO NOT CHANGE OR REMOVE THE MAKK HERE, KEEP IT ALWAYS THE FIRST LINE #

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_discovery.model.utils.modules import GAUBase, gau_test # DO NOT CHANGE THIS IMPORT STATEMENT #


# YOU CAN IMPORT MORE MODULES HERE #

# YOU CAN DEFINE MORE CLASSES OR FUNCTIONS HERE #


class UnitName(GAUBase):
    """
    FILL IN THE DOCSTRING HERE, FOLLOWING THE EXAMPLE BELOW:

    This is an example of how you can write docstrings.
    You can add multiple lines of those descriptions. Make sure to include
    useful information about your method.

    **Code Example:**

    .. code-block:: python

        # Here is a Python code block
        def foo(lst):
            ret = []
            for x in lst:
                ret.append(x * 2)
            return ret

    And here is a verbatim-text diagram example:

    .. code-block:: text

        .------+---------------------------------.-----------------------------
        |            Block A (first)             |       Block B (second)
        +------+------+--------------------------+------+------+---------------
        | Next | Prev |   usable space           | Next | Prev | usable space..
        +------+------+--------------------------+------+--+---+---------------
        ^  |                                     ^         |
        |  '-------------------------------------'         |
        |                                                  |
        '----------- Block B's prev points to Block A -----'

    Todo:
        * This is a TODO item.
        * And a second TODO item.

    Args:
        alignment (c_size_t): Description of the `alignment` value.
        param (float): Description of `param1`.

    Returns:
        Description of the method's return value.

    Raises:
        AttributeError: If there is an error with the attributes.
        ValueError: If `param` is equal to 3.14.

    Example:
        This is how you can use this function

        >>> print("Code blocks are supported")

    Note:
        For more info on reStructuredText docstrings, see
        `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__
        and
        `here <https://peps.python.org/pep-0287/>`__.
    """
    def __init__(self, embed_dim: int, block_loc: tuple, kwarg_all: dict, device=None, dtype=None,**kwargs): # YOU CAN ADD MORE ARGUMENTS WITH OPTIONAL DEFAULT VALUES, BUT KEEP THE ORIGINAL ONES #
        self.factory_kwargs = {"device": device, "dtype": dtype} # DO NOT CHANGE THIS LINE, REMEMBER TO PASS IT #
        super().__init__(embed_dim, block_loc, kwarg_all) # DO NOT CHANGE THIS LINE #
        
        # COMPLETING THE CODE HERE #

        raise NotImplementedError


    # YOU CAN ADD MORE FUNCTIONS HERE #


    def _forward(self, X, **Z): 
        
        # THE CODE HERE MUST BE COMPLETED #

        raise NotImplementedError


# WRITE YOUR UNIT TEST FUNCTIONS HERE #

@gau_test # DO NOT CHANGE THIS DECORATOR, OTHERWISE IT WON'T BE RECOGNIZED AS A UNIT TEST #
def unit_test_name(device=None, dtype=None)->None: # RENAME THIS FUNCTION, DO NOT CHANGE THE ARGUMENTS, IT SHOULD ALSO NOT RETURN ANYTHING #
    # AN AVAILABLE DEVICE AND DTYPE ARE PASSED AS ARGUMENTS, USE THEM TO INITIALIZE YOUR GAU AND MOCK INPUTS #

    # WRITE ASSERTIONS TO PERFORM THE TEST, USE PRINT TO DEBUG #
    
    raise NotImplementedError # YOU MUST IMPLEMENT THIS FUNCTION #
