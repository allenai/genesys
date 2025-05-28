import os
import collections 
import typing
import argparse

from typing import Union

__all__ = [
    "PathOrStr",
]

PathOrStr = typing.Union[str,os.PathLike]
ConfigType = Union[argparse.Namespace,collections.namedtuple,typing.NamedTuple]
