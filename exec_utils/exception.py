from .base import UtilBase

__all__ = [
    "ExecUtilException",
    "ExecConfigError",
]

class ExecUtilException(Exception):
    """Base class for all exceptions
    """

class ExecConfigError(ExecUtilException):
    pass

