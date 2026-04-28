# File to handle op3.device context manager
import abc
import contextlib
import warnings

# TODO: Consider if there is approach to actually identify host method calling the python script
# For now, we just have a constant argument.
HOST_DEVICE = "cpu" 

class Device(abc.ABC):
  pass
