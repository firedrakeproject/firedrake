# File to handle op3.device context manager
from abc import ABCMeta, abstractmethod
import contextlib
import warnings

# TODO: The constant should be elsewhere but temporary here 
HOST_DEVICE = 0 

class Device(metaclass=ABCMeta):
  _available: bool
  _name: str

  def __init__(self):
    pass

class GPU(Device):
  pass

@contextlib.contextmanager
def device(dev: Device): 
  # TODO: Update buffers and copy to
  try:
    yield "not implemented"
  finally:
    # TODO: Update buffers and copy back
    pass
    
