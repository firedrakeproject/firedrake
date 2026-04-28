# File to handle op3.device context manager
from abc import ABCMeta, abstractmethod
import contextlib
import contextvars
import warnings

# TODO: The constant should be elsewhere but temporary here 
HOST_DEVICE = "cpu" 

# NOTE: Use contextvars to act as a bridge between buffer and manager classes
_current_device = contextvars.ContextVar("current_device", default=HOST_DEVICE)

class Device(metaclass=ABCMeta):
  _available: bool
  _name: str

  def __init__(self):
    self._available = True
    pass

  @abstractmethod
  def __repr__(self):
    pass 

class GPU(Device):
  
  def __init__(self):
    super().__init__()
    self._name = "gpu"
  pass

  def __repr__(self):
    return self._name

@contextlib.contextmanager
def offloading(dev: Device): 
  # TODO: Update buffers and copy to
  token = _current_device.set(dev)
  try:
    yield
  finally:
    # TODO: Update buffers and copy back
    _current_device.reset(token)
    pass
    
