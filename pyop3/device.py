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
    _accessed_buffer: list

    def __init__(self):
        self._available = True
        self._accessed_buffer = []
        pass

    @abstractmethod
    def __repr__(self):
        pass 

    @abstractmethod
    def __str__(self):
        pass

class GPU(Device):
    
    def __init__(self):
        super().__init__()
        self._name = "gpu"
    pass

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

@contextlib.contextmanager
def offloading(dev: Device): 
    token = _current_device.set(dev)
    try:
        yield
    finally:
        _current_device.reset(token)
        # TODO: Update buffers and copy back
        # NOTE: Does this have to happen lazily if we do not have a record of accessed buffers?
