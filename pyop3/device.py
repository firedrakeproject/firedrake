# File to handle op3.device context manager
from abc import ABCMeta, abstractmethod
import contextlib
import contextvars
import warnings

import cupy as cp

class Device(metaclass=ABCMeta):
    _name: str
    _registered_arrays: set 

    def __init__(self, device: int | None = None):
        pass

    @staticmethod
    def current():
        device = _current_device.get()
        return device
    
    @abstractmethod
    def sync_buffers(self):
        pass

    @abstractmethod
    def register(self, arr):
        pass

    @abstractmethod
    def _reset_register(self):
        pass

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self._name
        
    def __str__(self):
        return self._name

class CPU(Device):

    def __init__(self, device: int | None = None):
        super().__init__()
        self._name = "cpu"
        self._registered_arrays = set()
        self.device = None

    # NOTE: Is it necessary to have any implementation here? 
    # Maybe perfunctory implementations but no real purpose
    def sync_buffers(self):
        pass

    def register(self, arr):
        self._registered_arrays.add(arr)

    def _reset_register(self):
        self._registered_arrays = set()

class GPU(Device):
    
    def __init__(self, device: int | None = None):
        super().__init__()
        self._name = "gpu"
        self._registered_arrays = set()
        self._token = None
        self.device = cp.cuda.Device(device)

    def sync_buffers(self):
        for arr in self._registered_arrays:
            arr.maybe_sync_to_host()

    def register(self, arr):
        self._registered_arrays.add(arr)

    def _reset_register(self):
        self._registered_arrays = set()

@contextlib.contextmanager
def offloading(device: Device):
    # TODO: Not device exception
    if not isinstance(device, Device):
        raise NotImplementedError

    token = _current_device.set(device)
    try:
        yield
    finally:
        device.sync_buffers()
        device._reset_register()
        _current_device.reset(token)

# TODO: Should this const variable be here? 
HOST_DEVICE = CPU() 

# NOTE: Use contextvars to act as a bridge between buffer and manager class
_current_device = contextvars.ContextVar("current_device", default=HOST_DEVICE)
