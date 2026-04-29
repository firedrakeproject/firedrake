# File to handle op3.device context manager
from abc import ABCMeta, abstractmethod
import contextlib
import contextvars
import warnings

import cupy as cp


# TODO: The constant should be elsewhere but temporary here 
HOST_DEVICE = "cpu"

# NOTE: Use contextvars to act as a bridge between buffer and manager class
_current_device = contextvars.ContextVar("current_device", default=HOST_DEVICE)

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
    def register(self, arr):
        pass

    @abstractmethod
    def __repr__(self):
        pass 

    @abstractmethod
    def __str__(self):
        pass

# TODO: Necessary to make this?
class CPU(Device):
    pass

# Implementation follows similar idea to CuPY (with GPU(): ...)
class GPU(Device):
    
    def __init__(self, device: int | None = None):
        super().__init__()
        self._name = "gpu"
        self._registered_arrays = set()
        self._token = None
        self.device = cp.cuda.Device(device)

    def __enter__(self):
        self._token = _current_device.set(self)
        return self

    def __exit__(self, type, value, traceback):
        self.sync_buffers()
        _current_device.reset(self._token)
        self._reset_register()

    def sync_buffers(self):
        for arr in self._registered_arrays:
            arr.maybe_sync_to_host()

    def register(self, arr):
        self._registered_arrays.add(arr)

    def _reset_register(self):
        self._registered_arrays = set()

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name

