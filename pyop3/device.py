# File to handle op3.device context manager
from abc import ABCMeta, abstractmethod
import contextlib
import contextvars
import warnings

import numpy as np

class Device(metaclass=ABCMeta):
    _name: str
    _device_index: int | None

    def __init__(self, device_index: int | None = None):
        pass

    @property
    def name(self):
        return self._name

    @property
    def device_index(self):
        return self._device_index

    @abstractmethod
    def asarray(self, arr):
        pass

    @abstractmethod
    def zeros_like(self, arr):
        pass

    def __repr__(self):
        return self._name
        
    def __str__(self):
        return self._name

class CPU(Device):

    def __init__(self, device_index: int | None = None):
        super().__init__()
        self._name = "cpu"
        self._registered_arrays = set()
        self._device_index = device_index

    def asarray(self, arr):
        # NOTE: Better logic needed if we switch from just NumPy/CuPy
        if not isinstance(arr, np.ndarray):
            import cupy as cp
            return cp.asnumpy(arr)
    
        return np.array(arr)

    def zeros_like(self, arr):
        return np.zeros_like(arr)

class CUDAGPU(Device):
    
    def __init__(self, device_index: int | None = None):
        super().__init__()
        self._name = "CudaGPU"
        self._registered_arrays = set()
        self._token = None
        self._device_index = device_index

        try:
            import cupy as cp
            assert cp.is_available() 
        except:
            # TODO: Raise No GPU exception
            raise NotImplementedError 

    def asarray(self, arr):
        import cupy as cp
        return cp.asarray(arr)
    
    def zeros_like(self, arr):
        import cupy as cp
        return cp.zeros_like(arr)

@contextlib.contextmanager
def offloading(dev: Device):
    # TODO: Not Device exception
    if not isinstance(dev, Device):
        raise NotImplementedError

    token = _current_device.set(dev)
    try:
        yield
    finally:
        _current_device.reset(token)

# TODO: Should this const variable be here? 
HOST_DEVICE = CPU() 

# NOTE: Use contextvars to act as a bridge between buffer and manager
_current_device = contextvars.ContextVar("current_device", default=HOST_DEVICE)
