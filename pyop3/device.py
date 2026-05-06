# File to handle op3.device context manager
from abc import ABCMeta, abstractmethod
import contextlib
import contextvars
import warnings

import numpy as np

class Device(metaclass=ABCMeta):
    name: str

    def __init__(self):
        pass

    @abstractmethod
    def asarray(self, arr, *, constant=False):
        pass

    @abstractmethod
    def zeros_like(self, arr):
        pass

    def __repr__(self):
        return self.name
        
    def __str__(self):
        return self.name

class CPU(Device):
    name = "CPU"

    def __init__(self):
        super().__init__()

    def asarray(self, arr, *, constant=False):
        # NOTE: Better logic needed if we switch from just NumPy/CuPy
        output = arr
        if not isinstance(arr, np.ndarray):
            import cupy as cp
            output = cp.asnumpy(arr)
        else:
            output = np.array(output) 

        if constant:
            output.flags.writeable = False
        return output

    def zeros_like(self, arr):
        return np.zeros_like(arr)

class CUDAGPU(Device):
    name = "CudaGPU"
    
    def __init__(self):
        super().__init__()

        try:
            import cupy as cp
            assert cp.is_available() 
        except:
            # TODO: Raise No GPU exception
            raise NotImplementedError 

    def asarray(self, arr, *, constant=False):
        import cupy as cp
        return cp.asarray(arr)
    
    def zeros_like(self, arr):
        import cupy as cp
        return cp.zeros_like(arr)

HOST_DEVICE = CPU() 
_current_device = contextvars.ContextVar("current_device", default=HOST_DEVICE)

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

def on_host(func):

    def wrapper(*args, **kwargs):
        token = _current_device.set(HOST_DEVICE)
        try:
            return func(*args, **kwargs)
        finally:
            _current_device.reset(token)

    return wrapper 

def get_current_device():
    return _current_device.get()
