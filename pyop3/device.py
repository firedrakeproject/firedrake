# File to handle op3.device context manager
from abc import ABCMeta, abstractmethod
import contextlib
import contextvars
import warnings

import numpy as np

class Device(metaclass=ABCMeta):
    """
    Device - Abstract class
    - Base for future GPU implementations
    - All device-specific logic should be kept in here
    """
    name: str

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
    """
    CPU Class, designed to be host object, inheriting Device
    - Plausible to have multiple CPUs, functionally similar to having GPU
    """
    name = "CPU"

    def asarray(self, arr, *, constant=False):
        """ Convert GPU/CuPy/NumPy input array to CPU-compliant NumPy array """
        try:
            import cupy as cp
        except ImportError:
            cp = None
        
        if cp and isinstance(arr, cp.ndarray):
            output = cp.asnumpy(arr)
        elif isinstance(arr, np.ndarray):
            output = np.array(arr)
            if constant:
                output.flags.writeable = False
        else:
            raise TypeError(f"{type(arr)} not supported.")

        return output


    def zeros_like(self, arr):
        return np.zeros_like(arr)

class CUDAGPU(Device):
    """ 
    GPU class for Nvidia GPUs. inheriting Device.
    - All offloading will be done through CuPy
    - Multiple instantiations will be independent of each other 
    """
    name = "CudaGPU"
    
    def __init__(self):
        try:
            assert self.cp.is_available() 
        except:
            # TODO: Raise No GPU exception
            raise NotImplementedError 

    @property
    def cp(self):
        import cupy as cp
        return cp

    def asarray(self, arr, *, constant=False):
        return self.cp.asarray(arr)
    
    def zeros_like(self, arr):
        return self.cp.zeros_like(arr)

HOST_DEVICE = CPU() 

""" 
    Global context variable for determining device context
    - This should not be imported to other modules - value accessed through getter
    - All modification should be controlled via the offloading function.
"""
_current_device = contextvars.ContextVar("current_device", default=HOST_DEVICE)

@contextlib.contextmanager
def offloading(dev: Device):
    """ 
    Context Manager for offloading components to select device
    This function should be the only way to modfiy the current device variable

    Updates current context to the given `dev` variable.
    Former device is stored in stack, to be restored when finished
        - This also allows for stacking of context windows

    Context variables are also async safe.

    --- Example:
    gpu = op3.CUDAGPU()
    with op3.offloading(gpu):
        g.dat.assign(23, eager=True, eager_strategy="array")
    """

    # TODO: Not Device exception
    if not isinstance(dev, Device):
        raise NotImplementedError

    token = _current_device.set(dev)
    try:
        yield
    finally:
        _current_device.reset(token)

def on_host(func):
    """
    Decorator for components that we want to stay on host device
    i.e. MPI communications/StarForest
    """
    def wrapper(*args, **kwargs):
        with offloading(HOST_DEVICE):
            return func(*args, **kwargs)

    return wrapper 

def get_current_device():
    return _current_device.get()
