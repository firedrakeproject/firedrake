import abc
import contextlib
from cuda.core.experimental import Device as CuDevice, Stream
import cupy as cp
from cuda.bindings.driver import CUstream
#import cutlass.cute as cute
#import cutlass
import os
class ComputeDevice(abc.ABC):

    pass

class CPUDevice(ComputeDevice):

    def context_manager(self):
        yield self

class GPUDevice(ComputeDevice):

    def __init__(self, num_threads=32):
        self.num_threads=num_threads
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.kernel_string = []

    def context_manager(self):    
        with self.stream:
            self.stream.begin_capture()
            yield self
            g = self.stream.end_capture()
        g.launch(self.stream)
        self.stream.synchronize()

compute_device = CPUDevice()

@contextlib.contextmanager
def device(type=None):
    if type=="gpu":
        gpu = GPUDevice()
        global compute_device
        orig_device = compute_device
        compute_device = gpu
        os.environ["FIREDRAKE_USE_GPU"] = "1" 
        yield from compute_device.context_manager()
        compute_device = orig_device
        del os.environ['FIREDRAKE_USE_GPU']
    else:
        yield from compute_device.context_manager()
    
def add_kernel_string(k_str):
    global compute_device
    assert isinstance(compute_device, GPUDevice)        
    compute_device.kernel_string += [k_str]
