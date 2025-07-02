import abc
import contextlib
from cuda.core.experimental import Device as CuDevice, Stream
import cupy as cp
from cuda.bindings.driver import CUstream
#import cutlass.cute as cute
#import cutlass

class ComputeDevice(abc.ABC):
    pass

class CPUDevice(ComputeDevice):

    def context_manager(self):
        yield

class GPUDevice(ComputeDevice):

    def __init__(self, num_threads=32):
        self.num_threads=num_threads
        self.stream = cp.cuda.Stream(non_blocking=True)

    def context_manager(self):    
        with self.stream:
            self.stream.begin_capture()
            yield
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
        yield from compute_device.context_manager()
        compute_device = orig_device
    else:
        yield from compute_device.context_manager()
    
