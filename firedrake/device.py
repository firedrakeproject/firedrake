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


@contextlib.contextmanager
def device(type=None):
    print(type)
    if type=="gpu":
        gpu = GPUDevice()
        yield from gpu.context_manager()
    else:
        cpu = CPUDevice()
        yield from cpu.context_manager()
        #stream = cp.cuda.Stream(non_blocking=True)
        #with stream:
        #    stream.begin_capture()
        #    yield
        #    g =stream.end_capture()
        #g.launch(stream)
        #stream.synchronize()
    
