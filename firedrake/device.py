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
        self.kernel_args = []

    
    def write_file(self):
        
        with open("./temp_kernel_minimal.py",'w') as file:
            file.write("import cupy as cp\n")
            for kernel in self.kernel_string:
                file.write(kernel+ "\n")
                file.write("\n")
            
            file.write("def __main__():")
            file.write("\t pass")
            # cell loop needed here
            for i, kernel in enumerate(self.kernel_string):
                for arg in self.kernel_args[i]: 
                    # get arg data 
                    pass
    

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
    
def add_kernel_string(k_str, args):
    global compute_device
    assert isinstance(compute_device, GPUDevice)        
    compute_device.kernel_string += [k_str.replace("cupy_kernel", f"cupy_kernel{len(compute_device.kernel_string)}")] 
    compute_device.kernel_args += [tuple(args)]
