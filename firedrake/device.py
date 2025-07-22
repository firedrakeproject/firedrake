import abc
import contextlib
from cuda.core.experimental import Device as CuDevice, Stream
import numpy as np
import cupy as cp
from cuda.bindings.driver import CUstream
#import cutlass.cute as cute
#import cutlass
import os

class ComputeDevice(abc.ABC):
    pass

class CPUDevice(ComputeDevice):
    identity = "cpu"
    array_type = np.ndarray

    def array(self, arr):
        return np.array(arr)

    def context_manager(self):
        yield self

class GPUDevice(ComputeDevice):

    identity = "gpu"
    array_type = cp.ndarray
    
    def __init__(self, num_threads=32):
        self.identity = "gpu"
        self.num_threads=num_threads
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.kernel_string = []
        self.kernel_args = []
        self.file_name = "temp_kernel_device"

    def array(self, arr):
        print("GPU array")
        return cp.array(arr)

    
    def write_file(self, arrays, maps):
        
        with open(f"./{self.file_name}.py",'w') as file:
            file.write("import cupy as cp\n")
            file.write("import cupyx as cpx\n")
            for kernel in self.kernel_string:
                file.write(kernel+ "\n")
                file.write("\n")
            
                
            num_cells = None 
            file.write("def gpu_parloop():\n")
            for array, map, i in zip(arrays, maps, [i for i in range(len(arrays)+2)]):
                print(i)
                file.write(f"\ta{i} = cp.{repr(array.astype(object)).replace("object", "cp.float64")}\n")
                file.write(f"\tm{i} = cp.{repr(map.astype(object)).replace("object", "cp.int32")}\n")
                if num_cells is None:
                    num_cells = len(map)
                else:
                    assert num_cells == len(map)
            # cell loop needed here
            file.write(f"\tfor i in range({num_cells}):\n")
            for j, kernel in enumerate(self.kernel_string):
                for k, arg in enumerate(self.kernel_args[j]): 
                    # get arg data
                    if arg == "coords": 
                        file.write(f"\t\ta_g{k} = cp.take(a{k}, m{k}[i], axis=0).flatten()\n")
                    elif arg == "A":
                        file.write(f"\t\ta_g{k} = cp.zeros_like(m{k}[i], dtype=cp.float64)\n")
                    else:
                        file.write(f"\t\ta_g{k} = cp.take(a{k}, m{k}[i], axis=0)\n")
                arg_str = ",".join([f"a_g{j}" for j in range(len(self.kernel_args[j]))])
                file.write(f"\t\tcupy_kernel{j}({arg_str})\n")
                for k, arg in enumerate(self.kernel_args[j]): 
                    # get arg data
                    if arg == "A": 
                        file.write(f"\t\tcpx.scatter_add(a{k}, m{k}[i], a_g{k})\n")
                file.write(f"\tprint(a{k})")

    def context_manager(self):    
        yield self
        #with self.stream:
        #    self.stream.begin_capture()
        #    yield self
        #    g = self.stream.end_capture()
        #g.launch(self.stream)
        #self.stream.synchronize()

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
