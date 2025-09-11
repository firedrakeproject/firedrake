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
        if isinstance(arr, cp.ndarray):
            return arr.get()
        return np.array(arr)

    def context_manager(self):
        yield self

class GPUDevice(ComputeDevice):

    identity = "gpu"
    array_type = cp.ndarray
    
    def __init__(self, blocks = [("cell", 1)]):
        self.identity = "gpu"
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.kernel_string = []
        self.kernel_args = []
        self.file_name = "temp_kernel_device"
        self.kernel_type = None
        self.blocks = {0:[], 1:[]}
        self.block_names = {}
        for block_type, dim in blocks:
            self.block_names[block_type] = "BLOCK_SIZE_" + block_type[0].capitalize()
            if 2**int(np.log2(dim)) != dim:
                raise ValueError(f"Block dim must be a power of two, not {dim}")
            if block_type == "cell":
                block_def = (block_type, "BLOCK_SIZE_C", dim, None)
                self.blocks[0] += [block_def]
                self.blocks[1] += [block_def]
            elif block_type == "quad":
                self.blocks[0] += [(block_type,"BLOCK_SIZE_Q", dim, None)]
            elif block_type == "basis":
                self.blocks[1] += [(block_type, "BLOCK_SIZE_B", dim, None)]
            else:   
                raise NotImplementedError(f"Other block types {block_type} not yet supported")
                
    def add_info(self, block_name, data):
        for key in self.blocks.keys():
            for i in range(len(self.blocks[key])):
                if self.blocks[key][i][0] == block_name:
                    self.blocks[key][i] = (self.blocks[key][i][0], self.blocks[key][i][1], data)

    def block_dims(self):
        return_dict = {}
        for key in self.blocks.keys():
            return_dict[key] = [val[0] for val in self.blocks[key]]
        return return_dict

    def array(self, arr):
        return cp.array(arr)

    def triton_slicing(self):
        return """
# Ops for slicing (take/put) local tensor (extension to https://github.com/triton-lang/triton/pull/2715)
# extension found https://github.com/triton-lang/triton/issues/656
@triton.jit
def _indicator_(n_dims: tl.constexpr, idx: tl.constexpr, pos: tl.constexpr, pos_dim: tl.constexpr):
    tl.static_assert(idx < n_dims)
    tl.static_assert(pos < pos_dim)
    y = tl.arange(0, pos_dim)
    y = tl.where(y==pos, 1, 0)
    
    for n in tl.static_range(0, n_dims):
        if n != n_dims - 1 - idx:
            y = tl.expand_dims(y, n)
    return y

@triton.jit
def _take_slice_(x, n_dims: tl.constexpr, idx: tl.constexpr, pos: tl.constexpr, pos_dim:tl.constexpr, keep_dim: tl.constexpr = True):
    ind = _indicator_(n_dims, idx, pos, pos_dim)
    y = tl.sum(x * ind, n_dims - 1 - idx)
    if keep_dim:
        y = tl.expand_dims(y, n_dims - 1 - idx)

    return y

@triton.jit
def _put_slice_(x, n_dims: tl.constexpr, idx: tl.constexpr, pos: tl.constexpr, pos_dim:tl.constexpr, input_slice):
    ind = _indicator_(n_dims, idx, pos, pos_dim)
    y = tl.where(ind==1, input_slice, x)
    return y

"""    

    def write_file(self, arrays, maps):
        
        with open(f"./{self.file_name}.py",'w') as file:
            file.write("import cupy as cp\n")
            file.write("import cupyx as cpx\n")
            if self.kernel_type == "triton":
                file.write("import torch\n")
                file.write("import numpy as np\n")
                file.write("import triton\n")
                file.write("import triton.language as tl\n")
                file.write("from triton.language.extra import libdevice\n")
                file.write("import pdb\n")
                file.write("DEVICE=triton.runtime.driver.active.get_active_torch_device()\n")
                file.write(self.triton_slicing())
            for kernel in self.kernel_string:
                file.write(kernel+ "\n")
                file.write("\n")
            
                
            num_cells = None 
            file.write("def gpu_parloop():\n")
            for array, map_i, i in zip(arrays, maps, [i for i in range(len(arrays)+2)]):
                print(i)
                a = repr(array).replace("object", "cp.float64")
                m = repr(map_i).replace("int32", "cp.int32")
                file.write(f"\ta{i} = cp.{a}\n")
                file.write(f"\tm{i} = cp.{m}\n")
                if num_cells is None:
                    num_cells = len(map_i)
                else:
                    assert num_cells == len(map_i)

            if self.kernel_type =="triton":
                for name, array in self.kernel_data["arrays"]:
                    if array is not None:
                        file.write(f"\t{name} = torch.from_numpy(np.{repr(array)}).float().to(DEVICE)\n")
                
            # hate
            #real_kernel_args = list(list(zip(*(self.kernel_data["sizes_pow2"] + self.kernel_data["sizes_actual"] + self.kernel_data["strides"])))[1])
            #real_kernel_args = [str(int(i)) for i in real_kernel_args]
            # cell loop needed here
            indent = 1
            index = ""
            grid = ""
            if self.kernel_type != "triton":
                file.write(f"\tfor i in range({num_cells}):\n")
                index = "[i]"
                indent += 1
            else:
                file.write(f"\tgrid = lambda meta: (triton.cdiv({num_cells}, meta['BLOCK_SIZE_C']), )\n")
                grid = "[grid]"
            for j, kernel in enumerate(self.kernel_string):
                for k, arg in enumerate(self.kernel_args[j]): 
                    # get arg data
                    if arg == "coords": 
                        file.write(indent * "\t" + f"a_g{k} = cp.take(a{k}, m{k}{index}, axis=0)\n")
                    elif arg == "A":
                        file.write(indent * "\t" + f"a_g{k} = cp.zeros_like(m{k}{index}, dtype=cp.float64)\n")
                    else:
                        file.write(indent * "\t" + f"a_g{k} = cp.take(a{k}, m{k}{index}, axis=0)\n")
                    if self.kernel_type == "triton":
                        file.write(indent * "\t" + f"a_g{k} = torch.from_numpy(a_g{k}.get()).float().to(DEVICE)\n") 
                gathered_args = [f"a_g{j}" for j in range(len(self.kernel_args[j]))] + [name for name, val in self.kernel_data["arrays"] if val is not None]
                block_sizes = [f"size_{block[0][-1]}={block[2] if block[2] is not None else 2}" for block in self.blocks['vars'] + self.blocks['temps']]
                arg_str = ",".join(gathered_args + block_sizes + [f"{block[0]}={block[1]}" for block in self.blocks['vars'] + self.blocks['temps']])
                #file.write(indent * "\t" + "breakpoint()\n")
                file.write(indent * "\t" + f"{self.kernel_type}_kernel{j}({arg_str})\n")
                for k, arg in enumerate(self.kernel_args[j]): 
                    # get arg data
                    if arg == "A": 
                        file.write(indent * "\t" + f"cpx.scatter_add(a{k}, m{k}{index}, a_g{k})\n")
                        #file.write("\tbreakpoint()\n")
                        file.write(f"\tprint(a{k})\n")

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
def device(type="cpu", blocks=[("cell", 1)]):
    global compute_device
    if type=="gpu":
        gpu = GPUDevice(blocks)
        orig_device = compute_device
        compute_device = gpu
        os.environ["FIREDRAKE_USE_GPU"] = "1" 
        yield from compute_device.context_manager()
        compute_device = orig_device
        del os.environ['FIREDRAKE_USE_GPU']
    elif type=="cpu":
        cpu = CPUDevice()
        orig_device = compute_device
        compute_device = cpu
        yield from compute_device.context_manager()
        compute_device = orig_device
    else:
       raise NotImplementedError(f"Device identity {type} unrecognised") 
    
def add_kernel_string(k_str, args, k_type):
    global compute_device
    assert isinstance(compute_device, GPUDevice)        
    compute_device.kernel_string += [k_str.replace(f"{k_type}_kernel", f"{k_type}_kernel{len(compute_device.kernel_string)}")] 
    if k_type=="triton":
        compute_device.kernel_data = args
        compute_device.kernel_args += [tuple([a[0] for a in args["arrays"] if a[1] is None])]
    compute_device.kernel_args += [tuple(args)]
    compute_device.kernel_type = k_type
