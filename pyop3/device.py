import abc
import contextlib
import contextvars
import ctypes
import dataclasses
import functools
import numbers
from typing import Any

import numpy as np
from mpi4py import MPI

from pyop3 import utils
from pyop3.dtypes import DTypeT

try:
    import cupy as cp
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
else:
    CUPY_AVAILABLE = True


# NOTE: Temporary fix to forcibly set the MLIR stream to match CuPy
# As the CuPy stream can sometimes clash with MLIR and cause bug
# Also required for multi-GPU
_STREAM_OVERRIDE = """
static void *stream;
void pyop3_set_stream(void *s) { stream = s; }
void *mgpuStreamCreate(void) { return stream; }
void mgpuStreamDestroy(void *s) { (void)s; }
"""


class Device(abc.ABC):
    """
    Device - Abstract class
    - Base for future GPU implementations
    - All device-specific logic should be kept in here
    """
    name: str

    arch: str | None = None
    """The architecture that code is compiled for, or `None` if the compiler selects it."""

    private_address_space: int | None = None
    """The address space of the temporaries of one thread, or `None` for the default address space."""

    @abc.abstractmethod
    def asarray(self, arr, *, constant=False):
        pass

    @abc.abstractmethod
    def zeros_like(self, arr):
        pass

    @abc.abstractmethod
    def empty(self, shape: int | tuple[int, ...], *, dtype: DTypeT | None = None):
        """Return a new array on this device whose values are not initialized.
        """

    @abc.abstractmethod
    def zeros(self, shape: int | tuple[int, ...], *, dtype: DTypeT | None = None):
        """Return a new array on this device that is filled with zeros.
        """

    @abc.abstractmethod
    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value: numbers.Number,
        *,
        dtype: DTypeT | None = None,
    ):
        """Return a new array on this device that is filled with fill_value.
        """

    @abc.abstractmethod
    def array(self, obj: Any, *, dtype: DTypeT | None = None):
        """Return a new array on this device that contains a copy of obj.
        """

    def __str__(self):
        return self.name


class CPU(Device):
    """
    CPU Class, designed to be host object, inheriting Device
    - Plausible to have multiple CPUs, functionally similar to having GPU
    """
    name = "CPU"

    # NOTE: We want to make sure we don't do a copy here
    def asarray(self, arr, *, constant=False):
        """ Convert GPU/CuPy/NumPy input array to CPU-compliant NumPy array """
        if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
            output = cp.asnumpy(arr)
        elif isinstance(arr, np.ndarray):
            output = arr
            if constant:
                output = flag_constant(output)
        elif isinstance(arr, list):
            output = np.array(arr) 
            if constant:
                output = flag_constant(arr)
        else:
            raise TypeError(f"{type(arr)} not supported.")

        return output

    def zeros_like(self, arr):
        return np.zeros_like(arr)

    def empty(self, shape, *, dtype=None):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, *, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def full(self, shape, fill_value, *, dtype=None):
        return np.full(shape, fill_value, dtype=dtype)

    def array(self, obj, *, dtype=None):
        return np.array(obj, dtype=dtype)


@dataclasses.dataclass(frozen=True)
class GPUTarget:
    """GPU limits
    
    i.e. blocks per thread and available static shared memory
    """

    shared_memory_per_block: int
    shared_memory_per_sm: int
    max_threads_per_block: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    warp_size: int


class GPU(Device):
    """A GPU that is driven through CuPy and an MLIR GPU runtime

    Instances that have the same `device_id` compare equal, so they share
    the device arrays of a buffer.

    The MLIR GPU runtime keeps the context of the first device that it uses
    for the lifetime of the process. Each process can therefore use only one
    GPU, and the constructor selects it before any kernel is launched.

    """

    runtime_library: str
    """The MLIR GPU runtime that the kernels link against, as a name for `-l`"""

    is_hip: bool
    """Whether CuPy must be built for ROCm (HIP) rather than CUDA"""

    _selected_device_id: int | None = None
    """The ordinal that this process uses, or None if no GPU is selected yet"""

    def __init__(self, device_id: int | None = None) -> None:
        try:
            assert self.cp.is_available()
        except:
            # TODO: Raise No GPU exception
            raise NotImplementedError
        if self.cp.cuda.runtime.is_hip != self.is_hip:
            platform = "ROCm" if self.is_hip else "CUDA"
            raise RuntimeError(f"{type(self).__name__} needs a CuPy that is built for {platform}")

        if device_id is None:
            device_id = _node_local_rank() % self.cp.cuda.runtime.getDeviceCount()

        if GPU._selected_device_id is None:
            self.cp.cuda.Device(device_id).use()
            ctypes.CDLL(f"lib{self.runtime_library}.so").mgpuSetDefaultDevice(ctypes.c_int32(device_id))

            # This method runs before all other GPU streams so MLIR and CuPy should
            # default here. Hacky way to force MLIR and CuPy to act on same stream
            from pyop3.cc import load

            override = ctypes.CDLL(load(_STREAM_OVERRIDE, "c", comm=MPI.COMM_SELF)._name, mode=ctypes.RTLD_GLOBAL)
            override.pyop3_set_stream(ctypes.c_void_p(self.cp.cuda.get_current_stream().ptr))
            GPU._selected_device_id = device_id
        elif device_id != GPU._selected_device_id:
            raise ValueError(
                f"This process already uses GPU {GPU._selected_device_id}, "
                f"it cannot also use GPU {device_id}"
            )
        self.device_id = device_id

        DEVICE_TO_ARRAY_TYPE[self] = cp.ndarray

    def __eq__(self, other: Any, /) -> bool:
        return type(other) is type(self) and other.device_id == self.device_id

    def __hash__(self) -> int:
        return hash((type(self), self.device_id))

    @property
    def cp(self):
        import cupy as cp
        return cp

    @functools.cached_property
    def target(self) -> GPUTarget:
        """The limits of the GPU."""
        attributes = self.cp.cuda.Device(self.device_id).attributes
        return GPUTarget(
            shared_memory_per_block=attributes["MaxSharedMemoryPerBlock"],
            shared_memory_per_sm=attributes["MaxSharedMemoryPerMultiprocessor"],
            max_threads_per_block=attributes["MaxThreadsPerBlock"],
            max_threads_per_sm=attributes["MaxThreadsPerMultiProcessor"],
            max_blocks_per_sm=attributes["MaxBlocksPerMultiprocessor"],
            warp_size=attributes["WarpSize"],
        )

    def asarray(self, arr, *, constant=False):
        return self.cp.asarray(arr)
    
    def zeros_like(self, arr):
        return self.cp.zeros_like(arr)

    def empty(self, shape, *, dtype=None):
        return self.cp.empty(shape, dtype=dtype)

    def zeros(self, shape, *, dtype=None):
        return self.cp.zeros(shape, dtype=dtype)

    def full(self, shape, fill_value, *, dtype=None):
        return self.cp.full(shape, fill_value, dtype=dtype)

    def array(self, obj, *, dtype=None):
        return self.cp.array(obj, dtype=dtype)


class CUDAGPU(GPU):
    """An NVIDIA GPU that is driven through CuPy and the MLIR CUDA runtime"""

    name = "CudaGPU"
    runtime_library = "mlir_cuda_runtime"
    is_hip = False

    @property
    def arch(self) -> str:
        """The name of the GPU architecture, for example ``sm_89``."""
        return f"sm_{self.cp.cuda.Device(self.device_id).compute_capability}"


class AMDGPU(GPU):
    """An AMD GPU that is driven through a ROCm build of CuPy and the MLIR ROCm runtime"""

    name = "AMDGPU"
    runtime_library = "mlir_rocm_runtime"
    is_hip = True
    # AMDGPU requires each alloca to be in the private address space. NVPTX
    # miscompiles allocas in its local address space, so CUDAGPU keeps the default.
    private_address_space = 5

    @property
    def arch(self) -> str:
        """The name of the GPU architecture without its target features"""
        name = self.cp.cuda.runtime.getDeviceProperties(self.device_id)["gcnArchName"]
        return name.decode().split(":")[0]


def _node_local_rank() -> int:
    """Return the rank of this process among the processes on its node.

    The first call is collective over `MPI.COMM_WORLD`.

    """
    node_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    rank = node_comm.rank
    node_comm.Free()
    return rank


"""Mapping between device and array type.

When new devices are created they should register themselves here.

Note that this mapping is not invertible. A single array type could map to
multiple devices.

"""
HOST_DEVICE = CPU()
DEVICE_TO_ARRAY_TYPE = {
    HOST_DEVICE: np.ndarray,
}



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


@functools.singledispatch
def flag_constant(array: Any) -> None:
    """Mark an array as being constant."""
    utils.raise_missing_dispatch_handler(array)


@flag_constant.register
def _(array: np.ndarray) -> None:
    array.flags.writeable = False


if CUPY_AVAILABLE:
    @flag_constant.register
    def _(array: cp.ndarray) -> None:
        # CuPy arrays don't support setting flags
        pass
