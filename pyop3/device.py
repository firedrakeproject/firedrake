import abc
import contextlib
import warnings


class OffloadingDevice(abc.ABC):
    pass


class CPUDevice(OffloadingDevice):
    pass


class GPUDevice(OffloadingDevice, abc.ABC):
    def __init__(self, num_threads=32):
        self.num_threads = num_threads


class CUDADevice(GPUDevice):
    pass


class OpenCLDevice(GPUDevice):
    pass


host_device = CUDADevice()
offloading_device = host_device


@contextlib.contextmanager
def offloading(device: OffloadingDevice):
    global offloading_device

    orig_offloading_device = offloading_device
    if isinstance(orig_offloading_device, GPUDevice):
        warnings.warn("Offloading from a GPUDevice is not fully supported and may lead to unexpected behavior.")

    offloading_device = device
    yield
    offloading_device = orig_offloading_device
