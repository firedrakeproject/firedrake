import abc
import contextlib
from cuda.core.experimental import Device, Stream

class ComputeDevice(abc.ABC):
    pass

class CPUDevice(ComputeDevice):
    pass

class GPUDevice(ComputeDevice):

    def __init__(self, num_threads=32):
        self.num_threads=num_threads



class gpu_assembly:

    def __init__(self):
        self.device = Device(0)

    def __enter__(self):
        self.gb = self.device.create_graph_builder()
        self.gb.begin_building()
        return self.gb.stream.handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gb.end_building()
        graph = self.gb.complete()

        graph.launch(self.device.default_stream)
        self.device.sync()
