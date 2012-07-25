import sequential as op2
from utils import verify_reshape

class Kernel(op2.Kernel):
    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)
        self._bin = None

    def compile(self):
        if self._bin is None:
            self._bin = self._code

    def handle(self):
        pass

class DeviceDataMixin:
    def fetch_data(self):
        return self._data

class Dat(op2.Dat, DeviceDataMixin):
    def __init__(self, dataset, dim, data=None, dtype=None, name=None, soa=None):
        op2.Dat.__init__(self, dataset, dim, data, dtype, name, soa)
        self._on_device = False

class Mat(op2.Mat, DeviceDataMixin):
    def __init__(self, datasets, dim, dtype=None, name=None):
        op2.Mat.__init__(self, datasets, dim, dtype, name)
        self._on_device = False

class Const(op2.Const, DeviceDataMixin):
    def __init__(self, dim, data, name, dtype=None):
        op2.Const.__init__(self, dim, data, name, dtype)
        self._on_device = False

class Global(op2.Global, DeviceDataMixin):
    def __init__(self, dim, data, dtype=None, name=None):
        op2.Global.__init__(self, dim, data, dtype, name)
        self._on_device = False

    @property
    def data(self):
        self._data = self.fetch_data()
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        self._on_device = False

class Map(op2.Map):
    def __init__(self, iterset, dataset, dim, values, name=None):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)
        self._on_device = False

def par_loop(kernel, it_space, *args):
    kernel.compile()
    pass
