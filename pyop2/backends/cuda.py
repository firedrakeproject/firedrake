from common import Access, IterationSpace, Set, IdentityMap
from common import READ, WRITE, RW, INC, MIN, MAX
import common

class Kernel(common.Kernel):
    def __init__(self, code, name):
        common.Kernel.__init__(self, code, name)
        self._bin = None

    def compile(self):
        if self._bin is None:
            self._bin = self._code

    def handle(self):
        pass

class DataCarrier(common.DataCarrier):
    def fetch_data(self):
        pass

class Dat(common.Dat, DataCarrier):
    def __init__(self, dataset, dim, datatype, data, name):
        common.Dat.__init__(self, dataset, dim, datatype, data, name)
        self._on_device = False

class Mat(common.Mat, DataCarrier):
    def __init__(self, datasets, dim, datatype, name):
        common.Mat.__init__(self, datasets, dim, datatype, data, name)
        self._on_device = False

class Const(common.Const, DataCarrier):
    def __init__(self, dim, value, name):
        common.Const.__init__(self, dim, value, name)
        self._on_device = False

class Global(common.Global, DataCarrier):
    def __init__(self, dim, value, name):
        common.Global.__init__(self, dim, value, name)
        self._on_device = False

    @property
    def value(self):
        self._value = self.fetch_data()
        return self._value

class Map(common.Map):
    def __init__(self, iterset, dataset, dim, values, name):
        common.Map.__init__(self, iterset, dataset, dim, values, name)
        self._on_device = False

def par_loop(kernel, it_space, *args):
    kernel.compile()
    pass
