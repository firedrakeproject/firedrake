from common import Access, IterationSpace, Set, READ, WRITE, RW, INC, MIN, MAX
import common

class Kernel(common.Kernel):
    def __init__(self, code, name):
        common.Kernel.__init__(self, code, name)

class DataCarrier(common.DataCarrier):
    def fetch_data(self):
        pass

class Dat(common.Dat, DataCarrier):
    def __init__(self, dataset, dim, datatype, data, name):
        common.Dat.__init__(self, dataset, dim, datatype, data, name)

class Mat(common.Mat, DataCarrier):
    def __init__(self, datasets, dim, datatype, name):
        common.Mat.__init__(self, datasets, dim, datatype, data, name)

class Const(common.Const, DataCarrier):
    def __init__(self, dim, value, name):
        common.Const.__init__(self, dim, value, name)

class Global(common.Global, DataCarrier):
    def __init__(self, dim, value, name):
        common.Global.__init__(self, dim, value, name)

    @property
    def value(self):
        self._value = self.fetch_data()
        return self._value

class Map(common.Map):
    def __init__(self, iterset, dataset, dim, values, name):
        common.Map.__init__(self, iterset, dataset, dim, values, name)

def par_loop(kernel, it_space, *args):
    pass
