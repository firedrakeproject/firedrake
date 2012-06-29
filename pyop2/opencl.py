from op2 import Access, IterationSpace, Set, IdentityMap, \
        READ, WRITE, RW, INC, MIN, MAX
import op2

class Kernel(op2.Kernel):
    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)

class DataCarrier(op2.DataCarrier):
    def fetch_data(self):
        pass

class Dat(op2.Dat, DataCarrier):
    def __init__(self, dataset, dim, datatype, data, name):
        op2.Dat.__init__(self, dataset, dim, datatype, data, name)

class Mat(op2.Mat, DataCarrier):
    def __init__(self, datasets, dim, datatype, name):
        op2.Mat.__init__(self, datasets, dim, datatype, data, name)

class Const(op2.Const, DataCarrier):
    def __init__(self, dim, value, name):
        op2.Const.__init__(self, dim, value, name)

class Global(op2.Global, DataCarrier):
    def __init__(self, dim, value, name):
        op2.Global.__init__(self, dim, value, name)

    @property
    def value(self):
        self._value = self.fetch_data()
        return self._value

class Map(op2.Map):
    def __init__(self, iterset, dataset, dim, values, name):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)

def par_loop(kernel, it_space, *args):
    pass
