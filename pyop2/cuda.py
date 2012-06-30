import sequential as op2

class Kernel(op2.Kernel):
    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)
        self._bin = None

    def compile(self):
        if self._bin is None:
            self._bin = self._code

    def handle(self):
        pass

class DataCarrier(op2.DataCarrier):
    def fetch_data(self):
        pass

class Dat(op2.Dat, DataCarrier):
    def __init__(self, dataset, dim, datatype, data, name):
        op2.Dat.__init__(self, dataset, dim, datatype, data, name)
        self._on_device = False

class Mat(op2.Mat, DataCarrier):
    def __init__(self, datasets, dim, datatype, name):
        op2.Mat.__init__(self, datasets, dim, datatype, data, name)
        self._on_device = False

class Const(op2.Const, DataCarrier):
    def __init__(self, dim, value, name):
        op2.Const.__init__(self, dim, value, name)
        self._on_device = False

class Global(op2.Global, DataCarrier):
    def __init__(self, dim, value, name):
        op2.Global.__init__(self, dim, value, name)
        self._on_device = False

    @property
    def value(self):
        self._value = self.fetch_data()
        return self._value

class Map(op2.Map):
    def __init__(self, iterset, dataset, dim, values, name):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)
        self._on_device = False

def par_loop(kernel, it_space, *args):
    kernel.compile()
    pass
