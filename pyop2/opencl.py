import sequential as op2

class Kernel(op2.Kernel):
    def __init__(self, code, name=None):
        op2.Kernel.__init__(self, code, name)

class DeviceDataMixin:
    def fetch_data(self):
        pass

class Dat(op2.Dat, DeviceDataMixin):
    def __init__(self, dataset, dim, datatype=None, data=None, name=None):
        op2.Dat.__init__(self, dataset, dim, datatype, data, name)

class Mat(op2.Mat, DeviceDataMixin):
    def __init__(self, datasets, dim, datatype=None, name=None):
        op2.Mat.__init__(self, datasets, dim, datatype, data, name)

class Const(op2.Const, DeviceDataMixin):
    def __init__(self, dim, value, name=None):
        op2.Const.__init__(self, dim, value, name)

class Global(op2.Global, DeviceDataMixin):
    def __init__(self, dim, value, name=None):
        op2.Global.__init__(self, dim, value, name)

    @property
    def value(self):
        self._value = self.fetch_data()
        return self._value

class Map(op2.Map):
    def __init__(self, iterset, dataset, dim, values, name=None):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)

def par_loop(kernel, it_space, *args):
    pass
