class Access(object):
    def __init__(self, *args):
        raise NotImplementedError("Please call op2.init to select a backend")

class IterationSpace(object):
    def __init__(self, *args):
        raise NotImplementedError("Please call op2.init to select a backend")

class Set(object):
    def __init__(self, *args):
        raise NotImplementedError("Please call op2.init to select a backend")

class Kernel(object):
    def __init__(self, *args):
        raise NotImplementedError("Please call op2.init to select a backend")

class Dat(object):
    def __init__(self, *args):
        raise NotImplementedError("Please call op2.init to select a backend")

class Mat(object):
    def __init__(self, *args):
        raise NotImplementedError("Please call op2.init to select a backend")

class Const(object):
    def __init__(self, *args):
        raise NotImplementedError("Please call op2.init to select a backend")

class Global(object):
    def __init__(self, *args):
        raise NotImplementedError("Please call op2.init to select a backend")

class Map(object):
    def __init__(self, *args):
        raise NotImplementedError("Please call op2.init to select a backend")

def par_loop(*args):
    raise NotImplementedError("Please call op2.init to select a backend")
