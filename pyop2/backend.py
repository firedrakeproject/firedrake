import op2

class Backend(object):
    """
    Generic backend interface
    """

    def __init__(self):
        raise NotImplementedError()

    def handle_kernel_declaration(self, kernel):
        raise NotImplementedError()

    def handle_datacarrier_declaration(self, datacarrier):
        raise NotImplementedError()

    def handle_map_declaration(self, map):
        raise NotImplementedError()

    def handle_par_loop_call(self, kernel, it_space, *args):
        raise NotImplementedError()

    def handle_datacarrier_retrieve_value(self, datacarrier):
        raise NotImplementedError()

class VoidBackend(Backend):
    """
    Checks for valid usage of the interface,
    but actually does nothing
    """

    def __init__(self):
        pass

    def handle_kernel_declaration(self, kernel):
        assert isinstance(kernel, op2.Kernel)

    def handle_datacarrier_declaration(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)

    def handle_map_declaration(self, map):
        assert isinstance(map, op2.Map)

    def handle_par_loop_call(self, kernel, it_space, *args):
        pass

    def handle_datacarrier_retrieve_value(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)

backends = { 'void': VoidBackend() }
