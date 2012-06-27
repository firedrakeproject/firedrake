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

class OpenCLBackend(Backend):
    """
    Checks for valid usage of the interface,
    but actually does nothing
    """

    def __init__(self):
        self._ctx = cl.create_some_context()
        self._queue = cl.CommandQueue(self._ctx)
        self._warpsize = 1
        self._threads_per_block = self._ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        self._blocks_per_grid = 200

    def handle_kernel_declaration(self, kernel):
        assert isinstance(kernel, op2.Kernel)

    def handle_datacarrier_declaration(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)
        if (isinstance(datacarrier, op2.Dat)):
            buf = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE, size=datacarrier._data.nbytes)
            cl.enqueue_write_buffer(self._queue, buf, datacarrier._data).wait()
            self._buffers[datacarrier] = buf
        else:
            raise NotImplementedError()

    def handle_map_declaration(self, map):
        assert isinstance(map, op2.Map)
        # dirty how to do that properly ?
        if not map._name == 'identity':
            #FIX: READ ONLY
            buf = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE, size=map._values.nbytes)
            cl.enqueue_write_buffer(self._queue, buf, map._values).wait()
            self._buffers[map] = buf

    def handle_par_loop_call(self, kernel, it_space, *args):
        pass

    def handle_datacarrier_retrieve_value(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)

backends = {
    'void': VoidBackend(),
    'opencl': OpenCLBackend
    }
