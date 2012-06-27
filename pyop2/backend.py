import op2

class ParLoopCall(object):
    """
    Backend Agnostic support code
    """

    def __init__(self, kernel, it_space, *args):
        assert ParLoopCall.check(kernel, it_space, *args)
        self._kernel = kernel
        self._it_space = it_space
        self._args = args

    @staticmethod
    def check(kernel, it_space, *args):
        #TODO
        return True

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
        self._handle_par_loop_call(ParLoopCall(kernel, it_space, args))

    def _handle_par_loop_call(self, parloop):
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

    def _handle_par_loop_call(self, parloop):
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

    def _handle_par_loop_call(self, parloop):
        pass

    def handle_datacarrier_retrieve_value(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)

backends = {
    'void': VoidBackend(),
    'opencl': OpenCLBackend
    }
