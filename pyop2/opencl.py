# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

import sequential as op2
from utils import verify_reshape
from sequential import IdentityMap, READ, WRITE, RW, INC, MIN, MAX
import pyopencl as cl
import pkg_resources
import stringtemplate3
import pycparser
import numpy as np
import collections

def round_up(bytes):
    return (bytes + 15) & ~15

class Kernel(op2.Kernel):

    _cparser = pycparser.CParser()

    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)
        self._ast = Kernel._cparser.parse(self._code)

class Arg(op2.Arg):
    def __init__(self, data=None, map=None, idx=None, access=None):
        op2.Arg.__init__(self, data, map, idx, access)

    @property
    def _d_is_INC(self):
        return self._access == INC

    @property
    def _d_is_staged(self):
        # FIX; stagged only if dim > 1
        return isinstance(self._dat, Dat) and self._access in [READ, WRITE, RW]

    @property
    def _i_direct(self):
        return isinstance(self._dat, Dat) and self._map != IdentityMap

class DeviceDataMixin:

    ClTypeInfo = collections.namedtuple('ClTypeInfo', ['clstring', 'zero'])
    CL_TYPES = {np.dtype('uint32'): ClTypeInfo('unsigned int', '0u')}

    def fetch_data(self):
        cl.enqueue_read_buffer(_queue, self._buffer, self._data).wait()

class Dat(op2.Dat, DeviceDataMixin):

    _arg_type = Arg

    def __init__(self, dataset, dim, data=None, dtype=None, name=None, soa=None):
        op2.Dat.__init__(self, dataset, dim, data, dtype, name, soa)
        self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
        cl.enqueue_write_buffer(_queue, self._buffer, self._data).wait()

    @property
    def bytes_per_elem(self):
        #FIX: probably not the best way to do... (pad, alg ?)
        return self._data.nbytes / self._dataset.size

    @property
    def data(self):
        cl.enqueue_read_buffer(_queue, self._buffer, self._data).wait()
        return self._data

    @property
    def _cl_type(self):
        return DataCarrier.CL_TYPES[self._data.dtype].clstring

    @property
    def _cl_type_zero(self):
        return DataCarrier.CL_TYPES[self._data.dtype].zero

class Mat(op2.Mat, DeviceDataMixin):

    _arg_type = Arg

    def __init__(self, datasets, dim, dtype=None, name=None):
        op2.Mat.__init__(self, datasets, dim, dtype, name)
        raise NotImplementedError('Matrix data is unsupported yet')

class Const(op2.Const, DeviceDataMixin):

    def __init__(self, dim, data, name, dtype=None):
        op2.Const.__init__(self, dim, data, name, dtype)
        raise NotImplementedError('Const data is unsupported yet')

class Global(op2.Global, DeviceDataMixin):

    _arg_type = Arg

    def __init__(self, dim, data, dtype=None, name=None):
        op2.Global.__init__(self, dim, data, dtype, name)
        self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
        cl.enqueue_write_buffer(_queue, self._buffer, self._data).wait()

    def _allocate_reduction_array(self, nelems):
        self._h_reduc_array = np.zeros ((round_up(nelems * self._datatype(0).nbytes),), dtype=self._datatype)
        self._d_reduc_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._h_reduc_array.nbytes)
        #NOTE: the zeroing of the buffer could be made with an opencl kernel call
        cl.enqueue_write_buffer(_queue, self._d_reduc_buffer, self._h_reduc_array).wait()

    def _host_reduction(self, nelems):
        cl.enqueue_read_buffer(_queue, self._d_reduc_buffer, self._h_reduc_array).wait()
        for i in range(nelems):
            for j in range(self._dim[0]):
                self._data[j] += self._h_reduc_array[j + i * self._dim[0]]

        # update on device buffer
        cl.enqueue_write_buffer(_queue, self._d_reduc_buffer, self._h_reduc_array).wait()

        # get rid of the buffer and host temporary arrays
        del self._h_reduc_array
        del self._d_reduc_buffer

    @property
    def data(self):
        self.fetch_data()
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        self._on_device = False

class Map(op2.Map):

    _arg_type = Arg

    def __init__(self, iterset, dataset, dim, values, name=None):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)
        if self._iterset._size != 0:
            self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, self._values.nbytes)
            cl.enqueue_write_buffer(_queue, self._buffer, self._values).wait()

class DatMapPair(object):
    """ Dummy class needed for codegen
        could do without but would obfuscate codegen templates
    """
    def __init__(self, dat, map):
        self._dat = dat
        self._map = map

#FIXME: some of this can probably be factorised up in common
class ParLoopCall(object):

    def __init__(self, kernel, it_space, *args):
        self._it_space = it_space
        self._kernel = kernel
        self._args = list(args)
        self.compute()

    @property
    def _d_staged_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(self._d_staged_in_args + self._d_staged_out_args))

    @property
    def _d_nonreduction_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(filter(lambda a: not isinstance(a._dat, Global), self._args)))

    @property
    def _d_staged_in_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(filter(lambda a: isinstance(a._dat, Dat) and a._access in [READ, RW], self._args)))

    @property
    def _d_staged_out_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(filter(lambda a: isinstance(a._dat, Dat) and a._access in [WRITE, RW], self._args)))

    @property
    def _d_reduction_args(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return list(set(filter(lambda a: isinstance(a._dat, Global) and a._access in [INC, MIN, MAX], self._args)))

    """ maximum shared memory required for staging an op_arg """
    def _d_max_dynamic_shared_memory(self):
        assert self.is_direct(), "Should only be called on direct loops"
        return max(map(lambda a: a._dat.bytes_per_elem, self._d_staged_args))

    def compute(self):
        if self.is_direct():
            thread_count = _threads_per_block * _blocks_per_grid
            dynamic_shared_memory_size = self._d_max_dynamic_shared_memory()
            shared_memory_offset = dynamic_shared_memory_size * _warpsize
            dynamic_shared_memory_size = dynamic_shared_memory_size * _threads_per_block
            dloop = _stg_direct_loop.getInstanceOf("direct_loop")
            dloop['parloop'] = self
            dloop['const'] = {"warpsize": _warpsize,\
                              "shared_memory_offset": shared_memory_offset,\
                              "dynamic_shared_memory_size": dynamic_shared_memory_size,\
                              "threads_per_block": _threads_per_block}
            source = str(dloop)
            prg = cl.Program (_ctx, source).build(options="-Werror")
            kernel = prg.__getattr__(self._kernel._name + '_stub')
            for i, a in enumerate(self._d_nonreduction_args):
                kernel.set_arg(i, a._dat._buffer)
            for i, a in enumerate(self._d_reduction_args):
                a._dat._allocate_reduction_array(_blocks_per_grid)
                kernel.set_arg(i + len(self._d_nonreduction_args), a._dat._d_reduc_buffer)

            cl.enqueue_nd_range_kernel(_queue, kernel, (thread_count,), (_threads_per_block,), g_times_l=False).wait()
            for i, a in enumerate(self._d_reduction_args):
                a._dat._host_reduction(_blocks_per_grid)
        else:
            raise NotImplementedError()

    def is_direct(self):
        return all(map(lambda a: isinstance(a._dat, Global) or (isinstance(a._dat, Dat) and a._map == IdentityMap), self._args))

def par_loop(kernel, it_space, *args):
    ParLoopCall(kernel, it_space, *args)

_ctx = cl.create_some_context()
_queue = cl.CommandQueue(_ctx)
_threads_per_block = _ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
_warpsize = 1

#preload string template groups
_stg_direct_loop = stringtemplate3.StringTemplateGroup(file=stringtemplate3.StringIO(pkg_resources.resource_string(__name__, "assets/opencl_direct_loop.stg")), lexer="default")
