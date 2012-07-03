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

class Kernel(op2.Kernel):
    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)

class DeviceDataMixin:
    def fetch_data(self):
        cl.enqueue_read_buffer(_queue, self._buffer, self._data).wait()

class Dat(op2.Dat, DeviceDataMixin):
    def __init__(self, dataset, dim, data=None, dtype=None, name=None, soa=None):
        op2.Dat.__init__(self, dataset, dim, data, dtype, name, soa)
        self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
        cl.enqueue_write_buffer(_queue, self._buffer, self._data).wait()

    @property
    def data(self):
        self.fetch_data()
        return self._data

class Mat(op2.Mat, DeviceDataMixin):
    def __init__(self, datasets, dim, dtype=None, name=None):
        op2.Mat.__init__(self, datasets, dim, dtype, name)
        raise NotImplementedError('Matrix data is unsupported yet')

class Const(op2.Const, DeviceDataMixin):
    def __init__(self, dim, data, name, dtype=None):
        op2.Const.__init__(self, dim, data, name, dtype)
        raise NotImplementedError('Const data is unsupported yet')

class Global(op2.Global, DeviceDataMixin):
    def __init__(self, dim, data, dtype=None, name=None):
        op2.Global.__init__(self, dim, data, dtype, name)
        raise NotImplementedError('Global data is unsupported yet')

    @property
    def data(self):
        self.fetch_data()
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        self._on_device = False

class Map(op2.Map):
    def __init__(self, iterset, dataset, dim, values, name=None):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)
        if self._iterset._size != 0:
            self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, self._values.nbytes)
            cl.enqueue_write_buffer(_queue, self._buffer, self._values).wait()

#FIXME: some of this can probably be factorised up in common
class ParLoopCall(object):
    def __init__(self, kernel, it_space, *args):
        self._it_space = it_space
        self._kernel = kernel
        self._args = args
        print self._args
        self.compute()

    def compute(self):
        if self.is_direct:
            print 'COMPUTE.........'
            thread_count = _threads_per_block * _blocks_per_grid
            dynamic_shared_memory_size = max(map(lambda a: a['dat'].dim * a['dat'].datatype.nbytes, self._args))
            shared_memory_offset = dynamic_shared_memory_size * _warpsize
            dynamic_shared_memory_size = dynamic_shared_memory_size * _threads_per_block
            dloop = group.getInstanceOf("direct_loop")
            dloop['parloop'] = self
            dloop['const'] = {"warpsize": _warpsize,\
                              "shared_memory_offset": shared_memory_offset,\
                              "dynamic_shared_memory_size": dynamic_shared_memory_size}
            source = str(dloop)
            prg = cl.Program (op2['ctx'], source).build(options="-Werror")
            kernel = prg.__getattr__(self._kernel._name + '_stub')
            for i, a in enumerate(self._args):
                kernel.set_arg(i, a._dat._buffer)
            cl.enqueue_nd_range_kernel(_queue, self._kernel, (thread_count,), (threads_per_block,), g_times_l=False).wait()
        else:
            raise NotImplementedError()

    def is_direct(self):
        return all(map(lambda a: a['map'] == IdentityMap), self._args)

def par_loop(kernel, it_space, *args):
    ParLoopCall(kernel, it_space, *args)

_ctx = cl.create_some_context()
_queue = cl.CommandQueue(_ctx)
_threads_per_block = _ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
_warpsize = 1

#preload string template groups
_stg_direct_loop = stringtemplate3.StringTemplateGroup(file=stringtemplate3.StringIO(pkg_resources.resource_string(__name__, "assets/opencl_direct_loop.stg")), lexer="default")
