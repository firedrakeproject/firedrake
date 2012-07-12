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
import op_lib_core as core
import pyopencl as cl
import pkg_resources
import stringtemplate3
import pycparser
import numpy as np
import collections
import itertools
import warnings

_sum = 0

def trace():
    def decorator(f):
        def wrapper(*args, **kargs):
            print "%s (%s, %s)" % (f.__name__, args, kargs)
            print "%d" % kargs['size']
            global _sum
            _sum += kargs['size']
            print "running total %d" % (_sum)
            return f(*args, **kargs)
        return wrapper
    return decorator

#cl.Buffer = trace()(cl.Buffer)

def round_up(bytes):
    return (bytes + 15) & ~15

#TODO: use this instead of the unordered sets to ensure order is preserved
def _del_dup_keep_order(l):
    uniq = set()
    return [ x for x in l if x not in uniq and not uniq.add(x)]

class Kernel(op2.Kernel):

    _cparser = pycparser.CParser()

    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)
        # deactivate until we have the memory attribute generator
        # in order to allow passing "opencl" C kernels
        # self._ast = Kernel._cparser.parse(self._code)

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
    def _i_is_direct(self):
        return isinstance(self._dat, Dat) and self._map is IdentityMap

    @property
    def _i_is_indirect(self):
        return isinstance(self._dat, Dat) and self._map not in [None, IdentityMap]

    @property
    def _i_is_reduction(self):
        return isinstance(self._dat, Dat) and self._map != None and self._access in [INC, MIN, MAX]

    @property
    def _i_is_global_reduction(self):
        return isinstance(self._dat, Global)

class DeviceDataMixin:

    ClTypeInfo = collections.namedtuple('ClTypeInfo', ['clstring', 'zero'])
    CL_TYPES = {np.dtype('int16'): ClTypeInfo('short', '0'),
                np.dtype('uint32'): ClTypeInfo('unsigned int', '0u'),
                np.dtype('int32'): ClTypeInfo('int', '0')}

    @property
    def _cl_type(self):
        return DeviceDataMixin.CL_TYPES[self._data.dtype].clstring

    @property
    def _cl_type_zero(self):
        return DeviceDataMixin.CL_TYPES[self._data.dtype].zero

class Dat(op2.Dat, DeviceDataMixin):

    _arg_type = Arg

    def __init__(self, dataset, dim, data=None, dtype=None, name=None, soa=None):
        op2.Dat.__init__(self, dataset, dim, data, dtype, name, soa)
        self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
        cl.enqueue_copy(_queue, self._buffer, self._data, is_blocking=True).wait()

    @property
    def bytes_per_elem(self):
        #FIX: probably not the best way to do... (pad, alg ?)
        return self._data.nbytes / self._dataset.size

    @property
    def data(self):
        cl.enqueue_copy(_queue, self._data, self._buffer, is_blocking=True).wait()
        return self._data

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
        cl.enqueue_copy(_queue, self._buffer, self._data, is_blocking=True).wait()

    def _allocate_reduction_array(self, nelems):
        self._h_reduc_array = np.zeros ((round_up(nelems * self._data.itemsize),), dtype=self._data.dtype)
        self._d_reduc_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._h_reduc_array.nbytes)
        #NOTE: the zeroing of the buffer could be made with an opencl kernel call
        cl.enqueue_copy(_queue, self._d_reduc_buffer, self._h_reduc_array, is_blocking=True).wait()

    def _host_reduction(self, nelems):
        cl.enqueue_copy(_queue, self._h_reduc_array, self._d_reduc_buffer, is_blocking=True).wait()
        for j in range(self._dim[0]):
            self._data[j] = 0

        for i in range(nelems):
            for j in range(self._dim[0]):
                self._data[j] += self._h_reduc_array[j + i * self._dim[0]]

        warnings.warn('missing: updating buffer value')
        # get rid of the buffer and host temporary arrays
        del self._h_reduc_array
        del self._d_reduc_buffer

class Map(op2.Map):

    _arg_type = Arg

    def __init__(self, iterset, dataset, dim, values, name=None):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)
        if self._iterset._size != 0:
            self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._values.nbytes)
            cl.enqueue_copy(_queue, self._buffer, self._values, is_blocking=True).wait()

class OpPlan(core.op_plan):
    """ Helper wrapper
    """

    def __init__(self, kernel, itset, *args, **kargs):
        #FIX partition size by the our caller
        core.op_plan.__init__(self, kernel, *args, **kargs)
        self.itset = itset
        self.load()

    def reclaim(self):
        del self._ind_map_buffers
        del self._loc_map_buffers
        del self._ind_sizes_buffer
        del self._ind_offs_buffer
        del self._blkmap_buffer
        del self._offset_buffer
        del self._nelems_buffer
        del self._nthrcol_buffer
        del self._thrcol_buffer


    def load(self):
        self._ind_map_buffers = [None] * self.ninds
        for i in range(self.ninds):
            self._ind_map_buffers[i] = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=int(np.int32(0).itemsize * self.nindirect[i]))
            s = i * self.itset.size
            e = s + self.nindirect[i]
            cl.enqueue_copy(_queue, self._ind_map_buffers[i], self.ind_map[s:e], is_blocking=True).wait()

        self._loc_map_buffers = [None] * self.ninds
        for i in range(self.ninds):
            self._loc_map_buffers[i] = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=int(np.int16(0).itemsize * self.itset.size))
            s = i * self.itset.size
            e = s + self.itset.size
            cl.enqueue_copy(_queue, self._loc_map_buffers[i], self.loc_map[s:e], is_blocking=True).wait()

        self._ind_sizes_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self.ind_sizes.nbytes)
        cl.enqueue_copy(_queue, self._ind_sizes_buffer, self.ind_sizes, is_blocking=True).wait()

        self._ind_offs_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self.ind_offs.nbytes)
        cl.enqueue_copy(_queue, self._ind_offs_buffer, self.ind_offs, is_blocking=True).wait()

        self._blkmap_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self.blkmap.nbytes)
        cl.enqueue_copy(_queue, self._blkmap_buffer, self.blkmap, is_blocking=True).wait()

        self._offset_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self.offset.nbytes)
        cl.enqueue_copy(_queue, self._offset_buffer, self.offset, is_blocking=True).wait()

        self._nelems_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self.nelems.nbytes)
        cl.enqueue_copy(_queue, self._nelems_buffer, self.nelems, is_blocking=True).wait()

        self._nthrcol_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self.nthrcol.nbytes)
        cl.enqueue_copy(_queue, self._nthrcol_buffer, self.nthrcol, is_blocking=True).wait()

        self._thrcol_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self.thrcol.nbytes)
        cl.enqueue_copy(_queue, self._thrcol_buffer, self.thrcol, is_blocking=True).wait()

class DatMapPair(object):
    """ Dummy class needed for codegen
        could do without but would obfuscate codegen templates
    """
    def __init__(self, dat, map):
        self._dat = dat
        self._map = map

    @property
    def _i_direct(self):
        return isinstance(self._dat, Dat) and self._map != IdentityMap

#FIXME: some of this can probably be factorised up in common
class ParLoopCall(object):

    def __init__(self, kernel, it_space, *args):
        self._it_space = it_space
        self._kernel = kernel
        self._args = list(args)

    """ code generation specific """
    """ a lot of this can rewriten properly """
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

    @property
    def _unique_dats(self):
        return _del_dup_keep_order(map(lambda arg: arg._dat, filter(lambda arg: not isinstance(arg._dat, Global), self._args)))

    @property
    def _i_staged_dat_map_pairs(self):
        assert not self.is_direct(), "Should only be called on indirect loops"
        return set(map(lambda arg: DatMapPair(arg._dat, arg._map), filter(lambda a: not (a._map == IdentityMap or isinstance(a._dat, Global)), self._args)))

    @property
    def _i_staged_in_dat_map_pairs(self):
        assert not self.is_direct(), "Should only be called on indirect loops"
        return set(map(lambda arg: DatMapPair(arg._dat, arg._map), filter(lambda a: a._map != IdentityMap and not isinstance(a._dat, Global) and a._access in [READ, RW], self._args)))

    @property
    def _i_staged_out_dat_map_pairs(self):
        assert not self.is_direct(), "Should only be called on indirect loops"
        return set(map(lambda arg: DatMapPair(arg._dat, arg._map), filter(lambda a: a._map != IdentityMap and not isinstance(a._dat, Global) and a._access in [WRITE, RW], self._args)))

    @property
    def _i_reduc_args(self):
        assert not self.is_direct(), "Should only be called on indirect loops"
        return list(set(filter(lambda a: a._access in [INC, MIN, MAX] and a._map != IdentityMap and not isinstance(a._dat, Global), self._args)))

    @property
    def _i_global_reduc_args(self):
        assert not self.is_direct(), "Should only be called on indirect loops"
        return list(set(filter(lambda a: isinstance(a._dat, Global), self._args)))

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
                              "threads_per_block": _threads_per_block,
                              "partition_size": _threads_per_block}
            source = str(dloop)
            prg = cl.Program (_ctx, source).build(options="-Werror")
            kernel = prg.__getattr__(self._kernel._name + '_stub')
            self._karg = 0
            for a in self._d_nonreduction_args:
                self._kernel_arg_append(kernel, a._dat._buffer)

            for a in self._d_reduction_args:
                a._dat._allocate_reduction_array(_blocks_per_grid)
                self._kernel_arg_append(kernel, a._dat._d_reduc_buffer)

            cl.enqueue_nd_range_kernel(_queue, kernel, (thread_count,), (_threads_per_block,), g_times_l=False).wait()
            for i, a in enumerate(self._d_reduction_args):
                a._dat._host_reduction(_blocks_per_grid)
        else:
            psize = 1024
            plan = OpPlan(self._kernel, self._it_space, *self._args, partition_size=psize)

            # codegen
            iloop = _stg_indirect_loop.getInstanceOf("indirect_loop")
            iloop['parloop'] = self
            iloop['const'] = {'dynamic_shared_memory_size': plan.nshared, 'ninds':plan.ninds, 'partition_size':psize}
            source = str(iloop)

            # for debugging purpose, refactor that properly at some point
            #f = open(self._kernel._name + '.cl.c', 'w')
            #f.write(source)
            #f.close

            prg = cl.Program(_ctx, source).build(options="-Werror")
            kernel = prg.__getattr__(self._kernel._name + '_stub')

            self._karg = 0
            for a in self._unique_dats:
                self._kernel_arg_append(kernel, a._buffer)

            for i in range(plan.ninds):
                self._kernel_arg_append(kernel, plan._ind_map_buffers[i])

            for i in range(plan.ninds):
                    self._kernel_arg_append(kernel, plan._loc_map_buffers[i])

            for arg in self._i_global_reduc_args:
                arg._dat._allocate_reduction_array(plan.nblocks)
                self._kernel_arg_append(kernel, arg._dat._d_reduc_buffer)


            self._kernel_arg_append(kernel, plan._ind_sizes_buffer)
            self._kernel_arg_append(kernel, plan._ind_offs_buffer)
            self._kernel_arg_append(kernel, plan._blkmap_buffer)
            self._kernel_arg_append(kernel, plan._offset_buffer)
            self._kernel_arg_append(kernel, plan._nelems_buffer)
            self._kernel_arg_append(kernel, plan._nthrcol_buffer)
            self._kernel_arg_append(kernel, plan._thrcol_buffer)

            block_offset = 0
            for i in range(plan.ncolors):
                blocks_per_grid = int(plan.ncolblk[i])
                threads_per_block = _threads_per_block
                thread_count = threads_per_block * blocks_per_grid

                kernel.set_arg(self._karg, np.int32(block_offset))
                cl.enqueue_nd_range_kernel(_queue, kernel, (thread_count,), (threads_per_block,), g_times_l=False).wait()
                block_offset += blocks_per_grid

            for arg in self._i_global_reduc_args:
                arg._dat._host_reduction(plan.nblocks)

            plan.reclaim()


    def _kernel_arg_append(self, kernel, arg):
        kernel.set_arg(self._karg, arg)
        self._karg += 1

    def is_direct(self):
        return all(map(lambda a: isinstance(a._dat, Global) or ((isinstance(a._dat, Dat) and a._map == IdentityMap)), self._args))

def par_loop(kernel, it_space, *args):
    ParLoopCall(kernel, it_space, *args).compute()

_ctx = cl.create_some_context()
_queue = cl.CommandQueue(_ctx)
_threads_per_block = _ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
_warpsize = 1

#preload string template groups
_stg_direct_loop = stringtemplate3.StringTemplateGroup(file=stringtemplate3.StringIO(pkg_resources.resource_string(__name__, "assets/opencl_direct_loop.stg")), lexer="default")
_stg_indirect_loop = stringtemplate3.StringTemplateGroup(file=stringtemplate3.StringIO(pkg_resources.resource_string(__name__, "assets/opencl_indirect_loop.stg")), lexer="default")
