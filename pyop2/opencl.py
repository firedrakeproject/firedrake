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
import sys
import math
from pycparser import c_parser, c_ast, c_generator
from utils import align, uniquify

class Kernel(op2.Kernel):
    """OP2 OpenCL kernel type."""

    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)

    class Instrument(c_ast.NodeVisitor):
        """C AST visitor for instrumenting user kernels.
             - adds memory space attribute to user kernel declaration
             - adds a separate function declaration for user kernel
        """
        def instrument(self, ast, kernel_name, instrument):
            self._kernel_name = kernel_name
            self._instrument = instrument
            self._ast = ast
            self.generic_visit(ast)
            idx = ast.ext.index(self._func_node)
            ast.ext.insert(0, self._func_node.decl)

        def visit_FuncDef(self, node):
            if node.decl.name == self._kernel_name:
                self._func_node = node
                self.visit(node.decl)

        def visit_ParamList(self, node):
            for i, p in enumerate(node.params):
                if self._instrument[i][0]:
                    p.storage.append(self._instrument[i][0])
                if self._instrument[i][1]:
                    p.type.quals.append(self._instrument[i][1])

    def instrument(self, instrument):
        ast = c_parser.CParser().parse(self._code)
        Kernel.Instrument().instrument(ast, self._name, instrument)
        self._inst_code = c_generator.CGenerator().visit(ast)

class Arg(op2.Arg):
    """OP2 OpenCL argument type."""

    def __init__(self, data=None, map=None, idx=None, access=None):
        op2.Arg.__init__(self, data, map, idx, access)

    # Codegen specific

    @property
    def _d_is_staged(self):
        return self._is_direct and not self._dat._is_scalar

    @property
    def _i_gen_vec(self):
        assert self._is_vec_map
        return map(lambda i: Arg(self._dat, self._map, i, self._access), range(self._map._dim))


class DeviceDataMixin:
    """Codegen mixin for datatype and literal translation."""

    ClTypeInfo = collections.namedtuple('ClTypeInfo', ['clstring', 'zero'])
    CL_TYPES = {np.dtype('uint8'): ClTypeInfo('uchar', '0'),
                np.dtype('int8'): ClTypeInfo('char', '0'),
                np.dtype('uint16'): ClTypeInfo('ushort', '0'),
                np.dtype('int16'): ClTypeInfo('short', '0'),
                np.dtype('uint32'): ClTypeInfo('uint', '0u'),
                np.dtype('int32'): ClTypeInfo('int', '0'),
                np.dtype('uint64'): ClTypeInfo('ulong', '0ul'),
                np.dtype('int64'): ClTypeInfo('long', '0l'),
                np.dtype('float32'): ClTypeInfo('float', '0.0f'),
                np.dtype('float64'): ClTypeInfo('double', '0.0')}

    @property
    def bytes_per_elem(self):
        return self.dtype.itemsize * np.prod(self.dim)

    @property
    def _is_scalar(self):
        return np.prod(self.dim) == 1

    @property
    def _cl_type(self):
        return DeviceDataMixin.CL_TYPES[self._data.dtype].clstring

    @property
    def _cl_type_zero(self):
        return DeviceDataMixin.CL_TYPES[self._data.dtype].zero

class Dat(op2.Dat, DeviceDataMixin):
    """OP2 OpenCL vector data type."""

    _arg_type = Arg

    def __init__(self, dataset, dim, data=None, dtype=None, name=None, soa=None):
        op2.Dat.__init__(self, dataset, dim, data, dtype, name, soa)
        if data is not None:
            self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
            cl.enqueue_copy(_queue, self._buffer, self._data, is_blocking=True).wait()
        else:
            self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE,
                                     size=int(dataset.size * self.dtype.itemsize * np.prod(self.dim)))

    @property
    def data(self):
        if len(self._data) is 0:
            raise RuntimeError("Temporary dat has no data on the host")
        cl.enqueue_copy(_queue, self._data, self._buffer, is_blocking=True).wait()
        if self._soa:
            np.transpose(self._data)
        return self._data

class Mat(op2.Mat, DeviceDataMixin):
    """OP2 OpenCL matrix data type."""

    _arg_type = Arg

    def __init__(self, datasets, dim, dtype=None, name=None):
        op2.Mat.__init__(self, datasets, dim, dtype, name)
        raise NotImplementedError('Matrix data is unsupported yet')

class Const(op2.Const, DeviceDataMixin):
    """OP2 OpenCL data that is constant for any element of any set."""

    def __init__(self, dim, data, name, dtype=None):
        op2.Const.__init__(self, dim, data, name, dtype)

    @property
    def _cl_value(self):
        return list(self._data)

class Global(op2.Global, DeviceDataMixin):
    """OP2 OpenCL global value."""

    _arg_type = Arg

    def __init__(self, dim, data, dtype=None, name=None):
        op2.Global.__init__(self, dim, data, dtype, name)
        self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._data.nbytes)
        cl.enqueue_copy(_queue, self._buffer, self._data, is_blocking=True).wait()

    def _allocate_reduction_array(self, nelems):
        self._h_reduc_array = np.zeros ((align(nelems * self._data.itemsize, 16),), dtype=self._data.dtype)
        self._d_reduc_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._h_reduc_array.nbytes)
        cl.enqueue_copy(_queue, self._d_reduc_buffer, self._h_reduc_array, is_blocking=True).wait()

    @property
    def data(self):
        cl.enqueue_copy(_queue, self._data, self._buffer, is_blocking=True).wait()
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        cl.enqueue_copy(_queue, self._buffer, self._data, is_blocking=True).wait()

    def _post_kernel_reduction_task(self, nelems):
        src = """
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel
void %(name)s_reduction (
  __global %(type)s* dat,
  __global %(type)s* tmp,
  __private int count
)
{
  __private %(type)s accumulator[%(dim)d];
  for (int j = 0; j < %(dim)d; ++j)
  {
    accumulator[j] = %(zero)s;
  }
  for (int i = 0; i < count; ++i)
  {
    for (int j = 0; j < %(dim)d; ++j)
    {
      accumulator[j] += *(tmp + i * %(dim)d + j);
    }
  }
  for (int j = 0; j < %(dim)d; ++j)
  {
    *(dat + j) = accumulator[j];
  }

}
""" % {'name': self._name,
       'dim': np.prod(self._dim),
       'type': self._cl_type,
       'zero': self._cl_type_zero}

        prg = cl.Program(_ctx, src).build(options="-Werror")
        kernel = prg.__getattr__(self._name + '_reduction')
        kernel.append_arg(self._buffer)
        kernel.append_arg(self._d_reduc_buffer)
        kernel.append_arg(np.int32(nelems))
        cl.enqueue_task(_queue, kernel).wait()

        del self._d_reduc_buffer

class Map(op2.Map):
    """OP2 OpenCL map, a relation between two Sets."""

    _arg_type = Arg

    def __init__(self, iterset, dataset, dim, values, name=None):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)
        if self._iterset._size != 0:
            self._buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._values.nbytes)
            cl.enqueue_copy(_queue, self._buffer, self._values, is_blocking=True).wait()

class OpPlanCache():
    """Cache for OpPlan."""

    def __init__(self):
        self._cache = dict()

    def get_plan(self, parloop, **kargs):
        cp = core.op_plan(parloop._kernel, parloop._it_space, *parloop._args, **kargs)
        try:
            plan = self._cache[cp.hsh]
        except KeyError:
            plan = OpPlan(parloop, cp)
            self._cache[cp.hsh] = plan

        return plan

class GenCodeCache():
    """Cache for generated code.
         Keys: OP2 kernels
         Entries: generated code strings, OpenCL built programs tuples
    """

    def __init__(self):
        self._cache = dict()

    def get_code(self, kernel):
        try:
            return self._cache[kernel]
        except KeyError:
            return (None, None)

    def cache_code(self, kernel, code):
        self._cache[kernel] = code

class OpPlan():
    """ Helper proxy for core.op_plan."""

    def __init__(self, parloop, core_plan):
        self._parloop = parloop
        self._core_plan = core_plan
        self._loaded = False

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
        self.nuinds = sum(map(lambda a: a._is_indirect, self._parloop._args))
        _ind_desc = [-1] * len(self._parloop._args)
        _d = {}
        _c = 0
        for i, arg in enumerate(self._parloop._args):
            if arg._is_indirect:
                if _d.has_key((arg._dat, arg._map)):
                    _ind_desc[i] = _d[(arg._dat, arg._map)]
                else:
                    _ind_desc[i] = _c
                    _d[(arg._dat, arg._map)] = _c
                    _c += 1
        del _c
        del _d

        _off = [0] * (self._core_plan.ninds + 1)
        for i in range(self._core_plan.ninds):
            _c = 0
            for idesc in _ind_desc:
                if idesc == i:
                    _c += 1
            _off[i+1] = _off[i] + _c

        self._ind_map_buffers = [None] * self._core_plan.ninds
        for i in range(self._core_plan.ninds):
            self._ind_map_buffers[i] = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=int(np.int32(0).itemsize * (_off[i+1] - _off[i]) * self._parloop._it_space.size))
            s = self._parloop._it_space.size * _off[i]
            e = s + (_off[i+1] - _off[i]) * self._parloop._it_space.size
            cl.enqueue_copy(_queue, self._ind_map_buffers[i], self._core_plan.ind_map[s:e], is_blocking=True).wait()

        self._loc_map_buffers = [None] * self.nuinds
        for i in range(self.nuinds):
            self._loc_map_buffers[i] = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=int(np.int16(0).itemsize * self._parloop._it_space.size))
            s = i * self._parloop._it_space.size
            e = s + self._parloop._it_space.size
            cl.enqueue_copy(_queue, self._loc_map_buffers[i], self._core_plan.loc_map[s:e], is_blocking=True).wait()

        self._ind_sizes_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.ind_sizes.nbytes)
        cl.enqueue_copy(_queue, self._ind_sizes_buffer, self._core_plan.ind_sizes, is_blocking=True).wait()

        self._ind_offs_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.ind_offs.nbytes)
        cl.enqueue_copy(_queue, self._ind_offs_buffer, self._core_plan.ind_offs, is_blocking=True).wait()

        self._blkmap_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.blkmap.nbytes)
        cl.enqueue_copy(_queue, self._blkmap_buffer, self._core_plan.blkmap, is_blocking=True).wait()

        self._offset_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.offset.nbytes)
        cl.enqueue_copy(_queue, self._offset_buffer, self._core_plan.offset, is_blocking=True).wait()

        self._nelems_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.nelems.nbytes)
        cl.enqueue_copy(_queue, self._nelems_buffer, self._core_plan.nelems, is_blocking=True).wait()

        self._nthrcol_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.nthrcol.nbytes)
        cl.enqueue_copy(_queue, self._nthrcol_buffer, self._core_plan.nthrcol, is_blocking=True).wait()

        self._thrcol_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.thrcol.nbytes)
        cl.enqueue_copy(_queue, self._thrcol_buffer, self._core_plan.thrcol, is_blocking=True).wait()

        if _debug:
            print 'plan ind_map ' + str(self._core_plan.ind_map)
            print 'plan loc_map ' + str(self._core_plan.loc_map)
            print '_ind_desc ' + str(_ind_desc)
            print 'nuinds %d' % self.nuinds
            print 'ninds %d' % self.ninds
            print '_off ' + str(_off)
            for i in range(self.ninds):
                print 'ind_map[' + str(i) + '] = ' + str(self.ind_map[s:e])
                pass
            for i in range(self.nuinds):
                print 'loc_map[' + str(i) + '] = ' + str(self.loc_map[s:e])
                pass
            print 'ind_sizes :' + str(self.ind_sizes)
            print 'ind_offs :' + str(self.ind_offs)
            print 'blk_map :' + str(self.blkmap)
            print 'offset :' + str(self.offset)
            print 'nelems :' + str(self.nelems)
            print 'nthrcol :' + str(self.nthrcol)
            print 'thrcol :' + str(self.thrcol)

    @property
    def nshared(self):
        return self._core_plan.nshared

    @property
    def ninds(self):
        return self._core_plan.ninds

    @property
    def ncolors(self):
        return self._core_plan.ncolors

    @property
    def ncolblk(self):
        return self._core_plan.ncolblk

    @property
    def nblocks(self):
        return self._core_plan.nblocks

class DatMapPair(object):
    """ Dummy class needed for codegen
        (could do without but would obfuscate codegen templates)
    """
    def __init__(self, dat, map):
        self._dat = dat
        self._map = map

    def __hash__(self):
        return hash(self._dat) ^ hash(self._map)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class ParLoopCall(object):
    """Invocation of an OP2 OpenCL kernel with an access descriptor"""

    def __init__(self, kernel, it_space, *args):
        self._kernel = kernel
        self._it_space = it_space
        self._actual_args = list(args)

        self._args = list()
        for a in self._actual_args:
            if a._is_vec_map:
                for i in range(a._map._dim):
                    self._args.append(Arg(a._dat, a._map, i, a._access))
            else:
                self._args.append(a)

    # generic

    @property
    def _global_reduction_args(self):
        return uniquify(a for a in self._args if a._is_global_reduction)

    @property
    def _global_non_reduction_args(self):
        return uniquify(a for a in self._args if a._is_global and not a._is_global_reduction)

    @property
    def _unique_dats(self):
        return uniquify(a._dat for a in self._args if a._is_dat)

    @property
    def _indirect_reduc_args(self):
        return uniquify(a for a in self._args if a._is_indirect_reduction)

    # code generation specific

    @property
    def _direct_args(self):
        return uniquify(a for a in self._args if a._is_direct)

    @property
    def _direct_non_scalar_args(self):
        return [a for a in self._direct_args if not a._dat._is_scalar]

    @property
    def _direct_non_scalar_read_args(self):
        return [a for a in self._direct_non_scalar_args if a._access in [READ, RW]]

    @property
    def _direct_non_scalar_written_args(self):
        return [a for a in self._direct_non_scalar_args if a._access in [WRITE, RW]]

    def _d_max_dynamic_shared_memory(self):
        """Computes the maximum shared memory requirement per iteration set elements."""
        assert self.is_direct(), "Should only be called on direct loops"
        if self._direct_non_scalar_args:
            staging = max(a._dat.bytes_per_elem for a in self._direct_non_scalar_args)
        else:
            staging = 0

        if self._global_reduction_args:
            reduction = max(a._dat._data.itemsize for a in self._global_reduction_args)
        else:
            reduction = 0

        return max(staging, reduction)

    @property
    def _indirect_args(self):
        return [a for a in self._args if a._is_indirect]

    @property
    def _vec_map_args(self):
        return [a for a in self._actual_args if a._is_vec_map]

    @property
    def _dat_map_pairs(self):
        return uniquify(DatMapPair(a._dat, a._map) for a in self._indirect_args)

    @property
    def _nonreduc_vec_dat_map_pairs(self):
        return uniquify(DatMapPair(a._dat, a._map) for a in self._vec_map_args if a._access is not INC)

    @property
    def _reduc_vec_dat_map_pairs(self):
        return uniquify(DatMapPair(a._dat, a._map) for a in self._vec_map_args if a._access is INC)

    @property
    def _read_dat_map_pairs(self):
        return uniquify(DatMapPair(a._dat, a._map) for a in self._indirect_args if a._access in [READ, RW])

    @property
    def _written_dat_map_pairs(self):
        return uniquify(DatMapPair(a._dat, a._map) for a in self._indirect_args if a._access in [WRITE, RW])

    @property
    def _indirect_reduc_dat_map_pairs(self):
        return uniquify(DatMapPair(a._dat, a._map) for a in self._args if a._is_indirect_reduction)

    def compute(self):
        source, prg = _gen_code_cache.get_code(self._kernel)

        if self.is_direct():
            per_elem_max_local_mem_req = self._d_max_dynamic_shared_memory()
            shared_memory_offset = per_elem_max_local_mem_req * _warpsize
            if per_elem_max_local_mem_req == 0:
                wgs = _queue.device.max_work_group_size
            else:
                warnings.warn('temporary fix to available local memory computation (-512)')
                available_local_memory = _queue.device.local_mem_size - 512
                # 16bytes local mem used for global / local indices and sizes
                available_local_memory -= 16
                # (4/8)ptr bytes for each dat buffer passed to the kernel
                available_local_memory -= (len(self._unique_dats) + len(self._global_non_reduction_args))\
                                          * (_queue.device.address_bits / 8)
                # (4/8)ptr bytes for each temporary global reduction buffer passed to the kernel
                available_local_memory -= len(self._global_reduction_args) * (_queue.device.address_bits / 8)
                # 7: 7bytes potentialy lost for aligning the shared memory buffer to 'long'
                available_local_memory -= 7
                ps = available_local_memory / per_elem_max_local_mem_req
                wgs = min(_queue.device.max_work_group_size, (ps / _warpsize) * _warpsize)
            nwg = min(_pref_work_group_count, int(math.ceil(self._it_space.size / float(wgs))))
            ttc = wgs * nwg

            local_memory_req = per_elem_max_local_mem_req * wgs

            if not source:
                inst = []
                for i, arg in enumerate(self._args):
                    if arg._is_direct and arg._dat._is_scalar:
                        inst.append(("__global", None))
                    elif arg._is_direct:
                        inst.append(("__private", None))
                    elif arg._is_global_reduction:
                        inst.append(("__private", None))
                    elif arg._is_global:
                        inst.append(("__global", None))

                self._kernel.instrument(inst)

                dloop = _stg_direct_loop.getInstanceOf("direct_loop")
                dloop['parloop'] = self
                dloop['const'] = {"warpsize": _warpsize,\
                                  "shared_memory_offset": shared_memory_offset,\
                                  "dynamic_shared_memory_size": local_memory_req,\
                                  "threads_per_block": wgs,
                                  "block_count": nwg}
                dloop['op2const'] = list(Const._defs)
                source = str(dloop)

                # for debugging purpose, refactor that properly at some point
                if _kernel_dump:
                    f = open(self._kernel._name + '.cl.c', 'w')
                    f.write(source)
                    f.close

                prg = cl.Program (_ctx, source).build(options="-Werror")
                _gen_code_cache.cache_code(self._kernel, (source, prg))

            kernel = prg.__getattr__(self._kernel._name + '_stub')

            for a in self._unique_dats:
                kernel.append_arg(a._buffer)

            for a in self._global_reduction_args:
                a._dat._allocate_reduction_array(nwg)
                kernel.append_arg(a._dat._d_reduc_buffer)

            for a in self._global_non_reduction_args:
                kernel.append_arg(a._dat._buffer)

            kernel.append_arg(np.int32(self._it_space.size))

            cl.enqueue_nd_range_kernel(_queue, kernel, (int(ttc),), (int(wgs),), g_times_l=False).wait()
            for i, a in enumerate(self._global_reduction_args):
                a._dat._post_kernel_reduction_task(nwg)
        else:
            psize = self._i_compute_partition_size()
            plan = _plan_cache.get_plan(self, partition_size=psize)

            if not source:
                inst = []
                for i, arg in enumerate(self._actual_args):
                    if arg._map == IdentityMap:
                        inst.append(("__global", None))
                    elif arg._is_vec_map and arg._is_indirect_reduction:
                        inst.append(("__private", None))
                    elif arg._is_vec_map and not arg._is_indirect_reduction:
                        inst.append(("__local", None))
                    elif isinstance(arg._dat, Dat) and arg._access not in [INC, MIN, MAX]:
                        inst.append(("__local", None))
                    elif arg._is_global and not arg._is_global_reduction:
                        inst.append(("__global", None))
                    else:
                        inst.append(("__private", None))

                self._kernel.instrument(inst)

                # codegen
                iloop = _stg_indirect_loop.getInstanceOf("indirect_loop")
                iloop['parloop'] = self
                iloop['const'] = {'dynamic_shared_memory_size': plan.nshared,\
                                  'ninds':plan.ninds,\
                                  'block_count': 'dynamic',\
                                  'threads_per_block': min(_max_work_group_size, psize),\
                                  'partition_size':psize,\
                                  'warpsize': _warpsize}
                iloop['op2const'] = list(Const._defs)
                source = str(iloop)

                # for debugging purpose, refactor that properly at some point
                if _kernel_dump:
                    f = open(self._kernel._name + '.cl.c', 'w')
                    f.write(source)
                    f.close

                prg = cl.Program(_ctx, source).build(options="-Werror")

                _gen_code_cache.cache_code(self._kernel, (source, prg))


            kernel = prg.__getattr__(self._kernel._name + '_stub')

            for a in self._unique_dats:
                kernel.append_arg(a._buffer)

            for a in self._global_non_reduction_args:
                kernel.append_arg(a._dat._buffer)

            for i in range(plan.ninds):
                kernel.append_arg(plan._ind_map_buffers[i])

            for i in range(plan.nuinds):
                kernel.append_arg(plan._loc_map_buffers[i])

            for arg in self._global_reduction_args:
                arg._dat._allocate_reduction_array(plan.nblocks)
                kernel.append_arg(arg._dat._d_reduc_buffer)

            kernel.append_arg(plan._ind_sizes_buffer)
            kernel.append_arg(plan._ind_offs_buffer)
            kernel.append_arg(plan._blkmap_buffer)
            kernel.append_arg(plan._offset_buffer)
            kernel.append_arg(plan._nelems_buffer)
            kernel.append_arg(plan._nthrcol_buffer)
            kernel.append_arg(plan._thrcol_buffer)

            block_offset = 0
            for i in range(plan.ncolors):
                blocks_per_grid = int(plan.ncolblk[i])
                threads_per_block = min(_max_work_group_size, psize)
                thread_count = threads_per_block * blocks_per_grid

                kernel.set_last_arg(np.int32(block_offset))
                cl.enqueue_nd_range_kernel(_queue, kernel, (int(thread_count),), (int(threads_per_block),), g_times_l=False).wait()
                block_offset += blocks_per_grid

            for arg in self._global_reduction_args:
                arg._dat._post_kernel_reduction_task(plan.nblocks)

    def is_direct(self):
        return all(map(lambda a: a._is_direct or isinstance(a._dat, Global), self._args))

    def _i_compute_partition_size(self):
        staged_args = filter(lambda a: a._map != IdentityMap, self._args)
        assert staged_args
        # will have to fix for vec dat
        #TODO FIX: something weird here
        #available_local_memory
        warnings.warn('temporary fix to available local memory computation (-512)')
        available_local_memory = _max_local_memory - 512
        # 16bytes local mem used for global / local indices and sizes
        available_local_memory -= 16
        # (4/8)ptr size per dat passed as argument (dat)
        available_local_memory -= (_queue.device.address_bits / 8) * (len(self._unique_dats) + len(self._global_non_reduction_args))
        # (4/8)ptr size per dat/map pair passed as argument (ind_map)
        available_local_memory -= (_queue.device.address_bits / 8) * len(self._dat_map_pairs)
        # (4/8)ptr size per global reduction temp array
        available_local_memory -= (_queue.device.address_bits / 8) * len(self._global_reduction_args)
        # (4/8)ptr size per indirect arg (loc_map)
        available_local_memory -= (_queue.device.address_bits / 8) * len(filter(lambda a: not a._is_indirect, self._args))
        # (4/8)ptr size * 7: for plan objects
        available_local_memory -= (_queue.device.address_bits / 8) * 7
        # 1 uint value for block offset
        available_local_memory -= 4
        # 7: 7bytes potentialy lost for aligning the shared memory buffer to 'long'
        available_local_memory -= 7
        # 12: shared_memory_offset, active_thread_count, active_thread_count_ceiling variables (could be 8 or 12 depending)
        #     and 3 for potential padding after shared mem buffer
        available_local_memory -= 12 + 3
        # 2 * (4/8)ptr size + 1uint32: DAT_via_MAP_indirection(./_size/_map) per dat map pairs
        available_local_memory -= 4 + (_queue.device.address_bits / 8) * 2 * len(self._dat_map_pairs)
        # inside shared memory padding
        available_local_memory -= 2 * (len(self._dat_map_pairs) - 1)

        max_bytes = sum(map(lambda a: a._dat.bytes_per_elem, staged_args))
        return available_local_memory / (2 * _warpsize * max_bytes) * (2 * _warpsize)

#Monkey patch pyopencl.Kernel for convenience
_original_clKernel = cl.Kernel

class CLKernel (_original_clKernel):
    def __init__(self, *args, **kargs):
        super(CLKernel, self).__init__(*args, **kargs)
        self._karg = 0

    def reset_args(self):
        self._karg = 0;

    def append_arg(self, arg):
        self.set_arg(self._karg, arg)
        self._karg += 1

    def set_last_arg(self, arg):
        self.set_arg(self._karg, arg)

cl.Kernel = CLKernel

def par_loop(kernel, it_space, *args):
    ParLoopCall(kernel, it_space, *args).compute()

_debug = False
_kernel_dump = False
_ctx = cl.create_some_context()
_queue = cl.CommandQueue(_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
_pref_work_group_count = _queue.device.max_compute_units
_max_local_memory = _queue.device.local_mem_size
_address_bits = _queue.device.address_bits
_max_work_group_size = _queue.device.max_work_group_size
_has_dpfloat = 'cl_khr_fp64' in _queue.device.extensions or 'cl_amd_fp64' in _queue.device.extensions

# CPU
if _queue.device.type == cl.device_type.CPU:
    _warpsize = 1
# GPU
elif _queue.device.type == cl.device_type.GPU:
    # assumes nvidia, will probably fail with AMD gpus
    _warpsize = 32

if not _has_dpfloat:
    warnings.warn('device does not support double precision floating point computation, expect undefined behavior for double')

_stg_direct_loop = stringtemplate3.StringTemplateGroup(file=stringtemplate3.StringIO(pkg_resources.resource_string(__name__, "assets/opencl_direct_loop.stg")), lexer="default")
_stg_indirect_loop = stringtemplate3.StringTemplateGroup(file=stringtemplate3.StringIO(pkg_resources.resource_string(__name__, "assets/opencl_indirect_loop.stg")), lexer="default")

_plan_cache = OpPlanCache()
_gen_code_cache = GenCodeCache()
