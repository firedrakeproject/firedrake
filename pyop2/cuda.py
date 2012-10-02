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

import runtime_base as op2
import numpy as np
from runtime_base import Set, IterationSpace, Sparsity
from utils import verify_reshape
import jinja2
import op_lib_core as core
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

class Kernel(op2.Kernel):
    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)
        self._code = "__device__ %s" % self._code

class Arg(op2.Arg):
    @property
    def _d_is_staged(self):
        return self._is_direct and not (self.data._is_scalar or self._is_soa)

    def _indirect_kernel_arg_name(self, idx):
        name = self.data.name
        if self._is_global:
            if self._is_global_reduction:
                return "%s_l" % name
            else:
                return name
        if self._is_direct:
            if self.data.soa:
                return "%s + (%s + offset_b)" % (name, idx)
            return "%s + (%s + offset_b) * %s" % (name, idx, self.data.cdim)
        if self._is_indirect:
            if self._is_vec_map:
                return "%s_vec" % name
            if self.access is op2.INC:
                return "%s%s_l" % (name, self.idx)
            else:
                return "%s_s + loc_map[%s * set_size + %s + offset_b]*%s" \
                    % (name, self._which_indirect, idx, self.data.cdim)


    def _kernel_arg_name(self, idx=None):
        name = self.data.name
        if self._d_is_staged:
            return "%s_local" % name
        elif self._is_global_reduction:
            return "%s_reduc_local" % name
        elif self._is_global:
            return name
        else:
            return "%s + %s" % (name, idx)

class DeviceDataMixin(object):
    UNALLOCATED = 0             # device_data is not yet allocated
    GPU = 1                     # device_data is valid, data is invalid
    CPU = 2                     # device_data is allocated, but invalid
    BOTH = 3                    # device_data and data are both valid

    @property
    def bytes_per_elem(self):
        return self.dtype.itemsize * self.cdim
    @property
    def state(self):
        return self._state
    @state.setter
    def state(self, value):
        self._state = value

    def _allocate_device(self):
        if self.state is DeviceDataMixin.UNALLOCATED:
            if self.soa:
                shape = self._data.T.shape
            else:
                shape = self._data.shape
            self._device_data = gpuarray.empty(shape=shape, dtype=self.dtype)
            self.state = DeviceDataMixin.CPU

    def _to_device(self):
        self._allocate_device()
        if self.state is DeviceDataMixin.CPU:
            if self.soa:
                shape = self._device_data.shape
                tmp = self._data.T.ravel().reshape(shape)
            else:
                tmp = self._data
            self._device_data.set(tmp)
        self.state = DeviceDataMixin.BOTH

    def _from_device(self):
        if self.state is DeviceDataMixin.GPU:
            self._device_data.get(self._data)
            if self.soa:
                shape = self._data.T.shape
                self._data = self._data.reshape(shape).T
                print self._data
            self.state = DeviceDataMixin.BOTH

    @property
    def data(self):
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Dat!")
        self._from_device()
        if self.state is not DeviceDataMixin.UNALLOCATED:
            self.state = DeviceDataMixin.CPU
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        if self.state is not DeviceDataMixin.UNALLOCATED:
            self.state = DeviceDataMixin.CPU

class Dat(DeviceDataMixin, op2.Dat):

    _arg_type = Arg

    @property
    def _is_scalar(self):
        return self.cdim == 1

    def __init__(self, dataset, dim, data=None, dtype=None, name=None, soa=None):
        op2.Dat.__init__(self, dataset, dim, data, dtype, name, soa)
        self.state = DeviceDataMixin.UNALLOCATED

class Mat(DeviceDataMixin, op2.Mat):

    _arg_type = Arg

    def __init__(self, datasets, dtype=None, name=None):
        op2.Mat.__init__(self, datasets, dtype, name)
        self.state = DeviceDataMixin.UNALLOCATED

class Const(DeviceDataMixin, op2.Const):

    _arg_type = Arg

    def __init__(self, dim, data, name, dtype=None):
        op2.Const.__init__(self, dim, data, name, dtype)
        self.state = DeviceDataMixin.CPU

    @property
    def data(self):
        self.state = DeviceDataMixin.CPU
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        self.state = DeviceDataMixin.CPU

    def _format_declaration(self):
        d = {'dim' : self.cdim,
             'type' : self.ctype,
             'name' : self.name}

        if self.cdim == 1:
            return "__constant__ %(type)s %(name)s;" % d
        return "__constant__ %(type)s %(name)s[%(dim)s];" % d

    def _to_device(self, module):
        ptr, size = module.get_global(self.name)
        if size != self.data.nbytes:
            raise RuntimeError("Const %s needs %d bytes, but only space for %d" % (self, self.data.nbytes, size))
        if self.state is DeviceDataMixin.CPU:
            driver.memcpy_htod(ptr, self._data)
            self.state = DeviceDataMixin.BOTH

    def _from_device(self):
        raise RuntimeError("Copying Const %s from device makes no sense" % self)

class Global(DeviceDataMixin, op2.Global):

    _arg_type = Arg

    def __init__(self, dim, data, dtype=None, name=None):
        op2.Global.__init__(self, dim, data, dtype, name)
        self.state = DeviceDataMixin.UNALLOCATED
        self._reduction_buffer = None
        self._host_reduction_buffer = None

    def _allocate_reduction_buffer(self, grid_size, op):
        if self._reduction_buffer is None:
            self._host_reduction_buffer = np.zeros(np.prod(grid_size) * self.cdim,
                                                   dtype=self.dtype).reshape((-1,)+self._dim)
            if op is not op2.INC:
                self._host_reduction_buffer[:] = self._data
            self._reduction_buffer = gpuarray.to_gpu(self._host_reduction_buffer)

    @property
    def soa(self):
        return False

    @property
    def data(self):
        if self.state is not DeviceDataMixin.UNALLOCATED:
            self.state = DeviceDataMixin.CPU
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        if self.state is not DeviceDataMixin.UNALLOCATED:
            self.state = DeviceDataMixin.CPU

    def _finalise_reduction_begin(self, grid_size, op):
        self._stream = driver.Stream()
        driver.memcpy_dtoh_async(self._host_reduction_buffer,
                                 self._reduction_buffer.ptr,
                                 self._stream)
    def _finalise_reduction_end(self, grid_size, op):
        self.state = DeviceDataMixin.CPU
        self._stream.synchronize()
        del self._stream
        tmp = self._host_reduction_buffer
        if op is op2.MIN:
            tmp = np.min(tmp, axis=0)
            fn = min
        elif op is op2.MAX:
            tmp = np.max(tmp, axis=0)
            fn = max
        else:
            tmp = np.sum(tmp, axis=0)
        for i in range(self.cdim):
            if op is op2.INC:
                self._data[i] += tmp[i]
            else:
                self._data[i] = fn(self._data[i], tmp[i])

class Map(op2.Map):

    _arg_type = Arg

    def __init__(self, iterset, dataset, dim, values, name=None):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)
        self._device_values = None

    def _to_device(self):
        if self._device_values is None:
            self._device_values = gpuarray.to_gpu(self._values)
        else:
            from warnings import warn
            warn("Copying Map data for %s again, do you really want to do this?" % \
                 self)
            self._device_values.set(self._values)

    def _from_device(self):
        if self._device_values is None:
            raise RuntimeError("No values for Map %s on device" % self)
        self._device_values.get(self._values)

_plan_cache = dict()

def empty_plan_cache():
    _plan_cache.clear()

def ncached_plans():
    return len(_plan_cache)

class Plan(core.op_plan):
    def __new__(cls, kernel, iset, *args, **kwargs):
        ps = kwargs.get('partition_size', 0)
        key = Plan.cache_key(iset, ps, *args)
        cached = _plan_cache.get(key, None)
        if cached is not None:
            return cached
        else:
            return super(Plan, cls).__new__(cls, kernel, iset, *args,
                                            **kwargs)
    def __init__(self, kernel, iset, *args, **kwargs):
        ps = kwargs.get('partition_size', 0)
        key = Plan.cache_key(iset, ps, *args)
        cached = _plan_cache.get(key, None)
        if cached is not None:
            return
        core.op_plan.__init__(self, kernel, iset, *args, **kwargs)
        self._nthrcol = None
        self._thrcol = None
        self._offset = None
        self._ind_map = None
        self._ind_offs = None
        self._ind_sizes = None
        self._loc_map = None
        self._nelems = None
        self._blkmap = None
        _plan_cache[key] = self

    @classmethod
    def cache_key(cls, iset, partition_size, *args):
        # Set size
        key = (iset.size, )
        # Size of partitions (amount of smem)
        key += (partition_size, )

        # For each indirect arg, the map and the indices into the map
        # are important
        inds = {}
        for arg in args:
            if arg._is_indirect:
                dat = arg.data
                map = arg.map
                l = inds.get((dat, map), [])
                l.append(arg.idx)
                inds[(dat, map)] = l

        for k,v in inds.iteritems():
            key += (k[1],) + tuple(sorted(v))

        return key

    @property
    def nthrcol(self):
        if self._nthrcol is None:
            self._nthrcol = gpuarray.to_gpu(super(Plan, self).nthrcol)
        return self._nthrcol

    @property
    def thrcol(self):
        if self._thrcol is None:
            self._thrcol = gpuarray.to_gpu(super(Plan, self).thrcol)
        return self._thrcol

    @property
    def offset(self):
        if self._offset is None:
            self._offset = gpuarray.to_gpu(super(Plan, self).offset)
        return self._offset

    @property
    def ind_map(self):
        if self._ind_map is None:
            self._ind_map = gpuarray.to_gpu(super(Plan, self).ind_map)
        return self._ind_map

    @property
    def ind_offs(self):
        if self._ind_offs is None:
            self._ind_offs = gpuarray.to_gpu(super(Plan, self).ind_offs)
        return self._ind_offs

    @property
    def ind_sizes(self):
        if self._ind_sizes is None:
            self._ind_sizes = gpuarray.to_gpu(super(Plan, self).ind_sizes)
        return self._ind_sizes

    @property
    def loc_map(self):
        if self._loc_map is None:
            self._loc_map = gpuarray.to_gpu(super(Plan, self).loc_map)
        return self._loc_map

    @property
    def nelems(self):
        if self._nelems is None:
            self._nelems = gpuarray.to_gpu(super(Plan, self).nelems)
        return self._nelems

    @property
    def blkmap(self):
        if self._blkmap is None:
            self._blkmap = gpuarray.to_gpu(super(Plan, self).blkmap)
        return self._blkmap

def par_loop(kernel, it_space, *args):
    ParLoop(kernel, it_space, *args).compute()

class ParLoop(op2.ParLoop):
    def __init__(self, kernel, it_space, *args):
        op2.ParLoop.__init__(self, kernel, it_space, *args)
        self._src = None
        self.__unique_args = []
        self._unwound_args = []
        seen = set()
        c = 0
        for arg in self.args:
            if arg._is_vec_map:
                for i in range(arg.map.dim):
                    self._unwound_args.append(arg.data(arg.map[i],
                                                       arg.access))
            elif arg._is_mat:
                pass
            elif arg._uses_itspace:
                for i in range(self._it_space.extents[arg.idx.index]):
                    self._unwound_args.append(arg.data(arg.map[i],
                                                       arg.access))
            else:
                self._unwound_args.append(arg)

            if arg._is_dat:
                k = (arg.data, arg.map)
                if arg._is_indirect:
                    arg._which_indirect = c
                    if arg._is_vec_map:
                        c += arg.map.dim
                    else:
                        c += 1
                if k in seen:
                    pass
                else:
                    self.__unique_args.append(arg)
                    seen.add(k)
            else:
                self.__unique_args.append(arg)

    def __hash__(self):
        """Canonical representation of a parloop wrt generated code caching."""
        # FIXME, make clearer, converge on hashing with opencl code
        def argdimacc(arg):
            if self.is_direct():
                if arg._is_global or (arg._is_dat and not arg.data._is_scalar):
                    return (arg.data.cdim, arg.access)
                else:
                    return ()
            else:
                if (arg._is_global and arg.access is op2.READ) or arg._is_direct:
                    return ()
                else:
                    return (arg.data.cdim, arg.access)

        argdesc = []
        seen = dict()
        c = 0
        for arg in self.args:
            if arg._is_indirect:
                if not seen.has_key((arg.data,arg.map)):
                    seen[(arg.data,arg.map)] = c
                    idesc = (c, (- arg.map.dim) if arg._is_vec_map else arg.idx)
                    c += 1
                else:
                    idesc = (seen[(arg.data,arg.map)], (- arg.map.dim) if arg._is_vec_map else arg.idx)
            else:
                idesc = ()

            d = (arg.data.__class__,
                 arg.data.dtype) + argdimacc(arg) + idesc

            argdesc.append(d)

        hsh = hash(self._kernel)
        hsh ^= hash(self._it_space)
        hsh ^= hash(tuple(argdesc))
        for c in Const._definitions():
            hsh ^= hash(c)

        return hsh

    @property
    def _unique_args(self):
        return self.__unique_args

    @property
    def _unique_vec_map_args(self):
        return [a for a in self._unique_args if a._is_vec_map]

    @property
    def _unique_indirect_dat_args(self):
        return [a for a in self._unique_args if a._is_indirect]

    @property
    def _unique_read_indirect_dat_args(self):
        return [a for a in self._unique_indirect_dat_args \
                if a.access in [op2.READ, op2.RW]]

    @property
    def _unique_written_indirect_dat_args(self):
        return [a for a in self._unique_indirect_dat_args \
                if a.access in [op2.RW, op2.WRITE, op2.INC]]

    @property
    def _vec_map_args(self):
        return [a for a in self.args if a._is_vec_map]

    @property
    def _unique_inc_indirect_dat_args(self):
        return [a for a in self._unique_indirect_dat_args \
                if a.access is op2.INC]

    @property
    def _inc_indirect_dat_args(self):
        return [a for a in self.args if a.access is op2.INC and
                a._is_indirect]

    @property
    def _inc_non_vec_map_indirect_dat_args(self):
        return [a for a in self.args if a.access is op2.INC and
                a._is_indirect and not a._is_vec_map]

    @property
    def _non_inc_vec_map_args(self):
        return [a for a in self._vec_map_args if a.access is not op2.INC]

    @property
    def _inc_vec_map_args(self):
        return [a for a in self._vec_map_args if a.access is op2.INC]

    @property
    def _needs_smem(self):
        if not self.is_direct():
            return True
        for a in self.args:
            if a._is_global_reduction:
                return True
            if not a.data._is_scalar:
                return True
        return False

    @property
    def _global_reduction_args(self):
        return [a for a in self.args if a._is_global_reduction]
    @property
    def _direct_args(self):
        return [a for a in self.args if a._is_direct]

    @property
    def _direct_non_scalar_args(self):
        return [a for a in self._direct_args if not (a.data._is_scalar or a._is_soa)]

    @property
    def _direct_non_scalar_read_args(self):
        return [a for a in self._direct_non_scalar_args if a.access is not op2.WRITE]

    @property
    def _direct_non_scalar_written_args(self):
        return [a for a in self._direct_non_scalar_args if a.access is not op2.READ]

    @property
    def _stub_name(self):
        return "__%s_stub" % self.kernel.name

    def is_direct(self):
        return all([a._is_direct or a._is_global for a in self.args])

    def device_function(self):
        return self._module.get_function(self._stub_name)

    def compile(self, config=None):

        self._module, self._fun = op2._parloop_cache.get(hash(self),
                                                         (None, None))
        if self._module is not None:
            return
        if self.is_direct():
            self.generate_direct_loop(config)
            self._module = SourceModule(self._src, options=['-O3', '--use_fast_math'])
            self._fun = self.device_function()
            argtypes = np.dtype('int32').char
            for arg in self.args:
                argtypes += "P"
            self._fun.prepare(argtypes)
            op2._parloop_cache[hash(self)] = self._module, self._fun
        else:
            self.generate_indirect_loop()
            self._module = SourceModule(self._src, options=['-O3', '--use_fast_math'])
            self._fun = self.device_function()
            argtypes = np.dtype('int32').char
            for arg in self._unique_args:
                argtypes += "P"
            itype = np.dtype('int32').char
            argtypes += "PPPP"
            argtypes += itype
            argtypes += "PPPPP"
            argtypes += itype
            self._fun.prepare(argtypes)
            op2._parloop_cache[hash(self)] = self._module, self._fun

    def _max_smem_per_elem_direct(self):
        m_stage = 0
        m_reduc = 0
        if self._direct_non_scalar_args:
            m_stage = max(a.data.bytes_per_elem for a in self._direct_non_scalar_args)
        if self._global_reduction_args:
            m_reduc = max(a.dtype.itemsize for a in self._global_reduction_args)
        return max(m_stage, m_reduc)

    def launch_configuration(self):
        if self.is_direct():
            max_smem = self._max_smem_per_elem_direct()
            smem_offset = max_smem * _WARPSIZE
            max_block = _device.get_attribute(driver.device_attribute.MAX_BLOCK_DIM_X)
            if max_smem == 0:
                block_size = max_block
            else:
                available_smem = _device.get_attribute(driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
                threads_per_sm = available_smem / max_smem
                block_size = min(max_block, (threads_per_sm / _WARPSIZE) * _WARPSIZE)
            max_grid = _device.get_attribute(driver.device_attribute.MAX_GRID_DIM_X)
            grid_size = min(max_grid, (block_size + self._it_space.size) / block_size)

            grid_size = np.asscalar(np.int64(grid_size))
            block_size = (block_size, 1, 1)
            grid_size = (grid_size, 1, 1)

            required_smem = np.asscalar(max_smem * np.prod(block_size))
            return {'smem_offset' : smem_offset,
                    'WARPSIZE' : _WARPSIZE,
                    'required_smem' : required_smem,
                    'block_size' : block_size,
                    'grid_size' : grid_size}

    def generate_direct_loop(self, config):
        if self._src is not None:
            return
        d = {'parloop' : self,
             'launch' : config,
             'constants' : Const._definitions()}
        self._src = _direct_loop_template.render(d).encode('ascii')

    def generate_indirect_loop(self):
        if self._src is not None:
            return
        config = {'WARPSIZE': 32}
        d = {'parloop' : self,
             'launch' : config,
             'constants' : Const._definitions()}
        self._src = _indirect_loop_template.render(d).encode('ascii')

    def compute(self):
        if self._has_soa:
            op2stride = Const(1, self._it_space.size, name='op2stride',
                              dtype='int32')
        arglist = [np.int32(self._it_space.size)]
        if self.is_direct():
            config = self.launch_configuration()
            self.compile(config=config)
            block_size = config['block_size']
            grid_size = config['grid_size']
            shared_size = config['required_smem']
            for c in Const._definitions():
                c._to_device(self._module)
            for arg in self.args:
                arg.data._allocate_device()
                if arg.access is not op2.WRITE:
                    arg.data._to_device()
                karg = arg.data._device_data
                if arg._is_global_reduction:
                    arg.data._allocate_reduction_buffer(grid_size, arg.access)
                    karg = arg.data._reduction_buffer
                arglist.append(np.intp(karg.gpudata))
            self._fun.prepared_call(grid_size, block_size, *arglist,
                                    shared_size=shared_size)
            for arg in self.args:
                if arg._is_global_reduction:
                    arg.data._finalise_reduction_begin(grid_size, arg.access)
                    arg.data._finalise_reduction_end(grid_size, arg.access)
                if arg.access is not op2.READ:
                    arg.data.state = DeviceDataMixin.GPU
        else:
            self.compile()
            maxbytes = sum([a.dtype.itemsize * a.data.cdim for a in self._unwound_args \
                                if a._is_indirect])
            part_size = ((47 * 1024) / (64 * maxbytes)) * 64
            self._plan = Plan(self.kernel, self._it_space.iterset,
                              *self._unwound_args,
                              partition_size=part_size)
            max_grid_size = self._plan.ncolblk.max()
            for c in Const._definitions():
                c._to_device(self._module)
            for arg in self._unique_args:
                arg.data._allocate_device()
                if arg.access is not op2.WRITE:
                    arg.data._to_device()
                karg = arg.data._device_data
                if arg._is_global_reduction:
                    arg.data._allocate_reduction_buffer(max_grid_size,
                                                        arg.access)
                    karg = arg.data._reduction_buffer
                arglist.append(karg.gpudata)
            arglist.append(self._plan.ind_map.gpudata)
            arglist.append(self._plan.loc_map.gpudata)
            arglist.append(self._plan.ind_sizes.gpudata)
            arglist.append(self._plan.ind_offs.gpudata)
            arglist.append(None) # Block offset
            arglist.append(self._plan.blkmap.gpudata)
            arglist.append(self._plan.offset.gpudata)
            arglist.append(self._plan.nelems.gpudata)
            arglist.append(self._plan.nthrcol.gpudata)
            arglist.append(self._plan.thrcol.gpudata)
            arglist.append(None) # Number of colours in this block
            block_offset = 0
            for col in xrange(self._plan.ncolors):
                # if col == self._plan.ncolors_core: wait for mpi

                blocks = self._plan.ncolblk[col]
                if blocks <= 0:
                    continue

                arglist[-1] = np.int32(blocks)
                arglist[-7] = np.int32(block_offset)
                blocks = np.asscalar(blocks)
                if blocks >= 2**16:
                    grid_size = (2**16 - 1, (blocks - 1)/(2**16-1) + 1, 1)
                else:
                    grid_size = (blocks, 1, 1)

                block_size = (128, 1, 1)
                shared_size = np.asscalar(self._plan.nsharedCol[col])

                self._fun.prepared_call(grid_size, block_size, *arglist,
                                        shared_size=shared_size)

                if col == self._plan.ncolors_owned - 1:
                    for arg in self.args:
                        if arg._is_global_reduction:
                            arg.data._finalise_reduction_begin(max_grid_size,
                                                               arg.access)
                block_offset += blocks
            for arg in self.args:
                if arg._is_global_reduction:
                    arg.data._finalise_reduction_end(max_grid_size,
                                                     arg.access)
                if arg.access is not op2.READ:
                    arg.data.state = DeviceDataMixin.GPU
        if self._has_soa:
            op2stride.remove_from_namespace()

_device = None
_context = None
_WARPSIZE = 32
_direct_loop_template = None
_indirect_loop_template = None

def _setup():
    global _device
    global _context
    global _WARPSIZE
    if _device is None or _context is None:
        import pycuda.autoinit
        _device = pycuda.autoinit.device
        _context = pycuda.autoinit.context
        _WARPSIZE=_device.get_attribute(driver.device_attribute.WARP_SIZE)
        pass
    global _direct_loop_template
    global _indirect_loop_template
    env = jinja2.Environment(loader=jinja2.PackageLoader('pyop2', 'assets'))
    if _direct_loop_template is None:
        _direct_loop_template = env.get_template('cuda_direct_loop.jinja2')

    if _indirect_loop_template is None:
        _indirect_loop_template = env.get_template('cuda_indirect_loop.jinja2')
