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

from device import *
import device as op2
import numpy as np
from utils import verify_reshape, maybe_setflags
import jinja2
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

class Kernel(op2.Kernel):
    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)
        self._code = "__device__ %s" % self._code

class Arg(op2.Arg):
    def _indirect_kernel_arg_name(self, idx):
        if self._is_mat:
            rmap = self.map[0]
            ridx = self.idx[0]
            cmap = self.map[1]
            cidx = self.idx[1]
            esize = np.prod(self.data.dims)
            size = esize * rmap.dim * cmap.dim
            d = {'n' : self._name,
                 'offset' : self._lmaoffset_name,
                 'idx' : idx,
                 't' : self.ctype,
                 'size' : size,
                 '0' : ridx.index,
                 '1' : cidx.index,
                 'lcdim' : self.data.dims[1],
                 'roff' : cmap.dim * esize,
                 'coff' : esize}
            # We walk through the lma-data in order of the
            # alphabet:
            #  A B C
            #  D E F
            #  G H I
            #       J K
            #       L M
            #  where each sub-block is walked in the same order:
            #  A1 A2
            #  A3 A4
            return """(%(t)s (*)[%(lcdim)s])(%(n)s + %(offset)s +
            (ele_offset + %(idx)s) * %(size)s +
            i%(0)s * %(roff)s + i%(1)s * %(coff)s)""" % d
        if self._is_global:
            if self._is_global_reduction:
                return self._reduction_local_name
            else:
                return self._name
        if self._is_direct:
            if self.data.soa:
                return "%s + (%s + offset_b)" % (self._name, idx)
            return "%s + (%s + offset_b) * %s" % (self._name, idx,
                                                  self.data.cdim)
        if self._is_indirect:
            if self._is_vec_map:
                return self._vec_name
            if self._uses_itspace:
                if self.access is op2.INC:
                    return "%s[i%s]" % (self._vec_name, self.idx.index)
                return "%s + loc_map[(%s+i%s) * set_size + %s + offset_b]*%s" \
                    % (self._shared_name, self._which_indirect,
                       self.idx.index, idx, self.data.cdim)
            if self.access is op2.INC:
                return self._local_name()
            else:
                return "%s + loc_map[%s * set_size + %s + offset_b]*%s" \
                    % (self._shared_name, self._which_indirect, idx,
                       self.data.cdim)

    def _direct_kernel_arg_name(self, idx=None):
        if self._is_staged_direct:
            return self._local_name()
        elif self._is_global_reduction:
            return self._reduction_local_name
        elif self._is_global:
            return self._name
        else:
            return "%s + %s" % (self._name, idx)

class DeviceDataMixin(op2.DeviceDataMixin):
    def _allocate_device(self):
        if self.state is DeviceDataMixin.DEVICE_UNALLOCATED:
            if self.soa:
                shape = self._data.T.shape
            else:
                shape = self._data.shape
            self._device_data = gpuarray.empty(shape=shape, dtype=self.dtype)
            self.state = DeviceDataMixin.HOST

    def _to_device(self):
        self._allocate_device()
        if self.state is DeviceDataMixin.HOST:
            self._device_data.set(self._maybe_to_soa(self._data))
            self.state = DeviceDataMixin.BOTH

    def _from_device(self):
        if self.state is DeviceDataMixin.DEVICE:
            self._device_data.get(self._data)
            self._data = self._maybe_to_aos(self._data)
            self.state = DeviceDataMixin.BOTH

class Dat(DeviceDataMixin, op2.Dat):
    _arg_type = Arg

    @property
    def norm(self):
        """The L2-norm on the flattened vector."""
        return np.sqrt(gpuarray.dot(self.array, self.array).get())

class Mat(DeviceDataMixin, op2.Mat):
    _arg_type = Arg

    def _assemble(self):
        from warnings import warn
        warn("Conversion from LMA to CSR not yet implemented")

    @property
    def _lmadata(self):
        if not hasattr(self, '__lmadata'):
            nentries = 0
            # dense block of rmap.dim x cmap.dim for each rmap/cmap
            # pair
            for rmap, cmap in self.sparsity.maps:
                nentries += rmap.dim * cmap.dim

            entry_size = 0
            # all pairs of maps in the sparsity must have the same
            # iterset, there are sum(iterset.size) * nentries total
            # entries in the LMA data
            for rmap, cmap in self.sparsity.maps:
                entry_size += rmap.iterset.size
            # each entry in the block is size dims[0] x dims[1]
            entry_size *= np.asscalar(np.prod(self.dims))
            nentries *= entry_size
            setattr(self, '__lmadata',
                    gpuarray.zeros(shape=nentries, dtype=self.dtype))
        return getattr(self, '__lmadata')

    def _lmaoffset(self, iterset):
        offset = 0
        size = self.sparsity.maps[0][0].dataset.size
        size *= np.asscalar(np.prod(self.dims))
        for rmap, cmap in self.sparsity.maps:
            if rmap.iterset is iterset:
                break
            offset += rmap.dim * cmap.dim
        return offset * size

    @property
    def _rowptr(self):
        if not hasattr(self, '__rowptr'):
            setattr(self, '__rowptr',
                    gpuarray.to_device(self._sparsity._c_handle.rowptr))
        return getattr(self, '__rowptr')

    @property
    def _colidx(self):
        if not hasattr(self, '__colidx'):
            setattr(self, '__colidx',
                    gpuarray.to_device(self._sparsity._c_handle.colidx))
        return getattr(self, '__colidx')

    @property
    def _csrdata(self):
        if not hasattr(self, '__csrdata'):
            setattr(self, '__csrdata',
                    gpuarray.zeros(shape=self._sparsity._c_handle.total_nz,
                                   dtype=self.dtype))
        return getattr(self, '__csrdata')

    def zero(self):
        self._csrdata.fill(0)


class Const(DeviceDataMixin, op2.Const):
    _arg_type = Arg

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
        if self.state is DeviceDataMixin.HOST:
            driver.memcpy_htod(ptr, self._data)
            self.state = DeviceDataMixin.BOTH

    def _from_device(self):
        raise RuntimeError("Copying Const %s from device makes no sense" % self)

class Global(DeviceDataMixin, op2.Global):
    _arg_type = Arg

    def _allocate_reduction_buffer(self, grid_size, op):
        if not hasattr(self, '_reduction_buffer') or \
           self._reduction_buffer.size != grid_size:
            self._host_reduction_buffer = np.zeros(np.prod(grid_size) * self.cdim,
                                                   dtype=self.dtype).reshape((-1,)+self._dim)
            if op is not op2.INC:
                self._host_reduction_buffer[:] = self._data
            self._reduction_buffer = gpuarray.to_gpu(self._host_reduction_buffer)
        else:
            if op is not op2.INC:
                self._reduction_buffer.fill(self._data)
            else:
                self._reduction_buffer.fill(0)

    @property
    def data(self):
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST

    def _finalise_reduction_begin(self, grid_size, op):
        self._stream = driver.Stream()
        self._reduction_buffer.get_async(ary=self._host_reduction_buffer,
                                         stream=self._stream)

    def _finalise_reduction_end(self, grid_size, op):
        self.state = DeviceDataMixin.HOST
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

    def _to_device(self):
        if not hasattr(self, '_device_values'):
            self._device_values = gpuarray.to_gpu(self._values)
        elif self._state is not DeviceDataMixin.BOTH:
            self._device_values.set(self._values)
        self._state = DeviceDataMixin.BOTH

    def _from_device(self):
        if not hasattr(self, '_device_values') is None:
            raise RuntimeError("No values for Map %s on device" % self)
        self._state = DeviceDataMixin.HOST
        self._device_values.get(self._values)

class Plan(op2.Plan):
    @property
    def nthrcol(self):
        if not hasattr(self, '_nthrcol'):
            self._nthrcol = gpuarray.to_gpu(super(Plan, self).nthrcol)
        return self._nthrcol

    @property
    def thrcol(self):
        if not hasattr(self, '_thrcol'):
            self._thrcol = gpuarray.to_gpu(super(Plan, self).thrcol)
        return self._thrcol

    @property
    def offset(self):
        if not hasattr(self, '_offset'):
            self._offset = gpuarray.to_gpu(super(Plan, self).offset)
        return self._offset

    @property
    def ind_map(self):
        if not hasattr(self, '_ind_map'):
            self._ind_map = gpuarray.to_gpu(super(Plan, self).ind_map)
        return self._ind_map

    @property
    def ind_offs(self):
        if not hasattr(self, '_ind_offs'):
            self._ind_offs = gpuarray.to_gpu(super(Plan, self).ind_offs)
        return self._ind_offs

    @property
    def ind_sizes(self):
        if not hasattr(self, '_ind_sizes'):
            self._ind_sizes = gpuarray.to_gpu(super(Plan, self).ind_sizes)
        return self._ind_sizes

    @property
    def loc_map(self):
        if not hasattr(self, '_loc_map'):
            self._loc_map = gpuarray.to_gpu(super(Plan, self).loc_map)
        return self._loc_map

    @property
    def nelems(self):
        if not hasattr(self, '_nelems'):
            self._nelems = gpuarray.to_gpu(super(Plan, self).nelems)
        return self._nelems

    @property
    def blkmap(self):
        if not hasattr(self, '_blkmap'):
            self._blkmap = gpuarray.to_gpu(super(Plan, self).blkmap)
        return self._blkmap

def par_loop(kernel, it_space, *args):
    ParLoop(kernel, it_space, *args).compute()

class ParLoop(op2.ParLoop):
    def device_function(self):
        return self._module.get_function(self._stub_name)

    def compile(self, config=None):

        key = self._cache_key
        self._module, self._fun = op2._parloop_cache.get(key, (None, None))
        if self._module is not None:
            return

        compiler_opts = ['-m64', '-Xptxas', '-dlcm=ca',
                         '-Xptxas=-v', '-O3', '-use_fast_math', '-DNVCC']
        inttype = np.dtype('int32').char
        argtypes = inttype      # set size
        if self._is_direct:
            self.generate_direct_loop(config)
            for arg in self.args:
                argtypes += "P" # pointer to each Dat's data
        else:
            self.generate_indirect_loop()
            for arg in self._unique_args:
                if arg._is_mat:
                    # pointer to lma data, offset into lma data
                    # for case of multiple map pairs.
                    argtypes += "P"
                    argtypes += inttype
                else:
                    # pointer to each unique Dat's data
                    argtypes += "P"
            argtypes += "PPPP"  # ind_map, loc_map, ind_sizes, ind_offs
            argtypes += inttype # block offset
            argtypes += "PPPPP" # blkmap, offset, nelems, nthrcol, thrcol
            argtypes += inttype # number of colours in the block

        self._module = SourceModule(self._src, options=compiler_opts)
        self._fun = self.device_function()
        self._fun.prepare(argtypes)
        op2._parloop_cache[key] = self._module, self._fun

    def launch_configuration(self):
        if self._is_direct:
            max_smem = self._max_shared_memory_needed_per_set_element
            smem_offset = max_smem * _WARPSIZE
            max_block = _device.get_attribute(driver.device_attribute.MAX_BLOCK_DIM_X)
            if max_smem == 0:
                block_size = max_block
            else:
                threads_per_sm = _AVAILABLE_SHARED_MEMORY / max_smem
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
        config = self.launch_configuration()
        self.compile(config=config)

        if self._is_direct:
            _args = self.args
            block_size = config['block_size']
            max_grid_size = config['grid_size']
            shared_size = config['required_smem']
        else:
            _args = self._unique_args
            maxbytes = sum([a.dtype.itemsize * a.data.cdim \
                            for a in self._unwound_args if a._is_indirect])
            # shared memory as reported by the device, divided by some
            # factor.  This is the same calculation as done inside
            # op_plan_core, but without assuming 48K shared memory.
            # It would be much nicer if we could tell op_plan_core "I
            # have X bytes shared memory"
            part_size = (_AVAILABLE_SHARED_MEMORY / (64 * maxbytes)) * 64
            self._plan = Plan(self.kernel, self._it_space.iterset,
                              *self._unwound_args,
                              partition_size=part_size)
            max_grid_size = self._plan.ncolblk.max()

        # Upload Const data.
        for c in Const._definitions():
            c._to_device(self._module)

        for arg in _args:
            if arg._is_mat:
                d = arg.data._lmadata.gpudata
                offset = arg.data._lmaoffset(self._it_space.iterset)
                arglist.append(np.intp(d))
                arglist.append(np.int32(offset))
            else:
                arg.data._allocate_device()
                if arg.access is not op2.WRITE:
                    arg.data._to_device()
                karg = arg.data._device_data
                if arg._is_global_reduction:
                    arg.data._allocate_reduction_buffer(max_grid_size,
                                                        arg.access)
                    karg = arg.data._reduction_buffer
                arglist.append(np.intp(karg.gpudata))

        if self._is_direct:
            self._fun.prepared_call(max_grid_size, block_size, *arglist,
                                    shared_size=shared_size)
            for arg in self.args:
                if arg._is_global_reduction:
                    arg.data._finalise_reduction_begin(max_grid_size, arg.access)
                    arg.data._finalise_reduction_end(max_grid_size, arg.access)
                else:
                    # Set write state to False
                    maybe_setflags(arg.data._data, write=False)
                    # Data state is updated in finalise_reduction for Global
                    if arg.access is not op2.READ:
                        arg.data.state = DeviceDataMixin.DEVICE
        else:
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
                # At this point, before we can continue processing in
                # the MPI case, we'll need to wait for halo swaps to
                # complete, but at the moment we don't support that
                # use case, so we just pass through for now.
                if col == self._plan.ncolors_core:
                    pass

                blocks = self._plan.ncolblk[col]
                if blocks <= 0:
                    continue

                arglist[-1] = np.int32(blocks)
                arglist[-7] = np.int32(block_offset)
                blocks = np.asscalar(blocks)
                # Compute capability < 3 can handle at most 2**16  - 1
                # blocks in any one dimension of the grid.
                if blocks >= 2**16:
                    grid_size = (2**16 - 1, (blocks - 1)/(2**16-1) + 1, 1)
                else:
                    grid_size = (blocks, 1, 1)

                block_size = (128, 1, 1)
                shared_size = np.asscalar(self._plan.nsharedCol[col])

                self._fun.prepared_call(grid_size, block_size, *arglist,
                                        shared_size=shared_size)

                # We've reached the end of elements that should
                # contribute to a reduction (this is only different
                # from the total number of elements in the MPI case).
                # So copy the reduction array back to the host now (so
                # that we don't double count halo elements).  We'll
                # finalise the reduction a little later.
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
                elif not arg._is_mat:
                    # Data state is updated in finalise_reduction for Global
                    if arg.access is not op2.READ:
                        arg.data.state = DeviceDataMixin.DEVICE
                else:
                    # Mat, assemble from lma->csr
                    arg.data._assemble()
        if self._has_soa:
            op2stride.remove_from_namespace()

_device = None
_context = None
_WARPSIZE = 32
_AVAILABLE_SHARED_MEMORY = 0
_direct_loop_template = None
_indirect_loop_template = None

def _setup():
    global _device
    global _context
    global _WARPSIZE
    global _AVAILABLE_SHARED_MEMORY
    if _device is None or _context is None:
        import pycuda.autoinit
        _device = pycuda.autoinit.device
        _context = pycuda.autoinit.context
        _WARPSIZE=_device.get_attribute(driver.device_attribute.WARP_SIZE)
        _AVAILABLE_SHARED_MEMORY = _device.get_attribute(driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
    global _direct_loop_template
    global _indirect_loop_template
    env = jinja2.Environment(loader=jinja2.PackageLoader('pyop2', 'assets'))
    if _direct_loop_template is None:
        _direct_loop_template = env.get_template('cuda_direct_loop.jinja2')

    if _indirect_loop_template is None:
        _indirect_loop_template = env.get_template('cuda_indirect_loop.jinja2')
