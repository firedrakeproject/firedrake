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
        return self._is_direct and not self.data._is_scalar

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
            self._device_data = gpuarray.empty(shape=self._data.shape, dtype=self.dtype)
            self.state = DeviceDataMixin.CPU

    def _to_device(self):
        self._allocate_device()
        if self.state is DeviceDataMixin.CPU:
            self._device_data.set(self._data)
        self.state = DeviceDataMixin.BOTH

    def _from_device(self):
        if self.state is DeviceDataMixin.GPU:
            self._device_data.get(self._data)
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
        self.state = DeviceDataMixin.UNALLOCATED

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        if self.state is not DeviceDataMixin.UNALLOCATED:
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
    def data(self):
        if self.state is not DeviceDataMixin.UNALLOCATED:
            self.state = DeviceDataMixin.CPU
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        if self.state is not DeviceDataMixin.UNALLOCATED:
            self.state = DeviceDataMixin.CPU

    def _finalise_reduction(self, grid_size, op):
        self.state = DeviceDataMixin.CPU
        tmp = self._host_reduction_buffer
        driver.memcpy_dtoh(tmp, self._reduction_buffer.ptr)
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

def par_loop(kernel, it_space, *args):
    ParLoop(kernel, it_space, *args).compute()

class ParLoop(op2.ParLoop):
    def __init__(self, kernel, it_space, *args):
        op2.ParLoop.__init__(self, kernel, it_space, *args)
        self._src = None

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
        return [a for a in self._direct_args if not a.data._is_scalar]

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

    def compile(self):
        self._module = SourceModule(self._src)

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
             'launch' : config}
        self._src = _direct_loop_template.render(d).encode('ascii')

    def device_function(self):
        return self._module.get_function(self._stub_name)

    def compute(self):
        if self.is_direct():
            config = self.launch_configuration()
            self.generate_direct_loop(config)
            self.compile()
            fun = self.device_function()
            arglist = [np.int32(self._it_space.size)]
            block_size = config['block_size']
            grid_size = config['grid_size']
            shared_size = config['required_smem']
            for arg in self.args:
                arg.data._allocate_device()
                if arg.access is not op2.WRITE:
                    arg.data._to_device()
                karg = arg.data._device_data
                if arg._is_global_reduction:
                    arg.data._allocate_reduction_buffer(grid_size, arg.access)
                    karg = arg.data._reduction_buffer
                arglist.append(karg)
            fun(*arglist, block=block_size, grid=grid_size,
                shared=shared_size)
            for arg in self.args:
                if arg._is_global_reduction:
                    arg.data._finalise_reduction(grid_size, arg.access)
                if arg.access is not op2.READ:
                    arg.data.state = DeviceDataMixin.GPU
        else:
            raise NotImplementedError("Indirect loops in CUDA not yet implemented")

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
        pass
