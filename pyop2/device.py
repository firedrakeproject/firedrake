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

import base
from base import *

from coffee.plan import ASTKernel

from mpi import collective


class Kernel(base.Kernel):

    def _ast_to_c(self, ast, opts={}):
        """Transform an Abstract Syntax Tree representing the kernel into a
        string of code (C syntax) suitable to GPU execution."""
        ast_handler = ASTKernel(ast)
        ast_handler.plan_gpu()
        return ast.gencode()

    def __init__(self, code, name, opts={}, include_dirs=[]):
        if self._initialized:
            return
        self._code = preprocess(self._ast_to_c(code, opts), include_dirs)
        super(Kernel, self).__init__(self._code, name, opts=opts, include_dirs=include_dirs)


class Arg(base.Arg):

    @property
    def name(self):
        """The generated argument name."""
        if self._is_indirect:
            return "ind_arg%d" % self.indirect_position
        return "arg%d" % self.position

    @property
    def _lmaoffset_name(self):
        return "%s_lmaoffset" % self.name

    @property
    def _shared_name(self):
        return "%s_shared" % self.name

    def _local_name(self, idx=None):
        if self._is_direct:
            return "%s_local" % self.name
        else:
            if self._is_vec_map and idx is not None:
                return "%s_%s_local" % (self.name, self._which_indirect + idx)
            if self._uses_itspace:
                if idx is not None:
                    return "%s_%s_local" % (self.name, self._which_indirect + idx)
                return "%s_%s_local" % (self.name, self.idx.index)
            return "%s_%s_local" % (self.name, self.idx)

    @property
    def _reduction_local_name(self):
        return "%s_reduction_local" % self.name

    @property
    def _reduction_tmp_name(self):
        return "%s_reduction_tmp" % self.name

    @property
    def _reduction_kernel_name(self):
        return "%s_reduction_kernel" % self.name

    @property
    def _vec_name(self):
        return "%s_vec" % self.name

    @property
    def _map_name(self):
        return "%s_map" % self.name

    @property
    def _size_name(self):
        return "%s_size" % self.name

    @property
    def _mat_entry_name(self):
        return "%s_entry" % self.name

    @property
    def _is_staged_direct(self):
        return self._is_direct and not (self.data._is_scalar or self._is_soa)


class DeviceDataMixin(object):

    DEVICE_UNALLOCATED = 'DEVICE_UNALLOCATED'  # device_data not allocated
    HOST_UNALLOCATED = 'HOST_UNALLOCATED'     # host data not allocated
    DEVICE = 'DEVICE'                         # device valid, host invalid
    HOST = 'HOST'                             # host valid, device invalid
    BOTH = 'BOTH'                             # both valid

    @property
    def _bytes_per_elem(self):
        return self.dtype.itemsize * self.cdim

    @property
    def _is_scalar(self):
        return self.cdim == 1

    @property
    def state(self):
        """Current allocation state of the data."""
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    @collective
    def data(self):
        """Numpy array containing the data values."""
        base._trace.evaluate(self, self)
        if len(self._data) is 0 and self.dataset.total_size > 0:
            raise RuntimeError("Illegal access: No data associated with this Dat!")
        maybe_setflags(self._data, write=True)
        self.needs_halo_update = True
        self._from_device()
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST
        return self._data[:self.dataset.size]

    @data.setter
    @collective
    def data(self, value):
        base._trace.evaluate(set(), set([self]))
        maybe_setflags(self._data, write=True)
        self.needs_halo_update = True
        self._data = verify_reshape(value, self.dtype, self.shape)
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST

    @property
    def data_ro(self):
        """Numpy array containing the data values.  Read-only"""
        base._trace.evaluate(reads=self)
        if len(self._data) is 0 and self.dataset.total_size > 0:
            raise RuntimeError("Illegal access: No data associated with this Dat!")
        maybe_setflags(self._data, write=True)
        self._from_device()
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.BOTH
        maybe_setflags(self._data, write=False)
        v = self._data[:self.dataset.size].view()
        v.setflags(write=False)
        return v

    def _maybe_to_soa(self, data):
        """Convert host data to SoA order for device upload if necessary

        If self.soa is True, return data in SoA order, otherwise just
        return data.
        """
        if self.soa:
            shape = data.T.shape
            return data.T.ravel().reshape(shape)
        return data

    def _maybe_to_aos(self, data):
        """Convert host data to AoS order after copy back from device

        If self.soa is True, we will have copied data from device in
        SoA order, convert these into AoS.
        """
        if self.soa:
            tshape = data.T.shape
            shape = data.shape
            return data.reshape(tshape).T.ravel().reshape(shape)
        return data

    def _allocate_device(self):
        """Allocate device data array."""
        raise RuntimeError("Abstract device class can't do this")

    def _to_device(self):
        """Upload data array from host to device."""
        raise RuntimeError("Abstract device class can't do this")

    def _from_device(self):
        """Download data array from device to host."""
        raise RuntimeError("Abstract device class can't do this")


class Dat(DeviceDataMixin, base.Dat):

    def __init__(self, dataset, data=None, dtype=None, name=None,
                 soa=None, uid=None):
        self.state = DeviceDataMixin.DEVICE_UNALLOCATED
        base.Dat.__init__(self, dataset, data, dtype, name, soa, uid)

    @property
    def array(self):
        """The data array on the device."""
        return self._device_data

    @array.setter
    def array(self, ary):
        assert not getattr(self, '_device_data') or self.shape == ary.shape
        self._device_data = ary
        self.state = DeviceDataMixin.DEVICE

    def _check_shape(self, other):
        """Check if ``other`` has compatible shape."""
        if not self.shape == other.shape:
            raise ValueError("operands could not be broadcast together with shapes %s, %s"
                             % (self.shape, other.shape))

    def halo_exchange_begin(self):
        if self.dataset.halo is None:
            return
        maybe_setflags(self._data, write=True)
        self._from_device()
        super(Dat, self).halo_exchange_begin()

    def halo_exchange_end(self):
        if self.dataset.halo is None:
            return
        maybe_setflags(self._data, write=True)
        super(Dat, self).halo_exchange_end()
        if self.state in [DeviceDataMixin.DEVICE,
                          DeviceDataMixin.BOTH]:
            self._halo_to_device()
            self.state = DeviceDataMixin.DEVICE

    def _halo_to_device(self):
        _lim = self.dataset.size * self.dataset.cdim
        self._device_data.ravel()[_lim:].set(self._data[self.dataset.size:])


class Const(DeviceDataMixin, base.Const):

    def __init__(self, dim, data, name, dtype=None):
        base.Const.__init__(self, dim, data, name, dtype)
        self.state = DeviceDataMixin.HOST

    @property
    def data(self):
        """Numpy array containing the data values."""
        self.state = DeviceDataMixin.HOST
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        self.state = DeviceDataMixin.HOST

    def _to_device(self):
        """Upload data array from host to device."""
        raise RuntimeError("Abstract device class can't do this")

    def _from_device(self):
        """Download data array from device to host."""
        raise RuntimeError("Copying Const %s from device not allowed" % self)


class Global(DeviceDataMixin, base.Global):

    def __init__(self, dim, data=None, dtype=None, name=None):
        base.Global.__init__(self, dim, data, dtype, name)
        self.state = DeviceDataMixin.DEVICE_UNALLOCATED

    @property
    def data_ro(self):
        return self.data


class Map(base.Map):

    def __init__(self, iterset, dataset, arity, values=None, name=None,
                 offset=None, parent=None, bt_masks=None):
        base.Map.__init__(self, iterset, dataset, arity, values, name, offset,
                          parent, bt_masks)
        # The base.Map base class allows not passing values. We do not allow
        # that on the device, but want to keep the API consistent. So if the
        # user doesn't pass values, we fail with MapValueError rather than
        # a (confusing) error telling the user the function requires
        # additional parameters
        if len(self.values_with_halo) == 0 and self.iterset.total_size > 0:
            raise MapValueError("Map values must be populated.")

    def _to_device(self):
        """Upload mapping values from host to device."""
        raise RuntimeError("Abstract device class can't do this")

    def _from_device(self):
        """Download mapping values from device to host."""
        raise RuntimeError("Abstract device class can't do this")


class Mat(base.Mat):

    def __init__(self, datasets, dtype=None, name=None):
        base.Mat.__init__(self, datasets, dtype, name)
        self.state = DeviceDataMixin.DEVICE_UNALLOCATED


class ParLoop(base.ParLoop):

    def __init__(self, kernel, itspace, *args, **kwargs):
        base.ParLoop.__init__(self, kernel, itspace, *args, **kwargs)
        # List of arguments with vector-map/iteration-space indexes
        # flattened out
        # Does contain Mat arguments (cause of coloring)
        self.__unwound_args = []
        # List of unique arguments:
        #  - indirect dats with the same dat/map pairing only appear once
        # Does contain Mat arguments
        self.__unique_args = []
        # Argument lists filtered by various criteria
        self._arg_dict = {}
        seen = set()
        c = 0
        for arg in self._actual_args:
            if arg._is_mat:
                for a in arg:
                    self.__unwound_args.append(a)
            elif arg._is_vec_map or arg._uses_itspace:
                for d, m in zip(arg.data, arg.map):
                    for i in range(m.arity):
                        a = d(arg.access, m[i])
                        a.position = arg.position
                        self.__unwound_args.append(a)
            else:
                for a in arg:
                    self.__unwound_args.append(a)

            if arg._is_dat:
                key = (arg.data, arg.map)
                if arg._is_indirect:
                    # Needed for indexing into ind_map/loc_map
                    arg._which_indirect = c
                    if arg._is_vec_map or arg._flatten:
                        c += arg.map.arity
                    elif arg._uses_itspace:
                        c += self._it_space.extents[arg.idx.index]
                    else:
                        c += 1
                if key not in seen:
                    self.__unique_args.append(arg)
                    seen.add(key)
            else:
                self.__unique_args.append(arg)

    def _get_arg_list(self, propname, arglist_name, keep=lambda x: True):
        attr = self._arg_dict.get(propname)
        if attr:
            return attr
        attr = filter(keep, getattr(self, arglist_name))
        self._arg_dict[propname] = attr
        return attr

    @property
    def _is_direct(self):
        for arg in self.__unwound_args:
            if arg._is_indirect:
                return False
        return True

    @property
    def _is_indirect(self):
        return not self._is_direct

    @property
    def _max_shared_memory_needed_per_set_element(self):
        staged = self._all_staged_direct_args
        reduction = self._all_global_reduction_args
        smax = 0
        rmax = 0
        if staged:
            # We stage all the dimensions of the Dat at once
            smax = max(a.data._bytes_per_elem for a in staged)
        if reduction:
            # We reduce over one dimension of the Global at a time
            rmax = max(a.dtype.itemsize for a in reduction)
        return max(smax, rmax)

    @property
    def _stub_name(self):
        return "__%s_stub" % self.kernel.name

    @property
    def _has_itspace(self):
        return len(self._it_space.extents) > 0

    @property
    def _needs_shared_memory(self):
        if self._is_indirect:
            return True
        for arg in self._actual_args:
            if arg._is_global_reduction:
                return True
            if arg._is_staged_direct:
                return True
        return False

    @property
    def _requires_coloring(self):
        """Direct code generation to follow use colored execution scheme."""
        return not not self._all_inc_indirect_dat_args or self._requires_matrix_coloring

    @property
    def _requires_matrix_coloring(self):
        """Direct code generation to follow colored execution for global
        matrix insertion."""
        return False

    @property
    def _unique_args(self):
        return self.__unique_args

    @property
    def _unwound_args(self):
        return self.__unwound_args

    @property
    def _unwound_indirect_args(self):
        keep = lambda x: x._is_indirect
        return self._get_arg_list('__unwound_indirect_args',
                                  '_unwound_args', keep)

    @property
    def _unique_dat_args(self):
        keep = lambda x: x._is_dat
        return self._get_arg_list('__unique_dat_args',
                                  '_unique_args', keep)

    @property
    def _aliased_dat_args(self):
        keep = lambda x: x._is_dat and all(x is not y for y in self._unique_dat_args)
        return self._get_arg_list('__aliased_dat_args',
                                  '_unwound_args', keep)

    @property
    def _unique_vec_map_args(self):
        keep = lambda x: x._is_vec_map
        return self._get_arg_list('__unique_vec_map_args',
                                  '_unique_args', keep)

    @property
    def _unique_indirect_dat_args(self):
        keep = lambda x: x._is_indirect
        return self._get_arg_list('__unique_indirect_dat_args',
                                  '_unique_args', keep)

    @property
    def _unique_read_or_rw_indirect_dat_args(self):
        keep = lambda x: x._is_indirect and x.access in [READ, RW]
        return self._get_arg_list('__unique_read_or_rw_indirect_dat_args',
                                  '_unique_args', keep)

    @property
    def _unique_write_or_rw_indirect_dat_args(self):
        keep = lambda x: x._is_indirect and x.access in [WRITE, RW]
        return self._get_arg_list('__unique_write_or_rw_indirect_dat_args',
                                  '_unique_args', keep)

    @property
    def _unique_inc_indirect_dat_args(self):
        keep = lambda x: x._is_indirect and x.access is INC
        return self._get_arg_list('__unique_inc_indirect_dat_args',
                                  '_unique_args', keep)

    @property
    def _all_inc_indirect_dat_args(self):
        keep = lambda x: x._is_indirect and x.access is INC
        return self._get_arg_list('__all_inc_indirect_dat_args',
                                  '_actual_args', keep)

    @property
    def _all_inc_non_vec_map_indirect_dat_args(self):
        keep = lambda x: x._is_indirect and x.access is INC and \
            not (x._is_vec_map or x._uses_itspace)
        return self._get_arg_list('__all_inc_non_vec_map_indirect_dat_args',
                                  '_actual_args', keep)

    @property
    def _all_vec_map_args(self):
        keep = lambda x: x._is_vec_map
        return self._get_arg_list('__all_vec_map_args',
                                  '_actual_args', keep)

    @property
    def _all_itspace_dat_args(self):
        keep = lambda x: x._is_dat and x._uses_itspace
        return self._get_arg_list('__all_itspace_dat_args',
                                  '_actual_args', keep)

    @property
    def _all_inc_itspace_dat_args(self):
        keep = lambda x: x.access is INC
        return self._get_arg_list('__all_inc_itspace_dat_args',
                                  '_all_itspace_dat_args', keep)

    @property
    def _all_non_inc_itspace_dat_args(self):
        keep = lambda x: x.access is not INC
        return self._get_arg_list('__all_non_inc_itspace_dat_args',
                                  '_all_itspace_dat_args', keep)

    @property
    def _all_inc_vec_map_args(self):
        keep = lambda x: x._is_vec_map and x.access is INC
        return self._get_arg_list('__all_inc_vec_map_args',
                                  '_actual_args', keep)

    @property
    def _all_non_inc_vec_map_args(self):
        keep = lambda x: x._is_vec_map and x.access is not INC
        return self._get_arg_list('__all_non_inc_vec_map_args',
                                  '_actual_args', keep)

    @property
    def _all_vec_like_args(self):
        keep = lambda x: x._is_vec_map or (x._is_dat and x._uses_itspace)
        return self._get_arg_list('__all_vec_like_args',
                                  '_actual_args', keep)

    @property
    def _all_inc_vec_like_args(self):
        keep = lambda x: x.access is INC
        return self._get_arg_list('__all_inc_vec_like_args',
                                  '_all_vec_like_args', keep)

    @property
    def _all_indirect_args(self):
        keep = lambda x: x._is_indirect
        return self._get_arg_list('__all_indirect_args',
                                  '_unwound_args', keep)

    @property
    def _all_direct_args(self):
        keep = lambda x: x._is_direct
        return self._get_arg_list('__all_direct_args',
                                  '_actual_args', keep)

    @property
    def _all_staged_direct_args(self):
        keep = lambda x: x._is_staged_direct
        return self._get_arg_list('__all_non_scalar_direct_args',
                                  '_actual_args', keep)

    @property
    def _all_staged_in_direct_args(self):
        keep = lambda x: x.access is not WRITE
        return self._get_arg_list('__all_staged_in_direct_args',
                                  '_all_staged_direct_args', keep)

    @property
    def _all_staged_out_direct_args(self):
        keep = lambda x: x.access is not READ
        return self._get_arg_list('__all_staged_out_direct_args',
                                  '_all_staged_direct_args', keep)

    @property
    def _all_global_reduction_args(self):
        keep = lambda x: x._is_global_reduction
        return self._get_arg_list('__all_global_reduction_args',
                                  '_actual_args', keep)

    @property
    def _all_global_non_reduction_args(self):
        keep = lambda x: x._is_global and not x._is_global_reduction
        return self._get_arg_list('__all_global_non_reduction_args',
                                  '_actual_args', keep)

    @property
    def _has_matrix_arg(self):
        return any(arg._is_mat for arg in self._unique_args)
