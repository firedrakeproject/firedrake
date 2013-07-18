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

try:
    from collections import OrderedDict
# OrderedDict was added in Python 2.7. Earlier versions can use ordereddict
# from PyPI
except ImportError:
    from ordereddict import OrderedDict
import numpy
import op_lib_core as core
import base
from base import *


class Arg(base.Arg):

    @property
    def _name(self):
        return self.data.name

    @property
    def _lmaoffset_name(self):
        return "%s_lmaoffset" % self._name

    @property
    def _shared_name(self):
        return "%s_shared" % self._name

    def _local_name(self, idx=None):
        if self._is_direct:
            return "%s_local" % self._name
        else:
            if self._is_vec_map and idx is not None:
                return "%s%s_local" % (self._name, self._which_indirect + idx)
            if self._uses_itspace:
                if idx is not None:
                    return "%s%s_local" % (self._name, self._which_indirect + idx)
                return "%s%s_local" % (self._name, self.idx.index)
            return "%s%s_local" % (self._name, self.idx)

    @property
    def _reduction_local_name(self):
        return "%s_reduction_local" % self._name

    @property
    def _reduction_tmp_name(self):
        return "%s_reduction_tmp" % self._name

    @property
    def _reduction_kernel_name(self):
        return "%s_reduction_kernel" % self._name

    @property
    def _vec_name(self):
        return "%s_vec" % self._name

    @property
    def _map_name(self):
        return "%s_map" % self._name

    @property
    def _size_name(self):
        return "%s_size" % self._name

    @property
    def _mat_entry_name(self):
        return "%s_entry" % self._name

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
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def data(self):
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Dat!")
        maybe_setflags(self._data, write=True)
        self._from_device()
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST
        return self._data

    @data.setter
    def data(self, value):
        maybe_setflags(self._data, write=True)
        self._data = verify_reshape(value, self.dtype, self._data.shape)
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST

    @property
    def data_ro(self):
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Dat!")
        maybe_setflags(self._data, write=True)
        self._from_device()
        self.state = DeviceDataMixin.BOTH
        maybe_setflags(self._data, write=False)
        return self._data

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
        raise RuntimeError("Abstract device class can't do this")

    def _to_device(self):
        raise RuntimeError("Abstract device class can't do this")

    def _from_device(self):
        raise RuntimeError("Abstract device class can't do this")


class Dat(DeviceDataMixin, base.Dat):

    def __init__(self, dataset, data=None, dtype=None, name=None,
                 soa=None, uid=None):
        base.Dat.__init__(self, dataset, data, dtype, name, soa, uid)
        self.state = DeviceDataMixin.DEVICE_UNALLOCATED

    @property
    def array(self):
        self._to_device()
        return self._device_data

    @array.setter
    def array(self, ary):
        assert not getattr(self, '_device_data') or self._device_data.shape == ary.shape
        self._device_data = ary
        self.state = DeviceDataMixin.DEVICE

    def _check_shape(self, other):
        if not self.array.shape == other.array.shape:
            raise ValueError("operands could not be broadcast together with shapes %s, %s"
                             % (self.array.shape, other.array.shape))


class Const(DeviceDataMixin, base.Const):

    def __init__(self, dim, data, name, dtype=None):
        base.Const.__init__(self, dim, data, name, dtype)
        self.state = DeviceDataMixin.HOST

    @property
    def data(self):
        self.state = DeviceDataMixin.HOST
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        self.state = DeviceDataMixin.HOST

    def _to_device(self):
        raise RuntimeError("Abstract device class can't do this")

    def _from_device(self):
        raise RuntimeError("Copying Const %s from device not allowed" % self)


class Global(DeviceDataMixin, base.Global):

    def __init__(self, dim, data, dtype=None, name=None):
        base.Global.__init__(self, dim, data, dtype, name)
        self.state = DeviceDataMixin.DEVICE_UNALLOCATED


class Map(base.Map):

    def __init__(self, iterset, dataset, dim, values=None, name=None):
        # The base.Map base class allows not passing values. We do not allow
        # that on the device, but want to keep the API consistent. So if the
        # user doesn't pass values, we fail with MapValueError rather than
        # a (confusing) error telling the user the function requires
        # additional parameters
        if values is None:
            raise MapValueError("Map values must be populated.")
        base.Map.__init__(self, iterset, dataset, dim, values, name)

    def _to_device(self):
        raise RuntimeError("Abstract device class can't do this")

    def _from_device(self):
        raise RuntimeError("Abstract device class can't do this")


class Mat(base.Mat):

    def __init__(self, datasets, dtype=None, name=None):
        base.Mat.__init__(self, datasets, dtype, name)
        self.state = DeviceDataMixin.DEVICE_UNALLOCATED


class _GenericPlan(base.Cached):

    _cache = {}

    @classmethod
    def _cache_key(cls, kernel, iset, *args, **kwargs):
        # Disable caching if requested
        if kwargs.pop('refresh_cache', False):
            return
        partition_size = kwargs.get('partition_size', 0)
        matrix_coloring = kwargs.get('matrix_coloring', False)

        key = (iset.size, partition_size, matrix_coloring)

        # For each indirect arg, the map, the access type, and the
        # indices into the map are important
        inds = OrderedDict()
        for arg in args:
            if arg._is_indirect:
                dat = arg.data
                map = arg.map
                acc = arg.access
                # Identify unique dat-map-acc tuples
                k = (dat, map, acc is base.INC)
                l = inds.get(k, [])
                l.append(arg.idx)
                inds[k] = l

        # order of indices doesn't matter
        subkey = ('dats', )
        for k, v in inds.iteritems():
            # Only dimension of dat matters, but identity of map does
            subkey += (k[0].cdim, k[1:],) + tuple(sorted(v))
        key += subkey

        # For each matrix arg, the maps and indices
        subkey = ('mats', )
        for arg in args:
            if arg._is_mat:
                # For colouring, we only care about the rowmap
                # and the associated iteration index
                idxs = (arg.idx[0].__class__,
                        arg.idx[0].index)
                subkey += (as_tuple(arg.map[0]), idxs)
        key += subkey

        return key


class CPlan(_GenericPlan, core.op_plan):

    """
    Legacy plan function.
        Does not support matrix coloring.
    """
    pass


class PPlan(_GenericPlan, core.Plan):

    """
    PyOP2's cython plan function.
        Support matrix coloring, selective staging and thread color computation.
    """
    pass

# _GenericPlan, CPlan, and PPlan are not meant to be instantiated directly.
# one should instead use Plan. The actual class that is instanciated is defined
# at configuration time see (op2.py::init())
Plan = PPlan


def compare_plans(kernel, iset, *args, **kwargs):
    """This can only be used if caching is disabled."""

    ps = kwargs.get('partition_size', 0)
    mc = kwargs.get('matrix_coloring', False)

    assert not mc, "CPlan does not support matrix coloring, can not compare"
    assert ps > 0, "need partition size"

    # filter the list of access descriptor arguments:
    #  - drop mat arguments (not supported by the C plan
    #  - expand vec arguments
    fargs = list()
    for arg in args:
        if arg._is_vec_map:
            for i in range(arg.map.dim):
                fargs.append(arg.data(arg.map[i], arg.access))
        elif arg._is_mat:
            fargs.append(arg)
        elif arg._uses_itspace:
            for i in range(self._it_space.extents[arg.idx.index]):
                fargs.append(arg.data(arg.map[i], arg.access))
        else:
            fargs.append(arg)

    s = iset._iterset if isinstance(iset, IterationSpace) else iset

    kwargs['refresh_cache'] = True

    cplan = CPlan(kernel, s, *fargs, **kwargs)
    pplan = PPlan(kernel, s, *fargs, **kwargs)

    assert cplan is not pplan
    assert pplan.ninds == cplan.ninds
    assert pplan.nblocks == cplan.nblocks
    assert pplan.ncolors == cplan.ncolors
    assert pplan.nshared == cplan.nshared
    assert (pplan.nelems == cplan.nelems).all()
    # slice is ok cause op2 plan function seems to allocate an
    # arbitrarily longer array
    assert (pplan.ncolblk == cplan.ncolblk[:len(pplan.ncolblk)]).all()
    assert (pplan.blkmap == cplan.blkmap).all()
    assert (pplan.nthrcol == cplan.nthrcol).all()
    assert (pplan.thrcol == cplan.thrcol).all()
    assert (pplan.offset == cplan.offset).all()
    assert (pplan.nindirect == cplan.nindirect).all()
    assert ((pplan.ind_map == cplan.ind_map) | (pplan.ind_map == -1)).all()
    assert (pplan.ind_offs == cplan.ind_offs).all()
    assert (pplan.ind_sizes == cplan.ind_sizes).all()
    assert (pplan.loc_map == cplan.loc_map).all()


class ParLoop(base.ParLoop):

    def __init__(self, kernel, itspace, *args):
        base.ParLoop.__init__(self, kernel, itspace, *args)
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
            if arg._is_vec_map:
                for i in range(arg.map.dim):
                    self.__unwound_args.append(arg.data(arg.map[i],
                                                        arg.access))
            elif arg._is_mat:
                self.__unwound_args.append(arg)
            elif arg._uses_itspace:
                for i in range(self._it_space.extents[arg.idx.index]):
                    self.__unwound_args.append(arg.data(arg.map[i],
                                                        arg.access))
            else:
                self.__unwound_args.append(arg)

            if arg._is_dat:
                key = (arg.data, arg.map)
                if arg._is_indirect:
                    # Needed for indexing into ind_map/loc_map
                    arg._which_indirect = c
                    if arg._is_vec_map:
                        c += arg.map.dim
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
