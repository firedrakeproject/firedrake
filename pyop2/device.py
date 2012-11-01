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

from collections import OrderedDict
import numpy
import op_lib_core as core
import runtime_base as op2
from runtime_base import *
from runtime_base import _parloop_cache, _empty_parloop_cache
from runtime_base import _parloop_cache_size

class Arg(op2.Arg):

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
    DEVICE_UNALLOCATED = 'DEVICE_UNALLOCATED' # device_data not allocated
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

class Dat(DeviceDataMixin, op2.Dat):
    _arg_type = Arg

    def __init__(self, dataset, dim, data=None, dtype=None, name=None, soa=None):
        op2.Dat.__init__(self, dataset, dim, data, dtype, name, soa)
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
            raise ValueError("operands could not be broadcast together with shapes %s, %s" \
                    % (self.array.shape, other.array.shape))

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        self._check_shape(other)
        self.array += as_type(other.array, self.dtype)
        return self

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        self._check_shape(other)
        self.array -= as_type(other.array, self.dtype)
        return self

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        if numpy.isscalar(other):
            self.array *= as_type(other, self.dtype)
        else:
            self._check_shape(other)
            self.array *= as_type(other.array, self.dtype)
        return self

    def __idiv__(self, other):
        """Pointwise division or scaling of fields."""
        if numpy.isscalar(other):
            self.array /= as_type(other, self.dtype)
        else:
            self._check_shape(other)
            self.array /= as_type(other.array, self.dtype)
        return self

class Const(DeviceDataMixin, op2.Const):
    def __init__(self, dim, data, name, dtype=None):
        op2.Const.__init__(self, dim, data, name, dtype)
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

class Global(DeviceDataMixin, op2.Global):
    _arg_type = Arg
    def __init__(self, dim, data, dtype=None, name=None):
        op2.Global.__init__(self, dim, data, dtype, name)
        self.state = DeviceDataMixin.DEVICE_UNALLOCATED

class Map(op2.Map):
    _arg_type = Arg
    def __init__(self, iterset, dataset, dim, values, name=None):
        op2.Map.__init__(self, iterset, dataset, dim, values, name)

    def _to_device(self):
        raise RuntimeError("Abstract device class can't do this")

    def _from_device(self):
        raise RuntimeError("Abstract device class can't do this")

class Mat(op2.Mat):
    _arg_type = Arg
    def __init__(self, datasets, dtype=None, name=None):
        op2.Mat.__init__(self, datasets, dtype, name)
        self.state = DeviceDataMixin.DEVICE_UNALLOCATED


_plan_cache = dict()

def _empty_plan_cache():
    _plan_cache.clear()

def _plan_cache_size():
    return len(_plan_cache)

class Plan(core.op_plan):
    def __new__(cls, kernel, iset, *args, **kwargs):
        ps = kwargs.get('partition_size', 0)
        mc = kwargs.get('matrix_coloring', False)
        key = Plan._cache_key(iset, ps, mc, *args)
        cached = _plan_cache.get(key, None)
        if cached is not None:
            return cached
        else:
            return super(Plan, cls).__new__(cls, kernel, iset, *args,
                                            **kwargs)

    def __init__(self, kernel, iset, *args, **kwargs):
        # This is actually a cached instance, everything's in place,
        # so just return.
        if getattr(self, '_cached', False):
            return
        core.op_plan.__init__(self, kernel, iset, *args, **kwargs)
        ps = kwargs.get('partition_size', 0)
        mc = kwargs.get('matrix_coloring', False)
        key = Plan._cache_key(iset,
                              ps,
                              mc,
                              *args)

        self._fixed_coloring = False
        if mc and any(arg._is_mat for arg in args):
            self._fix_coloring(iset, ps, *args)
            self._fixed_coloring = True

        _plan_cache[key] = self
        self._cached = True

    @classmethod
    def _cache_key(cls, iset, partition_size, matrix_coloring, *args):
        # Set size
        key = (iset.size, )
        # Size of partitions (amount of smem)
        key += (partition_size, )
        # do use matrix cooring ?
        key += (matrix_coloring, )

        # For each indirect arg, the map, the access type, and the
        # indices into the map are important
        inds = OrderedDict()
        for arg in args:
            if arg._is_indirect:
                dat = arg.data
                map = arg.map
                acc = arg.access
                # Identify unique dat-map-acc tuples
                k = (dat, map, acc is op2.INC)
                l = inds.get(k, [])
                l.append(arg.idx)
                inds[k] = l

        # order of indices doesn't matter
        subkey = ('dats', )
        for k,v in inds.iteritems():
            # Only dimension of dat matters, but identity of map does
            subkey += (k[0].cdim, k[1:],) + tuple(sorted(v))
        key += subkey

        # For each matrix arg, the maps and indices
        subkey = ('mats', )
        for arg in args:
            if arg._is_mat:
                idxs = (arg.idx[0].__class__,
                        arg.idx[0].index,
                        arg.idx[1].index)
                subkey += (as_tuple(arg.map), idxs)
        key += subkey

        return key

    def _fix_coloring(self, iset, ps, *args):
        # list of indirect reductions args
        cds = OrderedDict()
        for arg in args:
            if arg._is_indirect_reduction:
                k = arg.data
                l = cds.get(k, [])
                l.append((arg.map, arg.idx))
                cds[k] = l
            elif arg._is_mat:
                k = arg.data
                rowmap = k.sparsity.maps[0][0]
                l = cds.get(k, [])
                for i in range(rowmap.dim):
                    l.append((rowmap, i))
                cds[k] = l

        cds_work = dict()
        for cd in cds.iterkeys():
            if isinstance(cd, Dat):
                s = cd.dataset.size
            elif isinstance(cd, Mat):
                s = cd.sparsity.maps[0][0].dataset.size
            cds_work[cd] = numpy.empty((s,), dtype=numpy.uint32)

        # intra partition coloring
        self._fixed_thrcol = numpy.empty((iset.size, ),
                                         dtype=numpy.int32)
        self._fixed_thrcol.fill(-1)

        tidx = 0
        for p in range(self.nblocks):
            base_color = 0
            terminated = False
            while not terminated:
                terminated = True

                # zero out working array:
                for w in cds_work.itervalues():
                    w.fill(0)

                # color threads
                for t in range(tidx, tidx + super(Plan, self).nelems[p]):
                    if self._fixed_thrcol[t] == -1:
                        mask = 0
                        for cd in cds.iterkeys():
                            for m, i in cds[cd]:
                                mask |= cds_work[cd][m.values[t][i]]

                        if mask == 0xffffffff:
                            terminated = False
                        else:
                            c = 0
                            while mask & 0x1:
                                mask = mask >> 1
                                c += 1
                            self._fixed_thrcol[t] = base_color + c
                            mask = 1 << c
                            for cd in cds.iterkeys():
                                for m, i in cds[cd]:
                                    cds_work[cd][m.values[t][i]] |= mask
                base_color += 32
            tidx += super(Plan, self).nelems[p]

        self._fixed_nthrcol = numpy.zeros(self.nblocks,dtype=numpy.int32)
        tidx = 0
        for p in range(self.nblocks):
            self._fixed_nthrcol[p] = max(self._fixed_thrcol[tidx:(tidx + super(Plan, self).nelems[p])]) + 1
            tidx += super(Plan, self).nelems[p]

        # partition coloring
        pcolors = numpy.empty(self.nblocks, dtype=numpy.int32)
        pcolors.fill(-1)
        base_color = 0
        terminated = False
        while not terminated:
            terminated = True

            # zero out working array:
            for w in cds_work.itervalues():
                w.fill(0)

            tidx = 0
            for p in range(self.nblocks):
                if pcolors[p] == -1:
                    mask = 0
                    for t in range(tidx, tidx + super(Plan, self).nelems[p]):
                        for cd in cds.iterkeys():
                            for m, i in cds[cd]:
                                mask |= cds_work[cd][m.values[t][i]]

                    if mask == 0xffffffff:
                        terminated = False
                    else:
                        c = 0
                        while mask & 0x1:
                            mask = mask >> 1
                            c += 1
                        pcolors[p] = base_color + c

                        mask = 1 << c
                        for t in range(tidx, tidx + super(Plan, self).nelems[p]):
                            for cd in cds.iterkeys():
                                for m, i in cds[cd]:
                                    cds_work[cd][m.values[t][i]] |= mask
                tidx += super(Plan, self).nelems[p]

            base_color += 32

        self._fixed_ncolors = max(pcolors) + 1
        self._fixed_ncolblk = numpy.bincount(pcolors)
        self._fixed_blkmap = numpy.argsort(pcolors, kind='mergesort').astype(numpy.int32)

    @property
    def blkmap(self):
        return self._fixed_blkmap if self._fixed_coloring else super(Plan, self).blkmap

    @property
    def ncolors(self):
        return self._fixed_ncolors if self._fixed_coloring else super(Plan, self).ncolors

    @property
    def ncolblk(self):
        return self._fixed_ncolblk if self._fixed_coloring else super(Plan, self).ncolblk

    @property
    def thrcol(self):
        return self._fixed_thrcol if self._fixed_coloring else super(Plan, self).thrcol

    @property
    def nthrcol(self):
        return self._fixed_nthrcol if self._fixed_coloring else super(Plan, self).nthrcol

class ParLoop(op2.ParLoop):
    def __init__(self, kernel, itspace, *args):
        op2.ParLoop.__init__(self, kernel, itspace, *args)
        self._src = None
        # List of arguments with vector-map/iteration-space indexes
        # flattened out
        # Does contain Mat arguments (cause of coloring)
        self.__unwound_args = []
        # List of unique arguments:
        #  - indirect dats with the same dat/map pairing only appear once
        # Does contain Mat arguments
        self.__unique_args = []
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

    def _get_arg_list(self, propname, arglist_name, keep=None):
        attr = getattr(self, propname, None)
        if attr:
            return attr
        attr = []
        if not keep:
            keep = lambda x: True
        for arg in getattr(self, arglist_name):
            if keep(arg):
                attr.append(arg)
        setattr(self, propname, attr)
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
        """Direct code generation to follow colored execution for global matrix insertion."""
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
