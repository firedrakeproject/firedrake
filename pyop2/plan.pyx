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

"""
Cython implementation of the Plan construction.
"""

import base
from utils import align, as_tuple
import math
import numpy
cimport numpy
from libc.stdlib cimport malloc, free
try:
    from collections import OrderedDict
# OrderedDict was added in Python 2.7. Earlier versions can use ordereddict
# from PyPI
except ImportError:
    from ordereddict import OrderedDict

# C type declarations
ctypedef struct map_idx_t:
    # pointer to the raw numpy array containing the map values
    int * map_base
    # arity of the map
    int arity
    int idx

ctypedef struct flat_race_args_t:
    # Dat size
    int size
    # Temporary array for coloring purpose
    unsigned int* tmp
    # lenght of mip (ie, number of occurences of Dat in the access descriptors)
    int count
    map_idx_t * mip

cdef class _Plan:
    """Plan object contains necessary information for data staging and execution scheduling."""

    # NOTE:
    #  - do not rename fields: _nelems, _ind_map, etc in order to get ride of the boilerplate
    # property definitions, these are necessary to allow CUDA and OpenCL to override them without
    # breaking this code

    cdef numpy.ndarray _nelems
    cdef numpy.ndarray _ind_map
    cdef numpy.ndarray _loc_map
    cdef numpy.ndarray _ind_sizes
    cdef numpy.ndarray _nindirect
    cdef numpy.ndarray _ind_offs
    cdef numpy.ndarray _offset
    cdef numpy.ndarray _thrcol
    cdef numpy.ndarray _nthrcol
    cdef numpy.ndarray _ncolblk
    cdef numpy.ndarray _blkmap
    cdef int _nblocks
    cdef int _nargs
    cdef int _ninds
    cdef int _nshared
    cdef int _ncolors

    def __init__(self, iset, *args, partition_size=1,
                 matrix_coloring=False, staging=True, thread_coloring=True,
                 **kwargs):
        assert partition_size > 0, "partition size must be strictly positive"

        self._compute_partition_info(iset, partition_size, matrix_coloring, args)
        if staging:
            self._compute_staging_info(iset, partition_size, matrix_coloring, args)

        self._compute_coloring(iset, partition_size, matrix_coloring, thread_coloring, args)

    def _compute_partition_info(self, iset, partition_size, matrix_coloring, args):
        self._nblocks = int(math.ceil(iset.size / float(partition_size)))
        self._nelems = numpy.array([min(partition_size, iset.size - i * partition_size) for i in range(self._nblocks)],
                                  dtype=numpy.int32)

        def offset_iter(offset):
            _offset = offset
            for pi in range(self._nblocks):
                yield _offset
                _offset += self._nelems[pi]
        self._offset = numpy.fromiter(offset_iter(iset.offset), dtype=numpy.int32)

    def _compute_staging_info(self, iset, partition_size, matrix_coloring, args):
        """Constructs:
            - nindirect : Number of unique Dat/Map pairs in the argument list
            - ind_map   : Indirection map - array of arrays of indices into the
                          Dat of all indirect arguments
            - loc_map   : Array of offsets of staged data in shared memory for
                          each Dat/Map pair for each partition
            - ind_sizes : array of sizes of indirection maps for each block
            - ind_offs  : array of offsets into indirection maps for each block
            - offset    : List of offsets of each partition
            - nshared   : Bytes of shared memory required per partition
        """
        indices = {}  # indices referenced for a given dat-map pair

        self._ninds = 0
        self._nargs = len([arg for arg in args if not arg._is_mat])
        d = OrderedDict()
        for arg in args:
            if arg._is_indirect and not arg._is_mat:
                k = arg.data, arg.map
                if not k in d:
                    indices[k] = [a.idx for a in args
                                  if a.data is arg.data and a.map is arg.map]
                    d[k] = self._ninds
                    self._ninds += 1

        inds = {}   # Indices referenced by dat via map in given partition
        locs = {}   # Offset of staged data in shared memory by dat via map in
                    # given partition
        sizes = {}  # # of indices references by dat via map in given partition

        for pi in range(self._nblocks):
            start = self._offset[pi]
            end = start + self._nelems[pi]

            for dat,map in d.iterkeys():
                ii = indices[dat, map]
                l = len(ii)

                if (isinstance(iset.set, base.Subset)):
                    staged_values = map.values_with_halo[iset.set.indices[start:end]][:, ii]
                else:
                    staged_values = map.values_with_halo[start:end, ii]

                inds[dat, map, pi], inv = numpy.unique(staged_values, return_inverse=True)
                sizes[dat, map, pi] = len(inds[dat, map, pi])

                for i, ind in enumerate(sorted(ii)):
                    locs[dat, map, ind, pi] = inv[i::l]

        def ind_iter():
            for dat,map in d.iterkeys():
                cumsum = 0
                for pi in range(self._nblocks):
                    cumsum += len(inds[dat, map, pi])
                    yield inds[dat, map, pi]
                # creates a padding to conform with op2 plan objects
                # fills with -1 for debugging
                # this should be removed and generated code changed
                # once we switch to python plan only
                pad = numpy.empty(len(indices[dat, map]) * iset.size - cumsum, dtype=numpy.int32)
                pad.fill(-1)
                yield pad
        t = tuple(ind_iter())
        self._ind_map = numpy.concatenate(t) if t else numpy.array([], dtype=numpy.int32)

        def size_iter():
            for pi in range(self._nblocks):
                for dat,map in d.iterkeys():
                    yield sizes[(dat,map,pi)]
        self._ind_sizes = numpy.fromiter(size_iter(), dtype=numpy.int32)

        def nindirect_iter():
            for dat,map in d.iterkeys():
                yield sum(sizes[(dat,map,pi)] for pi in range(self._nblocks))
        self._nindirect = numpy.fromiter(nindirect_iter(), dtype=numpy.int32)

        locs_t = tuple(locs[dat, map, i, pi].astype(numpy.int16)
                       for dat, map in d.iterkeys()
                       for i in indices[dat, map]
                       for pi in range(self._nblocks))
        self._loc_map = numpy.concatenate(locs_t) if locs_t else numpy.array([], dtype=numpy.int16)

        def off_iter():
            _off = dict()
            for dat, map in d.iterkeys():
                _off[dat, map] = 0
            for pi in range(self._nblocks):
                for dat, map in d.iterkeys():
                    yield _off[dat, map]
                    _off[dat, map] += sizes[dat, map, pi]
        self._ind_offs = numpy.fromiter(off_iter(), dtype=numpy.int32)

        # max shared memory required by work groups
        nshareds = [0] * self._nblocks
        for pi in range(self._nblocks):
            for k in d.iterkeys():
                dat, map = k
                nshareds[pi] += align(sizes[(dat,map,pi)] * dat.dtype.itemsize * dat.cdim)
        self._nshared = max(nshareds)

    def _compute_coloring(self, iset, partition_size, matrix_coloring, thread_coloring, args):
        """Constructs:
            - thrcol  : Thread colours for each element of iteration space
            - nthrcol : Array of numbers of thread colours for each partition
            - ncolors : Total number of block colours
            - blkmap  : List of blocks ordered by colour
            - ncolblk : Array of numbers of block with any given colour
        """
        # args requiring coloring (ie, indirect reduction and matrix args)
        #  key: Dat
        #  value: [(map, idx)] (sorted as they appear in the access descriptors)
        race_args = OrderedDict()
        for arg in args:
            if arg._is_indirect_reduction:
                k = arg.data
                l = race_args.get(k, [])
                l.append((arg.map, arg.idx))
                race_args[k] = l
            elif matrix_coloring and arg._is_mat:
                k = arg.data
                rowmap = arg.map[0]
                l = race_args.get(k, [])
                for i in range(rowmap.arity):
                    l.append((rowmap, i))
                race_args[k] = l

        # convert 'OrderedDict race_args' into a flat array for performant access in cython
        cdef int n_race_args = len(race_args)
        cdef flat_race_args_t* flat_race_args = <flat_race_args_t*> malloc(n_race_args * sizeof(flat_race_args_t))
        pcds = [None] * n_race_args
        for i, ra in enumerate(race_args.iterkeys()):
            if isinstance(ra, base.Dat):
                s = ra.dataset.exec_size
            elif isinstance(ra, base.Mat):
                s = ra.sparsity.maps[0][0].toset.exec_size

            pcds[i] = numpy.empty((s,), dtype=numpy.uint32)
            flat_race_args[i].size = s
            flat_race_args[i].tmp = <unsigned int *> numpy.PyArray_DATA(pcds[i])

            flat_race_args[i].count = len(race_args[ra])
            flat_race_args[i].mip = <map_idx_t*> malloc(flat_race_args[i].count * sizeof(map_idx_t))
            for j, mi in enumerate(race_args[ra]):
                map, idx = mi
                if map._parent is not None:
                    map = map._parent
                flat_race_args[i].mip[j].map_base = <int *> numpy.PyArray_DATA(map.values_with_halo)
                flat_race_args[i].mip[j].arity = map.arity
                flat_race_args[i].mip[j].idx = idx

        # type constraining a few variables
        cdef int _p
        cdef unsigned int _base_color
        cdef int _t
        cdef unsigned int _mask
        cdef unsigned int _color
        cdef int _rai
        cdef int _mi
        cdef int _i

        # indirection array:
        # array containing the iteration set index given a thread index
        #  - id for normal sets
        #  - Subset::indices for subsets
        # (the extra indirection is to avoid a having a test in the inner most
        # loops and to avoid splitting code: set vs subset)
        cdef int * iteridx
        if isinstance(iset.set, base.Subset):
            iteridx = <int *> numpy.PyArray_DATA(iset.set.indices)
        else:
            _id = numpy.arange(iset.set.total_size, dtype=numpy.uint32)
            iteridx = <int *> numpy.PyArray_DATA(_id)

        # intra partition coloring
        self._thrcol = numpy.empty((iset.set.exec_size, ), dtype=numpy.int32)
        self._thrcol.fill(-1)

        # create direct reference to numpy array storage
        cdef int * thrcol = <int *> numpy.PyArray_DATA(self._thrcol)
        cdef int * nelems = <int *> numpy.PyArray_DATA(self._nelems)
        cdef int * offset = <int *> numpy.PyArray_DATA(self._offset)

        # Colour threads of each partition
        if thread_coloring:
            # For each block
            for _p in range(self._nblocks):
                _base_color = 0
                terminated = False
                while not terminated:
                    terminated = True

                    # zero out working array:
                    for _rai in range(n_race_args):
                        for _i in range(flat_race_args[_rai].size):
                            flat_race_args[_rai].tmp[_i] = 0

                    # color threads
                    for _t in range(offset[_p], offset[_p] + nelems[_p]):
                        if thrcol[_t] == -1:
                            _mask = 0

                            # Find an available colour (the first colour not
                            # touched by the current thread)
                            for _rai in range(n_race_args):
                                for _mi in range(flat_race_args[_rai].count):
                                    _mask |= flat_race_args[_rai].tmp[flat_race_args[_rai].mip[_mi].map_base[iteridx[_t] * flat_race_args[_rai].mip[_mi].arity + flat_race_args[_rai].mip[_mi].idx]]

                            # Check if colour is available i.e. mask isn't full
                            if _mask == 0xffffffffu:
                                terminated = False
                            else:
                                # Find the first available colour
                                _color = 0
                                while _mask & 0x1:
                                    _mask = _mask >> 1
                                    _color += 1
                                thrcol[_t] = _base_color + _color
                                # Mark everything touched by the current
                                # thread with that colour
                                _mask = 1 << _color
                                for _rai in range(n_race_args):
                                    for _mi in range(flat_race_args[_rai].count):
                                        flat_race_args[_rai].tmp[flat_race_args[_rai].mip[_mi].map_base[iteridx[_t] * flat_race_args[_rai].mip[_mi].arity + flat_race_args[_rai].mip[_mi].idx]] |= _mask

                    # We've run out of colours, so we start over and offset
                    _base_color += 32

            self._nthrcol = numpy.zeros(self._nblocks,dtype=numpy.int32)
            for _p in range(self._nblocks):
                self._nthrcol[_p] = max(self._thrcol[offset[_p]:(offset[_p] + nelems[_p])]) + 1
            self._thrcol = self._thrcol[iset.offset:(iset.offset + iset.size)]

        # partition coloring
        pcolors = numpy.empty(self._nblocks, dtype=numpy.int32)
        pcolors.fill(-1)

        cdef int * _pcolors = <int *> numpy.PyArray_DATA(pcolors)

        _base_color = 0
        terminated = False
        while not terminated:
            terminated = True

            # zero out working array:
            for _rai in range(n_race_args):
                for _i in range(flat_race_args[_rai].size):
                    flat_race_args[_rai].tmp[_i] = 0

            # For each partition
            for _p in range(self._nblocks):
                # If this partition doesn't already have a colour
                if _pcolors[_p] == -1:
                    _mask = 0
                    # Find an available colour (the first colour not touched
                    # by the current partition)
                    for _t in range(offset[_p], offset[_p] + nelems[_p]):
                        for _rai in range(n_race_args):
                            for _mi in range(flat_race_args[_rai].count):
                                _mask |= flat_race_args[_rai].tmp[flat_race_args[_rai].mip[_mi].map_base[iteridx[_t] * flat_race_args[_rai].mip[_mi].arity + flat_race_args[_rai].mip[_mi].idx]]

                    # Check if a colour is available i.e. the mask isn't full
                    if _mask == 0xffffffffu:
                        terminated = False
                    else:
                        # Find the first available colour
                        _color = 0
                        while _mask & 0x1:
                            _mask = _mask >> 1
                            _color += 1
                        _pcolors[_p] = _base_color + _color

                        # Mark everything touched by the current partition with
                        # that colour
                        _mask = 1 << _color
                        for _t in range(offset[_p], offset[_p] + nelems[_p]):
                            for _rai in range(n_race_args):
                                for _mi in range(flat_race_args[_rai].count):
                                    flat_race_args[_rai].tmp[flat_race_args[_rai].mip[_mi].map_base[iteridx[_t] * flat_race_args[_rai].mip[_mi].arity + flat_race_args[_rai].mip[_mi].idx]] |= _mask

            # We've run out of colours, so we start over and offset by 32
            _base_color += 32

        # memory free
        for i in range(n_race_args):
            free(flat_race_args[i].mip)
        free(flat_race_args)

        self._pcolors = pcolors
        self._ncolors = max(pcolors) + 1
        self._ncolblk = numpy.bincount(pcolors).astype(numpy.int32)
        self._blkmap = numpy.argsort(pcolors, kind='mergesort').astype(numpy.int32)

    @property
    def nargs(self):
        """Number of arguments."""
        return self._nargs

    @property
    def ninds(self):
        """Number of indirect non-matrix arguments."""
        return self._ninds

    @property
    def nshared(self):
        """Bytes of shared memory required per partition."""
        return self._nshared

    @property
    def nblocks(self):
        """Number of partitions."""
        return self._nblocks

    @property
    def ncolors(self):
        """Total number of block colours."""
        return self._ncolors

    @property
    def ncolblk(self):
        """Array of numbers of block with any given colour."""
        return self._ncolblk

    @property
    def nindirect(self):
        """Number of unique Dat/Map pairs in the argument list."""
        return self._nindirect

    @property
    def ind_map(self):
        """Indirection map: array of arrays of indices into the Dat of all
        indirect arguments (nblocks x nindirect x nvalues)."""
        return self._ind_map

    @property
    def ind_sizes(self):
        """2D array of sizes of indirection maps for each block (nblocks x
        nindirect)."""
        return self._ind_sizes

    @property
    def ind_offs(self):
        """2D array of offsets into the indirection maps for each block
        (nblocks x nindirect)."""
        return self._ind_offs

    @property
    def loc_map(self):
        """Array of offsets of staged data in shared memory for each Dat/Map
        pair for each partition (nblocks x nindirect x partition size)."""
        return self._loc_map

    @property
    def blkmap(self):
        """List of blocks ordered by colour."""
        return self._blkmap

    @property
    def offset(self):
        """List of offsets of each partition."""
        return self._offset

    @property
    def nelems(self):
        """Array of numbers of elements for each partition."""
        return self._nelems

    @property
    def nthrcol(self):
        """Array of numbers of thread colours for each partition."""
        return self._nthrcol

    @property
    def thrcol(self):
        """Array of thread colours for each element of iteration space."""
        return self._thrcol

    #dummy values for now, to make it run with the cuda backend
    @property
    def nsharedCol(self):
        """Array of shared memory sizes for each colour."""
        return numpy.array([self._nshared] * self._ncolors, dtype=numpy.int32)


class Plan(base.Cached, _Plan):

    def __init__(self, iset, *args, **kwargs):
        if self._initialized:
            Plan._cache_hit[self] += 1
            return
        _Plan.__init__(self, iset, *args, **kwargs)
        Plan._cache_hit[self] = 0
        self._initialized = True

    _cache_hit = {}
    _cache = {}

    @classmethod
    def _cache_key(cls, part, *args, **kwargs):
        # Disable caching if requested
        if kwargs.pop('refresh_cache', False):
            return
        partition_size = kwargs.get('partition_size', 0)
        matrix_coloring = kwargs.get('matrix_coloring', False)

        key = (part.set.size, part.offset, part.size,
               partition_size, matrix_coloring)

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
