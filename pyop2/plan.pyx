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

import runtime_base as op2
from utils import align
import math
from collections import OrderedDict
import numpy
cimport numpy
from libc.stdlib cimport malloc, free

# C type declarations
ctypedef struct map_idx_t:
    int * map_base
    int dim
    int idx

ctypedef struct flat_cds_t:
    int size
    unsigned int* tmp
    int count
    map_idx_t * mip

cdef class Plan:
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

    def __cinit__(self, kernel, iset, *args, **kwargs):
        ps = kwargs.get('partition_size', 1)
        mc = kwargs.get('matrix_coloring', False)

        assert ps > 0, "partition size must be strictly positive"

        self._nblocks = int(math.ceil(iset.size / float(ps)))
        self._nelems = numpy.array([min(ps, iset.size - i * ps) for i in range(self._nblocks)],
                                  dtype=numpy.int32)

        self._compute_staging_info(iset, ps, mc, args)
        self._compute_coloring(iset, ps, mc, args)

    def _compute_staging_info(self, iset, ps, mc, args):
        """Constructs:
            - nelems
            - nindirect
            - ind_map
            - loc_map
            - ind_sizes
            - ind_offs
            - offset
            - nshared
        """
        # (indices referenced for this dat-map pair, inverse)
        def indices(dat, map):
            return [arg.idx for arg in args if arg.data == dat and arg.map == map]

        self._ninds = 0
        self._nargs = len([arg for arg in args if not arg._is_mat])
        d = OrderedDict()
        for i, arg in enumerate([arg for arg in args if not arg._is_mat]):
            if arg._is_indirect:
                k = (arg.data,arg.map)
                if not d.has_key(k):
                    d[k] = i
                    self._ninds += 1

        inds = dict()
        locs = dict()
        sizes = dict()

        for pi in range(self._nblocks):
            start = pi * ps
            end = start + self._nelems[pi]

            for dat,map in d.iterkeys():
                ii = indices(dat,map)
                l = len(ii)

                inds[(dat,map,pi)], inv = numpy.unique(map.values[start:end,ii], return_inverse=True)
                sizes[(dat,map,pi)] = len(inds[(dat,map,pi)])

                for i, ind in enumerate(sorted(ii)):
                    locs[(dat,map,ind,pi)] = inv[i::l]

        def ind_iter():
            for dat,map in d.iterkeys():
                cumsum = 0
                for pi in range(self._nblocks):
                    cumsum += len(inds[(dat,map,pi)])
                    yield inds[(dat,map,pi)]
                # creates a padding to conform with op2 plan objects
                # fills with -1 for debugging
                # this should be removed and generated code changed
                # once we switch to python plan only
                pad = numpy.empty(len(indices(dat,map)) * iset.size - cumsum, dtype=numpy.int32)
                pad.fill(-1)
                yield pad
        self._ind_map = numpy.concatenate(tuple(ind_iter()))

        def size_iter():
            for pi in range(self._nblocks):
                for dat,map in d.iterkeys():
                    yield sizes[(dat,map,pi)]
        self._ind_sizes = numpy.fromiter(size_iter(), dtype=numpy.int32)

        def nindirect_iter():
            for dat,map in d.iterkeys():
                yield sum(sizes[(dat,map,pi)] for pi in range(self._nblocks))
        self._nindirect = numpy.fromiter(nindirect_iter(), dtype=numpy.int32)

        def loc_iter():
            for dat,map in d.iterkeys():
                for i in indices(dat, map):
                    for pi in range(self._nblocks):
                        yield locs[(dat,map,i,pi)].astype(numpy.int16)
        self._loc_map = numpy.concatenate(tuple(loc_iter()))

        def off_iter():
            _off = dict()
            for dat,map in d.iterkeys():
                _off[(dat,map)] = 0
            for pi in range(self._nblocks):
                for dat,map in d.iterkeys():
                    yield _off[(dat,map)]
                    _off[(dat,map)] += sizes[(dat,map,pi)]
        self._ind_offs = numpy.fromiter(off_iter(), dtype=numpy.int32)

        def offset_iter():
            _offset = 0
            for pi in range(self._nblocks):
                yield _offset
                _offset += self._nelems[pi]
        self._offset = numpy.fromiter(offset_iter(), dtype=numpy.int32)

        # max shared memory required by work groups
        nshareds = [0] * self._nblocks
        for pi in range(self._nblocks):
            for k in d.iterkeys():
                dat, map = k
                nshareds[pi] += align(sizes[(dat,map,pi)] * dat.dtype.itemsize * dat.cdim)
        self._nshared = max(nshareds)

    def _compute_coloring(self, iset, ps, mc, args):
        """Constructs:
            - thrcol
            - nthrcol
            - ncolors
            - blkmap
            - ncolblk
        """
        # list of indirect reductions args
        cds = OrderedDict()
        for arg in args:
            if arg._is_indirect_reduction:
                k = arg.data
                l = cds.get(k, [])
                l.append((arg.map, arg.idx))
                cds[k] = l
            elif mc and arg._is_mat:
                k = arg.data
                rowmap = k.sparsity.maps[0][0]
                l = cds.get(k, [])
                for i in range(rowmap.dim):
                    l.append((rowmap, i))
                cds[k] = l

        # convert cds into a flat array for performant access in cython
        cdef flat_cds_t* fcds

        nfcds = len(cds)
        fcds = <flat_cds_t*> malloc(nfcds * sizeof(flat_cds_t))
        pcds = [None] * nfcds
        for i, cd in enumerate(cds.iterkeys()):
            if isinstance(cd, op2.Dat):
                s = cd.dataset.size
            elif isinstance(cd, op2.Mat):
                s = cd.sparsity.maps[0][0].dataset.size

            pcds[i] = numpy.empty((s,), dtype=numpy.uint32)
            fcds[i].size = s
            fcds[i].tmp = <unsigned int *> numpy.PyArray_DATA(pcds[i])

            fcds[i].count = len(cds[cd])
            fcds[i].mip = <map_idx_t*> malloc(fcds[i].count * sizeof(map_idx_t))
            for j, mi in enumerate(cds[cd]):
                map, idx = mi
                fcds[i].mip[j].map_base = <int *> numpy.PyArray_DATA(map.values)
                fcds[i].mip[j].dim = map.dim
                fcds[i].mip[j].idx = idx

        # intra partition coloring
        self._thrcol = numpy.empty((iset.size, ),
                                         dtype=numpy.int32)
        self._thrcol.fill(-1)

        # type constraining a few variables
        cdef int tidx
        cdef int p
        cdef unsigned int base_color
        cdef int t
        cdef unsigned int mask
        cdef unsigned int c

        # create direct reference to numpy array storage
        cdef int * thrcol
        thrcol = <int *> numpy.PyArray_DATA(self._thrcol)
        cdef int * nelems
        nelems = <int *> numpy.PyArray_DATA(self._nelems)

        tidx = 0
        for p in range(self._nblocks):
            base_color = 0
            terminated = False
            while not terminated:
                terminated = True

                # zero out working array:
                for cd in range(nfcds):
                    for i in range(fcds[cd].size):
                        fcds[cd].tmp[i] = 0

                # color threads
                for t in range(tidx, tidx + nelems[p]):
                    if thrcol[t] == -1:
                        mask = 0

                        for cd in range(nfcds):
                            for mi in range(fcds[cd].count):
                                mask |= fcds[cd].tmp[fcds[cd].mip[mi].map_base[t * fcds[cd].mip[mi].dim + fcds[cd].mip[mi].idx]]

                        if mask == 0xffffffffu:
                            terminated = False
                        else:
                            c = 0
                            while mask & 0x1:
                                mask = mask >> 1
                                c += 1
                            thrcol[t] = base_color + c
                            mask = 1 << c
                            for cd in range(nfcds):
                                for mi in range(fcds[cd].count):
                                    fcds[cd].tmp[fcds[cd].mip[mi].map_base[t * fcds[cd].mip[mi].dim + fcds[cd].mip[mi].idx]] |= mask

                base_color += 32
            tidx += nelems[p]

        self._nthrcol = numpy.zeros(self._nblocks,dtype=numpy.int32)
        tidx = 0
        for p in range(self._nblocks):
            self._nthrcol[p] = max(self._thrcol[tidx:(tidx + nelems[p])]) + 1
            tidx += nelems[p]

        # partition coloring
        pcolors = numpy.empty(self._nblocks, dtype=numpy.int32)
        pcolors.fill(-1)
        base_color = 0
        terminated = False
        while not terminated:
            terminated = True

            # zero out working array:
            for cd in range(nfcds):
                for i in range(fcds[cd].size):
                    fcds[cd].tmp[i] = 0

            tidx = 0
            for p in range(self._nblocks):
                if pcolors[p] == -1:
                    mask = 0
                    for t in range(tidx, tidx + nelems[p]):
                        for cd in range(nfcds):
                            for mi in range(fcds[cd].count):
                                mask |= fcds[cd].tmp[fcds[cd].mip[mi].map_base[t * fcds[cd].mip[mi].dim + fcds[cd].mip[mi].idx]]

                    if mask == 0xffffffff:
                        terminated = False
                    else:
                        c = 0
                        while mask & 0x1:
                            mask = mask >> 1
                            c += 1
                        pcolors[p] = base_color + c

                        mask = 1 << c
                        for t in range(tidx, tidx + nelems[p]):
                            for cd in range(nfcds):
                                for mi in range(fcds[cd].count):
                                    fcds[cd].tmp[fcds[cd].mip[mi].map_base[t * fcds[cd].mip[mi].dim + fcds[cd].mip[mi].idx]] |= mask
                tidx += nelems[p]

            base_color += 32

        # memory free
        for i in range(nfcds):
            free(fcds[i].mip)
        free(fcds)

        self._ncolors = max(pcolors) + 1
        self._ncolblk = numpy.bincount(pcolors).astype(numpy.int32)
        self._blkmap = numpy.argsort(pcolors, kind='mergesort').astype(numpy.int32)

    @property
    def nargs(self):
        return self._nargs

    @property
    def ninds(self):
        return self._ninds

    @property
    def nshared(self):
        return self._nshared

    @property
    def nblocks(self):
        return self._nblocks

    @property
    def ncolors(self):
        return self._ncolors

    @property
    def ncolblk(self):
        return self._ncolblk

    @property
    def nindirect(self):
        return self._nindirect

    @property
    def ind_map(self):
        return self._ind_map

    @property
    def ind_sizes(self):
        return self._ind_sizes

    @property
    def ind_offs(self):
        return self._ind_offs

    @property
    def loc_map(self):
        return self._loc_map

    @property
    def blkmap(self):
        return self._blkmap

    @property
    def offset(self):
        return self._offset

    @property
    def nelems(self):
        return self._nelems

    @property
    def nthrcol(self):
        return self._nthrcol

    @property
    def thrcol(self):
        return self._thrcol

    #dummy values for now, to make it run with the cuda backend
    @property
    def ncolors_core(self):
        return self._ncolors

    #dummy values for now, to make it run with the cuda backend
    @property
    def ncolors_owned(self):
        return self._ncolors

    #dummy values for now, to make it run with the cuda backend
    @property
    def nsharedCol(self):
        return numpy.array([self._nshared] * self._ncolors, dtype=numpy.int32)
