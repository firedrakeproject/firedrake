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

from libcpp.vector cimport vector
from vecset cimport vecset
from cython.operator cimport dereference as deref, preincrement as inc
from cpython cimport bool
import numpy as np
cimport numpy as np
import cython

np.import_array()

ctypedef np.int32_t DTYPE_t

ctypedef struct cmap:
    int from_size
    int from_exec_size
    int to_size
    int to_exec_size
    int arity
    int* values
    int* offset
    int layers

cdef cmap init_map(omap):
    cdef cmap out
    out.from_size = omap.iterset.size
    out.from_exec_size = omap.iterset.exec_size
    out.to_size = omap.toset.size
    out.to_exec_size = omap.toset.exec_size
    out.arity = omap.arity
    out.values = <int *>np.PyArray_DATA(omap.values_with_halo)
    out.offset = <int *>np.PyArray_DATA(omap.offset)
    if omap.iterset._extruded:
        out.layers = omap.iterset.layers
    else:
        out.layers = 0
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef build_sparsity_pattern_seq(int rmult, int cmult, int nrows, list maps):
    """Create and populate auxiliary data structure: for each element of the
    from set, for each row pointed to by the row map, add all columns pointed
    to by the col map."""
    cdef:
        int e, i, r, d, c, layer, l
        int lsize, rsize, row
        cmap rowmap, colmap
        vector[vecset[int]] s_diag
        vecset[int].const_iterator it

    lsize = nrows*rmult
    s_diag = vector[vecset[int]](lsize)

    for ind, (rmap, cmap) in enumerate(maps):
        rowmap = init_map(rmap)
        colmap = init_map(cmap)
        rsize = rowmap.from_size
        if not s_diag[0].capacity():
            # Preallocate set entries heuristically based on arity
            for i in range(lsize):
                s_diag[i].reserve(6*rowmap.arity)
        # In the case of extruded meshes, in particular, when iterating over
        # horizontal facets, the iteration region determines which part of the
        # mesh the sparsity should be constructed for.
        #
        # ON_BOTTOM: create the sparsity only for the bottom layer of cells
        # ON_TOP: create the sparsity only for the top layers
        # ON_INTERIOR_FACETS: the sparsity creation requires the dynamic
        # computation of the full facet map. Because the extruded direction
        # is structured, the map can be computed dynamically. The map is made up
        # of a lower half given by the base map and an upper part which is obtained
        # by adding the offset to the base map. This produces a map which has double
        # the arity of the initial map.
        if rowmap.layers > 1:
            row_iteration_region = maps[ind][0].iteration_region
            col_iteration_region = maps[ind][1].iteration_region
            for it_sp in row_iteration_region:
                if it_sp.where == 'ON_BOTTOM':
                    for e in range(rsize):
                        for i in range(rowmap.arity):
                            for r in range(rmult):
                                row = rmult * (rowmap.values[i + e*rowmap.arity]) + r
                                for d in range(colmap.arity):
                                    for c in range(cmult):
                                        s_diag[row].insert(cmult * (colmap.values[d + e * colmap.arity]) + c)
                elif it_sp.where == "ON_TOP":
                    layer = rowmap.layers - 2
                    for e in range(rsize):
                        for i in range(rowmap.arity):
                            for r in range(rmult):
                                row = rmult * (rowmap.values[i + e*rowmap.arity] + layer * rowmap.offset[i]) + r
                                for d in range(colmap.arity):
                                    for c in range(cmult):
                                        s_diag[row].insert(cmult * (colmap.values[d + e * colmap.arity] +
                                                           layer * colmap.offset[d]) + c)
                elif it_sp.where == "ON_INTERIOR_FACETS":
                    for e in range(rsize):
                        for i in range(rowmap.arity * 2):
                            for r in range(rmult):
                                for l in range(rowmap.layers - 2):
                                    row = rmult * (rowmap.values[i % rowmap.arity + e*rowmap.arity] + (l + i / rowmap.arity) * rowmap.offset[i % rowmap.arity]) + r
                                    for d in range(colmap.arity * 2):
                                        for c in range(cmult):
                                            s_diag[row].insert(cmult * (colmap.values[d % colmap.arity + e * colmap.arity] +
                                                               (l + d / rowmap.arity) * colmap.offset[d % colmap.arity]) + c)
                else:
                    for e in range(rsize):
                        for i in range(rowmap.arity):
                            for r in range(rmult):
                                for l in range(rowmap.layers - 1):
                                    row = rmult * (rowmap.values[i + e*rowmap.arity] + l * rowmap.offset[i]) + r
                                    for d in range(colmap.arity):
                                        for c in range(cmult):
                                            s_diag[row].insert(cmult * (colmap.values[d + e * colmap.arity] +
                                                               l * colmap.offset[d]) + c)

        else:
            for e in range(rsize):
                for i in range(rowmap.arity):
                    for r in range(rmult):
                            row = rmult * rowmap.values[i + e*rowmap.arity] + r
                            for d in range(colmap.arity):
                                for c in range(cmult):
                                    s_diag[row].insert(cmult * colmap.values[d + e * colmap.arity] + c)

    # Create final sparsity structure
    cdef np.ndarray[DTYPE_t, ndim=1] nnz = np.empty(lsize, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] rowptr = np.empty(lsize + 1, dtype=np.int32)
    rowptr[0] = 0
    for row in range(lsize):
        nnz[row] = s_diag[row].size()
        rowptr[row+1] = rowptr[row] + nnz[row]

    cdef np.ndarray[DTYPE_t, ndim=1] colidx = np.empty(rowptr[lsize], dtype=np.int32)
    # Note: elements in a set are always sorted, so no need to sort colidx
    for row in range(lsize):
        s_diag[row].sort()
        i = rowptr[row]
        it = s_diag[row].begin()
        while it != s_diag[row].end():
            colidx[i] = deref(it)
            inc(it)
            i += 1

    return rowptr[lsize], nnz, rowptr, colidx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef build_sparsity_pattern_mpi(int rmult, int cmult, int nrows, int ncols, list maps):
    """Create and populate auxiliary data structure: for each element of the
    from set, for each row pointed to by the row map, add all columns pointed
    to by the col map."""
    cdef:
        int lrsize, lcsize, rsize, row, entry
        int e, i, r, d, c, l
        cmap rowmap, colmap
        vector[vecset[int]] s_diag, s_odiag

    lrsize = nrows*rmult
    lcsize = ncols*cmult
    s_diag = vector[vecset[int]](lrsize)
    s_odiag = vector[vecset[int]](lrsize)

    for rmap, cmap in maps:
        rowmap = init_map(rmap)
        colmap = init_map(cmap)
        rsize = rowmap.from_exec_size;
        if not s_diag[0].capacity():
            # Preallocate set entries heuristically based on arity
            for i in range(lrsize):
                s_diag[i].reserve(6*rowmap.arity)
                s_odiag[i].reserve(6*rowmap.arity)
        if rowmap.layers > 1:
            for e in range (rsize):
                for i in range(rowmap.arity):
                    for r in range(rmult):
                        for l in range(rowmap.layers - 1):
                            row = rmult * (rowmap.values[i + e*rowmap.arity] + l * rowmap.offset[i]) + r
                            # NOTE: this hides errors due to invalid map entries
                            if row < lrsize:
                                for d in range(colmap.arity):
                                    for c in range(cmult):
                                        entry = cmult * (colmap.values[d + e * colmap.arity] + l * colmap.offset[d]) + c
                                        if entry < lcsize:
                                            s_diag[row].insert(entry)
                                        else:
                                            s_odiag[row].insert(entry)
        else:
            for e in range (rsize):
                for i in range(rowmap.arity):
                    for r in range(rmult):
                            row = rmult * rowmap.values[i + e*rowmap.arity] + r
                            # NOTE: this hides errors due to invalid map entries
                            if row < lrsize:
                                for d in range(colmap.arity):
                                    for c in range(cmult):
                                        entry = cmult * colmap.values[d + e * colmap.arity] + c
                                        if entry < lcsize:
                                            s_diag[row].insert(entry)
                                        else:
                                            s_odiag[row].insert(entry)

    # Create final sparsity structure
    cdef np.ndarray[DTYPE_t, ndim=1] d_nnz = np.empty(lrsize, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] o_nnz = np.empty(lrsize, dtype=np.int32)
    cdef int d_nz = 0
    cdef int o_nz = 0
    for row in range(lrsize):
        d_nnz[row] = s_diag[row].size()
        d_nz += d_nnz[row]
        o_nnz[row] = s_odiag[row].size()
        o_nz += o_nnz[row]

    return d_nnz, o_nnz, d_nz, o_nz

@cython.boundscheck(False)
@cython.wraparound(False)
def build_sparsity(object sparsity, bool parallel):
    cdef int rmult, cmult
    rmult, cmult = sparsity._dims
    cdef int nrows = sparsity._nrows
    cdef int ncols = sparsity._ncols

    if parallel:
        sparsity._d_nnz, sparsity._o_nnz, sparsity._d_nz, sparsity._o_nz = \
            build_sparsity_pattern_mpi(rmult, cmult, nrows, ncols, sparsity.maps)
        sparsity._rowptr = []
        sparsity._colidx = []
    else:
        sparsity._d_nz, sparsity._d_nnz, sparsity._rowptr, sparsity._colidx = \
            build_sparsity_pattern_seq(rmult, cmult, nrows, sparsity.maps)
        sparsity._o_nnz = []
        sparsity._o_nz = 0
