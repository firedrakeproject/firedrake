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

from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector
from libcpp.set cimport set
from cython.operator cimport dereference as deref, preincrement as inc
from cpython cimport bool
import numpy as np
cimport numpy as np

np.import_array()

cdef data_to_numpy_array_with_spec(void * ptr, np.npy_intp size, int t):
    """Return an array of SIZE elements (each of type T) with data from PTR."""
    return np.PyArray_SimpleNewFromData(1, &size, t, ptr)

def free_sparsity(object sparsity):
    cdef np.ndarray tmp
    for attr in ['_rowptr', '_colidx', '_d_nnz', '_o_nnz']:
        try:
            tmp = getattr(sparsity, attr)
            free(<void *>np.PyArray_DATA(tmp))
        except:
            pass

ctypedef struct cmap:
    int from_size
    int from_exec_size
    int to_size
    int to_exec_size
    int arity
    int* values
    int* offset

cdef cmap init_map(omap):
    cdef cmap out
    out.from_size = omap.iterset.size
    out.from_exec_size = omap.iterset.exec_size
    out.to_size = omap.toset.size
    out.to_exec_size = omap.toset.exec_size
    out.arity = omap.arity
    out.values = <int *>np.PyArray_DATA(omap.values)
    out.offset = <int *>np.PyArray_DATA(omap.offset)
    return out

cdef void build_sparsity_pattern_seq (int rmult, int cmult, int nrows, list maps,
                                      int ** _nnz, int ** _rowptr, int ** _colidx,
                                      int * _nz):
    """Create and populate auxiliary data structure: for each element of the
    from set, for each row pointed to by the row map, add all columns pointed
    to by the col map."""
    cdef:
        int e, i, r, d, c
        int lsize, rsize, row
        int *nnz, *rowptr, *colidx
        cmap rowmap, colmap
        vector[set[int]] s_diag
        set[int].iterator it

    lsize = nrows*rmult
    s_diag = vector[set[int]](lsize)

    for rmap, cmap in maps:
        rowmap = init_map(rmap)
        colmap = init_map(cmap)
        rsize = rowmap.from_size
        for e in range(rsize):
            for i in range(rowmap.arity):
                for r in range(rmult):
                    row = rmult * rowmap.values[i + e*rowmap.arity] + r
                    for d in range(colmap.arity):
                        for c in range(cmult):
                            s_diag[row].insert(cmult * colmap.values[d + e * colmap.arity] + c)

    # Create final sparsity structure
    nnz = <int*>malloc(lsize * sizeof(int))
    rowptr = <int*>malloc((lsize+1) * sizeof(int))
    rowptr[0] = 0
    for row in range(lsize):
        nnz[row] = s_diag[row].size()
        rowptr[row+1] = rowptr[row] + nnz[row]

    colidx = <int*>malloc(rowptr[lsize] * sizeof(int))
    # Note: elements in a set are always sorted, so no need to sort colidx
    for row in range(lsize):
        i = rowptr[row]
        it = s_diag[row].begin()
        while it != s_diag[row].end():
            colidx[i] = deref(it)
            inc(it)
            i += 1

    _nz[0] = rowptr[lsize]
    _nnz[0] = nnz
    _rowptr[0] = rowptr
    _colidx[0] = colidx

cdef void build_sparsity_pattern_mpi (int rmult, int cmult, int nrows, list maps,
                                      int ** _d_nnz, int ** _o_nnz,
                                      int * _d_nz, int * _o_nz ):
    """Create and populate auxiliary data structure: for each element of the
    from set, for each row pointed to by the row map, add all columns pointed
    to by the col map."""
    cdef:
        int lsize, rsize, row, entry
        int e, i, r, d, c
        int dnz, o_nz
        int *d_nnz, *o_nnz
        cmap rowmap, colmap
        vector[set[int]] s_diag, s_odiag

    lsize = nrows*rmult
    s_diag = vector[set[int]](lsize)
    s_odiag = vector[set[int]](lsize)

    for rmap, cmap in maps:
        rowmap = init_map(rmap)
        colmap = init_map(cmap)
        rsize = rowmap.from_exec_size;
        for e in range (rsize):
            for i in range(rowmap.arity):
                for r in range(rmult):
                    row = rmult * rowmap.values[i + e*rowmap.arity] + r
                    # NOTE: this hides errors due to invalid map entries
                    if row < lsize:
                        for d in range(colmap.arity):
                            for c in range(cmult):
                                entry = cmult * colmap.values[d + e * colmap.arity] + c
                                if entry < lsize:
                                    s_diag[row].insert(entry)
                                else:
                                    s_odiag[row].insert(entry)

    # Create final sparsity structure
    d_nnz = <int*> malloc(lsize * sizeof(int))
    o_nnz = <int*> malloc(lsize * sizeof(int))
    d_nz = 0
    o_nz = 0
    for row in range(lsize):
        d_nnz[row] = s_diag[row].size()
        d_nz += d_nnz[row]
        o_nnz[row] = s_odiag[row].size()
        o_nz += o_nnz[row]

    _d_nnz[0] = d_nnz;
    _o_nnz[0] = o_nnz;
    _d_nz[0] = d_nz;
    _o_nz[0] = o_nz;

def build_sparsity(object sparsity, bool parallel):
    cdef int rmult, cmult
    rmult, cmult = sparsity._dims
    cdef int nrows = sparsity._nrows
    cdef int lsize = nrows*rmult
    cdef int nmaps = len(sparsity._rmaps)
    cdef int *d_nnz, *o_nnz, *rowptr, *colidx
    cdef int d_nz, o_nz

    if parallel:
        build_sparsity_pattern_mpi(rmult, cmult, nrows, sparsity.maps,
                                   &d_nnz, &o_nnz, &d_nz, &o_nz)
        sparsity._d_nnz = data_to_numpy_array_with_spec(d_nnz, lsize,
                                                        np.NPY_INT32)
        sparsity._o_nnz = data_to_numpy_array_with_spec(o_nnz, lsize,
                                                        np.NPY_INT32)
        sparsity._rowptr = []
        sparsity._colidx = []
        sparsity._d_nz = d_nz
        sparsity._o_nz = o_nz
    else:
        build_sparsity_pattern_seq(rmult, cmult, nrows, sparsity.maps,
                                   &d_nnz, &rowptr, &colidx, &d_nz)
        sparsity._d_nnz = data_to_numpy_array_with_spec(d_nnz, lsize,
                                                        np.NPY_INT32)
        sparsity._o_nnz = []
        sparsity._rowptr = data_to_numpy_array_with_spec(rowptr, lsize+1,
                                                        np.NPY_INT32)
        sparsity._colidx = data_to_numpy_array_with_spec(colidx,
                                                        rowptr[lsize],
                                                        np.NPY_INT32)
        sparsity._d_nz = d_nz
        sparsity._o_nz = 0
