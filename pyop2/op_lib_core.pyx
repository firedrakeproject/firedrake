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
from cpython cimport bool
import base
import numpy as np
cimport numpy as np
cimport _op_lib_core as core

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

cdef core.cmap init_map(omap):
    cdef core.cmap out
    out.from_size = omap.iterset.size
    out.from_exec_size = omap.iterset.exec_size
    out.to_size = omap.toset.size
    out.to_exec_size = omap.toset.exec_size
    out.arity = omap.arity
    out.values = <int *>np.PyArray_DATA(omap.values)
    return out

def build_sparsity(object sparsity, bool parallel):
    cdef int rmult, cmult
    rmult, cmult = sparsity._dims
    cdef int nrows = sparsity._nrows
    cdef int lsize = nrows*rmult
    cdef int nmaps = len(sparsity._rmaps)
    cdef int *d_nnz, *o_nnz, *rowptr, *colidx
    cdef int d_nz, o_nz

    cdef core.cmap *rmaps = <core.cmap *>malloc(nmaps * sizeof(core.cmap))
    if rmaps is NULL:
        raise MemoryError("Unable to allocate space for rmaps")
    cdef core.cmap *cmaps = <core.cmap *>malloc(nmaps * sizeof(core.cmap))
    if cmaps is NULL:
        raise MemoryError("Unable to allocate space for cmaps")

    try:
        for i in range(nmaps):
            rmaps[i] = init_map(sparsity._rmaps[i])
            cmaps[i] = init_map(sparsity._cmaps[i])

        if parallel:
            core.build_sparsity_pattern_mpi(rmult, cmult, nrows, nmaps,
                                            rmaps, cmaps, &d_nnz, &o_nnz,
                                            &d_nz, &o_nz)
            sparsity._d_nnz = data_to_numpy_array_with_spec(d_nnz, lsize,
                                                            np.NPY_INT32)
            sparsity._o_nnz = data_to_numpy_array_with_spec(o_nnz, lsize,
                                                            np.NPY_INT32)
            sparsity._rowptr = []
            sparsity._colidx = []
            sparsity._d_nz = d_nz
            sparsity._o_nz = o_nz
        else:
            core.build_sparsity_pattern_seq(rmult, cmult, nrows, nmaps,
                                            rmaps, cmaps,
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
    finally:
        free(rmaps)
        free(cmaps)

