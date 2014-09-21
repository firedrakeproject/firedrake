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
cimport petsc4py.PETSc as PETSc
from petsc4py import PETSc

np.import_array()

cdef extern from "petsc.h":
    ctypedef long PetscInt
    ctypedef double PetscScalar
    ctypedef enum PetscInsertMode "InsertMode":
        PETSC_INSERT_VALUES "INSERT_VALUES"
    int PetscCalloc1(size_t, void*)
    int PetscMalloc1(size_t, void*)
    int PetscFree(void*)
    int MatSetValuesBlockedLocal(PETSc.PetscMat, PetscInt, PetscInt*, PetscInt, PetscInt*,
                                 PetscScalar*, PetscInsertMode)
    int MatSetValuesLocal(PETSc.PetscMat, PetscInt, PetscInt*, PetscInt, PetscInt*,
                          PetscScalar*, PetscInsertMode)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef build_sparsity_pattern(int rmult, int cmult, list maps, bool have_odiag):
    """Build a sparsity pattern defined by a list of pairs of maps

    :arg rmult: the dataset dimension of the rows of the sparsity (the row block size).
    :arg cmult: the dataset dimension of the columns of the sparsity (column block size).
    :arg maps: a list of pairs of row, column maps defining the sparsity pattern.

    The sparsity pattern is built from the outer products of the pairs
    of maps.  This code works for both the serial and (MPI-) parallel
    case."""
    cdef:
        int e, i, r, d, c
        int layer, layer_start, layer_end
        int local_nrows, local_ncols, set_size
        int row, col, tmp_row, tmp_col, reps, rrep, crep
        int rarity, carity
        vector[vecset[int]] s_diag, s_odiag
        vecset[int].const_iterator it
        int *rmap_vals
        int *cmap_vals
        int *roffset
        int *coffset

    # Number of rows and columns "local" to this process
    # In parallel, the matrix is distributed row-wise, so all
    # processes always see all columns, but we distinguish between
    # local (process-diagonal) and remote (process-off-diagonal)
    # columns.
    local_nrows = rmult * maps[0][0].toset.size
    local_ncols = cmult * maps[0][1].toset.size

    if local_nrows == 0:
        # We don't own any rows, return something appropriate.
        dummy = np.empty(0, dtype=np.int32).reshape(-1)
        return 0, 0, dummy, dummy, dummy, dummy

    s_diag = vector[vecset[int]](local_nrows)
    if have_odiag:
        s_odiag = vector[vecset[int]](local_nrows)

    extruded = maps[0][0].iterset._extruded

    for rmap, cmap in maps:
        set_size = rmap.iterset.exec_size
        rarity = rmap.arity
        carity = cmap.arity
        rmap_vals = <int *>np.PyArray_DATA(rmap.values_with_halo)
        cmap_vals = <int *>np.PyArray_DATA(cmap.values_with_halo)
        if not s_diag[0].capacity():
            # Preallocate set entries heuristically based on arity
            for i in range(local_nrows):
                s_diag[i].reserve(6*rarity)
                # Always reserve space for diagonal entry
                if i < local_ncols:
                    s_diag[i].insert(i)
            if have_odiag:
                for i in range(local_nrows):
                    s_odiag[i].reserve(6*rarity)
        if not extruded:
            # Non extruded case, reasonably straightfoward
            for e in range(set_size):
                for i in range(rarity):
                    tmp_row = rmult * rmap_vals[e * rarity + i]
                    # Not a process-local row, carry on.
                    if tmp_row >= local_nrows:
                        continue
                    for r in range(rmult):
                        row = tmp_row + r
                        for d in range(carity):
                            for c in range(cmult):
                                col = cmult * cmap_vals[e * carity + d] + c
                                # Process-local column?
                                if col < local_ncols:
                                    s_diag[row].insert(col)
                                else:
                                    assert have_odiag, "Should never happen"
                                    s_odiag[row].insert(col)
        else:
            # Now the slightly trickier extruded case
            roffset = <int *>np.PyArray_DATA(rmap.offset)
            coffset = <int *>np.PyArray_DATA(cmap.offset)
            layers = rmap.iterset.layers
            for region in rmap.iteration_region:
                # The rowmap will have an iteration region attached to
                # it specifying which bits of the "implicit" (walking
                # up the column) map we want.  This mostly affects the
                # range of the loop over layers, except in the
                # ON_INTERIOR_FACETS where we also have to "double" up
                # the map.
                layer_start = 0
                layer_end = layers - 1
                reps = 1
                if region.where == "ON_BOTTOM":
                    layer_end = 1
                elif region.where == "ON_TOP":
                    layer_start = layers - 2
                elif region.where == "ON_INTERIOR_FACETS":
                    layer_end = layers - 2
                    reps = 2
                elif region.where != "ALL":
                    raise RuntimeError("Unhandled iteration region %s", region)
                for e in range(set_size):
                    for i in range(rarity):
                        tmp_row = rmult * (rmap_vals[e * rarity + i] + layer_start * roffset[i])
                        # Not a process-local row, carry on
                        if tmp_row >= local_nrows:
                            continue
                        for r in range(rmult):
                            # Double up for interior facets
                            for rrep in range(reps):
                                row = tmp_row + r + rmult*rrep*roffset[i]
                                for layer in range(layer_start, layer_end):
                                    for d in range(carity):
                                        for c in range(cmult):
                                            for crep in range(reps):
                                                col = cmult * (cmap_vals[e * carity + d] +
                                                               (layer + crep) * coffset[d]) + c
                                                if col < local_ncols:
                                                    s_diag[row].insert(col)
                                                else:
                                                    assert have_odiag, "Should never happen"
                                                    s_odiag[row].insert(col)
                                    row += rmult * roffset[i]

    # Create final sparsity structure
    cdef np.ndarray[np.int32_t, ndim=1] dnnz = np.zeros(local_nrows, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] onnz = np.zeros(local_nrows, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] rowptr
    cdef np.ndarray[np.int32_t, ndim=1] colidx
    cdef int dnz, onz
    if have_odiag:
        # Don't need these, so create dummy arrays
        rowptr = np.empty(0, dtype=np.int32).reshape(-1)
        colidx = np.empty(0, dtype=np.int32).reshape(-1)
    else:
        rowptr = np.empty(local_nrows + 1, dtype=np.int32)

    dnz = 0
    onz = 0
    if have_odiag:
        # Have off-diagonals (i.e. we're in parallel).
        for row in range(local_nrows):
            dnnz[row] = s_diag[row].size()
            dnz += dnnz[row]
            onnz[row] = s_odiag[row].size()
            onz += onnz[row]
    else:
        # Not in parallel, in which case build the explicit row
        # pointer and column index data structure petsc wants.
        rowptr[0] = 0
        for row in range(local_nrows):
            dnnz[row] = s_diag[row].size()
            rowptr[row+1] = rowptr[row] + dnnz[row]
            dnz += dnnz[row]
        colidx = np.empty(dnz, dtype=np.int32)
        for row in range(local_nrows):
            # each row's entries in colidx need to be sorted.
            s_diag[row].sort()
            i = rowptr[row]
            it = s_diag[row].begin()
            while it != s_diag[row].end():
                colidx[i] = deref(it)
                inc(it)
                i += 1

    return dnz, onz, dnnz, onnz, rowptr, colidx


def fill_with_zeros(PETSc.Mat mat not None, dims, maps):
    """Fill a PETSc matrix with zeros in all slots we might end up inserting into

    :arg mat: the PETSc Mat (must already be preallocated)
    :arg dims: the dimensions of the sparsity (block size)
    :arg maps: the pairs of maps defining the sparsity pattern"""
    cdef:
        PetscInt rdim, cdim
        PetscScalar *values
        int set_entry
        int set_size
        int layer_start, layer_end
        int layer
        PetscInt i
        PetscScalar zero = 0.0
        PetscInt nrow, ncol
        PetscInt rarity, carity, tmp_rarity, tmp_carity
        PetscInt[:, ::1] rmap, cmap
        PetscInt *rvals
        PetscInt *cvals
        PetscInt *roffset
        PetscInt *coffset

    rdim, cdim = dims
    # Always allocate space for diagonal
    nrow, ncol = mat.getLocalSize()
    for i in range(nrow):
        if i < ncol:
            MatSetValuesLocal(mat.mat, 1, &i, 1, &i, &zero, PETSC_INSERT_VALUES)
    extruded = maps[0][0].iterset._extruded
    for pair in maps:
        # Iterate over row map values including value entries
        set_size = pair[0].iterset.exec_size
        if set_size == 0:
            continue
        # Map values
        rmap = pair[0].values_with_halo
        cmap = pair[1].values_with_halo
        # Arity of maps
        rarity = pair[0].arity
        carity = pair[1].arity

        if not extruded:
            # The non-extruded case is easy, we just walk over the
            # rmap and cmap entries and set a block of values.
            PetscCalloc1(rarity*carity*rdim*cdim, &values)
            for set_entry in range(set_size):
                MatSetValuesBlockedLocal(mat.mat, rarity, &rmap[set_entry, 0],
                                         carity, &cmap[set_entry, 0],
                                         values, PETSC_INSERT_VALUES)
        else:
            # The extruded case needs a little more work.
            layers = pair[0].iterset.layers
            # We only need the *2 if we have an ON_INTERIOR_FACETS
            # iteration region, but it doesn't hurt to make them all
            # bigger, since we can special case less code below.
            PetscCalloc1(2*rarity*carity*rdim*cdim, &values)
            # Row values (generally only rarity of these)
            PetscMalloc1(2 * rarity, &rvals)
            # Col values (generally only rarity of these)
            PetscMalloc1(2 * carity, &cvals)
            # Offsets (for walking up the column)
            PetscMalloc1(rarity, &roffset)
            PetscMalloc1(carity, &coffset)
            # Walk over the iteration regions on this map.
            for r in pair[0].iteration_region:
                # Default is "ALL"
                layer_start = 0
                layer_end = layers - 1
                tmp_rarity = rarity
                tmp_carity = carity
                if r.where == "ON_BOTTOM":
                    # Finish after first layer
                    layer_end = 1
                elif r.where == "ON_TOP":
                    # Start on penultimate layer
                    layer_start = layers - 2
                elif r.where == "ON_INTERIOR_FACETS":
                    # Finish on penultimate layer
                    layer_end = layers - 2
                    # Double up rvals and cvals (the map is over two
                    # cells, not one)
                    tmp_rarity *= 2
                    tmp_carity *= 2
                elif r.where != "ALL":
                    raise RuntimeError("Unhandled iteration region %s", r)
                for i in range(rarity):
                    roffset[i] = pair[0].offset[i]
                for i in range(carity):
                    coffset[i] = pair[1].offset[i]
                for set_entry in range(set_size):
                    # In the case of tmp_rarity == rarity this is just:
                    #
                    # rvals[i] = rmap[set_entry, i] + layer_start * roffset[i]
                    #
                    # But this means less special casing.
                    for i in range(tmp_rarity):
                        rvals[i] = rmap[set_entry, i % rarity] + \
                                   (layer_start + i / rarity) * roffset[i % rarity]
                    # Ditto
                    for i in range(tmp_carity):
                        cvals[i] = cmap[set_entry, i % carity] + \
                                   (layer_start + i / carity) * coffset[i % carity]
                    for layer in range(layer_start, layer_end):
                        MatSetValuesBlockedLocal(mat.mat, tmp_rarity, rvals,
                                                 tmp_carity, cvals,
                                                 values, PETSC_INSERT_VALUES)
                        # Move to the next layer
                        for i in range(tmp_rarity):
                            rvals[i] += roffset[i % rarity]
                        for i in range(tmp_carity):
                            cvals[i] += coffset[i % carity]
            PetscFree(rvals)
            PetscFree(cvals)
            PetscFree(roffset)
            PetscFree(coffset)
        PetscFree(values)
    # Aaaand, actually finalise the assembly.
    mat.assemble()


def build_sparsity(object sparsity, bool parallel):
    cdef int rmult, cmult
    rmult, cmult = sparsity._dims

    # Build sparsity pattern for block sparse matrix
    if rmult == cmult and rmult > 1:
        rmult = cmult = 1
    pattern = build_sparsity_pattern(rmult, cmult, sparsity.maps, have_odiag=parallel)

    sparsity._d_nz = pattern[0]
    sparsity._o_nz = pattern[1]
    sparsity._d_nnz = pattern[2]
    sparsity._o_nnz = pattern[3]
    sparsity._rowptr = pattern[4]
    sparsity._colidx = pattern[5]
