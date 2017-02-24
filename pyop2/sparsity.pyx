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
from pyop2.datatypes import IntType

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


cdef object set_writeable(map):
     flag = map.values_with_halo.flags['WRITEABLE']
     map.values_with_halo.setflags(write=True)
     return flag

cdef void restore_writeable(map, flag):
     map.values_with_halo.setflags(write=flag)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void add_entries(rset, rmap, cset, cmap,
                             PetscInt row_offset,
                             vector[vecset[PetscInt]]& diag,
                             vector[vecset[PetscInt]]& odiag,
                             bint should_block):
    cdef:
        PetscInt nrows, ncols, i, j, k, l, nent, e
        PetscInt rarity, carity, row, col, rdim, cdim
        PetscInt[:, ::1] rmap_vals, cmap_vals

    nent = rmap.iterset.exec_size

    if should_block:
        rdim = cdim = 1
    else:
        rdim = rset.cdim
        cdim = cset.cdim

    rmap_vals = rmap.values_with_halo
    cmap_vals = cmap.values_with_halo

    nrows = rset.size * rdim
    ncols = cset.size * cdim

    rarity = rmap.arity
    carity = cmap.arity

    for e in range(nent):
        for i in range(rarity):
            row = rdim * rmap_vals[e, i]
            if row >= nrows:
                # Not a process local row
                continue
            row += row_offset
            for j in range(rdim):
                for k in range(carity):
                    for l in range(cdim):
                        col = cdim * cmap_vals[e, k] + l
                        if col < ncols:
                            diag[row + j].insert(col)
                        else:
                            odiag[row + j].insert(col)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void add_entries_extruded(rset, rmap, cset, cmap,
                                      PetscInt row_offset,
                                      vector[vecset[PetscInt]]& diag,
                                      vector[vecset[PetscInt]]& odiag,
                                      bint should_block):
    cdef:
        PetscInt nrows, ncols, i, j, k, l, nent, e, start, end, layer
        PetscInt rarity, carity, row, col, rdim, cdim, layers, tmp_row
        PetscInt reps, crep, rrep
        PetscInt[:, ::1] rmap_vals, cmap_vals
        PetscInt[::1] roffset, coffset

    nent = rmap.iterset.exec_size

    if should_block:
        rdim = cdim = 1
    else:
        rdim = rset.cdim
        cdim = cset.cdim

    rmap_vals = rmap.values_with_halo
    cmap_vals = cmap.values_with_halo

    nrows = rset.size * rdim
    ncols = cset.size * cdim

    rarity = rmap.arity
    carity = cmap.arity

    roffset = rmap.offset
    coffset = cmap.offset

    layers = rmap.iterset.layers

    for region in rmap.iteration_region:
        # The rowmap will have an iteration region attached to
        # it specifying which bits of the "implicit" (walking
        # up the column) map we want.  This mostly affects the
        # range of the loop over layers, except in the
        # ON_INTERIOR_FACETS where we also have to "double" up
        # the map.
        start = 0
        end = layers - 1
        reps = 1
        if region.where == "ON_BOTTOM":
            end = 1
        elif region.where == "ON_TOP":
            start = layers - 2
        elif region.where == "ON_INTERIOR_FACETS":
            end = layers - 2
            reps = 2
        elif region.where != "ALL":
            raise RuntimeError("Unhandled iteration region %s", region)

        for e in range(nent):
            for i in range(rarity):
                tmp_row = rdim * (rmap_vals[e, i] + start * roffset[i])
                if tmp_row >= nrows:
                    continue
                tmp_row += row_offset
                for j in range(rdim):
                    for rrep in range(reps):
                        row = tmp_row + j + rdim*rrep*roffset[i]
                        for layer in range(start, end):
                            for k in range(carity):
                                for l in range(cdim):
                                    for crep in range(reps):
                                        col = cdim * (cmap_vals[e, k] +
                                                      (layer + crep) * coffset[k]) + l
                                        if col < ncols:
                                            diag[row].insert(col)
                                        else:
                                            odiag[row].insert(col)
                            row += rdim * roffset[i]


@cython.boundscheck(False)
@cython.cdivision(True)
def build_sparsity(object sparsity, bint parallel, bool block=True):
    """Build a sparsity pattern defined by a list of pairs of maps

    :arg sparsity: the Sparsity object to build a pattern for
    :arg parallel: Are we running in parallel?
    :arg block: Should we build a block sparsity

    The sparsity pattern is built from the outer products of the pairs
    of maps.  This code works for both the serial and (MPI-) parallel
    case, as well as for MixedMaps"""
    cdef:
        vector[vector[vecset[PetscInt]]] diag, odiag
        vecset[PetscInt].const_iterator it
        PetscInt nrows, ncols, i, cur_nrows, rarity
        PetscInt row_offset, row, val
        int c
        bint should_block = False
        bint make_rowptr = False
        bint alloc_diag

    rset, cset = sparsity.dsets

    if block and len(rset) == 1 and len(cset) == 1 and rset.cdim == cset.cdim:
        should_block = True

    if not (parallel or len(rset) > 1 or len(cset) > 1):
        make_rowptr = True

    if should_block:
        nrows = sum(s.size for s in rset)
    else:
        nrows = sum(s.cdim * s.size for s in rset)

    maps = sparsity.maps
    extruded = maps[0][0].iterset._extruded

    if nrows == 0:
        # We don't own any rows, return something appropriate.
        dummy = np.empty(0, dtype=IntType).reshape(-1)
        sparsity._d_nz = 0
        sparsity._o_nz = 0
        sparsity._d_nnz = dummy
        sparsity._o_nnz = dummy
        sparsity._rowptr = dummy
        sparsity._colidx = dummy

    # Exposition:
    # When building a monolithic sparsity for a mixed space, we build
    # the contributions from each column set separately and then sum
    # them up at the end.  This is because otherwise we need to carry
    # out communication to figure out which column entries are
    # actually off diagonal and which are not.
    diag = vector[vector[vecset[PetscInt]]](len(cset))
    for c in range(len(cset)):
        diag[c] = vector[vecset[PetscInt]](nrows)
    if parallel:
        odiag = vector[vector[vecset[PetscInt]]](len(cset))
        for c in range(len(cset)):
            odiag[c] = vector[vecset[PetscInt]](nrows)

    for rmaps, cmaps in maps:
        row_offset = 0
        for r, rmap in enumerate(rmaps):
            if should_block:
                rdim = 1
            else:
                rdim = rset[r].cdim
            # Memoryviews require writeable buffers
            rflag = set_writeable(rmap)
            for c, cmap in enumerate(cmaps):
                cflag = set_writeable(cmap)
                if not diag[c][row_offset].capacity():
                    if should_block:
                        ncols = cset[c].size
                    else:
                        ncols = cset[c].size * cset[c].cdim
                    # Preallocate set entries heuristically based on arity
                    cur_nrows = rset[r].size * rdim
                    rarity = rmap.arity
                    alloc_diag = r == c and sparsity._has_diagonal
                    for i in range(cur_nrows):
                        diag[c][row_offset + i].reserve(6*rarity)
                        if alloc_diag and i < ncols:
                            # Always allocate space for diagonal.
                            # Note we only add the row_offset to the
                            # index, not the inserted value, since
                            # when we walk over the column maps we
                            # don't add offsets.
                            diag[c][row_offset + i].insert(i)
                        if parallel:
                            odiag[c][row_offset + i].reserve(6*rarity)
                if extruded:
                    add_entries_extruded(rset[r], rmap,
                                         cset[c], cmap,
                                         row_offset,
                                         diag[c], odiag[c],
                                         should_block)
                else:
                    add_entries(rset[r], rmap,
                                cset[c], cmap,
                                row_offset,
                                diag[c], odiag[c],
                                should_block)
                restore_writeable(cmap, cflag)
            # Increment only by owned rows
            row_offset += rset[r].size * rdim
            restore_writeable(rmap, rflag)

    cdef np.ndarray[PetscInt, ndim=1] nnz = np.zeros(nrows, dtype=IntType)
    cdef np.ndarray[PetscInt, ndim=1] onnz = np.zeros(nrows, dtype=IntType)
    cdef np.ndarray[PetscInt, ndim=1] rowptr
    cdef np.ndarray[PetscInt, ndim=1] colidx
    cdef int nz, onz
    if make_rowptr:
        rowptr = np.empty(nrows + 1, dtype=IntType)
        rowptr[0] = 0
    else:
        # Can't build these, so create dummy arrays
        rowptr = np.empty(0, dtype=IntType).reshape(-1)
        colidx = np.empty(0, dtype=IntType).reshape(-1)

    nz = 0
    onz = 0
    for c in range(len(cset)):
        for row in range(nrows):
            val = diag[c][row].size()
            nnz[row] += val
            nz += val
    if parallel:
        for c in range(len(cset)):
            for row in range(nrows):
                val = odiag[c][row].size()
                onnz[row] += val
                onz += val

    if make_rowptr:
        colidx = np.empty(nz, dtype=IntType)
        assert diag.size() == 1, "Can't make rowptr for mixed monolithic mat"
        for row in range(nrows):
            diag[0][row].sort()
            rowptr[row+1] = rowptr[row] + nnz[row]
            i = rowptr[row]
            it = diag[0][row].begin()
            while it != diag[0][row].end():
                colidx[i] = deref(it)
                inc(it)
                i += 1

    sparsity._d_nz = nz
    sparsity._o_nz = onz
    sparsity._d_nnz = nnz
    sparsity._o_nnz = onnz
    sparsity._rowptr = rowptr
    sparsity._colidx = colidx


def fill_with_zeros(PETSc.Mat mat not None, dims, maps, set_diag=True):
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
    if set_diag:
        for i in range(nrow):
            if i < ncol:
                MatSetValuesLocal(mat.mat, 1, &i, 1, &i, &zero, PETSC_INSERT_VALUES)
    extruded = maps[0][0].iterset._extruded
    for pair in maps:
        # Iterate over row map values including value entries
        set_size = pair[0].iterset.exec_size
        if set_size == 0:
            continue
        # Memoryviews require writeable buffers
        rflag = set_writeable(pair[0])
        cflag = set_writeable(pair[1])
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
            # We only need the *4 if we have an ON_INTERIOR_FACETS
            # iteration region, but it doesn't hurt to make them all
            # bigger, since we can special case less code below.
            PetscCalloc1(4*rarity*carity*rdim*cdim, &values)
            # Row values (generally only rarity of these)
            PetscMalloc1(2 * rarity, &rvals)
            # Col values (generally only rarity of these)
            PetscMalloc1(2 * carity, &cvals)
            # Offsets (for walking up the column)
            PetscMalloc1(rarity, &roffset)
            PetscMalloc1(carity, &coffset)
            # Walk over the iteration regions on this map.
            if pair[0].iteration_region != pair[1].iteration_region:
                raise NotImplementedError("fill_with_zeros: iteration regions of row and col maps don't match")
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
        restore_writeable(pair[0], rflag)
        restore_writeable(pair[1], cflag)
        PetscFree(values)
    # Aaaand, actually finalise the assembly.
    mat.assemble()
