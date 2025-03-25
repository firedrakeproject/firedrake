# cython: language_level=3

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
    ctypedef enum PetscBool:
        PETSC_TRUE, PETSC_FALSE
    ctypedef enum PetscInsertMode "InsertMode":
        PETSC_INSERT_VALUES "INSERT_VALUES"
    int PetscCalloc1(size_t, void*)
    int PetscMalloc1(size_t, void*)
    int PetscMalloc2(size_t, void*, size_t, void*)
    int PetscFree(void*)
    int PetscFree2(void*,void*)
    int MatSetValuesBlockedLocal(PETSc.PetscMat, PetscInt, PetscInt*, PetscInt, PetscInt*,
                                 PetscScalar*, PetscInsertMode)
    int MatSetValuesLocal(PETSc.PetscMat, PetscInt, PetscInt*, PetscInt, PetscInt*,
                          PetscScalar*, PetscInsertMode)
    int MatPreallocatorPreallocate(PETSc.PetscMat, PetscBool, PETSc.PetscMat)
    int MatXAIJSetPreallocation(PETSc.PetscMat, PetscInt, const PetscInt[], const PetscInt[],
                                const PetscInt[], const PetscInt[])

cdef extern from "petsc/private/matimpl.h":
    struct _p_Mat:
        void *data

ctypedef struct Mat_Preallocator:
    void *ht
    PetscInt *dnz
    PetscInt *onz

cdef extern from *:
    void PyErr_SetObject(object, object)
    void *PyExc_RuntimeError

cdef object PetscError = <object>PyExc_RuntimeError

cdef inline int SETERR(int ierr) with gil:
    if (<void*>PetscError) != NULL:
        PyErr_SetObject(PetscError, <long>ierr)
    else:
        PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)
    return ierr

cdef inline int CHKERR(int ierr) nogil except -1:
    if ierr == 0:
        return 0 # no error
    else:
        SETERR(ierr)
        return -1

cdef object set_writeable(map):
     flag = map.values_with_halo.flags['WRITEABLE']
     map.values_with_halo.setflags(write=True)
     return flag

cdef void restore_writeable(map, flag):
     map.values_with_halo.setflags(write=flag)


def get_preallocation(PETSc.Mat preallocator, PetscInt nrow):
    cdef:
        _p_Mat *A = <_p_Mat *>(preallocator.mat)
        Mat_Preallocator *p = <Mat_Preallocator *>(A.data)

    if p.dnz != NULL:
        dnz = <PetscInt[:nrow]>p.dnz
        dnz = np.asarray(dnz).copy()
    else:
        dnz = np.zeros(0, dtype=IntType)
    if p.onz != NULL:
        onz = <PetscInt[:nrow]>p.onz
        onz = np.asarray(onz).copy()
    else:
        onz = np.zeros(0, dtype=IntType)
    return dnz, onz


def build_sparsity(sparsity):
    rset, cset = sparsity.dsets
    mixed = len(rset) > 1 or len(cset) > 1
    nest = sparsity.nested
    if mixed and sparsity.nested:
        raise ValueError("Can't build sparsity on mixed nest, build the sparsity on the blocks")
    preallocator = PETSc.Mat().create(comm=sparsity.comm)
    preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
    if mixed:
        # Sparsity is the dof sparsity.
        nrows = rset.layout_vec.local_size
        ncols = cset.layout_vec.local_size
        preallocator.setLGMap(rmap=rset.unblocked_lgmap, cmap=cset.unblocked_lgmap)
    else:
        # Sparsity is the block sparsity
        nrows = rset.layout_vec.local_size // rset.layout_vec.block_size
        ncols = cset.layout_vec.local_size // cset.layout_vec.block_size
        preallocator.setLGMap(rmap=rset.scalar_lgmap, cmap=cset.scalar_lgmap)

    preallocator.setSizes(size=((nrows, None), (ncols, None)),
                          bsize=1)
    preallocator.setUp()

    if mixed:
        for i, r in enumerate(rset):
            for j, c in enumerate(cset):
                maps = sparsity.rcmaps.get((i, j), [])
                iter_regions = sparsity.iteration_regions.get((i, j), [])
                mat = preallocator.getLocalSubMatrix(isrow=rset.local_ises[i],
                                                     iscol=cset.local_ises[j])
                fill_with_zeros(mat, (r.cdim, c.cdim),
                                maps,
                                iter_regions,
                                set_diag=((i == j) and sparsity._has_diagonal))
                mat.assemble()
                preallocator.restoreLocalSubMatrix(isrow=rset.local_ises[i],
                                                   iscol=cset.local_ises[j],
                                                   submat=mat)
        preallocator.assemble()
        nnz, onnz = get_preallocation(preallocator, nrows)
    else:
        fill_with_zeros(preallocator, (1, 1),
                        sparsity.rcmaps[(0, 0)],
                        sparsity.iteration_regions[(0, 0)],
                        set_diag=sparsity._has_diagonal)
        preallocator.assemble()
        nnz, onnz = get_preallocation(preallocator, nrows)
        if not (sparsity._block_sparse and rset.cdim == cset.cdim):
            # We only build baij for the the square blocks, so unwind if we didn't
            nnz = nnz * cset.cdim
            nnz = np.repeat(nnz, rset.cdim)
            onnz = onnz * cset.cdim
            onnz = np.repeat(onnz, rset.cdim)
    preallocator.destroy()
    return nnz, onnz


def fill_with_zeros(PETSc.Mat mat not None, dims, maps, iteration_regions, set_diag=True):
    """Fill a PETSc matrix with zeros in all slots we might end up inserting into

    :arg mat: the PETSc Mat (must already be preallocated)
    :arg dims: the dimensions of the sparsity (block size)
    :arg maps: the pairs of maps defining the sparsity pattern

    You must call ``mat.assemble()`` after this call."""
    cdef:
        PetscInt rdim, cdim
        PetscScalar *values
        PetscScalar *diag_values
        int set_entry
        int set_size
        int region_selector
        bint constant_layers, extruded_periodic
        PetscInt layer_start, layer_end, layer_bottom, num_layers, effective_offset, layer
        PetscInt[:, ::1] layers
        PetscInt i, k, irem
        PetscInt nrow, ncol
        PetscInt rarity, carity, tmp_rarity, tmp_carity
        PetscInt[:, ::1] rmap, cmap, tempmap
        PetscInt **rcomposedmaps = NULL
        PetscInt **ccomposedmaps = NULL
        PetscInt nrcomposedmaps, nccomposedmaps, rset_entry, cset_entry
        PetscInt *rvals
        PetscInt *cvals
        PetscInt *roffset
        PetscInt *coffset
        PetscInt *roffset_quotient
        PetscInt *coffset_quotient

    from pyop2 import op2
    rdim, cdim = dims
    # Always allocate space for diagonal
    nrow, ncol = mat.getLocalSize()
    if set_diag:
        CHKERR(PetscCalloc1(rdim*cdim, &diag_values))
        for i in range(nrow // rdim):
            if i < ncol  // cdim:
                CHKERR(MatSetValuesBlockedLocal(mat.mat, 1, &i, 1, &i, diag_values, PETSC_INSERT_VALUES))
        CHKERR(PetscFree(diag_values))
    if len(maps) == 0:
        return
    extruded = maps[0][0].iterset._extruded
    for pair, iteration_region in zip(maps, iteration_regions):
        # Iterate over row map values including value entries
        set_size = pair[0].iterset.size
        if set_size == 0:
            continue
        rflags = []
        cflags = []
        if isinstance(pair[0], op2.ComposedMap):
            m = pair[0].flattened_maps[0]
            rflags.append(set_writeable(m))
            rmap = m.values_with_halo
            nrcomposedmaps = len(pair[0].flattened_maps) - 1
        else:
            rflags.append(set_writeable(pair[0]))  # Memoryviews require writeable buffers
            rmap = pair[0].values_with_halo  # Map values
            nrcomposedmaps = 0
        if isinstance(pair[1], op2.ComposedMap):
            m = pair[1].flattened_maps[0]
            cflags.append(set_writeable(m))
            cmap = m.values_with_halo
            nccomposedmaps = len(pair[1].flattened_maps) - 1
        else:
            cflags.append(set_writeable(pair[1]))
            cmap = pair[1].values_with_halo
            nccomposedmaps = 0
        # Handle ComposedMaps
        CHKERR(PetscMalloc2(nrcomposedmaps, &rcomposedmaps, nccomposedmaps, &ccomposedmaps))
        for i in range(nrcomposedmaps):
            m = pair[0].flattened_maps[1 + i]
            rflags.append(set_writeable(m))
            tempmap = m.values_with_halo
            rcomposedmaps[i] = &tempmap[0, 0]
        for i in range(nccomposedmaps):
            m = pair[1].flattened_maps[1 + i]
            cflags.append(set_writeable(m))
            tempmap = m.values_with_halo
            ccomposedmaps[i] = &tempmap[0, 0]
        # Arity of maps
        rarity = pair[0].arity
        carity = pair[1].arity
        if not extruded:
            # The non-extruded case is easy, we just walk over the
            # rmap and cmap entries and set a block of values.
            CHKERR(PetscCalloc1(rarity*carity*rdim*cdim, &values))
            for set_entry in range(set_size):
                rset_entry = <PetscInt>set_entry
                cset_entry = <PetscInt>set_entry
                for i in range(nrcomposedmaps):
                    rset_entry = rcomposedmaps[nrcomposedmaps - 1 - i][rset_entry]
                    if rset_entry < 0:
                        break
                if rset_entry < 0:
                    continue
                for i in range(nccomposedmaps):
                    cset_entry = ccomposedmaps[nccomposedmaps - 1 - i][cset_entry]
                    if cset_entry < 0:
                        break
                if cset_entry < 0:
                    continue
                CHKERR(MatSetValuesBlockedLocal(mat.mat, rarity, &rmap[<int>rset_entry, 0],
                                                carity, &cmap[<int>cset_entry, 0],
                                                values, PETSC_INSERT_VALUES))
        else:
            # The extruded case needs a little more work.
            layers = pair[0].iterset.layers_array
            constant_layers = pair[0].iterset.constant_layers
            extruded_periodic = pair[0].iterset._extruded_periodic
            # We only need the *4 if we have an ON_INTERIOR_FACETS
            # iteration region, but it doesn't hurt to make them all
            # bigger, since we can special case less code below.
            CHKERR(PetscCalloc1(4*rarity*carity*rdim*cdim, &values))
            # Row values (generally only rarity of these)
            CHKERR(PetscMalloc1(2 * rarity, &rvals))
            # Col values (generally only rarity of these)
            CHKERR(PetscMalloc1(2 * carity, &cvals))
            # Offsets (for walking up the column)
            CHKERR(PetscMalloc1(rarity, &roffset))
            CHKERR(PetscMalloc1(carity, &coffset))
            # Offset quotients (for walking up the column)
            CHKERR(PetscMalloc1(rarity, &roffset_quotient))
            CHKERR(PetscMalloc1(carity, &coffset_quotient))
            # Walk over the iteration regions on this map.
            for r in iteration_region:
                region_selector = -1
                tmp_rarity = rarity
                tmp_carity = carity
                if r == op2.ON_BOTTOM:
                    region_selector = 1
                elif r == op2.ON_TOP:
                    region_selector = 2
                elif r == op2.ON_INTERIOR_FACETS:
                    region_selector = 3
                    # Double up rvals and cvals (the map is over two
                    # cells, not one)
                    tmp_rarity *= 2
                    tmp_carity *= 2
                elif r != op2.ALL:
                    raise RuntimeError("Unhandled iteration region %s", r)
                for i in range(rarity):
                    roffset[i] = pair[0].offset[i]
                for i in range(carity):
                    coffset[i] = pair[1].offset[i]
                for i in range(rarity):
                    roffset_quotient[i] = 0 if pair[0].offset_quotient is None else pair[0].offset_quotient[i]
                for i in range(carity):
                    coffset_quotient[i] = 0 if pair[1].offset_quotient is None else pair[1].offset_quotient[i]
                for set_entry in range(set_size):
                    rset_entry = <PetscInt>set_entry
                    cset_entry = <PetscInt>set_entry
                    for i in range(nrcomposedmaps):
                        rset_entry = rcomposedmaps[nrcomposedmaps - 1 - i][rset_entry]
                        if rset_entry < 0:
                            break
                    if rset_entry < 0:
                        continue
                    for i in range(nccomposedmaps):
                        cset_entry = ccomposedmaps[nccomposedmaps - 1 - i][cset_entry]
                        if cset_entry < 0:
                            break
                    if cset_entry < 0:
                        continue
                    if constant_layers:
                        layer_start = layers[0, 0]
                        layer_end = layers[0, 1] - 1
                    else:
                        layer_start = layers[set_entry, 0]
                        layer_end = layers[set_entry, 1] - 1
                    layer_bottom = layer_start
                    num_layers = layer_end - layer_start
                    if region_selector == 1:
                        # Bottom, finish after first layer
                        layer_end = layer_start + 1
                    elif region_selector == 2:
                        # Top, start on penultimate layer
                        layer_start = layer_end - 1
                    elif region_selector == 3:
                        if not extruded_periodic:
                            # interior, finish on penultimate layer
                            layer_end = layer_end - 1
                    for layer in range(layer_start, layer_end):
                        # Make sure that the following cases are covered:
                        #
                        # - extrusion type            : standard, periodic
                        # - num_layers                : 1, 2, and N (general)
                        # - integration_type          : ON_INTERIOR_FACET, ALL
                        # - {r,c}offset_quotient[irem]: 0 and 1 (for FEM)
                        #
                        # For the standard extrusion, the following reduces to
                        # the conventional logic;
                        # note that {r,c}offset_quotient[:] == 0 in that case.
                        for i in range(tmp_rarity):
                            k = i // rarity  # always 0 if not ON_INTERIOR_FACETS
                            irem = i % rarity  # always i if not ON_INTERIOR_FACETS
                            effective_offset = layer + k + roffset_quotient[irem]
                            rvals[i] = rmap[<int>rset_entry, irem] + \
                                       roffset[irem] * (effective_offset % num_layers - roffset_quotient[irem] % num_layers)
                        for i in range(tmp_carity):
                            k = i // carity
                            irem = i % carity
                            effective_offset = layer + k + coffset_quotient[irem]
                            cvals[i] = cmap[<int>cset_entry, irem] + \
                                       coffset[irem] * (effective_offset % num_layers - coffset_quotient[irem] % num_layers)
                        CHKERR(MatSetValuesBlockedLocal(mat.mat, tmp_rarity, rvals,
                                                        tmp_carity, cvals,
                                                        values, PETSC_INSERT_VALUES))
            CHKERR(PetscFree(rvals))
            CHKERR(PetscFree(cvals))
            CHKERR(PetscFree(roffset))
            CHKERR(PetscFree(coffset))
            CHKERR(PetscFree(roffset_quotient))
            CHKERR(PetscFree(coffset_quotient))
        CHKERR(PetscFree2(rcomposedmaps, ccomposedmaps))
        if isinstance(pair[0], op2.ComposedMap):
            for m, rflag in zip(pair[0].flattened_maps, rflags):
                restore_writeable(m, rflag)
        else:
            restore_writeable(pair[0], rflags[0])
        if isinstance(pair[1], op2.ComposedMap):
            for m, cflag in zip(pair[1].flattened_maps, cflags):
                restore_writeable(m, cflag)
        else:
            restore_writeable(pair[1], cflags[0])
        CHKERR(PetscFree(values))
