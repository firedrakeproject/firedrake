import cython
import numpy
from collections import namedtuple
cimport numpy
cimport petsc4py.PETSc as PETSc

from mpi4py import MPI
cimport mpi4py.MPI as MPI

from patches cimport *

numpy.import_array()

class RaggedArray(tuple):

    @property
    def offset(self):
        return super(RaggedArray, self).__getitem__(0)

    @property
    def value(self):
        return super(RaggedArray, self).__getitem__(1)

    def __len__(self):
        return self.offset.shape[0] - 1

    def __getitem__(self, i):
        assert i <= len(self)
        return self.value[self.offset[i]:self.offset[i+1]]

    def __repr__(self):
        ret = ["RaggedArray(["]
        for i in range(len(self.offset) - 1):
            s = self.offset[i]
            e = self.offset[i+1]
            ret.append("  " + repr(self.value[s:e]))
        ret.append("])")
        return "\n".join(ret)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_cell_facet_patches(PETSc.DM dm, PETSc.Section cell_numbering):
    cdef:
        PetscInt i, j, k, c, ci, v, start, end
        PetscInt vStart, vEnd, nvtx
        PetscInt fStart, fEnd
        PetscInt cStart, cEnd
        PetscInt nfacet, ncell = 0, nclosure, nfacet_cell
        PetscBool flg, flg1, flg2
        PetscInt *closure = NULL
        PetscInt *facets = NULL
        PetscInt *facet_cells = NULL
        PetscInt fidx = 0
        bint boundary_facet
        hash_t ht
        khiter_t iter = 0, ret = 0
        DMLabel facet_label, core, non_core
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] csr_rows
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] csr_facet_rows
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] csr_facets
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] csr_cells
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] tmp

    vStart, vEnd = dm.getDepthStratum(0)
    fStart, fEnd = dm.getHeightStratum(1)
    cStart, cEnd = dm.getHeightStratum(0)


    DMGetLabel(dm.dm, <char *>"exterior_facets", &facet_label)
    DMLabelCreateIndex(facet_label, fStart, fEnd)
    DMGetLabel(dm.dm, <char *>"op2_core", &core)
    DMGetLabel(dm.dm, <char *>"op2_non_core", &non_core)
    start, end = dm.getChart()
    DMLabelCreateIndex(core, start, end)
    DMLabelCreateIndex(non_core, start, end)

    nvtx = 0
    for v in range(vStart, vEnd):
        DMLabelHasPoint(core, v, &flg1)
        DMLabelHasPoint(non_core, v, &flg2)
        if flg1 or flg2:
            nvtx += 1
    csr_rows = numpy.empty(nvtx + 1, dtype=numpy.int32)
    csr_rows[0] = 0

    # Count number of cells per patch
    # This needs to change if we want a different determination of a
    # "patch".
    for v in range(vStart, vEnd):
        DMLabelHasPoint(core, v, &flg1)
        DMLabelHasPoint(non_core, v, &flg2)
        if not (flg1 or flg2):
            continue
        # Get iterated support of the vertex (star)
        DMPlexGetTransitiveClosure(dm.dm, v, PETSC_FALSE,
                                   &nclosure, &closure)
        for ci in range(nclosure):
            if cStart <= closure[2*ci] < cEnd:
                ncell += 1
        # Store cell counts
        csr_rows[v - vStart + 1] = ncell

    # Allocate cells
    csr_cells = numpy.empty(ncell, dtype=numpy.int32)
    csr_facet_rows = numpy.empty(nvtx + 1, dtype=numpy.int32)
    csr_facet_rows[0] = 0
    # Guess at how many boundary facets there will be
    csr_facets = numpy.empty(1 + ncell/4, dtype=numpy.int32)

    ht = kh_init(32)
    for v in range(vStart, vEnd):
        DMLabelHasPoint(core, v, &flg1)
        DMLabelHasPoint(non_core, v, &flg2)
        if not (flg1 or flg2):
            continue
        kh_clear(32, ht)
        DMPlexGetTransitiveClosure(dm.dm, v, PETSC_FALSE,
                                   &nclosure, &closure)
        i = 0
        start = csr_rows[v - vStart]
        end = csr_rows[v - vStart + 1]
        # Store patch's cells
        for ci in range(nclosure):
            if cStart <= closure[2*ci] < cEnd:
                iter = kh_put(32, ht, closure[2*ci], &ret)
                csr_cells[start + i] = closure[2*ci]
                i += 1

        # Determine boundary
        for i in range(start, end):
            # Get the cell's facets
            DMPlexGetCone(dm.dm, csr_cells[i], <const PetscInt **>&facets)
            # How many facets on the cell
            DMPlexGetConeSize(dm.dm, csr_cells[i], &nfacet)
            for j in range(nfacet):
                DMLabelHasPoint(facet_label, facets[j], &flg)
                if flg:
                    # facet is on domain boundary, don't select it
                    continue
                # Get the cells incident to the facet (2 of them)
                DMPlexGetSupport(dm.dm, facets[j], <const PetscInt **>&facet_cells)
                DMPlexGetSupportSize(dm.dm, facets[j], &nfacet_cell)
                # On process boundary, therefore also on subdomain
                # boundary if we only have one cell.
                boundary_facet = nfacet_cell == 1
                if not boundary_facet:
                    for k in range(nfacet_cell):
                        iter = kh_get(32, ht, facet_cells[k])
                        if iter != kh_end(ht):
                            # Facet's cell is inside patch
                            continue
                        boundary_facet = True
                        break
                # Cell not in patch, therefore facet on boundary
                if boundary_facet:
                    # Realloc if necessary
                    if fidx == csr_facets.shape[0]:
                        tmp = csr_facets
                        csr_facets = numpy.empty(int(fidx * 1.5),
                                                 dtype=numpy.int32)
                        csr_facets[:fidx] = tmp
                    csr_facets[fidx] = facets[j]
                    fidx += 1
        csr_facet_rows[v - vStart + 1] = fidx
    kh_destroy(32, ht)
    # Truncate
    csr_facets = csr_facets[:fidx].copy()
    for i in range(ncell):
        # Convert from PETSc numbering to Firedrake numbering
        PetscSectionGetOffset(cell_numbering.sec, csr_cells[i], &c)
        csr_cells[i] = c

    if closure != NULL:
        DMPlexRestoreTransitiveClosure(dm.dm, 0, PETSC_FALSE, NULL, &closure)

    return RaggedArray([csr_rows, csr_cells]), RaggedArray([csr_facet_rows, csr_facets])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_dof_patches(PETSc.DM dm, PETSc.Section dof_section,
                    numpy.ndarray[numpy.int32_t, ndim=2, mode="c"] cell_node_map,
                    numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] bc_nodes,
                    cells,
                    facets):

    cdef:
        PetscInt i, j, c, ci, f, start, end, p
        PetscInt gdof, ldof, bcdof
        PetscInt dof_per_cell, size
        PetscInt offset
        PetscInt dofidx = 0, gidx = 0, bcidx = 0
        PetscInt nclosure
        PetscInt *closure = NULL
        PetscInt *bc_patch = NULL
        PetscBool flg
        hash_t ht, global_bcs, local_bcs
        khiter_t iter = 0, ret = 0
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] csr_cell_rows = cells.offset
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] csr_cells = cells.value
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] csr_facet_rows = facets.offset
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] csr_facets = facets.value
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] patch_dofs
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] patch_rows
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] bc_dofs
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] bc_rows
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] global_dofs
        numpy.ndarray[numpy.int32_t, ndim=1, mode="c"] global_rows

    dof_per_cell = cell_node_map.shape[1]
    idx = len(csr_cell_rows) - 1
    patch_dofs = numpy.empty(csr_cell_rows[idx] * dof_per_cell, dtype=numpy.int32)
    patch_rows = numpy.empty_like(csr_cell_rows)
    global_dofs = numpy.empty(1 + dof_per_cell*csr_cell_rows.shape[0], dtype=numpy.int32)
    global_rows = numpy.empty_like(csr_cell_rows)
    bc_dofs = numpy.empty(1 + csr_cell_rows[idx], dtype=numpy.int32)
    bc_rows = numpy.empty_like(csr_cell_rows)

    patch_rows[0] = 0
    global_rows[0] = 0
    bc_rows[0] = 0

    global_bcs = kh_init(32)
    for i in range(bc_nodes.shape[0]):
        bc = bc_nodes[i]
        iter = kh_put(32, global_bcs, bc, &ret)

    ht = kh_init(32)
    local_bcs = kh_init(32)
    for p in range(csr_cell_rows.shape[0] - 1):
        # Reset for new patch
        kh_clear(32, ht)
        kh_clear(32, local_bcs)
        start = csr_cell_rows[p]
        end = csr_cell_rows[p+1]
        size = 0
        # Determine patch cell->node map
        for i in range(end - start):
            c = csr_cells[i + start]
            for j in range(dof_per_cell):
                gdof = cell_node_map[c, j]
                ldof = -1
                iter = kh_get(32, ht, gdof)
                if iter != kh_end(ht):
                    ldof = kh_val(ht, iter)
                if ldof == -1:
                    ldof = size
                    size += 1
                    iter = kh_put(32, ht, gdof, &ret)
                    kh_set_val(ht, iter, ldof)
                patch_dofs[dofidx] = ldof
                # This dof is globally a bc, so make it a patch
                # local bc.
                iter = kh_get(32, global_bcs, gdof)
                if iter != kh_end(global_bcs):
                    iter = kh_put(32, local_bcs, ldof, &ret)
                dofidx += 1
        patch_rows[p + 1] = dofidx

        # Realloc?
        if gidx + size >= global_dofs.shape[0]:
            tmp = global_dofs
            global_dofs = numpy.empty(int((gidx + size) * 1.5),
                                       dtype=numpy.int32)
            global_dofs[:tmp.shape[0]] = tmp

        iter = kh_begin(ht)
        while iter != kh_end(ht):
            if kh_exist(ht, iter):
                gdof = kh_key(ht, iter)
                offset = kh_val(ht, iter)
                global_dofs[gidx + offset] = gdof
            iter += 1
        gidx += size
        global_rows[p + 1] = gidx

        # Now build the facet related stuff
        start = csr_facet_rows[p]
        end = csr_facet_rows[p+1]
        for i in range(start, end):
            f = csr_facets[i]
            DMPlexGetTransitiveClosure(dm.dm, f, PETSC_TRUE, &nclosure, &closure)
            for ci in range(nclosure):
                PetscSectionGetDof(dof_section.sec, closure[2*ci], &size)
                if size > 0:
                    PetscSectionGetOffset(dof_section.sec, closure[2*ci], &offset)
                    for j in range(size):
                        # All the dofs on the boundary facets are part
                        # of the local bcs
                        iter = kh_get(32, ht, offset + j)
                        if iter != kh_end(ht):
                            ldof = kh_val(ht, iter)
                            iter = kh_put(32, local_bcs, ldof, &ret)
                        else:
                            raise RuntimeError

        # Allocate bcs data structure
        size = kh_size(local_bcs)
        PetscMalloc1(size, &bc_patch)
        j = 0
        # Get the local dofs
        iter = kh_begin(local_bcs)
        while iter != kh_end(local_bcs):
            if kh_exist(local_bcs, iter):
                bc_patch[j] = kh_key(local_bcs, iter)
                j += 1
            iter += 1
        assert j == size
        PetscSortInt(size, bc_patch)
        if bcidx + size >= bc_dofs.shape[0]:
            tmp = bc_dofs
            bc_dofs = numpy.empty(int((bcidx + size) * 1.5),
                                     dtype=numpy.int32)
            bc_dofs[:tmp.shape[0]] = tmp

        for j in range(size):
            bc_dofs[bcidx + j] = bc_patch[j]
        bcidx += size
        bc_rows[p + 1] = bcidx
        PetscFree(bc_patch)

    kh_destroy(32, local_bcs)
    kh_destroy(32, global_bcs)
    kh_destroy(32, ht)
    global_dofs = global_dofs[:gidx].copy()
    bc_dofs = bc_dofs[:bcidx].copy()
    if closure != NULL:
        DMPlexRestoreTransitiveClosure(dm.dm, 0, PETSC_FALSE, NULL, &closure)

    return RaggedArray([patch_rows, patch_dofs]), RaggedArray([global_rows, global_dofs]), \
        RaggedArray([bc_rows, bc_dofs])


cdef inline void insert_forward(PETSc.Vec g, PETSc.Vec l, PETSc.IS iset) nogil:
    cdef:
        const PetscScalar *garr
        PetscScalar *arr
        const PetscInt *indices
        PetscInt bs
        PetscInt nind
        PetscInt i, j, idx

    ISGetBlockSize(iset.iset, &bs)
    ISBlockGetLocalSize(iset.iset, &nind)
    ISBlockGetIndices(iset.iset, &indices)
    VecGetArrayRead(g.vec, &garr)
    VecGetArray(l.vec, &arr)
    for i in range(nind):
        for j in range(bs):
            arr[bs*i + j] = garr[bs*indices[i] + j]
    ISBlockRestoreIndices(iset.iset, &indices)
    VecRestoreArrayRead(g.vec, &garr)
    VecRestoreArray(l.vec, &arr)


cdef inline void add_reverse(PETSc.Vec l, PETSc.Vec g, PETSc.IS iset) nogil:
    cdef:
        const PetscScalar *arr
        PetscScalar *garr
        const PetscInt *indices
        PetscInt bs
        PetscInt nind
        PetscInt i, j

    ISGetBlockSize(iset.iset, &bs)
    ISBlockGetLocalSize(iset.iset, &nind)
    ISBlockGetIndices(iset.iset, &indices)
    VecGetArrayRead(l.vec, &arr)
    VecGetArray(g.vec, &garr)
    for i in range(nind):
        for j in range(bs):
            garr[bs*indices[i] + j] += arr[bs*i + j]
    ISBlockRestoreIndices(iset.iset, &indices)
    VecRestoreArrayRead(l.vec, &arr)
    VecRestoreArray(g.vec, &garr)


def apply_patch(self, PETSc.Vec x, PETSc.Vec y):
    cdef:
        PETSc.SF sf = self._sf
        MPI.Datatype dtype = self._mpi_type
        PetscInt i, j, k, num_patches
        PETSc.Vec ly, b, local
        PETSc.KSP ksp
        PETSc.Mat Ai
        PETSc.IS bcind
        PETSc.IS gind
        const PetscInt *bcindices
        PetscInt nind
        PetscInt bs
        PetscScalar *arr
        const PetscScalar *xarr
    ctx = self.ctx
    local = self._local
    y.set(0)
    local.set(0)

    g2l_begin(sf, x, local, dtype)
    g2l_end(sf, x, local, dtype)

    num_patches = len(ctx.matrices)
    for i in range(num_patches):
        ly = self._ys[i]
        b = self._bs[i]
        bcind = ctx.bc_patches[i]
        gind = ctx.glob_patches[i]
        ksp = self.ksps[i]

        insert_forward(local, b, gind)

        ISBlockGetIndices(bcind.iset, &bcindices)
        ISGetBlockSize(bcind.iset, &bs)
        ISBlockGetLocalSize(bcind.iset, &nind)
        VecGetArray(b.vec, &arr)
        for j in range(nind):
            for k in range(bs):
                arr[bs*bcindices[j] + k] = 0
        ISBlockRestoreIndices(bcind.iset, &bcindices)
        VecRestoreArray(b.vec, &arr)

        KSPSolve(ksp.ksp, b.vec, ly.vec)

    local.set(0)
    for i in range(num_patches):
        ly = self._ys[i]
        gind = ctx.glob_patches[i]
        add_reverse(ly, local, gind)

    l2g_begin(sf, local, y, dtype)
    l2g_end(sf, local, y, dtype)
    VecGetArrayRead(x.vec, &xarr)
    VecGetArray(y.vec, &arr)
    bcind = ctx.bc_nodes
    ISBlockGetIndices(bcind.iset, &bcindices)
    ISGetBlockSize(bcind.iset, &bs)
    ISBlockGetLocalSize(bcind.iset, &nind)
    for i in range(nind):
        for j in range(bs):
            idx = bs*bcindices[i] + j
            arr[idx] = xarr[idx]
    ISBlockRestoreIndices(bcind.iset, &bcindices)
    VecRestoreArrayRead(x.vec, &xarr)
    VecRestoreArray(y.vec, &arr)


def g2l_begin(PETSc.SF sf, PETSc.Vec gvec, PETSc.Vec lvec,
              MPI.Datatype dtype):
    cdef:
        const PetscScalar *garray
        PetscScalar *larray

    VecGetArray(lvec.vec, &larray)
    VecGetArrayRead(gvec.vec, &garray)

    PetscSFBcastBegin(sf.sf, dtype.ob_mpi, garray, larray)

    VecRestoreArray(lvec.vec, &larray)
    VecRestoreArrayRead(gvec.vec, &garray)


def g2l_end(PETSc.SF sf, PETSc.Vec gvec, PETSc.Vec lvec,
            MPI.Datatype dtype):
    cdef:
        const PetscScalar *garray
        PetscScalar *larray

    VecGetArray(lvec.vec, &larray)
    VecGetArrayRead(gvec.vec, &garray)

    PetscSFBcastEnd(sf.sf, dtype.ob_mpi, garray, larray)

    VecRestoreArray(lvec.vec, &larray)
    VecRestoreArrayRead(gvec.vec, &garray)


def l2g_begin(PETSc.SF sf, PETSc.Vec lvec, PETSc.Vec gvec,
              MPI.Datatype dtype):
    cdef:
        MPI.Op op = MPI.SUM
        const PetscScalar *larray
        PetscScalar *garray

    VecGetArrayRead(lvec.vec, &larray)
    VecGetArray(gvec.vec, &garray)

    PetscSFReduceBegin(sf.sf, dtype.ob_mpi, larray, garray, op.ob_mpi)

    VecRestoreArrayRead(lvec.vec, &larray)
    VecRestoreArray(gvec.vec, &garray)


def l2g_end(PETSc.SF sf, PETSc.Vec lvec, PETSc.Vec gvec,
            MPI.Datatype dtype):
    cdef:
        MPI.Op op = MPI.SUM
        const PetscScalar *larray
        PetscScalar *garray

    VecGetArrayRead(lvec.vec, &larray)
    VecGetArray(gvec.vec, &garray)

    PetscSFReduceEnd(sf.sf, dtype.ob_mpi, larray, garray, op.ob_mpi)

    VecRestoreArrayRead(lvec.vec, &larray)
    VecRestoreArray(gvec.vec, &garray)
