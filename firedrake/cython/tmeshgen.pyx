# cython: language_level=3

import functools
import cython
import numpy as np
import firedrake
from firedrake.petsc import PETSc
from mpi4py import MPI
from firedrake.utils import IntType, ScalarType
from libc.string cimport memset
from libc.stdlib cimport qsort
from tsfc.finatinterface import as_fiat_cell

cimport numpy as np
cimport mpi4py.MPI as MPI
cimport petsc4py.PETSc as PETSc

np.import_array()

include "petschdr.pxi"


# Do nurbs first.
# Tsplines would require removing constrained DoFs in create_section.
# +---+---+
# |   |b  |
# |  a+-- +
# |   |c  |
# +---+---+
# Remove "a" as these are not linearly independent?
# or remove "b" and "c"?
# Support is the same.
# Can do the same for hanging nodes?


# section should now be cached with edofs_key instead of nodes_per_entity,
# as entity_dofs is now anisotropic.


# Make:
# -- idx_section
# -- dof_section
# -- 
#
# Make another section for periodic coordinates.
#




@cython.boundscheck(False)
@cython.wraparound(False)
def tmesh_create_section(PETSc.DM dm,
                         entity_dofs,
                         PetscInt block_size=1,
                         dm_renumbering=None):
    """Create the section describing a global numbering.

    :arg mesh: The mesh.
    :arg nodes_per_entity: Number of nodes on each
        type of topological entity of the mesh.  Or, if the mesh is
        extruded, the number of nodes on, and on top of, each
        topological entity in the base mesh.
    :arg on_base: If True, assume extruded space is actually Foo x Real.
    :arg block_size: The integer by which nodes_per_entity is uniformly multiplied
        to get the true data layout.

    :returns: A PETSc Section providing the number of dofs, and offset
        of each dof, on each mesh point.


    Notes
    -----
    ``dm`` represents T-mesh, i.e., pure topology, where each face/edge
    is aligned with the parametric coordinate axes, e.g., xi, eta, and zeta in 3D.

    This function assumes that dm cells are oriented with respect to
    the parametric coordinate axes with orientation 0.

    The "local element" is always to be mapped to the T-mesh with orientation 0.

    """
    cdef:
        PETSc.DM dm
        PETSc.Section section
        PETSc.IS dm_renumbering
        PetscInt i, p, layers, pStart, pEnd
        PetscInt dimension, ndof
        np.ndarray[PetscInt, ndim=2, mode="c"] nodes
        np.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        bint variable, extruded, on_base_

    nodes_per_entity = np.asarray(nodes_per_entity, dtype=IntType)
    section = PETSc.Section().create(comm=mesh._comm)
    get_chart(dm.dm, &pStart, &pEnd)
    section.setChart(pStart, pEnd)
    if dm_renumbering is not None:
        CHKERR(PetscSectionSetPermutation(section.sec, dm_renumbering.iset))
    dimension = get_topological_dimension(dm)
    nodes = nodes_per_entity.reshape(dimension + 1, -1)
    for i in range(dimension + 1):
        get_depth_stratum(dm.dm, i, &pStart, &pEnd)
        ndof = nodes[i, 0]
        for p in range(pStart, pEnd):
            CHKERR(PetscSectionSetDof(section.sec, p, block_size * ndof))
    section.setUp()
    return section


@cython.boundscheck(False)
@cython.wraparound(False)
def _tmesh_nurbs_base(PetscInt dim,
                      length_tuple,
                      nelem_tuple,
                      degree_tuple,
                      periodic_tuple,
):
    """
    """
    cdef:
        PetscInt d, i, q, koff = 0, qoff = 0, ppp_sum = 0, mod = 1
        PetscScalar *lll
        PetscInt *nnn, *ppp, *mmm, *periodic
        np.ndarray[PetscInt, ndim=1, mode="c"] kkk
        PETSc.IS global_knot_vector_sizes
        PETSc.Vec global_knot_vectors
        PetscScalar *global_knot_arrays
        PETSc.DM dm
        PetscInt pStart, pEnd, p, cStart, cEnd
        PETSc.Section idx_sec
        PETSc.IS idx_iset
        np.ndarray[PetscInt, ndim=1, mode="c"] idx_array
        PETSc.Section dof_sec
        PETSc.Vec w_vec, X_vec
        PetscScalar *w_array, *X_array
        PetscInt rank = <PetscInt>comm.rank

    assert len(length_tuple) == dim
    assert len(nelem_tuple) == dim
    assert len(degree_tuple) == dim
    assert len(periodic_tuple) == dim
    CHKERR(PetscMalloc5(dim, &lll, dim, &nnn, dim, &ppp, dim, &mmm, dim, &periodic))
    for d in range(dim):
        lll[d] = <PetscScalar>length_tuple[d]
        nnn[d] = <PetscInt>nelem_tuple[d]
        ppp[d] = <PetscInt>degree_tuple[d]
        periodic[d] = <PetscInt>periodic_tuple[d]
    # Define the global knot-vectors.
    # This is a concatenation of x and y global knot-vectors.
    kkk = np.array([nnn[d] + (1 - periodic[d]) * (1 + 2 * ppp[d]) for d in range(dim)], dtype=IntType)
    global_knot_vector_sizes = PETSc.IS().createGeneral(kkk, comm=PETSc.COMM_SELF)
    global_knot_vectors = PETSc.Vec().create(comm=dm.comm)
    global_knot_vectors.setType(PETSc.Vec.Type.STANDARD)
    if rank == 0:
        global_knot_vectors.setSizes((kkk.sum(), PETSc.DETERMINE), 1)
    else:
        global_knot_vectors.setSizes((0, PETSc.DETERMINE), 1)
    CHKERR(VecGetArray(global_knot_vectors.vec, &global_knot_arrays))
    if rank == 0:
        for d in range(dim):
            if periodic[d] == 0:
                for i in range(ppp[d]):
                    global_knot_arrays[koff + i] = 0.
                for i in range(ppp[d], kkk[d] - ppp[d]):
                    global_knot_arrays[koff + i] = (lll[d] / nnn[d]) * (i - ppp[d])
                for i in range(kkk[d] - ppp[d], kkk[d]):
                    global_knot_arrays[koff + i] = lll[d]
            else:
                for i in range(kkk[d]):
                    global_knot_arrays[koff + i] = (lll[d] / nnn[d]) * i
            koff += kkk[d]
    CHKERR(VecRestoreArray(global_knot_vectors.vec, &global_knot_arrays))
    # Define the "mesh" defined by knot-spans.
    # Here, we are only interested in the topology and the coordinates will be ignored.
    # Some cells may represent cells with zero volume (cells that we do not loop over).
    for d in range(dim):
        mmm[d] = nnn[d] + (1 - periodic[d]) * (ppp[d] // 2) * 2
    dm = PETSc.DMPlex().createBoxMesh(tuple(mmm[d] for d in range(dim)),
                                      lower=tuple(0. for _ in range(dim)),  # not significant
                                      upper=tuple(1. for _ in range(dim)),  # not significant
                                      simplex=False,
                                      periodic=periodic_tuple,
                                      interpolate=True, comm=comm)
    # Define the entity-local (local-)knot-indices.
    # Here, just use knot-indices = global_knot_indices.
    idx_sec = PETSc.Section().create(comm=PETSc.COMM_SELF)
    dof_sec = PETSc.Section().create(comm=PETSc.COMM_SELF)
    pStart, pEnd = dm.getChart()
    idx_sec.setChart(pStart, pEnd)
    dof_sec.setChart(pStart, pEnd)
    # The following assumes even degrees.
    if any(ppp[d] % 2 == 1 for d in range(dim)):
        # TODO: Should manually make the dm instead of using BoxMesh to know exactly where each point is located.
        raise NotImplementedError("All degrees must be even.")
    for d in range(dim):
        ppp_sum += ppp[d] + 2
    cStart, cEnd = dm.getHeightStratum(0)
    if rank == 0:
        for p in range(cStart, cEnd):
            CHKERR(PetscSectionSetDof(idx_sec.sec, p, ppp_sum))
            CHKERR(PetscSectionSetDof(dof_sec.sec, p, 1))
    idx_sec.setUp()
    dof_sec.setUp()
    idx_array = np.empty(idx_sec.getStorageSize(), dtype=IntType)  # knot indices needed to make each basis function.
    if rank == 0:
        mod = 1
        qoff = 0
        for d in range(dim):
            if periodic[d] == 0:
                for p in range(cEnd - cStart):
                    i = p // mod % mmm[d]:
                    for q in range(ppp[d] + 2):
                        idx_array[ppp_sum * p + qoff + q] = i + q
            else:
                for p in range(cEnd - cStart):
                    i = p // mod % mmm[d]:
                    for q in range(ppp[d] + 2):
                        idx_array[ppp_sum * p + qoff + q] = (- ppp[d] // 2 + i + q) % kkk[d]
            mod *= mmm[d]
            qoff += ppp[d] + 2
    idx_iset = PETSc.IS().createGeneral(idx_array, comm=PETSc.COMM_SELF)
    # Separately treat periodic case.
    # For periodicity, gather active dofs for a given cell
    # Can "order" active dofs on each cell using knot indices: sort (ixmin, iymin, izmin).
    w_vec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
    w_vec.setType(PETSc.Vec.Type.STANDARD)
    w_vec.setSizes((dof_sec.getStorageSize(), PETSc.DETERMINE), 1)
    X_vec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
    X_vec.setType(PETSc.Vec.Type.STANDARD)
    X_vec.setSizes((dim * dof_sec.getStorageSize(), PETSc.DETERMINE), bsize=dim)
    CHKERR(VecGetArray(w_vec.vec, &w_array))
    CHKERR(VecGetArray(X_vec.vec, &X_array))
    if rank == 0:
        for d in range(dim):
            for i in range(mmm[d]):
        for ix in range(mx):
            for iy in range(my):
                poffset = ((px + 2) + (py + 2)) * (mx * iy + ix)  # cell order in BoxMesh
                w_array[] =
                X_array[] =# what to do with periodic -> Use IGACoordElem (DG like element; discontinuous across periodic boundary).
    CHKERR(VecRestoreArray(w_vec.vec, &w_array))
    CHKERR(VecRestoreArray(X_vec.vec, &X_array))
    CHKERR(PetscFree5(lll, nnn, ppp, mmm, periodic))
    return dm, global_knot_vector_sizes, global_knot_vectors, idx_sec, idx_iset, dof_sec, w_vec, X_vec


@cython.boundscheck(False)
@cython.wraparound(False)
def _tmesh_square_with_hole(PETSc.DM dm):
    """
    Builds the DoF mapping.

    :arg mesh: The mesh
    :arg global_numbering: Section describing the global DoF numbering
    :arg entity_dofs: FInAT element entity dofs for the cell
    :arg entity_permutations: FInAT element entity permutations for the cell
    :arg offset: offsets for each entity dof walking up a column.

    Preconditions: This function assumes that cell_closures contains mesh
    entities ordered by dimension, i.e. vertices first, then edges, faces, and
    finally the cell. For quadrilateral meshes, edges corresponding to
    dimension (0, 1) in the FInAT element must precede edges corresponding to
    dimension (1, 0) in the FInAT element.
    """
    cdef:
        PETSc.DM  dmDist
        PetscInt  dim = 2
        PetscInt *c0 = [11, 12, 13, 14]
        PetscInt *c1 = [15, 16, 17, 18]
        PetscInt *c2 = [19, 18, 20, 21]
        PetscInt *c11 = [3, 4]
        PetscInt *c12 = [3, 7]
        PetscInt *c13 = [7, 8]
        PetscInt *c14 = [4, 8]
        PetscInt *c15 = [7, 10]
        PetscInt *c16 = [7, 5]
        PetscInt *c17 = [5, 9]
        PetscInt *c18 = [10, 9]
        PetscInt *c19 = [10, 8]
        PetscInt *c20 = [9, 6]
        PetscInt *c21 = [8, 6]
        DMLabel   label
        PetscInt  numValues, maxValues = 0, i
        PetscInt  rank = <PetscInt>dm.comm.rank

    CHKERR(DMSetDimension(dm.dm, dim))
    if rank == 0:
        CHKERR(DMPlexSetChart(dm.dm, 0, 22))
        CHKERR(DMPlexSetConeSize(dm.dm, 0, 4))
        CHKERR(DMPlexSetConeSize(dm.dm, 1, 4))
        CHKERR(DMPlexSetConeSize(dm.dm, 2, 4))
        CHKERR(DMPlexSetConeSize(dm.dm, 11, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 12, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 13, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 14, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 15, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 16, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 17, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 18, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 19, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 20, 2))
        CHKERR(DMPlexSetConeSize(dm.dm, 21, 2))
    CHKERR(DMSetUp(dm.dm))
    if rank == 0:
        CHKERR(DMPlexSetCone(dm.dm, 0, c0))
        CHKERR(DMPlexSetCone(dm.dm, 1, c1))
        CHKERR(DMPlexSetCone(dm.dm, 2, c2))
        CHKERR(DMPlexSetCone(dm.dm, 11, c11))
        CHKERR(DMPlexSetCone(dm.dm, 12, c12))
        CHKERR(DMPlexSetCone(dm.dm, 13, c13))
        CHKERR(DMPlexSetCone(dm.dm, 14, c14))
        CHKERR(DMPlexSetCone(dm.dm, 15, c15))
        CHKERR(DMPlexSetCone(dm.dm, 16, c16))
        CHKERR(DMPlexSetCone(dm.dm, 17, c17))
        CHKERR(DMPlexSetCone(dm.dm, 18, c18))
        CHKERR(DMPlexSetCone(dm.dm, 19, c19))
        CHKERR(DMPlexSetCone(dm.dm, 20, c20))
        CHKERR(DMPlexSetCone(dm.dm, 21, c21))
    # CHKERR(DMPlexSymmetrize(dm))
    CHKERR(DMCreateLabel(dm.dm, b"depth"))
    CHKERR(DMGetLabel(dm.dm, b"depth", &label))
    if rank == 0:
      for i in range(22):
        if i < 3:
            CHKERR(DMLabelSetValue(label, i, 2))
        elif i < 11:
            CHKERR(DMLabelSetValue(label, i, 0))
        elif i < 22:
            CHKERR(DMLabelSetValue(label, i, 1))
    CHKERR(DMLabelGetNumValues(label, &numValues))
    maxValues = dm.comm.tompi4py().allreduce(numValues, op=MPI.MAX)
    for i in range(numValues, maxValues):
        CHKERR(DMLabelAddStratum(label, i))
    #CHKERR(DMPlexDistribute(dm.dm, 1, NULL, &dmDist))
    #if (dmDist) {
    #  CHKERR(DMDestroy(&dm))
    #  dm = dmDist
    #}
