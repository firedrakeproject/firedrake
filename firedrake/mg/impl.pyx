# Low-level numbering for multigrid support
from firedrake.petsc import PETSc
from firedrake import dmplex
from pyop2.datatypes import IntType
import numpy as np
cimport numpy as np
import cython
cimport petsc4py.PETSc as PETSc

np.import_array()

include "../dmplexinc.pxi"
include "firedrakeimpl.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
def get_entity_renumbering(PETSc.DM plex, PETSc.Section section, entity_type):
    """
    Given a section numbering a type of topological entity, return the
    renumberings from original plex numbers to new firedrake numbers
    (and vice versa)

    :arg plex: The DMPlex object
    :arg section: The Section defining the renumbering
    :arg entity_type: The type of entity (either ``"cell"`` or
        ``"vertex"``)
    """
    cdef:
        PetscInt start, end, p, ndof, entity
        np.ndarray[PetscInt, ndim=1] old_to_new
        np.ndarray[PetscInt, ndim=1] new_to_old

    if entity_type == "cell":
        start, end = plex.getHeightStratum(0)
    elif entity_type == "vertex":
        start, end = plex.getDepthStratum(0)
    else:
        raise RuntimeError("Entity renumbering for entities of type %s not implemented",
                           entity_type)

    old_to_new = np.empty(end - start, dtype=PETSc.IntType)
    new_to_old = np.empty(end - start, dtype=PETSc.IntType)

    for p in range(start, end):
        CHKERR(PetscSectionGetDof(section.sec, p, &ndof))
        if ndof > 0:
            CHKERR(PetscSectionGetOffset(section.sec, p, &entity))
            new_to_old[entity] = p - start
            old_to_new[p - start] = entity

    return old_to_new, new_to_old


@cython.boundscheck(False)
@cython.wraparound(False)
def coarse_to_fine_nodes(Vc, Vf, np.ndarray[PetscInt, ndim=2, mode="c"] coarse_to_fine_cells):
    cdef:
        np.ndarray[PetscInt, ndim=2, mode="c"] fine_map, coarse_map, coarse_to_fine_map
        np.ndarray[PetscInt, ndim=1, mode="c"] coarse_offset, fine_offset
        PetscInt i, j, k, l, m, node, fine, layer
        PetscInt coarse_per_cell, fine_per_cell, fine_cell_per_coarse_cell, coarse_cells
        PetscInt layers
        bint extruded

    fine_map = Vf.cell_node_map().values
    coarse_map = Vc.cell_node_map().values

    fine_cell_per_coarse_cell = coarse_to_fine_cells.shape[1]
    extruded = Vc.extruded

    if extruded:
        coarse_offset = Vc.offset
        fine_offset = Vf.offset
        layers = Vc.mesh().layers - 1
    coarse_cells = coarse_map.shape[0]
    coarse_per_cell = coarse_map.shape[1]
    fine_per_cell = fine_map.shape[1]
    coarse_to_fine_map = np.full((Vc.dof_dset.total_size,
                                  fine_per_cell * fine_cell_per_coarse_cell),
                                 -1,
                                 dtype=IntType)
    for i in range(coarse_cells):
        for j in range(coarse_per_cell):
            node = coarse_map[i, j]
            if extruded:
                for layer in range(layers):
                    k = 0
                    for l in range(fine_cell_per_coarse_cell):
                        fine = coarse_to_fine_cells[i, l]
                        for m in range(fine_per_cell):
                            coarse_to_fine_map[node + coarse_offset[j]*layer, k] = (fine_map[fine, m] +
                                                                                    fine_offset[m]*layer)
                            k += 1
            else:
                k = 0
                for l in range(fine_cell_per_coarse_cell):
                    fine = coarse_to_fine_cells[i, l]
                    for m in range(fine_per_cell):
                        coarse_to_fine_map[node, k] = fine_map[fine, m]
                        k += 1

    return coarse_to_fine_map


@cython.boundscheck(False)
@cython.wraparound(False)
def fine_to_coarse_nodes(Vf, Vc, np.ndarray[PetscInt, ndim=2, mode="c"] fine_to_coarse_cells):
    cdef:
        np.ndarray[PetscInt, ndim=2, mode="c"] fine_map, coarse_map, fine_to_coarse_map
        np.ndarray[PetscInt, ndim=1, mode="c"] coarse_offset, fine_offset
        PetscInt i, j, k, node, layer, layers
        PetscInt coarse_per_cell, fine_per_cell, coarse_cell, fine_cells
        bint extruded

    fine_map = Vf.cell_node_map().values
    coarse_map = Vc.cell_node_map().values

    extruded = Vc.extruded

    if extruded:
        coarse_offset = Vc.offset
        fine_offset = Vf.offset
        layers = Vc.mesh().layers - 1

    fine_cells = fine_to_coarse_cells.shape[0]
    coarse_per_fine = fine_to_coarse_cells.shape[1]
    coarse_per_cell = coarse_map.shape[1]
    fine_per_cell = fine_map.shape[1]
    fine_to_coarse_map = np.full((Vf.dof_dset.total_size,
                                  coarse_per_fine*coarse_per_cell),
                                 -1,
                                 dtype=IntType)

    for i in range(fine_cells):
        for coarse_cell in fine_to_coarse_cells[i, :]:
            for j in range(fine_per_cell):
                node = fine_map[i, j]
                if extruded:
                    for layer in range(layers):
                        for k in range(coarse_per_cell):
                            fine_to_coarse_map[node + fine_offset[j]*layer, k] = (coarse_map[coarse_cell, k] +
                                                                                  coarse_offset[k]*layer)
                else:
                    for k in range(coarse_per_cell):
                        fine_to_coarse_map[node, k] = coarse_map[coarse_cell, k]

    return fine_to_coarse_map


def create_lgmap(PETSc.DM dm):
    """Create a local to global map for all points in the given DM.

    :arg dm: The DM to create the map for.

    Returns a petsc4py LGMap."""
    cdef:
        PETSc.IS iset = PETSc.IS()
        PETSc.LGMap lgmap = PETSc.LGMap()
        PetscInt *indices
        PetscInt i, size
        PetscInt start, end

    # Not necessary on one process
    if dm.comm.size == 1:
        return None
    CHKERR(DMPlexCreatePointNumbering(dm.dm, &iset.iset))
    CHKERR(ISLocalToGlobalMappingCreateIS(iset.iset, &lgmap.lgm))
    CHKERR(ISLocalToGlobalMappingGetSize(lgmap.lgm, &size))
    CHKERR(ISLocalToGlobalMappingGetBlockIndices(lgmap.lgm, <const PetscInt**>&indices))
    for i in range(size):
        if indices[i] < 0:
            indices[i] = -(indices[i]+1)

    CHKERR(ISLocalToGlobalMappingRestoreBlockIndices(lgmap.lgm, <const PetscInt**>&indices))

    return lgmap


# Exposition:
#
# These next functions compute maps from coarse mesh cells to fine
# mesh cells and provide a consistent vertex reordering of each fine
# cell inside each coarse cell.  In parallel, this is somewhat
# complicated because the DMs only provide information about
# relationships between non-overlapped meshes, and we only have
# overlapped meshes.  We there need to translate non-overlapped DM
# numbering into overlapped-DM numbering and vice versa, as well as
# translating between firedrake numbering and DM numbering.
#
# A picture is useful here to make things clearer.
#
# To translate between overlapped and non-overlapped DM points, we
# need to go via global numbers (which don't change)
#
#      DM_orig<--.    ,-<--DM_new
#         |      |    |      |
#     L2G v  G2L ^    v L2G  ^ G2L
#         |      |    |      |
#         '-->-->Global-->---'
#
# Mapping between Firedrake numbering and DM numbering is carried out
# by computing the section permutation `get_entity_renumbering` above.
#
#            .->-o2n->-.
#      DM_new          Firedrake
#            `-<-n2o-<-'
#
# Finally, coarse to fine maps are produced on the non-overlapped DM
# and subsequently composed with the appropriate sequence of maps to
# get to Firedrake numbering (and vice versa).
#
#     DM_orig_coarse
#           |
#           v coarse_to_fine_cells [coarse_cell = floor(fine_cell / 2**tdim)]
#           |
#      DM_orig_fine
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def coarse_to_fine_cells(mc, mf, clgmaps, flgmaps):
    """Return a map from (renumbered) cells in a coarse mesh to those
    in a refined fine mesh.

    :arg mc: the coarse mesh to create the map from.
    :arg mf: the fine mesh to map to.
    :arg clgmaps: coarse lgmaps (non-overlapped and overlapped)
    :arg flgmaps: fine lgmaps (non-overlapped and overlapped)
    :returns: Two arrays, one mapping coarse to fine cells, the second fine to coarse cells.
    """
    cdef:
        PETSc.DM cdm, fdm
        PetscInt cStart, cEnd, c, val, dim, nref, ncoarse
        PetscInt i, ccell, fcell, nfine
        np.ndarray[PetscInt, ndim=2, mode="c"] coarse_to_fine
        np.ndarray[PetscInt, ndim=2, mode="c"] fine_to_coarse
        np.ndarray[PetscInt, ndim=1, mode="c"] co2n, fn2o, idx

    cdm = mc._plex
    fdm = mf._plex
    dim = cdm.getDimension()
    nref = 2 ** dim
    ncoarse = mc.cell_set.size
    nfine = mf.cell_set.size
    co2n, _ = get_entity_renumbering(cdm, mc._cell_numbering, "cell")
    _, fn2o = get_entity_renumbering(fdm, mf._cell_numbering, "cell")
    coarse_to_fine = np.full((ncoarse, nref), -1, dtype=PETSc.IntType)
    fine_to_coarse = np.full((nfine, 1), -1, dtype=PETSc.IntType)
    # Walk owned fine cells:
    cStart, cEnd = 0, nfine

    if mc.comm.size > 1:
        cno, co = clgmaps
        fno, fo = flgmaps
        # Compute global numbers of original cell numbers
        fo.apply(fn2o, result=fn2o)
        # Compute local numbers of original cells on non-overlapped mesh
        fn2o = fno.applyInverse(fn2o, PETSc.LGMap.MapMode.MASK)
        # Need to permute order of co2n so it maps from non-overlapped
        # cells to new cells (these may have changed order).  Need to
        # map all known cells through.
        idx = np.arange(mc.cell_set.total_size, dtype=PETSc.IntType)
        # LocalToGlobal
        co.apply(idx, result=idx)
        # GlobalToLocal
        # Drop values that did not exist on non-overlapped mesh
        idx = cno.applyInverse(idx, PETSc.LGMap.MapMode.DROP)
        co2n = co2n[idx]

    for c in range(cStart, cEnd):
        # get original (overlapped) cell number
        fcell = fn2o[c]
        # The owned cells should map into non-overlapped cell numbers
        # (due to parallel growth strategy)
        assert 0 <= fcell < cEnd

        # Find original coarse cell (fcell / nref) and then map
        # forward to renumbered coarse cell (again non-overlapped
        # cells should map into owned coarse cells)
        ccell = co2n[fcell // nref]
        assert 0 <= ccell < ncoarse
        fine_to_coarse[c, 0] = ccell
        for i in range(nref):
            if coarse_to_fine[ccell, i] == -1:
                coarse_to_fine[ccell, i] = c
                break
    return coarse_to_fine, fine_to_coarse


@cython.boundscheck(False)
@cython.wraparound(False)
def filter_exterior_facet_labels(PETSc.DM plex):
    """Remove exterior facet labels from things that aren't facets.

    When refining, every point "underneath" the refined entity
    receives its label.  But we want the facet label to really only
    apply to facets, so clear the labels from everything else."""
    cdef:
        PetscInt pStart, pEnd, fStart, fEnd, p, value
        PetscBool has_bdy_ids, has_bdy_faces
        DMLabel exterior_facets = NULL
        DMLabel boundary_ids = NULL
        DMLabel boundary_faces = NULL

    pStart, pEnd = plex.getChart()
    fStart, fEnd = plex.getHeightStratum(1)

    # Plex will always have an exterior_facets label (maybe
    # zero-sized), but may not always have boundary_ids or
    # boundary_faces.
    has_bdy_ids = plex.hasLabel(dmplex.FACE_SETS_LABEL)
    has_bdy_faces = plex.hasLabel("boundary_faces")

    CHKERR(DMGetLabel(plex.dm, <const char*>b"exterior_facets", &exterior_facets))
    if has_bdy_ids:
        label = dmplex.FACE_SETS_LABEL.encode()
        CHKERR(DMGetLabel(plex.dm, <const char*>label, &boundary_ids))
    if has_bdy_faces:
        CHKERR(DMGetLabel(plex.dm, <const char*>b"boundary_faces", &boundary_faces))
    for p in range(pStart, pEnd):
        if p < fStart or p >= fEnd:
            CHKERR(DMLabelGetValue(exterior_facets, p, &value))
            if value >= 0:
                CHKERR(DMLabelClearValue(exterior_facets, p, value))
            if has_bdy_ids:
                CHKERR(DMLabelGetValue(boundary_ids, p, &value))
                if value >= 0:
                    CHKERR(DMLabelClearValue(boundary_ids, p, value))
            if has_bdy_faces:
                CHKERR(DMLabelGetValue(boundary_faces, p, &value))
                if value >= 0:
                    CHKERR(DMLabelClearValue(boundary_faces, p, value))
