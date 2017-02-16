# Low-level numbering for multigrid support
from __future__ import absolute_import, print_function, division

import FIAT
from tsfc.fiatinterface import create_element

from firedrake.petsc import PETSc
import firedrake.mg.utils as utils
from pyop2 import MPI
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
#
#
#     DM_orig_coarse
#           |
#           v coarse_to_fine_vertices (via DMPlexCreateCoarsePointIS)
#           |
#      DM_orig_fine
#
# Phew.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def coarse_to_fine_cells(mc, mf):
    """Return a map from (renumbered) cells in a coarse mesh to those
    in a refined fine mesh.

    :arg mc: the coarse mesh to create the map from.
    :arg mf: the fine mesh to map to.
    :arg parents: a Section mapping original fine cell numbers to
         their corresponding coarse parent cells"""
    cdef:
        PETSc.DM cdm, fdm
        PetscInt fStart, fEnd, c, val, dim, nref, ncoarse
        PetscInt i, ccell, fcell, nfine
        np.ndarray[PetscInt, ndim=2, mode="c"] coarse_to_fine
        np.ndarray[PetscInt, ndim=1, mode="c"] co2n, fn2o, idx

    cdm = mc._plex
    fdm = mf._plex
    dim = cdm.getDimension()
    nref = 2 ** dim
    ncoarse = mc.cell_set.size
    nfine = mf.cell_set.size
    co2n, _ = get_entity_renumbering(cdm, mc._cell_numbering, "cell")
    _, fn2o = get_entity_renumbering(fdm, mf._cell_numbering, "cell")
    coarse_to_fine = np.empty((ncoarse, nref), dtype=PETSc.IntType)
    coarse_to_fine[:] = -1

    # Walk owned fine cells:
    fStart, fEnd = 0, nfine

    if mc.comm.size > 1:
        # Compute global numbers of original cell numbers
        mf._overlapped_lgmap.apply(fn2o, result=fn2o)
        # Compute local numbers of original cells on non-overlapped mesh
        fn2o = mf._non_overlapped_lgmap.applyInverse(fn2o, PETSc.LGMap.MapType.MASK)
        # Need to permute order of co2n so it maps from non-overlapped
        # cells to new cells (these may have changed order).  Need to
        # map all known cells through.
        idx = np.arange(mc.cell_set.total_size, dtype=PETSc.IntType)
        # LocalToGlobal
        mc._overlapped_lgmap.apply(idx, result=idx)
        # GlobalToLocal
        # Drop values that did not exist on non-overlapped mesh
        idx = mc._non_overlapped_lgmap.applyInverse(idx, PETSc.LGMap.MapType.DROP)
        co2n = co2n[idx]

    for c in range(fStart, fEnd):
        # get original (overlapped) cell number
        fcell = fn2o[c]
        # The owned cells should map into non-overlapped cell numbers
        # (due to parallel growth strategy)
        assert 0 <= fcell < fEnd

        # Find original coarse cell (fcell / nref) and then map
        # forward to renumbered coarse cell (again non-overlapped
        # cells should map into owned coarse cells)
        ccell = co2n[fcell // nref]
        assert 0 <= ccell < ncoarse
        for i in range(nref):
            if coarse_to_fine[ccell, i] == -1:
                coarse_to_fine[ccell, i] = c
                break
    return coarse_to_fine


@cython.wraparound(False)
@cython.boundscheck(False)
def compute_orientations(P1c, P1f, np.ndarray[PetscInt, ndim=2, mode="c"] c2f):
    """Compute consistent orientations for refined cells

    :arg P1c: A P1 function space on the coarse mesh
    :arg P1f: A P1 function space on the fine mesh
    :arg c2f: A map from coarse to fine cells

    Returns a reordered map from coarse to fine cells (such that
    traversing the fine cells on a coarse cell always happens in a
    consistent order) and a permutation of the vertices of each fine
    cell such that the dofs on the each fine cell can also be
    traversed in a consistent order."""
    cdef:
        PetscInt vcStart_orig, vcEnd_orig, vfStart_orig, vfEnd_orig
        PetscInt vcStart_new, vcEnd_new, vfStart_new, vfEnd_new
        PetscInt ncoarse, nfine, ccell, fcell, i, j, k, vtx, ovtx, fcvtx, cvtx
        PetscInt nvertex, ofcell
        bint found
        np.ndarray[PetscInt, ndim=2, mode="c"] new_c2f = -np.ones_like(c2f)
        np.ndarray[PetscInt, ndim=1, mode="c"] cn2o, fo2n, indices
        np.ndarray[PetscInt, ndim=2, mode="c"] cvertices, fvertices
        np.ndarray[PetscInt, ndim=2, mode="c"] vertex_perm
    coarse = P1c.mesh()
    fine = P1f.mesh()

    if coarse.ufl_cell().cellname() not in ["interval", "triangle"]:
        raise NotImplementedError("Only implemented for intervals and triangles, sorry")
    ncoarse = coarse.cell_set.size
    nfine = fine.cell_set.size

    vcStart_new, vcEnd_new = coarse._plex.getDepthStratum(0)
    vfStart_new, vfEnd_new = fine._plex.getDepthStratum(0)
    vcStart_orig, vcEnd_orig = coarse._non_overlapped_nent[0]
    vfStart_orig, vfEnd_orig = fine._non_overlapped_nent[0]

    # Get renumbering to original (plex) vertex numbers
    _, cn2o = get_entity_renumbering(coarse._plex,
                                     coarse._vertex_numbering, "vertex")
    fo2n, _ = get_entity_renumbering(fine._plex,
                                     fine._vertex_numbering, "vertex")

    # Get map from coarse points into corresponding fine mesh points.
    # Note this is only valid for "owned" entities (non-overlapped)
    indices = coarse._fpointIS.indices[vcStart_orig:vcEnd_orig]

    cvertices = P1c.cell_node_map().values
    fvertices = P1f.cell_node_map().values
    if coarse.comm.size > 1:
        # Convert values in indices to points in the overlapped
        # (rather than non-overlapped) mesh.
        # Convert to global numbers
        fine._non_overlapped_lgmap.apply(indices, result=indices)
        # Send back to local numbers on the overlapped mesh
        indices = fine._overlapped_lgmap.applyInverse(indices,
                                                      PETSc.LGMap.MapType.MASK)
        indices -= vfStart_new

        # Need to map the new-to-old map back onto the original
        # (non-overlapped) plex points
        # Convert from vertex numbers to plex point numbers
        cn2o += vcStart_new
        # Go to global numbers
        coarse._overlapped_lgmap.apply(cn2o, result=cn2o)
        # Back to local numbers on the original (non-overlapped) mesh
        cn2o = coarse._non_overlapped_lgmap.applyInverse(cn2o,
                                                         PETSc.LGMap.MapType.MASK)
        # Go from point numbers back to vertex numbers
        cn2o -= vcStart_orig
        # Note that unlike in coarse_to_fine_cells, we don't need to
        # permute fo2n because we always access it using overlapped
        # plex point indices, rather than non-overlapped indices.
    else:
        indices = indices - vfStart_new

    nvertex = P1c.cell_node_map().arity
    dim = coarse._plex.getDimension()
    vertex_perm = -np.ones((ncoarse, nvertex*(2 ** dim)), dtype=PETSc.IntType)

    # Intervals
    if dim == 1:
        # Cell order (given coarse reference cell numbering) is:
        # 0---0---*---1---1
        for ccell in range(ncoarse):
            for fcell in range(2):
                found = False
                for j in range(nvertex):
                    if found:
                        break
                    vtx = fvertices[c2f[ccell, fcell], j]
                    for i in range(nvertex):
                        cvtx = cvertices[ccell, i]
                        fcvtx = fo2n[indices[cn2o[cvtx]]]
                        if vtx == fcvtx:
                            new_c2f[ccell, i] = c2f[ccell, fcell]
                            found = True
                            break
            # Having computed the fine cell ordering on this coarse cell,
            # we derive the permutation of each fine cell vertex.
            # Vertex order on each fine cell is given by:
            #
            # 0---0---a---1---1
            # 0_f => [0, a]
            # 1_f => [1, a]
            #
            for fcell in range(2):
                for i in range(nvertex):
                    vtx = fvertices[new_c2f[ccell, fcell], i]
                    found = False
                    for j in range(nvertex):
                        cvtx = cvertices[ccell, j]
                        fcvtx = fo2n[indices[cn2o[cvtx]]]
                        if vtx == fcvtx:
                            found = True
                            break
                    vertex_perm[ccell, fcell*nvertex + i] = 0 if found else 1
        return new_c2f, vertex_perm

    # Now triangles
    for ccell in range(ncoarse):
        for fcell in range(4):
            # Cell order (given coarse reference cell numbering) is as below:
            # 2
            # |\
            # | \
            # |  \
            # | 2 \
            # *----*
            # |\ 3 |\
            # | \  | \
            # |  \ |  \
            # | 0 \| 1 \
            # 0----*----1
            #
            found = False
            # Check if this cell shares a vertex with the coarse cell
            # In which case, if it shares coarse vertex i it is in position i.
            for j in range(nvertex):
                if found:
                    break
                vtx = fvertices[c2f[ccell, fcell], j]
                for i in range(nvertex):
                    cvtx = cvertices[ccell, i]
                    fcvtx = fo2n[indices[cn2o[cvtx]]]
                    if vtx == fcvtx:
                        new_c2f[ccell, i] = c2f[ccell, fcell]
                        found = True
                        break
            # Doesn't share any vertices, must be central cell, which comes last.
            if not found:
                new_c2f[ccell, 3] = c2f[ccell, fcell]
        # Having computed the fine cell ordering on this coarse cell,
        # we derive the permutation of each fine cell vertex.
        # Vertex order on each fine cell is given by:
        # 2
        # |\
        # | \
        # |  \
        # | 2 \
        # b----a
        # |\ 3 |\
        # | \  | \
        # |  \ |  \
        # | 0 \| 1 \
        # 0----c----1
        #
        # 0_f => [0, c, b]
        # 1_f => [1, a, c]
        # 2_f => [2, b, a]
        # 3_f => [a, b, c]
        #
        for fcell in range(3):
            # "Other" cell, vertex neither shared with coarse cell
            # vertex nor this cell is vertex 1, (the shared vertex is
            # vertex 2).
            ofcell = (fcell + 2) % 3
            for i in range(nvertex):
                vtx = fvertices[new_c2f[ccell, fcell], i]
                # Is this vertex shared with the coarse grid?
                found = False
                for j in range(nvertex):
                    cvtx = cvertices[ccell, j]
                    fcvtx = fo2n[indices[cn2o[cvtx]]]
                    if vtx == fcvtx:
                        found = True
                        break
                if found:
                    # Yes, this is vertex 0.
                    vertex_perm[ccell, fcell*nvertex + i] = 0
                    continue

                # Is this vertex shared with "other" cell
                found = False
                for j in range(nvertex):
                    ovtx = fvertices[new_c2f[ccell, ofcell], j]
                    if vtx == ovtx:
                        found = True
                        break
                if found:
                    # Yes, this is vertex 2.
                    vertex_perm[ccell, fcell*nvertex + i] = 2
                    # Find vertex in cell 3 that matches this one.
                    # It is numbered by the other cell's other
                    # cell.
                    for j in range(nvertex):
                        ovtx = fvertices[new_c2f[ccell, 3], j]
                        if vtx == ovtx:
                            vertex_perm[ccell, 3*nvertex + j] = (fcell + 4) % 3
                            break
                if not found:
                    # No, this is vertex 1
                    vertex_perm[ccell, fcell*nvertex + i] = 1
                    # Find vertex in cell 3 that matches this one.
                    # It is numbered by the "other" cell
                    for j in range(nvertex):
                        ovtx = fvertices[new_c2f[ccell, 3], j]
                        if vtx == ovtx:
                            vertex_perm[ccell, 3*nvertex + j] = ofcell
                            break
    return new_c2f, vertex_perm


cdef inline PetscInt hash_perm(PetscInt p0, PetscInt p1):
    if p0 == 0:
        if p1 == 1:
            return 0
        return 1
    if p0 == 1:
        if p1 == 0:
            return 2
        return 3
    if p0 == 2:
        if p1 == 0:
            return 4
        return 5


@cython.wraparound(False)
@cython.boundscheck(False)
def create_cell_node_map(coarse, fine, np.ndarray[PetscInt, ndim=2, mode="c"] c2f,
                         np.ndarray[PetscInt, ndim=2, mode="c"] vertex_perm):
    """Compute a map from coarse cells to fine nodes (dofs) in a consistent order

    :arg coarse: The coarse function space
    :arg fine: The fine function space
    :arg c2f: A map from coarse to fine cells
    :arg vertex_perm: A permutation of each of the fine cell vertices
        into a consistent order.
    """
    cdef:
        np.ndarray[PetscInt, ndim=1, mode="c"] indices, cell_map
        np.ndarray[PetscInt, ndim=2, mode="c"] permutations
        np.ndarray[PetscInt, ndim=2, mode="c"] new_cell_map, old_cell_map
        PetscInt ccell, fcell, ncoarse, ndof, i, j, perm, nfdof, nfcell, tdim

    cell = coarse.finat_element.cell
    if isinstance(cell, FIAT.reference_element.TensorProductCell):
        basecell, _ = cell.cells
        tdim = basecell.get_spatial_dimension()
    else:
        tdim = cell.get_spatial_dimension()

    ncoarse = coarse.mesh().cell_set.size
    ndof = coarse.cell_node_map().arity

    fiat_element = create_element(coarse.ufl_element(), vector_is_mixed=False)
    perms = utils.get_node_permutations(fiat_element)
    permutations = np.empty((len(perms), len(perms.values()[0])), dtype=np.int32)
    for k, v in perms.iteritems():
        if tdim == 1:
            permutations[k[0], :] = v[:]
        else:
            p0, p1 = np.asarray(k, dtype=PETSc.IntType)[0:2]
            permutations[hash_perm(p0, p1), :] = v[:]

    old_cell_map = fine.cell_node_map().values[c2f, ...].reshape(ncoarse, -1)

    # We're going to uniquify the maps we get out, so the first step
    # is to apply the permutation to one entry to find out which
    # indices we need to keep.
    indices, offset = utils.get_unique_indices(fiat_element,
                                               old_cell_map[0, :],
                                               vertex_perm[0, :],
                                               offset=fine.cell_node_map().offset)

    nfdof = indices.shape[0]
    nfcell = 2**tdim
    new_cell_map = -np.ones((ncoarse, nfdof), dtype=PETSc.IntType)

    cell_map = np.empty(nfcell*ndof, dtype=PETSc.IntType)
    for ccell in range(ncoarse):
        # 2**tdim fine cells per coarse
        for fcell in range(nfcell):
            if tdim == 1:
                perm = vertex_perm[ccell, fcell*2]
            else:
                perm = hash_perm(vertex_perm[ccell, fcell*3],
                                 vertex_perm[ccell, fcell*3 + 1])
            for j in range(ndof):
                cell_map[fcell*ndof + j] = old_cell_map[ccell, fcell*ndof +
                                                        permutations[perm, j]]
        for j in range(nfdof):
            new_cell_map[ccell, j] = cell_map[indices[j]]
    return new_cell_map, offset


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
    has_bdy_ids = plex.hasLabel("boundary_ids")
    has_bdy_faces = plex.hasLabel("boundary_faces")

    CHKERR(DMGetLabel(plex.dm, <char*>"exterior_facets", &exterior_facets))
    if has_bdy_ids:
        CHKERR(DMGetLabel(plex.dm, <char*>"boundary_ids", &boundary_ids))
    if has_bdy_faces:
        CHKERR(DMGetLabel(plex.dm, <char*>"boundary_faces", &boundary_faces))
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
