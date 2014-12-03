# Low-level numbering for multigrid support
from firedrake.petsc import PETSc
from pyop2 import MPI
import numpy as np
cimport numpy as np
import cython
cimport petsc4py.PETSc as PETSc

np.import_array()

include "../dmplex.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
def get_entity_renumbering(PETSc.DM plex, PETSc.Section section, entity_type):
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
        np.ndarray[PetscInt, ndim=1, mode="c"] co2n, cn2o, fo2n, fn2o

    cdm = mc._plex
    fdm = mf._plex
    dim = cdm.getDimension()
    nref = 2 ** dim
    ncoarse = mc.cell_set.size
    nfine = mf.cell_set.size
    co2n, cn2o  = get_entity_renumbering(cdm, mc._cell_numbering, "cell")
    fo2n, fn2o  = get_entity_renumbering(fdm, mf._cell_numbering, "cell")
    coarse_to_fine = np.empty((ncoarse, nref), dtype=PETSc.IntType)
    coarse_to_fine[:] = -1

    # Walk owned fine cells:
    fStart, fEnd = 0, nfine
    for c in range(fStart, fEnd):
        # get original (overlapped) cell number
        fcell = fn2o[c]
        # The owned cells should map into non-overlapped cell numbers
        # (due to parallel growth strategy)
        assert fcell < fEnd

        # Find original coarse cell (fcell / nref) and then map
        # forward to renumbered coarse cell (again non-overlapped
        # cells should map into owned coarse cells)
        ccell = co2n[fcell / nref]
        assert ccell < ncoarse
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
        PetscInt vcStart, vcEnd, vfStart, vfEnd, cshift, fshift,
        PetscInt ncoarse, nfine, ccell, fcell, i, j, k, vtx, ovtx, fcvtx, cvtx
        PetscInt nvertex, ofcell
        bint found
        np.ndarray[PetscInt, ndim=2, mode="c"] new_c2f = -np.ones_like(c2f)
        np.ndarray[PetscInt, ndim=1, mode="c"] inv_cvertex, fvertex, indices
        np.ndarray[PetscInt, ndim=2, mode="c"] cvertices, fvertices
        np.ndarray[PetscInt, ndim=2, mode="c"] vertex_perm
    coarse = P1c.mesh()
    fine = P1f.mesh()

    if coarse.ufl_cell().cellname() != "triangle":
        raise NotImplementedError("Only implemented for triangles, sorry")
    ncoarse = coarse.cell_set.size
    nfine = fine.cell_set.size
    cshift = coarse.cell_set.total_size - ncoarse
    fshift = fine.cell_set.total_size - nfine

    vcStart, vcEnd = coarse._plex.getDepthStratum(0)
    vfStart, vfEnd = fine._plex.getDepthStratum(0)

    # Get renumbering to original (plex) vertex numbers
    _, inv_cvertex = get_entity_renumbering(coarse._plex,
                                            coarse._vertex_numbering, "vertex")
    fvertex, _ = get_entity_renumbering(fine._plex,
                                        fine._vertex_numbering, "vertex")

    # Get map from coarse points into corresponding fine mesh points.
    # Note this is only valid for "owned" entities (non-overlapped)
    indices = coarse._fpointIS.indices

    cvertices = P1c.cell_node_map().values

    fmap = P1f.cell_node_map().values
    fvertices = P1f.cell_node_map().values
    nvertex = P1c.cell_node_map().arity
    vertex_perm = -np.ones((ncoarse, nvertex*4), dtype=PETSc.IntType)
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
                    fcvtx = fvertex[indices[inv_cvertex[cvtx] + vcStart - cshift]
                                    - vfStart + fshift]
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
                    fcvtx = fvertex[indices[inv_cvertex[cvtx] + vcStart - cshift]
                                    - vfStart + fshift]
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
