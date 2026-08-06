# cython: language_level=3
import cython
import numpy as np
from firedrake.petsc import PETSc
from firedrake.utils import IntType
from libc.stdint cimport uintptr_t

include "petschdr.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
def create_star_points(PETSc.DM dm, seeds, offsets):
    """Collect the mesh points in the star of each seed point.

    These are the points PCPatch solves for with ``-pc_patch_construct_type
    star``. Seeds that this process does not own are skipped, so a patch can
    come out empty.

    Parameters
    ----------
    dm : PETSc.DM
        The mesh topology.
    seeds : numpy.ndarray
        The mesh points to build the patches around, patch by patch.
    offsets : numpy.ndarray
        Where each patch starts in ``seeds``, of length one more than the
        number of patches.

    Returns
    -------
    tuple of numpy.ndarray
        The star of every owned seed by decreasing topological dimension, the
        offsets saying where each star starts in them, the offsets saying which
        stars make up each patch, and the owned seeds themselves, one per star.

    """
    cdef:
        PetscInt npatch = len(offsets) - 1
        PetscInt npoints = 0, nstars = 0, maxpoints = 0
        PetscInt v, k, s, seed, starSize = 0
        PetscInt *star = NULL
        PetscInt *points = NULL
        PetscInt *newpoints = NULL
        PetscInt pStart = 0, pEnd = 0
        PetscInt[::1] cseeds = np.asarray(seeds, dtype=IntType)
        PetscInt[::1] coffsets = np.asarray(offsets, dtype=IntType)
        PetscInt[::1] cstar_offsets = np.zeros(cseeds.shape[0] + 1, dtype=IntType)
        PetscInt[::1] cowned_seeds = np.zeros(cseeds.shape[0], dtype=IntType)
        PetscInt[::1] cpatch_offsets = np.zeros(npatch + 1, dtype=IntType)
        PetscInt[::1] view
        DMLabel ghost = NULL
        PetscBool isghost = PETSC_FALSE

    CHKERR(DMPlexGetChart(dm.dm, &pStart, &pEnd))
    # Only build patches around the points this process owns, exactly as PCPatch does
    ghost_name = b"pyop2_ghost"
    if dm.hasLabel("pyop2_ghost"):
        CHKERR(DMGetLabel(dm.dm, <const char*>ghost_name, &ghost))
        CHKERR(DMLabelCreateIndex(ghost, pStart, pEnd))

    maxpoints = 16 * (cseeds.shape[0] + 1)
    CHKERR(PetscMalloc1(maxpoints, &points))
    try:
        for v in range(npatch):
            cpatch_offsets[v] = nstars
            for k in range(coffsets[v], coffsets[v+1]):
                seed = cseeds[k]
                if ghost != NULL:
                    CHKERR(DMLabelHasPoint(ghost, seed, &isghost))
                    if isghost == PETSC_TRUE:
                        continue
                CHKERR(DMPlexGetTransitiveClosure(dm.dm, seed, PETSC_FALSE, &starSize, &star))
                while npoints + starSize > maxpoints:
                    maxpoints *= 2
                    CHKERR(PetscMalloc1(maxpoints, &newpoints))
                    for s in range(npoints):
                        newpoints[s] = points[s]
                    CHKERR(PetscFree(points))
                    points = newpoints
                    newpoints = NULL
                # By decreasing topological dimension, so that a patch lists its
                # interiors first and its vertices last
                for s in range(starSize - 1, -1, -1):
                    points[npoints] = star[2*s]
                    npoints += 1
                CHKERR(DMPlexRestoreTransitiveClosure(dm.dm, seed, PETSC_FALSE, &starSize, &star))
                cowned_seeds[nstars] = seed
                nstars += 1
                cstar_offsets[nstars] = npoints
        cpatch_offsets[npatch] = nstars

        star_points = np.empty(npoints, dtype=IntType)
        if npoints > 0:
            view = star_points
            for s in range(npoints):
                view[s] = points[s]
    finally:
        CHKERR(PetscFree(points))
        if ghost != NULL:
            CHKERR(DMLabelDestroyIndex(ghost))
    return (star_points,
            np.asarray(cstar_offsets)[:nstars + 1],
            np.asarray(cpatch_offsets),
            np.asarray(cowned_seeds)[:nstars])


@cython.boundscheck(False)
@cython.wraparound(False)
def create_closure_points(PETSc.DM dm, points, offsets):
    """Collect the mesh points in the closure of each group of points.

    Applied to the stars of `create_star_points`, these are the points PCPatch
    solves for with ``-pc_patch_construct_type vanka``.

    Parameters
    ----------
    dm : PETSc.DM
        The mesh topology.
    points : numpy.ndarray
        The mesh points to close over, group by group.
    offsets : numpy.ndarray
        Where each group starts in ``points``, of length one more than the
        number of groups.

    Returns
    -------
    tuple of numpy.ndarray
        The closure of every group by decreasing topological dimension, and the
        offsets saying where each closure starts in them.

    """
    cdef:
        PetscInt ngroups = len(offsets) - 1
        PetscInt npoints = 0, maxpoints = 0, start = 0
        PetscInt g, k, q, p, closureSize = 0
        PetscInt *closure = NULL
        PetscInt *points_ = NULL
        PetscInt *newpoints = NULL
        PetscInt *seen = NULL
        PetscInt pStart = 0, pEnd = 0
        PetscInt[::1] cpoints = np.asarray(points, dtype=IntType)
        PetscInt[::1] coffsets = np.asarray(offsets, dtype=IntType)
        PetscInt[::1] cclosure_offsets = np.zeros(ngroups + 1, dtype=IntType)
        PetscInt[::1] view

    CHKERR(DMPlexGetChart(dm.dm, &pStart, &pEnd))
    # A point already taken carries the group that took it, one-based so that
    # the zeroed initial state marks a point no group has taken
    CHKERR(PetscCalloc1(pEnd - pStart, &seen))
    maxpoints = 16 * (cpoints.shape[0] + 1)
    CHKERR(PetscMalloc1(maxpoints, &points_))
    try:
        for g in range(ngroups):
            start = npoints
            # Against the order of the group, so that the closure of its lowest
            # dimensional point comes first and survives the deduplication
            for k in range(coffsets[g+1] - 1, coffsets[g] - 1, -1):
                CHKERR(DMPlexGetTransitiveClosure(dm.dm, cpoints[k], PETSC_TRUE, &closureSize, &closure))
                while npoints + closureSize > maxpoints:
                    maxpoints *= 2
                    CHKERR(PetscMalloc1(maxpoints, &newpoints))
                    for q in range(npoints):
                        newpoints[q] = points_[q]
                    CHKERR(PetscFree(points_))
                    points_ = newpoints
                    newpoints = NULL
                for q in range(closureSize):
                    p = closure[2*q]
                    if seen[p - pStart] == g + 1:
                        continue
                    seen[p - pStart] = g + 1
                    points_[npoints] = p
                    npoints += 1
                CHKERR(DMPlexRestoreTransitiveClosure(dm.dm, cpoints[k], PETSC_TRUE, &closureSize, &closure))
            # Undo the reversed walk, leaving the closure by decreasing topological dimension
            for k in range((npoints - start) // 2):
                p = points_[start + k]
                points_[start + k] = points_[npoints - 1 - k]
                points_[npoints - 1 - k] = p
            cclosure_offsets[g+1] = npoints

        closure_points = np.empty(npoints, dtype=IntType)
        if npoints > 0:
            view = closure_points
            for k in range(npoints):
                view[k] = points_[k]
    finally:
        CHKERR(PetscFree(points_))
        CHKERR(PetscFree(seen))
    return closure_points, np.asarray(cclosure_offsets)


@cython.boundscheck(False)
@cython.wraparound(False)
def create_patch_ises(points, offsets, sections, bsizes, indices):
    """Gather the degrees of freedom carried by the mesh points of each patch.

    Parameters
    ----------
    points : list of numpy.ndarray
        The mesh points each subspace reads, patch by patch. Subspaces reading
        the same points may share one array.
    offsets : list of numpy.ndarray
        Where each patch starts in the points of each subspace, each of length
        one more than the number of patches.
    sections : list of PETSc.Section
        The local section of each subspace.
    bsizes : list of int
        The block size of each subspace.
    indices : list of numpy.ndarray
        The local indices of each subspace within the mixed space, negative
        where the degree of freedom is masked out.

    Returns
    -------
    list of PETSc.IS
        The degrees of freedom of each patch, in local numbering.

    Raises
    ------
    ValueError
        If the subspaces disagree on the number of patches, or if a mesh point
        appears more than once in a patch.

    """
    cdef:
        PetscInt nsub = len(sections)
        PetscInt npatch = 0
        PetscInt ndofs = 0, maxdofs = 0
        PetscInt i, k, p, v, dof, off, bs, index
        PetscInt *dofs = NULL
        PetscInt *cbs = NULL
        PetscInt **cindices = NULL
        PetscInt **cpoints = NULL
        PetscInt **coffsets = NULL
        PETSc.PetscSection *csections = NULL
        PetscInt[::1] view

    if nsub > 0:
        npatch = len(offsets[0]) - 1
        if any(len(offsets[i]) - 1 != npatch for i in range(nsub)):
            raise ValueError("The subspaces disagree on the number of patches")

    CHKERR(PetscMalloc1(nsub, &csections))
    CHKERR(PetscMalloc1(nsub, &cindices))
    CHKERR(PetscMalloc1(nsub, &cpoints))
    CHKERR(PetscMalloc1(nsub, &coffsets))
    CHKERR(PetscMalloc1(nsub, &cbs))
    # The memoryviews must outlive the pointers taken out of them
    views = [np.asarray(indices[i], dtype=IntType) for i in range(nsub)]
    point_views = [np.asarray(points[i], dtype=IntType) for i in range(nsub)]
    offset_views = [np.asarray(offsets[i], dtype=IntType) for i in range(nsub)]
    for i in range(nsub):
        csections[i] = (<PETSc.Section?>sections[i]).sec
        cbs[i] = bsizes[i]
        view = views[i]
        cindices[i] = &view[0] if view.shape[0] > 0 else NULL
        maxdofs += view.shape[0]
        view = point_views[i]
        cpoints[i] = &view[0] if view.shape[0] > 0 else NULL
        view = offset_views[i]
        coffsets[i] = &view[0]

    # No mesh point appears twice in a patch, so a patch cannot hold more degrees
    # of freedom than the process has
    CHKERR(PetscMalloc1(maxdofs, &dofs))
    ises = []
    try:
        for v in range(npatch):
            # Subspace by subspace, so that a patch lists its fields in order
            ndofs = 0
            for i in range(nsub):
                bs = cbs[i]
                for p in range(coffsets[i][v], coffsets[i][v+1]):
                    CHKERR(PetscSectionGetDof(csections[i], cpoints[i][p], &dof))
                    if dof <= 0:
                        continue
                    CHKERR(PetscSectionGetOffset(csections[i], cpoints[i][p], &off))
                    if ndofs + dof*bs > maxdofs:
                        raise ValueError("A mesh point appears more than once in a patch")
                    for k in range(off*bs, (off + dof)*bs):
                        index = cindices[i][k]
                        if index >= 0:
                            dofs[ndofs] = index
                            ndofs += 1

            patch = np.empty(ndofs, dtype=IntType)
            if ndofs > 0:
                view = patch
                for k in range(ndofs):
                    view[k] = dofs[k]
            ises.append(PETSc.IS().createGeneral(patch, comm=PETSc.COMM_SELF))
    finally:
        CHKERR(PetscFree(dofs))
        CHKERR(PetscFree(csections))
        CHKERR(PetscFree(cindices))
        CHKERR(PetscFree(cpoints))
        CHKERR(PetscFree(coffsets))
        CHKERR(PetscFree(cbs))
    return ises


def pcpatch_set_compute_operator(patch, function, ctx):
    CHKERR(PCPatchSetComputeOperator((<PETSc.PC?>patch).pc,
                                     <PetscPCPatchComputeOperator><uintptr_t>function,
                                     <void *><uintptr_t>ctx))


def pcpatch_set_compute_operator_interior_facets(patch, function, ctx):
    CHKERR(PCPatchSetComputeOperatorInteriorFacets((<PETSc.PC?>patch).pc,
                                                   <PetscPCPatchComputeOperator><uintptr_t>function,
                                                   <void *><uintptr_t>ctx))


def pcpatch_set_compute_operator_exterior_facets(patch, function, ctx):
    CHKERR(PCPatchSetComputeOperatorExteriorFacets((<PETSc.PC?>patch).pc,
                                                   <PetscPCPatchComputeOperator><uintptr_t>function,
                                                   <void *><uintptr_t>ctx))


def snespatch_set_compute_operator(patch, function, ctx):
    CHKERR(SNESPatchSetComputeOperator((<PETSc.SNES?>patch).snes,
                                        <PetscPCPatchComputeOperator><uintptr_t>function,
                                        <void *><uintptr_t>ctx))


def snespatch_set_compute_function(patch, function, ctx):
    CHKERR(SNESPatchSetComputeFunction((<PETSc.SNES?>patch).snes,
                                       <PetscPCPatchComputeFunction><uintptr_t>function,
                                       <void *><uintptr_t>ctx))
