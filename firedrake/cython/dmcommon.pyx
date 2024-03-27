# cython: language_level=3

# Utility functions common to all DMs used in Firedrake
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

FACE_SETS_LABEL = "Face Sets"
CELL_SETS_LABEL = "Cell Sets"


def get_topological_dimension(PETSc.DM dm):
    """
    Get the topological dimension of a DMPlex or DMSwarm.  Assumes that
    a DMSwarm represents a mesh of vertices (tdim 0).

    :arg dm: A DMPlex or DMSwarm.

    :returns: For a DMPlex ``dm.getDimension()``, for a DMSwarm ``0``.
    """
    if isinstance(dm, PETSc.DMPlex):
        return dm.getDimension()
    elif isinstance(dm, PETSc.DMSwarm):
        return 0
    else:
        raise ValueError("dm must be a DMPlex or DMSwarm")


cdef inline void get_height_stratum(PETSc.PetscDM dm, PetscInt stratum, PetscInt *start, PetscInt *end):
    """
    Get the bounds [start, end) for all points at a certain height in
    the topology of a DM.

    :arg dm: The input ``PETSc.PetscDM`` - must be DMPlex or DMSwarm.
    :arg stratum: The height to get
    :arg start: Output parameter for the first point at the requested
        height
    :arg end: Output parameter for one beyond the last point at the
        requested height

    For DMPlex calls
    ``CHKERR(DMPlexGetHeightStratum(dm, stratum, start, end))``.

    For DMSwarm sets ``start[0] = 0`` then calls
    ``CHKERR(DMSwarmGetLocalSize(dm, end))`` if ``stratum`` is 0 otherwise
    sets ``end[0] = 0`` since a DMSwarm is assumed to consist only of
    vertices.
    """
    cdef:
        PetscBool isplex, isswarm

    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.PLEX.encode(), &isplex))
    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.SWARM.encode(), &isswarm))
    if isswarm:
        start[0] = 0
        if stratum == 0:
            CHKERR(DMSwarmGetLocalSize(dm, end))
        else:
            end[0] = 0
    elif isplex:
        CHKERR(DMPlexGetHeightStratum(dm, stratum, start, end))
    else:
        raise ValueError("dm must be a DMPlex or DMSwarm")


cdef inline void get_depth_stratum(PETSc.PetscDM dm, PetscInt stratum, PetscInt *start, PetscInt *end):
    """
    Get the bounds [start, end) for all points at a certain depth in
    the topology of a DM.

    :arg dm: The input ``PETSc.PetscDM`` - must be DMPlex or DMSwarm.
    :arg stratum: The depth to get
    :arg start: Output parameter for the first point at the requested
        depth
    :arg end: Output parameter for one beyond the last point at the
        requested depth

    For DMPlex calls
    ``CHKERR(DMPlexGetDepthStratum(dm, stratum, start, end))``.

    For DMSwarm sets ``start[0] = 0`` then calls
    ``CHKERR(DMSwarmGetLocalSize(dm, end))`` if ``stratum`` is 0 otherwise
    sets ``end[0] = 0`` since a DMSwarm is assumed to consist only of
    vertices.
    """
    cdef:
        PetscBool isplex, isswarm

    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.PLEX.encode(), &isplex))
    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.SWARM.encode(), &isswarm))
    if isswarm:
        start[0] = 0
        if stratum == 0:
            CHKERR(DMSwarmGetLocalSize(dm, end))
        else:
            end[0] = 0
    elif isplex:
        CHKERR(DMPlexGetDepthStratum(dm, stratum, start, end))
    else:
        raise ValueError("dm must be a DMPlex or DMSwarm")


cdef inline void get_transitive_closure(PETSc.PetscDM dm, PetscInt p, PetscBool useCone, PetscInt *nclosure, PetscInt **closure):
    """
    Get the points on the transitive closure of the in-edges or
    out-edges for the point ``p`` in the topology DAG.


    :arg dm: The input ``PETSc.PetscDM`` - must be DMPlex or DMSwarm.
    :arg p: A point which lies in the interval of all mesh entities.
    :arg useCone: ``PETSC_TRUE`` for in-edges, otherwise use out-edges.
    :arg nclosure: The number of points in the closure, so points[] is
        of size 2*nclosure.
    :arg closure: The points and point orientations, interleaved as pairs
        ``[p0, o0, p1, o1, ...]``. If ``closure`` is ``NULL`` on input,
        internal storage will be returned, otherwise the provided array
        is used.

    For DMPlex calls
    ``CHKERR(DMPlexGetTransitiveClosure(dm, p, useCone, nclosure, closure))``

    For DMSwarm sets ``nclosure[0] = 1``, allocates if needed, then sets
    ``closure[0][0] = p`` and ``closure[0][1] = 0`` (orientation is
    unimportant for a mesh of disconnected vertices).

    ..note:: If ``closure`` is ``NULL`` on input, be sure to call
             `restore_transitive_closure` to deallocate memory!
    """
    cdef:
        PetscBool isplex, isswarm

    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.PLEX.encode(), &isplex))
    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.SWARM.encode(), &isswarm))
    if isswarm:
        if closure[0] == NULL:
            CHKERR(PetscMalloc1(2, closure))
        if nclosure != NULL:
            # though not documented in DMPlexGetTransitiveClosure,
            # *nclosure can be NULL on input
            nclosure[0] = 1
        closure[0][0] = p
        closure[0][1] = 0
    elif isplex:
        CHKERR(DMPlexGetTransitiveClosure(dm, p, useCone, nclosure, closure))
    else:
        raise ValueError("dm must be a DMPlex or DMSwarm")


cdef inline void restore_transitive_closure(PETSc.PetscDM dm, PetscInt p, PetscBool useCone, PetscInt *nclosure, PetscInt **closure):
    """
    Restore the transitive closure accessed with get_transitive_closure

    :arg dm: The input ``PETSc.PetscDM`` - must be DMPlex or DMSwarm.
    :arg p: A point which lies in the interval of all mesh entities.
    :arg useCone: ``PETSC_TRUE`` for in-edges, otherwise use out-edges.
    :arg nclosure: The number of points in the closure, so points[] is
        of size 2*nclosure. Zero after calling this.
    :arg closure: The points and point orientations, interleaved as pairs
        ``[p0, o0, p1, o1, ...]``. Deallocated after calling this.

    For DMPlex calls
    ``CHKERR(DMPlexRestoreTransitiveClosure(dm, p, useCone, nclosure, closure))``

    For DMSwarm restores memory if allocated in get_transitive_closure.
    """
    cdef:
        PetscBool isplex, isswarm

    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.PLEX.encode(), &isplex))
    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.SWARM.encode(), &isswarm))
    if isswarm:
        CHKERR(PetscFree(closure[0]))
        if nclosure != NULL:
            # though not documented in DMPlexGetTransitiveClosure,
            # *nclosure can be NULL on input
            nclosure[0] = 0
    elif isplex:
        CHKERR(DMPlexRestoreTransitiveClosure(dm, p, useCone, nclosure, closure))
    else:
        raise ValueError("dm must be a DMPlex or DMSwarm")

cdef inline void get_chart(PETSc.PetscDM dm, PetscInt *pStart, PetscInt *pEnd):
    """
    Get the bounds [pStart, pEnd) for all mesh points.

    :arg dm: The input ``PETSc.PetscDM`` - must be DMPlex or DMSwarm.
    :arg pStart: Output parameter for the first point in the mesh.
    :arg pEnd: Output parameter for one beyond the last point at the
        requested depth

    For DMPlex calls
    ``CHKERR(DMPlexGetChart(dm, pStart, pEnd))``.

    For DMSwarm sets ``pStart[0] = 0`` then calls
    ``CHKERR(DMSwarmGetLocalSize(dm, pEnd))`` to get the upper bound.

    """
    cdef:
        PetscBool isplex, isswarm

    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.PLEX.encode(), &isplex))
    CHKERR(PetscObjectTypeCompare(<PETSc.PetscObject>dm, PETSc.DM.Type.SWARM.encode(), &isswarm))
    if isswarm:
        pStart[0] = 0
        CHKERR(DMSwarmGetLocalSize(dm, pEnd))
    elif isplex:
        CHKERR(DMPlexGetChart(dm, pStart, pEnd))
    else:
        raise ValueError("dm must be a DMPlex or DMSwarm")


def count_labelled_points(PETSc.DM dm, name,
                          PetscInt start, PetscInt end):
    """Return the number of points in the chart [start, end)
    that are marked by the given label.

    .. note::

       This destroys any index on the label.

    :arg dm: The DM containing the label
    :arg name: The label name.
    :arg start: The smallest point to consider.
    :arg end: One past the largest point to consider."""
    cdef PetscInt pStart, pEnd, p, n
    cdef PetscBool has_point
    cdef DMLabel label
    get_chart(dm.dm, &pStart, &pEnd)
    CHKERR(DMGetLabel(dm.dm, name.encode(), &label))
    CHKERR(DMLabelCreateIndex(label, pStart, pEnd))
    n = 0
    for p in range(start, end):
        CHKERR(DMLabelHasPoint(label, p, &has_point))
        if has_point:
            n += 1
    CHKERR(DMLabelDestroyIndex(label))
    return n

@cython.boundscheck(False)
@cython.wraparound(False)
def facet_numbering(PETSc.DM plex, kind,
                    np.ndarray[PetscInt, ndim=1, mode="c"] facets,
                    PETSc.Section cell_numbering,
                    np.ndarray[PetscInt, ndim=2, mode="c"] cell_closures):
    """Compute the parent cell(s) and the local facet number within
    each parent cell for each given facet.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg kind: String indicating the facet kind (interior or exterior)
    :arg facets: Array of input facets
    :arg cell_numbering: Section describing the global cell numbering
    :arg cell_closures: 2D array of ordered cell closures
    """
    cdef:
        PetscInt f, fStart, fEnd, fi, cell
        PetscInt nfacets, nclosure, ncells, cells_per_facet
        const PetscInt *cells = NULL
        np.ndarray[PetscInt, ndim=2, mode="c"] facet_cells
        np.ndarray[PetscInt, ndim=2, mode="c"] facet_local_num

    get_height_stratum(plex.dm, 1, &fStart, &fEnd)
    nfacets = facets.shape[0]
    nclosure = cell_closures.shape[1]

    assert kind in ["interior", "exterior"]
    if kind == "interior":
        cells_per_facet = 2
    else:
        cells_per_facet = 1
    facet_local_num = np.empty((nfacets, cells_per_facet), dtype=IntType)
    facet_cells = np.empty((nfacets, cells_per_facet), dtype=IntType)

    # First determine the parent cell(s) for each facet
    for f in range(nfacets):
        CHKERR(DMPlexGetSupport(plex.dm, facets[f], &cells))
        CHKERR(DMPlexGetSupportSize(plex.dm, facets[f], &ncells))
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, cells[0], &cell))
        facet_cells[f,0] = cell
        if cells_per_facet > 1:
            if ncells > 1:
                CHKERR(PetscSectionGetOffset(cell_numbering.sec,
                                             cells[1], &cell))
                facet_cells[f,1] = cell
            else:
                facet_cells[f,1] = -1

    # Run through the sorted closure to get the
    # local facet number within each parent cell
    for f in range(nfacets):
        # First cell
        cell = facet_cells[f,0]
        fi = 0
        for c in range(nclosure):
            if cell_closures[cell, c] == facets[f]:
                facet_local_num[f,0] = fi
            if fStart <= cell_closures[cell, c] < fEnd:
                fi += 1

        # Second cell
        if facet_cells.shape[1] > 1:
            cell = facet_cells[f,1]
            if cell >= 0:
                fi = 0
                for c in range(nclosure):
                    if cell_closures[cell, c] == facets[f]:
                        facet_local_num[f,1] = fi
                    if fStart <= cell_closures[cell, c] < fEnd:
                        fi += 1
            else:
                facet_local_num[f,1] = -1
    return facet_local_num, facet_cells


cdef inline PetscInt _reorder_plex_cone(PETSc.DM dm,
                                        PetscInt p,
                                        PetscInt *plex_cone_old,
                                        PetscInt *plex_cone_new):
    """Reorder DMPlex cones for tensor product cells of intervals.

    :arg dm: The DMPlex object
    :arg p: The plex point
    :arg plex_cone_old: The original DMPlex cone
    :arg plex_cone_new: The reorderd cone (output)

    This function defines fixed rules to reorder DMPlex cones that facilitates
    comparison with the associated FIAT cones.
    The reordered DMPlex cones are later used to construct cell_closure and to
    compute orientations.
    """
    if dm.getCellType(p) == PETSc.DM.PolytopeType.POINT:
        raise RuntimeError(f"POINT has no cone")
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.SEGMENT:
        raise NotImplementedError(f"Not implemented for {dm.getCellType(p)}")
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.TRIANGLE:
        raise NotImplementedError(f"Not implemented for {dm.getCellType(p)}")
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.TETRAHEDRON:
        raise NotImplementedError(f"Not implemented for {dm.getCellType(p)}")
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.QUADRILATERAL:
        # UFCQuadrilateral:       +---3---+
        #                         |       |
        #                         0       1
        #                         |       |
        #                         +---2---+
        #
        # PETSc.DM.PolytopeType.  +---3---+
        # QUADRILATERAL:          |       |
        #                         0       2
        #                         |       |
        #                         +---1---+
        plex_cone_new[0] = plex_cone_old[0]
        plex_cone_new[1] = plex_cone_old[2]
        plex_cone_new[2] = plex_cone_old[1]
        plex_cone_new[3] = plex_cone_old[3]
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.HEXAHEDRON:
        # UFCHexahedron:            +-------+     +-------+
        #                          /.       |    /   5   /|
        #                         + .   3   |   +-------+ |
        #                         |0.       |   |       |1|
        #                         | . . . . +   |   2   | +
        #                         |.   4   /    |       |/
        #                         +-------+     +-------+
        #
        # PETSc.DM.PolytopeType.    +-------+     +-------+
        # HEXAHEDRON:              /.       |    /   1   /|
        #                         + .   3   |   +-------+ |
        #                         |5.       |   |       |4|
        #                         | . . . . +   |   2   | +
        #                         |.   0   /    |       |/
        #                         +-------+     +-------+
        plex_cone_new[0] = plex_cone_old[5]
        plex_cone_new[1] = plex_cone_old[4]
        plex_cone_new[2] = plex_cone_old[2]
        plex_cone_new[3] = plex_cone_old[3]
        plex_cone_new[4] = plex_cone_old[0]
        plex_cone_new[5] = plex_cone_old[1]
    else:
        raise NotImplementedError(f"Not implemented for {dm.getCellType(p)}")


cdef inline PetscInt _reorder_plex_closure(PETSc.DM dm,
                                           PetscInt p,
                                           PetscInt *plex_closure,
                                           PetscInt *fiat_closure):
    """Reorder DMPlex closure for FIAT closure.

    :arg dm: The DMPlex object
    :arg p: The plex point
    :arg plex_closure: The original DMPlex closure
    :arg fiat_closure: The reorderd closure (output)

    This function defines fixed rules to reorder DMPlex closures
    for FIAT closures.
    Constructing cell_closure using _reorder_plex_closure() and
    reordering plex_cone using _reorder_plex_cone(), we make it
    sure that the cell orientation is always 0 in entity_orientations().
    Indeed the same FIAT closure can be obtained merely using
    _reorder_plex_cone() and ensuring that the cell orientation is 0.
    """
    if dm.getCellType(p) == PETSc.DM.PolytopeType.POINT:
        raise RuntimeError(f"POINT has no cone")
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.SEGMENT:
        # UFCInterval:            0---2---1
        #
        # PETSc.DM.PolytopeType.  1---0---2
        # SEGMENT:
        raise NotImplementedError(f"Not implemented for {dm.getCellType(p)}")
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.TRIANGLE:
        # UFCTriangle:            1 
        #                         | \
        #                         5   3
        #                         |  6   \
        #                         0---4---2
        #
        # PETSc.DM.PolytopeType.  4
        # TRIANGLE:               | \
        #                         3   1
        #                         |  0   \
        #                         6---2---5
        raise NotImplementedError(f"Not implemented for {dm.getCellType(p)}")
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.TETRAHEDRON:
        # UFCTetrahedron:         0---9---1---9---0
        #                          \ 12  / \ 13  /
        # cell = 15                 7   5   6   8
        #                            \ / 10  \ /
        #                             3---4---2
        #                              \ 11  /
        #                               7   8
        #                                \ /
        #                                 0
        #         
        # PETSc.DM.PolytopeType. 14--10--13--10---14
        # TETRAHEDRON:             \  3  / \  4  /
        #                           8   7   6   9
        # cell = 0                   \ /  1  \ /
        #                            11---5---12
        #                              \  2  /
        #                               8   9
        #                                \ /
        #                                14
        raise NotImplementedError(f"Not implemented for {dm.getCellType(p)}")
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.QUADRILATERAL:
        # UFCQuadrilateral:       1---7---3
        #                         |       |
        #                         4   8   5
        #                         |       |
        #                         0---6---2
        #
        # PETSc.DM.PolytopeType.  5---4---8
        # QUADRILATERAL:          |       |
        #                         1   0   3
        #                         |       |
        #                         6---2---7
        raise NotImplementedError(f"Not implemented for {dm.getCellType(p)}")
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.HEXAHEDRON:
        # UFCHexahedron:            3--19---7     3--19---7
        #                         13.       |   13  25  15|
        # cell = 26               1 9  23  11   1--17---5 11
        #                         |20       |   |       |21
        #                         8 2...18..6   8  22  10 6
        #                         |12  24  14   |       |14
        #                         0---16--4     0--16---4
        #
        # PETSc.DM.PolytopeType.   26--13--25    26--13---25
        # HEXAHEDRON:             14.       |   14   2  12|
        #                        23 17  4  18  23--11--24 18
        # cell = 0                |6.       |   |       |5|
        #                        16 20..8..21  16   3  15 21
        #                         |7   1   9    |       |9
        #                        19---10--22   19--10--22
        #
        # To check, run the following with "-dm_view ascii::ascii_info_detail":
        #
        # >>> mesh = UnitCubeMesh(1, 1, 1, hexahedral=True)
        # >>> fiat_cell = as_fiat_cell(mesh.ufl_cell())
        # >>> print(fiat_cell.vertices)
        # >>> print(fiat_cell.topology)
        # >>> mesh.topology_dm.viewFromOptions("-dm_view")
        # >>> closure, _ = mesh.topology_dm.getTransitiveClosure(0)
        # >>> print(closure)
        fiat_closure[0] = plex_closure[2 * 19]
        fiat_closure[1] = plex_closure[2 * 23]
        fiat_closure[2] = plex_closure[2 * 20]
        fiat_closure[3] = plex_closure[2 * 26]
        fiat_closure[4] = plex_closure[2 * 22]
        fiat_closure[5] = plex_closure[2 * 24]
        fiat_closure[6] = plex_closure[2 * 21]
        fiat_closure[7] = plex_closure[2 * 25]
        fiat_closure[8] = plex_closure[2 * 16]
        fiat_closure[9] = plex_closure[2 * 17]
        fiat_closure[10] = plex_closure[2 * 15]
        fiat_closure[11] = plex_closure[2 * 18]
        fiat_closure[12] = plex_closure[2 * 7]
        fiat_closure[13] = plex_closure[2 * 14]
        fiat_closure[14] = plex_closure[2 * 9]
        fiat_closure[15] = plex_closure[2 * 12]
        fiat_closure[16] = plex_closure[2 * 10]
        fiat_closure[17] = plex_closure[2 * 11]
        fiat_closure[18] = plex_closure[2 * 8]
        fiat_closure[19] = plex_closure[2 * 13]
        fiat_closure[20] = plex_closure[2 * 6]
        fiat_closure[21] = plex_closure[2 * 5]
        fiat_closure[22] = plex_closure[2 * 3]
        fiat_closure[23] = plex_closure[2 * 4]
        fiat_closure[24] = plex_closure[2 * 1]
        fiat_closure[25] = plex_closure[2 * 2]
        fiat_closure[26] = plex_closure[2 * 0]
    else:
        raise NotImplementedError(f"Not implemented for {dm.getCellType(p)}")


@cython.boundscheck(False)
@cython.wraparound(False)
def create_cell_closure(PETSc.DM dm,
                        PETSc.Section cell_numbering,
                        _closureSize):
    """Create a map from FIAT local entity numbers to DMPlex point numbers for each cell.

    :arg dm: The DM object encapsulating the mesh topology
    :arg cell_numbering: Section describing the global cell numbering
    :arg _closureSize: Number of entities in the cell
    """
    cdef:
        PetscInt c, cStart, cEnd, cell, i
        PetscInt closureSize = _closureSize, closureSize1
        PetscInt *closure = NULL
        PetscInt *fiat_closure = NULL
        np.ndarray[PetscInt, ndim=2, mode="c"] cell_closure

    get_height_stratum(dm.dm, 0, &cStart, &cEnd)
    if cEnd == cStart:
        return np.empty((cEnd - cStart, closureSize), dtype=IntType)
    for c in range(cStart, cEnd):
        get_transitive_closure(dm.dm, c, PETSC_TRUE, &closureSize1, &closure)
        if closureSize1 != closureSize:
            raise RuntimeError(f"point 0 and point {c} have different cell types")
        restore_transitive_closure(dm.dm, c, PETSC_TRUE, &closureSize1, &closure)
    cell_closure = np.empty((cEnd - cStart, closureSize), dtype=IntType)
    CHKERR(PetscMalloc1(closureSize, &fiat_closure))
    for c in range(cStart, cEnd):
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        get_transitive_closure(dm.dm, c, PETSC_TRUE, &closureSize1, &closure)
        _reorder_plex_closure(dm, c, closure, fiat_closure)
        restore_transitive_closure(dm.dm, c, PETSC_TRUE, &closureSize1, &closure)
        for i in range(closureSize):
            cell_closure[cell, i] = fiat_closure[i]
    CHKERR(PetscFree(fiat_closure))
    return cell_closure


@cython.boundscheck(False)
@cython.wraparound(False)
def closure_ordering(PETSc.DM dm,
                     PETSc.Section vertex_numbering,
                     PETSc.Section cell_numbering,
                     np.ndarray[PetscInt, ndim=1, mode="c"] entity_per_cell):
    """Apply Fenics local numbering to a cell closure.

    :arg dm: The DM object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_numbering: Section describing the global cell numbering
    :arg entity_per_cell: List of the number of entity points in each dimension

    Vertices    := Ordered according to global/universal
                   vertex numbering
    Edges/faces := Ordered according to lexicographical
                   ordering of non-incident vertices
    """
    cdef:
        PetscInt c, cStart, cEnd, v, vStart, vEnd
        PetscInt f, fStart, fEnd, e, eStart, eEnd
        PetscInt dim, vi, ci, fi, v_per_cell, f_per_cell, cell
        PetscInt offset, cell_offset, nfaces, nfacets
        PetscInt nclosure, nfacet_closure, nface_vertices
        PetscInt *vertices = NULL
        PetscInt *v_global = NULL
        PetscInt *closure = NULL
        PetscInt *facets = NULL
        PetscInt *faces = NULL
        PetscInt *face_indices = NULL
        const PetscInt *face_vertices = NULL
        PetscInt *facet_vertices = NULL
        np.ndarray[PetscInt, ndim=2, mode="c"] cell_closure

    dim = get_topological_dimension(dm)
    get_height_stratum(dm.dm, 0, &cStart, &cEnd)
    get_height_stratum(dm.dm, 1, &fStart, &fEnd)
    get_depth_stratum(dm.dm, 1, &eStart, &eEnd)
    get_depth_stratum(dm.dm, 0, &vStart, &vEnd)

    v_per_cell = entity_per_cell[0]
    if len(entity_per_cell) > 1:
        f_per_cell = entity_per_cell[1]
    else:
        f_per_cell = 0

    cell_offset = sum(entity_per_cell) - 1

    CHKERR(PetscMalloc1(v_per_cell, &vertices))
    CHKERR(PetscMalloc1(v_per_cell, &v_global))
    CHKERR(PetscMalloc1(v_per_cell, &facets))
    if v_per_cell > 0:
        CHKERR(PetscMalloc1(v_per_cell-1, &facet_vertices))
    if len(entity_per_cell) > 1:
        CHKERR(PetscMalloc1(f_per_cell, &faces))
        CHKERR(PetscMalloc1(f_per_cell, &face_indices))
    cell_closure = np.empty((cEnd - cStart, sum(entity_per_cell)), dtype=IntType)

    for c in range(cStart, cEnd):
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        get_transitive_closure(dm.dm, c, PETSC_TRUE, &nclosure, &closure)

        # Find vertices and translate universal numbers
        vi = 0
        for ci in range(nclosure):
            if vStart <= closure[2*ci] < vEnd:
                vertices[vi] = closure[2*ci]
                CHKERR(PetscSectionGetOffset(vertex_numbering.sec,
                                             closure[2*ci], &v))
                # Correct -ve offsets for non-owned entities
                if v >= 0:
                    v_global[vi] = v
                else:
                    v_global[vi] = -(v+1)
                vi += 1

        # Sort vertices by universal number
        CHKERR(PetscSortIntWithArray(v_per_cell,v_global,vertices))
        for vi in range(v_per_cell):
            if dim == 1:
                # Correct 1D edge numbering
                cell_closure[cell, vi] = vertices[v_per_cell-vi-1]
            else:
                cell_closure[cell, vi] = vertices[vi]
        offset = v_per_cell

        # Find all edges (dim=1) (only relevant for `DMPlex`)
        if dim > 2:
            assert isinstance(dm, PETSc.DMPlex)
            nfaces = 0
            for ci in range(nclosure):
                if eStart <= closure[2*ci] < eEnd:
                    faces[nfaces] = closure[2*ci]

                    CHKERR(DMPlexGetConeSize(dm.dm, closure[2*ci],
                                             &nface_vertices))
                    CHKERR(DMPlexGetCone(dm.dm, closure[2*ci],
                                         &face_vertices))

                    # Edges in 3D are tricky because we need a
                    # lexicographical sort with two keys (the local
                    # numbers of the two non-incident vertices).

                    # Find non-incident vertices
                    fi = 0
                    face_indices[nfaces] = 0
                    for v in range(v_per_cell):
                        incident = 0
                        for vi in range(nface_vertices):
                            if cell_closure[cell,v] == face_vertices[vi]:
                                incident = 1
                                break
                        if incident == 0:
                            face_indices[nfaces] += v * 10**(1-fi)
                            fi += 1
                    nfaces += 1

            # Sort by local numbers of non-incident vertices
            CHKERR(PetscSortIntWithArray(f_per_cell,
                                         face_indices, faces))
            for fi in range(nfaces):
                cell_closure[cell, offset+fi] = faces[fi]
            offset += nfaces

        # Calling get_transitive_closure() again invalidates the
        # current work array, so we need to get the facets and cell
        # out before getting the facet closures.

        # Find all facets (co-dim=1)
        nfacets = 0
        for ci in range(nclosure):
            if fStart <= closure[2*ci] < fEnd:
                facets[nfacets] = closure[2*ci]
                nfacets += 1

        # The cell itself is always the first entry in the Plex closure
        cell_closure[cell, cell_offset] = closure[0]

        # Now we can deal with facets (only relevant for `DMPlex`)
        if dim > 1:
            for f in range(nfacets):
                # Derive facet vertices from facet_closure
                get_transitive_closure(dm.dm, facets[f],
                                       PETSC_TRUE,
                                       &nfacet_closure,
                                       &closure)
                vi = 0
                for fi in range(nfacet_closure):
                    if vStart <= closure[2*fi] < vEnd:
                        facet_vertices[vi] = closure[2*fi]
                        vi += 1

                # Find non-incident vertices
                for v in range(v_per_cell):
                    incident = 0
                    for vi in range(v_per_cell-1):
                        if cell_closure[cell,v] == facet_vertices[vi]:
                            incident = 1
                            break
                    # Only one non-incident vertex per facet, so
                    # local facet no. = non-incident vertex no.
                    if incident == 0:
                        cell_closure[cell,offset+v] = facets[f]
                        break

            offset += nfacets

    if closure != NULL:
        restore_transitive_closure(dm.dm, 0, PETSC_TRUE, &nclosure, &closure)

    CHKERR(PetscFree(vertices))
    CHKERR(PetscFree(v_global))
    CHKERR(PetscFree(facets))
    CHKERR(PetscFree(facet_vertices))
    CHKERR(PetscFree(faces))
    CHKERR(PetscFree(face_indices))

    return cell_closure


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def quadrilateral_closure_ordering(PETSc.DM plex,
                                   PETSc.Section vertex_numbering,
                                   PETSc.Section cell_numbering,
                                   np.ndarray[PetscInt, ndim=1, mode="c"] cell_orientations):
    """Cellwise orders mesh entities according to the given cell orientations.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_numbering: Section describing the cell numbering
    :arg cell_orientations: Specifies the starting vertex for each cell,
                            and the order of traversal (CCW or CW).
    """
    cdef:
        PetscInt c, cStart, cEnd, cell
        PetscInt fStart, fEnd, vStart, vEnd
        PetscInt entity_per_cell, ncells
        PetscInt nclosure, p, vi, v, fi, i
        PetscInt start_v, off
        PetscInt *closure = NULL
        PetscInt c_vertices[4]
        PetscInt c_facets[4]
        PetscInt g_vertices[4]
        PetscInt vertices[4]
        PetscInt facets[4]
        const PetscInt *cell_cone = NULL
        int reverse
        np.ndarray[PetscInt, ndim=2, mode="c"] cell_closure

    get_height_stratum(plex.dm, 0, &cStart, &cEnd)
    get_height_stratum(plex.dm, 1, &fStart, &fEnd)
    get_depth_stratum(plex.dm, 0, &vStart, &vEnd)

    ncells = cEnd - cStart
    entity_per_cell = 4 + 4 + 1

    cell_closure = np.empty((ncells, entity_per_cell), dtype=IntType)
    for c in range(cStart, cEnd):
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        get_transitive_closure(plex.dm, c, PETSC_TRUE, &nclosure, &closure)

        # First extract the facets (edges) and the vertices
        # from the transitive closure into c_facets and c_vertices.
        # Here we assume that DMPlex gives entities in the order:
        #
        #   8--3--7
        #   |     |
        #   4  0  2
        #   |     |
        #   5--1--6
        #
        # where the starting vertex and order of traversal is arbitrary.
        # (We fix that later.)
        #
        # For the vertices, we also retrieve the global numbers into g_vertices.
        vi = 0
        fi = 0
        for p in range(nclosure):
            if vStart <= closure[2*p] < vEnd:
                CHKERR(PetscSectionGetOffset(vertex_numbering.sec, closure[2*p], &v))
                c_vertices[vi] = closure[2*p]
                g_vertices[vi] = cabs(v)
                vi += 1
            elif fStart <= closure[2*p] < fEnd:
                c_facets[fi] = closure[2*p]
                fi += 1

        # The first vertex is given by the entry in cell_orientations.
        start_v = cell_orientations[cell]

        # Based on the cell orientation, we reorder the vertices and facets
        # (edges) from 'c_vertices' and 'c_facets' into 'vertices' and 'facets'.
        off = 0
        while off < 4 and g_vertices[off] != start_v:
            off += 1
        assert off < 4

        # The second vertex is chosen so that the first facet appering in
        # the cone of c (cell_cone[0] in the below) would match up with
        # the first facet or the second facet of the FInAT cell.
        # The plex-intrinsic quadrilateral cell in general can be oriented
        # in 8 different ways relative to the FInAT reference cell, but
        # embedding this constraint in the construction of cell_closure
        # reduces the total number of possible orientations by a factor of
        # 2, letting the remaining 4 orientations naturally represented by
        # the tensor product of the orientations (= (0, 1)) of the two
        # intervals that compose the tensor-product quadrilateral cell of FInAT.
        #
        # Our using cell_one(c)[0] here ensures that cells in the halo are
        # consistently oriented across MPI processes, and allows us to
        # order global DoFs in the plex-intrinsic way that is meaningful
        # during the save/load cycles.
        CHKERR(DMPlexGetCone(plex.dm, c, &cell_cone))
        if cell_cone[0] == c_facets[off] or cell_cone[0] == c_facets[(off + 2) % 4]:
            for i in range(off, 4):
                vertices[i - off] = c_vertices[i]
                facets[i - off] = c_facets[i]
            for i in range(0, off):
                vertices[i + (4-off)] = c_vertices[i]
                facets[i + (4-off)] = c_facets[i]
        else:
            for i in range(off, -1, -1):
                vertices[off - i] = c_vertices[i]
            for i in range(3, off, -1):
                vertices[off+1 + (3-i)] = c_vertices[i]
            for i in range(off-1, -1, -1):
                facets[off-1 - i] = c_facets[i]
            for i in range(3, off-1, -1):
                facets[off + (3-i)] = c_facets[i]

        # At this point the cell "has" the right starting vertex
        # and order of traversal. If the starting vertex is one with an X,
        # and arrows on the edges show the order of traversal:
        #
        #   o--<--o
        #   |     |
        #   v     ^
        #   |     |
        #   o-->--X
        #
        # then outer product elements assume edge directions like this:
        #
        #   o--<--o
        #   |     |
        #   ^     ^
        #   |     |
        #   o--<--X
        #
        # ... and a vertex ordering like this:
        #
        #   3-----1
        #   |     |
        #   |     |
        #   |     |
        #   2-----0
        #
        # ... and a facet (edge) ordering like this:
        #
        #   o--3--o
        #   |     |
        #   1     0
        #   |     |
        #   o--2--o
        #
        # So let us permute.
        cell_closure[cell, 0] = vertices[0]
        cell_closure[cell, 1] = vertices[1]
        cell_closure[cell, 2] = vertices[3]
        cell_closure[cell, 3] = vertices[2]
        cell_closure[cell, 4 + 0] = facets[0]
        cell_closure[cell, 4 + 1] = facets[2]
        cell_closure[cell, 4 + 2] = facets[3]
        cell_closure[cell, 4 + 3] = facets[1]
        cell_closure[cell, 8] = c

    if closure != NULL:
        restore_transitive_closure(plex.dm, 0, PETSC_TRUE, &nclosure, &closure)

    return cell_closure


cdef inline PetscInt _compute_orientation_simplex(PetscInt *fiat_cone,
                                                  PetscInt *plex_cone,
                                                  PetscInt coneSize):
    """Compute orientation of a simplex cell/facet etc.

    Example:
    Compute orientaion of an edge (interval) of a triange

    cell_closure[cell, :] = [32, 31, 33, 41, 44, 47, 53]
    fiat_cone = [32, 33]  # DMPlex cone of the interval of interest
    plex_cone = [33, 32]  # DMPlex cone of the interval of interest
    coneSize = 2

    UFCTriangle

     2
     | \
     4   3
     | 6   \
     0--5---1

    physical triangle mapped onto UFCTriangle

    33
     | \
    44  41
     |53   \
    32--47--31

    Conceptually, the following happens:

    inds = []
    for e in fiat_cone:
        if e in plex_cone:
            inds.append(plex_cone.index(e))
            plex_cone.remove(e)

    -> inds = [1, 0]
    o = 1! * inds[0] + 0! * inds[1] = 1
    """
    cdef:
        PetscInt k, n, e, q, o = 0, coneSize1 = coneSize
        PetscInt *cone1 = NULL
        PetscInt *inds = NULL

    CHKERR(PetscMalloc1(coneSize, &cone1))
    CHKERR(PetscMalloc1(coneSize, &inds))
    for k in range(coneSize1):
        cone1[k] = plex_cone[k]
    n = 0
    for e in range(coneSize):
        q = fiat_cone[e]
        for k in range(coneSize1):
            if q == cone1[k]:
                inds[n] = k
                n += 1
                break
        else:
            raise ValueError(f"Point {q} is in fiat_cone, but not in plex_cone")
        while k < coneSize1 - 1:
            cone1[k] = cone1[k + 1]
            k += 1
        coneSize1 -= 1
    assert n == coneSize
    for k in range(n):
        o += np.math.factorial(n - 1 - k) * inds[k]
    CHKERR(PetscFree(cone1))
    CHKERR(PetscFree(inds))
    return o


cdef inline PetscInt _compute_orientation_interval_tensor_product(PetscInt *fiat_cone,
                                                                  PetscInt *plex_cone,
                                                                  PetscInt *plex_cone_copy,
                                                                  PetscInt dim):
    """Compute orientation of tensor product entity of intervals.

    :arg fiat_cone: The FIAT cone
    :arg plex_cone: The plex cone (reordered)
    :arg plex_cone_copy: A work array
    :arg dim: The dimension
    :returns: The orientation
    """
    cdef:
        PetscInt i, j, k, eo = 0, io = 0, dim1

    # eo (extrinsic orientation): swap axes
    # io (intrinsic orientation): reflect component intervals
    # o (total orientation): (2 ** dim) * eo + io
    #
    # Example:
    #
    # UFCQuadrilateral:
    #
    #  eo\io    0      1      2      3
    #
    #         1---3  0---2  3---1  2---0
    #    0    |   |  |   |  |   |  |   |
    #         0---2  1---3  2---0  3---1
    #
    #         2---3  3---2  0---1  1---0
    #    1    |   |  |   |  |   |  |   |
    #         0---1  1---0  2---3  3---2
    #
    dim1 = dim
    for i in range(2 * dim):
        plex_cone_copy[i] = plex_cone[i]
    for i in range(dim):
        for j in range(dim1):
            if plex_cone_copy[2 * j] == fiat_cone[2 * i] or plex_cone_copy[2 * j + 1] == fiat_cone[2 * i]:
                if plex_cone_copy[2 * j] == fiat_cone[2 * i] and plex_cone_copy[2 * j + 1] == fiat_cone[2 * i + 1]:
                    # io += (2**(dim - 1 - i)) * 0
                    pass
                elif plex_cone_copy[2 * j + 1] == fiat_cone[2 * i] and plex_cone_copy[2 * j] == fiat_cone[2 * i + 1]:
                    io += (2**(dim - 1 - i)) * 1
                else:
                    raise RuntimeError("Found inconsistent fiat_cone and plex_cone")
                eo += np.math.factorial(dim - 1 - i) * j
                for k in range(j, dim1 - 1):
                    plex_cone_copy[2 * k] = plex_cone_copy[2 * k + 2]
                    plex_cone_copy[2 * k + 1] = plex_cone_copy[2 * k + 3]
                dim1 -= 1
                break
        else:
            raise RuntimeError("Found inconsistent fiat_cone and plex_cone")
    assert dim1 == 0
    return (2**dim) * eo + io


cdef inline PetscInt _compute_orientation(PETSc.DM dm,
                                          np.ndarray[PetscInt, ndim=2, mode="c"] cell_closure,
                                          PetscInt cell,
                                          PetscInt e,
                                          PetscInt *fiat_cone,
                                          PetscInt *plex_cone,
                                          PetscInt *plex_cone_copy,
                                          PetscInt *entity_cone_map,
                                          PetscInt *entity_cone_map_offset):
    """Compute orientation of a given entity on a given cell.

    :arg dm: The DMPlex object
    :arg cell_closure: The cell_closure array that defines a map
        from FIAT entities to plex points for each cell
    :arg cell: The cell (Firedrake numbering)
    :arg e: The entity (FIAT numbering)
    :arg fiat_cone: The work array to store a FIAT cone
    :arg plex_cone: The work array to store a plex cone
    :arg plex_cone_copy: A work array
    :arg entity_cone_map: The concatenated array of entity cones in FIAT local entity numbers
    :arg entity_cone_map_offset: The map from FIAT local entity numbers to offsets into entity_cone_map

    Dispatches on PETSc.DM.PolytopeType.
    """
    cdef:
        PetscInt p, coneSize, offset, i, dim, o
        PetscInt *cone = NULL

    p = cell_closure[cell, e]
    if dm.getCellType(p) == PETSc.DM.PolytopeType.POINT:
        return 0
    CHKERR(DMPlexGetConeSize(dm.dm, p, &coneSize))
    CHKERR(DMPlexGetCone(dm.dm, p, &cone))
    if (entity_cone_map_offset[e + 1] - entity_cone_map_offset[e]) != coneSize:
        raise RuntimeError("FIAT entity cone size != plex point cone size")
    offset = entity_cone_map_offset[e]
    for i in range(coneSize):
        fiat_cone[i] = cell_closure[cell, entity_cone_map[offset + i]]
    if dm.getCellType(p) == PETSc.DM.PolytopeType.SEGMENT or \
       dm.getCellType(p) == PETSc.DM.PolytopeType.TRIANGLE or \
       dm.getCellType(p) == PETSc.DM.PolytopeType.TETRAHEDRON:
        # UFCInterval      <- PETSc.DM.PolytopeType.SEGMENT
        # UFCTriangle      <- PETSc.DM.PolytopeType.TRIANGLE
        # UFCTetrahedron   <- PETSc.DM.PolytopeType.TETRAHEDRON
        return _compute_orientation_simplex(fiat_cone, cone, coneSize)
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.QUADRILATERAL:
        # UFCQuadrilateral <- PETSc.DM.PolytopeType.QUADRILATERAL
        dim = 2
        _reorder_plex_cone(dm, p, cone, plex_cone)
        return _compute_orientation_interval_tensor_product(fiat_cone, plex_cone, plex_cone_copy, dim)
    elif dm.getCellType(p) == PETSc.DM.PolytopeType.HEXAHEDRON:
        # UFCHexahedron    <- PETSc.DM.PolytopeType.HEXAHEDRON
        dim = 3
        _reorder_plex_cone(dm, p, cone, plex_cone)
        return _compute_orientation_interval_tensor_product(fiat_cone, plex_cone, plex_cone_copy, dim)
    else:
        raise ValueError(f"Unknown cell type: {dm.getCellType(p)}")


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def entity_orientations(mesh,
                        np.ndarray[PetscInt, ndim=2, mode="c"] cell_closure):
    """Compute entity orientations.

    :arg mesh: The :class:`~.MeshTopology` object encapsulating the mesh topology
    :arg cell_closure: The two-dimensional array, each row of which contains
        the closure of the associated cell
    :returns: A 2D array of the same shape as cell_closure, each row of which
        contains orientations of the entities in the closure of the associated cell

    See :meth:`~.AbstractMeshTopology.entity_orientations` for details on the
    returned array.

    See :func:`~.get_cell_nodes` for the usage of the returned array.
    """
    cdef:
        PETSc.DM dm
        PetscInt numCells, numEntities, cell, e, maxConeSize
        PetscInt *fiat_cone = NULL
        PetscInt *plex_cone = NULL
        PetscInt *plex_cone_copy = NULL
        PetscInt *entity_cone_map = NULL
        PetscInt *entity_cone_map_offset = NULL
        np.ndarray[PetscInt, ndim=2, mode="c"] entity_orientations

    if type(mesh) is not firedrake.mesh.MeshTopology:
        raise TypeError(f"Unexpected mesh type: {type(mesh)}")

    # Make entity-cone map for the FIAT cell.
    def make_entity_cone_lists(fiat_cell):
        _dim = fiat_cell.get_dimension()
        _connectivity = fiat_cell.connectivity
        _list = []
        _offset_list = [0 for _ in _connectivity[(0, 0)]]  # vertices have no cones
        _offset = 0
        _n = 0  # num. of entities up to dimension = _d
        for _d in range(_dim):
            _n1 = len(_offset_list)
            for _conn in _connectivity[(_d + 1, _d)]:
                _list += [_c + _n for _c in _conn]  # These are indices into cell_closure[some_cell]
                _offset_list.append(_offset)
                _offset += len(_conn)
            _n = _n1
        _offset_list.append(_offset)
        return _list, _offset_list

    fiat_cell = as_fiat_cell(mesh.ufl_cell())
    entity_cone_list, entity_cone_list_offset = make_entity_cone_lists(fiat_cell)
    CHKERR(PetscMalloc1(len(entity_cone_list), &entity_cone_map))
    CHKERR(PetscMalloc1(len(entity_cone_list_offset), &entity_cone_map_offset))
    for i in range(len(entity_cone_list)):
        entity_cone_map[i] = entity_cone_list[i]
    for i in range(len(entity_cone_list_offset)):
        entity_cone_map_offset[i] = entity_cone_list_offset[i]
    #
    dm = mesh.topology_dm
    dim = dm.getDimension()
    numCells = cell_closure.shape[0]
    numEntities = cell_closure.shape[1]
    entity_orientations = np.zeros_like(cell_closure)
    # Define work arrays to avoid repeated mallocs/frees.
    CHKERR(DMPlexGetMaxSizes(dm.dm, &maxConeSize, NULL))
    CHKERR(PetscMalloc1(maxConeSize, &fiat_cone))  # work array
    CHKERR(PetscMalloc1(maxConeSize, &plex_cone))  # work array
    CHKERR(PetscMalloc1(maxConeSize, &plex_cone_copy))  # work array
    for cell in range(numCells):
        for e in range(numEntities):
            entity_orientations[cell, e] = _compute_orientation(dm, cell_closure, cell, e,
                                                                fiat_cone,
                                                                plex_cone,
                                                                plex_cone_copy,
                                                                entity_cone_map,
                                                                entity_cone_map_offset)
    CHKERR(PetscFree(fiat_cone))
    CHKERR(PetscFree(plex_cone))
    CHKERR(PetscFree(plex_cone_copy))
    CHKERR(PetscFree(entity_cone_map))
    CHKERR(PetscFree(entity_cone_map_offset))
    return entity_orientations


@cython.boundscheck(False)
@cython.wraparound(False)
def create_section(mesh, nodes_per_entity, on_base=False, block_size=1):
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
    """
    # We don't use DMPlexCreateSection because we only ever put one
    # field in each section.
    cdef:
        PETSc.DM dm
        PETSc.Section section
        PETSc.IS renumbering
        PetscInt i, p, layers, pStart, pEnd
        PetscInt dimension, ndof
        np.ndarray[PetscInt, ndim=2, mode="c"] nodes
        np.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        bint variable, extruded, on_base_

    dm = mesh.topology_dm
    if isinstance(dm, PETSc.DMSwarm) and on_base:
        raise NotImplementedError("Vertex Only Meshes cannot be extruded.")
    variable = mesh.variable_layers
    extruded = mesh.cell_set._extruded
    extruded_periodic = mesh.cell_set._extruded_periodic
    on_base_ = on_base
    nodes_per_entity = np.asarray(nodes_per_entity, dtype=IntType)
    if variable:
        layer_extents = mesh.layer_extents
    elif extruded:
        if on_base:
            nodes_per_entity = sum(nodes_per_entity[:, i] for i in range(2))
        else:
            if extruded_periodic:
                nodes_per_entity = sum(nodes_per_entity[:, i]*(mesh.layers - 1) for i in range(2))
            else:
                nodes_per_entity = sum(nodes_per_entity[:, i]*(mesh.layers - i) for i in range(2))
    section = PETSc.Section().create(comm=mesh._comm)
    get_chart(dm.dm, &pStart, &pEnd)
    section.setChart(pStart, pEnd)
    renumbering = mesh._dm_renumbering
    CHKERR(PetscSectionSetPermutation(section.sec, renumbering.iset))
    dimension = get_topological_dimension(dm)
    nodes = nodes_per_entity.reshape(dimension + 1, -1)
    for i in range(dimension + 1):
        get_depth_stratum(dm.dm, i, &pStart, &pEnd)
        if not variable:
            ndof = nodes[i, 0]
        for p in range(pStart, pEnd):
            if variable:
                if on_base_:
                    ndof = nodes[i, 1]
                else:
                    layers = layer_extents[p, 1] - layer_extents[p, 0]
                    ndof = layers*nodes[i, 0] + (layers - 1)*nodes[i, 1]
            CHKERR(PetscSectionSetDof(section.sec, p, block_size * ndof))
    section.setUp()
    return section


@cython.boundscheck(False)
@cython.wraparound(False)
def _make_entity_permutations_c(entity_dofs,
                                entity_permutations):
    entity_permutations_list = []
    num_orientations_list = []
    for dim in sorted(entity_dofs.keys()):
        for entity_num in xrange(len(entity_dofs[dim])):
            perm = entity_permutations[dim][entity_num]
            # Remember number of possible orientations for this entity
            num_orientations_list.append(len(perm))
            # Assert orientations are [0, 1, 2, ..., n-1]
            assert sorted(perm) == list(range(len(perm)))
            for orient in sorted(perm):
                # Concatenate all permutation arrays
                entity_permutations_list.extend(perm[orient])
    entity_permutations_size = len(entity_permutations_list)
    num_orientations_size = len(num_orientations_list)
    entity_permutations_c = np.empty(entity_permutations_size, dtype=IntType)
    num_orientations_c = np.empty(num_orientations_size, dtype=IntType)
    for i in range(entity_permutations_size):
        entity_permutations_c[i] = entity_permutations_list[i]
    for i in range(num_orientations_size):
        num_orientations_c[i] = num_orientations_list[i]
    return entity_permutations_c, num_orientations_c


@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_nodes(mesh,
                   PETSc.Section global_numbering,
                   entity_dofs,
                   entity_permutations,
                   np.ndarray[PetscInt, ndim=1, mode="c"] offset):
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
        PETSc.DM dm
        PETSc.Section cell_numbering
        PetscInt nclosure, c, cStart, cEnd, cell, entity, i
        PetscInt dofs_per_cell, ndofs, off, j, k, orient
        PetscInt entity_permutations_size, num_orientations_size, perm_offset
        int *ceil_ndofs = NULL
        int *flat_index = NULL
        np.ndarray[PetscInt, ndim=1, mode="c"] entity_permutations_c
        np.ndarray[PetscInt, ndim=1, mode="c"] num_orientations_c
        np.ndarray[PetscInt, ndim=2, mode="c"] cell_nodes
        np.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        np.ndarray[PetscInt, ndim=2, mode="c"] cell_closures
        np.ndarray[PetscInt, ndim=2, mode="c"] entity_orientations
        bint is_swarm, variable, extruded_periodic_1_layer

    dm = mesh.topology_dm
    is_swarm = isinstance(dm, PETSc.DMSwarm)
    variable = mesh.variable_layers
    cell_closures = mesh.cell_closure
    entity_orientations = mesh.entity_orientations
    if not is_swarm and entity_orientations is None:
        raise ValueError("entity_orientations can only be None for swarm meshes")
    if variable:
        layer_extents = mesh.layer_extents
        if offset is None:
            raise ValueError("Offset cannot be None with variable layer extents")
    # Special case: DoFs on the top layer are identified as those on the bottom layer.
    extruded_periodic_1_layer = isinstance(mesh, firedrake.mesh.ExtrudedMeshTopology) and \
                                mesh.extruded_periodic and \
                                mesh.layers == 1 + 1
    nclosure = cell_closures.shape[1]
    # Extract ordering from FInAT element entity DoFs
    ndofs_list = []
    flat_index_list = []
    for dim in sorted(entity_dofs.keys()):
        for entity_num in xrange(len(entity_dofs[dim])):
            dofs = entity_dofs[dim][entity_num]
            ndofs_list.append(len(dofs))
            flat_index_list.extend(dofs)
    # Coerce lists into C arrays
    assert nclosure == len(ndofs_list)
    dofs_per_cell = len(flat_index_list)
    CHKERR(PetscMalloc1(nclosure, &ceil_ndofs))
    CHKERR(PetscMalloc1(dofs_per_cell, &flat_index))
    for i in range(nclosure):
        ceil_ndofs[i] = ndofs_list[i]
    for i in range(dofs_per_cell):
        flat_index[i] = flat_index_list[i]
    # Preprocess entity_permutations
    if entity_permutations is not None:
        entity_permutations_c, num_orientations_c = _make_entity_permutations_c(entity_dofs, entity_permutations)
    # Fill cell nodes
    get_height_stratum(dm.dm, 0, &cStart, &cEnd)
    cell_nodes = np.empty((cEnd - cStart, dofs_per_cell), dtype=IntType)
    cell_numbering = mesh._cell_numbering
    for c in range(cStart, cEnd):
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        k = 0
        perm_offset = 0
        for i in range(nclosure):
            entity = cell_closures[cell, i]
            orient = 0 if is_swarm else entity_orientations[cell, i]
            CHKERR(PetscSectionGetDof(global_numbering.sec, entity, &ndofs))
            if ndofs > 0:
                CHKERR(PetscSectionGetOffset(global_numbering.sec, entity, &off))
                # The cell we're looking at the entity through is
                # higher than the lowest cell the column touches, so
                # we need to offset by the difference from the bottom.
                if variable:
                    off += offset[flat_index[k]]*(layer_extents[c, 0] - layer_extents[entity, 0])
                if entity_permutations is not None:
                    if extruded_periodic_1_layer:
                        for j in range(ceil_ndofs[i]):
                            cell_nodes[cell, flat_index[k]] = off + entity_permutations_c[perm_offset + ceil_ndofs[i] * orient + j] % offset[flat_index[k]]
                            k += 1
                    else:
                        for j in range(ceil_ndofs[i]):
                            cell_nodes[cell, flat_index[k]] = off + entity_permutations_c[perm_offset + ceil_ndofs[i] * orient + j]
                            k += 1
                    perm_offset += ceil_ndofs[i] * num_orientations_c[i]
                else:
                    # FInAT element must eventually add entity_permutations() method
                    if extruded_periodic_1_layer:
                        for j in range(ceil_ndofs[i]):
                            cell_nodes[cell, flat_index[k]] = off + j % offset[flat_index[k]]
                            k += 1
                    else:
                        for j in range(ceil_ndofs[i]):
                            cell_nodes[cell, flat_index[k]] = off + j
                            k += 1
    CHKERR(PetscFree(ceil_ndofs))
    CHKERR(PetscFree(flat_index))
    return cell_nodes


@cython.boundscheck(False)
@cython.wraparound(False)
def get_facet_nodes(mesh, np.ndarray[PetscInt, ndim=2, mode="c"] cell_nodes, label,
                    np.ndarray[PetscInt, ndim=1, mode="c"] offset):
    """Build to DoF mapping from facets.

    :arg mesh: The mesh.
    :arg cell_nodes: numpy array mapping from cells to function space nodes.
    :arg label: which set of facets to ask for (interior_facets or exterior_facets).
    :arg offset: optional offset (extruded only).
    :returns: numpy array mapping from facets to nodes in the closure
        of the support of that facet.
    """
    cdef:
        PETSc.DM dm
        PETSc.Section cell_numbering
        DMLabel clabel = NULL
        np.ndarray[PetscInt, ndim=2, mode="c"] facet_nodes
        np.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        PetscInt f, p, i, j, pStart, pEnd, fStart, fEnd, point
        PetscInt supportSize, facet, cell, ndof, dof
        const PetscInt *renumbering
        const PetscInt *support
        PetscBool flg
        bint variable, add_offset

    if label not in {"interior_facets", "exterior_facets"}:
        raise ValueError("Unsupported facet label '%s'", label)

    dm = mesh.topology_dm
    variable = mesh.variable_layers

    if variable and offset is None:
        raise ValueError("Offset cannot be None with variable layer extents")

    get_height_stratum(dm.dm, 1, &fStart, &fEnd)

    ndof = cell_nodes.shape[1]

    get_chart(dm.dm, &pStart, &pEnd)
    nfacet = count_labelled_points(dm, label, fStart, fEnd)
    shape = {"interior_facets": (nfacet, ndof*2),
             "exterior_facets": (nfacet, ndof)}[label]

    facet_nodes = np.full(shape, -1, dtype=IntType)

    CHKERR(DMGetLabel(dm.dm, label.encode(), &clabel))
    CHKERR(DMLabelCreateIndex(clabel, pStart, pEnd))

    CHKERR(ISGetIndices((<PETSc.IS?>mesh._dm_renumbering).iset, &renumbering))
    cell_numbering = mesh._cell_numbering

    facet = 0

    if variable:
        layer_extents = mesh.layer_extents
    for p in range(pStart, pEnd):
        point = renumbering[p]
        if fStart <= point < fEnd:
            CHKERR(DMLabelHasPoint(clabel, point, &flg))
            if not flg:
                # Not a facet we want.
                continue

            CHKERR(DMPlexGetSupportSize(dm.dm, point, &supportSize))
            CHKERR(DMPlexGetSupport(dm.dm, point, &support))
            for i in range(supportSize):
                CHKERR(PetscSectionGetOffset(cell_numbering.sec, support[i], &cell))
                for j in range(ndof):
                    dof = cell_nodes[cell, j]
                    if variable:
                        # This facet iterates from higher than the
                        # cell numbering of the cell, so we need
                        # to add on the difference.
                        dof += offset[j]*(layer_extents[point, 2] - layer_extents[support[i], 0])
                    facet_nodes[facet, ndof*i + j] = dof
            facet += 1

    CHKERR(DMLabelDestroyIndex(clabel))
    CHKERR(ISRestoreIndices((<PETSc.IS?>mesh._dm_renumbering).iset, &renumbering))
    return facet_nodes


@cython.boundscheck(False)
@cython.wraparound(False)
def facet_closure_nodes(V, sub_domain):
    """Extract nodes in the closure of facets with a given marker.

    This works fine for interior as well as exterior facets.

    :arg V: the function space
    :arg sub_domain: a tuple of mesh markers selecting the facets, or
        the magic string "on_boundary" indicating the entire boundary.
    :returns: a numpy array of unique nodes in the closure of facets
        with the given marker.
    """
    cdef:
        PETSc.Section sec = V.dm.getSection()
        PETSc.DM dm = V.mesh().topology_dm
        PetscInt nnodes, p, i, dof, offset, n, j, d
        np.ndarray[PetscInt, ndim=1, mode="c"] points
        np.ndarray[PetscInt, ndim=1, mode="c"] nodes
    if sub_domain == "on_boundary":
        label = "exterior_facets"
        sub_domain = (1, )
    else:
        label = FACE_SETS_LABEL
        if V.mesh().variable_layers:
            # We can't use the generic code in this case because we
            # need to manually take closure of facets on each external
            # face (rather than using the label completion and section
            # information).
            # The reason for this is that (for example) a stack of
            # vertices may extend higher than the faces
            #
            #            Y---.
            #            |   |
            #    .---x---x---.
            #    |   | O |
            #    .---x---x
            #    |   |
            #    .---Y
            #
            # BCs on the facet marked with 'O' should produce 4 values
            # for a P1 field (marked X). But if we just include
            # everything from the sections we'd get 6: additionally we
            # see the points marked Y.
            from firedrake.cython import extrusion_numbering as extnum
            return extnum.facet_closure_nodes(V, sub_domain)

    if not dm.hasLabel(label) or all(dm.getStratumSize(label, marker)
                                     for marker in sub_domain) == 0:
        return np.empty(0, dtype=IntType)

    nnodes = 0
    for marker in sub_domain:
        n = dm.getStratumSize(label, marker)
        if n == 0:
            continue
        points = dm.getStratumIS(label, marker).indices
        for i in range(n):
            p = points[i]
            CHKERR(PetscSectionGetDof(sec.sec, p, &dof))
            nnodes += dof

    nodes = np.empty(nnodes, dtype=IntType)
    j = 0
    for marker in sub_domain:
        n = dm.getStratumSize(label, marker)
        if n == 0:
            continue
        points = dm.getStratumIS(label, marker).indices
        for i in range(n):
            p = points[i]
            CHKERR(PetscSectionGetDof(sec.sec, p, &dof))
            CHKERR(PetscSectionGetOffset(sec.sec, p, &offset))
            for d in range(dof):
                nodes[j] = offset + d
                j += 1
    assert j == nnodes
    return np.unique(nodes)


@cython.boundscheck(False)
@cython.wraparound(False)
def label_facets(PETSc.DM plex):
    """Add labels to facets in the the plex

    Facets on the boundary are marked with "exterior_facets" while all
    others are marked with "interior_facets".

    """
    cdef:
        PetscInt fStart, fEnd, facet, pStart, pEnd
        char *ext_label = <char *>"exterior_facets"
        char *int_label = <char *>"interior_facets"
        DMLabel lbl_ext, lbl_int
        PetscBool has_point

    if get_topological_dimension(plex) == 0:
        return
    plex.removeLabel(ext_label)
    plex.removeLabel(int_label)
    plex.createLabel(ext_label)
    plex.createLabel(int_label)
    get_height_stratum(plex.dm, 1, &fStart, &fEnd)
    get_chart(plex.dm, &pStart, &pEnd)
    CHKERR(DMGetLabel(plex.dm, ext_label, &lbl_ext))
    # Mark boundaries as exterior_facets.
    plex.markBoundaryFaces(ext_label)
    CHKERR(DMGetLabel(plex.dm, int_label, &lbl_int))
    CHKERR(DMLabelCreateIndex(lbl_ext, pStart, pEnd))
    for facet in range(fStart, fEnd):
        CHKERR(DMLabelHasPoint(lbl_ext, facet, &has_point))
        # Not marked, must be interior
        if not has_point:
            CHKERR(DMLabelSetValue(lbl_int, facet, 1))
    CHKERR(DMLabelDestroyIndex(lbl_ext))


def complete_facet_labels(PETSc.DM dm):
    """Transfer label values from the facet labels to everything in
    the closure of the facets."""
    cdef PETSc.DMLabel label

    if get_topological_dimension(dm) == 0:
        return
    for name in [FACE_SETS_LABEL, "exterior_facets", "interior_facets"]:
        if dm.hasLabel(name):
            label = dm.getLabel(name)
            CHKERR( DMPlexLabelComplete(dm.dm, label.dmlabel) )


@cython.boundscheck(False)
@cython.wraparound(False)
def cell_facet_labeling(PETSc.DM plex,
                        PETSc.Section cell_numbering,
                        np.ndarray[PetscInt, ndim=2, mode="c"] cell_closures):
    """Computes a labeling for the facet numbers on a particular cell
    (interior and exterior facet labels with subdomain markers). The
    i-th local facet is represented as:

    cell_facets[c, i]

    If `cell_facets[c, i, 0]` is ``0``, then the local facet
    ``i`` is an exterior facet, otherwise if the result is ``1``
    it is interior. `cell_facets[c, i, 1]` returns the subdomain marker
    for the local facet.

    :arg plex: The DMPlex object representing the mesh topology.
    :arg cell_numbering: PETSc.Section describing the global cell numbering
    :arg cell_closures: 2D array of ordered cell closures.
    """
    cdef:
        PetscInt c, cstart, cend, fi, cell, nfacet, p, nclosure
        PetscInt f, fstart, fend, point, marker, pstart, pend
        PetscBool is_exterior
        const PetscInt *facets
        DMLabel exterior = NULL, subdomain = NULL
        np.ndarray[np.int8_t, ndim=3, mode="c"] cell_facets

    from firedrake.slate.slac.compiler import cell_to_facets_dtype
    nclosure = cell_closures.shape[1]
    get_height_stratum(plex.dm, 0, &cstart, &cend)
    # FIXME: wrong for mixed element meshes
    nfacet = plex.getConeSize(cstart)
    get_height_stratum(plex.dm, 1, &fstart, &fend)
    cell_facets = np.full((cend - cstart, nfacet, 2), -1, dtype=cell_to_facets_dtype)

    get_chart(plex.dm, &pstart, &pend)
    CHKERR(DMGetLabel(plex.dm, "exterior_facets".encode(), &exterior))
    CHKERR(DMGetLabel(plex.dm, FACE_SETS_LABEL.encode(), &subdomain))
    CHKERR(DMLabelCreateIndex(exterior, pstart, pend))

    for c in range(cstart, cend):
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        fi = 0
        for p in range(nclosure):
            point = cell_closures[cell, p]
            # This is a facet point
            if fstart <= point < fend:
                # Get exterior label
                DMLabelHasPoint(exterior, point, &is_exterior)
                cell_facets[cell, fi, 0] = <np.int8_t>(not is_exterior)
                if subdomain != NULL:
                    # Get subdomain marker
                    CHKERR(DMLabelGetValue(subdomain, point, &marker))
                    cell_facets[cell, fi, 1] = <np.int8_t>marker
                else:
                    cell_facets[cell, fi, 1] = -1

                fi += 1

    CHKERR(DMLabelDestroyIndex(exterior))
    return cell_facets


def _get_firedrake_plex_permutation_dg_transitive_closure(PETSc.DM dm):
    # PETSc DG1 whose DoFs are ordered according to the order
    # of vertices appering in the transitive closure of the cell.
    # This is the default PETSc DG coordinate representation,
    # which works with the default PETSc CG coordinate FE
    # in refinement.
    # See _reorder_plex_closure for the transitive closure orderings.
    cStart, cEnd = dm.getHeightStratum(0)
    if cEnd == cStart:
        dm_cell_type = -1
    else:
        dm_cell_type = dm.getCellType(0)  # assume single cell type
    dm_cell_type = dm.comm.tompi4py().allreduce(dm_cell_type, op=MPI.MAX)
    if dm_cell_type == PETSc.DM.PolytopeType.POINT:
        ndofs = np.array([1], dtype=IntType)
        perm = np.array([0], dtype=IntType)
    elif dm_cell_type == PETSc.DM.PolytopeType.SEGMENT:
        ndofs = np.array([2, 0], dtype=IntType)
        perm = np.array([0, 1], dtype=IntType)
    elif dm_cell_type == PETSc.DM.PolytopeType.TRIANGLE:
        ndofs = np.array([3, 0, 0], dtype=IntType)
        perm = np.array([2, 0, 1], dtype=IntType)
    elif dm_cell_type == PETSc.DM.PolytopeType.TETRAHEDRON:
        ndofs = np.array([4, 0, 0, 0], dtype=IntType)
        perm = np.array([3, 2, 1, 0], dtype=IntType)
    elif dm_cell_type == PETSc.DM.PolytopeType.QUADRILATERAL:
        ndofs = np.array([4, 0, 0], dtype=IntType)
        perm = np.array([1, 0, 2, 3], dtype=IntType)
    elif dm_cell_type == PETSc.DM.PolytopeType.HEXAHEDRON:
        ndofs = np.array([8, 0, 0, 0], dtype=IntType)
        perm = np.array([0, 4, 1, 7, 3, 5, 2, 6], dtype=IntType)
    else:
        raise NotImplementedError(f"Not implemented for dm_cell_type ({dm_cell_type})")
    perm_offsets = np.add.accumulate(np.concatenate((np.array([0], dtype=IntType), ndofs)), dtype=IntType)
    return ndofs, perm, perm_offsets


@cython.boundscheck(False)
@cython.wraparound(False)
def transform_vec_from_firedrake_to_petsc(PETSc.DM dm,
                                          finat_element,
                                          PETSc.Section firedrake_sec,
                                          PETSc.Vec firedrake_vec,
                                          PETSc.Section petsc_sec,
                                          PETSc.Vec petsc_vec,
                                          use_transitive_closure_dg=False):
    """Transform Firedrake vec to PETSc vec.

    Parameters
    ----------
    dm: PETSc.DM
        `PETSc.DM`.
    finat_element: finat.finiteelementbase.FiniteElementBase
        FInAT element.
    firedrake_sec: PETSc.Section
        `PETSc.Section` representing the Firedrake DoF layout.
    firedrake_vec: PETSc.Vec
        `PETSc.Vec` representing the Firedrake DoF vector.
    petsc_sec: PETSc.Section
        `PETSc.Section` representing the PETSc DoF layout.
    petsc_vec: PETSc.Vec
        `PETSc.Vec` representing the PETSc DoF vector.
    use_transitive_closure_dg: bool
        Whether to use PETSc DG element whose DoFs are ordered according to the transitive closure of the cell.

    """
    cdef:
        const PetscScalar *firedrake_array
        PetscScalar *petsc_array
        PetscInt n, bs, petsc_n, petsc_bs, pStart, pEnd, firedrake_pStart, firedrake_pEnd, petsc_pStart, petsc_pEnd, p, firedrake_dof, petsc_dof, total_dof = 0, firedrake_offset, petsc_offset, i, j, height
        np.ndarray[PetscInt, ndim=1, mode="c"] ndofs, perm, perm_offsets

    n, _ = firedrake_vec.getSizes()
    petsc_n, _ = petsc_vec.getSizes()
    if petsc_n != n:
        raise RuntimeError(f"petsc_vec local size ({petsc_n}) != firedrake_vec local size ({n})")
    bs = firedrake_vec.getBlockSize()
    petsc_bs = petsc_vec.getBlockSize()
    if petsc_bs != bs:
        raise RuntimeError(f"petsc_vec block size ({petsc_bs}) != firedrake_vec block size ({bs})")
    firedrake_pStart, firedrake_pEnd = firedrake_sec.getChart()
    petsc_pStart, petsc_pEnd = petsc_sec.getChart()
    pStart = firedrake_pStart if firedrake_pStart > petsc_pStart else petsc_pStart
    pEnd = firedrake_pEnd if firedrake_pEnd < petsc_pEnd else petsc_pEnd
    if use_transitive_closure_dg:
        ndofs, perm, perm_offsets = _get_firedrake_plex_permutation_dg_transitive_closure(dm)
    else:
        raise NotImplementedError("Currently not implemented for {finat_element}")
    CHKERR(VecGetArrayRead(firedrake_vec.vec, &firedrake_array))
    CHKERR(VecGetArray(petsc_vec.vec, &petsc_array))
    for p in range(pStart, pEnd):
        CHKERR(PetscSectionGetDof(firedrake_sec.sec, p, &firedrake_dof))  # scalar offset
        CHKERR(PetscSectionGetDof(petsc_sec.sec, p, &petsc_dof))
        if petsc_dof != bs * firedrake_dof:
            raise RuntimeError(f"petsc_dof ({petsc_dof}) != bs * firedrake_dof ({bs} * {firedrake_dof})")
        CHKERR(DMPlexGetPointHeight(dm.dm, p, &height))
        CHKERR(PetscSectionGetOffset(firedrake_sec.sec, p, &firedrake_offset))  # scalar offset
        CHKERR(PetscSectionGetOffset(petsc_sec.sec, p, &petsc_offset))
        for i in range(ndofs[height]):
            for j in range(bs):
                petsc_array[petsc_offset + bs * perm[perm_offsets[height] + i] + j] = firedrake_array[bs * firedrake_offset + bs * i + j]
        total_dof += petsc_dof
    CHKERR(VecRestoreArray(petsc_vec.vec, &petsc_array))
    CHKERR(VecRestoreArrayRead(firedrake_vec.vec, &firedrake_array))
    if total_dof != n:
        raise RuntimeError(f"total number of local dofs ({total_dof}) != local vec size ({n})")


@cython.boundscheck(False)
@cython.wraparound(False)
def _set_dg_coordinates(PETSc.DM dm,
                        finat_element,
                        PETSc.Section firedrake_dg_coord_sec,
                        PETSc.Vec firedrake_dg_coord_vec):
    """Set DG coordinates in plex for a periodic mesh.

    Parameters
    ----------
    dm: PETSc.DM
        `PETSc.DM` representing the periodic mesh topology.
    finat_element: finat.finiteelementbase.FiniteElementBase
        Scalar DG finat element.
    firedrake_dg_coord_sec: Function
        `PETSc.Section` containing the Firedrake scalar DG DoF layout.
    firedrake_dg_coord_vec: Function
        `PETSc.Vec` containing the Firedrake DG coordinates.

    """
    cdef:
        PETSc.DM coord_dm, dg_coord_dm
        PETSc.Section dg_coord_sec
        PETSc.Vec dg_coord_vec
        PetscScalar *dg_coords
        const PetscScalar *firedrake_dg_coords
        PetscInt n, gdim, cStart, cEnd, c, offset, firedrake_offset, i, j, coord_size, ndof

    gdim = firedrake_dg_coord_vec.getBlockSize()
    coord_dm = dm.getCoordinateDM()
    dg_coord_dm = coord_dm.clone()
    dm.setCellCoordinateDM(dg_coord_dm)
    dg_coord_sec = PETSc.Section().create(comm=dm.comm)
    dg_coord_sec.setNumFields(1)
    dg_coord_sec.setFieldComponents(0, gdim)
    cStart, cEnd = dm.getHeightStratum(0)
    dg_coord_sec.setChart(cStart, cEnd)
    if finat_element.entity_closure_dofs() != finat_element.entity_dofs():
        raise RuntimeError(f"Expecting discontinuous element: got {finat_element}")
    ndof = finat_element.space_dimension()
    for c in range(cStart, cEnd):
        CHKERR(PetscSectionSetDof(dg_coord_sec.sec, c, ndof * gdim))
        CHKERR(PetscSectionSetFieldDof(dg_coord_sec.sec, c, 0, ndof * gdim))
    dg_coord_sec.setUp()
    dm.setCellCoordinateSection(gdim, dg_coord_sec)  # gdim ignored
    dm.setCoordinateDim(gdim)
    coord_size = dg_coord_sec.getStorageSize()
    dg_coord_vec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
    dg_coord_vec.setName("coordinates_dg")
    dg_coord_vec.setSizes((coord_size, PETSc.DETERMINE), gdim)
    dg_coord_vec.setType(dm.getCoordinatesLocal().getType())
    transform_vec_from_firedrake_to_petsc(dm,
                                          finat_element,
                                          firedrake_dg_coord_sec,
                                          firedrake_dg_coord_vec,
                                          dg_coord_sec,
                                          dg_coord_vec,
                                          True)
    dm.setCellCoordinatesLocal(dg_coord_vec)


@cython.boundscheck(False)
@cython.wraparound(False)
def reordered_coords(PETSc.DM dm, PETSc.Section global_numbering, shape, reference_coord=False):
    """Return coordinates for the dm, reordered according to the
    global numbering permutation for the coordinate function space.

    Shape is a tuple of (mesh.num_vertices(), geometric_dim)."""
    cdef:
        PETSc.Section dm_sec, coord_sec
        PetscInt v, vStart, vEnd, offset, dm_offset, c, cStart, cEnd
        PetscInt i, j, dim = shape[1]
        np.ndarray[PetscScalar, ndim=2, mode="c"] dm_coords, coords
        np.ndarray[PetscInt, ndim=1, mode="c"] ndofs, perm, perm_offsets

    get_depth_stratum(dm.dm, 0, &vStart, &vEnd)
    if isinstance(dm, PETSc.DMPlex):
        if not dm.getCoordinatesLocalized():
            # Use CG coordiantes.
            dm_sec = dm.getCoordinateSection()
            dm_coords = dm.getCoordinatesLocal().array.reshape(shape)
            coords = np.empty_like(dm_coords)
            for v in range(vStart, vEnd):
                CHKERR(PetscSectionGetOffset(global_numbering.sec, v, &offset))
                CHKERR(PetscSectionGetOffset(dm_sec.sec, v, &dm_offset))
                dm_offset = dm_offset//dim
                for i in range(dim):
                    coords[offset, i] = dm_coords[dm_offset, i]
        else:
            # Use DG coordiantes.
            get_height_stratum(dm.dm, 0, &cStart, &cEnd)
            dim = dm.getCoordinateDim()
            ndofs, perm, perm_offsets = _get_firedrake_plex_permutation_dg_transitive_closure(dm)
            dm_sec = dm.getCellCoordinateSection()
            dm_coords = dm.getCellCoordinatesLocal().array.reshape(((cEnd - cStart) * ndofs[0], dim))
            coords = np.empty_like(dm_coords)
            for c in range(cStart, cEnd):
                CHKERR(PetscSectionGetOffset(global_numbering.sec, c, &offset))  # scalar offset
                CHKERR(PetscSectionGetOffset(dm_sec.sec, c, &dm_offset))
                dm_offset = dm_offset // dim
                for j in range(ndofs[0]):
                    for i in range(dim):
                        coords[offset + j, i] = dm_coords[dm_offset + perm[perm_offsets[0] + j], i]
    elif isinstance(dm, PETSc.DMSwarm):
        # NOTE DMSwarm coords field isn't copied so make sure
        # dm.restoreField is called too!
        # NOTE DMSwarm coords field DMSwarmPIC_coor always stored as real
        if reference_coord:
            swarm_field_name = "refcoord"
        else:
            swarm_field_name = "DMSwarmPIC_coor"
        dm_coords = dm.getField(swarm_field_name).reshape(shape).astype(ScalarType)
        coords = np.empty_like(dm_coords)
        for v in range(vStart, vEnd):
            CHKERR(PetscSectionGetOffset(global_numbering.sec, v, &offset))
            for i in range(dim):
                coords[offset, i] = dm_coords[v - vStart, i]
        dm.restoreField(swarm_field_name)
    else:
        raise ValueError("Only DMPlex and DMSwarm are supported.")
    return coords

@cython.boundscheck(False)
@cython.wraparound(False)
def mark_entity_classes(PETSc.DM dm):
    """Mark all points in a given DM according to the PyOP2 entity
    classes:

    core   : owned and not in send halo
    owned  : owned and in send halo
    ghost  : in halo

    by inspecting the `pointSF` graph.

    :arg dm: The DM object encapsulating the mesh topology
    """
    cdef:
        PetscInt pStart, pEnd, cStart, cEnd
        PetscInt c, ci, p
        PetscInt nleaves
        PetscInt *closure = NULL
        PetscInt nclosure
        const PetscInt *ilocal = NULL
        PetscBool non_exec
        const PetscSFNode *iremote = NULL
        PETSc.SF point_sf = None
        PetscBool is_ghost, is_owned
        DMLabel lbl_core, lbl_owned, lbl_ghost

    get_chart(dm.dm, &pStart, &pEnd)
    get_height_stratum(dm.dm, 0, &cStart, &cEnd)

    dm.createLabel("pyop2_core")
    dm.createLabel("pyop2_owned")
    dm.createLabel("pyop2_ghost")

    CHKERR(DMGetLabel(dm.dm, b"pyop2_core", &lbl_core))
    CHKERR(DMGetLabel(dm.dm, b"pyop2_owned", &lbl_owned))
    CHKERR(DMGetLabel(dm.dm, b"pyop2_ghost", &lbl_ghost))

    if dm.comm.size > 1:
        # Mark ghosts from point overlap SF
        point_sf = dm.getPointSF()
        CHKERR(PetscSFGetGraph(point_sf.sf, NULL, &nleaves, &ilocal, NULL))
        for p in range(nleaves):
            # If ilocal is NULL but we have leaves then ilocal is contiguous
            # (0, 1, 2...)
            if ilocal:
                CHKERR(DMLabelSetValue(lbl_ghost, ilocal[p], 1))
            else:
                CHKERR(DMLabelSetValue(lbl_ghost, p, 1))
    else:
        # If sequential mark all points as core
        for p in range(pStart, pEnd):
            CHKERR(DMLabelSetValue(lbl_core, p, 1))
        return

    CHKERR(DMLabelCreateIndex(lbl_ghost, pStart, pEnd))
    # If any entity in closure(cell) is in the halo, then all those
    # entities in closure(cell) that are not in the halo are owned,
    # but not core.
    for c in range(cStart, cEnd):
        get_transitive_closure(dm.dm, c, PETSC_TRUE, &nclosure, &closure)
        is_owned = PETSC_FALSE
        for ci in range(nclosure):
            p = closure[2*ci]
            CHKERR(DMLabelHasPoint(lbl_ghost, p, &is_ghost))
            if is_ghost:
                is_owned = PETSC_TRUE
                break
        if is_owned:
            for ci in range(nclosure):
                p = closure[2*ci]
                CHKERR(DMLabelHasPoint(lbl_ghost, p, &is_ghost))
                if not is_ghost:
                    CHKERR(DMLabelSetValue(lbl_owned, p, 1))
    if closure != NULL:
        restore_transitive_closure(dm.dm, 0, PETSC_TRUE, &nclosure, &closure)
    # Mark all remaining points as core
    CHKERR(DMLabelCreateIndex(lbl_owned, pStart, pEnd))
    for p in range(pStart, pEnd):
        CHKERR(DMLabelHasPoint(lbl_owned, p, &is_owned))
        CHKERR(DMLabelHasPoint(lbl_ghost, p, &is_ghost))
        if not is_ghost and not is_owned:
            CHKERR(DMLabelSetValue(lbl_core, p, 1))
    CHKERR(DMLabelDestroyIndex(lbl_owned))
    CHKERR(DMLabelDestroyIndex(lbl_ghost))


@cython.boundscheck(False)
@cython.wraparound(False)
def mark_entity_classes_using_cell_dm(PETSc.DM swarm):
    """
    Mark all points in a given Particle in Cell (PIC) DMSwarm according to the
    PyOP2 entity classes (core, owned, ghost) using the markers of the parent
    DMPlex or DMSwarm cells (i.e. the cells within which the swarm points are
    located).
    """
    cdef:
        PETSc.DM plex
        PetscInt cStart, cEnd, c
        PetscInt *plex_cell_classes = NULL, plex_cell_class
        DMLabel swarm_labels[3], plex_label
        PetscInt label_value = 1, op2class_size, i, ilabel
        PETSc.PetscIS op2class_is = NULL
        const PetscInt *class_indices = NULL
        PetscInt nswarmCells, swarmCell, blocksize
        PetscInt *swarmParentCells = NULL
        PetscDataType ctype = PETSC_DATATYPE_UNKNOWN

    plex = swarm.getCellDM()
    get_height_stratum(plex.dm, 0, &cStart, &cEnd)
    CHKERR(PetscMalloc1(cEnd - cStart, &plex_cell_classes))
    for c in range(cStart, cEnd):
        plex_cell_classes[c - cStart] = -1
    for ilabel, op2class in enumerate([b"pyop2_core", b"pyop2_owned", b"pyop2_ghost"]):
        CHKERR(DMGetLabel(plex.dm, op2class, &plex_label))
        # Get number of plex points labeled as this op2class.
        CHKERR(DMLabelGetStratumSize(plex_label, label_value, &op2class_size))
        if op2class_size > 0:
            # Get an IS containing the plex points labeled as this op2class.
            CHKERR(DMLabelGetStratumIS(plex_label, label_value, &op2class_is))
            CHKERR(ISGetIndices(op2class_is, &class_indices))
            for i in range(op2class_size):
                if cStart <= class_indices[i] < cEnd:  # plex cell points are in [cStart, cEnd)
                    plex_cell_classes[class_indices[i] - cStart] = ilabel
            CHKERR(ISRestoreIndices(op2class_is, &class_indices))
            CHKERR(ISDestroy(&op2class_is))
    for c in range(cStart, cEnd):
        if plex_cell_classes[c - cStart] < 0:
            raise RuntimeError("Cell point %d in the parent plex does not belong to any pyop2 class" % c)
    for ilabel, op2class in enumerate([b"pyop2_core", b"pyop2_owned", b"pyop2_ghost"]):
        CHKERR(DMCreateLabel(swarm.dm, op2class))
        CHKERR(DMGetLabel(swarm.dm, op2class, &swarm_labels[ilabel]))
    CHKERR(DMSwarmGetField(swarm.dm, b"DMSwarm_cellid", &blocksize, &ctype, <void**>&swarmParentCells))
    assert ctype == PETSC_INT
    assert blocksize == 1
    CHKERR(DMSwarmGetLocalSize(swarm.dm, &nswarmCells))
    for swarmCell in range(nswarmCells):
        plex_cell_class = plex_cell_classes[swarmParentCells[swarmCell] - cStart]
        CHKERR(DMLabelSetValue(swarm_labels[plex_cell_class], swarmCell, label_value))
    CHKERR(DMSwarmRestoreField(swarm.dm, b"DMSwarm_cellid", &blocksize, &ctype, <void**>&swarmParentCells))
    CHKERR(PetscFree(plex_cell_classes))


@cython.boundscheck(False)
@cython.wraparound(False)
def get_entity_classes(PETSc.DM dm):
    """Builds PyOP2 entity class offsets for all entity levels.

    :arg dm: The DM object encapsulating the mesh topology
    """
    cdef:
        np.ndarray[PetscInt, ndim=2, mode="c"] entity_class_sizes
        np.ndarray[PetscInt, mode="c"] eStart, eEnd
        PetscInt depth, d, i, ci, class_size, start, end
        const PetscInt *indices = NULL
        PETSc.IS class_is

    depth = get_topological_dimension(dm) + 1

    entity_class_sizes = np.zeros((depth, 3), dtype=IntType)
    eStart = np.zeros(depth, dtype=IntType)
    eEnd = np.zeros(depth, dtype=IntType)
    for d in range(depth):
        get_depth_stratum(dm.dm, d, &start, &end)
        eStart[d] = start
        eEnd[d] = end

    for i, op2class in enumerate([b"pyop2_core",
                                  b"pyop2_owned",
                                  b"pyop2_ghost"]):
        class_is = dm.getStratumIS(op2class, 1)
        class_size = dm.getStratumSize(op2class, 1)
        if class_size > 0:
            CHKERR(ISGetIndices(class_is.iset, &indices))
            for ci in range(class_size):
                for d in range(depth):
                    if eStart[d] <= indices[ci] < eEnd[d]:
                        entity_class_sizes[d, i] += 1
                        break
            CHKERR(ISRestoreIndices(class_is.iset, &indices))

    # PyOP2 entity class indices are additive
    for d in range(depth):
        for i in range(1, 3):
            entity_class_sizes[d, i] += entity_class_sizes[d, i-1]
    return entity_class_sizes


@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_markers(PETSc.DM dm, PETSc.Section cell_numbering,
                     subdomain_id):
    """Get the cells marked by a given subdomain_id.

    :arg dm: The DM for the mesh topology
    :arg cell_numbering: Section mapping dm cell points to firedrake cell indices.
    :arg subdomain_id: The subdomain_id to look for.

    :raises ValueError: if the subdomain_id is not valid.
    :returns: A numpy array (possibly empty) of the cell ids.
    """
    cdef:
        PetscInt i, j, n, offset, c, cStart, cEnd, ncells
        np.ndarray[PetscInt, ndim=1, mode="c"] cells
        np.ndarray[PetscInt, ndim=1, mode="c"] indices

    if not dm.hasLabel(CELL_SETS_LABEL):
        return np.empty(0, dtype=IntType)
    vals = dm.getLabelIdIS(CELL_SETS_LABEL).indices
    comm = dm.comm.tompi4py()

    def merge_ids(x, y, datatype):
        return x.union(y)

    op = MPI.Op.Create(merge_ids, commute=True)

    all_ids = np.asarray(sorted(comm.allreduce(set(vals), op=op)),
                         dtype=IntType)
    op.Free()
    if subdomain_id not in all_ids:
        raise ValueError("Invalid subdomain_id %d not in %s" % (subdomain_id, vals))

    if subdomain_id not in vals:
        return np.empty(0, dtype=IntType)

    n = dm.getStratumSize(CELL_SETS_LABEL, subdomain_id)
    if n == 0:
        return np.empty(0, dtype=IntType)
    indices = dm.getStratumIS(CELL_SETS_LABEL, subdomain_id).indices
    ncells = 0
    get_height_stratum(dm.dm, 0, &cStart, &cEnd)
    for i in range(n):
        c = indices[i]
        if cStart <= c < cEnd:
            ncells += 1
    if ncells == 0:
        return np.empty(0, dtype=IntType)
    cells = np.empty(ncells, dtype=IntType)
    j = 0
    for i in range(n):
        c = indices[i]
        if cStart <= c < cEnd:
            CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &offset))
            cells[j] = offset
            j += 1
    return cells


@cython.boundscheck(False)
@cython.wraparound(False)
def get_facet_ordering(PETSc.DM plex, PETSc.Section facet_numbering):
    """Builds a list of all facets ordered according to the given numbering.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg facet_numbering: A Section describing the global facet numbering
    """
    cdef:
        PetscInt fi, fStart, fEnd, offset
        np.ndarray[PetscInt, ndim=1, mode="c"] facets

    size = facet_numbering.getStorageSize()
    facets = np.empty(size, dtype=IntType)
    get_height_stratum(plex.dm, 1, &fStart, &fEnd)
    for fi in range(fStart, fEnd):
        CHKERR(PetscSectionGetOffset(facet_numbering.sec, fi, &offset))
        facets[offset] = fi
    return facets


@cython.boundscheck(False)
@cython.wraparound(False)
def get_facets_by_class(PETSc.DM plex, label,
                        np.ndarray[PetscInt, ndim=1, mode="c"] ordering):
    """Builds a list of all facets ordered according to PyOP2 entity
    classes and computes the respective class offsets.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg ordering: An array giving the global traversal order of facets
    :arg label: Label string that marks the facets to order
    """
    cdef:
        PetscInt dim, fi, ci, nfacets, nclass, lbl_val, o, f, fStart, fEnd
        PetscInt pStart, pEnd, i, n
        PetscInt *indices = NULL
        PETSc.IS class_is = None
        PetscBool has_point, is_class
        DMLabel lbl_facets, lbl_class
        np.ndarray[PetscInt, ndim=1, mode="c"] facets

    dim = get_topological_dimension(plex)
    get_height_stratum(plex.dm, 1, &fStart, &fEnd)
    get_chart(plex.dm, &pStart, &pEnd)
    nfacets = count_labelled_points(plex, label, fStart, fEnd)
    facets = np.empty(nfacets, dtype=IntType)
    facet_classes = [0, 0, 0]
    fi = 0

    CHKERR(DMGetLabel(plex.dm, label.encode(), &lbl_facets))
    CHKERR(DMLabelCreateIndex(lbl_facets, pStart, pEnd))
    for i, op2class in enumerate([b"pyop2_core",
                                  b"pyop2_owned",
                                  b"pyop2_ghost"]):
        CHKERR(DMGetLabel(plex.dm, op2class, &lbl_class))
        CHKERR(DMLabelCreateIndex(lbl_class, pStart, pEnd))
        nclass = plex.getStratumSize(op2class, 1)
        if nclass > 0:
            for o in range(ordering.shape[0]):
                f = ordering[o]
                CHKERR(DMLabelHasPoint(lbl_facets, f, &has_point))
                CHKERR(DMLabelHasPoint(lbl_class, f, &is_class))
                if has_point and is_class:
                    facets[fi] = f
                    fi += 1
        facet_classes[i] = fi
        CHKERR(DMLabelDestroyIndex(lbl_class))
    CHKERR(DMLabelDestroyIndex(lbl_facets))
    return facets, facet_classes


@cython.boundscheck(False)
@cython.wraparound(False)
def validate_mesh(PETSc.DM dm):
    """Perform some validation of the input mesh.

    :arg dm: The DM object encapsulating the mesh topology."""
    cdef:
        PetscInt  pStart, pEnd, cStart, cEnd, p, c, ci
        PetscInt  nclosure, nseen
        PetscInt *closure = NULL
        PetscBT   seen = NULL
        PetscBool flag

    from mpi4py import MPI

    get_chart(dm.dm, &pStart, &pEnd)
    get_height_stratum(dm.dm, 0, &cStart, &cEnd)

    CHKERR(PetscBTCreate(pEnd - pStart, &seen))
    nseen = 0
    # Walk the cells, counting the number of points we can traverse in
    # the closure.
    for c in range(cStart, cEnd):
        get_transitive_closure(dm.dm, c, PETSC_TRUE, &nclosure, &closure)
        for ci in range(nclosure):
            p = closure[2*ci]
            if not PetscBTLookup(seen, p):
                nseen += 1
                PetscBTSet(seen, p)

    if closure != NULL:
        restore_transitive_closure(dm.dm, 0, PETSC_TRUE, &nclosure, &closure)
    CHKERR(PetscBTDestroy(&seen))

    # Check validity on all processes
    valid = dm.comm.tompi4py().allreduce(nseen == pEnd - pStart, op=MPI.LAND)
    if not valid:
        raise ValueError("Provided mesh has some entities not reachable by traversing cells (maybe rogue vertices?)")


@cython.boundscheck(False)
@cython.wraparound(False)
def plex_renumbering(PETSc.DM plex,
                     np.ndarray entity_classes,
                     np.ndarray[PetscInt, ndim=1, mode="c"] reordering=None):
    """
    Build a global node renumbering as a permutation of Plex points.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg entity_classes: Array of entity class offsets for
         each dimension.
    :arg reordering: A reordering from reordered to original plex
         points used to provide the traversal order of the cells
         (i.e. the inverse of the ordering obtained from
         DMPlexGetOrdering).  Optional, if not provided (or ``None``),
         no reordering is applied and the plex is traversed in
         original order.

    The node permutation is derived from a depth-first traversal of
    the Plex graph over each entity class in turn. The returned IS
    is the Plex -> PyOP2 permutation.
    """
    cdef:
        PetscInt dim, cStart, cEnd, nfacets, nclosure, c, ci, l, p, f
        PetscInt pStart, pEnd, cell
        np.ndarray[PetscInt, ndim=1, mode="c"] lidx, ncells
        PetscInt *facets = NULL
        PetscInt *closure = NULL
        PetscInt *perm = NULL
        PETSc.IS facet_is = None
        PETSc.IS perm_is = None
        PetscBT seen = NULL
        PetscBool has_point
        DMLabel labels[3]
        bint reorder = reordering is not None

    dim = get_topological_dimension(plex)
    get_chart(plex.dm, &pStart, &pEnd)
    get_height_stratum(plex.dm, 0, &cStart, &cEnd)
    CHKERR(PetscMalloc1(pEnd - pStart, &perm))
    CHKERR(PetscBTCreate(pEnd - pStart, &seen))
    ncells = np.zeros(3, dtype=IntType)

    # Get label pointers and label-specific array indices
    CHKERR(DMGetLabel(plex.dm, b"pyop2_core", &labels[0]))
    CHKERR(DMGetLabel(plex.dm, b"pyop2_owned", &labels[1]))
    CHKERR(DMGetLabel(plex.dm, b"pyop2_ghost", &labels[2]))
    for l in range(3):
        CHKERR(DMLabelCreateIndex(labels[l], pStart, pEnd))
    entity_classes = entity_classes.astype(IntType)
    lidx = np.zeros(3, dtype=IntType)
    lidx[1] = sum(entity_classes[:, 0])
    lidx[2] = sum(entity_classes[:, 1])

    for c in range(pStart, pEnd):
        if reorder:
            cell = reordering[c]
        else:
            cell = c

        # We always re-order cell-wise so that we inherit any cache
        # coherency from the reordering provided by the Plex
        if cStart <= cell < cEnd:

            # Get  cell closure
            get_transitive_closure(plex.dm, cell, PETSC_TRUE, &nclosure, &closure)
            for ci in range(nclosure):
                p = closure[2*ci]
                if not PetscBTLookup(seen, p):
                    for l in range(3):
                        CHKERR(DMLabelHasPoint(labels[l], p, &has_point))
                        if has_point:
                            PetscBTSet(seen, p)
                            perm[lidx[l]] = p
                            lidx[l] += 1
                            break

    if closure != NULL:
        restore_transitive_closure(plex.dm, 0, PETSC_TRUE, &nclosure, &closure)
    for c in range(3):
        CHKERR(DMLabelDestroyIndex(labels[c]))

    CHKERR(PetscBTDestroy(&seen))
    perm_is = PETSc.IS().create(comm=plex.comm)
    perm_is.setType("general")
    CHKERR(ISGeneralSetIndices(perm_is.iset, pEnd - pStart,
                               perm, PETSC_OWN_POINTER))
    return perm_is

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_remote_ranks(PETSc.DM plex):
    """Returns an array assigning the rank of the owner to each
    locally visible cell. Locally owned cells have -1 assigned to them.

    :arg plex: The DMPlex object encapsulating the mesh topology
    """
    cdef:
        PetscInt cStart, cEnd, ncells, i
        PETSc.SF sf
        PetscInt nroots, nleaves
        const PetscInt *ilocal = NULL
        const PetscSFNode *iremote = NULL
        np.ndarray[PetscInt, ndim=1, mode="c"] result

    get_height_stratum(plex.dm, 0, &cStart, &cEnd)
    ncells = cEnd - cStart

    result = np.full(ncells, -1, dtype=IntType)
    if plex.comm.size > 1:
        sf = plex.getPointSF()
        CHKERR(PetscSFGetGraph(sf.sf, &nroots, &nleaves, &ilocal, &iremote))

        for i in range(nleaves):
            if cStart <= ilocal[i] < cEnd:
                result[ilocal[i] - cStart] = iremote[i].rank

    return result

cdef inline PetscInt cneg(PetscInt i):
    """complementary inverse"""
    return -(i + 1)

cdef inline PetscInt cabs(PetscInt i):
    """complementary absolute value"""
    if i >= 0:
        return i
    else:
        return cneg(i)

cdef inline void get_edge_global_vertices(PETSc.DM plex,
                                          PETSc.Section vertex_numbering,
                                          PetscInt facet,
                                          PetscInt *global_v):
    """Returns the global numbers of the vertices of an edge.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg facet: The edge
    :arg global_v: Return buffer, must have capacity for 2 values
    """
    cdef:
        PetscInt nvertices, ndofs
        const PetscInt *vs = NULL

    CHKERR(DMPlexGetConeSize(plex.dm, facet, &nvertices))
    assert nvertices == 2

    CHKERR(DMPlexGetCone(plex.dm, facet, &vs))

    CHKERR(PetscSectionGetDof(vertex_numbering.sec, vs[0], &ndofs))
    assert cabs(ndofs) == 1
    CHKERR(PetscSectionGetDof(vertex_numbering.sec, vs[1], &ndofs))
    assert cabs(ndofs) == 1

    CHKERR(PetscSectionGetOffset(vertex_numbering.sec, vs[0], &global_v[0]))
    CHKERR(PetscSectionGetOffset(vertex_numbering.sec, vs[1], &global_v[1]))

    global_v[0] = cabs(global_v[0])
    global_v[1] = cabs(global_v[1])

cdef inline np.int8_t get_global_edge_orientation(PETSc.DM plex,
                                                  PETSc.Section vertex_numbering,
                                                  PetscInt facet):
    """Returns the local plex direction (ordering in plex cone) relative to
    the global edge direction (from smaller to greater global vertex number).

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg facet: The edge
    """
    cdef PetscInt v[2]
    get_edge_global_vertices(plex, vertex_numbering, facet, v)
    return v[0] > v[1]

cdef struct CommFacet:
    PetscInt remote_rank
    PetscInt global_u, global_v
    PetscInt local_facet

cdef int CommFacet_cmp(void *x_, void *y_) nogil:
    """Three-way comparison C function for CommFacet structs."""
    cdef:
        CommFacet *x = <CommFacet *>x_
        CommFacet *y = <CommFacet *>y_

    if x.remote_rank < y.remote_rank:
        return -1
    elif x.remote_rank > y.remote_rank:
        return 1

    if x.global_u < y.global_u:
        return -1
    elif x.global_u > y.global_u:
        return 1

    if x.global_v < y.global_v:
        return -1
    elif x.global_v > y.global_v:
        return 1

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void get_communication_lists(
    PETSc.DM plex, PETSc.Section vertex_numbering,
    np.ndarray[PetscInt, ndim=1, mode="c"] cell_ranks,
    # Output parameters:
    PetscInt *nranks, PetscInt **ranks, PetscInt **offsets,
    PetscInt **facets, PetscInt **facet2index):

    """Creates communication lists for shared facet information exchange.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.

    :arg nranks: Number of neighbouring MPI nodes (return value)
    :arg ranks: MPI ranks of neigbours (return value)
    :arg offsets: Offset for each neighbour in the data buffer (return value)
    :arg facets: Array of local plex facet numbers of shared facets
                 (return value)
    :arg facet2index: Maps local facet numbers to indices in the communication
                      buffer, inverse of 'facets' (return value)
    """
    cdef:
        int comm_size = plex.comm.size
        PetscInt cStart, cEnd
        PetscInt nfacets, fStart, fEnd, f
        PetscInt i, k, support_size
        const PetscInt *support = NULL
        PetscInt local_count, remote
        PetscInt v[2]
        PetscInt *facet_ranks = NULL
        PetscInt *nfacets_per_rank = NULL

        CommFacet *cfacets = NULL

    get_height_stratum(plex.dm, 0, &cStart, &cEnd)
    get_height_stratum(plex.dm, 1, &fStart, &fEnd)
    nfacets = fEnd - fStart

    CHKERR(PetscMalloc1(nfacets, &facet_ranks))
    memset(facet_ranks, -1, nfacets * sizeof(PetscInt))

    # Determines which facets are shared, and which MPI process
    # they are shared with.
    for f in range(fStart, fEnd):
        CHKERR(DMPlexGetSupportSize(plex.dm, f, &support_size))
        CHKERR(DMPlexGetSupport(plex.dm, f, &support))

        local_count = 0
        remote = -1
        for i in range(support_size):
            if cell_ranks[support[i] - cStart] >= 0:
                remote = cell_ranks[support[i] - cStart]
            else:
                local_count += 1

        if local_count == 1:
            facet_ranks[f - fStart] = remote

    # Counts how many facets are shared with each MPI node
    CHKERR(PetscMalloc1(comm_size, &nfacets_per_rank))
    memset(nfacets_per_rank, 0, comm_size * sizeof(PetscInt))

    for i in range(nfacets):
        if facet_ranks[i] != -1:
            nfacets_per_rank[facet_ranks[i]] += 1

    # Counts how many MPI nodes shall this node communicate with
    nranks[0] = 0
    for i in range(comm_size):
        if nfacets_per_rank[i] != 0:
            nranks[0] += 1

    # Creates list of neighbours, and their offsets
    # in the communication buffer.
    #
    # Information about facets shared with rank 'i'
    # should be between offsets[i] (inclusive) and
    # offset[i+1] (exclusive) in the buffer.
    CHKERR(PetscMalloc1(nranks[0], ranks))
    CHKERR(PetscMalloc1(nranks[0]+1, offsets))

    offsets[0][0] = 0
    k = 0
    for i in range(comm_size):
        if nfacets_per_rank[i] != 0:
            ranks[0][k] = i
            offsets[0][k+1] = offsets[0][k] + nfacets_per_rank[i]
            k += 1

    CHKERR(PetscFree(nfacets_per_rank))

    # Sort the facets based on
    # 1. Remote rank - so they occupy the right section of the buffer.
    # 2. Global vertex numbers - so the same order is used on both sides.
    CHKERR(PetscMalloc1(offsets[0][nranks[0]], &cfacets))

    k = 0
    for f in range(fStart, fEnd):
        if facet_ranks[f - fStart] != -1:
            cfacets[k].remote_rank = facet_ranks[f - fStart]
            get_edge_global_vertices(plex, vertex_numbering, f, v)
            if v[0] < v[1]:
                cfacets[k].global_u = v[0]
                cfacets[k].global_v = v[1]
            else:
                cfacets[k].global_u = v[1]
                cfacets[k].global_v = v[0]
            cfacets[k].local_facet = f
            k += 1
    CHKERR(PetscFree(facet_ranks))
    qsort(cfacets, offsets[0][nranks[0]], sizeof(CommFacet), &CommFacet_cmp)

    # For debugging purposes:
    #
    # for i in range(offsets[0][nranks[0]]):
    #     print "(%d/%d): %d = (%d, %d) -> %d" % (_MPI.comm.rank,
    #                                             _MPI.comm.size,
    #                                             cfacets[i].local_facet,
    #                                             cfacets[i].global_u,
    #                                             cfacets[i].global_v,
    #                                             cfacets[i].remote_rank)

    CHKERR(PetscMalloc1(offsets[0][nranks[0]], facets))
    CHKERR(PetscMalloc1(nfacets, facet2index))
    memset(facet2index[0], -1, nfacets * sizeof(PetscInt))

    for i in range(offsets[0][nranks[0]]):
        facets[0][i] = cfacets[i].local_facet
        facet2index[0][facets[0][i] - fStart] = i
    CHKERR(PetscFree(cfacets))

    # For debugging purposes:
    #
    # for i in range(nfacets):
    #     if facet2index[0][i] != -1:
    #         print "(%d/%d): [%d] = %d" % (_MPI.comm.rank,
    #                                       _MPI.comm.size,
    #                                       facet2index[0][i],
    #                                       fStart + i)

@cython.profile(False)
cdef inline void plex_get_restricted_support(PETSc.DM plex,
                                             PetscInt *cell_ranks,
                                             PetscInt f,
                                             # Output parameters:
                                             PetscInt *size,
                                             PetscInt *outbuf):
    """Returns the owned cells incident to a given facet.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.
    :arg f: Facet, whose owned support is the result
    :arg size: Length of the result
    :arg outbuf: Preallocated output buffer
    """
    cdef:
        PetscInt cStart, cEnd, c
        PetscInt support_size
        const PetscInt *support = NULL
        PetscInt i, k

    get_height_stratum(plex.dm, 0, &cStart, &cEnd)

    CHKERR(DMPlexGetSupportSize(plex.dm, f, &support_size))
    CHKERR(DMPlexGetSupport(plex.dm, f, &support))

    k = 0
    for i in range(support_size):
        if cell_ranks[support[i] - cStart] < 0:
            outbuf[k] = support[i]
            k += 1
    size[0] = k

@cython.cdivision(True)
cdef inline PetscInt traverse_cell_string(PETSc.DM plex,
                                          PetscInt first_facet,
                                          PetscInt cell,
                                          PetscInt *cell_ranks,
                                          np.int8_t *orientations):
    """Takes a start facet, and a direction (which of the, possibly two, cells
    it is adjacent to) and propagates that facet's orientation as far as
    possible by orienting the "opposite" facet in the cell then moving to the
    next cell.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg first_facet: Facet to start traversal with
    :arg cell: One of the cells incident to 'first_facet', determines
               the direction of traversal.
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.
    :arg orientations: Facet orientations relative to the plex.
                       -1: orientation not set
                        0: orientation is same as in (local) plex
                       +1: orientation is the opposite of that in plex
                       THIS ARRAY IS UPDATED BY THIS FUNCTION.

    Returns the plex number of the facet visited last, or -1 if
    'first_facet' is part of a closed loop.

    Facets not incident to an owned cell are ignored.

    Facet orientations are aligned to match the orientation, which was
    assigned to orientations[first_facet - fStart].
    """
    cdef:
        PetscInt fStart, fEnd
        PetscInt from_facet = first_facet
        PetscInt to_facet
        PetscInt c = cell
        np.int8_t plex_orientation

        PetscInt local_from, local_to

        PetscInt cone_size, support_size
        const PetscInt *cone = NULL
        const PetscInt *cone_orient = NULL
        PetscInt support[2]
        PetscInt i, ncells_adj

    get_height_stratum(plex.dm, 1, &fStart, &fEnd)

    # Retrieve orientation of first facet
    plex_orientation = orientations[first_facet - fStart]

    while True:
        CHKERR(DMPlexGetConeSize(plex.dm, c, &cone_size))
        assert cone_size == 4

        CHKERR(DMPlexGetCone(plex.dm, c, &cone))
        local_from = 0
        while cone[local_from] != from_facet and local_from < cone_size:
            local_from += 1
        assert local_from < cone_size

        local_to = (local_from + 2) % 4
        to_facet = cone[local_to]

        CHKERR(DMPlexGetConeOrientation(plex.dm, c, &cone_orient))
        plex_orientation ^= (cone_orient[local_from] < 0) ^ True ^ (cone_orient[local_to] < 0)

        # Store orientation of next facet
        orientations[to_facet - fStart] = plex_orientation

        if to_facet == first_facet:
            # Closed loop
            return -1

        plex_get_restricted_support(plex, cell_ranks, to_facet, &support_size, support)

        ncells_adj = 0
        for i in range(support_size):
            if support[i] != c:
                ncells_adj += 1

        if ncells_adj == 0:
            # Reached boundary of local domain
            return to_facet
        elif ncells_adj == 1:
            # Continue with next cell
            for i in range(support_size):
                if support[i] != c:
                    from_facet = to_facet
                    c = support[i]
                    break
        else:
            assert ncells_adj > 1
            raise RuntimeError("Facet belongs to more than two quadrilaterals!")

    # We must not reach this point here.
    raise RuntimeError("This should never happen!")

@cython.boundscheck(False)
@cython.wraparound(False)
cdef locally_orient_quadrilateral_plex(PETSc.DM plex,
                                       PETSc.Section vertex_numbering,
                                       PetscInt *cell_ranks,
                                       PetscInt *facet2index,
                                       PetscInt nfacets_shared,
                                       np.int8_t *orientations):
    """Locally orient the facets (edges) of a quadrilateral plex, and
    derive the dependency information of shared facets.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.
    :arg facet2index: Maps plex facet numbers to their index in the buffer
                      of shared facets.
    :arg nfacets_shared: Number of facets shared with other MPI nodes.
    :arg orientations: Facet orientations relative to the plex.
                       -1: orientation not set
                        0: orientation is same as in (local) plex
                       +1: orientation is the opposite of that in plex
                       THIS ARRAY IS UPDATED BY THIS FUNCTION.

    Returns an array of size 'nfacets_shared', which tells for each shared
    facet which other shared facet needs update, if any, when a shared facet
    is flipped.
     * Equal to 'nfacets_shared': no other facet requires update.
     * Non-negative value: index of shared facet,
                           which must have the same global orientation.
     * Negative value 'i': cneg(i) is the index of the shared facet,
                           which must have opposite global orientation.
    """
    cdef:
        PetscInt nfacets, fStart, fEnd, f
        PetscInt size
        PetscInt support[2]
        PetscInt start_facet, end_facet
        np.int8_t twist
        PetscInt i, j
        np.ndarray[PetscInt, ndim=1, mode="c"] result

    get_height_stratum(plex.dm, 1, &fStart, &fEnd)
    nfacets = fEnd - fStart

    result = np.empty(nfacets_shared, dtype=IntType)

    # Here we walk over all the known facets, if it is not oriented already.
    for f in range(fStart, fEnd):
        if orientations[f - fStart] < 0:
            orientations[f - fStart] = 0

            plex_get_restricted_support(plex, cell_ranks, f, &size, support)
            assert 0 <= size <= 2
            if size == 0:
                # Facet is interior to some other MPI process, ignored
                continue

            # Propagate the orientation of this facet as far as possible
            end_facet = traverse_cell_string(plex, f, support[0],
                                             cell_ranks, orientations)
            if end_facet == -1:
                # Closed loop
                if orientations[f - fStart]:
                    # Moebius strip found
                    #
                    # So we came round a loop and found that the last cell we
                    # hit claims that the end_facet must be flipped then there
                    # must be a twist in the loop, because the first facet
                    # (which is the same) should have had no flip.
                    raise RuntimeError("Moebius strip found in the mesh.")
            else:
                if size == 1:
                    # 'f' is at local domain boundary
                    start_facet = f
                else:
                    # Here we potentially walk off in the other direction
                    start_facet = traverse_cell_string(plex, f, support[1],
                                                       cell_ranks, orientations)

                i = facet2index[start_facet - fStart]
                j = facet2index[end_facet - fStart]
                if i >= 0 or j >= 0:
                    # Either the start or the end facet is shared
                    # with remote processes
                    twist = 0
                    twist ^= get_global_edge_orientation(plex,
                                                         vertex_numbering,
                                                         start_facet)
                    twist ^= orientations[start_facet - fStart]
                    twist ^= orientations[end_facet - fStart]
                    twist ^= get_global_edge_orientation(plex,
                                                         vertex_numbering,
                                                         end_facet)

                    # If the other end of the string is local (not shared), then
                    # no propagation to remote ranks at the other end is needed.
                    if i == -1:
                        result[j] = nfacets_shared
                    elif j == -1:
                        result[i] = nfacets_shared
                    # If other end of the string is shared, then propagation
                    # must take place, the sign tells you whether an orientation
                    # flip is required and the value tells you which facet.
                    elif twist == 0:
                        result[i] = j
                        result[j] = i
                    else:
                        result[i] = cneg(j)
                        result[j] = cneg(i)

    # At the end of this function we have provided a consistent orientation
    # to the local plex, in O(nfacets) time, and we are returning, for all
    # shared facets, information about whether they will require a round of
    # communications when we try and provide a globally consistent orientation.
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void exchange_edge_orientation_data(
    PetscInt nranks, PetscInt *ranks, PetscInt *offsets,
    np.ndarray[PetscInt, ndim=1, mode="c"] ours,
    np.ndarray[PetscInt, ndim=1, mode="c"] theirs,
    MPI.Comm comm):

    """Exchange edge orientation data between neighbouring MPI nodes.

    :arg nranks: Number of neighbouring MPI nodes
    :arg ranks: MPI ranks of neigbours
    :arg offsets: Offset for each neighbour in the data buffer
    :arg ours: Local data, to be sent to neigbours
    :arg theirs: Remote data, to be received from neighbours (return value)
    :arg comm: MPI Communicator.
    """
    cdef PetscInt ri

    # Initiate receiving
    recv_reqs = []
    for ri in range(nranks):
        recv_reqs.append(comm.Irecv(theirs[offsets[ri] : offsets[ri+1]], ranks[ri]))

    # Initiate sending
    send_reqs = []
    for ri in range(nranks):
        send_reqs.append(comm.Isend(ours[offsets[ri] : offsets[ri+1]], ranks[ri]))

    # Wait for completion
    for req in recv_reqs:
        req.Wait()
    for req in send_reqs:
        req.Wait()

@cython.boundscheck(False)
@cython.wraparound(False)
def quadrilateral_facet_orientations(
    PETSc.DM plex, PETSc.Section vertex_numbering,
    np.ndarray[PetscInt, ndim=1, mode="c"] cell_ranks):

    """Returns globally synchronised facet orientations (edge directions)
    incident to locally owned quadrilateral cells.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_ranks: MPI rank of the owner of each (visible) non-owned cell,
                     or -1 for (locally) owned cell.
    """
    cdef:
        PetscInt nranks
        PetscInt *ranks = NULL
        PetscInt *offsets = NULL
        PetscInt *facets = NULL
        PetscInt *facet2index = NULL

        MPI.Comm comm = plex.comm.tompi4py()
        PetscInt nfacets, nfacets_shared, fStart, fEnd

        np.ndarray[PetscInt, ndim=1, mode="c"] affects
        np.ndarray[PetscInt, ndim=1, mode="c"] ours, theirs
        PetscInt conflict, value, f, i, j

        PetscInt ci, size
        PetscInt cells[2]

        np.ndarray[np.int8_t, ndim=1, mode="c"] result

    # Get communication lists
    get_communication_lists(plex, vertex_numbering, cell_ranks,
                            &nranks, &ranks, &offsets, &facets, &facet2index)
    nfacets_shared = offsets[nranks]

    # Discover edge direction dependencies in the mesh locally
    get_height_stratum(plex.dm, 1, &fStart, &fEnd)
    nfacets = fEnd - fStart

    result = np.full(nfacets, -1, dtype=np.int8)
    affects = locally_orient_quadrilateral_plex(plex,
                                                vertex_numbering,
                                                <PetscInt *>cell_ranks.data,
                                                facet2index,
                                                nfacets_shared,
                                                <np.int8_t *>result.data)
    CHKERR(PetscFree(facet2index))

    # Initialise shared edge directions and assign weights
    #
    # "cabs" of the values in 'ours' and 'theirs' is voting strength, and
    # the sign tells the edge direction. Positive sign implies that the edge
    # points from the vertex with the smaller global number to the vertex with
    # the greater global number, negative implies otherwise.
    ours = comm.size * np.arange(nfacets_shared, dtype=IntType) + comm.rank

    # We update these values based on the local connections
    # before we do any communication.
    for i in range(nfacets_shared):
        if affects[i] != nfacets_shared:
            j = cabs(affects[i])
            if cabs(ours[i]) < cabs(ours[j]):
                if affects[i] >= 0:
                    ours[i] = ours[j]
                else:
                    ours[i] = cneg(ours[j])

    # 'ours' is full of the local view of what the orientations are,
    # and 'theirs' will be filled by the remote orientation view.
    theirs = np.empty_like(ours)

    # Synchronise shared edge directions in parallel
    conflict = int(comm.size > 1)
    while conflict != 0:
        # Populate 'theirs' by communication from the 'ours' of others.
        exchange_edge_orientation_data(nranks, ranks, offsets, ours, theirs, comm)

        conflict = 0
        for i in range(nfacets_shared):
            if ours[i] != theirs[i] and cabs(ours[i]) == cabs(theirs[i]):
                # Moebius strip found
                raise RuntimeError("Moebius strip found in the mesh.")

            # If the remote value is stronger, ...
            if cabs(ours[i]) < cabs(theirs[i]):
                # ... we adopt it, ...
                ours[i] = theirs[i]

                # ... and propagate, if the other end is shared as well.
                if affects[i] != nfacets_shared:
                    j = cabs(affects[i])  # connected facet at the other end

                    # If the ribbon is twisted locally,
                    # we propagate the orientation accordingly.
                    if affects[i] >= 0:
                        value = ours[i]
                    else:
                        value = cneg(ours[i])

                    # If the other end does not have the same orientation as the
                    # orientation which propagates there, then the twist might
                    # need to travel further in that direction, therefore we
                    # require another round of orientation exchange.
                    if (ours[j] >= 0) ^ (value >= 0):
                        conflict = 1

                    # Please note that at this point cabs(value) is
                    # always greater than cabs(ours[j]).
                    ours[j] = value

        # If there was a conflict anywhere, do another round
        # of communication everywhere.
        conflict = comm.allreduce(conflict)

    CHKERR(PetscFree(ranks))
    CHKERR(PetscFree(offsets))

    # Reorient the strings of all the shared facets, so that
    # they will match the globally agreed orientations.
    for i in range(nfacets_shared):
        result[facets[i] - fStart] = -1

    for i in range(nfacets_shared):
        f = facets[i]
        if result[f - fStart] == -1:
            if get_global_edge_orientation(plex, vertex_numbering, f) ^ (ours[i] >= 0):
                orientation = 0
            else:
                orientation = 1

            plex_get_restricted_support(plex, <PetscInt *>cell_ranks.data, f,
                                        &size, cells)

            result[f - fStart] = orientation
            for ci in range(size):
                traverse_cell_string(plex, f, cells[ci],
                                     <PetscInt *>cell_ranks.data,
                                     <np.int8_t *>result.data)

    CHKERR(PetscFree(facets))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def orientations_facet2cell(
    PETSc.DM plex, PETSc.Section vertex_numbering,
    np.ndarray[PetscInt, ndim=1, mode="c"] cell_ranks,
    np.ndarray[np.int8_t, ndim=1, mode="c"] facet_orientations,
    PETSc.Section cell_numbering):

    """Converts local quadrilateral facet orientations into
    global quadrilateral cell orientations.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg facet_orientations: Facet orientations (edge directions) relative
                             to the local DMPlex ordering.
    :arg cell_numbering: Section describing the cell numbering
    """
    cdef:
        PetscInt c, cStart, cEnd, ncells, cell
        PetscInt fStart, fEnd
        const PetscInt *cone = NULL
        const PetscInt *cone_orient = NULL
        np.int8_t dst_orient[4]
        int i, off
        PetscInt facet, v, V
        np.ndarray[PetscInt, ndim=1, mode="c"] cell_orientations

    get_height_stratum(plex.dm, 0, &cStart, &cEnd)
    get_height_stratum(plex.dm, 1, &fStart, &fEnd)
    ncells = cEnd - cStart

    cell_orientations = np.zeros(ncells, dtype=IntType)

    for c in range(cStart, cEnd):
        if cell_ranks[c - cStart] < 0:
            CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))

            CHKERR(DMPlexGetCone(plex.dm, c, &cone))
            CHKERR(DMPlexGetConeOrientation(plex.dm, c, &cone_orient))

            # Cone orientations describe which edges to flip (relative to
            # plex edge directions) to get circularly directed edges.
            #
            #   o--<--o
            #   |     |
            #   v     ^
            #   |     |
            #   o-->--o
            #
            # "facet_orientations" describe which edge to flip (relative
            # to plex edge directions) to get edge directions like below
            # for each quadrilateral:
            #
            #   o-->--o
            #   |     |
            #   ^     ^
            #   |     |
            #   X-->--o
            #
            # Their XOR describes the desired edge directions relative to
            # the traversal direction of the cone. This is always a
            # circular permutation of:
            #
            #   straight -- straight -- reverse -- reverse
            #
            for i in range(4):
                dst_orient[i] = (cone_orient[i] < 0) ^ facet_orientations[cone[i] - fStart]

            # We select vertex X (figure above) as starting vertex.
            # We only calculate starting vertices for the locally owned
            # cells, and later exchange these values on the halo cells.
            if dst_orient[2] and dst_orient[3]:
                off = 0
            elif dst_orient[3] and dst_orient[0]:
                off = 1
            elif dst_orient[0] and dst_orient[1]:
                off = 2
            elif dst_orient[1] and dst_orient[2]:
                off = 3
            else:
                raise RuntimeError("Please get the facet orientation right first!")

            # Cell orientation values are defined to be
            # the global number of the starting vertex.
            facet = cone[off]

            CHKERR(DMPlexGetCone(plex.dm, facet, &cone))
            if cone_orient[off] >= 0:
                v = cone[0]
            else:
                v = cone[1]

            CHKERR(PetscSectionGetOffset(vertex_numbering.sec, v, &V))
            cell_orientations[cell] = cabs(V)

    return cell_orientations


@cython.boundscheck(False)
@cython.wraparound(False)
def exchange_cell_orientations(
    PETSc.DM plex, PETSc.Section section,
    np.ndarray[PetscInt, ndim=1, mode="c"] orientations):

    """Halo exchange of cell orientations.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg section: Section describing the cell numbering
    :arg orientations: Cell orientations to exchange,
                       values in the halo will be overwritten.
    """
    cdef:
        PETSc.SF sf
        PetscInt nroots, nleaves
        const PetscInt *ilocal = NULL
        const PetscSFNode *iremote = NULL
        MPI.Datatype dtype
        PETSc.Section new_section
        PetscInt *new_values = NULL
        PetscInt i, c, cStart, cEnd, l, r

    try:
        try:
            dtype = MPI.__TypeDict__[np.dtype(IntType).char]
        except AttributeError:
            dtype = MPI._typedict[np.dtype(IntType).char]
    except KeyError:
        raise ValueError("Don't know how to create datatype for %r", PETSc.IntType)
    # Halo exchange of cell orientations, i.e. receive orientations
    # from the owners in the halo region.
    if plex.comm.size > 1 and plex.isDistributed():
        sf = plex.getPointSF()
        CHKERR(PetscSFGetGraph(sf.sf, &nroots, &nleaves, &ilocal, &iremote))

        new_section = PETSc.Section().create(comm=plex.comm)
        CHKERR(DMPlexDistributeData(plex.dm, sf.sf, section.sec,
                                    dtype.ob_mpi, <void *>orientations.data,
                                    new_section.sec, <void **>&new_values))

        # Overwrite values in the halo region with remote values
        get_height_stratum(plex.dm, 0, &cStart, &cEnd)
        for i in range(nleaves):
            c = ilocal[i]
            if cStart <= c < cEnd:
                CHKERR(PetscSectionGetOffset(section.sec, c, &l))
                CHKERR(PetscSectionGetOffset(new_section.sec, c, &r))

                orientations[l] = new_values[r]

    if new_values != NULL:
        CHKERR(PetscFree(new_values))


@cython.boundscheck(False)
@cython.wraparound(False)
def make_global_numbering(PETSc.Section lsec, PETSc.Section gsec):
    """Build an array of global numbers for local dofs

    :arg lsec: Section describing local dof layout and numbers.
    :arg gsec: Section describing global dof layout and numbers."""
    cdef:
        PetscInt c, p, pStart, pEnd, dof, loff, goff
        np.ndarray[PetscInt, ndim=1, mode="c"] val

    val = np.empty(lsec.getStorageSize(), dtype=IntType)
    pStart, pEnd = lsec.getChart()

    for p in range(pStart, pEnd):
        CHKERR(PetscSectionGetDof(lsec.sec, p, &dof))
        if dof > 0:
            CHKERR(PetscSectionGetOffset(lsec.sec, p, &loff))
            CHKERR(PetscSectionGetOffset(gsec.sec, p, &goff))
            goff = cabs(goff)
            for c in range(dof):
                val[loff + c] = goff + c
    return val


cdef int DMPlexGetAdjacency_Facet_Support(PETSc.PetscDM dm,
                                          PetscInt p,
                                          PetscInt *adjSize,
                                          PetscInt adj[],
                                          void *ctx) nogil:
    """Custom adjacency callback for halo growth.

    :arg dm: The DMPlex object.
    :arg p: The mesh point to compute the adjacency of.
    :arg adjSize: Output parameter, the size of the computed adjacency.
    :arg adj: Output parameter, the adjacent mesh points.
    :arg ctx: User context.

    The halo we need for owner-computes is everything in the stencil
    of the owned mesh points.  For cells, we already have everything,
    for facets, if we own the facet, we need the mesh points in
    closure(support(facet)).  This function returns non-zero adjacency
    only for facets, which then means that everything else falls
    through right.
    """
    cdef:
        const PetscInt *support = NULL;
        PetscInt numAdj = 0
        PetscInt maxAdjSize = adjSize[0]
        PetscInt supportSize
        PetscInt s
        PetscInt fStart, fEnd
        PetscInt point, closureSize, ci, q
        PetscInt *closure = NULL
        DMLabel label = <DMLabel>ctx;
        PetscBool flg = PETSC_TRUE

    CHKERR(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd))
    if not (fStart <= p < fEnd):
        # Not a facet, no adjacent points
        adjSize[0] = 0
        return 0
    if label != NULL:
        # If a label is provided to filter out points, use it.
        # Requires that the label has already had an index created.
        # The label should mark those points that are not owned.
        # If the point is owned, then we would like to grow the halo.
        # So we need the remote process to donate those points.
        # Hence, if we own the point, we return an empty adjacency (we
        # don't want to donate those points to the remote process),
        # and vice versa.
        CHKERR(DMLabelHasPoint(label, p, &flg))
        if not flg:
            # This point is owned, no adjacency.
            adjSize[0] = 0
            return 0
    # OK, it's a remote point, let's gather the adjacency
    CHKERR(DMPlexGetSupportSize(dm, p, &supportSize))
    CHKERR(DMPlexGetSupport(dm, p, &support))
    for s in range(supportSize):
        point = support[s]
        CHKERR(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure))
        for ci in range(closureSize):
            # This is just ensuring that the adjacency is unique.
            for q in range(numAdj):
                if closure[2*ci] == adj[q]:
                    break
            else:
                adj[numAdj] = closure[2*ci]
                numAdj += 1
            # Too many adjacent points for the provided output array.
            if numAdj > maxAdjSize:
                SETERR(77)
    CHKERR(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &closureSize, &closure))
    adjSize[0] = numAdj
    return 0


def set_adjacency_callback(PETSc.DM dm not None):
    """Set the callback for DMPlexGetAdjacency.

    :arg dm: The DMPlex object.

    This is used during DMPlexDistributeOverlap to determine where to
    grow the halos."""
    cdef:
        PetscInt pStart, pEnd, p
        DMLabel label = NULL
        PETSc.SF sf
        PetscInt nleaves
        const PetscInt *ilocal

    if False:
        # In theory we can grow halos asymmetrically, but in practice
        # the implementation of parallel quad orientation relies on
        # the halo being symmetric.

        # Mark remote points from point overlap SF
        sf = dm.getPointSF()
        CHKERR(PetscSFGetGraph(sf.sf, NULL, &nleaves, &ilocal, NULL))
        dm.createLabel("ghost_region")
        CHKERR(DMGetLabel(dm.dm, "ghost_region", &label))
        get_chart(dm.dm, &pStart, &pEnd)
        for p in range(nleaves):
            CHKERR(DMLabelSetValue(label, ilocal[p], 1))
        CHKERR(DMLabelCreateIndex(label, pStart, pEnd))
    CHKERR(DMPlexSetAdjacencyUser(dm.dm, DMPlexGetAdjacency_Facet_Support, NULL))


def clear_adjacency_callback(PETSc.DM dm not None):
    """Clear the callback for DMPlexGetAdjacency.

    :arg dm: The DMPlex object"""
    cdef:
        DMLabel label = NULL

    if False:
        CHKERR(DMGetLabel(dm.dm, "ghost_region", &label))
        CHKERR(DMLabelDestroyIndex(label))
        dm.removeLabel("ghost_region")
        CHKERR(DMLabelDestroy(&label))
    CHKERR(DMPlexSetAdjacencyUser(dm.dm, NULL, NULL))


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_point_cone_global_sizes(PETSc.DM dm):
    """Compute the total number of DMPlex points and the global sum of cone sizes for dm.

    :arg dm: The dm.
    :returns: A list of global number of points and global sum of cone sizes.
    """
    cdef:
        PETSc.SF sf
        PetscInt nleaves
        const PetscInt *ilocal = NULL
        const PetscSFNode *iremote = NULL
        PetscInt i, p, pStart, pEnd, coneSize
        np.ndarray[PetscInt, ndim=1, mode="c"] arraySizes
        np.ndarray[PetscInt, ndim=1, mode="c"] out

    sf = dm.getPointSF()
    CHKERR(PetscSFGetGraph(sf.sf, NULL, &nleaves, &ilocal, NULL))
    if nleaves < 0:
        # comm.size == 1
        nleaves = 0
    CHKERR(DMPlexGetChart(dm.dm, &pStart, &pEnd))
    arraySizes = np.zeros((2, ), dtype=IntType)
    arraySizes[0] = (pEnd - pStart) - nleaves
    for p in range(pStart, pEnd):
        CHKERR(DMPlexGetConeSize(dm.dm, p, &coneSize))
        arraySizes[1] += coneSize;
    for i in range(nleaves):
        CHKERR(DMPlexGetConeSize(dm.dm, ilocal[i], &coneSize))
        arraySizes[1] -= coneSize;
    out = np.zeros((2, ), dtype=IntType)
    dm.comm.tompi4py().Allreduce(arraySizes, out, op=MPI.SUM)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def mark_points_with_function_array(PETSc.DM plex,
                                    PETSc.Section section,
                                    PetscInt height,
                                    np.ndarray[PetscInt, ndim=1, mode="c"] array,
                                    PETSc.DMLabel dmlabel,
                                    PetscInt label_value):

    """Marks points in a DMLabel using an indicator function array.

    :arg plex: DMPlex representing the mesh topology
    :arg section: Section describing the function space DoF layout and order
    :arg height: Height of marked points (0 for cells, 1 for facets)
    :arg array: Array representing the indicator function whose layout is
        defined by plex, section, and height
    :arg dmlabel: DMLabel that records marked entities
    :arg label_value: Value used in dmlabel
    """
    cdef:
        PetscInt pStart, pEnd, p, offset

    get_height_stratum(plex.dm, height, &pStart, &pEnd)
    for p in range(pStart, pEnd):
        CHKERR(PetscSectionGetOffset(section.sec, p, &offset))
        if array[offset] == 1:
            CHKERR(DMLabelSetValue(<DMLabel>dmlabel.dmlabel, p, label_value))


def to_petsc_local_numbering(PETSc.Vec vec, V):
    """
    Reorder a PETSc Vec corresponding to a Firedrake Function w.r.t.
    the PETSc natural numbering.

    :arg vec: the PETSc Vec to reorder; must be a global vector
    :arg V: the FunctionSpace of the Function which the Vec comes from
    :ret out: a copy of the Vec, ordered with the PETSc natural numbering
    """
    cdef int dim, idx, start, end, p, d, k
    cdef PetscInt dof, off
    cdef PETSc.Vec out
    cdef PETSc.Section section
    cdef np.ndarray[PetscReal, mode="c", ndim=1] varray, oarray

    section = V.dm.getGlobalSection()
    out = vec.duplicate()
    varray = vec.array_r
    oarray = out.array
    dim = V.value_size
    idx = 0
    start, end = vec.getOwnershipRange()
    for p in range(*section.getChart()):
        CHKERR(PetscSectionGetDof(section.sec, p, &dof))
        if dof > 0:
            CHKERR(PetscSectionGetOffset(section.sec, p, &off))
            assert off >= 0
            off *= dim
            for d in range(dof):
                for k in range(dim):
                    oarray[idx] = varray[off + dim * d + k - start]
                    idx += 1
    assert idx == (end - start)
    return out


def create_halo_exchange_sf(PETSc.DM dm):
    """Create the halo exchange sf.

    Parameters
    ----------
    dm : PETSc.DM
        The section dm.

    Returns
    -------
    PETSc.SF
        The halo exchange sf.

    Notes
    -----
    The output sf is to update all ghost DoFs including constrained ones if any.

    """
    cdef:
        PETSc.SF halo_exchange_sf
        PetscInt dof_nroots, dof_nleaves
        PetscInt *dof_ilocal = NULL
        PetscSFNode *dof_iremote = NULL
        PETSc.SF point_sf
        PetscInt nroots, nleaves
        const PetscInt *ilocal = NULL
        const PetscSFNode *iremote = NULL
        PETSc.Section local_sec
        PetscInt pStart, pEnd, p, dof, off, m, n, i, j
        np.ndarray[PetscInt, ndim=1, mode="c"] local_offsets
        np.ndarray[PetscInt, ndim=1, mode="c"] remote_offsets

    point_sf = dm.getPointSF()
    local_sec = dm.getLocalSection()
    CHKERR(PetscSFGetGraph(point_sf.sf, &nroots, &nleaves, &ilocal, &iremote))
    pStart, pEnd = local_sec.getChart()
    assert pEnd - pStart == nroots, f"pEnd - pStart ({pEnd - pStart}) != nroots ({nroots})"
    assert pStart == 0
    m = 0
    local_offsets = np.empty(pEnd - pStart, dtype=IntType)
    remote_offsets = np.full(pEnd - pStart, -1, dtype=IntType)
    for p in range(pStart, pEnd):
        CHKERR(PetscSectionGetDof(local_sec.sec, p, &dof))
        CHKERR(PetscSectionGetOffset(local_sec.sec, p, &off))
        local_offsets[p] = off
        m += dof
    unit = MPI._typedict[np.dtype(IntType).char]
    point_sf.bcastBegin(unit, local_offsets, remote_offsets, MPI.REPLACE)
    point_sf.bcastEnd(unit, local_offsets, remote_offsets, MPI.REPLACE)
    n = 0
    # ilocal == NULL if local leaf points are [0, 1, 2, ...).
    for i in range(nleaves):
        p = ilocal[i] if ilocal != NULL else i
        CHKERR(PetscSectionGetDof(local_sec.sec, p, &dof))
        n += dof
    CHKERR(PetscMalloc1(n, &dof_ilocal))
    CHKERR(PetscMalloc1(n, &dof_iremote))
    n = 0
    for i in range(nleaves):
        # ilocal == NULL if local leaf points are [0, 1, 2, ...).
        p = ilocal[i] if ilocal != NULL else i
        assert remote_offsets[p] >= 0
        CHKERR(PetscSectionGetDof(local_sec.sec, p, &dof))
        CHKERR(PetscSectionGetOffset(local_sec.sec, p, &off))
        for j in range(dof):
            dof_ilocal[n] = off + j
            dof_iremote[n].rank = iremote[i].rank
            dof_iremote[n].index = remote_offsets[p] + j
            n += 1
    halo_exchange_sf = PETSc.SF().create(comm=point_sf.comm)
    CHKERR(PetscSFSetGraph(halo_exchange_sf.sf, m, n, dof_ilocal, PETSC_OWN_POINTER, dof_iremote, PETSC_OWN_POINTER))
    return halo_exchange_sf
