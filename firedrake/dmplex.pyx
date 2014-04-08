# Utility functions to derive global and local numbering from DMPlex
from petsc import PETSc
from pyop2 import MPI
import numpy as np
cimport numpy as np
import cython
cimport petsc4py.PETSc as PETSc
from operator import itemgetter

np.import_array()

cdef extern from "petsc.h":
   ctypedef long PetscInt
   ctypedef enum PetscBool:
       PETSC_TRUE, PETSC_FALSE

cdef extern from "petscsys.h":
   int PetscMalloc1(PetscInt,PetscInt**)
   int PetscFree(PetscInt*)
   int PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[])

cdef extern from "petscdmplex.h":
    int DMPlexGetConeSize(PETSc.PetscDM,PetscInt,PetscInt*)
    int DMPlexGetCone(PETSc.PetscDM,PetscInt,PetscInt*[])

    int DMPlexGetTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])
    int DMPlexRestoreTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])

cdef extern from "petscis.h":
    int PetscSectionGetOffset(PETSc.PetscSection,PetscInt,PetscInt*)

def _from_cell_list(dim, cells, coords, comm=None):
    """
    Create a DMPlex from a list of cells and coords.

    :arg dim: The topological dimension of the mesh
    :arg cells: The vertices of each cell
    :arg coords: The coordinates of each vertex
    :arg comm: An optional communicator to build the plex on (defaults to COMM_WORLD)
    """

    if comm is None:
        comm = MPI.comm
    if comm.rank == 0:
        cells = np.asarray(cells, dtype=PETSc.IntType)
        coords = np.asarray(coords, dtype=float)
        comm.bcast(cells.shape, root=0)
        comm.bcast(coords.shape, root=0)
        # Provide the actual data on rank 0.
        return PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=comm)

    cell_shape = list(comm.bcast(None, root=0))
    coord_shape = list(comm.bcast(None, root=0))
    cell_shape[0] = 0
    coord_shape[0] = 0
    # Provide empty plex on other ranks
    # A subsequent call to plex.distribute() takes care of parallel partitioning
    return PETSc.DMPlex().createFromCellList(dim,
                                             np.zeros(cell_shape, dtype=PETSc.IntType),
                                             np.zeros(coord_shape, dtype=float),
                                             comm=comm)

def facet_numbering(plex, vertex_numbering, facet):
    """Derive local facet number according to Fenics"""
    cells = plex.getSupport(facet)
    local_facet = []
    for c in cells:
        closure = plex.getTransitiveClosure(c)[0]

        # Local vertex numbering according to universal vertex numbering
        vStart, vEnd = plex.getDepthStratum(0)   # vertices
        is_vertex = lambda v: vStart <= v < vEnd
        vertices = filter(is_vertex, closure)
        v_glbl = [vertex_numbering.getOffset(v) for v in vertices]
        v_glbl = [v if v >= 0 else -(v+1) for v in v_glbl]
        vertices, v_glbl = zip(*sorted(zip(vertices, v_glbl), key=itemgetter(1)))

        # Local facet number := local number of non-incident vertex
        v_incident = filter(is_vertex, plex.getTransitiveClosure(facet)[0])
        v_non_incident = [v for v in vertices if v not in v_incident][0]
        local_facet.append(np.where(vertices == v_non_incident)[0][0])
    return local_facet

@cython.boundscheck(False)
@cython.wraparound(False)
def closure_ordering(PETSc.DM plex,
                     PETSc.Section vertex_numbering,
                     PETSc.Section cell_numbering,
                     np.ndarray[np.int32_t] entity_per_cell):
    """Apply Fenics local numbering to a cell closure.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg cell_numbering: Section describing the universal vertex numbering
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
        PetscInt dim, vi, ci, fi, v_per_cell, cell
        PetscInt offset, cell_offset, nfaces, nfacets
        PetscInt nclosure, nfacet_closure, nface_vertices
        PetscInt *vertices = NULL
        PetscInt *v_global = NULL
        PetscInt *closure = NULL
        PetscInt *facets = NULL
        PetscInt *facet_closure = NULL
        PetscInt *faces = NULL
        PetscInt *face_indices = NULL
        PetscInt *face_vertices = NULL
        PetscInt *facet_vertices = NULL
        np.ndarray[np.int32_t, ndim=2] cell_closure

    dim = plex.getDimension()
    cStart, cEnd = plex.getHeightStratum(0)
    fStart, fEnd = plex.getHeightStratum(1)
    eStart, eEnd = plex.getDepthStratum(1)
    vStart, vEnd = plex.getDepthStratum(0)
    v_per_cell = entity_per_cell[0]
    cell_offset = sum(entity_per_cell) - 1

    PetscMalloc1(v_per_cell, &vertices)
    PetscMalloc1(v_per_cell, &v_global)
    PetscMalloc1(v_per_cell-1, &facets)
    PetscMalloc1(v_per_cell-1, &facet_vertices)
    PetscMalloc1(entity_per_cell[1], &faces)
    PetscMalloc1(entity_per_cell[1], &face_indices)
    cell_closure = np.empty((cEnd - cStart, sum(entity_per_cell)), dtype=np.int32)

    for c in range(cStart, cEnd):
        PetscSectionGetOffset(cell_numbering.sec, c, &cell)
        DMPlexGetTransitiveClosure(plex.dm, c, PETSC_TRUE, &nclosure,&closure)

        # Find vertices and translate universal numbers
        vi = 0
        for ci in range(nclosure):
            if vStart <= closure[2*ci] < vEnd:
                vertices[vi] = closure[2*ci]
                PetscSectionGetOffset(vertex_numbering.sec, closure[2*ci], &v)
                # Correct -ve offsets for non-owned entities
                if v >= 0:
                    v_global[vi] = v
                else:
                    v_global[vi] = -(v+1)
                vi += 1

        # Sort vertices by universal number
        PetscSortIntWithArray(v_per_cell,v_global,vertices)
        for vi in range(v_per_cell):
            if dim == 1:
                # Correct 1D edge numbering
                cell_closure[cell, vi] = vertices[v_per_cell-vi-1]
            else:
                cell_closure[cell, vi] = vertices[vi]
        offset = v_per_cell

        # Find all faces (dim=1)
        if dim > 2:
            nfaces = 0
            for ci in range(nclosure):
                if eStart <= closure[2*ci] < eEnd:
                    faces[nfaces] = closure[2*ci]

                    DMPlexGetConeSize(plex.dm, closure[2*ci], &nface_vertices)
                    DMPlexGetCone(plex.dm, closure[2*ci], &face_vertices)

                    # Faces in 3D are tricky because we need a
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
            PetscSortIntWithArray(entity_per_cell[1], face_indices, faces)
            for fi in range(nfaces):
                cell_closure[cell, offset+fi] = faces[fi]
            offset += nfaces

        # Calling DMPlexGetTransitiveClosure() again invalidates the
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

        # Now we can deal with facets
        if dim > 1:
            for f in range(nfacets):
                # Derive facet vertices from facet_closure
                DMPlexGetTransitiveClosure(plex.dm, facets[f],
                                           PETSC_TRUE,
                                           &nfacet_closure,
                                           &facet_closure)
                vi = 0
                for fi in range(nfacet_closure):
                    if vStart <= facet_closure[2*fi] < vEnd:
                        facet_vertices[vi] = facet_closure[2*fi]
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

            DMPlexRestoreTransitiveClosure(plex.dm, facets[f], PETSC_TRUE,
                                           &nfacet_closure, &facet_closure)
            offset += nfacets

    DMPlexRestoreTransitiveClosure(plex.dm, c, PETSC_TRUE, &nclosure,&closure)
    PetscFree(vertices)
    PetscFree(v_global)
    PetscFree(facets)
    PetscFree(facet_vertices)
    PetscFree(faces)

    return cell_closure


def mark_entity_classes(plex):
    """Mark all points in a given Plex according to the PyOP2 entity classes:
    core      : owned and not in send halo
    non_core  : owned and in send halo
    exec_halo : in halo, but touch owned entity
    """
    plex.createLabel("op2_core")
    plex.createLabel("op2_non_core")
    plex.createLabel("op2_exec_halo")

    if MPI.comm.size > 1:
        # Mark exec_halo from point overlap SF
        point_sf = plex.getPointSF()
        nroots, nleaves, local, remote = point_sf.getGraph()
        for p in local:
            depth = plex.getLabelValue("depth", p)
            plex.setLabelValue("op2_exec_halo", p, depth)
    else:
        # If sequential mark all points as core
        pStart, pEnd = plex.getChart()
        for p in range(pStart, pEnd):
            depth = plex.getLabelValue("depth", p)
            plex.setLabelValue("op2_core", p, depth)
        return

    # Mark all unmarked points in the closure of adjacent cells as non_core
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)
    dim = plex.getDimension()
    halo_cells = plex.getStratumIS("op2_exec_halo", dim).getIndices()
    adjacent_cells = []
    for c in halo_cells:
        halo_closure = plex.getTransitiveClosure(c)[0]
        for vertex in filter(lambda x: x >= vStart and x < vEnd, halo_closure):
            star = plex.getTransitiveClosure(vertex, useCone=False)[0]
            for adj in filter(lambda x: x >= cStart and x < cEnd, star):
                if plex.getLabelValue("op2_exec_halo", adj) < 0:
                    adjacent_cells.append(adj)

    for adj_cell in adjacent_cells:
        for p in plex.getTransitiveClosure(adj_cell)[0]:
            if plex.getLabelValue("op2_exec_halo", p) < 0:
                depth = plex.getLabelValue("depth", p)
                plex.setLabelValue("op2_non_core", p, depth)

    # Mark all remaining points as core
    pStart, pEnd = plex.getChart()
    for p in range(pStart, pEnd):
        exec_halo = plex.getLabelValue("op2_exec_halo", p)
        non_core = plex.getLabelValue("op2_non_core", p)
        if exec_halo < 0 and non_core < 0:
            depth = plex.getLabelValue("depth", p)
            plex.setLabelValue("op2_core", p, depth)


def get_entities_by_class(plex, depth, condition=None):
    """Get a list of Plex entities sorted by the PyOP2 entity classes"""
    entity_classes = [0, 0, 0, 0]
    entities = np.array([], dtype=np.int32)
    if plex.getStratumSize("op2_core", depth) > 0:
        core = plex.getStratumIS("op2_core", depth).getIndices()
        if condition:
            core = filter(condition, core)
        entities = np.concatenate([entities, core])
    entity_classes[0] = entities.size
    if plex.getStratumSize("op2_non_core", depth) > 0:
        non_core = plex.getStratumIS("op2_non_core", depth).getIndices()
        if condition:
            non_core = filter(condition, non_core)
        entities = np.concatenate([entities, non_core])
    entity_classes[1] = entities.size
    if plex.getStratumSize("op2_exec_halo", depth) > 0:
        exec_halo = plex.getStratumIS("op2_exec_halo", depth).getIndices()
        if condition:
            exec_halo = filter(condition, exec_halo)
        entities = np.concatenate([entities, exec_halo])
    entity_classes[2] = entities.size
    entity_classes[3] = entities.size
    return entities, entity_classes


def plex_renumbering(plex):
    """
    Build a global node renumbering as a permutation of Plex points.

    :arg plex: The DMPlex object encapsulating the mesh topology

    The node permutation is derived from a depth-first traversal of
    the Plex graph over each OP2 entity class in turn. The returned IS
    is the Plex -> OP2 permutation.
    """
    dim = plex.getDimension()
    pStart, pEnd = plex.getChart()
    perm = np.empty(pEnd - pStart, dtype=np.int32)
    p_glbl = 0

    # Renumber core DoFs
    seen = set()
    if plex.getStratumSize("op2_core", dim) > 0:
        for cell in plex.getStratumIS("op2_core", dim).getIndices():
            for p in plex.getTransitiveClosure(cell)[0]:
                if p in seen:
                    continue

                if plex.getLabelValue("op2_core", p) >= 0:
                    seen.add(p)
                    perm[p_glbl] = p
                    p_glbl += 1

    # Renumber non-core DoFs
    if plex.getStratumSize("op2_non_core", dim) > 0:
        for cell in plex.getStratumIS("op2_non_core", dim).getIndices():
            for p in plex.getTransitiveClosure(cell)[0]:
                if p in seen:
                    continue

                if plex.getLabelValue("op2_non_core", p) >= 0:
                    seen.add(p)
                    perm[p_glbl] = p
                    p_glbl += 1

    # Renumber halo DoFs
    if plex.getStratumSize("op2_exec_halo", dim) > 0:
        for cell in plex.getStratumIS("op2_exec_halo", dim).getIndices():
            for p in plex.getTransitiveClosure(cell)[0]:
                if p in seen:
                    continue

                if plex.getLabelValue("op2_exec_halo", p) >= 0:
                    seen.add(p)
                    perm[p_glbl] = p
                    p_glbl += 1

    return PETSc.IS().createGeneral(perm)
