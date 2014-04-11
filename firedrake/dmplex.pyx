# Utility functions to derive global and local numbering from DMPlex
from petsc import PETSc
from pyop2 import MPI
import numpy as np
cimport numpy as np
import cython
cimport petsc4py.PETSc as PETSc

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
    int DMPlexGetSupportSize(PETSc.PetscDM,PetscInt,PetscInt*)
    int DMPlexGetSupport(PETSc.PetscDM,PetscInt,PetscInt*[])

    int DMPlexGetTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])
    int DMPlexRestoreTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])

    int DMPlexGetLabelValue(PETSc.PetscDM,char[],PetscInt,PetscInt*)

cdef extern from "petscis.h":
    int PetscSectionGetOffset(PETSc.PetscSection,PetscInt,PetscInt*)
    int ISGetIndices(PETSc.PetscIS,PetscInt*[])

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

@cython.boundscheck(False)
@cython.wraparound(False)
def facet_numbering(PETSc.DM plex, kind,
                    np.ndarray[np.int32_t] facets,
                    PETSc.Section cell_numbering,
                    np.ndarray[np.int32_t, ndim=2] cell_closures):
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
        PetscInt *cells = NULL
        np.ndarray[np.int32_t, ndim=2] facet_cells
        np.ndarray[np.int32_t, ndim=2] facet_local_num

    fStart, fEnd = plex.getHeightStratum(1)
    nfacets = facets.shape[0]
    nclosure = cell_closures.shape[1]

    assert(kind in ["interior", "exterior"])
    if kind == "interior":
        cells_per_facet = 2
    else:
        cells_per_facet = 1
    facet_local_num = np.empty((nfacets, cells_per_facet), dtype=np.int32)
    facet_cells = np.empty((nfacets, cells_per_facet), dtype=np.int32)

    # First determine the parent cell(s) for each facet
    for f in range(nfacets):
        DMPlexGetSupport(plex.dm, facets[f], &cells)
        DMPlexGetSupportSize(plex.dm, facets[f], &ncells)
        PetscSectionGetOffset(cell_numbering.sec, cells[0], &cell)
        facet_cells[f,0] = cell
        if cells_per_facet > 1:
            if ncells > 1:
                PetscSectionGetOffset(cell_numbering.sec,
                                      cells[1], &cell)
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
            if cell > 0:
                fi = 0
                for c in range(nclosure):
                    if cell_closures[cell, c] == facets[f]:
                        facet_local_num[f,1] = fi
                    if fStart <= cell_closures[cell, c] < fEnd:
                        fi += 1
            else:
                facet_local_num[f,1] = -1
    return facet_local_num, facet_cells

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

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cells_by_class(PETSc.DM plex):
    """Builds a list of all cells ordered according to OP2 entity
    classes and computes the respective class offsets.

    :arg plex: The DMPlex object encapsulating the mesh topology
    """
    cdef:
        PetscInt dim, c, ci, nclass
        PetscInt *indices = NULL
        PETSc.IS class_is = None
        np.ndarray[np.int32_t] cells

    dim = plex.getDimension()
    cStart, cEnd = plex.getHeightStratum(0)
    cells = np.empty(cEnd - cStart, dtype=np.int32)
    cell_classes = [0, 0, 0, 0]
    c = 0

    for i, op2class in enumerate(["op2_core",
                                  "op2_non_core",
                                  "op2_exec_halo"]):
        nclass = plex.getStratumSize(op2class, dim)
        if nclass > 0:
            class_is = plex.getStratumIS(op2class, dim)
            ISGetIndices(class_is.iset, &indices)
            for ci in range(nclass):
                cells[c] = indices[ci]
                c += 1
        cell_classes[i] = c

    cell_classes[3] = cell_classes[2]
    return cells, cell_classes

@cython.boundscheck(False)
@cython.wraparound(False)
def get_facets_by_class(PETSc.DM plex, label):
    """Builds a list of all facets ordered according to OP2 entity
    classes and computes the respective class offsets.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg label: Label string that marks the facets to order
    """
    cdef:
        PetscInt dim, fi, ci, nfacets, nclass, lbl_val
        PetscInt *indices = NULL
        PETSc.IS class_is = None
        char *class_chr = NULL
        np.ndarray[np.int32_t] facets

    label_chr = <char*>label
    dim = plex.getDimension()
    nfacets = plex.getStratumSize(label, 1)
    facets = np.empty(nfacets, dtype=np.int32)
    facet_classes = [0, 0, 0, 0]
    fi = 0

    for i, op2class in enumerate(["op2_core",
                                  "op2_non_core",
                                  "op2_exec_halo"]):
        nclass = plex.getStratumSize(op2class, dim-1)
        if nclass > 0:
            class_is = plex.getStratumIS(op2class, dim-1)
            ISGetIndices(class_is.iset, &indices)
            for ci in range(nclass):
                DMPlexGetLabelValue(plex.dm, label_chr,
                                    indices[ci], &lbl_val)
                if lbl_val == 1:
                    facets[fi] = indices[ci]
                    fi += 1
        facet_classes[i] = fi

    facet_classes[3] = facet_classes[2]
    return facets, facet_classes

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
