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
   ctypedef enum PetscCopyMode:
       PETSC_COPY_VALUES,
       PETSC_OWN_POINTER,
       PETSC_USE_POINTER

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
    int DMPlexSetLabelValue(PETSc.PetscDM,char[],PetscInt,PetscInt)

cdef extern from "petscis.h":
    int PetscSectionGetOffset(PETSc.PetscSection,PetscInt,PetscInt*)
    int PetscSectionGetDof(PETSc.PetscSection,PetscInt,PetscInt*)
    int ISGetIndices(PETSc.PetscIS,PetscInt*[])
    int ISGeneralSetIndices(PETSc.PetscIS,PetscInt,PetscInt[],PetscCopyMode)

cdef extern from "petscsf.h":
    struct PetscSFNode:
        PetscInt rank
        PetscInt index
    ctypedef PetscSFNode PetscSFNode "PetscSFNode"

    int PetscSFGetGraph(PETSc.PetscSF,PetscInt*,PetscInt*,PetscInt**,PetscSFNode**)

cdef extern from "petscbt.h":
    ctypedef char * PetscBT
    int PetscBTCreate(PetscInt,PetscBT*)
    int PetscBTDestroy(PetscBT*)
    char PetscBTLookup(PetscBT,PetscInt)
    int PetscBTSet(PetscBT,PetscInt)

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

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_nodes(PETSc.Section global_numbering,
                   np.ndarray[np.int32_t, ndim=2] cell_closures,
                   dofs_per_cell):
    """
    Builds the DoF mapping for non-extruded meshes.

    :arg global_numbering: Section describing the global DoF numbering
    :arg cell_closures: 2D array of ordered cell closures
    :arg dofs_per_cell: Number of DoFs associated with each mesh cell
    """
    cdef:
        PetscInt c, ncells, ci, nclosure, offset, p, pi, dof, off, i
        np.ndarray[np.int32_t, ndim=2] cell_nodes

    ncells = cell_closures.shape[0]
    nclosure = cell_closures.shape[1]
    cell_nodes = np.empty((ncells, dofs_per_cell), dtype=np.int32)

    for c in range(ncells):
        offset = 0
        for ci in range(nclosure):
            p = cell_closures[c, ci]
            PetscSectionGetDof(global_numbering.sec, p, &dof)
            PetscSectionGetOffset(global_numbering.sec, p, &off)
            for i in range(dof):
                cell_nodes[c, offset+i] = off+i
            offset += dof
    return cell_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def get_extruded_cell_nodes(PETSc.DM plex,
                            PETSc.Section global_numbering,
                            np.ndarray[np.int32_t, ndim=2] cell_closures,
                            fiat_element, dofs_per_cell):
    """
    Builds the DoF mapping for extruded meshes.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg global_numbering: Section describing the global DoF numbering
    :arg cell_closures: 2D array of ordered cell closures
    :arg fiat_element: The FIAT element for the extruded cell
    :arg dofs_per_cell: Number of DoFs associated with each mesh cell
    """
    cdef:
        PetscInt c, ncells, ci, nclosure, d, dim
        PetscInt offset, p, pdof, off, glbl, lcl, i
        PetscInt *pStarts = NULL
        PetscInt *pEnds = NULL
        PetscInt *hdofs = NULL
        PetscInt *vdofs = NULL
        np.ndarray[np.int32_t, ndim=2] cell_nodes

    ncells = cell_closures.shape[0]
    nclosure = cell_closures.shape[1]
    cell_nodes = np.empty((ncells, dofs_per_cell), dtype=np.int32)

    dim = plex.getDimension()
    PetscMalloc1(dim+1, &pStarts)
    PetscMalloc1(dim+1, &pEnds)
    for d in range(dim+1):
        pStarts[d], pEnds[d] = plex.getDepthStratum(d)

    entity_dofs = fiat_element.entity_dofs()
    PetscMalloc1(dim+1, &hdofs)
    PetscMalloc1(dim+1, &vdofs)
    for d in range(dim+1):
        hdofs[d] = len(entity_dofs[(d,0)][0])
        vdofs[d] = len(entity_dofs[(d,1)][0])

    flattened_element = fiat_element.flattened_element()
    flat_entity_dofs = flattened_element.entity_dofs()

    for c in range(ncells):
        offset = 0
        for d in range(dim+1):
            pi = 0
            for ci in range(nclosure):
                if pStarts[d] <= cell_closures[c, ci] < pEnds[d]:
                    p = cell_closures[c, ci]
                    PetscSectionGetDof(global_numbering.sec, p, &pdof)
                    if pdof > 0:
                        PetscSectionGetOffset(global_numbering.sec, p, &glbl)

                        # For extruded entities the numberings are:
                        # Global: [bottom[:], top[:], side[:]]
                        # Local:  [bottom[i], top[i], side[i] for i in bottom[:]]
                        #
                        # eg. extruded P3 facet:
                        #       Local            Global
                        #  --1---6---11--   --12---13---14--
                        #  | 4   9   14 |   |  5    8   11 |
                        #  | 3   8   13 |   |  4    7   10 |
                        #  | 2   7   12 |   |  3    6    9 |
                        #  --0---5---10--   ---0----1----2--
                        #
                        # cell_nodes = [0,12,3,4,5,1,13,6,7,8,2,14,9,10,11]

                        lcl_dofs = flat_entity_dofs[d][pi]
                        for i in range(hdofs[d]):
                            lcl = lcl_dofs[i]
                            cell_nodes[c, lcl] = glbl + i
                        for i in range(vdofs[d]):
                            lcl = lcl_dofs[hdofs[d] + i]
                            cell_nodes[c, lcl] = glbl + hdofs[d] + i
                        for i in range(hdofs[d]):
                            lcl = lcl_dofs[hdofs[d] + vdofs[d] + i]
                            cell_nodes[c, lcl] = glbl + hdofs[d] + vdofs[d] + i

                        offset += 2*hdofs[d] + vdofs[d]
                        pi += 1
    PetscFree(pStarts)
    PetscFree(pEnds)
    PetscFree(hdofs)
    PetscFree(vdofs)
    return cell_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def mark_entity_classes(PETSc.DM plex):
    """Mark all points in a given Plex according to the PyOP2 entity
    classes:

    core      : owned and not in send halo
    non_core  : owned and in send halo
    exec_halo : in halo, but touch owned entity

    :arg plex: The DMPlex object encapsulating the mesh topology
    """
    cdef:
        PetscInt p, pStart, pEnd, cStart, cEnd, vStart, vEnd
        PetscInt c, ncells, ci, nclosure, vi, dim
        PetscInt depth, non_core, exec_halo, nroots, nleaves
        PetscInt *cells = NULL
        PetscInt *vertices = NULL
        PetscInt *closure = NULL
        PetscInt *ilocal = NULL
        PetscSFNode *iremote = NULL
        PETSc.SF point_sf = None
        PETSc.IS cell_is = None

    dim = plex.getDimension()
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)
    v_per_cell = plex.getConeSize(cStart)
    PetscMalloc1(v_per_cell, &vertices)

    plex.createLabel("op2_core")
    plex.createLabel("op2_non_core")
    plex.createLabel("op2_exec_halo")

    lbl_depth = <char*>"depth"
    lbl_core = <char*>"op2_core"
    lbl_non_core = <char*>"op2_non_core"
    lbl_halo = <char*>"op2_exec_halo"

    if MPI.comm.size > 1:
        # Mark exec_halo from point overlap SF
        point_sf = plex.getPointSF()
        PetscSFGetGraph(point_sf.sf, &nroots, &nleaves, &ilocal, &iremote)
        for p in range(nleaves):
            DMPlexGetLabelValue(plex.dm, lbl_depth, ilocal[p], &depth)
            DMPlexSetLabelValue(plex.dm, lbl_halo, ilocal[p], depth)
    else:
        # If sequential mark all points as core
        pStart, pEnd = plex.getChart()
        for p in range(pStart, pEnd):
            DMPlexGetLabelValue(plex.dm, lbl_depth, p, &depth)
            DMPlexSetLabelValue(plex.dm, lbl_core, p, depth)
        return

    # Mark all cells adjacent to halo cells as non_core,
    # where adjacent(c) := star(closure(c))
    ncells = plex.getStratumSize("op2_exec_halo", dim)
    cell_is = plex.getStratumIS("op2_exec_halo", dim)
    ISGetIndices(cell_is.iset, &cells)
    for c in range(ncells):
        DMPlexGetTransitiveClosure(plex.dm, cells[c], PETSC_TRUE,
                                   &nclosure, &closure)
        # Copy vertices out of the work array (closure)
        vi = 0
        for ci in range(nclosure):
            if vStart <= closure[2*ci] < vEnd:
                vertices[vi] = closure[2*ci]
                vi += 1

        # Mark all cells in the star of each vertex
        for vi in range(v_per_cell):
            vertex = vertices[vi]
            DMPlexGetTransitiveClosure(plex.dm, vertices[vi], PETSC_FALSE,
                                       &nclosure, &closure)
            for ci in range(nclosure):
                if cStart <= closure[2*ci] < cEnd:
                    p = closure[2*ci]
                    DMPlexGetLabelValue(plex.dm, lbl_halo, p, &exec_halo)
                    if exec_halo < 0:
                        DMPlexSetLabelValue(plex.dm, lbl_non_core, p, dim)

    # Mark the closures of non_core cells as non_core
    ncells = plex.getStratumSize("op2_non_core", dim)
    cell_is = plex.getStratumIS("op2_non_core", dim)
    ISGetIndices(cell_is.iset, &cells)
    for c in range(ncells):
        DMPlexGetTransitiveClosure(plex.dm, cells[c], PETSC_TRUE,
                                   &nclosure, &closure)
        for ci in range(nclosure):
            p = closure[2*ci]
            DMPlexGetLabelValue(plex.dm, lbl_halo, p, &exec_halo)
            if exec_halo < 0:
                DMPlexGetLabelValue(plex.dm, lbl_depth, p, &depth)
                DMPlexSetLabelValue(plex.dm, lbl_non_core, p, depth)

    DMPlexRestoreTransitiveClosure(plex.dm, cells[c], PETSC_TRUE,
                                   &nclosure, &closure)

    # Mark all remaining points as core
    pStart, pEnd = plex.getChart()
    for p in range(pStart, pEnd):
        DMPlexGetLabelValue(plex.dm, lbl_halo, p, &exec_halo)
        DMPlexGetLabelValue(plex.dm, lbl_non_core, p, &non_core)
        if exec_halo < 0 and non_core < 0:
            DMPlexGetLabelValue(plex.dm, lbl_depth, p, &depth)
            DMPlexSetLabelValue(plex.dm, lbl_core, p, depth)

    PetscFree(vertices)

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

@cython.boundscheck(False)
@cython.wraparound(False)
def plex_renumbering(PETSc.DM plex):
    """
    Build a global node renumbering as a permutation of Plex points.

    :arg plex: The DMPlex object encapsulating the mesh topology

    The node permutation is derived from a depth-first traversal of
    the Plex graph over each OP2 entity class in turn. The returned IS
    is the Plex -> OP2 permutation.
    """
    cdef:
        PetscInt dim, ncells, nclosure, c, ci, p, p_glbl, lbl_val
        PetscInt *cells = NULL
        PetscInt *closure = NULL
        PetscInt *perm = NULL
        PETSc.IS cell_is = None
        PETSc.IS perm_is = None
        char *lbl_chr = NULL
        PetscBT seen = NULL

    dim = plex.getDimension()
    pStart, pEnd = plex.getChart()
    PetscMalloc1(pEnd - pStart, &perm)
    PetscBTCreate(pEnd - pStart, &seen)
    p_glbl = 0

    for op2class in ["op2_core",
                     "op2_non_core",
                     "op2_exec_halo"]:
        lbl_chr = <char*>op2class
        ncells = plex.getStratumSize(op2class, dim)
        if ncells > 0:
            cell_is = plex.getStratumIS(op2class, dim)
            ISGetIndices(cell_is.iset, &cells)
            for c in range(ncells):
                DMPlexGetTransitiveClosure(plex.dm, cells[c],
                                           PETSC_TRUE,
                                           &nclosure,
                                           &closure)
                for ci in range(nclosure):
                    p = closure[2*ci]
                    if not PetscBTLookup(seen, p):
                        DMPlexGetLabelValue(plex.dm, lbl_chr,
                                            p, &lbl_val)
                        if lbl_val >= 0:
                            PetscBTSet(seen, p)
                            perm[p_glbl] = p
                            p_glbl += 1

            DMPlexRestoreTransitiveClosure(plex.dm, cells[c],
                                           PETSC_TRUE,
                                           &nclosure,
                                           &closure)
    PetscBTDestroy(&seen)

    perm_is = PETSc.IS().create()
    perm_is.setType("general")
    ISGeneralSetIndices(perm_is.iset, pEnd - pStart,
                        perm, PETSC_OWN_POINTER)
    return perm_is
