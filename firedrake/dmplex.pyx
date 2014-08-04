# Utility functions to derive global and local numbering from DMPlex
from petsc import PETSc
from pyop2 import MPI
import numpy as np
cimport numpy as np
import cython
cimport petsc4py.PETSc as PETSc

np.import_array()

include "dmplex.pxi"

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
                    np.ndarray[np.int32_t, ndim=1, mode="c"] facets,
                    PETSc.Section cell_numbering,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] cell_closures):
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
        np.ndarray[np.int32_t, ndim=2, mode="c"] facet_cells
        np.ndarray[np.int32_t, ndim=2, mode="c"] facet_local_num

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

@cython.boundscheck(False)
@cython.wraparound(False)
def closure_ordering(PETSc.DM plex,
                     PETSc.Section vertex_numbering,
                     PETSc.Section cell_numbering,
                     np.ndarray[np.int32_t, ndim=1, mode="c"] entity_per_cell):
    """Apply Fenics local numbering to a cell closure.

    :arg plex: The DMPlex object encapsulating the mesh topology
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
        PetscInt dim, vi, ci, fi, v_per_cell, cell
        PetscInt offset, cell_offset, nfaces, nfacets
        PetscInt nclosure, nfacet_closure, nface_vertices
        PetscInt *vertices = NULL
        PetscInt *v_global = NULL
        PetscInt *closure = NULL
        PetscInt *facets = NULL
        PetscInt *faces = NULL
        PetscInt *face_indices = NULL
        PetscInt *face_vertices = NULL
        PetscInt *facet_vertices = NULL
        np.ndarray[np.int32_t, ndim=2, mode="c"] cell_closure

    dim = plex.getDimension()
    cStart, cEnd = plex.getHeightStratum(0)
    fStart, fEnd = plex.getHeightStratum(1)
    eStart, eEnd = plex.getDepthStratum(1)
    vStart, vEnd = plex.getDepthStratum(0)
    v_per_cell = entity_per_cell[0]
    cell_offset = sum(entity_per_cell) - 1

    CHKERR(PetscMalloc1(v_per_cell, &vertices))
    CHKERR(PetscMalloc1(v_per_cell, &v_global))
    CHKERR(PetscMalloc1(v_per_cell-1, &facets))
    CHKERR(PetscMalloc1(v_per_cell-1, &facet_vertices))
    CHKERR(PetscMalloc1(entity_per_cell[1], &faces))
    CHKERR(PetscMalloc1(entity_per_cell[1], &face_indices))
    cell_closure = np.empty((cEnd - cStart, sum(entity_per_cell)), dtype=np.int32)

    for c in range(cStart, cEnd):
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        CHKERR(DMPlexGetTransitiveClosure(plex.dm, c, PETSC_TRUE,
                                          &nclosure,&closure))

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

        # Find all faces (dim=1)
        if dim > 2:
            nfaces = 0
            for ci in range(nclosure):
                if eStart <= closure[2*ci] < eEnd:
                    faces[nfaces] = closure[2*ci]

                    CHKERR(DMPlexGetConeSize(plex.dm, closure[2*ci],
                                             &nface_vertices))
                    CHKERR(DMPlexGetCone(plex.dm, closure[2*ci],
                                         &face_vertices))

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
            CHKERR(PetscSortIntWithArray(entity_per_cell[1],
                                         face_indices, faces))
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
                CHKERR(DMPlexGetTransitiveClosure(plex.dm, facets[f],
                                                  PETSC_TRUE,
                                                  &nfacet_closure,
                                                  &closure))
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
        CHKERR(DMPlexRestoreTransitiveClosure(plex.dm, 0, PETSC_TRUE,
                                              NULL, &closure))
    CHKERR(PetscFree(vertices))
    CHKERR(PetscFree(v_global))
    CHKERR(PetscFree(facets))
    CHKERR(PetscFree(facet_vertices))
    CHKERR(PetscFree(faces))
    CHKERR(PetscFree(face_indices))

    return cell_closure

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_nodes(PETSc.Section global_numbering,
                   np.ndarray[np.int32_t, ndim=2, mode="c"] cell_closures,
                   dofs_per_cell):
    """
    Builds the DoF mapping for non-extruded meshes.

    :arg global_numbering: Section describing the global DoF numbering
    :arg cell_closures: 2D array of ordered cell closures
    :arg dofs_per_cell: Number of DoFs associated with each mesh cell
    """
    cdef:
        PetscInt c, ncells, ci, nclosure, offset, p, pi, dof, off, i
        np.ndarray[np.int32_t, ndim=2, mode="c"] cell_nodes

    ncells = cell_closures.shape[0]
    nclosure = cell_closures.shape[1]
    cell_nodes = np.empty((ncells, dofs_per_cell), dtype=np.int32)

    for c in range(ncells):
        offset = 0
        for ci in range(nclosure):
            p = cell_closures[c, ci]
            CHKERR(PetscSectionGetDof(global_numbering.sec,
                                      p, &dof))
            CHKERR(PetscSectionGetOffset(global_numbering.sec,
                                         p, &off))
            for i in range(dof):
                cell_nodes[c, offset+i] = off+i
            offset += dof
    return cell_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def get_extruded_cell_nodes(PETSc.DM plex,
                            PETSc.Section global_numbering,
                            np.ndarray[np.int32_t, ndim=2, mode="c"] cell_closures,
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
        PetscInt offset, p, pdof, off, glbl, lcl, i, pi
        PetscInt *pStarts = NULL
        PetscInt *pEnds = NULL
        PetscInt *hdofs = NULL
        PetscInt *vdofs = NULL
        np.ndarray[np.int32_t, ndim=2, mode="c"] cell_nodes

    ncells = cell_closures.shape[0]
    nclosure = cell_closures.shape[1]
    cell_nodes = np.empty((ncells, dofs_per_cell), dtype=np.int32)

    dim = plex.getDimension()
    CHKERR(PetscMalloc1(dim+1, &pStarts))
    CHKERR(PetscMalloc1(dim+1, &pEnds))
    for d in range(dim+1):
        pStarts[d], pEnds[d] = plex.getDepthStratum(d)

    entity_dofs = fiat_element.entity_dofs()
    CHKERR(PetscMalloc1(dim+1, &hdofs))
    CHKERR(PetscMalloc1(dim+1, &vdofs))
    for d in range(dim+1):
        hdofs[d] = len(entity_dofs[(d,0)][0])
        vdofs[d] = len(entity_dofs[(d,1)][0])

    flattened_element = fiat_element.flattened_element()
    flat_entity_dofs = flattened_element.entity_dofs()

    cdef PetscInt ***fled_ptr = NULL
    cdef PetscInt *lcl_dofs = NULL
    CHKERR(PetscMalloc1(len(flat_entity_dofs), &fled_ptr))
    for i in range(len(flat_entity_dofs)):
        CHKERR(PetscMalloc1(len(flat_entity_dofs[i]), &(fled_ptr[i])))
        for j in range(len(flat_entity_dofs[i])):
            CHKERR(PetscMalloc1(len(flat_entity_dofs[i][j]), &(fled_ptr[i][j])))
            for k in range(len(flat_entity_dofs[i][j])):
                fled_ptr[i][j][k] = flat_entity_dofs[i][j][k]

    for c in range(ncells):
        offset = 0
        for d in range(dim+1):
            pi = 0
            for ci in range(nclosure):
                if pStarts[d] <= cell_closures[c, ci] < pEnds[d]:
                    p = cell_closures[c, ci]
                    CHKERR(PetscSectionGetDof(global_numbering.sec,
                                              p, &pdof))
                    if pdof > 0:
                        CHKERR(PetscSectionGetOffset(global_numbering.sec,
                                                     p, &glbl))

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

                        lcl_dofs = fled_ptr[d][pi]
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

    for i in range(len(flat_entity_dofs)):
        for j in range(len(flat_entity_dofs[i])):
            CHKERR(PetscFree(fled_ptr[i][j]))
        CHKERR(PetscFree(fled_ptr[i]))
    CHKERR(PetscFree(fled_ptr))
    CHKERR(PetscFree(pStarts))
    CHKERR(PetscFree(pEnds))
    CHKERR(PetscFree(hdofs))
    CHKERR(PetscFree(vdofs))
    return cell_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def get_facet_nodes(np.ndarray[np.int32_t, ndim=2, mode="c"] facet_cells,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] cell_nodes):
    """
    Derives the DoF mapping for a given facet list.

    :arg facet_cells: 2D array of parent cells for each facet
    :arg cell_nodes: 2D array of cell DoFs
    """
    cdef:
        int f, i, cell, nfacets, ncells, ndofs
        np.ndarray[np.int32_t, ndim=2, mode="c"] facet_nodes

    nfacets = facet_cells.shape[0]
    ncells = facet_cells.shape[1]
    ndofs = cell_nodes.shape[1]
    facet_nodes = np.empty((nfacets, ncells*ndofs), dtype=np.int32)

    for f in range(nfacets):
        # First parent cell
        cell = facet_cells[f, 0]
        for i in range(ndofs):
            facet_nodes[f, i] = cell_nodes[cell, i]

        # Second parent cell for internal facets
        if ncells > 1:
            cell = facet_cells[f, 1]
            if cell >= 0:
                for i in range(ndofs):
                    facet_nodes[f, ndofs+i] = cell_nodes[cell, i]
            else:
                for i in range(ndofs):
                    facet_nodes[f, ndofs+i] = -1

    return facet_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def label_facets(PETSc.DM plex, label_boundary=True):
    """Add labels to facets in the the plex

    Facets on the boundary are marked with "exterior_facets" while all
    others are marked with "interior_facets".

    :arg label_boundary: if False, don't label the boundary faces
         (they must have already been labelled)."""
    cdef:
        PetscInt fStart, fEnd, facet, val
        char *ext_label = <char *>"exterior_facets"
        char *int_label = <char *>"interior_facets"

    # Mark boundaries as exterior_facets
    if label_boundary:
        plex.markBoundaryFaces(ext_label)
    plex.createLabel(int_label)

    fStart, fEnd = plex.getHeightStratum(1)

    for facet in range(fStart, fEnd):
        CHKERR(DMPlexGetLabelValue(plex.dm, ext_label, facet, &val))
        # Not marked, must be interior
        if val == -1:
            CHKERR(DMPlexSetLabelValue(plex.dm, int_label, facet, 1))

@cython.boundscheck(False)
@cython.wraparound(False)
def filter_exterior_facet_labels(PETSc.DM plex):
    """Remove exterior facet labels from things that aren't facets.

    When refining, every point "underneath" the refined entity
    receives its label.  But we want the facet label to really only
    apply to facets, so clear the labels from everything else."""
    cdef:
        PetscInt pStart, pEnd, fStart, fEnd, p, ext_val
        PetscBool has_bdy_ids, has_bdy_faces

    pStart, pEnd = plex.getChart()
    fStart, fEnd = plex.getHeightStratum(1)

    # Plex will always have an exterior_facets label (maybe
    # zero-sized), but may not always have boundary_ids or
    # boundary_faces.
    has_bdy_ids = plex.hasLabel("boundary_ids")
    has_bdy_faces = plex.hasLabel("boundary_faces")

    for p in range(pStart, pEnd):
        if p < fStart or p >= fEnd:
            CHKERR(DMPlexGetLabelValue(plex.dm, <char *>"exterior_facets", p, &ext_val))
            if ext_val >= 0:
                CHKERR(DMPlexClearLabelValue(plex.dm, <char *>"exterior_facets", p, ext_val))
            if has_bdy_ids:
                CHKERR(DMPlexGetLabelValue(plex.dm, <char *>"boundary_ids", p, &ext_val))
                if ext_val >= 0:
                    CHKERR(DMPlexClearLabelValue(plex.dm, <char *>"boundary_ids", p, ext_val))
            if has_bdy_faces:
                CHKERR(DMPlexGetLabelValue(plex.dm, <char *>"boundary_faces", p, &ext_val))
                if ext_val >= 0:
                    CHKERR(DMPlexClearLabelValue(plex.dm, <char *>"boundary_faces", p, ext_val))


@cython.boundscheck(False)
@cython.wraparound(False)
def facet_numbering_section(mesh, typ):
    """Return a section providing a numbering for facets of specified type.

    :arg mesh: the mesh containing the facets.
    :arg typ: the type of facet (``interior`` or ``exterior``).

    The returned section can be used to build a halo object for the
    specified set of facets."""
    cdef:
        PETSc.DM plex = mesh._plex
        PETSc.Section section = PETSc.Section()
        PETSc.IS perm = mesh._plex_renumbering
        PetscInt f, fStart, fEnd, val, pStart, pEnd
        char *facet_label

    if typ == "interior":
        facet_label = "interior_facets"
    elif typ == "exterior":
        facet_label = "exterior_facets"
    else:
        raise RuntimeError("Unknown facet type '%s'" % typ)

    pStart, pEnd = plex.getChart()
    section.create()
    section.setChart(pStart, pEnd)
    CHKERR(PetscSectionSetPermutation(section.sec, perm.iset))
    fStart, fEnd = plex.getHeightStratum(1)
    for f in range(fStart, fEnd):
        CHKERR(DMPlexGetLabelValue(plex.dm, facet_label, f, &val))
        # We've got a facet of the correct type, set the number of dofs to 1
        if val >= 0:
            CHKERR(PetscSectionSetDof(section.sec, f, 1))

    section.setUp()
    return section

@cython.boundscheck(False)
@cython.wraparound(False)
def reordered_coords(PETSc.DM plex, PETSc.Section global_numbering, shape):
    """Return coordinates for the plex, reordered according to the
    global numbering permutation for the coordinate function space.

    Shape is a tuple of (plex.numVertices(), geometric_dim)."""
    cdef:
        PetscInt v, vStart, vEnd, offset
        PetscInt i, dim = shape[1]
        np.ndarray[np.float64_t, ndim=2, mode="c"] plex_coords, coords

    plex_coords = plex.getCoordinatesLocal().array.reshape(shape)
    coords = np.empty_like(plex_coords)
    vStart, vEnd = plex.getDepthStratum(0)

    for v in range(vStart, vEnd):
        CHKERR(PetscSectionGetOffset(global_numbering.sec, v, &offset))
        for i in range(dim):
            coords[offset, i] = plex_coords[v - vStart, i]

    return coords

@cython.boundscheck(False)
@cython.wraparound(False)
def mark_entity_classes(PETSc.DM plex):
    """Mark all points in a given Plex according to the PyOP2 entity
    classes:

    core      : owned and not in send halo
    non_core  : owned and in send halo
    exec_halo : in halo, but touch owned entity
    non_exec_halo : in halo and only touch halo entities

    :arg plex: The DMPlex object encapsulating the mesh topology
    """
    cdef:
        PetscInt p, pStart, pEnd, cStart, cEnd, vStart, vEnd
        PetscInt c, ncells, f, nfacets, ci, nclosure, vi, dim
        PetscInt depth, non_core, exec_halo, nroots, nleaves
        PetscInt v_per_cell
        PetscInt *cells = NULL
        PetscInt *facets = NULL
        PetscInt *vertices = NULL
        PetscInt *closure = NULL
        PetscInt *ilocal = NULL
        PetscBool non_exec
        PetscSFNode *iremote = NULL
        PETSc.SF point_sf = None
        PETSc.IS cell_is = None
        PETSc.IS facet_is = None

    dim = plex.getDimension()
    cStart, cEnd = plex.getHeightStratum(0)
    vStart, vEnd = plex.getDepthStratum(0)
    v_per_cell = plex.getConeSize(cStart)
    CHKERR(PetscMalloc1(v_per_cell, &vertices))

    plex.createLabel("op2_core")
    plex.createLabel("op2_non_core")
    plex.createLabel("op2_exec_halo")
    plex.createLabel("op2_non_exec_halo")

    lbl_depth = <char*>"depth"
    lbl_core = <char*>"op2_core"
    lbl_non_core = <char*>"op2_non_core"
    lbl_halo = <char*>"op2_exec_halo"
    lbl_non_exec_halo = <char*>"op2_non_exec_halo"

    if MPI.comm.size > 1:
        # Mark exec_halo from point overlap SF
        point_sf = plex.getPointSF()
        CHKERR(PetscSFGetGraph(point_sf.sf, &nroots, &nleaves,
                               &ilocal, &iremote))
        for p in range(nleaves):
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                       ilocal[p], &depth))
            CHKERR(DMPlexSetLabelValue(plex.dm, lbl_halo,
                                       ilocal[p], depth))
    else:
        # If sequential mark all points as core
        pStart, pEnd = plex.getChart()
        for p in range(pStart, pEnd):
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                       p, &depth))
            CHKERR(DMPlexSetLabelValue(plex.dm, lbl_core,
                                       p, depth))
        return

    # Mark all cells adjacent to halo cells as non_core,
    # where adjacent(c) := star(closure(c))
    ncells = plex.getStratumSize("op2_exec_halo", dim)
    cell_is = plex.getStratumIS("op2_exec_halo", dim)
    CHKERR(ISGetIndices(cell_is.iset, &cells))
    for c in range(ncells):
        CHKERR(DMPlexGetTransitiveClosure(plex.dm, cells[c],
                                          PETSC_TRUE,
                                          &nclosure,
                                          &closure))
        # Copy vertices out of the work array (closure)
        vi = 0
        for ci in range(nclosure):
            if vStart <= closure[2*ci] < vEnd:
                vertices[vi] = closure[2*ci]
                vi += 1

        # Mark all cells in the star of each vertex
        for vi in range(v_per_cell):
            vertex = vertices[vi]
            CHKERR(DMPlexGetTransitiveClosure(plex.dm, vertices[vi],
                                              PETSC_FALSE,
                                              &nclosure,
                                              &closure))
            for ci in range(nclosure):
                if cStart <= closure[2*ci] < cEnd:
                    p = closure[2*ci]
                    CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                               p, &exec_halo))
                    if exec_halo < 0:
                        CHKERR(DMPlexSetLabelValue(plex.dm,
                                                   lbl_non_core,
                                                   p, dim))

    # Mark the closures of non_core cells as non_core
    ncells = plex.getStratumSize("op2_non_core", dim)
    cell_is = plex.getStratumIS("op2_non_core", dim)
    CHKERR(ISGetIndices(cell_is.iset, &cells))
    for c in range(ncells):
        CHKERR(DMPlexGetTransitiveClosure(plex.dm, cells[c],
                                          PETSC_TRUE,
                                          &nclosure,
                                          &closure))
        for ci in range(nclosure):
            p = closure[2*ci]
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                       p, &exec_halo))
            if exec_halo < 0:
                CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                           p, &depth))
                CHKERR(DMPlexSetLabelValue(plex.dm, lbl_non_core,
                                           p, depth))

    # Mark all remaining points as core
    pStart, pEnd = plex.getChart()
    for p in range(pStart, pEnd):
        CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                   p, &exec_halo))
        CHKERR(DMPlexGetLabelValue(plex.dm, lbl_non_core,
                                   p, &non_core))
        if exec_halo < 0 and non_core < 0:
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                       p, &depth))
            CHKERR(DMPlexSetLabelValue(plex.dm, lbl_core,
                                       p, depth))

    # Halo facets that only touch halo vertices and halo cells need to
    # be marked as non-exec.
    nfacets = plex.getStratumSize("op2_exec_halo", dim-1)
    facet_is = plex.getStratumIS("op2_exec_halo", dim-1)
    CHKERR(ISGetIndices(facet_is.iset, &facets))
    for f in range(nfacets):
        non_exec = PETSC_TRUE
        # Check for halo vertices
        CHKERR(DMPlexGetTransitiveClosure(plex.dm, facets[f],
                                          PETSC_TRUE,
                                          &nclosure,
                                          &closure))
        for ci in range(nclosure):
            if vStart <= closure[2*ci] < vEnd:
                CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                           closure[2*ci], &exec_halo))
                if exec_halo < 0:
                    # Touches a non-halo vertex, needs to be executed
                    # over.
                    non_exec = PETSC_FALSE
        if non_exec:
            # If we still think we're non-exec, check for halo cells
            CHKERR(DMPlexGetTransitiveClosure(plex.dm, facets[f],
                                              PETSC_FALSE,
                                              &nclosure,
                                              &closure))
            for ci in range(nclosure):
                if cStart <= closure[2*ci] < cEnd:
                    CHKERR(DMPlexGetLabelValue(plex.dm, lbl_halo,
                                               closure[2*ci], &exec_halo))
                    if exec_halo < 0:
                        # Touches a non-halo cell, needs to be
                        # executed over.
                        non_exec = PETSC_FALSE
        if non_exec:
            CHKERR(DMPlexGetLabelValue(plex.dm, lbl_depth,
                                       facets[f], &depth))
            CHKERR(DMPlexSetLabelValue(plex.dm, lbl_non_exec_halo,
                                       facets[f], depth))
            # Remove facet from exec-halo label
            CHKERR(DMPlexClearLabelValue(plex.dm, lbl_halo,
                                         facets[f], depth))

    if closure != NULL:
        CHKERR(DMPlexRestoreTransitiveClosure(plex.dm, 0, PETSC_TRUE,
                                              NULL, &closure))
    PetscFree(vertices)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_classes(PETSc.DM plex):
    """Builds a PyOP2 entity class offsets for cells.

    :arg plex: The DMPlex object encapsulating the mesh topology
    """
    cdef:
        PetscInt dim, c, ci, nclass

    dim = plex.getDimension()
    cStart, cEnd = plex.getHeightStratum(0)
    cell_classes = [0, 0, 0, 0]
    c = 0

    for i, op2class in enumerate(["op2_core",
                                  "op2_non_core",
                                  "op2_exec_halo"]):
        c += plex.getStratumSize(op2class, dim)
        cell_classes[i] = c
    cell_classes[3] = cell_classes[2]
    return cell_classes

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
        np.ndarray[np.int32_t, ndim=1, mode="c"] facets

    label_chr = <char*>label
    dim = plex.getDimension()
    nfacets = plex.getStratumSize(label, 1)
    facets = np.empty(nfacets, dtype=np.int32)
    facet_classes = [0, 0, 0, 0]
    fi = 0

    for i, op2class in enumerate(["op2_core",
                                  "op2_non_core",
                                  "op2_exec_halo",
                                  "op2_non_exec_halo"]):
        nclass = plex.getStratumSize(op2class, dim-1)
        if nclass > 0:
            class_is = plex.getStratumIS(op2class, dim-1)
            CHKERR(ISGetIndices(class_is.iset, &indices))
            for ci in range(nclass):
                CHKERR(DMPlexGetLabelValue(plex.dm, label_chr,
                                           indices[ci], &lbl_val))
                if lbl_val == 1:
                    facets[fi] = indices[ci]
                    fi += 1
        facet_classes[i] = fi

    return facets, facet_classes

@cython.boundscheck(False)
@cython.wraparound(False)
def plex_renumbering(PETSc.DM plex,
                     np.ndarray[PetscInt, ndim=1, mode="c"] reordering=None):
    """
    Build a global node renumbering as a permutation of Plex points.

    :arg plex: The DMPlex object encapsulating the mesh topology
    :arg reordering: A reordering from reordered to original plex
         points used to provide the traversal order of the cells
         (i.e. the inverse of the ordering obtained from
         DMPlexGetOrdering).  Optional, if not provided (or ``None``),
         no reordering is applied and the plex is traversed in
         original order.

    The node permutation is derived from a depth-first traversal of
    the Plex graph over each OP2 entity class in turn. The returned IS
    is the Plex -> OP2 permutation.
    """
    cdef:
        PetscInt dim, ncells, nfacets, nclosure, c, ci, p, p_glbl, lbl_val
        PetscInt cStart, cEnd, core_idx, non_core_idx, exec_halo_idx
        PetscInt *core_cells = NULL
        PetscInt *non_core_cells = NULL
        PetscInt *exec_halo_cells = NULL
        PetscInt *cells = NULL
        PetscInt *facets = NULL
        PetscInt *closure = NULL
        PetscInt *perm = NULL
        PETSc.IS cell_is = None
        PETSc.IS facet_is = None
        PETSc.IS perm_is = None
        char *lbl_chr = NULL
        PetscBT seen = NULL
        bint reorder = reordering is not None

    dim = plex.getDimension()
    pStart, pEnd = plex.getChart()
    cStart, cEnd = plex.getHeightStratum(0)
    CHKERR(PetscMalloc1(pEnd - pStart, &perm))
    CHKERR(PetscBTCreate(pEnd - pStart, &seen))
    p_glbl = 0

    if reorder:
        core_idx = plex.getStratumSize("op2_core", dim)
        non_core_idx = plex.getStratumSize("op2_non_core", dim)
        exec_halo_idx = plex.getStratumSize("op2_exec_halo", dim)

        CHKERR(PetscMalloc1(core_idx, &core_cells))
        CHKERR(PetscMalloc1(non_core_idx, &non_core_cells))
        CHKERR(PetscMalloc1(exec_halo_idx, &exec_halo_cells))

        core_idx = 0
        non_core_idx = 0
        exec_halo_idx = 0

        # Walk over the reordering
        for ci in range(reordering.size):
            # Have we hit all the cells yet, if so break out early
            if core_idx + non_core_idx + exec_halo_idx > cEnd - cStart:
                break
            p = reordering[ci]

            CHKERR(DMPlexGetLabelValue(plex.dm, "depth",
                                       p, &lbl_val))
            # This point is a cell
            if lbl_val == dim:
                # Which entity class is this point in?
                CHKERR(DMPlexGetLabelValue(plex.dm, "op2_core",
                                           p, &lbl_val))
                if lbl_val == dim:
                    core_cells[core_idx] = p
                    core_idx += 1
                    continue

                CHKERR(DMPlexGetLabelValue(plex.dm, "op2_non_core",
                                           p, &lbl_val))
                if lbl_val == dim:
                    non_core_cells[non_core_idx] = p
                    non_core_idx += 1
                    continue

                CHKERR(DMPlexGetLabelValue(plex.dm, "op2_exec_halo",
                                           p, &lbl_val))

                if lbl_val == dim:
                    exec_halo_cells[exec_halo_idx] = p
                    exec_halo_idx += 1
                    continue

                raise RuntimeError("Should never be reached")

    # Now we can walk over the cell classes and order all the plex points
    for op2class in ["op2_core", "op2_non_core", "op2_exec_halo"]:
        lbl_chr = <char *>op2class
        if reorder:
            if op2class == "op2_core":
                ncells = core_idx
                cells = core_cells
            elif op2class == "op2_non_core":
                ncells = non_core_idx
                cells = non_core_cells
            elif op2class == "op2_exec_halo":
                ncells = exec_halo_idx
                cells = exec_halo_cells
        else:
            ncells = plex.getStratumSize(op2class, dim)
            if ncells > 0:
                cell_is = plex.getStratumIS(op2class, dim)
                CHKERR(ISGetIndices(cell_is.iset, &cells))
        for c in range(ncells):
            CHKERR(DMPlexGetTransitiveClosure(plex.dm, cells[c],
                                              PETSC_TRUE,
                                              &nclosure,
                                              &closure))
            for ci in range(nclosure):
                p = closure[2*ci]
                if not PetscBTLookup(seen, p):
                    CHKERR(DMPlexGetLabelValue(plex.dm, lbl_chr,
                                               p, &lbl_val))
                    if lbl_val >= 0:
                        CHKERR(PetscBTSet(seen, p))
                        perm[p_glbl] = p
                        p_glbl += 1

        if not reorder and ncells > 0:
            CHKERR(ISRestoreIndices(cell_is.iset, &cells))

    if closure != NULL:
        CHKERR(DMPlexRestoreTransitiveClosure(plex.dm, 0, PETSC_TRUE,
                                              NULL, &closure))

    # We currently mark non-exec facets without marking non-exec
    # cells, so they will not get picked up by the cell closure loops
    # and we need to add them explicitly.
    op2class = "op2_non_exec_halo"
    nfacets = plex.getStratumSize(op2class, dim-1)
    if nfacets > 0:
        facet_is = plex.getStratumIS(op2class, dim-1)
        CHKERR(ISGetIndices(facet_is.iset, &facets))
        for f in range(nfacets):
            p = facets[f]
            if not PetscBTLookup(seen, p):
                CHKERR(PetscBTSet(seen, p))
                perm[p_glbl] = p
                p_glbl += 1
    if reorder:
        CHKERR(PetscFree(core_cells))
        CHKERR(PetscFree(non_core_cells))
        CHKERR(PetscFree(exec_halo_cells))

    CHKERR(PetscBTDestroy(&seen))
    perm_is = PETSc.IS().create()
    perm_is.setType("general")
    CHKERR(ISGeneralSetIndices(perm_is.iset, pEnd - pStart,
                               perm, PETSC_OWN_POINTER))
    return perm_is

@cython.cdivision(True)
def compute_parent_cells(PETSc.DM plex):
    """Return a Section mapping cells in a plex to their "parents"

    :arg plex: the refined plex

    The determination of a parent cell in a regularly refined plex is
    simply by taking the cell number and dividing by the number of
    refined cells per coarse cell.  However, we will subsequently go
    on to resize the refined mesh, breaking this map.  Instead, we
    create a Section that, on each fine cell point, contains the
    offset into the original numbering of coarse cells.  This wil
    persist through resizing and will still be correct."""
    cdef:
        PetscInt cStart, cEnd, c, dim, val, nref
        PETSc.Section parents

    dim = plex.getDimension()
    # Space for one "dof" on each cell
    ents = np.zeros(dim+1, dtype=PETSc.IntType)
    ents[dim] = 1
    parents = plex.createSection([1], ents)
    # Number of refined cells per coarse cell (simplex only)
    nref = 2 ** dim

    cStart, cEnd = plex.getHeightStratum(0)
    for c in range(cStart, cEnd):
        val = c / nref
        CHKERR(PetscSectionSetDof(parents.sec, c, 1))
        CHKERR(PetscSectionSetOffset(parents.sec, c, val))

    return parents

@cython.boundscheck(False)
@cython.wraparound(False)
def coarse_to_fine_cells(mc, mf, PETSc.Section parents):
    """Return a map from (renumbered) cells in a coarse mesh to those
    in a refined fine mesh.

    :arg mc: the coarse mesh to create the map from.
    :arg mf: the fine mesh to map to.
    :arg parents: a Section mapping original fine cell numbers to
         their corresponding coarse parent cells"""
    cdef:
        PETSc.DM cdm, fdm
        PETSc.Section co2n, fo2n
        PetscInt fStart, fEnd, c, val, dim, nref, ncoarse
        PetscInt i, ccell, fcell, nfine
        np.ndarray[PetscInt, ndim=2, mode="c"] coarse_to_fine

    cdm = mc._plex
    co2n = mc._cell_numbering
    fdm = mf._plex
    fo2n = mf._cell_numbering
    dim = cdm.getDimension()
    nref = 2 ** dim
    ncoarse = mc.cell_set.size
    nfine = mf.cell_set.size
    fStart, fEnd = fdm.getHeightStratum(0)

    coarse_to_fine = np.empty((ncoarse, nref), dtype=PETSc.IntType)
    coarse_to_fine[:] = -1

    # Walk fine cells, pull the parent cell from the label.  Then,
    # since these are the original (not renumbered) cells, map them to
    # our view of the world and push the values in the map
    for c in range(fStart, fEnd):
        # Find the (originally numbered) parent of the current cell
        CHKERR(PetscSectionGetDof(parents.sec, c, &ccell))
        if ccell != 1:
            raise RuntimeError("Didn't find map from fine to coarse cell")
        CHKERR(PetscSectionGetOffset(parents.sec, c, &val))
        CHKERR(PetscSectionGetDof(co2n.sec, val, &ccell))
        CHKERR(PetscSectionGetDof(fo2n.sec, c, &fcell))
        if fcell <= 0:
            raise RuntimeError("Didn't find renumbered fine cell, should never happen")
        if ccell <= 0:
            raise RuntimeError("Didn't find renumbered coarse cell, should never happen")
        # Find the new numbering of the parent
        CHKERR(PetscSectionGetOffset(co2n.sec, val, &ccell))
        # Find the new numbering of the cell
        CHKERR(PetscSectionGetOffset(fo2n.sec, c, &fcell))
        # Only care about owned coarse cells
        if ccell >= ncoarse:
            # But halo coarse cells should not contain owned fine cells
            if fcell < nfine:
                raise RuntimeError("Found halo coarse cell containing owned fine cell, should never happen")
            continue
        # Find an empty slot
        for i in range(nref):
            if coarse_to_fine[ccell, i] == -1:
                coarse_to_fine[ccell, i] = fcell
                break
    return coarse_to_fine

@cython.boundscheck(False)
@cython.wraparound(False)
def get_entity_renumbering(PETSc.DM plex, PETSc.Section section, entity_type):
    cdef:
        PetscInt start, end, p, ndof, entity
        np.ndarray[PetscInt, ndim=1, mode="c"] old_to_new
        np.ndarray[PetscInt, ndim=1, mode="c"] new_to_old

    if entity_type == "cell":
        start, end = plex.getHeightStratum(0)
    elif entity_type == "vertex":
        start, end = plex.getDepthStratum(0)
    else:
        raise RuntimeError("Entity renumbering for entities of type %s not implemented" % entity_type)

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
def p1_coarse_fine_map(Vc, Vf, np.ndarray[PetscInt, ndim=2, mode="c"] c2f_cells):
    """Build a map from a coarse P1 function space Vc to the finer space Vf.
    The resulting map is isomorphic in numbering to a P2 map in the coarse space.

    :arg Vc: The coarse space
    :arg Vf: The fine space
    :arg c2f_cells: The map from coarse cells to fine cells
    """
    cdef:
        PetscInt c, vStart, vEnd, i, ncoarse_cell, ndof, coarse_arity
        PetscInt l, j, k, tmp, other_cell, vfStart
        PetscInt coarse_vertex, fine_vertex, orig_coarse_vertex, ncell
        PetscInt *orig_c2f
        PETSc.DM dm
        PETSc.PetscIS fpointIS
        bint done
        np.ndarray[np.int32_t, ndim=2, mode="c"] coarse_map, map_vals, fine_map
        np.ndarray[PetscInt, ndim=1, mode="c"] coarse_inv, fine_forward

    coarse_mesh = Vc.mesh()
    fine_mesh = Vf.mesh()

    try:
        ndof, ncell = {'interval': (3, 2),
                       'triangle': (6, 4),
                       'tetrahedron': (10, 8)}[coarse_mesh.ufl_cell().cellname()]
    except KeyError:
        raise RuntimeError("Don't know how to make map")

    ncoarse_cell = Vc.mesh().cell_set.size
    map_vals = np.empty((ncoarse_cell, ndof), dtype=np.int32)
    map_vals[:, :] = -1
    coarse_map = Vc.cell_node_map().values
    fine_map = Vf.cell_node_map().values

    if not hasattr(coarse_mesh, '_vertex_new_to_old'):
        o, n = get_entity_renumbering(coarse_mesh._plex, coarse_mesh._vertex_numbering, "vertex")
        coarse_mesh._vertex_old_to_new = o
        coarse_mesh._vertex_new_to_old = n

    coarse_inv = coarse_mesh._vertex_new_to_old

    if not hasattr(fine_mesh, '_vertex_new_to_old'):
        o, n = get_entity_renumbering(fine_mesh._plex, fine_mesh._vertex_numbering, "vertex")
        fine_mesh._vertex_old_to_new = o
        fine_mesh._vertex_new_to_old = n

    fine_forward = fine_mesh._vertex_old_to_new

    vStart, vEnd = coarse_mesh._plex.getDepthStratum(0)
    vfStart, _ = fine_mesh._plex.getDepthStratum(0)
    # vertex numbers in the fine plex of coarse vertices
    dm = coarse_mesh._plex
    CHKERR( DMPlexCreateCoarsePointIS(dm.dm, &fpointIS) )
    CHKERR( ISGetIndices(fpointIS, &orig_c2f) )

    coarse_arity = coarse_map.shape[1]
    for c in range(ncoarse_cell):
        # Find the fine vertices that correspond to coarse vertices
        # and put them sequentially in the first ndof map entries
        for i in range(coarse_arity):
            coarse_vertex = coarse_map[c, i]
            orig_coarse_vertex = coarse_inv[coarse_vertex]
            fine_vertex = fine_forward[orig_c2f[orig_coarse_vertex + vStart] - vfStart]
            map_vals[c, i] = fine_vertex

        if coarse_arity == 2:
            # intervals, only one missing fine vertex dof
            # x----o----x
            done = False
            for j in range(ncell):
                for k in range(coarse_arity):
                    fine_vertex = fine_map[c2f_cells[c, j], k]
                    if fine_vertex != map_vals[c, 0] and fine_vertex != map_vals[c, 1]:
                        map_vals[c, 2] = fine_vertex
                        done = True
                        break
                if done:
                    break
        elif coarse_arity == 3:
            # triangles, 3 missing fine vertex dofs
            #
            #          x
            #         / \
            #        /   \
            #       o-----o
            #      / \   / \
            #     /   \ /   \
            #    x-----o-----x
            #
            for j in range(ncell):
                for k in range(coarse_arity):
                    fine_vertex = fine_map[c2f_cells[c, j], k]
                    done = False
                    # Original vertex or already found?
                    for i in range(coarse_arity):
                        if fine_vertex == map_vals[c, i] or \
                           fine_vertex == map_vals[c, i + 3]:
                            done = True
                            break
                    if done:
                        continue

                    other_cell = -1
                    # Find the cell this fine vertex is /not/ part of
                    for i in range(ncell):
                        done = False
                        for l in range(coarse_arity):
                            tmp = fine_map[c2f_cells[c, i], l]
                            if tmp == fine_vertex:
                                done = True
                        # Done is true if the fine_vertex was in the cell
                        if not done:
                            other_cell = i
                            break

                    # Now find the coarse vertex of of the cell we
                    # weren't in and put this fine vertex in the
                    # "opposite" position in the map
                    done = False
                    for l in range(coarse_arity):
                        tmp = fine_map[c2f_cells[c, other_cell], l]
                        for i in range(coarse_arity):
                            if tmp == map_vals[c, i]:
                                done = True
                                map_vals[c, i + 3] = fine_vertex
                                break
                        if done:
                            break

    CHKERR( ISRestoreIndices(fpointIS, &orig_c2f) )
    CHKERR( ISDestroy(&fpointIS) )
    return map_vals
