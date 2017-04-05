from __future__ import absolute_import, print_function, division

from firedrake.petsc import PETSc
import numpy
cimport numpy
import cython
cimport petsc4py.PETSc as PETSc
from pyop2.datatypes import IntType

numpy.import_array()

include "dmplexinc.pxi"


@cython.wraparound(False)
def layer_extents(mesh):
    """
    Compute the extents (start and stop layers) for an extruded mesh.

    :arg mesh: The extruded mesh.

    :returns: a numpy array of shape (npoints, 4) where npoints is the
        number of mesh points in the base mesh.  ``npoints[p, 0:2]``
        gives the start and stop layers for *allocation* for mesh
        point ``p`` (in plex ordering), while ``npoints[p, 2:4]``
        gives the start and stop layers for *iteration* over mesh
        point ``p`` (in plex ordering).

    .. warning::

       The indexing of this array uses DMPlex point ordering, *not*
       Firedrake ordering.  So you always need to iterate over plex
       points and translate to Firedrake numbers if necessary.
    """
    cdef:
        PETSc.DM dm
        PETSc.Section section
        numpy.ndarray[PetscInt, ndim=2, mode="c"] cell_extents
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        PetscInt cStart, cEnd, c, cell, ci, p
        PetscInt *closure = NULL
        PetscInt closureSize

    dm = mesh._plex
    section = mesh._cell_numbering
    cell_extents = mesh.cell_set.layers_array
    pStart, pEnd = dm.getChart()

    iinfo = numpy.iinfo(IntType)

    layer_extents = numpy.full((pEnd - pStart, 4),
                               (iinfo.max, iinfo.min, iinfo.min, iinfo.max),
                               dtype=IntType)

    cStart, cEnd = dm.getHeightStratum(0)
    for c in range(cStart, cEnd):
        CHKERR(DMPlexGetTransitiveClosure(dm.dm, c, PETSC_TRUE, &closureSize, &closure))
        CHKERR(PetscSectionGetOffset(section.sec, c, &cell))
        for ci in range(closureSize):
            p = closure[2*ci]
            # Allocation bounds
            # Each entity column is from bottom of lowest to top of highest
            layer_extents[p, 0] = min(layer_extents[p, 0], cell_extents[cell, 0])
            layer_extents[p, 1] = max(layer_extents[p, 1], cell_extents[cell, 1])
            # Iteration bounds
            # Each entity column is from top of lowest to bottom of highest
            layer_extents[p, 2] = max(layer_extents[p, 2], cell_extents[cell, 0])
            layer_extents[p, 3] = min(layer_extents[p, 3], cell_extents[cell, 1])
    CHKERR(DMPlexRestoreTransitiveClosure(dm.dm, 0, PETSC_TRUE, NULL, &closure))
    return layer_extents


@cython.wraparound(False)
def create_section(mesh, nodes_per_entity):
    """Create the section describing a global numbering.

    :arg mesh: The extruded mesh.
    :arg nodes_per_entity: Number of nodes on, and on top of, each
        type of topological entity on the base mesh for a single cell
        layer.  Multiplying up by the number of layers happens in this
        function.

    :returns: A PETSc Section providing the number of dofs, and offset
        of each dof, on each mesh point.
    """
    cdef:
        PETSc.DM dm
        PETSc.Section section
        PETSc.IS renumbering
        PetscInt i, p, layers, pStart, pEnd
        PetscInt dimension, ndof
        numpy.ndarray[PetscInt, ndim=2, mode="c"] nodes
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents = mesh.layer_extents

    dm = mesh._plex
    renumbering = mesh._plex_renumbering
    section = PETSc.Section().create(comm=mesh.comm)
    pStart, pEnd = dm.getChart()
    section.setChart(pStart, pEnd)
    CHKERR(PetscSectionSetPermutation(section.sec, renumbering.iset))

    nodes = numpy.asarray(nodes_per_entity, dtype=IntType)
    dimension = dm.getDimension()

    for i in range(dimension + 1):
        pStart, pEnd = dm.getDepthStratum(i)
        for p in range(pStart, pEnd):
            layers = layer_extents[p, 1] - layer_extents[p, 0]
            ndof = layers*nodes[i, 0] + (layers - 1)*nodes[i, 1]
            CHKERR(PetscSectionSetDof(section.sec, p, ndof))
    section.setUp()
    return section


@cython.wraparound(False)
def node_classes(mesh, nodes_per_entity):
    """Compute the node classes for a given extruded mesh.

    :arg mesh: the extruded mesh.
    :arg nodes_per_entity: Number of nodes on, and on top of, each
        type of topological entity on the base mesh for a single cell
        layer.  Multiplying up by the number of layers happens in this
        function.

    :returns: A numpy array of shape (4, ) giving the set entity sizes
        for the given nodes per entity.
    """
    cdef:
        PETSc.DM dm
        DMLabel label
        PetscInt p, point, layers, i, j, dimension
        numpy.ndarray[PetscInt, ndim=2, mode="c"] nodes
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents = mesh.layer_extents
        numpy.ndarray[PetscInt, ndim=2, mode="c"] stratum_bounds
        numpy.ndarray[PetscInt, ndim=1, mode="c"] node_classes
        numpy.ndarray[PetscInt, ndim=1, mode="c"] indices

    nodes = numpy.asarray(nodes_per_entity, dtype=IntType)

    node_classes = numpy.zeros(3, dtype=IntType)

    dm = mesh._plex
    dimension = dm.getDimension()
    stratum_bounds = numpy.zeros((dimension + 1, 2), dtype=IntType)
    for i in range(dimension + 1):
        stratum_bounds[i, :] = dm.getDepthStratum(i)

    for i, lbl in enumerate(["pyop2_core", "pyop2_owned", "pyop2_ghost"]):
        if dm.getStratumSize(lbl, 1) < 1:
            continue
        indices = dm.getStratumIS(lbl, 1).indices
        for p in range(indices.shape[0]):
            point = indices[p]
            layers = layer_extents[point, 1] - layer_extents[point, 0]
            for j in range(dimension + 1):
                if stratum_bounds[j, 0] <= point < stratum_bounds[j, 1]:
                    node_classes[i] += nodes[j, 0]*layers + nodes[j, 1]*(layers - 1)
                    break

    return numpy.cumsum(node_classes)


@cython.wraparound(False)
def entity_layers(mesh, height, label=None):
    """Compute the layers for a given entity type.

    :arg mesh: the extruded mesh to compute layers for.
    :arg height: the height of the entity to consider (in the DMPlex
       sense). e.g. 0 -> cells, 1 -> facets, etc...
    :arg label: optional label to select some subset of the points of
       the given height (may be None, meaning, select all points).

    :returns: a numpy array of shape (num_entities, 2) providing the
       layer extents for iteration on the requested entities.
    """
    cdef:
        PETSc.DM dm
        DMLabel clabel = NULL
        numpy.ndarray[PetscInt, ndim=1, mode="c"] facet_points
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layers
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        PetscInt f, p, i, fStart, fEnd
        PetscBool flg

    dm = mesh._base_mesh._plex

    if height == 0:
        if label is not None:
            raise ValueError("Not expecting non-None label")
        return mesh.cell_set.layers_array
    elif height == 1:
        fStart, fEnd = dm.getHeightStratum(height)
        if label is None:
            size = fEnd - fStart
        else:
            size = dm.getStratumSize(label, 1)

        layers = numpy.zeros((size, 2), dtype=IntType)
        facet_points = mesh._base_mesh._facet_ordering
        f = 0
        layer_extents = mesh.layer_extents

        if label is not None:
            CHKERR(DMGetLabel(dm.dm, <char *>label, &clabel))
            CHKERR(DMLabelCreateIndex(clabel, fStart, fEnd))

        for i in range(facet_points.shape[0]):
            p = facet_points[i]
            if clabel:
                CHKERR(DMLabelHasPoint(clabel, p, &flg))
            else:
                flg = PETSC_TRUE
            if flg:
                layers[f, 0] = layer_extents[p, 2]
                layers[f, 1] = layer_extents[p, 3]
                f += 1
        if label is not None:
            CHKERR(DMLabelDestroyIndex(clabel))
        return layers
    else:
        raise ValueError("Unsupported height '%s' (not 0 or 1)", height)


# More generic version (works on all entities)
def entity_layers2(mesh, height, label=None):
    cdef:
        PETSc.DM dm
        DMLabel clabel = NULL
        numpy.ndarray[PetscInt, ndim=1, mode="c"] facet_points
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layers
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        PetscInt f, p, i, pStart, pEnd
        PetscBool flg

    dm = mesh._base_mesh._plex

    pStart, pEnd = dm.getHeightStratum(height)
    if label is None:
        size = pEnd - pStart
    else:
        size = dm.getStratumSize(label, 1)

    layers = numpy.zeros((size, 2), dtype=IntType)

    layer_extents = mesh.layer_extents
    offset = 0
    if label is not None:
        CHKERR(DMGetLabel(dm.dm, <char *>label, &clabel))
        CHKERR(DMLabelCreateIndex(clabel, pStart, pEnd))
    for p in range(*dm.getChart()):
        plex_point = mesh._base_mesh._plex_renumbering.indices[p]
        if pStart <= plex_point < pEnd:
            if clabel:
                CHKERR(DMLabelHasPoint(clabel, plex_point, &flg))
            else:
                flg = PETSC_TRUE
            if flg:
                layers[offset, 0] = layer_extents[plex_point, 2]
                layers[offset, 1] = layer_extents[plex_point, 3]
                offset += 1

    if label is not None:
        CHKERR(DMLabelDestroyIndex(clabel))
    return layers
    

@cython.wraparound(False)
def get_cell_nodes(mesh,
                   PETSc.Section global_numbering,
                   entity_dofs,
                   numpy.ndarray[PetscInt, ndim=1, mode="c"] offset):
    """
    Builds the DoF mapping.

    :arg mesh: The mesh
    :arg global_numbering: Section describing the global DoF numbering
    :arg entity_dofs: FInAT element entity dofs for the cell
    :arg offset: offsets for each entity dof walking up a column.

    Preconditions: This function assumes that cell_closures contains mesh
    entities ordered by dimension, i.e. vertices first, then edges, faces, and
    finally the cell. For quadrilateral meshes, edges corresponding to
    dimension (0, 1) in the FInAT element must precede edges corresponding to
    dimension (1, 0) in the FInAT element.
    """
    cdef:
        int *ceil_ndofs = NULL
        int *flat_index = NULL
        PetscInt nclosure, dofs_per_cell
        PetscInt c, i, j, k, cStart, cEnd, cell
        PetscInt entity, ndofs, off
        PETSc.Section cell_numbering
        numpy.ndarray[PetscInt, ndim=2, mode="c"] cell_nodes
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        numpy.ndarray[PetscInt, ndim=2, mode="c"] cell_closures
        bint variable

    variable = not mesh.cell_set.constant_layers
    cell_closures = mesh.cell_closure
    if variable:
        layer_extents = mesh.layer_extents
        if offset is None:
            raise ValueError("Offset cannot be None with variable layer extents")
    
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

    # Fill cell nodes
    cStart, cEnd = mesh._plex.getHeightStratum(0)
    cell_nodes = numpy.empty((cEnd - cStart, dofs_per_cell), dtype=IntType)
    cell_numbering = mesh._cell_numbering
    for c in range(cStart, cEnd):
        k = 0
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        for i in range(nclosure):
            entity = cell_closures[cell, i]
            CHKERR(PetscSectionGetDof(global_numbering.sec, entity, &ndofs))
            if ndofs > 0:
                CHKERR(PetscSectionGetOffset(global_numbering.sec, entity, &off))
                # The cell we're looking at the entity through is
                # higher than the lowest cell the column touches, so
                # we need to offset by the difference from the bottom.
                if variable and layer_extents[entity, 0] != layer_extents[c, 0]:
                    off += offset[flat_index[k]]*(layer_extents[c, 0] - layer_extents[entity, 0])
                for j in range(ceil_ndofs[i]):
                    cell_nodes[cell, flat_index[k]] = off + j
                    k += 1

    CHKERR(PetscFree(ceil_ndofs))
    CHKERR(PetscFree(flat_index))
    return cell_nodes


def get_cell_nodes2(mesh,
                   global_numbering,
                   entity_dofs,
                   offset):
    """
    Builds the DoF mapping.

    :arg global_numbering: Section describing the global DoF numbering
    :arg cell_closures: 2D array of ordered cell closures
    :arg entity_dofs: FInAT element entity dofs for the cell

    Preconditions: This function assumes that cell_closures contains mesh
    entities ordered by dimension, i.e. vertices first, then edges, faces, and
    finally the cell. For quadrilateral meshes, edges corresponding to
    dimension (0, 1) in the FInAT element must precede edges corresponding to
    dimension (1, 0) in the FInAT element.
    """
    cell_closures = mesh.cell_closure
    dm = mesh._plex
    cell_numbering = mesh._cell_numbering
    layer_bounds = mesh.layer_extents
    ncells = cell_closures.shape[0]
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

    ceil_ndofs = ndofs_list
    flat_index = flat_index_list
    
    # Fill cell nodes
    cell_nodes = numpy.empty((ncells, dofs_per_cell), dtype=IntType)
    cStart, cEnd = dm.getHeightStratum(0)
    for cell in range(cStart, cEnd):
        k = 0
        c = cell_numbering.getOffset(cell)
        for i in range(nclosure):
            entity = cell_closures[c, i]
            ndofs = global_numbering.getDof(entity)
            if ndofs > 0:
                off = global_numbering.getOffset(entity)
                for j in range(ceil_ndofs[i]):
                    if layer_bounds[entity, 0] != layer_bounds[cell, 0]:
                        offs = off + offset[flat_index[k]]*(layer_bounds[entity, 2] - layer_bounds[entity, 0])
                    else:
                        offs = off
                    cell_nodes[c, flat_index[k]] = offs + j
                    k += 1

    return cell_nodes
