"""
Computation dof numberings for extruded meshes
==============================================

On meshes with a constant number of cell layers (i.e. each column
contains the same number of cells), it is possible to compute all the
correct numberings by just lying to DMPlex about how many degrees of
freedom there are on the base topological entities.

This ceases to be true as soon as we permit variable numbers of cells
in each column, since now, although the number of degrees of freedom
on a cell does not change from column to column, the number that are
stacked up on each topological entity does change.

This module implements the necessary chicanery to deal with it.

Computation of topological layer extents
----------------------------------------

First, a picture.

Consider a one-dimensional mesh::

    x---0---x---1---x---2---x

Extruded to form the following two-dimensional mesh::


                         x--------x
                         |        |
                         |        |
   2                     |        |
                         |        |
       x--------x--------x--------x
       |        |        |
       |        |        |
   1   |        |        |
       |        |        |
       x--------x--------x
       |        |
       |        |
   0   |        |
       |        |
       x--------x

This is constructed by providing the number of cells in each column as
well as the starting cell layer::

     [[0, 2],
      [1, 1],
      [2, 1]]

We need to promote this cell layering to layering for all topological
entities.  Our solution to "interior" facets that only have one side
is to require that they are geometrically zero sized, and then
guarantee that we never iterate over them.  We therefore need to keep
track of two bits of information, the layer extent for allocation
purposes and also the layer extent for iteration purposes.

We compute both by iterating over the cells and transferring cell
layers to points in the closure of each cell.  Allocation bounds use
min-max on the cell bounds, iteration bounds use max-min.

Computation of function space allocation size
---------------------------------------------

With the layer extents computed, we need to compute the dof
allocation.  For this, we need the number of degrees of freedom *on*
the base topological entity, and *above* it in each cell::

       x-------x
       |   o   |
       o   o   o
       o   o   o
       |   o   |
       o---o---o

This element has one degree of freedom on each base vertex and cell,
two degrees of freedom "above" each vertex, and four above each cell.
To compute the number of degrees of freedom on the column of
topological entities we sum the number on the entity, multiplied by
the number of layers with the number above, multiplied by the number
of layers minus one (due to the fencepost error difference).
This number of layers naturally changes from entity to entity, and so
we can't compute this up front, but must do it point by point,
constructing the PETSc Section as we go.

Computation of function space maps
----------------------------------

Now we need the maps from topological entities (cells and facets) to
the function space nodes they can see.  The allocation offsets that
the numbering section gives us are wrong, because when we have a step
in the column height, the offset will be wrong if we're looking from
the higher cell.  Consider a vertex discretisation on the previous
mesh, with a numbering::

                      8--------10
                      |        |
                      |        |
                      |        |
                      |        |
    2--------5--------7--------9
    |        |        |
    |        |        |
    |        |        |
    |        |        |
    1--------4--------6
    |        |
    |        |
    |        |
    |        |
    0--------3

The cell node map we get by just looking at allocation offsets is::

   [[0, 1, 3, 4],
    [3, 4, 6, 7],
    [6, 7, 9, 10]]

note how the second and third cells have the wrong value for their
"left" vertices.  Instead, we need to shift the numbering we obtain
from the allocation offset by the number of entities we're skipping
over, to result in::

   [[0, 1, 3, 4],
    [4, 5, 6, 7],
    [7, 8, 9, 10]]

Now, when we iterate over cells, we ensure that we access the correct
dofs.  The same trick needs to be applied to facet maps too.
"""

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
                if variable:
                    off += offset[flat_index[k]]*(layer_extents[c, 0] - layer_extents[entity, 0])
                for j in range(ceil_ndofs[i]):
                    cell_nodes[cell, flat_index[k]] = off + j
                    k += 1

    CHKERR(PetscFree(ceil_ndofs))
    CHKERR(PetscFree(flat_index))
    return cell_nodes


@cython.wraparound(False)
def get_facet_nodes(mesh, numpy.ndarray[PetscInt, ndim=2, mode="c"] cell_nodes, label,
                    numpy.ndarray[PetscInt, ndim=1, mode="c"] offset):
    cdef:
        PETSc.DM dm
        PETSc.Section cell_numbering
        DMLabel clabel = NULL
        numpy.ndarray[PetscInt, ndim=2, mode="c"] facet_nodes
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        PetscInt f, p, i, j, pStart, pEnd, fStart, fEnd, point
        PetscInt supportSize, facet, cell, ndof, dof
        const PetscInt *renumbering
        const PetscInt *support
        PetscBool flg
        bint variable, add_offset

    dm = mesh._plex
    try:
        variable = not mesh.cell_set.constant_layers
    except AttributeError:
        variable = False

    if variable and offset is None:
        raise ValueError("Offset cannot be None with variable layer extents")

    fStart, fEnd = dm.getHeightStratum(1)

    ndof = cell_nodes.shape[1]
    if label is None:
        nfacet = fEnd - fStart
    else:
        nfacet = dm.getStratumSize(label, 1)
    if label == "interior_facets":
        shape = (nfacet, ndof*2)
    elif label == "exterior_facets":
        shape = (nfacet, ndof)
    else:
        raise ValueError("Unsupported facet label '%s'", label)
    facet_nodes = numpy.full(shape, -1, dtype=IntType)

    if label is not None:
        CHKERR(DMGetLabel(dm.dm, <char *>label, &clabel))
        CHKERR(DMLabelCreateIndex(clabel, fStart, fEnd))
    pStart, pEnd = dm.getChart()
    CHKERR(ISGetIndices((<PETSc.IS?>mesh._plex_renumbering).iset, &renumbering))
    cell_numbering = mesh._cell_numbering

    facet = 0

    if variable:
        layer_extents = mesh.layer_extents
    for p in range(pStart, pEnd):
        point = renumbering[p]
        if fStart <= point < fEnd:
            if clabel:
                DMLabelHasPoint(clabel, point, &flg)
            else:
                flg = PETSC_TRUE

            if flg:
                DMPlexGetSupportSize(dm.dm, point, &supportSize)
                DMPlexGetSupport(dm.dm, point, &support)
                for i in range(supportSize):
                    PetscSectionGetOffset(cell_numbering.sec, support[i], &cell)
                    for j in range(ndof):
                        dof = cell_nodes[cell, j]
                        if variable:
                            # This facet iterates from higher than the
                            # cell numbering of the cell, so we need
                            # to add on the difference.
                            dof += offset[j]*(layer_extents[point, 2] - layer_extents[support[i], 0])
                        facet_nodes[facet, ndof*i + j] = dof
                facet += 1
    if label is not None:
        CHKERR(DMLabelDestroyIndex(clabel))
    CHKERR(ISRestoreIndices((<PETSc.IS?>mesh._plex_renumbering).iset, &renumbering))
    return facet_nodes

