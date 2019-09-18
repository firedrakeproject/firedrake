# cython: language_level=3
"""Computation dof numberings for extruded meshes
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

To simplify some things, we require that the resulting mesh is not
topologically disconnected anywhere.  Offset cells must, at least,
share a vertex with some other cell.

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

Computation of boundary nodes
-----------------------------

For the top and bottom boundary nodes, we walk over the cells at,
respectively, the top and bottom of the column and pull out those
nodes whose entity height matches the appropriate cell height.  As an
example::

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

The bottom boundary nodes are::

   [0, 3, 4, 6, 7, 9]

whereas the top are::

   [2, 5, 7, 8, 10]

For these strange "interior" facets, we first walk over the cells,
picking up the dofs in the closure of the base (ceiling) of the cell,
then we walk over facets, picking up all the dofs in the closure of
facets that are exposed (there may be more than one of these in the
cell column).  We don't have to worry about any lower-dimensional
entities, because if a co-dim 2 or greater entity is exposed in a
column, then the co-dim 1 entity in its star is also exposed.

For the side boundary nodes, we can make a simplification: we know
that the facet heights are always the same as the cell column heights
(because there is only one cell in the support of the facet).  Hence
we just walk over the boundary facets of the base mesh, extract out
the nodes on that facet on the bottom cell and walk up the column.
This is guaranteed to pick up all the nodes in the closure of the
facet column.
"""

cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport MPI_Op_create, MPI_Op_free, MPI_User_function, MPI_OP_NULL
from mpi4py import MPI
from firedrake.petsc import PETSc
import numpy
cimport numpy
import cython
cimport petsc4py.PETSc as PETSc
from pyop2.datatypes import IntType
from pyop2 import op2
from tsfc.fiatinterface import as_fiat_cell

import firedrake.extrusion_utils as eutils
numpy.import_array()

include "dmplexinc.pxi"

cdef extern from "mpi-compat.h":
    pass


cdef inline void extents_reduce(void *in_, void *out, int *count, MPI.MPI_Datatype *datatype) nogil:
    cdef:
        PetscInt *xin = <PetscInt *>in_
        PetscInt *xout = <PetscInt *>out

    if xin[0] < xout[0]:
        xout[0] = xin[0]
    if xin[1] > xout[1]:
        xout[1] = xin[1]
    if xin[2] > xout[2]:
        xout[2] = xin[2]
    if xin[3] < xout[3]:
        xout[3] = xin[3]


@cython.wraparound(False)
def layer_extents(PETSc.DM dm, PETSc.Section cell_numbering,
                  numpy.ndarray[PetscInt, ndim=2, mode="c"] cell_extents):
    """
    Compute the extents (start and stop layers) for an extruded mesh.

    :arg dm: The DMPlex.
    :arg cell_numbering: The cell numbering (plex points to Firedrake points).
    :arg cell_extents: The cell layers.

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
        PETSc.SF sf
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        numpy.ndarray[PetscInt, ndim=2, mode="c"] tmp
        PetscInt cStart, cEnd, c, cell, ci, p
        PetscInt *closure = NULL
        PetscInt closureSize
        MPI.Datatype contig, typ
        MPI.MPI_Op EXTENTS_REDUCER = MPI_OP_NULL

    pStart, pEnd = dm.getChart()

    iinfo = numpy.iinfo(IntType)

    layer_extents = numpy.full((pEnd - pStart, 4),
                               (iinfo.max, iinfo.min, iinfo.min, iinfo.max),
                               dtype=IntType)
    cStart, cEnd = dm.getHeightStratum(0)
    for c in range(cStart, cEnd):
        CHKERR(DMPlexGetTransitiveClosure(dm.dm, c, PETSC_TRUE, &closureSize, &closure))
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
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
    if dm.comm.size == 1:
        return layer_extents

    # OK, so now we have partially correct extents.  Those points on
    # the boundary of domains are not right yet, because we may not
    # see the cell that touches an owned vertex (say).
    sf = dm.getPointSF()
    try:
        tdict = MPI.__TypeDict__
    except AttributeError:
        tdict = MPI._typedict
    typ = tdict[layer_extents.dtype.char]
    contig = typ.Create_contiguous(4)
    contig.Commit()
    iinfo = numpy.iinfo(layer_extents.dtype)

    tmp = numpy.copy(layer_extents)
    # To get owned points correct, we do a reduction over the SF.
    CHKERR(MPI_Op_create(<MPI_User_function *>extents_reduce, 4, &EXTENTS_REDUCER))
    CHKERR(PetscSFReduceBegin(sf.sf, contig.ob_mpi,
                              <const void*>layer_extents.data,
                              <void *>tmp.data,
                              EXTENTS_REDUCER))
    CHKERR(PetscSFReduceEnd(sf.sf, contig.ob_mpi,
                            <const void*>layer_extents.data,
                            <void *>tmp.data,
                            EXTENTS_REDUCER))
    CHKERR(MPI_Op_free(&EXTENTS_REDUCER))
    layer_extents[:] = tmp[:]
    # OK, now we have the correct extents for owned points, but
    # potentially incorrect extents for ghost points, so do a SF Bcast
    # over the point SF to get it right.
    CHKERR(PetscSFBcastBegin(sf.sf, contig.ob_mpi,
                             <const void*>tmp.data,
                             <void *>layer_extents.data))
    CHKERR(PetscSFBcastEnd(sf.sf, contig.ob_mpi,
                           <const void*>tmp.data,
                           <void *>layer_extents.data))
    contig.Free()
    return layer_extents


@cython.wraparound(False)
def node_classes(mesh, nodes_per_entity):
    """Compute the node classes for a given extruded mesh.

    :arg mesh: the extruded mesh.
    :arg nodes_per_entity: Number of nodes on, and on top of, each
        type of topological entity on the base mesh for a single cell
        layer.  Multiplying up by the number of layers happens in this
        function.

    :returns: A numpy array of shape (3, ) giving the set entity sizes
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
       the given height (may be None meaning select all points).

    :returns: a numpy array of shape (num_entities, 2) providing the
       layer extents for iteration on the requested entities.
    """
    cdef:
        PETSc.DM dm
        DMLabel clabel = NULL
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layers
        PetscInt f, p, i, hStart, hEnd, pStart, pEnd
        PetscInt point, offset
        const PetscInt *renumbering
        PetscBool flg

    dm = mesh._plex

    hStart, hEnd = dm.getHeightStratum(height)
    if label is None:
        size = hEnd - hStart
    else:
        size = dm.getStratumSize(label, 1)

    layers = numpy.zeros((size, 2), dtype=IntType)

    layer_extents = mesh.layer_extents
    offset = 0
    if label is not None:
        label = label.encode()
        CHKERR(DMGetLabel(dm.dm, <const char *>label, &clabel))
        CHKERR(DMLabelCreateIndex(clabel, hStart, hEnd))
    pStart, pEnd = dm.getChart()
    CHKERR(ISGetIndices((<PETSc.IS?>mesh._plex_renumbering).iset, &renumbering))
    for p in range(pStart, pEnd):
        point = renumbering[p]
        if hStart <= point < hEnd:
            if clabel:
                CHKERR(DMLabelHasPoint(clabel, point, &flg))
                if not flg:
                    continue
            layers[offset, 0] = layer_extents[point, 2]
            layers[offset, 1] = layer_extents[point, 3]
            offset += 1

    CHKERR(ISRestoreIndices((<PETSc.IS?>mesh._plex_renumbering).iset, &renumbering))
    if label is not None:
        CHKERR(DMLabelDestroyIndex(clabel))
    return layers


@cython.wraparound(False)
def top_bottom_boundary_nodes(mesh,
                              numpy.ndarray[PetscInt, ndim=2, mode="c"] cell_node_list,
                              mask,
                              numpy.ndarray[PetscInt, ndim=1, mode="c"] offsets,
                              kind):
    """Extract top or bottom boundary nodes from an extruded function space.

    :arg mesh: The extruded mesh.
    :arg cell_node_list: The map from cells to nodes.
    :arg masks: masks for dofs in the closure of the facets of the
        cell.  First the vertical facets, then the horizontal facets
        (bottom then top).
    :arg offsets: Offsets to apply walking up the column.
    :arg kind: Whether we should select the bottom, or the top, nodes.
    :returns: a numpy array of unique indices of nodes on the bottom
        or top of the mesh.
    """
    cdef:
        bint top
        numpy.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        numpy.ndarray[PetscInt, ndim=2, mode="c"] cell_closure
        PETSc.Section section
        numpy.ndarray[PetscInt, ndim=1, mode="c"] indices
        PetscInt ncell, nclosure, n_vert_facet, fstart
        PetscInt idx, cell, facet, d, i, c, dof, p
        PetscInt initial_offset, exposed_layers, layer
        PetscInt ndof, offset, top_facet, bottom_facet
        numpy.ndarray[PetscInt, ndim=1, mode="c"] masked_indices
        numpy.ndarray[PetscInt, ndim=1, mode="c"] facet_points

    if kind not in {"bottom", "top"}:
        raise ValueError("Don't know how to extract nodes with kind '%s'", kind)

    section, masked_indices, facet_points = mask
    top = kind == "top"

    layer_extents = mesh.layer_extents
    cell_closure = mesh.cell_closure
    ncell, nclosure = mesh.cell_closure.shape
    n_vert_facet = mesh._base_mesh.ufl_cell().num_facets()
    assert facet_points.shape[0] == n_vert_facet + 2

    bottom_facet = facet_points[n_vert_facet]
    top_facet = facet_points[n_vert_facet+1]
    fstart = nclosure - n_vert_facet - 1
    ndof = cell_node_list.shape[1]
    # All vertical facets should have same number of masked dofs
    for i in range(n_vert_facet):
        if section.getDof(facet_points[i]) != section.getDof(facet_points[0]):
            raise ValueError("All vertical facets should mask same number of dofs")

    dm = mesh._plex
    fStart, fEnd = dm.getHeightStratum(1)
    if top:
        num_indices = (section.getDof(top_facet) * ncell
                       + section.getDof(facet_points[0]) * numpy.sum(layer_extents[fStart:fEnd, 1]
                                                                     - layer_extents[fStart:fEnd, 3]))
    else:
        num_indices = (section.getDof(bottom_facet) * ncell
                       + section.getDof(facet_points[0]) * numpy.sum(layer_extents[fStart:fEnd, 2]
                                                                     - layer_extents[fStart:fEnd, 0]))
    indices = numpy.full(num_indices, -1, dtype=IntType)
    idx = 0
    for cell in range(ncell):
        # Walk over all the cells, extract the plex cell this cell
        # corresponds to.
        c = cell_closure[cell, nclosure - 1]
        # First pick up the dofs in the closure of the horizontal
        # facet at the top or bottom of the cell column.
        if top:
            CHKERR(PetscSectionGetDof(section.sec, top_facet, &ndof))
            CHKERR(PetscSectionGetOffset(section.sec, top_facet, &offset))
            initial_offset = layer_extents[c, 1] - layer_extents[c, 0] - 2
        else:
            CHKERR(PetscSectionGetDof(section.sec, bottom_facet, &ndof))
            CHKERR(PetscSectionGetOffset(section.sec, bottom_facet, &offset))
            initial_offset = 0
        assert initial_offset >= 0, "Not expecting negative number of layers"

        for p in range(ndof):
            d = masked_indices[offset + p]
            dof = cell_node_list[cell, d]
            indices[idx] = dof + offsets[d] * initial_offset
            idx += 1
        # Now pick up dofs from any exposed facets.
        for i in range(n_vert_facet):
            CHKERR(PetscSectionGetDof(section.sec, facet_points[i], &ndof))
            CHKERR(PetscSectionGetOffset(section.sec, facet_points[i], &offset))
            if ndof <= 0:
                continue
            facet = cell_closure[cell, fstart + i]
            if top:
                # Is the facet exposed when viewed through this cell?
                if layer_extents[c, 1] == layer_extents[facet, 3]:
                    continue
                # Count number of exposed layers.
                initial_offset = layer_extents[facet, 3] - layer_extents[c, 0] - 1
                exposed_layers = layer_extents[facet, 1] - layer_extents[facet, 3]
            else:
                # Is the facet exposed when viewed through this cell?
                if layer_extents[c, 0] == layer_extents[facet, 2]:
                    continue
                initial_offset = 0
                exposed_layers = layer_extents[facet, 2] - layer_extents[facet, 0]
            assert initial_offset >= 0, "Not expecting negative number of layers"
            assert exposed_layers >= 1, "Expecting at least one exposed layer"
            for p in range(ndof):
                d = masked_indices[offset + p]
                dof = cell_node_list[cell, d]
                for layer in range(exposed_layers):
                    indices[idx] = dof + offsets[d] * (initial_offset + layer)
                    idx += 1
    return numpy.unique(indices[:idx])
