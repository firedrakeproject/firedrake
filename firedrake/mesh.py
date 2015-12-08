from __future__ import absolute_import
import numpy as np
import os
import ufl
import weakref

from pyop2 import op2
from pyop2.logger import info_red
from pyop2.profiling import timed_function, timed_region, profile
from pyop2.utils import as_tuple

import coffee.base as ast

import firedrake.dmplex as dmplex
import firedrake.extrusion_utils as eutils
import firedrake.spatialindex as spatialindex
import firedrake.utils as utils
from firedrake.parameters import parameters
from firedrake.petsc import PETSc


__all__ = ['Mesh', 'ExtrudedMesh', 'SubDomainData']


_cells = {
    1: {2: "interval"},
    2: {3: "triangle", 4: "quadrilateral"},
    3: {4: "tetrahedron"}
}


class _Facets(object):
    """Wrapper class for facet interation information on a :class:`Mesh`

    .. warning::

       The unique_markers argument **must** be the same on all processes."""
    def __init__(self, mesh, classes, kind, facet_cell, local_facet_number, markers=None,
                 unique_markers=None):

        self.mesh = mesh

        classes = as_tuple(classes, int, 4)
        self.classes = classes

        self.kind = kind
        assert(kind in ["interior", "exterior"])
        if kind == "interior":
            self._rank = 2
        else:
            self._rank = 1

        self.facet_cell = facet_cell

        self.local_facet_number = local_facet_number

        # assert that markers is a proper subset of unique_markers
        if markers is not None:
            for marker in markers:
                assert (marker in unique_markers), \
                    "Every marker has to be contained in unique_markers"

        self.markers = markers
        self.unique_markers = [] if unique_markers is None else unique_markers
        self._subsets = {}

    @utils.cached_property
    def set(self):
        size = self.classes
        halo = None
        if isinstance(self.mesh, ExtrudedMeshTopology):
            if self.kind == "interior":
                base = self.mesh._base_mesh.interior_facets.set
            else:
                base = self.mesh._base_mesh.exterior_facets.set
            return op2.ExtrudedSet(base, layers=self.mesh.layers)
        return op2.Set(size, "%s_%s_facets" % (self.mesh.name, self.kind), halo=halo)

    @property
    def bottom_set(self):
        '''Returns the bottom row of cells.'''
        return self.mesh.cell_set

    @utils.cached_property
    def _null_subset(self):
        '''Empty subset for the case in which there are no facets with
        a given marker value. This is required because not all
        markers need be represented on all processors.'''

        return op2.Subset(self.set, [])

    def measure_set(self, integral_type, subdomain_id):
        '''Return the iteration set appropriate to measure. This will
        either be for all the interior or exterior (as appropriate)
        facets, or for a particular numbered subdomain.'''

        # ufl.Measure doesn't have enums for these any more :(
        if subdomain_id in ["everywhere", "otherwise"]:
            if integral_type == "exterior_facet_bottom":
                return [(op2.ON_BOTTOM, self.bottom_set)]
            elif integral_type == "exterior_facet_top":
                return [(op2.ON_TOP, self.bottom_set)]
            elif integral_type == "interior_facet_horiz":
                return self.bottom_set
            else:
                return self.set
        else:
            return self.subset(subdomain_id)

    def subset(self, markers):
        """Return the subset corresponding to a given marker value.

        :param markers: integer marker id or an iterable of marker ids"""
        if self.markers is None:
            return self._null_subset
        markers = as_tuple(markers, int)
        try:
            return self._subsets[markers]
        except KeyError:
            # check that the given markers are valid
            for marker in markers:
                if marker not in self.unique_markers:
                    raise LookupError(
                        '{0} is not a valid marker'.
                        format(marker))

            # build a list of indices corresponding to the subsets selected by
            # markers
            indices = np.concatenate([np.nonzero(self.markers == i)[0]
                                      for i in markers])
            self._subsets[markers] = op2.Subset(self.set, indices)
            return self._subsets[markers]

    @utils.cached_property
    def local_facet_dat(self):
        """Dat indicating which local facet of each adjacent
        cell corresponds to the current facet."""

        return op2.Dat(op2.DataSet(self.set, self._rank), self.local_facet_number,
                       np.uintc, "%s_%s_local_facet_number" % (self.mesh.name, self.kind))

    @utils.cached_property
    def facet_cell_map(self):
        """Map from facets to cells."""
        return op2.Map(self.set, self.bottom_set, self._rank, self.facet_cell,
                       "facet_to_cell_map")


def _from_gmsh(filename):
    """Read a Gmsh .msh file from `filename`"""

    # Create a read-only PETSc.Viewer
    gmsh_viewer = PETSc.Viewer().create()
    gmsh_viewer.setType("ascii")
    gmsh_viewer.setFileMode("r")
    gmsh_viewer.setFileName(filename)
    gmsh_plex = PETSc.DMPlex().createGmsh(gmsh_viewer)

    if gmsh_plex.hasLabel("Face Sets"):
        boundary_ids = gmsh_plex.getLabelIdIS("Face Sets").getIndices()
        gmsh_plex.createLabel("boundary_ids")
        for bid in boundary_ids:
            faces = gmsh_plex.getStratumIS("Face Sets", bid).getIndices()
            for f in faces:
                gmsh_plex.setLabelValue("boundary_ids", f, bid)

    return gmsh_plex


def _from_exodus(filename):
    """Read an Exodus .e or .exo file from `filename`"""
    plex = PETSc.DMPlex().createExodusFromFile(filename)

    boundary_ids = dmplex.getLabelIdIS("Face Sets").getIndices()
    plex.createLabel("boundary_ids")
    for bid in boundary_ids:
        faces = plex.getStratumIS("Face Sets", bid).getIndices()
        for f in faces:
            plex.setLabelValue("boundary_ids", f, bid)

    return plex


def _from_cgns(filename):
    """Read a CGNS .cgns file from `filename`"""
    plex = PETSc.DMPlex().createCGNSFromFile(filename)

    # TODO: Add boundary IDs
    return plex


def _from_triangle(filename, dim):
    """Read a set of triangle mesh files from `filename`"""
    basename, ext = os.path.splitext(filename)

    if op2.MPI.comm.rank == 0:
        try:
            facetfile = open(basename+".face")
            tdim = 3
        except:
            try:
                facetfile = open(basename+".edge")
                tdim = 2
            except:
                facetfile = None
                tdim = 1
        if dim is None:
            dim = tdim
        op2.MPI.comm.bcast(tdim, root=0)

        with open(basename+".node") as nodefile:
            header = np.fromfile(nodefile, dtype=np.int32, count=2, sep=' ')
            nodecount = header[0]
            nodedim = header[1]
            assert nodedim == dim
            coordinates = np.loadtxt(nodefile, usecols=range(1, dim+1), skiprows=1)
            assert nodecount == coordinates.shape[0]

        with open(basename+".ele") as elefile:
            header = np.fromfile(elefile, dtype=np.int32, count=2, sep=' ')
            elecount = header[0]
            eledim = header[1]
            eles = np.loadtxt(elefile, usecols=range(1, eledim+1), dtype=np.int32, skiprows=1)
            assert elecount == eles.shape[0]

        cells = map(lambda c: c-1, eles)
    else:
        tdim = op2.MPI.comm.bcast(None, root=0)
        cells = None
        coordinates = None
    plex = _from_cell_list(tdim, cells, coordinates, comm=op2.MPI.comm)

    # Apply boundary IDs
    if op2.MPI.comm.rank == 0:
        facets = None
        try:
            header = np.fromfile(facetfile, dtype=np.int32, count=2, sep=' ')
            edgecount = header[0]
            facets = np.loadtxt(facetfile, usecols=range(1, tdim+2), dtype=np.int32, skiprows=0)
            assert edgecount == facets.shape[0]
        finally:
            facetfile.close()

        if facets is not None:
            vStart, vEnd = plex.getDepthStratum(0)   # vertices
            for facet in facets:
                bid = facet[-1]
                vertices = map(lambda v: v + vStart - 1, facet[:-1])
                join = plex.getJoin(vertices)
                plex.setLabelValue("boundary_ids", join[0], bid)

    return plex


def _from_cell_list(dim, cells, coords, comm=None):
    """
    Create a DMPlex from a list of cells and coords.

    :arg dim: The topological dimension of the mesh
    :arg cells: The vertices of each cell
    :arg coords: The coordinates of each vertex
    :arg comm: An optional MPI communicator to build the plex on
         (defaults to ``COMM_WORLD``)
    """

    if comm is None:
        comm = op2.MPI.comm
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


class MeshTopology(object):
    """A representation of mesh topology."""

    def __init__(self, plex, name, reorder, distribute):
        """Half-initialise a mesh topology.

        :arg plex: :class:`DMPlex` representing the mesh topology
        :arg name: name of the mesh
        :arg reorder: whether to reorder the mesh (bool)
        :arg distribute: whether to distribute the mesh to parallel processes
        """
        utils._init()

        self._plex = plex
        self.name = name

        # A cache of function spaces that have been built on this mesh
        self._cache = {}

        # Mark exterior and interior facets
        # Note.  This must come before distribution, because otherwise
        # DMPlex will consider facets on the domain boundary to be
        # exterior, which is wrong.
        with timed_region("Mesh: label facets"):
            label_boundary = (op2.MPI.comm.size == 1) or distribute
            dmplex.label_facets(plex, label_boundary=label_boundary)

        # Distribute the dm to all ranks
        if op2.MPI.comm.size > 1 and distribute:
            # We distribute with overlap zero, in case we're going to
            # refine this mesh in parallel.  Later, when we actually use
            # it, we grow the halo.
            plex.distribute(overlap=0)

        dim = plex.getDimension()

        cStart, cEnd = plex.getHeightStratum(0)  # cells
        cell_nfacets = plex.getConeSize(cStart)

        self._grown_halos = False
        self._ufl_cell = ufl.Cell(_cells[dim][cell_nfacets])

        def callback(self):
            """Finish initialisation."""
            del self._callback
            if op2.MPI.comm.size > 1:
                self._plex.distributeOverlap(1)
            self._grown_halos = True

            if reorder:
                with timed_region("Mesh: reorder"):
                    old_to_new = self._plex.getOrdering(PETSc.Mat.OrderingType.RCM).indices
                    reordering = np.empty_like(old_to_new)
                    reordering[old_to_new] = np.arange(old_to_new.size, dtype=old_to_new.dtype)
            else:
                # No reordering
                reordering = None
            self._did_reordering = bool(reorder)

            # Mark OP2 entities and derive the resulting Plex renumbering
            with timed_region("Mesh: renumbering"):
                dmplex.mark_entity_classes(self._plex)
                self._entity_classes = dmplex.get_entity_classes(self._plex)
                self._plex_renumbering = dmplex.plex_renumbering(self._plex,
                                                                 self._entity_classes,
                                                                 reordering)

            with timed_region("Mesh: cell numbering"):
                # Derive a cell numbering from the Plex renumbering
                entity_dofs = np.zeros(dim+1, dtype=np.int32)
                entity_dofs[-1] = 1

                self._cell_numbering = self._plex.createSection([1], entity_dofs,
                                                                perm=self._plex_renumbering)
                entity_dofs[:] = 0
                entity_dofs[0] = 1
                self._vertex_numbering = self._plex.createSection([1], entity_dofs,
                                                                  perm=self._plex_renumbering)

        self._callback = callback

    def init(self):
        """Finish the initialisation of the mesh."""
        if hasattr(self, '_callback'):
            self._callback(self)

    @property
    def topology(self):
        """The underlying mesh topology object."""
        return self

    @property
    def topological(self):
        """Alias of topology.

        This is to ensure consistent naming for some multigrid codes."""
        return self

    @property
    def layers(self):
        return None

    def ufl_cell(self):
        """The UFL :class:`~ufl.cell.Cell` associated with the mesh."""
        return self._ufl_cell

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        plex = self._plex
        dim = plex.getDimension()

        # Cell numbering and global vertex numbering
        cell_numbering = self._cell_numbering
        vertex_numbering = self._vertex_numbering.createGlobalSection(plex.getPointSF())

        cell = self.ufl_cell()
        if cell.is_simplex():
            # Simplex mesh
            cStart, cEnd = plex.getHeightStratum(0)
            a_closure = plex.getTransitiveClosure(cStart)[0]

            entity_per_cell = np.zeros(dim + 1, dtype=np.int32)
            for dim in xrange(dim + 1):
                start, end = plex.getDepthStratum(dim)
                entity_per_cell[dim] = sum(map(lambda idx: start <= idx < end,
                                               a_closure))

            return dmplex.closure_ordering(plex, vertex_numbering,
                                           cell_numbering, entity_per_cell)

        elif cell.cellname() == "quadrilateral":
            # Quadrilateral mesh
            cell_ranks = dmplex.get_cell_remote_ranks(plex)

            facet_orientations = dmplex.quadrilateral_facet_orientations(
                plex, vertex_numbering, cell_ranks)

            cell_orientations = dmplex.orientations_facet2cell(
                plex, vertex_numbering, cell_ranks,
                facet_orientations, cell_numbering)

            dmplex.exchange_cell_orientations(plex,
                                              cell_numbering,
                                              cell_orientations)

            return dmplex.quadrilateral_closure_ordering(
                plex, vertex_numbering, cell_numbering, cell_orientations)

        else:
            raise NotImplementedError("Cell type '%s' not supported." % cell)

    @utils.cached_property
    def exterior_facets(self):
        if self._plex.getStratumSize("exterior_facets", 1) > 0:
            # Compute the facet_numbering

            # Order exterior facets by OP2 entity class
            exterior_facets, exterior_facet_classes = \
                dmplex.get_facets_by_class(self._plex, "exterior_facets")

            # Derive attached boundary IDs
            if self._plex.hasLabel("boundary_ids"):
                boundary_ids = np.zeros(exterior_facets.size, dtype=np.int32)
                for i, facet in enumerate(exterior_facets):
                    boundary_ids[i] = self._plex.getLabelValue("boundary_ids", facet)

                unique_ids = np.sort(self._plex.getLabelIdIS("boundary_ids").indices)
            else:
                boundary_ids = None
                unique_ids = None

            exterior_local_facet_number, exterior_facet_cell = \
                dmplex.facet_numbering(self._plex, "exterior",
                                       exterior_facets,
                                       self._cell_numbering,
                                       self.cell_closure)

            return _Facets(self, exterior_facet_classes, "exterior",
                           exterior_facet_cell, exterior_local_facet_number,
                           boundary_ids, unique_markers=unique_ids)
        else:
            if self._plex.hasLabel("boundary_ids"):
                unique_ids = np.sort(self._plex.getLabelIdIS("boundary_ids").indices)
            else:
                unique_ids = None
            return _Facets(self, 0, "exterior", None, None,
                           unique_markers=unique_ids)

    @utils.cached_property
    def interior_facets(self):
        if self._plex.getStratumSize("interior_facets", 1) > 0:
            # Compute the facet_numbering

            # Order interior facets by OP2 entity class
            interior_facets, interior_facet_classes = \
                dmplex.get_facets_by_class(self._plex, "interior_facets")

            interior_local_facet_number, interior_facet_cell = \
                dmplex.facet_numbering(self._plex, "interior",
                                       interior_facets,
                                       self._cell_numbering,
                                       self.cell_closure)

            return _Facets(self, interior_facet_classes, "interior",
                           interior_facet_cell, interior_local_facet_number)
        else:
            return _Facets(self, 0, "interior", None, None)

    def make_cell_node_list(self, global_numbering, entity_dofs):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg fiat_element: The FIAT element for the cell
        """
        return dmplex.get_cell_nodes(global_numbering,
                                     self.cell_closure,
                                     entity_dofs)

    def make_dofs_per_plex_entity(self, entity_dofs):
        """Returns the number of DoFs per plex entity for each stratum,
        i.e. [#dofs / plex vertices, #dofs / plex edges, ...].

        :arg entity_dofs: FIAT element entity DoFs
        """
        return [len(entity_dofs[d][0]) for d in sorted(entity_dofs)]

    def make_offset(self, entity_dofs, ndofs):
        """Returns None (only for extruded use)."""
        return None

    def _order_data_by_cell_index(self, column_list, cell_data):
        return cell_data[column_list]

    def cell_orientations(self):
        """Return the orientation of each cell in the mesh.

        Use :func:`init_cell_orientations` on the mesh *geometry* to initialise."""
        if not hasattr(self, '_cell_orientations'):
            raise RuntimeError("No cell orientations found, did you forget to call init_cell_orientations?")
        return self._cell_orientations

    def num_cells(self):
        cStart, cEnd = self._plex.getHeightStratum(0)
        return cEnd - cStart

    def num_facets(self):
        fStart, fEnd = self._plex.getHeightStratum(1)
        return fEnd - fStart

    def num_faces(self):
        fStart, fEnd = self._plex.getDepthStratum(2)
        return fEnd - fStart

    def num_edges(self):
        eStart, eEnd = self._plex.getDepthStratum(1)
        return eEnd - eStart

    def num_vertices(self):
        vStart, vEnd = self._plex.getDepthStratum(0)
        return vEnd - vStart

    def num_entities(self, d):
        eStart, eEnd = self._plex.getDepthStratum(d)
        return eEnd - eStart

    def size(self, d):
        return self.num_entities(d)

    def cell_dimension(self):
        """Returns the cell dimension."""
        return self.ufl_cell().topological_dimension()

    def facet_dimension(self):
        """Returns the facet dimension."""
        # Facets have co-dimension 1
        return self.ufl_cell().topological_dimension() - 1

    @utils.cached_property
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "%s_cells" % self.name)


class ExtrudedMeshTopology(MeshTopology):
    """Representation of an extruded mesh topology."""

    def __init__(self, mesh, layers):
        """Build an extruded mesh topology from an input mesh topology

        :arg mesh:           the unstructured base mesh topology
        :arg layers:         number of extruded cell layers in the "vertical"
                             direction.
        """
        # A cache of function spaces that have been built on this mesh
        self._cache = {}

        mesh.init()
        self._base_mesh = mesh
        if layers < 1:
            raise RuntimeError("Must have at least one layer of extruded cells (not %d)" % layers)
        # All internal logic works with layers of base mesh (not layers of cells)
        self._layers = layers + 1
        self._ufl_cell = ufl.OuterProductCell(mesh.ufl_cell(), ufl.interval)

        # TODO: These attributes are copied so that FunctionSpaceBase can
        # access them directly.  Eventually we would want a better refactoring
        # of responsibilities between mesh and function space.
        self._plex = mesh._plex
        self._plex_renumbering = mesh._plex_renumbering
        self._entity_classes = mesh._entity_classes

    @property
    def name(self):
        return self._base_mesh.name

    @property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        return self._base_mesh.cell_closure

    @utils.cached_property
    def exterior_facets(self):
        exterior_facets = self._base_mesh.exterior_facets
        return _Facets(self, exterior_facets.classes,
                       "exterior",
                       exterior_facets.facet_cell,
                       exterior_facets.local_facet_number,
                       exterior_facets.markers,
                       unique_markers=exterior_facets.unique_markers)

    @utils.cached_property
    def interior_facets(self):
        interior_facets = self._base_mesh.interior_facets
        return _Facets(self, interior_facets.classes,
                       "interior",
                       interior_facets.facet_cell,
                       interior_facets.local_facet_number)

    def make_cell_node_list(self, global_numbering, entity_dofs):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg fiat_element: The FIAT element for the cell
        """
        flat_entity_dofs = {}
        for b, v in entity_dofs:
            # v in [0, 1].  Only look at the ones, then grab the data from zeros.
            if v == 0:
                continue
            flat_entity_dofs[b] = {}
            for i in entity_dofs[(b, v)]:
                # This line is fairly magic.
                # It works because an interval has two points.
                # We pick up the DoFs from the bottom point,
                # then the DoFs from the interior of the interval,
                # then finally the DoFs from the top point.
                flat_entity_dofs[b][i] = \
                    entity_dofs[(b, 0)][2*i] + entity_dofs[(b, 1)][i] + entity_dofs[(b, 0)][2*i+1]

        return dmplex.get_cell_nodes(global_numbering,
                                     self.cell_closure,
                                     flat_entity_dofs)

    def make_dofs_per_plex_entity(self, entity_dofs):
        """Returns the number of DoFs per plex entity for each stratum,
        i.e. [#dofs / plex vertices, #dofs / plex edges, ...].

        :arg entity_dofs: FIAT element entity DoFs
        """
        dofs_per_entity = [0] * (1 + self._base_mesh.cell_dimension())
        for (b, v), entities in entity_dofs.iteritems():
            dofs_per_entity[b] += (self.layers - v) * len(entities[0])
        return dofs_per_entity

    def make_offset(self, entity_dofs, ndofs):
        """Returns the offset between the neighbouring cells of a
        column for each DoF.

        :arg entity_dofs: FIAT element entity DoFs
        :arg ndofs: number of DoFs in the FIAT element
        """
        entity_offset = [0] * (1 + self._base_mesh.cell_dimension())
        for (b, v), entities in entity_dofs.iteritems():
            entity_offset[b] += len(entities[0])

        dof_offset = np.zeros(ndofs, dtype=np.int32)
        for (b, v), entities in entity_dofs.iteritems():
            for dof_indices in entities.itervalues():
                for i in dof_indices:
                    dof_offset[i] = entity_offset[b]
        return dof_offset

    @property
    def layers(self):
        """Return the number of layers of the extruded mesh
        represented by the number of occurences of the base mesh."""
        return self._layers

    def cell_dimension(self):
        """Returns the cell dimension."""
        return (self._base_mesh.cell_dimension(), 1)

    def facet_dimension(self):
        """Returns the facet dimension.

        .. note::

            This only returns the dimension of the "side" (vertical) facets,
            not the "top" or "bottom" (horizontal) facets.

        """
        return (self._base_mesh.facet_dimension(), 1)

    @utils.cached_property
    def cell_set(self):
        return op2.ExtrudedSet(self._base_mesh.cell_set, layers=self.layers)

    def _order_data_by_cell_index(self, column_list, cell_data):
        cell_list = []
        for col in column_list:
            cell_list += range(col, col + (self.layers - 1))
        return cell_data[cell_list]


class MeshGeometry(ufl.Mesh):
    """A representation of mesh topology and geometry."""

    def __new__(cls, element):
        """Create mesh geometry object."""
        utils._init()
        mesh = super(MeshGeometry, cls).__new__(cls)
        mesh.uid = utils._new_uid()
        assert isinstance(element, ufl.FiniteElementBase)
        ufl.Mesh.__init__(mesh, element, ufl_id=mesh.uid)
        return mesh

    def __init__(self, coordinates):
        """Initialise a mesh geometry from coordinates.

        :arg coordinates: a coordinateless function containing the coordinates
        """
        # Direct link to topology
        self._topology = coordinates.function_space().mesh()

        # Cache mesh object on the coordinateless coordinates function
        coordinates._as_mesh_geometry = weakref.ref(self)

        self._coordinates = coordinates

    def init(self):
        """Finish the initialisation of the mesh.  Most of the time
        this is carried out automatically, however, in some cases (for
        example accessing a property of the mesh directly after
        constructing it) you need to call this manually."""
        if hasattr(self, '_callback'):
            self._callback(self)

    @property
    def topology(self):
        """The underlying mesh topology object."""
        return self._topology

    @property
    def topological(self):
        """Alias of topology.

        This is to ensure consistent naming for some multigrid codes."""
        return self._topology

    @utils.cached_property
    def _coordinates_function(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        import firedrake.functionspace as functionspace
        import firedrake.function as function
        self.init()

        coordinates_fs = self._coordinates.function_space()
        V = functionspace.WithGeometry(coordinates_fs, self)
        f = function.Function(V, val=self._coordinates)
        return f

    @property
    def coordinates(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        return self._coordinates_function

    @coordinates.setter
    def coordinates(self, value):
        message = """Cannot re-assign the coordinates.

You are free to change the coordinate values, but if you need a
different coordinate function space, use Mesh(f) to create a new mesh
object whose coordinates are f's values.  (This will not copy the
values from f.)"""

        raise AttributeError(message)

    @utils.cached_property
    def spatial_index(self):
        """Spatial index to quickly find which cell contains a given point."""

        from firedrake import function, functionspace
        from firedrake.parloops import par_loop, READ, RW

        gdim = self.ufl_cell().geometric_dimension()
        if gdim <= 1:
            info_red("libspatialindex does not support 1-dimension, falling back on brute force.")
            return None

        # Calculate the bounding boxes for all cells by running a kernel
        V = functionspace.VectorFunctionSpace(self, "DG", 0, dim=gdim)
        coords_min = function.Function(V)
        coords_max = function.Function(V)

        coords_min.dat.data.fill(np.inf)
        coords_max.dat.data.fill(-np.inf)

        kernel = """
    for (int d = 0; d < gdim; d++) {
        for (int i = 0; i < nodes_per_cell; i++) {
            f_min[0][d] = fmin(f_min[0][d], f[i][d]);
            f_max[0][d] = fmax(f_max[0][d], f[i][d]);
        }
    }
"""

        cell_node_list = self.coordinates.function_space().cell_node_list
        nodes_per_cell = len(cell_node_list[0])

        kernel = kernel.replace("gdim", str(gdim))
        kernel = kernel.replace("nodes_per_cell", str(nodes_per_cell))

        par_loop(kernel, ufl.dx, {'f': (self.coordinates, READ),
                                  'f_min': (coords_min, RW),
                                  'f_max': (coords_max, RW)})

        # Reorder bounding boxes according to the cell indices we use
        column_list = V.cell_node_list.reshape(-1)
        coords_min = self._order_data_by_cell_index(column_list, coords_min.dat.data_ro_with_halos)
        coords_max = self._order_data_by_cell_index(column_list, coords_max.dat.data_ro_with_halos)

        # Build spatial index
        return spatialindex.from_regions(coords_min, coords_max)

    def init_cell_orientations(self, expr):
        """Compute and initialise :attr:`cell_orientations` relative to a specified orientation.

        :arg expr: an :class:`.Expression` evaluated to produce a
             reference normal direction.

        """
        import firedrake.function as function
        import firedrake.functionspace as functionspace

        if expr.value_shape()[0] != 3:
            raise NotImplementedError('Only implemented for 3-vectors')
        if self.ufl_cell() not in (ufl.Cell('triangle', 3), ufl.Cell("quadrilateral", 3), ufl.OuterProductCell(ufl.Cell('interval'), ufl.Cell('interval'), gdim=3)):
            raise NotImplementedError('Only implemented for triangles and quadrilaterals embedded in 3d')

        if hasattr(self.topology, '_cell_orientations'):
            raise RuntimeError("init_cell_orientations already called, did you mean to do so again?")

        v0 = lambda x: ast.Symbol("v0", (x,))
        v1 = lambda x: ast.Symbol("v1", (x,))
        n = lambda x: ast.Symbol("n", (x,))
        x = lambda x: ast.Symbol("x", (x,))
        coords = lambda x, y: ast.Symbol("coords", (x, y))

        body = []
        body += [ast.Decl("double", v(3)) for v in [v0, v1, n, x]]
        body.append(ast.Decl("double", "dot"))
        body.append(ast.Assign("dot", 0.0))
        body.append(ast.Decl("int", "i"))

        # if triangle, use v0 = x1 - x0, v1 = x2 - x0
        # otherwise, for the various quads, use v0 = x2 - x0, v1 = x1 - x0
        # recall reference element ordering:
        # triangle: 2        quad: 1 3
        #           0 1            0 2
        if self.ufl_cell() == ufl.Cell('triangle', 3):
            body.append(ast.For(ast.Assign("i", 0), ast.Less("i", 3), ast.Incr("i", 1),
                                [ast.Assign(v0("i"), ast.Sub(coords(1, "i"), coords(0, "i"))),
                                 ast.Assign(v1("i"), ast.Sub(coords(2, "i"), coords(0, "i"))),
                                 ast.Assign(x("i"), 0.0)]))
        else:
            body.append(ast.For(ast.Assign("i", 0), ast.Less("i", 3), ast.Incr("i", 1),
                                [ast.Assign(v0("i"), ast.Sub(coords(2, "i"), coords(0, "i"))),
                                 ast.Assign(v1("i"), ast.Sub(coords(1, "i"), coords(0, "i"))),
                                 ast.Assign(x("i"), 0.0)]))

        # n = v0 x v1
        body.append(ast.Assign(n(0), ast.Sub(ast.Prod(v0(1), v1(2)), ast.Prod(v0(2), v1(1)))))
        body.append(ast.Assign(n(1), ast.Sub(ast.Prod(v0(2), v1(0)), ast.Prod(v0(0), v1(2)))))
        body.append(ast.Assign(n(2), ast.Sub(ast.Prod(v0(0), v1(1)), ast.Prod(v0(1), v1(0)))))

        body.append(ast.For(ast.Assign("i", 0), ast.Less("i", 3), ast.Incr("i", 1),
                            [ast.Incr(x(j), coords("i", j)) for j in range(3)]))

        body.extend([ast.FlatBlock("dot += (%(x)s) * n[%(i)d];\n" % {"x": x_, "i": i})
                     for i, x_ in enumerate(expr.code)])
        body.append(ast.Assign("orientation[0][0]", ast.Ternary(ast.Less("dot", 0), 1, 0)))

        kernel = op2.Kernel(ast.FunDecl("void", "cell_orientations",
                                        [ast.Decl("int**", "orientation"),
                                         ast.Decl("double**", "coords")],
                                        ast.Block(body)),
                            "cell_orientations")

        # Build the cell orientations as a DG0 field (so that we can
        # pass it in for facet integrals and the like)
        fs = functionspace.FunctionSpace(self, 'DG', 0)
        cell_orientations = function.Function(fs, name="cell_orientations", dtype=np.int32)
        op2.par_loop(kernel, self.cell_set,
                     cell_orientations.dat(op2.WRITE, cell_orientations.cell_node_map()),
                     self.coordinates.dat(op2.READ, self.coordinates.cell_node_map()))
        self.topology._cell_orientations = cell_orientations

    def __getattr__(self, name):
        return getattr(self._topology, name)


def make_mesh_from_coordinates(coordinates):
    """Given a coordinate field build a new mesh, using said coordinate field.

    :arg coordinates: A :class:`~.Function`.
    """
    import firedrake.functionspace as functionspace
    from firedrake.ufl_expr import reconstruct_element

    if hasattr(coordinates, '_as_mesh_geometry'):
        mesh = coordinates._as_mesh_geometry()
        if mesh is not None:
            return mesh

    coordinates_fs = coordinates.function_space()
    if not isinstance(coordinates_fs, functionspace.VectorFunctionSpace):
        raise ValueError("Coordinates must have a VectorFunctionSpace.")
    assert coordinates_fs.mesh().ufl_cell().topological_dimension() <= coordinates_fs.dim
    # Build coordinate element
    element = coordinates.ufl_element()
    cell = element.cell().reconstruct(geometric_dimension=coordinates_fs.dim)
    element = reconstruct_element(element, cell=cell)

    mesh = MeshGeometry.__new__(MeshGeometry, element)
    mesh.__init__(coordinates)
    return mesh


@timed_function("Build mesh")
@profile
def Mesh(meshfile, **kwargs):
    """Construct a mesh object.

    Meshes may either be created by reading from a mesh file, or by
    providing a PETSc DMPlex object defining the mesh topology.

    :param meshfile: Mesh file name (or DMPlex object) defining
           mesh topology.  See below for details on supported mesh
           formats.
    :param dim: optional specification of the geometric dimension
           of the mesh (ignored if not reading from mesh file).
           If not supplied the geometric dimension is deduced from
           the topological dimension of entities in the mesh.
    :param reorder: optional flag indicating whether to reorder
           meshes for better cache locality.  If not supplied the
           default value in :data:`parameters["reorder_meshes"]`
           is used.

    When the mesh is read from a file the following mesh formats
    are supported (determined, case insensitively, from the
    filename extension):

    * GMSH: with extension `.msh`
    * Exodus: with extension `.e`, `.exo`
    * CGNS: with extension `.cgns`
    * Triangle: with extension `.node`

    .. note::

        When the mesh is created directly from a DMPlex object,
        the :data:`dim` parameter is ignored (the DMPlex already
        knows its geometric and topological dimensions).

    """
    import firedrake.functionspace as functionspace
    import firedrake.function as function

    if isinstance(meshfile, function.Function):
        coordinates = meshfile.topological
    elif isinstance(meshfile, function.CoordinatelessFunction):
        coordinates = meshfile
    else:
        coordinates = None

    if coordinates is not None:
        return make_mesh_from_coordinates(coordinates)

    utils._init()

    geometric_dim = kwargs.get("dim", None)
    reorder = kwargs.get("reorder", None)
    if reorder is None:
        reorder = parameters["reorder_meshes"]
    distribute = kwargs.get("distribute", True)

    if isinstance(meshfile, PETSc.DMPlex):
        name = "plexmesh"
        plex = meshfile
    else:
        name = meshfile
        basename, ext = os.path.splitext(meshfile)

        if ext.lower() in ['.e', '.exo']:
            plex = _from_exodus(meshfile)
        elif ext.lower() == '.cgns':
            plex = _from_cgns(meshfile)
        elif ext.lower() == '.msh':
            plex = _from_gmsh(meshfile)
        elif ext.lower() == '.node':
            plex = _from_triangle(meshfile, geometric_dim)
        else:
            raise RuntimeError("Mesh file %s has unknown format '%s'."
                               % (meshfile, ext[1:]))

    # Create mesh topology
    topology = MeshTopology(plex, name=name, reorder=reorder, distribute=distribute)

    tcell = topology.ufl_cell()
    if geometric_dim is None:
        geometric_dim = tcell.topological_dimension()
    cell = tcell.reconstruct(geometric_dimension=geometric_dim)

    element = ufl.VectorElement("Lagrange", cell, 1)
    # Create mesh object
    mesh = MeshGeometry.__new__(MeshGeometry, element)
    mesh._topology = topology

    def callback(self):
        """Finish initialisation."""
        del self._callback
        # Finish the initialisation of mesh topology
        self.topology.init()

        with timed_region("Mesh: coordinate field"):
            coordinates_fs = functionspace.VectorFunctionSpace(self.topology, "Lagrange", 1,
                                                               dim=geometric_dim)

            coordinates_data = dmplex.reordered_coords(plex, coordinates_fs._global_numbering,
                                                       (self.num_vertices(), geometric_dim))

            coordinates = function.CoordinatelessFunction(coordinates_fs,
                                                          val=coordinates_data,
                                                          name="Coordinates")

        self.__init__(coordinates)

    mesh._callback = callback
    return mesh


@timed_function("Build extruded mesh")
@profile
def ExtrudedMesh(mesh, layers, layer_height=None, extrusion_type='uniform', kernel=None, gdim=None):
    """Build an extruded mesh from an input mesh

    :arg mesh:           the unstructured base mesh
    :arg layers:         number of extruded cell layers in the "vertical"
                         direction.
    :arg layer_height:   the layer height, assuming all layers are evenly
                         spaced. If this is omitted, the value defaults to
                         1/layers (i.e. the extruded mesh has total height 1.0)
                         unless a custom kernel is used.
    :arg extrusion_type: the algorithm to employ to calculate the extruded
                         coordinates. One of "uniform", "radial",
                         "radial_hedgehog" or "custom". See below.
    :arg kernel:         a :class:`pyop2.Kernel` to produce coordinates for
                         the extruded mesh. See :func:`~.make_extruded_coords`
                         for more details.
    :arg gdim:           number of spatial dimensions of the
                         resulting mesh (this is only used if a
                         custom kernel is provided)

    The various values of ``extrusion_type`` have the following meanings:

    ``"uniform"``
        the extruded mesh has an extra spatial
        dimension compared to the base mesh. The layers exist
        in this dimension only.

    ``"radial"``
        the extruded mesh has the same number of
        spatial dimensions as the base mesh; the cells are
        radially extruded outwards from the origin. This
        requires the base mesh to have topological dimension
        strictly smaller than geometric dimension.
    ``"radial_hedgehog"``
        similar to `radial`, but the cells
        are extruded in the direction of the outward-pointing
        cell normal (this produces a P1dgxP1 coordinate field).
        In this case, a radially extruded coordinate field
        (generated with ``extrusion_type="radial"``) is
        available in the :attr:`radial_coordinates` attribute.
    ``"custom"``
        use a custom kernel to generate the extruded coordinates

    For more details see the :doc:`manual section on extruded meshes <extruded-meshes>`.
    """
    import firedrake.functionspace as functionspace
    import firedrake.function as function

    mesh.init()
    topology = ExtrudedMeshTopology(mesh.topology, layers)

    if extrusion_type == "uniform":
        pass
    elif extrusion_type in ("radial", "radial_hedgehog"):
        # do not allow radial extrusion if tdim = gdim
        if mesh.ufl_cell().geometric_dimension() == mesh.ufl_cell().topological_dimension():
            raise RuntimeError("Cannot radially-extrude a mesh with equal geometric and topological dimension")
    else:
        # check for kernel
        if kernel is None:
            raise RuntimeError("If the custom extrusion_type is used, a kernel must be provided")
        # otherwise, use the gdim that was passed in
        if gdim is None:
            raise RuntimeError("The geometric dimension of the mesh must be specified if a custom extrusion kernel is used")

    # Compute Coordinates of the extruded mesh
    if layer_height is None:
        # Default to unit
        layer_height = 1.0 / layers

    if extrusion_type == 'radial_hedgehog':
        hfamily = "DG"
    else:
        hfamily = mesh._coordinates.ufl_element().family()
    hdegree = mesh._coordinates.ufl_element().degree()

    if gdim is None:
        gdim = mesh.ufl_cell().geometric_dimension() + (extrusion_type == "uniform")
    coordinates_fs = functionspace.VectorFunctionSpace(topology, hfamily, hdegree, dim=gdim,
                                                       vfamily="Lagrange", vdegree=1)

    coordinates = function.CoordinatelessFunction(coordinates_fs, name="Coordinates")

    eutils.make_extruded_coords(topology, mesh._coordinates, coordinates,
                                layer_height, extrusion_type=extrusion_type, kernel=kernel)

    self = make_mesh_from_coordinates(coordinates)
    self._base_mesh = mesh

    if extrusion_type == "radial_hedgehog":
        fs = functionspace.VectorFunctionSpace(self, "CG", hdegree, dim=gdim,
                                               vfamily="CG", vdegree=1)
        self.radial_coordinates = function.Function(fs)
        eutils.make_extruded_coords(topology, mesh._coordinates, self.radial_coordinates,
                                    layer_height, extrusion_type="radial", kernel=kernel)

    return self


def SubDomainData(geometric_expr):
    import firedrake.functionspace as functionspace
    import firedrake.projection as projection

    # Find domain from expression
    m = geometric_expr.ufl_domain()

    # Find selected cells
    fs = functionspace.FunctionSpace(m, 'DG', 0)
    f = projection.project(ufl.conditional(geometric_expr, 1, 0), fs)

    # Create cell subset
    indices, = np.nonzero(f.dat.data > 0.5)
    return op2.Subset(m.cell_set, indices)
