from __future__ import absolute_import
import numpy as np
import os
import FIAT
import ufl
import weakref

from pyop2 import op2
from pyop2.profiling import timed_function, timed_region, profile
from pyop2.utils import as_tuple

import coffee.base as ast

import firedrake.dmplex as dmplex
import firedrake.extrusion_utils as eutils
import firedrake.fiat_utils as fiat_utils
import firedrake.utils as utils
from firedrake.parameters import parameters
from firedrake.petsc import PETSc


__all__ = ['Mesh', 'ExtrudedMesh']


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
        if isinstance(self.mesh, ExtrudedMesh):
            if self.kind == "interior":
                base = self.mesh._old_mesh.interior_facets.set
            else:
                base = self.mesh._old_mesh.exterior_facets.set
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


class Mesh(object):
    """A representation of mesh topology and geometry."""

    @timed_function("Build mesh")
    @profile
    def __init__(self, meshfile, **kwargs):
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
        :param periodic_coords: optional numpy array of coordinates
               used to replace those in the mesh object.  These are
               only supported in 1D and must have enough entries to be
               used as a DG1 field on the mesh.  Not supported when
               reading from file.

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

        utils._init()

        geometric_dim = kwargs.get("dim", None)
        reorder = kwargs.get("reorder", parameters["reorder_meshes"])
        periodic_coords = kwargs.get("periodic_coords", None)
        distribute = kwargs.get("distribute", True)

        if isinstance(meshfile, PETSc.DMPlex):
            name = "plexmesh"
            plex = meshfile
        else:
            name = meshfile
            basename, ext = os.path.splitext(meshfile)

            if periodic_coords is not None:
                raise RuntimeError("Periodic coordinates are unsupported when reading from file")
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

        # Mark exterior and interior facets
        # Note.  This must come before distribution, because otherwise
        # DMPlex will consider facets on the domain boundary to be
        # exterior, which is wrong.
        with timed_region("Mesh: label facets"):
            label_boundary = op2.MPI.comm.size == 1 or distribute
            dmplex.label_facets(plex, label_boundary=label_boundary)

        # Distribute the dm to all ranks
        if op2.MPI.comm.size > 1 and distribute:
            # We distribute with overlap zero, in case we're going to
            # refine this mesh in parallel.  Later, when we actually use
            # it, we grow the halo.
            plex.distribute(overlap=0)

        # A cache of function spaces that have been built on this mesh
        self._cache = {}
        self.parent = None

        self.name = name
        self._plex = plex
        self.uid = utils._new_uid()

        topological_dim = self._plex.getDimension()
        if geometric_dim is None:
            geometric_dim = topological_dim

        cStart, cEnd = self._plex.getHeightStratum(0)  # cells
        cell_facets = self._plex.getConeSize(cStart)

        self._ufl_cell = ufl.Cell(fiat_utils._cells[topological_dim][cell_facets],
                                  geometric_dimension=geometric_dim)
        self._ufl_domain = ufl.Domain(self.ufl_cell(), data=self)
        self._grown_halos = False

        def callback(self):
            import firedrake.function as function
            import firedrake.functionspace as functionspace

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
                entity_dofs = np.zeros(topological_dim+1, dtype=np.int32)
                entity_dofs[-1] = 1

                self._cell_numbering = self._plex.createSection([1], entity_dofs,
                                                                perm=self._plex_renumbering)
                entity_dofs[:] = 0
                entity_dofs[0] = 1
                self._vertex_numbering = self._plex.createSection([1], entity_dofs,
                                                                  perm=self._plex_renumbering)

            # Note that for bendy elements, this needs to change.
            with timed_region("Mesh: coordinate field"):
                if periodic_coords is not None:
                    if self.ufl_cell().geometric_dimension() != 1:
                        raise NotImplementedError("Periodic coordinates in more than 1D are unsupported")
                    # We've been passed a periodic coordinate field, so use that.
                    self._coordinate_fs = functionspace.VectorFunctionSpace(self, "DG", 1)
                    self.coordinates = function.Function(self._coordinate_fs,
                                                         val=periodic_coords,
                                                         name="Coordinates")
                else:
                    self._coordinate_fs = functionspace.VectorFunctionSpace(self, "Lagrange", 1)

                    coordinates = dmplex.reordered_coords(self._plex, self._coordinate_fs._global_numbering,
                                                          (self.num_vertices(), geometric_dim))
                    self.coordinates = function.Function(self._coordinate_fs,
                                                         val=coordinates,
                                                         name="Coordinates")
            self._ufl_domain = ufl.Domain(self.coordinates)
            # Build a new ufl element for this function space with the
            # correct domain.  This is necessary since this function space
            # is in the cache and will be picked up by later
            # VectorFunctionSpace construction.
            self._coordinate_fs._ufl_element = self._coordinate_fs.ufl_element().reconstruct(domain=self.ufl_domain())
            # HACK alert!
            # Replace coordinate Function by one that has a real domain on it (but don't copy values)
            self.coordinates = function.Function(self._coordinate_fs, val=self.coordinates.dat)
            # Add subdomain_data to the measure objects we store with
            # the mesh.  These are weakrefs for consistency with the
            # "global" measure objects
            self._dx = ufl.Measure('cell', subdomain_data=weakref.ref(self.coordinates))
            self._ds = ufl.Measure('exterior_facet', subdomain_data=weakref.ref(self.coordinates))
            self._dS = ufl.Measure('interior_facet', subdomain_data=weakref.ref(self.coordinates))
            # Set the subdomain_data on all the default measures to this
            # coordinate field.
            # We don't set the domain on the measure since this causes
            # an uncollectable reference in the global space (dx is
            # global).  Furthermore, it's never used anyway.
            for measure in [ufl.dx, ufl.ds, ufl.dS]:
                measure._subdomain_data = weakref.ref(self.coordinates)
        self._callback = callback

    def init(self):
        """Finish the initialisation of the mesh.  Most of the time
        this is carried out automatically, however, in some cases (for
        example accessing a property of the mesh directly after
        constructing it) you need to call this manually."""
        if hasattr(self, '_callback'):
            self._callback(self)

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

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        plex = self._plex
        topological_dim = plex.getDimension()

        # Cell numbering and global vertex numbering
        cell_numbering = self._cell_numbering
        vertex_numbering = self._vertex_numbering.createGlobalSection(plex.getPointSF())

        cStart, cEnd = plex.getHeightStratum(0)  # cells
        cell_facets = plex.getConeSize(cStart)

        if topological_dim + 1 == cell_facets:
            # Simplex mesh
            a_closure = plex.getTransitiveClosure(cStart)[0]

            entity_per_cell = np.zeros(topological_dim + 1, dtype=np.int32)
            for dim in xrange(topological_dim + 1):
                start, end = plex.getDepthStratum(dim)
                entity_per_cell[dim] = sum(map(lambda idx: start <= idx < end,
                                               a_closure))

            return dmplex.closure_ordering(plex, vertex_numbering,
                                           cell_numbering, entity_per_cell)

        elif topological_dim == 2 and cell_facets == 4:
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
            raise RuntimeError("Unsupported mesh: neither simplex, nor quadrilateral.")

    def create_cell_node_list(self, global_numbering, fiat_element):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg fiat_element: The FIAT element for the cell
        """
        return dmplex.get_cell_nodes(global_numbering,
                                     self.cell_closure,
                                     fiat_element)

    @property
    def coordinates(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        self.init()
        return self._coordinate_function

    @coordinates.setter
    def coordinates(self, value):
        self._coordinate_function = value

        # If the new coordinate field has a different dimension from
        # the geometric dimension of the existing cell, replace the
        # cell with one with the correct dimension.
        ufl_cell = self.ufl_cell()
        if value.element().value_shape()[0] != ufl_cell.geometric_dimension():
            if isinstance(ufl_cell, ufl.OuterProductCell):
                self._ufl_cell = ufl.OuterProductCell(ufl_cell._A, ufl_cell._B, value.element().value_shape()[0])
            else:
                self._ufl_cell = ufl.Cell(ufl_cell.cellname(),
                                          geometric_dimension=value.element().value_shape()[0])
            self._ufl_domain = ufl.Domain(self.ufl_cell(), data=self)

    @property
    def layers(self):
        """Return the number of layers of the extruded mesh
        represented by the number of occurences of the base mesh."""
        return self._layers

    def cell_orientations(self):
        """Return the orientation of each cell in the mesh.

        Use :func:`init_cell_orientations` to initialise this data."""
        if not hasattr(self, '_cell_orientations'):
            raise RuntimeError("No cell orientations found, did you forget to call init_cell_orientations?")
        return self._cell_orientations

    def init_cell_orientations(self, expr):
        """Compute and initialise :attr:`cell_orientations` relative to a specified orientation.

        :arg expr: an :class:`.Expression` evaluated to produce a
             reference normal direction.

        """
        import firedrake.function as function
        import firedrake.functionspace as functionspace

        if expr.value_shape()[0] != 3:
            raise NotImplementedError('Only implemented for 3-vectors')
        if self.ufl_cell() not in (ufl.Cell('triangle', 3), ufl.Cell("quadrilateral", 3), ufl.OuterProductCell(ufl.Cell('interval', 3), ufl.Cell('interval')), ufl.OuterProductCell(ufl.Cell('interval', 2), ufl.Cell('interval'), gdim=3)):
            raise NotImplementedError('Only implemented for triangles and quadrilaterals embedded in 3d')

        if hasattr(self, '_cell_orientations'):
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
        self._cell_orientations = cell_orientations

    def ufl_id(self):
        return id(self)

    def ufl_domain(self):
        return self._ufl_domain

    def ufl_cell(self):
        """The UFL :class:`~ufl.cell.Cell` associated with the mesh."""
        return self._ufl_cell

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

    @property
    def cell_classes(self):
        return self._entity_classes[self.cell_dimension(), :]

    @utils.cached_property
    def cell_set(self):
        size = list(self.cell_classes)
        return self.parent.cell_set if self.parent else \
            op2.Set(size, "%s_cells" % self.name)

    def facet_dimension(self):
        """Returns the facet dimension."""
        # Facets have co-dimension 1
        return self.ufl_cell().topological_dimension() - 1

    def cell_dimension(self):
        """Return the cell dimension"""
        return self.ufl_cell().topological_dimension()


class ExtrudedMesh(Mesh):
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

    @timed_function("Build extruded mesh")
    @profile
    def __init__(self, mesh, layers, layer_height=None, extrusion_type='uniform', kernel=None, gdim=None):
        # A cache of function spaces that have been built on this mesh
        import firedrake.function as function
        import firedrake.functionspace as functionspace

        self._cache = {}
        mesh.init()
        self._old_mesh = mesh
        if layers < 1:
            raise RuntimeError("Must have at least one layer of extruded cells (not %d)" % layers)
        # All internal logic works with layers of base mesh (not layers of cells)
        self._layers = layers + 1
        self.parent = mesh.parent
        self.uid = mesh.uid
        self.name = mesh.name
        self._plex = mesh._plex
        self._plex_renumbering = mesh._plex_renumbering
        self._cell_numbering = mesh._cell_numbering
        self._entity_classes = mesh._entity_classes

        interior_f = self._old_mesh.interior_facets
        self._interior_facets = _Facets(self, interior_f.classes,
                                        "interior",
                                        interior_f.facet_cell,
                                        interior_f.local_facet_number)
        exterior_f = self._old_mesh.exterior_facets
        self._exterior_facets = _Facets(self, exterior_f.classes,
                                        "exterior",
                                        exterior_f.facet_cell,
                                        exterior_f.local_facet_number,
                                        exterior_f.markers,
                                        unique_markers=exterior_f.unique_markers)

        self.ufl_cell_element = ufl.FiniteElement("Lagrange",
                                                  domain=mesh.ufl_cell(),
                                                  degree=1)
        self.ufl_interval_element = ufl.FiniteElement("Lagrange",
                                                      domain=ufl.Cell("interval", 1),
                                                      degree=1)

        self.fiat_base_element = fiat_utils.fiat_from_ufl_element(self.ufl_cell_element)
        self.fiat_vert_element = fiat_utils.fiat_from_ufl_element(self.ufl_interval_element)

        fiat_element = FIAT.tensor_finite_element.TensorFiniteElement(self.fiat_base_element, self.fiat_vert_element)

        if extrusion_type == "uniform":
            # *must* add a new dimension
            self._ufl_cell = ufl.OuterProductCell(mesh.ufl_cell(), ufl.Cell("interval", 1), gdim=mesh.ufl_cell().geometric_dimension() + 1)

        elif extrusion_type in ("radial", "radial_hedgehog"):
            # do not allow radial extrusion if tdim = gdim
            if mesh.ufl_cell().geometric_dimension() == mesh.ufl_cell().topological_dimension():
                raise RuntimeError("Cannot radially-extrude a mesh with equal geometric and topological dimension")
            # otherwise, all is fine, so make cell
            self._ufl_cell = ufl.OuterProductCell(mesh.ufl_cell(), ufl.Cell("interval", 1))

        else:
            # check for kernel
            if kernel is None:
                raise RuntimeError("If the custom extrusion_type is used, a kernel must be provided")
            # otherwise, use the gdim that was passed in
            if gdim is None:
                raise RuntimeError("The geometric dimension of the mesh must be specified if a custom extrusion kernel is used")
            self._ufl_cell = ufl.OuterProductCell(mesh.ufl_cell(), ufl.Cell("interval", 1), gdim=gdim)

        self._ufl_domain = ufl.Domain(self.ufl_cell(), data=self)
        flat_temp = fiat_utils.FlattenedElement(fiat_element)

        # Calculated dofs_per_column from flattened_element and layers.
        # The mirrored elements have to be counted only once.
        # Then multiply by layers and layers - 1 accordingly.
        self.dofs_per_column = eutils.compute_extruded_dofs(fiat_element, flat_temp.entity_dofs(),
                                                            layers)

        # Compute Coordinates of the extruded mesh
        if layer_height is None:
            # Default to unit
            layer_height = 1.0 / layers

        if extrusion_type == 'radial_hedgehog':
            hfamily = "DG"
        else:
            hfamily = mesh.coordinates.element().family()
        hdegree = mesh.coordinates.element().degree()

        self._coordinate_fs = functionspace.VectorFunctionSpace(self, hfamily,
                                                                hdegree,
                                                                vfamily="CG",
                                                                vdegree=1)

        self.coordinates = function.Function(self._coordinate_fs)
        self._ufl_domain = ufl.Domain(self.coordinates)
        eutils.make_extruded_coords(self, layer_height, extrusion_type=extrusion_type,
                                    kernel=kernel)
        if extrusion_type == "radial_hedgehog":
            fs = functionspace.VectorFunctionSpace(self, "CG", hdegree, vfamily="CG", vdegree=1)
            self.radial_coordinates = function.Function(fs)
            eutils.make_extruded_coords(self, layer_height, extrusion_type="radial",
                                        output_coords=self.radial_coordinates)

        # Build a new ufl element for this function space with the
        # correct domain.  This is necessary since this function space
        # is in the cache and will be picked up by later
        # VectorFunctionSpace construction.
        self._coordinate_fs._ufl_element = self._coordinate_fs.ufl_element().reconstruct(domain=self.ufl_domain())
        # HACK alert!
        # Replace coordinate Function by one that has a real domain on it (but don't copy values)
        self.coordinates = function.Function(self._coordinate_fs, val=self.coordinates.dat)
        # Add subdomain_data to the measure objects we store with
        # the mesh.  These are weakrefs for consistency with the
        # "global" measure objects
        self._dx = ufl.Measure('cell', subdomain_data=weakref.ref(self.coordinates))
        self._ds = ufl.Measure('exterior_facet', subdomain_data=weakref.ref(self.coordinates))
        self._dS = ufl.Measure('interior_facet', subdomain_data=weakref.ref(self.coordinates))
        self._ds_t = ufl.Measure('exterior_facet_top', subdomain_data=weakref.ref(self.coordinates))
        self._ds_b = ufl.Measure('exterior_facet_bottom', subdomain_data=weakref.ref(self.coordinates))
        self._ds_v = ufl.Measure('exterior_facet_vert', subdomain_data=weakref.ref(self.coordinates))
        self._dS_h = ufl.Measure('interior_facet_horiz', subdomain_data=weakref.ref(self.coordinates))
        self._dS_v = ufl.Measure('interior_facet_vert', subdomain_data=weakref.ref(self.coordinates))
        # Set the subdomain_data on all the default measures to this
        # coordinate field.  We don't set the domain on the measure
        # since this causes an uncollectable reference in the global
        # space (dx is global).  Furthermore, it's never used anyway.
        for measure in [ufl.ds, ufl.dS, ufl.dx, ufl.ds_t, ufl.ds_b, ufl.ds_v, ufl.dS_h, ufl.dS_v]:
            measure._subdomain_data = weakref.ref(self.coordinates)

    @property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        return self._old_mesh.cell_closure

    def create_cell_node_list(self, global_numbering, fiat_element):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg fiat_element: The FIAT element for the cell
        """
        return dmplex.get_cell_nodes(global_numbering,
                                     self.cell_closure,
                                     fiat_utils.FlattenedElement(fiat_element))

    @property
    def layers(self):
        """Return the number of layers of the extruded mesh
        represented by the number of occurences of the base mesh."""
        return self._layers

    @utils.cached_property
    def cell_set(self):
        return self.parent.cell_set if self.parent else \
            op2.ExtrudedSet(self._old_mesh.cell_set, layers=self._layers)

    @property
    def exterior_facets(self):
        return self._exterior_facets

    @property
    def interior_facets(self):
        return self._interior_facets

    @property
    def geometric_dimension(self):
        return self.ufl_cell().geometric_dimension()

    def cell_dimension(self):
        """Return the cell dimension"""
        if self.geometric_dimension == 3:
            return (2, 1)
        elif self.geometric_dimension == 2:
            return (1, 1)
        else:
            raise RuntimeError("Dimension computation not supported for gdim %d",
                               self.geometric_dimension)

    def facet_dimension(self):
        """Returns the facet dimension.

        .. note::

            This only returns the dimension of the "side" (vertical) facets,
            not the "top" or "bottom" (horizontal) facets.

        """
        # The facet is indexed by (base-ele-codim 1, 1) for
        # extruded meshes.
        # e.g. for the two supported options of
        # triangle x interval interval x interval it's (1, 1) and
        # (0, 1) respectively.
        if self.geometric_dimension == 3:
            return (1, 1)
        elif self.geometric_dimension == 2:
            return (0, 1)
        else:
            raise RuntimeError("Dimension computation for other than 2D or 3D extruded meshes not supported.")
