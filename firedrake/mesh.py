import numpy as np
import ctypes
import os
import sys
import ufl
import weakref
from collections import OrderedDict, defaultdict
from ufl.classes import ReferenceGrad
import enum
import numbers
import abc

from mpi4py import MPI
from firedrake.utils import IntType, RealType
import pyop2
from pyop2 import op2
from pyop2.mpi import COMM_WORLD, dup_comm
from pyop2.utils import as_tuple, tuplify

import firedrake.cython.dmcommon as dmcommon
import firedrake.cython.extrusion_numbering as extnum
import firedrake.extrusion_utils as eutils
import firedrake.cython.spatialindex as spatialindex
import firedrake.utils as utils
from firedrake.logging import info_red
from firedrake.parameters import parameters
from firedrake.petsc import PETSc, OptionsManager
from firedrake.adjoint import MeshGeometryMixin


__all__ = ['Mesh', 'ExtrudedMesh', 'VertexOnlyMesh', 'SubDomainData', 'unmarked',
           'DistributedMeshOverlapType']


_cells = {
    1: {2: "interval"},
    2: {3: "triangle", 4: "quadrilateral"},
    3: {4: "tetrahedron"}
}


unmarked = -1
"""A mesh marker that selects all entities that are not explicitly marked."""


class DistributedMeshOverlapType(enum.Enum):
    """How should the mesh overlap be grown for distributed meshes?

    Possible options are:

     - :attr:`NONE`:  Don't overlap distributed meshes, only useful for problems with
              no interior facet integrals.
     - :attr:`FACET`: Add ghost entities in the closure of the star of
              facets.
     - :attr:`VERTEX`: Add ghost entities in the closure of the star
              of vertices.

    Defaults to :attr:`FACET`.
    """
    NONE = 1
    FACET = 2
    VERTEX = 3


class _Facets(object):
    """Wrapper class for facet interation information on a :func:`Mesh`

    .. warning::

       The unique_markers argument **must** be the same on all processes."""

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, classes, kind, facet_cell, local_facet_number, markers=None,
                 unique_markers=None):

        self.mesh = mesh

        classes = as_tuple(classes, int, 3)
        self.classes = classes

        self.kind = kind
        assert kind in ["interior", "exterior"]
        if kind == "interior":
            self._rank = 2
        else:
            self._rank = 1

        self.facet_cell = facet_cell

        if isinstance(self.set, op2.ExtrudedSet):
            dset = op2.DataSet(self.set.parent, self._rank)
        else:
            dset = op2.DataSet(self.set, self._rank)

        # Dat indicating which local facet of each adjacent cell corresponds
        # to the current facet.
        self.local_facet_dat = op2.Dat(dset, local_facet_number, np.uintc,
                                       "%s_%s_local_facet_number" %
                                       (self.mesh.name, self.kind))

        # assert that markers is a proper subset of unique_markers
        if markers is not None:
            assert set(markers) <= set(unique_markers).union([unmarked]), \
                "Every marker has to be contained in unique_markers"

        self.markers = markers
        self.unique_markers = [] if unique_markers is None else unique_markers
        self._subsets = {}

    @utils.cached_property
    def set(self):
        size = self.classes
        if isinstance(self.mesh, ExtrudedMeshTopology):
            label = "%s_facets" % self.kind
            layers = self.mesh.entity_layers(1, label)
            base = getattr(self.mesh._base_mesh, label).set
            return op2.ExtrudedSet(base, layers=layers)
        return op2.Set(size, "%sFacets" % self.kind.capitalize()[:3],
                       comm=self.mesh.comm)

    @utils.cached_property
    def _null_subset(self):
        '''Empty subset for the case in which there are no facets with
        a given marker value. This is required because not all
        markers need be represented on all processors.'''

        return op2.Subset(self.set, [])

    @PETSc.Log.EventDecorator()
    def measure_set(self, integral_type, subdomain_id,
                    all_integer_subdomain_ids=None):
        """Return an iteration set appropriate for the requested integral type.

        :arg integral_type: The type of the integral (should be a facet measure).
        :arg subdomain_id: The subdomain of the mesh to iterate over.
             Either an integer, an iterable of integers or the special
             subdomains ``"everywhere"`` or ``"otherwise"``.
        :arg all_integer_subdomain_ids: Information to interpret the
             ``"otherwise"`` subdomain.  ``"otherwise"`` means all
             entities not explicitly enumerated by the integer
             subdomains provided here.  For example, if
             all_integer_subdomain_ids is empty, then ``"otherwise" ==
             "everywhere"``.  If it contains ``(1, 2)``, then
             ``"otherwise"`` is all entities except those marked by
             subdomains 1 and 2.

         :returns: A :class:`pyop2.Subset` for iteration.
        """
        if integral_type in ("exterior_facet_bottom",
                             "exterior_facet_top",
                             "interior_facet_horiz"):
            # these iterate over the base cell set
            return self.mesh.cell_subset(subdomain_id, all_integer_subdomain_ids)
        elif not (integral_type.startswith("exterior_")
                  or integral_type.startswith("interior_")):
            raise ValueError("Don't know how to construct measure for '%s'" % integral_type)
        if subdomain_id == "everywhere":
            return self.set
        if subdomain_id == "otherwise":
            if all_integer_subdomain_ids is None:
                return self.set
            key = ("otherwise", ) + all_integer_subdomain_ids
            try:
                return self._subsets[key]
            except KeyError:
                ids = [np.where(self.markers == sid)[0]
                       for sid in all_integer_subdomain_ids]
                to_remove = np.unique(np.concatenate(ids))
                indices = np.arange(self.set.total_size, dtype=np.int32)
                indices = np.delete(indices, to_remove)
                return self._subsets.setdefault(key, op2.Subset(self.set, indices))
        else:
            return self.subset(subdomain_id)

    @PETSc.Log.EventDecorator()
    def subset(self, markers):
        """Return the subset corresponding to a given marker value.

        :param markers: integer marker id or an iterable of marker ids
            (or ``None``, for an empty subset).
        """
        valid_markers = set([unmarked]).union(self.unique_markers)
        markers = as_tuple(markers, numbers.Integral)
        if self.markers is None and valid_markers.intersection(markers):
            return self._null_subset
        try:
            return self._subsets[markers]
        except KeyError:
            # check that the given markers are valid
            if len(set(markers).difference(valid_markers)) > 0:
                invalid = set(markers).difference(valid_markers)
                raise LookupError("{0} are not a valid markers (not in {1})".format(invalid, self.unique_markers))

            # build a list of indices corresponding to the subsets selected by
            # markers
            indices = np.concatenate([np.nonzero(self.markers == i)[0]
                                      for i in markers])
            return self._subsets.setdefault(markers, op2.Subset(self.set, indices))

    @utils.cached_property
    def facet_cell_map(self):
        """Map from facets to cells."""
        return op2.Map(self.set, self.mesh.cell_set, self._rank, self.facet_cell,
                       "facet_to_cell_map")


@deprecated
def MeshTopology(*args, **kwargs):
    return pyop2.mesh.UnstructuredMesh(*args, **kwargs)


@deprecated
def ExtrudedMeshTopology(*args, **kwargs):
    return pyop2.mesh.ExtrudedMesh(*args, **kwargs)


@deprecated
def VertexOnlyMeshTopology(*args, **kwargs):
    return pyop2.mesh.VertexOnlyMesh(*args, **kwargs)


class MeshGeometryCargo:
    """Helper class carrying data for a :class:`MeshGeometry`.

    It is required because it permits Firedrake to have stripped forms
    that still know that they are on an extruded mesh (for example).
    """

    def __init__(self, ufl_id):
        self._ufl_id = ufl_id

    def ufl_id(self):
        return self._ufl_id

    def init(self, coordinates):
        """Initialise the cargo.

        This function is separate to __init__ because of the two-step process we have
        for initialising a :class:`MeshGeometry`.
        """
        self.topology = coordinates.function_space().mesh()
        self.coordinates = coordinates
        self.geometric_shared_data_cache = defaultdict(dict)


class MeshGeometry(ufl.Mesh, MeshGeometryMixin):
    """A representation of mesh topology and geometry."""

    def __new__(cls, element):
        """Create mesh geometry object."""
        utils._init()
        mesh = super(MeshGeometry, cls).__new__(cls)
        uid = utils._new_uid()
        mesh.uid = uid
        cargo = MeshGeometryCargo(uid)
        assert isinstance(element, ufl.FiniteElementBase)
        ufl.Mesh.__init__(mesh, element, ufl_id=mesh.uid, cargo=cargo)
        return mesh

    @MeshGeometryMixin._ad_annotate_init
    def __init__(self, coordinates):
        """Initialise a mesh geometry from coordinates.

        :arg coordinates: a coordinateless function containing the coordinates
        """
        topology = coordinates.function_space().mesh()

        # this is codegen information so we attach it to the MeshGeometry rather than its cargo
        self.extruded = isinstance(topology, ExtrudedMeshTopology)
        self.variable_layers = self.extruded and topology.variable_layers

        # initialise the mesh cargo
        self.ufl_cargo().init(coordinates)

        # Cache mesh object on the coordinateless coordinates function
        coordinates._as_mesh_geometry = weakref.ref(self)

    def _ufl_signature_data_(self, *args, **kwargs):
        return (type(self), self.extruded, self.variable_layers,
                super()._ufl_signature_data_(*args, **kwargs))

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
        return self.ufl_cargo().topology

    @topology.setter
    def topology(self, val):
        self.ufl_cargo().topology = val

    @property
    def _topology(self):
        return self.topology

    @_topology.setter
    def _topology(self, val):
        self.topology = val

    @property
    def _parent_mesh(self):
        return self.ufl_cargo()._parent_mesh

    @_parent_mesh.setter
    def _parent_mesh(self, val):
        self.ufl_cargo()._parent_mesh = val

    @property
    def _coordinates(self):
        return self.ufl_cargo().coordinates

    @property
    def _geometric_shared_data_cache(self):
        return self.ufl_cargo().geometric_shared_data_cache

    @property
    def topological(self):
        """Alias of topology.

        This is to ensure consistent naming for some multigrid codes."""
        return self._topology

    @property
    def _topology_dm(self):
        """Alias of topology_dm"""
        from warnings import warn
        warn("_topology_dm is deprecated (use topology_dm instead)", DeprecationWarning, stacklevel=2)
        return self.topology_dm

    @property
    @MeshGeometryMixin._ad_annotate_coordinates_function
    def _coordinates_function(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        import firedrake.functionspaceimpl as functionspaceimpl
        import firedrake.function as function

        if hasattr(self.ufl_cargo(), "_coordinates_function"):
            return self.ufl_cargo()._coordinates_function
        else:
            self.init()
            coordinates_fs = self._coordinates.function_space()
            V = functionspaceimpl.WithGeometry.create(coordinates_fs, self)
            f = function.Function(V, val=self._coordinates)
            self.ufl_cargo()._coordinates_function = f
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
    def cell_sizes(self):
        """A :class`~.Function` in the :math:`P^1` space containing the local mesh size.

        This is computed by the :math:`L^2` projection of the local mesh element size."""
        from firedrake.ufl_expr import CellSize
        from firedrake.functionspace import FunctionSpace
        from firedrake.projection import project
        P1 = FunctionSpace(self, "Lagrange", 1)
        return project(CellSize(self), P1)

    def clear_cell_sizes(self):
        """Reset the :attr:`cell_sizes` field on this mesh geometry.

        Use this if you move the mesh.
        """
        try:
            del self.cell_size
        except AttributeError:
            pass

    def clear_spatial_index(self):
        """Reset the :attr:`spatial_index` on this mesh geometry.

        Use this if you move the mesh (for example by reassigning to
        the coordinate field)."""
        try:
            del self.spatial_index
        except AttributeError:
            pass

    @utils.cached_property
    def spatial_index(self):
        """Spatial index to quickly find which cell contains a given point."""

        from firedrake import function, functionspace
        from firedrake.parloops import par_loop, READ, MIN, MAX

        gdim = self.ufl_cell().geometric_dimension()
        if gdim <= 1:
            info_red("libspatialindex does not support 1-dimension, falling back on brute force.")
            return None

        # Calculate the bounding boxes for all cells by running a kernel
        V = functionspace.VectorFunctionSpace(self, "DG", 0, dim=gdim)
        coords_min = function.Function(V, dtype=RealType)
        coords_max = function.Function(V, dtype=RealType)

        coords_min.dat.data.fill(np.inf)
        coords_max.dat.data.fill(-np.inf)

        if utils.complex_mode:
            if not np.allclose(self.coordinates.dat.data_ro.imag, 0):
                raise ValueError("Coordinate field has non-zero imaginary part")
            coords = function.Function(self.coordinates.function_space(),
                                       val=self.coordinates.dat.data_ro_with_halos.real.copy(),
                                       dtype=RealType)
        else:
            coords = self.coordinates

        cell_node_list = self.coordinates.function_space().cell_node_list
        _, nodes_per_cell = cell_node_list.shape

        domain = "{{[d, i]: 0 <= d < {0} and 0 <= i < {1}}}".format(gdim, nodes_per_cell)
        instructions = """
        for d, i
            f_min[0, d] = fmin(f_min[0, d], f[i, d])
            f_max[0, d] = fmax(f_max[0, d], f[i, d])
        end
        """
        par_loop((domain, instructions), ufl.dx,
                 {'f': (coords, READ),
                  'f_min': (coords_min, MIN),
                  'f_max': (coords_max, MAX)},
                 is_loopy_kernel=True)

        # Reorder bounding boxes according to the cell indices we use
        column_list = V.cell_node_list.reshape(-1)
        coords_min = self._order_data_by_cell_index(column_list, coords_min.dat.data_ro_with_halos)
        coords_max = self._order_data_by_cell_index(column_list, coords_max.dat.data_ro_with_halos)

        # Build spatial index
        return spatialindex.from_regions(coords_min, coords_max)

    @PETSc.Log.EventDecorator()
    def locate_cell(self, x, tolerance=None):
        """Locate cell containing a given point.

        :arg x: point coordinates
        :kwarg tolerance: for checking if a point is in a cell. Default
            is None.
        :returns: cell number (int), or None (if the point is not
            in the domain)
        """
        return self.locate_cell_and_reference_coordinate(x, tolerance=tolerance)[0]

    def locate_reference_coordinate(self, x, tolerance=None):
        """Get reference coordinates of a given point in its cell. Which
        cell the point is in can be queried with the locate_cell method.

        :arg x: point coordinates
        :kwarg tolerance: for checking if a point is in a cell. Default
            is None.
        :returns: reference coordinates within cell (numpy array) or
            None (if the point is not in the domain)
        """
        return self.locate_cell_and_reference_coordinate(x, tolerance=tolerance)[1]

    def locate_cell_and_reference_coordinate(self, x, tolerance=None):
        """Locate cell containing a given point and the reference
        coordinates of the point within the cell.

        :arg x: point coordinates
        :kwarg tolerance: for checking if a point is in a cell. Default
            is None.
        :returns: tuple either (cell number, reference coordinates)
            (int, numpy array), or (None, None) (point is not in the domain)
        """
        if self.variable_layers:
            raise NotImplementedError("Cell location not implemented for variable layers")
        x = np.asarray(x, dtype=utils.ScalarType)
        if not np.allclose(x.imag, 0):
            raise ValueError("Point coordinates must have zero imaginary part")
        x = x.real.copy()
        if x.size != self.geometric_dimension():
            raise ValueError("Point coordinate dimension does not match mesh geometric dimension")
        X = np.empty_like(x)
        cell = self._c_locator(tolerance=tolerance)(self.coordinates._ctypes,
                                                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if cell == -1:
            return (None, None)
        else:
            return cell, X

    def _c_locator(self, tolerance=None):
        from pyop2 import compilation
        from pyop2.utils import get_petsc_dir
        import firedrake.function as function
        import firedrake.pointquery_utils as pq_utils

        cache = self.__dict__.setdefault("_c_locator_cache", {})
        try:
            return cache[tolerance]
        except KeyError:
            src = pq_utils.src_locate_cell(self, tolerance=tolerance)
            src += """
    int locator(struct Function *f, double *x, double *X)
    {
        struct ReferenceCoords reference_coords;
        int cell = locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &to_reference_coords_xtr, &reference_coords);
        for(int i=0; i<%(geometric_dimension)d; i++) {
            X[i] = reference_coords.X[i];
        }
        return cell;
    }
    """ % dict(geometric_dimension=self.geometric_dimension())

            locator = compilation.load(src, "c", "locator",
                                       cppargs=["-I%s" % os.path.dirname(__file__),
                                                "-I%s/include" % sys.prefix]
                                       + ["-I%s/include" % d for d in get_petsc_dir()],
                                       ldargs=["-L%s/lib" % sys.prefix,
                                               "-lspatialindex_c",
                                               "-Wl,-rpath,%s/lib" % sys.prefix])

            locator.argtypes = [ctypes.POINTER(function._CFunction),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double)]
            locator.restype = ctypes.c_int
            return cache.setdefault(tolerance, locator)

    @PETSc.Log.EventDecorator()
    def init_cell_orientations(self, expr):
        """Compute and initialise :attr:`cell_orientations` relative to a specified orientation.

        :arg expr: a UFL expression evaluated to produce a
             reference normal direction.

        """
        import firedrake.function as function
        import firedrake.functionspace as functionspace

        if self.ufl_cell() not in (ufl.Cell('triangle', 3),
                                   ufl.Cell("quadrilateral", 3),
                                   ufl.TensorProductCell(ufl.Cell('interval'), ufl.Cell('interval'),
                                                         geometric_dimension=3)):
            raise NotImplementedError('Only implemented for triangles and quadrilaterals embedded in 3d')

        if hasattr(self.topology, '_cell_orientations'):
            raise RuntimeError("init_cell_orientations already called, did you mean to do so again?")

        if isinstance(expr, ufl.classes.Expr):
            if expr.ufl_shape != (3,):
                raise NotImplementedError('Only implemented for 3-vectors')
        else:
            raise TypeError("UFL expression expected!")

        fs = functionspace.FunctionSpace(self, 'DG', 0)
        x = ufl.SpatialCoordinate(self)
        f = function.Function(fs)
        f.interpolate(ufl.dot(expr, ufl.cross(ReferenceGrad(x)[:, 0], ReferenceGrad(x)[:, 1])))

        cell_orientations = function.Function(fs, name="cell_orientations", dtype=np.int32)
        cell_orientations.dat.data[:] = (f.dat.data_ro < 0)
        self.topology._cell_orientations = cell_orientations

    def __getattr__(self, name):
        val = getattr(self._topology, name)
        setattr(self, name, val)
        return val

    def __dir__(self):
        current = super(MeshGeometry, self).__dir__()
        return list(OrderedDict.fromkeys(dir(self._topology) + current))


@PETSc.Log.EventDecorator()
def make_mesh_from_coordinates(coordinates):
    """Given a coordinate field build a new mesh, using said coordinate field.

    :arg coordinates: A :class:`~.Function`.
    """
    if hasattr(coordinates, '_as_mesh_geometry'):
        mesh = coordinates._as_mesh_geometry()
        if mesh is not None:
            return mesh

    V = coordinates.function_space()
    element = coordinates.ufl_element()
    if V.rank != 1 or len(element.value_shape()) != 1:
        raise ValueError("Coordinates must be from a rank-1 FunctionSpace with rank-1 value_shape.")
    assert V.mesh().ufl_cell().topological_dimension() <= V.value_size
    # Build coordinate element
    element = coordinates.ufl_element()
    cell = element.cell().reconstruct(geometric_dimension=V.value_size)
    element = element.reconstruct(cell=cell)

    mesh = MeshGeometry.__new__(MeshGeometry, element)
    mesh.__init__(coordinates)
    # Mark mesh as being made from coordinates
    mesh._made_from_coordinates = True
    return mesh


@PETSc.Log.EventDecorator("CreateMesh")
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
           default value in ``parameters["reorder_meshes"]``
           is used.
    :param distribution_parameters:  an optional dictionary of options for
           parallel mesh distribution.  Supported keys are:

             - ``"partition"``: which may take the value ``None`` (use
                 the default choice), ``False`` (do not) ``True``
                 (do), or a 2-tuple that specifies a partitioning of
                 the cells (only really useful for debugging).
             - ``"partitioner_type"``: which may take ``"chaco"``,
                 ``"ptscotch"``, ``"parmetis"``, or ``"shell"``.
             - ``"overlap_type"``: a 2-tuple indicating how to grow
                 the mesh overlap.  The first entry should be a
                 :class:`DistributedMeshOverlapType` instance, the
                 second the number of levels of overlap.

    :param comm: the communicator to use when creating the mesh.  If
           not supplied, then the mesh will be created on COMM_WORLD.
           Ignored if ``meshfile`` is a DMPlex object (in which case
           the communicator will be taken from there).

    When the mesh is read from a file the following mesh formats
    are supported (determined, case insensitively, from the
    filename extension):

    * GMSH: with extension `.msh`
    * Exodus: with extension `.e`, `.exo`
    * CGNS: with extension `.cgns`
    * Triangle: with extension `.node`

    .. note::

        When the mesh is created directly from a DMPlex object,
        the ``dim`` parameter is ignored (the DMPlex already
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

    geometric_dim = kwargs.get("dim")

    if isinstance(meshfile, pyop2.mesh.Mesh):
        topology = meshfile
    else:
        reorder = kwargs.get("reorder") or parameters["reorder_meshes"]
        distribution_parameters = kwargs.get("distribution_parameters") or {}

        if isinstance(meshfile, PETSc.DMPlex):
            name = "plexmesh"
            plex = meshfile
            # Create mesh topology
            # TODO put this inside if statement as it should be possible to pass in topology
            topology = MeshTopology(plex, name=name, reorder=reorder,
                                    distribution_parameters=distribution_parameters)
        else:
            topology = pyop2.mesh.Mesh.from_file(meshfile, name=name, reorder=reorder,
                                                 distribution_parameters=distribution_parameters)

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

        coordinates_fs = functionspace.VectorFunctionSpace(self.topology, "Lagrange", 1,
                                                           dim=geometric_dim)

        coordinates_data = dmcommon.reordered_coords(plex, coordinates_fs.dm.getDefaultSection(),
                                                     (self.num_vertices(), geometric_dim))

        coordinates = function.CoordinatelessFunction(coordinates_fs,
                                                      val=coordinates_data,
                                                      name="Coordinates")

        self.__init__(coordinates)

    mesh._callback = callback
    return mesh


@PETSc.Log.EventDecorator("CreateExtMesh")
def ExtrudedMesh(mesh, layers, layer_height=None, extrusion_type='uniform', kernel=None, gdim=None):
    """Build an extruded mesh from an input mesh

    :arg mesh:           the unstructured base mesh
    :arg layers:         number of extruded cell layers in the "vertical"
                         direction.  One may also pass an array of
                         shape (cells, 2) to specify a variable number
                         of layers.  In this case, each entry is a pair
                         ``[a, b]`` where ``a`` indicates the starting
                         cell layer of the column and ``b`` the number
                         of cell layers in that column.
    :arg layer_height:   the layer height.  A scalar value will result in
                         evenly-spaced layers, whereas an array of values
                         will vary the layer height through the extrusion.
                         If this is omitted, the value defaults to
                         1/layers (i.e. the extruded mesh has total height 1.0)
                         unless a custom kernel is used.  Must be
                         provided if using a variable number of layers.
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
    layers = np.asarray(layers, dtype=IntType)
    if layers.shape:
        if layers.shape != (mesh.cell_set.total_size, 2):
            raise ValueError("Must provide single layer number or array of shape (%d, 2), not %s",
                             mesh.cell_set.total_size, layers.shape)
        if layer_height is None:
            raise ValueError("Must provide layer height for variable layers")

        # variable-height layers need to be present for the maximum number
        # of extruded layers
        num_layers = layers.sum(axis=1).max() if mesh.cell_set.total_size else 0
        num_layers = mesh.comm.allreduce(num_layers, op=MPI.MAX)

        # Convert to internal representation
        layers[:, 1] += 1 + layers[:, 0]

    else:
        if layer_height is None:
            # Default to unit
            layer_height = 1 / layers

        num_layers = layers

        # All internal logic works with layers of base mesh (not layers of cells)
        layers = layers + 1

    try:
        assert num_layers == len(layer_height)
    except TypeError:
        # layer_height is a scalar; equi-distant layers are fine
        pass

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

    helement = mesh._coordinates.ufl_element().sub_elements()[0]
    if extrusion_type == 'radial_hedgehog':
        helement = helement.reconstruct(family="DG", variant="equispaced")
    velement = ufl.FiniteElement("Lagrange", ufl.interval, 1)
    element = ufl.TensorProductElement(helement, velement)

    if gdim is None:
        gdim = mesh.ufl_cell().geometric_dimension() + (extrusion_type == "uniform")
    coordinates_fs = functionspace.VectorFunctionSpace(topology, element, dim=gdim)

    coordinates = function.CoordinatelessFunction(coordinates_fs, name="Coordinates")

    eutils.make_extruded_coords(topology, mesh._coordinates, coordinates,
                                layer_height, extrusion_type=extrusion_type, kernel=kernel)

    self = make_mesh_from_coordinates(coordinates)
    self._base_mesh = mesh

    if extrusion_type == "radial_hedgehog":
        helement = mesh._coordinates.ufl_element().sub_elements()[0].reconstruct(family="CG")
        element = ufl.TensorProductElement(helement, velement)
        fs = functionspace.VectorFunctionSpace(self, element, dim=gdim)
        self.radial_coordinates = function.Function(fs)
        eutils.make_extruded_coords(topology, mesh._coordinates, self.radial_coordinates,
                                    layer_height, extrusion_type="radial", kernel=kernel)

    return self


@PETSc.Log.EventDecorator()
def VertexOnlyMesh(mesh, vertexcoords, missing_points_behaviour=None):
    """
    Create a vertex only mesh, immersed in a given mesh, with vertices
    defined by a list of coordinates.

    :arg mesh: The unstructured mesh in which to immerse the vertex only
        mesh.
    :arg vertexcoords: A list of coordinate tuples which defines the vertices.
    :kwarg missing_points_behaviour: optional string argument for what to do
        when vertices which are outside of the mesh are discarded. If ``'warn'``,
        will print a warning. If ``'error'`` will raise a ValueError. Note that
        setting this will cause all MPI ranks to check that they have the same
        list of vertices (else the test is not possible): this operation scales
        with number of vertices and number of ranks.

    .. note::

        The vertex only mesh uses the same communicator as the input ``mesh``.

    .. note::

        Meshes created from a coordinates :py:class:`~.Function` and immersed
        manifold meshes are not yet supported.

    .. note::

        This should also only be used for meshes which have not had their
        coordinates field modified as, at present, this does not update the
        coordinates field of the underlying DMPlex. Such meshes may cause
        unexpected behavioir or hangs when running in parallel.

    .. note::
        When running in parallel, ``vertexcoords`` are strictly confined
        to the local ``mesh`` cells of that rank. This means that if rank
        A has ``vertexcoords`` {X} that are not found in the mesh cells
        owned by rank A but are found in the mesh cells owned by rank B,
        **and rank B has not been supplied with those** ``vertexcoords``,
        then the ``vertexcoords`` {X} will be lost.

        This can be avoided by either

        #. making sure that all ranks are supplied with the same
           ``vertexcoords`` or by
        #. ensuring that ``vertexcoords`` are already found in cells
           owned by the ``mesh`` partition of the given rank.

        For more see `this github issue
        <https://github.com/firedrakeproject/firedrake/issues/2178>`_.

    """

    import firedrake.functionspace as functionspace
    import firedrake.function as function

    mesh.init()

    vertexcoords = np.asarray(vertexcoords, dtype=np.double)
    gdim = mesh.geometric_dimension()
    tdim = mesh.topological_dimension()
    _, pdim = vertexcoords.shape

    if isinstance(mesh.topology, ExtrudedMeshTopology):
        raise NotImplementedError("Extruded meshes are not supported")

    if gdim != tdim:
        raise NotImplementedError("Immersed manifold meshes are not supported")

    # TODO Some better method of matching points to cells will need to
    # be used for bendy meshes since our PETSc DMPlex implementation
    # only supports straight-edged mesh topologies and meshes made from
    # coordinate fields.
    # We can hopefully update the coordinates field correctly so that
    # the DMSwarm PIC can immerse itself in the DMPlex.
    # We can also hopefully provide a callback for PETSc to use to find
    # the parent cell id. We would add `DMLocatePoints` as an `op` to
    # `DMShell` types and do `DMSwarmSetCellDM(yourdmshell)` which has
    # `DMLocatePoints_Shell` implemented.
    # Whether one or both of these is needed is unclear.

    if mesh.coordinates.function_space().ufl_element().degree() > 1:
        raise NotImplementedError("Only straight edged meshes are supported")

    if hasattr(mesh, "_made_from_coordinates") and mesh._made_from_coordinates:
        raise NotImplementedError("Meshes made from coordinate fields are not yet supported")

    if pdim != gdim:
        raise ValueError(f"Mesh geometric dimension {gdim} must match point list dimension {pdim}")

    swarm = _pic_swarm_in_plex(mesh.topology.topology_dm, vertexcoords, fields=[("parentcellnum", 1, IntType), ("refcoord", tdim, RealType)])

    if missing_points_behaviour:

        def compare_arrays(x, y, datatype):
            x, eqx = x
            y, eqy = y
            if not (eqx and eqy):
                return (None, False)
            elif x.shape != y.shape:
                return (None, False)
            else:
                return (x, np.allclose(x, y))

        op = MPI.Op.Create(compare_arrays, commute=True)

        # check all ranks have the same vertexcoords so that check is valid
        # NOTE this operation scales with number of vertices and ranks
        _, allequal = mesh.comm.allreduce((vertexcoords, True), op=op)
        op.Free()
        if not allequal:
            raise ValueError("Cannot check for missing points if different vertices on each MPI rank!")

        # Check for missing points
        nlocal = len(swarm.getField("parentcellnum"))
        swarm.restoreField("parentcellnum")
        nglobal = mesh.comm.allreduce(nlocal, op=MPI.SUM)
        ninput = len(vertexcoords)
        if nglobal < ninput:
            msg = f"{ninput - nglobal} vertices are outside the mesh and have been removed from the VertexOnlyMesh"
            if missing_points_behaviour == 'error':
                raise ValueError(msg)
            elif missing_points_behaviour == 'warn':
                from warnings import warn
                warn(msg)
            else:
                raise ValueError("missing_points_behaviour must be None, 'error' or 'warn'")

    dmcommon.label_pic_parent_cell_info(swarm, mesh)

    # Topology
    topology = VertexOnlyMeshTopology(swarm, mesh.topology, name="swarmmesh", reorder=False)

    # Geometry
    tcell = topology.ufl_cell()
    cell = tcell.reconstruct(geometric_dimension=gdim)
    element = ufl.VectorElement("DG", cell, 0)
    # Create mesh object
    vmesh = MeshGeometry.__new__(MeshGeometry, element)
    vmesh._topology = topology
    vmesh._parent_mesh = mesh

    # Finish the initialisation of mesh topology
    vmesh.topology.init()

    # Initialise mesh geometry
    coordinates_fs = functionspace.VectorFunctionSpace(vmesh.topology, "DG", 0,
                                                       dim=gdim)

    coordinates_data = dmcommon.reordered_coords(swarm, coordinates_fs.dm.getDefaultSection(),
                                                 (vmesh.num_vertices(), gdim))

    coordinates = function.CoordinatelessFunction(coordinates_fs,
                                                  val=coordinates_data,
                                                  name="Coordinates")

    vmesh.__init__(coordinates)

    # Save vertex reference coordinate (within reference cell) in function
    reference_coordinates_fs = functionspace.VectorFunctionSpace(vmesh, "DG", 0, dim=tdim)
    vmesh.reference_coordinates = dmcommon.fill_reference_coordinates_function(function.Function(reference_coordinates_fs))

    return vmesh


def _pic_swarm_in_plex(plex, coords, fields=[]):
    """
    Create a Particle In Cell (PIC) DMSwarm, immersed in a DMPlex
    at given point coordinates.

    This should only by used for dmplexes associated with meshes with
    straight edges. If not, the particles may be placed in the wrong
    cells.

    :arg plex: the DMPlex within with the DMSwarm should be
        immersed.
    :arg coords: an ``ndarray`` of (npoints, coordsdim) shape.
    :kwarg fields: An optional list of named data which can be stored
        for each point in the DMSwarm. The format should be::

        [(fieldname1, blocksize1, dtype1),
          ...,
         (fieldnameN, blocksizeN, dtypeN)]

        For example, the swarm coordinates themselves are stored in a
        field named ``DMSwarmPIC_coor`` which, were it not created
        automatically, would be initialised with
        ``fields = [("DMSwarmPIC_coor", coordsdim, RealType)]``.
        All fields must have the same number of points. For more
        information see `the DMSWARM API reference
        <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DMSWARM/DMSWARM.html>_.
    :return: the immersed DMSwarm

    .. note::

        The created DMSwarm uses the communicator of the input DMPlex.

    .. note::

        In complex mode the "DMSwarmPIC_coor" field is still saved as a
        real number unlike the coordinates of a DMPlex which become
        complex (though usually with zeroed imaginary parts).

    .. note::
        When running in parallel, ``coords`` are strictly confined to
        the local DMPlex cells of that rank. This means that if rank A
        has ``coords`` {X} that are not found in the DMPlex cells of rank
        A but are found in the DMPlex cells of rank B, **and rank B has
        not been supplied with those** ``coords`` then the ``coords`` {X}
        will be lost.

        This can be avoided by either

        #. making sure that all ranks are supplied with the same list of
          ``coords`` or by
        #. ensuring that ``coords`` are already localised for to the
           DMPlex cells of the given rank.

        For more see `this github issue
        <https://github.com/firedrakeproject/firedrake/issues/2178>`_.
    """

    # Check coords
    coords = np.asarray(coords, dtype=RealType)
    _, coordsdim = coords.shape

    # Create a DMSWARM
    swarm = PETSc.DMSwarm().create(comm=plex.comm)

    # Set swarm DM dimension to match DMPlex dimension
    # NB: Unlike a DMPlex, this does not correspond to the topological
    #     dimension of a mesh (which would be 0). In all PETSc examples
    #     the dimension of the DMSwarm is set to match that of the
    #     DMPlex used with swarm.setCellDM
    swarm.setDimension(plex.getDimension())

    # Set coordinates dimension
    swarm.setCoordinateDim(coordsdim)

    # Link to DMPlex cells information for when swarm.migrate() is used
    swarm.setCellDM(plex)

    # Set to Particle In Cell (PIC) type
    swarm.setType(PETSc.DMSwarm.Type.PIC)

    # Register any fields
    for name, size, dtype in fields:
        swarm.registerField(name, size, dtype=dtype)
    swarm.finalizeFieldRegister()

    # Note that no new fields can now be associated with the DMSWARM.

    # Add point coordinates - note we set redundant mode to False
    # because we allow different MPI ranks to be given the overlapping
    # lists of coordinates. The cell DM (`plex`) will then attempt to
    # locate the coordinates within its rank-local sub domain and
    # disregard those which are outside it. See https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DMSWARM/DMSwarmSetPointCoordinates.html
    # for more information. The result is that all DMPlex cells,
    # including ghost cells on distributed meshes, have the relevent PIC
    # coordinates associated with them. The DMPlex cell id associated
    # with each PIC in the DMSwarm is accessed with the `DMSwarm_cellid`
    # field.
    swarm.setPointCoordinates(coords, redundant=False, mode=PETSc.InsertMode.INSERT_VALUES)

    # Remove PICs which have been placed into ghost cells of a distributed DMPlex
    dmcommon.remove_ghosts_pic(swarm, plex)

    # Set the `SF` graph to advertises no shared points (since the halo
    # is now empty) by setting the leaves to an empty list
    sf = swarm.getPointSF()
    nroots = swarm.getLocalSize()
    sf.setGraph(nroots, None, [])
    swarm.setPointSF(sf)

    return swarm


@PETSc.Log.EventDecorator()
def SubDomainData(geometric_expr):
    """Creates a subdomain data object from a boolean-valued UFL expression.

    The result can be attached as the subdomain_data field of a
    :class:`ufl.Measure`. For example:

    .. code-block:: python3

        x = mesh.coordinates
        sd = SubDomainData(x[0] < 0.5)
        assemble(f*dx(subdomain_data=sd))

    """
    import firedrake.functionspace as functionspace
    import firedrake.projection as projection

    # Find domain from expression
    m = geometric_expr.ufl_domain()

    # Find selected cells
    fs = functionspace.FunctionSpace(m, 'DG', 0)
    f = projection.project(ufl.conditional(geometric_expr, 1, 0), fs)

    # Create cell subset
    indices, = np.nonzero(f.dat.data_ro_with_halos > 0.5)
    return op2.Subset(m.cell_set, indices)
