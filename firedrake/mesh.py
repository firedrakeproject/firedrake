import numpy as np
import ctypes
import os
import sys
import ufl
import FIAT
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from ufl.classes import ReferenceGrad
from ufl.domain import extract_unique_domain
import enum
import numbers
import abc

from pyop2 import op2
from pyop2.mpi import (
    MPI, COMM_WORLD, internal_comm, decref, is_pyop2_comm, temp_internal_comm
)
from pyop2.utils import as_tuple, tuplify

import firedrake.cython.dmcommon as dmcommon
import firedrake.cython.extrusion_numbering as extnum
import firedrake.extrusion_utils as eutils
import firedrake.cython.spatialindex as spatialindex
import firedrake.utils as utils
from firedrake.utils import IntType, RealType
from firedrake.logging import info_red
from firedrake.parameters import parameters
from firedrake.petsc import PETSc, OptionsManager
from firedrake.adjoint import MeshGeometryMixin

try:
    import netgen
    from ngsolve import ngs2petsc
except ImportError:
    netgen = None


__all__ = ['Mesh', 'ExtrudedMesh', 'VertexOnlyMesh', 'RelabeledMesh', 'SubDomainData', 'unmarked',
           'DistributedMeshOverlapType', 'DEFAULT_MESH_NAME', 'MeshGeometry', 'MeshTopology', 'AbstractMeshTopology']


_cells = {
    1: {2: "interval"},
    2: {3: "triangle", 4: "quadrilateral"},
    3: {4: "tetrahedron", 6: "hexahedron"}
}


_supported_embedded_cell_types = [ufl.Cell('interval', 2),
                                  ufl.Cell('triangle', 3),
                                  ufl.Cell("quadrilateral", 3),
                                  ufl.TensorProductCell(ufl.Cell('interval'), ufl.Cell('interval'), geometric_dimension=3)]


unmarked = -1
"""A mesh marker that selects all entities that are not explicitly marked."""

DEFAULT_MESH_NAME = "_".join(["firedrake", "default"])
"""The default name of the mesh."""


def _generate_default_mesh_coordinates_name(name):
    """Generate the default mesh coordinates name from the mesh name.

    :arg name: the mesh name.
    :returns: the default mesh coordinates name.
    """
    return "_".join([name, "coordinates"])


def _generate_default_mesh_topology_name(name):
    """Generate the default mesh topology name from the mesh name.

    :arg name: the mesh name.
    :returns: the default mesh topology name.
    """
    return "_".join([name, "topology"])


def _generate_default_mesh_topology_distribution_name(comm_size, dist_param):
    """Generate the default mesh topology permutation name.

    :arg comm_size: the size of comm.
    :arg dist_param: the distribution_parameter dict.
    :returns: the default mesh topology distribution name.
    """
    return "_".join(["firedrake", "default",
                     str(comm_size),
                     str(dist_param["partition"]).replace(' ', ''),
                     str(dist_param["partitioner_type"]),
                     "(" + dist_param["overlap_type"][0].name + "," + str(dist_param["overlap_type"][1]) + ")"])


def _generate_default_mesh_topology_permutation_name(reorder):
    """Generate the default mesh topology permutation name.

    :arg reorder: the flag indicating if the reordering happened or not.
    :returns: the default mesh topology permutation name.
    """
    return "_".join(["firedrake", "default", str(reorder)])


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
    def __init__(self, mesh, facets, classes, kind, facet_cell, local_facet_number,
                 unique_markers=None):

        self.mesh = mesh
        self.facets = facets
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
                unmarked_points = self._collect_unmarked_points(all_integer_subdomain_ids)
                _, indices, _ = np.intersect1d(self.facets, unmarked_points, return_indices=True)
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
        try:
            return self._subsets[markers]
        except KeyError:
            # check that the given markers are valid
            if len(set(markers).difference(valid_markers)) > 0:
                invalid = set(markers).difference(valid_markers)
                raise LookupError("{0} are not a valid markers (not in {1})".format(invalid, self.unique_markers))

            # build a list of indices corresponding to the subsets selected by
            # markers
            marked_points_list = []
            for i in markers:
                if i == unmarked:
                    _markers = self.mesh.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices
                    # Can exclude points labeled with i\in markers here,
                    # as they will be included in the below anyway.
                    marked_points_list.append(self._collect_unmarked_points([_i for _i in _markers if _i not in markers]))
                else:
                    if self.mesh.topology_dm.getStratumSize(dmcommon.FACE_SETS_LABEL, i):
                        marked_points_list.append(self.mesh.topology_dm.getStratumIS(dmcommon.FACE_SETS_LABEL, i).indices)
            if marked_points_list:
                _, indices, _ = np.intersect1d(self.facets, np.concatenate(marked_points_list), return_indices=True)
                return self._subsets.setdefault(markers, op2.Subset(self.set, indices))
            else:
                return self._subsets.setdefault(markers, self._null_subset)

    def _collect_unmarked_points(self, markers):
        """Collect points that are not marked by markers."""
        plex = self.mesh.topology_dm
        indices_list = []
        for i in markers:
            if plex.getStratumSize(dmcommon.FACE_SETS_LABEL, i):
                indices_list.append(plex.getStratumIS(dmcommon.FACE_SETS_LABEL, i).indices)
        if indices_list:
            return np.setdiff1d(self.facets, np.concatenate(indices_list))
        else:
            return self.facets

    @utils.cached_property
    def facet_cell_map(self):
        """Map from facets to cells."""
        return op2.Map(self.set, self.mesh.cell_set, self._rank, self.facet_cell,
                       "facet_to_cell_map")


@PETSc.Log.EventDecorator()
def _from_netgen(ngmesh, comm=None):
    """
    Create a DMPlex from an Netgen mesh

    :arg ngmesh: Netgen Mesh
    TODO: Right now we construct Netgen mesh on a single worker, load it in Firedrake
    and then distribute. We should find a way to take advantage of the fact that
    Netgen can act as a parallel mesher.
    """
    meshMap = ngs2petsc.DMPlexMapping(ngmesh)
    return meshMap.plex


@PETSc.Log.EventDecorator()
def _from_gmsh(filename, comm=None):
    """Read a Gmsh .msh file from `filename`.

    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    comm = comm or COMM_WORLD
    # check the filetype of the gmsh file
    filetype = None
    if comm.rank == 0:
        with open(filename, 'rb') as fid:
            header = fid.readline().rstrip(b'\n\r')
            version = fid.readline().rstrip(b'\n\r')
        assert header == b'$MeshFormat'
        if version.split(b' ')[1] == b'1':
            filetype = "binary"
        else:
            filetype = "ascii"
    filetype = comm.bcast(filetype, root=0)
    # Create a read-only PETSc.Viewer
    gmsh_viewer = PETSc.Viewer().create(comm=comm)
    gmsh_viewer.setType(filetype)
    gmsh_viewer.setFileMode("r")
    gmsh_viewer.setFileName(filename)
    gmsh_plex = PETSc.DMPlex().createGmsh(gmsh_viewer, comm=comm)

    return gmsh_plex


@PETSc.Log.EventDecorator()
def _from_exodus(filename, comm):
    """Read an Exodus .e or .exo file from `filename`.

    :arg comm: communicator to build the mesh on.
    """
    plex = PETSc.DMPlex().createExodusFromFile(filename, comm=comm)

    return plex


@PETSc.Log.EventDecorator()
def _from_cgns(filename, comm):
    """Read a CGNS .cgns file from `filename`.

    :arg comm: communicator to build the mesh on.
    """
    plex = PETSc.DMPlex().createCGNSFromFile(filename, comm=comm)
    return plex


@PETSc.Log.EventDecorator()
def _from_triangle(filename, dim, comm):
    """Read a set of triangle mesh files from `filename`.

    :arg dim: The embedding dimension.
    :arg comm: communicator to build the mesh on.
    """
    basename, ext = os.path.splitext(filename)

    with temp_internal_comm(comm) as icomm:
        if icomm.rank == 0:
            try:
                facetfile = open(basename+".face")
                tdim = 3
            except FileNotFoundError:
                try:
                    facetfile = open(basename+".edge")
                    tdim = 2
                except FileNotFoundError:
                    facetfile = None
                    tdim = 1
            if dim is None:
                dim = tdim
            icomm.bcast(tdim, root=0)

            with open(basename+".node") as nodefile:
                header = np.fromfile(nodefile, dtype=np.int32, count=2, sep=' ')
                nodecount = header[0]
                nodedim = header[1]
                assert nodedim == dim
                coordinates = np.loadtxt(nodefile, usecols=list(range(1, dim+1)), skiprows=1, dtype=np.double)
                assert nodecount == coordinates.shape[0]

            with open(basename+".ele") as elefile:
                header = np.fromfile(elefile, dtype=np.int32, count=2, sep=' ')
                elecount = header[0]
                eledim = header[1]
                eles = np.loadtxt(elefile, usecols=list(range(1, eledim+1)), dtype=np.int32, skiprows=1)
                assert elecount == eles.shape[0]

            cells = list(map(lambda c: c-1, eles))
        else:
            tdim = icomm.bcast(None, root=0)
            cells = None
            coordinates = None
        plex = plex_from_cell_list(tdim, cells, coordinates, icomm)

        # Apply boundary IDs
        if icomm.rank == 0:
            facets = None
            try:
                header = np.fromfile(facetfile, dtype=np.int32, count=2, sep=' ')
                edgecount = header[0]
                facets = np.loadtxt(facetfile, usecols=list(range(1, tdim+2)), dtype=np.int32, skiprows=0)
                assert edgecount == facets.shape[0]
            finally:
                facetfile.close()

            if facets is not None:
                vStart, vEnd = plex.getDepthStratum(0)   # vertices
                for facet in facets:
                    bid = facet[-1]
                    vertices = list(map(lambda v: v + vStart - 1, facet[:-1]))
                    join = plex.getJoin(vertices)
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, join[0], bid)

    return plex


def plex_from_cell_list(dim, cells, coords, comm, name=None):
    """
    Create a DMPlex from a list of cells and coords.
    (Public interface to `_from_cell_list()`)

    :arg dim: The topological dimension of the mesh
    :arg cells: The vertices of each cell
    :arg coords: The coordinates of each vertex
    :arg comm: communicator to build the mesh on. Must be a PyOP2 internal communicator
    :kwarg name: name of the plex
    """
    with temp_internal_comm(comm) as icomm:
        # These types are /correct/, DMPlexCreateFromCellList wants int
        # and double (not PetscInt, PetscReal).
        if comm.rank == 0:
            cells = np.asarray(cells, dtype=np.int32)
            coords = np.asarray(coords, dtype=np.double)
            comm.bcast(cells.shape, root=0)
            comm.bcast(coords.shape, root=0)
            # Provide the actual data on rank 0.
            plex = PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=icomm)
        else:
            cell_shape = list(comm.bcast(None, root=0))
            coord_shape = list(comm.bcast(None, root=0))
            cell_shape[0] = 0
            coord_shape[0] = 0
            # Provide empty plex on other ranks
            # A subsequent call to plex.distribute() takes care of parallel partitioning
            plex = PETSc.DMPlex().createFromCellList(dim,
                                                     np.zeros(cell_shape, dtype=np.int32),
                                                     np.zeros(coord_shape, dtype=np.double),
                                                     comm=icomm)
    if name is not None:
        plex.setName(name)
    return plex


@PETSc.Log.EventDecorator()
def _from_cell_list(dim, cells, coords, comm, name=None):
    """
    Create a DMPlex from a list of cells and coords.
    This function remains for backward compatibility, but will be deprecated after 01/06/2023

    :arg dim: The topological dimension of the mesh
    :arg cells: The vertices of each cell
    :arg coords: The coordinates of each vertex
    :arg comm: communicator to build the mesh on. Must be a PyOP2 internal communicator
    :kwarg name: name of the plex
    """
    import warnings
    warnings.warn(
        "Private function `_from_cell_list` will be deprecated after 01/06/2023;"
        "use public fuction `plex_from_cell_list()` instead.",
        DeprecationWarning
    )
    assert is_pyop2_comm(comm)

    # These types are /correct/, DMPlexCreateFromCellList wants int
    # and double (not PetscInt, PetscReal).
    if comm.rank == 0:
        cells = np.asarray(cells, dtype=np.int32)
        coords = np.asarray(coords, dtype=np.double)
        comm.bcast(cells.shape, root=0)
        comm.bcast(coords.shape, root=0)
        # Provide the actual data on rank 0.
        plex = PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=comm)
    else:
        cell_shape = list(comm.bcast(None, root=0))
        coord_shape = list(comm.bcast(None, root=0))
        cell_shape[0] = 0
        coord_shape[0] = 0
        # Provide empty plex on other ranks
        # A subsequent call to plex.distribute() takes care of parallel partitioning
        plex = PETSc.DMPlex().createFromCellList(dim,
                                                 np.zeros(cell_shape, dtype=np.int32),
                                                 np.zeros(coord_shape, dtype=np.double),
                                                 comm=comm)
    if name is not None:
        plex.setName(name)
    return plex


class AbstractMeshTopology(object, metaclass=abc.ABCMeta):
    """A representation of an abstract mesh topology without a concrete
        PETSc DM implementation"""

    def __init__(self, name, tolerance=1.0):
        """Initialise an abstract mesh topology.

        :arg name: name of the mesh
        :kwarg tolerance: The relative tolerance (i.e. as defined on the
            reference cell) for the distance a point can be from a cell and
            still be considered to be in the cell. Note that
            this tolerance uses an L1 distance (aka 'manhatten', 'taxicab' or
            rectilinear distance) so will scale with the dimension of the mesh.
        """

        utils._init()

        self.name = name
        if not isinstance(tolerance, numbers.Number):
            raise TypeError("tolerance must be a number")
        self._tolerance = tolerance

        self.topology_dm = None
        r"The PETSc DM representation of the mesh topology."

        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)

        # Cell subsets for integration over subregions
        self._subsets = {}

        self._grown_halos = False

        # A set of weakrefs to meshes that are explicitly labelled as being
        # parallel-compatible for interpolation/projection/supermeshing
        # To set, do e.g.
        # target_mesh._parallel_compatible = {weakref.ref(source_mesh)}
        self._parallel_compatible = None

    def __del__(self):
        if hasattr(self, "_comm"):
            decref(self._comm)

    layers = None
    """No layers on unstructured mesh"""

    variable_layers = False
    """No variable layers on unstructured mesh"""

    @property
    def comm(self):
        return self.user_comm

    def mpi_comm(self):
        """The MPI communicator this mesh is built on (an mpi4py object)."""
        return self.comm

    @PETSc.Log.EventDecorator("CreateMesh")
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
    def _topology_dm(self):
        """Alias of topology_dm"""
        from warnings import warn
        warn("_topology_dm is deprecated (use topology_dm instead)", DeprecationWarning, stacklevel=2)
        return self.topology_dm

    def ufl_cell(self):
        """The UFL :class:`~ufl.classes.Cell` associated with the mesh.

        .. note::

            By convention, the UFL cells which specifically
            represent a mesh topology have geometric dimension equal their
            topological dimension. This is true even for immersed manifold
            meshes.

        """
        return self._ufl_mesh.ufl_cell()

    def ufl_mesh(self):
        """The UFL :class:`~ufl.classes.Mesh` associated with the mesh.

        .. note::

            By convention, the UFL cells which specifically
            represent a mesh topology have geometric dimension equal their
            topological dimension. This convention will be reflected in this
            UFL mesh and is true even for immersed manifold meshes.

        """
        return self._ufl_mesh

    @property
    @abc.abstractmethod
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        pass

    @property
    @abc.abstractmethod
    def entity_orientations(self):
        """2D array of entity orientations

        `entity_orientations` has the same shape as `cell_closure`.
        Each row of this array contains orientations of the entities
        in the closure of the associated cell. Here, for each cell in the mesh,
        orientation of an entity, say e, encodes how the the canonical
        representation of the entity defined by Cone(e) compares to
        that of the associated entity in the reference FInAT (FIAT) cell. (Note
        that `cell_closure` defines how each cell in the mesh is mapped to
        the FInAT (FIAT) reference cell and each entity of the FInAT (FIAT)
        reference cell has a canonical representation based on the entity ids of
        the lower dimensional entities.) Orientations of vertices are always 0.
        See ``FIAT.reference_element.Simplex`` and
        ``FIAT.reference_element.UFCQuadrilateral`` for example computations
        of orientations.
        """
        pass

    @abc.abstractmethod
    def _facets(self, kind):
        pass

    @property
    @abc.abstractmethod
    def exterior_facets(self):
        pass

    @property
    @abc.abstractmethod
    def interior_facets(self):
        pass

    @property
    @abc.abstractmethod
    def cell_to_facets(self):
        """Returns a :class:`pyop2.types.dat.Dat` that maps from a cell index to the local
        facet types on each cell, including the relevant subdomain markers.

        The `i`-th local facet on a cell with index `c` has data
        `cell_facet[c][i]`. The local facet is exterior if
        `cell_facet[c][i][0] == 0`, and interior if the value is `1`.
        The value `cell_facet[c][i][1]` returns the subdomain marker of the
        facet.
        """
        pass

    def create_section(self, nodes_per_entity, real_tensorproduct=False, block_size=1):
        """Create a PETSc Section describing a function space.

        :arg nodes_per_entity: number of function space nodes per topological entity.
        :arg real_tensorproduct: If True, assume extruded space is actually Foo x Real.
        :arg block_size: The integer by which nodes_per_entity is uniformly multiplied
            to get the true data layout.
        :returns: a new PETSc Section.
        """
        return dmcommon.create_section(self, nodes_per_entity, on_base=real_tensorproduct, block_size=block_size)

    def node_classes(self, nodes_per_entity, real_tensorproduct=False):
        """Compute node classes given nodes per entity.

        :arg nodes_per_entity: number of function space nodes per topological entity.
        :returns: the number of nodes in each of core, owned, and ghost classes.
        """
        return tuple(np.dot(nodes_per_entity, self._entity_classes))

    def make_cell_node_list(self, global_numbering, entity_dofs, entity_permutations, offsets):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg entity_dofs: FInAT element entity DoFs
        :arg entity_permutations: FInAT element entity permutations
        :arg offsets: layer offsets for each entity dof (may be None).
        """
        return dmcommon.get_cell_nodes(self, global_numbering,
                                       entity_dofs, entity_permutations, offsets)

    def make_dofs_per_plex_entity(self, entity_dofs):
        """Returns the number of DoFs per plex entity for each stratum,
        i.e. [#dofs / plex vertices, #dofs / plex edges, ...].

        :arg entity_dofs: FInAT element entity DoFs
        """
        return [len(entity_dofs[d][0]) for d in sorted(entity_dofs)]

    def make_offset(self, entity_dofs, ndofs, real_tensorproduct=False):
        """Returns None (only for extruded use)."""
        return None

    def _order_data_by_cell_index(self, column_list, cell_data):
        return cell_data[column_list]

    @abc.abstractmethod
    def num_cells(self):
        pass

    @abc.abstractmethod
    def num_facets(self):
        pass

    @abc.abstractmethod
    def num_faces(self):
        pass

    @abc.abstractmethod
    def num_edges(self):
        pass

    @abc.abstractmethod
    def num_vertices(self):
        pass

    @abc.abstractmethod
    def num_entities(self, d):
        pass

    def size(self, d):
        return self.num_entities(d)

    def cell_dimension(self):
        """Returns the cell dimension."""
        return self.ufl_cell().topological_dimension()

    def facet_dimension(self):
        """Returns the facet dimension."""
        # Facets have co-dimension 1
        return self.ufl_cell().topological_dimension() - 1

    @property
    @abc.abstractmethod
    def cell_set(self):
        pass

    @PETSc.Log.EventDecorator()
    def cell_subset(self, subdomain_id, all_integer_subdomain_ids=None):
        """Return a subset over cells with the given subdomain_id.

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

         :returns: A :class:`pyop2.types.set.Subset` for iteration.
        """
        if subdomain_id == "everywhere":
            return self.cell_set
        if subdomain_id == "otherwise":
            if all_integer_subdomain_ids is None:
                return self.cell_set
            key = ("otherwise", ) + all_integer_subdomain_ids
        else:
            key = subdomain_id
        try:
            return self._subsets[key]
        except KeyError:
            if subdomain_id == "otherwise":
                ids = tuple(dmcommon.get_cell_markers(self.topology_dm,
                                                      self._cell_numbering,
                                                      sid)
                            for sid in all_integer_subdomain_ids)
                to_remove = np.unique(np.concatenate(ids))
                indices = np.arange(self.cell_set.total_size, dtype=IntType)
                indices = np.delete(indices, to_remove)
            else:
                indices = dmcommon.get_cell_markers(self.topology_dm,
                                                    self._cell_numbering,
                                                    subdomain_id)
            return self._subsets.setdefault(key, op2.Subset(self.cell_set, indices))

    @PETSc.Log.EventDecorator()
    def measure_set(self, integral_type, subdomain_id,
                    all_integer_subdomain_ids=None):
        """Return an iteration set appropriate for the requested integral type.

        :arg integral_type: The type of the integral (should be a valid UFL measure).
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
             subdomains 1 and 2.  This should be a dict mapping
             ``integral_type`` to the explicitly enumerated subdomain ids.

         :returns: A :class:`pyop2.types.set.Subset` for iteration.
        """
        if all_integer_subdomain_ids is not None:
            all_integer_subdomain_ids = all_integer_subdomain_ids.get(integral_type, None)
        if integral_type == "cell":
            return self.cell_subset(subdomain_id, all_integer_subdomain_ids)
        elif integral_type in ("exterior_facet", "exterior_facet_vert",
                               "exterior_facet_top", "exterior_facet_bottom"):
            return self.exterior_facets.measure_set(integral_type, subdomain_id,
                                                    all_integer_subdomain_ids)
        elif integral_type in ("interior_facet", "interior_facet_vert",
                               "interior_facet_horiz"):
            return self.interior_facets.measure_set(integral_type, subdomain_id,
                                                    all_integer_subdomain_ids)
        else:
            raise ValueError("Unknown integral type '%s'" % integral_type)

    @property
    def tolerance(self):
        """The relative tolerance (i.e. as defined on the reference cell) for
        the distance a point can be from a cell and still be considered to be
        in the cell.

        Should always be set via the ``MeshGeometry.tolerance`` to ensure
        the spatial index is updated as necessary.
        """
        return self._tolerance

    @abc.abstractmethod
    def mark_entities(self, tf, label_name, label_value):
        """Mark selected entities.

        :arg tf: The :class:`.CoordinatelessFunction` object that marks
            selected entities as 1. f.function_space().ufl_element()
            must be "DP" or "DQ" (degree 0) to mark cell entities and
            "P" (degree 1) in 1D or "HDiv Trace" (degree 0) in 2D or 3D
            to mark facet entities.
        :arg label_name: The name of the label to store entity selections.
        :arg lable_value: The value used in the label.

        All entities must live on the same topological dimension. Currently,
        one can only mark cell or facet entities.
        """
        pass

    @utils.cached_property
    def extruded_periodic(self):
        return self.cell_set._extruded_periodic


class MeshTopology(AbstractMeshTopology):
    """A representation of mesh topology implemented on a PETSc DMPlex."""

    @PETSc.Log.EventDecorator("CreateMesh")
    def __init__(self, plex, name, reorder, distribution_parameters, sfXB=None, perm_is=None, distribution_name=None, permutation_name=None, comm=COMM_WORLD, tolerance=1.0):
        """Half-initialise a mesh topology.

        :arg plex: PETSc DMPlex representing the mesh topology
        :arg name: name of the mesh
        :arg reorder: whether to reorder the mesh (bool)
        :arg distribution_parameters: options controlling mesh
            distribution, see :func:`Mesh` for details.
        :kwarg sfXB: PETSc PetscSF that pushes forward the global point number
            slab :math:`[0, NX)` to input (naive) plex (only significant when
            the mesh topology is loaded from file and only passed from inside
            :class:`~.CheckpointFile`).
        :kwarg perm_is: PETSc IS that is used as `_plex_renumbering`; only
            makes sense if we know the exact parallel distribution of `plex`
            at the time of mesh topology construction like when we load mesh
            along with its distribution. If given, `reorder` param will be ignored.
        :kwarg distribution_name: name of the parallel distribution;
            if `None`, automatically generated.
        :kwarg permutation_name: name of the entity permutation (reordering);
            if `None`, automatically generated.
        :kwarg comm: MPI communicator
        :kwarg tolerance: The relative tolerance (i.e. as defined on the
            reference cell) for the distance a point can be from a cell and
            still be considered to be in the cell. Note that
            this tolerance uses an L1 distance (aka 'manhatten', 'taxicab' or
            rectilinear distance) so will scale with the dimension of the mesh.
        """

        super().__init__(name, tolerance=tolerance)

        self._distribution_parameters = distribution_parameters.copy()
        # Do some validation of the input mesh
        distribute = distribution_parameters.get("partition")
        if distribute is None:
            distribute = True
        self._distribution_parameters["partition"] = distribute
        partitioner_type = distribution_parameters.get("partitioner_type")
        self._distribution_parameters["partitioner_type"] = partitioner_type
        overlap_type, overlap = distribution_parameters.get("overlap_type",
                                                            (DistributedMeshOverlapType.FACET, 1))

        self._distribution_parameters["overlap_type"] = (overlap_type, overlap)
        if overlap < 0:
            raise ValueError("Overlap depth must be >= 0")
        if overlap_type == DistributedMeshOverlapType.NONE:
            def add_overlap():
                pass
            if overlap > 0:
                raise ValueError("Can't have NONE overlap with overlap > 0")
        elif overlap_type == DistributedMeshOverlapType.FACET:
            def add_overlap():
                dmcommon.set_adjacency_callback(self.topology_dm)
                original_name = self.topology_dm.getName()
                sfBC = self.topology_dm.distributeOverlap(overlap)
                self.topology_dm.setName(original_name)
                self.sfBC = self.sfBC.compose(sfBC) if self.sfBC else sfBC
                dmcommon.clear_adjacency_callback(self.topology_dm)
                self._grown_halos = True
        elif overlap_type == DistributedMeshOverlapType.VERTEX:
            def add_overlap():
                # Default is FEM (vertex star) adjacency.
                original_name = self.topology_dm.getName()
                sfBC = self.topology_dm.distributeOverlap(overlap)
                self.topology_dm.setName(original_name)
                self.sfBC = self.sfBC.compose(sfBC) if self.sfBC else sfBC
                self._grown_halos = True
        else:
            raise ValueError("Unknown overlap type %r" % overlap_type)

        dmcommon.validate_mesh(plex)
        # Currently, we do the distribution manually, so
        # disable auto distribution.
        plex.distributeSetDefault(False)
        # Similarly, disable auto plex reordering.
        plex.reorderSetDefault(PETSc.DMPlex.ReorderDefaultFlag.FALSE)
        plex.setFromOptions()

        self.topology_dm = plex
        r"The PETSc DM representation of the mesh topology."
        self.sfBC = None
        r"The PETSc SF that pushes the input (naive) plex to current (good) plex."
        self.sfXB = sfXB
        r"The PETSc SF that pushes the global point number slab [0, NX) to input (naive) plex."

        # User comm
        self.user_comm = comm
        # Internal comm
        self._comm = internal_comm(self.user_comm)

        # Mark exterior and interior facets
        # Note.  This must come before distribution, because otherwise
        # DMPlex will consider facets on the domain boundary to be
        # exterior, which is wrong.
        label_boundary = not plex.isDistributed()
        dmcommon.label_facets(plex, label_boundary=label_boundary)

        # Distribute/redistribute the dm to all ranks
        if self.comm.size > 1 and distribute:
            # We distribute with overlap zero, in case we're going to
            # refine this mesh in parallel.  Later, when we actually use
            # it, we grow the halo.
            self.set_partitioner(distribute, partitioner_type)
            self._distribution_parameters["partitioner_type"] = self.get_partitioner().getType()
            original_name = plex.getName()
            sfBC = plex.distribute(overlap=0)
            plex.setName(original_name)
            self.sfBC = sfBC
            # plex carries a new dm after distribute, which
            # does not inherit partitioner from the old dm.
            # It probably makes sense as chaco does not work
            # once distributed.

        tdim = plex.getDimension()

        # Allow empty local meshes on a process
        cStart, cEnd = plex.getHeightStratum(0)  # cells
        if cStart == cEnd:
            nfacets = -1
        else:
            nfacets = plex.getConeSize(cStart)

        # TODO: this needs to be updated for mixed-cell meshes.
        nfacets = self._comm.allreduce(nfacets, op=MPI.MAX)

        # Note that the geometric dimension of the cell is not set here
        # despite it being a property of a UFL cell. It will default to
        # equal the topological dimension.
        # Firedrake mesh topologies, by convention, which specifically
        # represent a mesh topology (as here) have geometric dimension
        # equal their topological dimension. This is reflected in the
        # corresponding UFL mesh.
        cell = ufl.Cell(_cells[tdim][nfacets])
        self._ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension()))
        # Set/Generate names to be used when checkpointing.
        self._distribution_name = distribution_name or _generate_default_mesh_topology_distribution_name(self.topology_dm.comm.size, self._distribution_parameters)
        self._permutation_name = permutation_name or _generate_default_mesh_topology_permutation_name(reorder)

        def callback(self):
            """Finish initialisation."""
            del self._callback
            if self.comm.size > 1:
                add_overlap()
            if self.sfXB is not None:
                self.sfXC = sfXB.compose(self.sfBC) if self.sfBC else self.sfXB
            dmcommon.complete_facet_labels(self.topology_dm)
            # TODO: Allow users to set distribution name if they want to save
            #       conceptually the same mesh but with different distributions,
            #       e.g., those generated by different partitioners.
            #       This currently does not make sense since those mesh instances
            #       of different distributions in general have different global
            #       point numbers (so they must be saved under different mesh names
            #       even though they are conceptually the same).
            # The name set here almost uniquely identifies a distribution, but
            # there is no gurantee that it really does or it continues to do so
            # there are lots of parameters that can change distributions.
            # Thus, when using CheckpointFile, it is recommended that the user set
            # distribution_name explicitly.
            if reorder:
                with PETSc.Log.Event("Mesh: reorder"):
                    old_to_new = self.topology_dm.getOrdering(PETSc.Mat.OrderingType.RCM).indices
                    reordering = np.empty_like(old_to_new)
                    reordering[old_to_new] = np.arange(old_to_new.size, dtype=old_to_new.dtype)
            else:
                # No reordering
                reordering = None
            self._did_reordering = bool(reorder)
            # Mark OP2 entities and derive the resulting Plex renumbering
            with PETSc.Log.Event("Mesh: numbering"):
                dmcommon.mark_entity_classes(self.topology_dm)
                self._entity_classes = dmcommon.get_entity_classes(self.topology_dm).astype(int)
                if perm_is:
                    self._plex_renumbering = perm_is
                else:
                    self._plex_renumbering = dmcommon.plex_renumbering(self.topology_dm,
                                                                       self._entity_classes,
                                                                       reordering)
                # Derive a cell numbering from the Plex renumbering
                entity_dofs = np.zeros(tdim+1, dtype=IntType)
                entity_dofs[-1] = 1
                self._cell_numbering = self.create_section(entity_dofs)
                entity_dofs[:] = 0
                entity_dofs[0] = 1
                self._vertex_numbering = self.create_section(entity_dofs)
                entity_dofs[:] = 0
                entity_dofs[-2] = 1
                facet_numbering = self.create_section(entity_dofs)
                self._facet_ordering = dmcommon.get_facet_ordering(self.topology_dm, facet_numbering)
        self._callback = callback

    def __del__(self):
        if hasattr(self, "_comm"):
            decref(self._comm)

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        plex = self.topology_dm
        tdim = plex.getDimension()

        # Cell numbering and global vertex numbering
        cell_numbering = self._cell_numbering
        vertex_numbering = self._vertex_numbering.createGlobalSection(plex.getPointSF())

        cell = self.ufl_cell()
        assert tdim == cell.topological_dimension()
        if cell.is_simplex():
            topology = FIAT.ufc_cell(cell).get_topology()
            entity_per_cell = np.zeros(len(topology), dtype=IntType)
            for d, ents in topology.items():
                entity_per_cell[d] = len(ents)

            return dmcommon.closure_ordering(plex, vertex_numbering,
                                             cell_numbering, entity_per_cell)

        elif cell.cellname() == "quadrilateral":
            from firedrake_citations import Citations
            Citations().register("Homolya2016")
            Citations().register("McRae2016")
            # Quadrilateral mesh
            cell_ranks = dmcommon.get_cell_remote_ranks(plex)

            facet_orientations = dmcommon.quadrilateral_facet_orientations(
                plex, vertex_numbering, cell_ranks)

            cell_orientations = dmcommon.orientations_facet2cell(
                plex, vertex_numbering, cell_ranks,
                facet_orientations, cell_numbering)

            dmcommon.exchange_cell_orientations(plex,
                                                cell_numbering,
                                                cell_orientations)

            return dmcommon.quadrilateral_closure_ordering(
                plex, vertex_numbering, cell_numbering, cell_orientations)
        elif cell.cellname() == "hexahedron":
            # TODO: Should change and use create_cell_closure() for all cell types.
            topology = FIAT.ufc_cell(cell).get_topology()
            closureSize = sum([len(ents) for _, ents in topology.items()])
            return dmcommon.create_cell_closure(plex, cell_numbering, closureSize)
        else:
            raise NotImplementedError("Cell type '%s' not supported." % cell)

    @utils.cached_property
    def entity_orientations(self):
        return dmcommon.entity_orientations(self, self.cell_closure)

    @PETSc.Log.EventDecorator()
    def _facets(self, kind):
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)

        dm = self.topology_dm
        facets, classes = dmcommon.get_facets_by_class(dm, (kind + "_facets"),
                                                       self._facet_ordering)
        label = dmcommon.FACE_SETS_LABEL
        if dm.hasLabel(label):
            from mpi4py import MPI
            local_markers = set(dm.getLabelIdIS(label).indices)

            def merge_ids(x, y, datatype):
                return x.union(y)

            op = MPI.Op.Create(merge_ids, commute=True)

            unique_markers = np.asarray(sorted(self._comm.allreduce(local_markers, op=op)),
                                        dtype=IntType)
            op.Free()
        else:
            unique_markers = None

        local_facet_number, facet_cell = \
            dmcommon.facet_numbering(dm, kind, facets,
                                     self._cell_numbering,
                                     self.cell_closure)

        point2facetnumber = np.full(facets.max(initial=0)+1, -1, dtype=IntType)
        point2facetnumber[facets] = np.arange(len(facets), dtype=IntType)
        obj = _Facets(self, facets, classes, kind,
                      facet_cell, local_facet_number,
                      unique_markers=unique_markers)
        obj.point2facetnumber = point2facetnumber
        return obj

    @utils.cached_property
    def exterior_facets(self):
        return self._facets("exterior")

    @utils.cached_property
    def interior_facets(self):
        return self._facets("interior")

    @utils.cached_property
    def cell_to_facets(self):
        """Returns a :class:`pyop2.types.dat.Dat` that maps from a cell index to the local
        facet types on each cell, including the relevant subdomain markers.

        The `i`-th local facet on a cell with index `c` has data
        `cell_facet[c][i]`. The local facet is exterior if
        `cell_facet[c][i][0] == 0`, and interior if the value is `1`.
        The value `cell_facet[c][i][1]` returns the subdomain marker of the
        facet.
        """
        cell_facets = dmcommon.cell_facet_labeling(self.topology_dm,
                                                   self._cell_numbering,
                                                   self.cell_closure)
        if isinstance(self.cell_set, op2.ExtrudedSet):
            dataset = op2.DataSet(self.cell_set.parent, dim=cell_facets.shape[1:])
        else:
            dataset = op2.DataSet(self.cell_set, dim=cell_facets.shape[1:])
        return op2.Dat(dataset, cell_facets, dtype=cell_facets.dtype,
                       name="cell-to-local-facet-dat")

    def num_cells(self):
        cStart, cEnd = self.topology_dm.getHeightStratum(0)
        return cEnd - cStart

    def num_facets(self):
        fStart, fEnd = self.topology_dm.getHeightStratum(1)
        return fEnd - fStart

    def num_faces(self):
        fStart, fEnd = self.topology_dm.getDepthStratum(2)
        return fEnd - fStart

    def num_edges(self):
        eStart, eEnd = self.topology_dm.getDepthStratum(1)
        return eEnd - eStart

    def num_vertices(self):
        vStart, vEnd = self.topology_dm.getDepthStratum(0)
        return vEnd - vStart

    def num_entities(self, d):
        eStart, eEnd = self.topology_dm.getDepthStratum(d)
        return eEnd - eStart

    @utils.cached_property
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "Cells", comm=self._comm)

    @PETSc.Log.EventDecorator()
    def set_partitioner(self, distribute, partitioner_type=None):
        """Set partitioner for (re)distributing underlying plex over comm.

        :arg distribute: Boolean or (sizes, points)-tuple.  If (sizes, point)-
            tuple is given, it is used to set shell partition. If Boolean, no-op.
        :kwarg partitioner_type: Partitioner to be used: "chaco", "ptscotch", "parmetis",
            "shell", or `None` (unspecified). Ignored if the distribute parameter
            specifies the distribution.
        """
        from firedrake_configuration import get_config
        plex = self.topology_dm
        partitioner = plex.getPartitioner()
        if type(distribute) is bool:
            if partitioner_type:
                if partitioner_type not in ["chaco", "ptscotch", "parmetis"]:
                    raise ValueError("Unexpected partitioner_type %s" % partitioner_type)
                if partitioner_type == "chaco":
                    if IntType.itemsize == 8:
                        raise ValueError("Unable to use 'chaco': 'chaco' is 32 bit only, "
                                         "but your Integer is %d bit." % IntType.itemsize * 8)
                    if plex.isDistributed():
                        raise ValueError("Unable to use 'chaco': 'chaco' is a serial "
                                         "patitioner, but the mesh is distributed.")
                if partitioner_type == "parmetis":
                    if not get_config().get("options", {}).get("with_parmetis", False):
                        raise ValueError("Unable to use 'parmetis': Firedrake is not "
                                         "installed with 'parmetis'.")
            else:
                if IntType.itemsize == 8 or plex.isDistributed():
                    # Default to PTSCOTCH on 64bit ints (Chaco is 32 bit int only).
                    # Chaco does not work on distributed meshes.
                    if get_config().get("options", {}).get("with_parmetis", False):
                        partitioner_type = "parmetis"
                    else:
                        partitioner_type = "ptscotch"
                else:
                    partitioner_type = "chaco"
            partitioner.setType({"chaco": partitioner.Type.CHACO,
                                 "ptscotch": partitioner.Type.PTSCOTCH,
                                 "parmetis": partitioner.Type.PARMETIS}[partitioner_type])
        else:
            sizes, points = distribute
            partitioner.setType(partitioner.Type.SHELL)
            partitioner.setShellPartition(self.comm.size, sizes, points)
        # Command line option `-petscpartitioner_type <type>` overrides.
        partitioner.setFromOptions()

    @PETSc.Log.EventDecorator()
    def get_partitioner(self):
        """Get partitioner actually used for (re)distributing underlying plex over comm."""
        return self.topology_dm.getPartitioner()

    def mark_entities(self, tf, label_name, label_value):
        import firedrake.function as function

        if label_name in (dmcommon.CELL_SETS_LABEL,
                          dmcommon.FACE_SETS_LABEL,
                          "Vertex Sets",
                          "depth",
                          "celltype",
                          "ghost",
                          "exterior_facets",
                          "interior_facets",
                          "pyop2_core",
                          "pyop2_owned",
                          "pyop2_ghost"):
            raise ValueError(f"Label name {label_name} is reserved")
        if not isinstance(tf, function.CoordinatelessFunction):
            raise TypeError(f"tf must be an instance of CoordinatelessFunction: {type(tf)} is not CoordinatelessFunction")
        tV = tf.function_space()
        elem = tV.ufl_element()
        if tV.mesh() is not self:
            raise RuntimeError(f"tf must be defined on {self}: {tf.mesh()} is not {self}")
        if elem.value_shape() != ():
            raise RuntimeError(f"tf must be scalar: {elem.value_shape()} != ()")
        if elem.family() in {"Discontinuous Lagrange", "DQ"} and elem.degree() == 0:
            # cells
            height = 0
        elif (elem.family() == "HDiv Trace" and elem.degree() == 0 and self.cell_dimension() > 1) or \
                (elem.family() == "Lagrange" and elem.degree() == 1 and self.cell_dimension() == 1):
            # facets
            height = 1
        else:
            raise ValueError(f"indicator functions must be 'DP' or 'DQ' (degree 0) to mark cells and 'P' (degree 1) in 1D or 'HDiv Trace' (degree 0) in 2D or 3D to mark facets: got (family, degree) = ({elem.family()}, {elem.degree()})")
        plex = self.topology_dm
        if not plex.hasLabel(label_name):
            plex.createLabel(label_name)
        label = plex.getLabel(label_name)
        section = tV.dm.getSection()
        array = tf.dat.data_ro_with_halos.real.astype(IntType)
        dmcommon.mark_points_with_function_array(plex, section, height, array, label, label_value)


class ExtrudedMeshTopology(MeshTopology):
    """Representation of an extruded mesh topology."""

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, layers, periodic=False, name=None, tolerance=1.0):
        """Build an extruded mesh topology from an input mesh topology

        :arg mesh:           the unstructured base mesh topology
        :arg layers:         number of occurence of base layer in the "vertical" direction.
        :arg periodic:       the flag for periodic extrusion; if True, only constant layer extrusion is allowed.
        :arg name:           optional name of the extruded mesh topology.
        :kwarg tolerance:    The relative tolerance (i.e. as defined on the
                             reference cell) for the distance a point can be
                             from a cell and still be considered to be in the
                             cell. Note that this tolerance
                             uses an L1 distance (aka 'manhatten', 'taxicab' or
                             rectilinear distance) so will scale with the
                             dimension of the mesh.
        """

        # TODO: refactor to call super().__init__

        from firedrake_citations import Citations
        Citations().register("McRae2016")
        Citations().register("Bercea2016")
        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)

        if isinstance(mesh.topology, VertexOnlyMeshTopology):
            raise NotImplementedError("Extrusion not implemented for VertexOnlyMeshTopology")
        if layers.shape and periodic:
            raise ValueError("Must provide constant layer for periodic extrusion")

        mesh.init()
        self._base_mesh = mesh
        self.user_comm = mesh.comm
        self._comm = internal_comm(mesh._comm)
        if name is not None and name == mesh.name:
            raise ValueError("Extruded mesh topology and base mesh topology can not have the same name")
        self.name = name if name is not None else mesh.name + "_extruded"
        self._tolerance = tolerance
        # TODO: These attributes are copied so that FunctionSpaceBase can
        # access them directly.  Eventually we would want a better refactoring
        # of responsibilities between mesh and function space.
        self.topology_dm = mesh.topology_dm
        r"The PETSc DM representation of the mesh topology."
        self._plex_renumbering = mesh._plex_renumbering
        self._cell_numbering = mesh._cell_numbering
        self._entity_classes = mesh._entity_classes
        self._did_reordering = mesh._did_reordering
        self._distribution_parameters = mesh._distribution_parameters
        self._subsets = {}
        cell = ufl.TensorProductCell(mesh.ufl_cell(), ufl.interval)
        self._ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension()))
        if layers.shape:
            self.variable_layers = True
            extents = extnum.layer_extents(self.topology_dm,
                                           self._cell_numbering,
                                           layers)
            if np.any(extents[:, 3] - extents[:, 2] <= 0):
                raise NotImplementedError("Vertically disconnected cells unsupported")
            self.layer_extents = extents
            """The layer extents for all mesh points.

            For variable layers, the layer extent does not match those for cells.
            A numpy array of layer extents (in PyOP2 format
            :math:`[start, stop)`), of shape ``(num_mesh_points, 4)`` where
            the first two extents are used for allocation and the last
            two for iteration.
            """
        else:
            self.variable_layers = False
        self.cell_set = op2.ExtrudedSet(mesh.cell_set, layers=layers, extruded_periodic=periodic)

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        return self._base_mesh.cell_closure

    @utils.cached_property
    def entity_orientations(self):
        return self._base_mesh.entity_orientations

    def _facets(self, kind):
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)
        base = getattr(self._base_mesh, "%s_facets" % kind)
        return _Facets(self, base.facets, base.classes,
                       kind,
                       base.facet_cell,
                       base.local_facet_dat.data_ro_with_halos,
                       unique_markers=base.unique_markers)

    def make_cell_node_list(self, global_numbering, entity_dofs, entity_permutations, offsets):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg entity_dofs: FInAT element entity DoFs
        :arg entity_permutations: FInAT element entity permutations
        :arg offsets: layer offsets for each entity dof.
        """
        if entity_permutations is None:
            # FInAT entity_permutations not yet implemented
            entity_dofs = eutils.flat_entity_dofs(entity_dofs)
            return super().make_cell_node_list(global_numbering, entity_dofs, None, offsets)
        assert sorted(entity_dofs.keys()) == sorted(entity_permutations.keys()), "Mismatching dimension tuples"
        for key in entity_dofs.keys():
            assert sorted(entity_dofs[key].keys()) == sorted(entity_permutations[key].keys()), "Mismatching entity tuples"
        assert all(v in {0, 1} for _, v in entity_permutations), "Vertical dim index must be in [0, 1]"
        entity_dofs = eutils.flat_entity_dofs(entity_dofs)
        entity_permutations = eutils.flat_entity_permutations(entity_permutations)
        return super().make_cell_node_list(global_numbering, entity_dofs, entity_permutations, offsets)

    def make_dofs_per_plex_entity(self, entity_dofs):
        """Returns the number of DoFs per plex entity for each stratum,
        i.e. [#dofs / plex vertices, #dofs / plex edges, ...].

        each entry is a 2-tuple giving the number of dofs on, and
        above the given plex entity.

        :arg entity_dofs: FInAT element entity DoFs

        """
        dofs_per_entity = np.zeros((1 + self._base_mesh.cell_dimension(), 2), dtype=IntType)
        for (b, v), entities in entity_dofs.items():
            dofs_per_entity[b, v] += len(entities[0])
        return tuplify(dofs_per_entity)

    @PETSc.Log.EventDecorator()
    def node_classes(self, nodes_per_entity, real_tensorproduct=False):
        """Compute node classes given nodes per entity.

        :arg nodes_per_entity: number of function space nodes per topological entity.
        :returns: the number of nodes in each of core, owned, and ghost classes.
        """
        if real_tensorproduct:
            nodes = np.asarray(nodes_per_entity)
            nodes_per_entity = sum(nodes[:, i] for i in range(2))
            return super(ExtrudedMeshTopology, self).node_classes(nodes_per_entity)
        elif self.variable_layers:
            return extnum.node_classes(self, nodes_per_entity)
        else:
            nodes = np.asarray(nodes_per_entity)
            if self.extruded_periodic:
                nodes_per_entity = sum(nodes[:, i]*(self.layers - 1) for i in range(2))
            else:
                nodes_per_entity = sum(nodes[:, i]*(self.layers - i) for i in range(2))
            return super(ExtrudedMeshTopology, self).node_classes(nodes_per_entity)

    @utils.cached_property
    def layers(self):
        """Return the layers parameter used to construct the mesh topology,
        which is the number of layers represented by the number of occurences
        of the base mesh for non-variable layer mesh and an array of size
        (num_cells, 2), each row representing the
        (first layer index, last layer index + 1) pair for the associated cell,
        for variable layer mesh."""
        if self.variable_layers:
            return self.cell_set.layers_array
        else:
            return self.cell_set.layers

    def entity_layers(self, height, label=None):
        """Return the number of layers on each entity of a given plex
        height.

        :arg height: The height of the entity to compute the number of
           layers (0 -> cells, 1 -> facets, etc...)
        :arg label: An optional label name used to select points of
           the given height (if None, then all points are used).
        :returns: a numpy array of the number of layers on the asked
           for entities (or a single layer number for the constant
           layer case).
        """
        if self.variable_layers:
            return extnum.entity_layers(self, height, label)
        else:
            return self.cell_set.layers

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

    def _order_data_by_cell_index(self, column_list, cell_data):
        cell_list = []
        for col in column_list:
            cell_list += list(range(col, col + (self.layers - 1)))
        return cell_data[cell_list]

    @property
    def _distribution_name(self):
        return self._base_mesh._distribution_name

    @property
    def _permutation_name(self):
        return self._base_mesh._permutation_name

    def mark_entities(self, tf, label_name, label_value):
        raise NotImplementedError("Currently not implemented for ExtrudedMesh")


# TODO: Could this be merged with MeshTopology given that dmcommon.pyx
# now covers DMSwarms and DMPlexes?
class VertexOnlyMeshTopology(AbstractMeshTopology):
    """
    Representation of a vertex-only mesh topology immersed within
    another mesh.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, swarm, parentmesh, name, reorder, tolerance=1.0):
        """
        Half-initialise a mesh topology.

        :arg swarm: Particle In Cell (PIC) :class:`DMSwarm` representing
            vertices immersed within a :class:`DMPlex` stored in the
            `parentmesh`
        :arg parentmesh: the mesh within which the vertex-only mesh
            topology is immersed.
        :arg name: name of the mesh
        :arg reorder: whether to reorder the mesh (bool)
        :tolerance: The relative tolerance (i.e. as defined on the
            reference cell) for the distance a point can be from a cell and
            still be considered to be in the cell.
        """

        super().__init__(name, tolerance=tolerance)

        # TODO: As a performance optimisation, we should renumber the
        # swarm to in parent-cell order so that we traverse efficiently.
        if reorder:
            raise NotImplementedError("Mesh reordering not implemented for vertex only meshes yet.")

        dmcommon.validate_mesh(swarm)
        swarm.setFromOptions()

        self._parent_mesh = parentmesh
        self.topology_dm = swarm
        r"The PETSc DM representation of the mesh topology."

        # Set up the comms the same as the parent mesh
        self.user_comm = parentmesh.comm
        self._comm = internal_comm(parentmesh._comm)
        if MPI.Comm.Compare(swarm.comm.tompi4py(), self._comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            ValueError("Parent mesh communicator and swarm communicator are not congruent")

        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)

        # Cell subsets for integration over subregions
        self._subsets = {}

        tdim = 0

        cell = ufl.Cell("vertex")
        self._ufl_mesh = ufl.Mesh(ufl.VectorElement("DG", cell, 0, dim=cell.topological_dimension()))

        # Mark OP2 entities and derive the resulting Swarm numbering
        with PETSc.Log.Event("Mesh: numbering"):
            dmcommon.mark_entity_classes(self.topology_dm)
            self._entity_classes = dmcommon.get_entity_classes(self.topology_dm).astype(int)

            # Derive a cell numbering from the Swarm numbering
            entity_dofs = np.zeros(tdim+1, dtype=IntType)
            entity_dofs[-1] = 1

            self._cell_numbering = self.create_section(entity_dofs)
            entity_dofs[:] = 0
            entity_dofs[0] = 1
            self._vertex_numbering = self.create_section(entity_dofs)

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        swarm = self.topology_dm
        tdim = 0

        # Cell numbering and global vertex numbering
        cell_numbering = self._cell_numbering
        vertex_numbering = self._vertex_numbering.createGlobalSection(swarm.getPointSF())

        cell = self.ufl_cell()
        assert tdim == cell.topological_dimension()
        assert cell.is_simplex()

        import FIAT
        topology = FIAT.ufc_cell(cell).get_topology()
        entity_per_cell = np.zeros(len(topology), dtype=IntType)
        for d, ents in topology.items():
            entity_per_cell[d] = len(ents)

        return dmcommon.closure_ordering(swarm, vertex_numbering,
                                         cell_numbering, entity_per_cell)

    entity_orientations = None

    def _facets(self, kind):
        """Raises an AttributeError since cells in a
        `VertexOnlyMeshTopology` have no facets.
        """
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)
        raise AttributeError("Cells in a VertexOnlyMeshTopology have no facets.")

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def exterior_facets(self):
        return self._facets("exterior")

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def interior_facets(self):
        return self._facets("interior")

    @utils.cached_property
    def cell_to_facets(self):
        """Raises an AttributeError since cells in a
        `VertexOnlyMeshTopology` have no facets.
        """
        raise AttributeError("Cells in a VertexOnlyMeshTopology have no facets.")

    def num_cells(self):
        return self.num_vertices()

    def num_facets(self):
        return 0

    def num_faces(self):
        return 0

    def num_edges(self):
        return 0

    def num_vertices(self):
        return self.topology_dm.getLocalSize()

    def num_entities(self, d):
        if d > 0:
            return 0
        else:
            return self.num_vertices()

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "Cells", comm=self.comm)

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_cell_list(self):
        """Return a list of parent mesh cells numbers in vertex only
        mesh cell order.
        """
        cell_parent_cell_list = np.copy(self.topology_dm.getField("parentcellnum"))
        self.topology_dm.restoreField("parentcellnum")
        return cell_parent_cell_list

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_cell_map(self):
        """Return the :class:`pyop2.types.map.Map` from vertex only mesh cells to
        parent mesh cells.
        """
        return op2.Map(self.cell_set, self._parent_mesh.cell_set, 1,
                       self.cell_parent_cell_list, "cell_parent_cell")

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_base_cell_list(self):
        """Return a list of parent mesh base cells numbers in vertex only
        mesh cell order.
        """
        if not isinstance(self._parent_mesh, ExtrudedMeshTopology):
            raise AttributeError("Parent mesh is not extruded")
        cell_parent_base_cell_list = np.copy(self.topology_dm.getField("parentcellbasenum"))
        self.topology_dm.restoreField("parentcellbasenum")
        return cell_parent_base_cell_list

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_base_cell_map(self):
        """Return the :class:`pyop2.types.map.Map` from vertex only mesh cells to
        parent mesh base cells.
        """
        if not isinstance(self._parent_mesh, ExtrudedMeshTopology):
            raise AttributeError("Parent mesh is not extruded.")
        return op2.Map(self.cell_set, self._parent_mesh.cell_set, 1,
                       self.cell_parent_base_cell_list, "cell_parent_base_cell")

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_extrusion_height_list(self):
        """Return a list of parent mesh extrusion heights in vertex only
        mesh cell order.
        """
        if not isinstance(self._parent_mesh, ExtrudedMeshTopology):
            raise AttributeError("Parent mesh is not extruded.")
        cell_parent_extrusion_height_list = np.copy(self.topology_dm.getField("parentcellextrusionheight"))
        self.topology_dm.restoreField("parentcellextrusionheight")
        return cell_parent_extrusion_height_list

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_extrusion_height_map(self):
        """Return the :class:`pyop2.types.map.Map` from vertex only mesh cells to
        parent mesh extrusion heights.
        """
        if not isinstance(self._parent_mesh, ExtrudedMeshTopology):
            raise AttributeError("Parent mesh is not extruded.")
        return op2.Map(self.cell_set, self._parent_mesh.cell_set, 1,
                       self.cell_parent_extrusion_height_list, "cell_parent_extrusion_height")

    def mark_entities(self, tf, label_name, label_value):
        raise NotImplementedError("Currently not implemented for VertexOnlyMesh")

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_global_index(self):
        """Return a list of unique cell IDs in vertex only mesh cell order."""
        cell_global_index = np.copy(self.topology_dm.getField("globalindex"))
        self.topology_dm.restoreField("globalindex")
        return cell_global_index


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

    def _init_topology(self, topology):
        """Initialise the topology.

        :arg topology: The :class:`.MeshTopology` object.

        A mesh is fully initialised with its topology and coordinates.
        In this method we partially initialise the mesh by registering
        its topology. We also set the `_callback` attribute that is
        later called to set its coordinates and finalise the initialisation.
        """
        import firedrake.functionspace as functionspace
        import firedrake.function as function

        self._topology = topology

        def callback(self):
            """Finish initialisation."""
            del self._callback
            # Finish the initialisation of mesh topology
            self.topology.init()
            coordinates_fs = functionspace.FunctionSpace(self.topology, self.ufl_coordinate_element())
            coordinates_data = dmcommon.reordered_coords(topology.topology_dm, coordinates_fs.dm.getDefaultSection(),
                                                         (self.num_vertices(), self.ufl_coordinate_element().cell().geometric_dimension()))
            coordinates = function.CoordinatelessFunction(coordinates_fs,
                                                          val=coordinates_data,
                                                          name=_generate_default_mesh_coordinates_name(self.name))
            self.__init__(coordinates)
        self._callback = callback

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

    @property
    def tolerance(self):
        """The relative tolerance (i.e. as defined on the reference cell) for
        the distance a point can be from a cell and still be considered to be
        in the cell.

        Increase this if points at mesh boundaries (either rank local or
        global) are reported as being outside the mesh, for example when
        creating a :class:`VertexOnlyMesh`. Note that this tolerance uses an L1
        distance (aka 'manhatten', 'taxicab' or rectilinear distance) so will
        scale with the dimension of the mesh.

        If this property is not set (i.e. set to ``None``) no tolerance is
        added to the bounding box and points deemed at all outside the mesh,
        even by floating point error distances, will be deemed to be outside
        it.

        Notes
        -----
        Modifying this property will modify the ``MeshTopology.tolerance``
        property of the underlying mesh topology. Furthermore, after changing
        it any requests for :attr:`spatial_index` will cause the spatial index
        to be rebuilt with the new tolerance which may take some time.
        """
        return self.topology.tolerance

    @tolerance.setter
    def tolerance(self, value):
        if not isinstance(value, numbers.Number):
            raise TypeError("tolerance must be a number")
        if value != self.topology.tolerance:
            self.clear_spatial_index()
            self.topology._tolerance = value

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
        """Spatial index to quickly find which cell contains a given point.

        Notes
        -----

        If this mesh has a :attr:`tolerance` property, which
        should be a float, this tolerance is added to the extrama of the
        spatial index so that points just outside the mesh, within tolerance,
        can be found.

        """

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

        # Push max and min out so we can find points on the boundary within
        # tolerance. Note that if tolerance is too small it might not actually
        # change the value!
        if hasattr(self, "tolerance") and self.tolerance is not None:
            coords_min -= self.tolerance
            coords_max += self.tolerance

        # Build spatial index
        return spatialindex.from_regions(coords_min, coords_max)

    @PETSc.Log.EventDecorator()
    def locate_cell(self, x, tolerance=None):
        """Locate cell containing a given point.

        :arg x: point coordinates
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the spatial index to be rebuilt which
            can take some time.
        :returns: cell number (int), or None (if the point is not
            in the domain)
        """
        return self.locate_cell_and_reference_coordinate(x, tolerance=tolerance)[0]

    def locate_reference_coordinate(self, x, tolerance=None):
        """Get reference coordinates of a given point in its cell. Which
        cell the point is in can be queried with the locate_cell method.

        :arg x: point coordinates
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the spatial index to be rebuilt which
            can take some time.
        :returns: reference coordinates within cell (numpy array) or
            None (if the point is not in the domain)
        """
        return self.locate_cell_and_reference_coordinate(x, tolerance=tolerance)[1]

    def locate_cell_and_reference_coordinate(self, x, tolerance=None):
        """Locate cell containing a given point and the reference
        coordinates of the point within the cell.

        :arg x: point coordinates
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the spatial index to be rebuilt which
            can take some time.
        :returns: tuple either
            (cell number, reference coordinates) of type (int, numpy array),
            or, when point is not in the domain, (None, None).
        """
        x = np.asarray(x)
        if x.size != self.geometric_dimension():
            raise ValueError("Point must have the same geometric dimension as the mesh")
        x = x.reshape((1, self.geometric_dimension()))
        cells, ref_coords, _ = self.locate_cells_ref_coords_and_dists(x, tolerance=tolerance)
        if cells[0] == -1:
            return None, None
        return cells[0], ref_coords[0]

    def locate_cells_ref_coords_and_dists(self, xs, tolerance=None):
        """Locate cell containing a given point and the reference
        coordinates of the point within the cell.

        :arg xs: 1 or more point coordinates of shape (npoints, gdim)
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the spatial index to be rebuilt which
            can take some time.
        :returns: tuple either
            (cell numbers array, reference coordinates array, ref_cell_dists_l1 array)
            of type
            (array of ints, array of floats of size (npoints, gdim), array of floats).
            The cell numbers array contains -1 for points not in the domain:
            the reference coordinates and distances are meaningless for these
            points.
        """
        if self.variable_layers:
            raise NotImplementedError("Cell location not implemented for variable layers")
        if tolerance is None:
            tolerance = self.tolerance
        else:
            self.tolerance = tolerance
        xs = np.asarray(xs, dtype=utils.ScalarType)
        xs = xs.real.copy()
        if xs.shape[1] != self.geometric_dimension():
            raise ValueError("Point coordinate dimension does not match mesh geometric dimension")
        Xs = np.empty_like(xs)
        npoints = len(xs)
        ref_cell_dists_l1 = np.empty(npoints, dtype=utils.RealType)
        cells = np.empty(npoints, dtype=IntType)
        assert xs.size == npoints * self.geometric_dimension()
        self._c_locator(tolerance=tolerance)(self.coordinates._ctypes,
                                             xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             Xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             ref_cell_dists_l1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             cells.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                             npoints)
        return cells, Xs, ref_cell_dists_l1

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
    int locator(struct Function *f, double *x, double *X, double *ref_cell_dists_l1, int *cells, size_t npoints)
    {
        size_t j = 0;  /* index into x and X */
        for(size_t i=0; i<npoints; i++) {
            /* i is the index into cells and ref_cell_dists_l1 */

            /* The type definitions and arguments used here are defined as
            statics in pointquery_utils.py */
            struct ReferenceCoords temp_reference_coords, found_reference_coords;

            cells[i] = locate_cell(f, &x[j], %(geometric_dimension)d, &to_reference_coords, &to_reference_coords_xtr, &temp_reference_coords, &found_reference_coords, &ref_cell_dists_l1[i]);

            for (int k = 0; k < %(geometric_dimension)d; k++) {
                X[j] = found_reference_coords.X[k];
                j++;
            }
        }
        return 0;
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
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double)]
            locator.restype = ctypes.c_int
            return cache.setdefault(tolerance, locator)

    def cell_orientations(self):
        """Return the orientation of each cell in the mesh.

        Use :meth:`.init_cell_orientations` to initialise."""
        # View `_cell_orientations` (`CoordinatelessFunction`) as a property of
        # `MeshGeometry` as opposed to one of `MeshTopology`, and treat it just like
        # `_coordinates` (`CoordinatelessFunction`) so that we have:
        # -- Regular MeshGeometry  = MeshTopology + `_coordinates`,
        # -- Immersed MeshGeometry = MeshTopology + `_coordinates` + `_cell_orientations`.
        # Here, `_coordinates` and `_cell_orientations` both represent some geometric
        # properties (i.e., "coordinates" and "cell normals").
        #
        # Two `MeshGeometry`s can share the same `MeshTopology` and `_coordinates` while
        # having distinct definition of "cell normals"; they are then simply regarded as two
        # distinct meshes as `dot(expr, cell_normal) * dx` in general gives different results.
        #
        # Storing `_cell_orientations` in `MeshTopology` would make the `MeshTopology`
        # object only useful for specific definition of "cell normals".
        if not hasattr(self, '_cell_orientations'):
            raise RuntimeError("No cell orientations found, did you forget to call init_cell_orientations?")
        return self._cell_orientations

    @PETSc.Log.EventDecorator()
    def init_cell_orientations(self, expr):
        """Compute and initialise meth:`cell_orientations` relative to a specified orientation.

        :arg expr: a UFL expression evaluated to produce a
             reference normal direction.

        """
        import firedrake.function as function
        import firedrake.functionspace as functionspace

        if self.ufl_cell() not in _supported_embedded_cell_types:
            raise NotImplementedError('Only implemented for intervals embedded in 2d and triangles and quadrilaterals embedded in 3d')

        if hasattr(self, '_cell_orientations'):
            raise RuntimeError("init_cell_orientations already called, did you mean to do so again?")

        if not isinstance(expr, ufl.classes.Expr):
            raise TypeError("UFL expression expected!")

        if expr.ufl_shape != (self.ufl_cell().geometric_dimension(), ):
            raise ValueError(f"Mismatching shapes: expr.ufl_shape ({expr.ufl_shape}) != (self.ufl_cell().geometric_dimension(), ) (({self.ufl_cell().geometric_dimension}, ))")

        fs = functionspace.FunctionSpace(self, 'DG', 0)
        x = ufl.SpatialCoordinate(self)
        f = function.Function(fs)

        if self.topological_dimension() == 1:
            normal = ufl.as_vector((-ReferenceGrad(x)[1, 0], ReferenceGrad(x)[0, 0]))
        else:  # self.topological_dimension() == 2
            normal = ufl.cross(ReferenceGrad(x)[:, 0], ReferenceGrad(x)[:, 1])

        f.interpolate(ufl.dot(expr, normal))

        cell_orientations = function.Function(fs, name="cell_orientations", dtype=np.int32)
        cell_orientations.dat.data[:] = (f.dat.data_ro < 0)
        self._cell_orientations = cell_orientations.topological

    def __getattr__(self, name):
        val = getattr(self._topology, name)
        setattr(self, name, val)
        return val

    def __dir__(self):
        current = super(MeshGeometry, self).__dir__()
        return list(OrderedDict.fromkeys(dir(self._topology) + current))

    def mark_entities(self, f, label_name, label_value):
        """Mark selected entities.

        :arg f: The :class:`.Function` object that marks
            selected entities as 1. f.function_space().ufl_element()
            must be "DP" or "DQ" (degree 0) to mark cell entities and
            "P" (degree 1) in 1D or "HDiv Trace" (degree 0) in 2D or 3D
            to mark facet entities.
        :arg label_name: The name of the label to store entity selections.
        :arg lable_value: The value used in the label.

        All entities must live on the same topological dimension. Currently,
        one can only mark cell or facet entities.
        """
        self.topology.mark_entities(f.topological, label_name, label_value)


@PETSc.Log.EventDecorator()
def make_mesh_from_coordinates(coordinates, name):
    """Given a coordinate field build a new mesh, using said coordinate field.

    :arg coordinates: A :class:`~.Function`.
    :arg name: The name of the mesh.
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
    cell = element.cell().reconstruct(geometric_dimension=V.value_size)
    element = element.reconstruct(cell=cell)

    mesh = MeshGeometry.__new__(MeshGeometry, element)
    mesh.__init__(coordinates)
    mesh.name = name
    # Mark mesh as being made from coordinates
    mesh._made_from_coordinates = True
    return mesh


def make_mesh_from_mesh_topology(topology, name, comm=COMM_WORLD):
    # Construct coordinate element
    # TODO: meshfile might indicates higher-order coordinate element
    cell = topology.ufl_cell()
    geometric_dim = topology.topology_dm.getCoordinateDim()
    cell = cell.reconstruct(geometric_dimension=geometric_dim)
    element = ufl.VectorElement("Lagrange", cell, 1)
    # Create mesh object
    mesh = MeshGeometry.__new__(MeshGeometry, element)
    mesh._init_topology(topology)
    mesh.name = name
    return mesh


@PETSc.Log.EventDecorator("CreateMesh")
def Mesh(meshfile, **kwargs):
    """Construct a mesh object.

    Meshes may either be created by reading from a mesh file, or by
    providing a PETSc DMPlex object defining the mesh topology.

    :param meshfile: the mesh file name, a DMPlex object or a Netgen mesh object defining
           mesh topology.  See below for details on supported mesh
           formats.
    :param name: optional name of the mesh object.
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

    :param distribution_name: the name of parallel distribution used
           when checkpointing; if not given, the name is automatically
           generated.

    :param permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if not given, the name is automatically
           generated.

    :param comm: the communicator to use when creating the mesh.  If
           not supplied, then the mesh will be created on COMM_WORLD.
           If ``meshfile`` is a DMPlex object then must be indentical
           to or congruent with the DMPlex communicator.

    :param tolerance: The relative tolerance (i.e. as defined on the reference
           cell) for the distance a point can be from a cell and still be
           considered to be in the cell. Defaults to 1.0. Increase
           this if point at mesh boundaries (either rank local or global) are
           reported as being outside the mesh, for example when creating a
           :class:`VertexOnlyMesh`. Note that this tolerance uses an L1
           distance (aka 'manhatten', 'taxicab' or rectilinear distance) so
           will scale with the dimension of the mesh.

    When the mesh is read from a file the following mesh formats
    are supported (determined, case insensitively, from the
    filename extension):

    * GMSH: with extension `.msh`
    * Exodus: with extension `.e`, `.exo`
    * CGNS: with extension `.cgns`
    * Triangle: with extension `.node`
    * HDF5: with extension `.h5`, `.hdf5`
      (Can only load HDF5 files created by ``MeshGeometry.save`` method.)

    .. note::

        When the mesh is created directly from a DMPlex object or a Netgen
        mesh object, the ``dim`` parameter is ignored (the DMPlex already
        knows its geometric and topological dimensions).

    """
    import firedrake.function as function

    user_comm = kwargs.get("comm", COMM_WORLD)
    name = kwargs.get("name", DEFAULT_MESH_NAME)
    reorder = kwargs.get("reorder", None)
    if reorder is None:
        reorder = parameters["reorder_meshes"]
    distribution_parameters = kwargs.get("distribution_parameters", None)
    if distribution_parameters is None:
        distribution_parameters = {}
    if isinstance(meshfile, str) and \
       any(meshfile.lower().endswith(ext) for ext in ['.h5', '.hdf5']):
        from firedrake.output import CheckpointFile

        with CheckpointFile(meshfile, 'r', comm=user_comm) as afile:
            return afile.load_mesh(name=name, reorder=reorder,
                                   distribution_parameters=distribution_parameters)
    elif isinstance(meshfile, function.Function):
        coordinates = meshfile.topological
    elif isinstance(meshfile, function.CoordinatelessFunction):
        coordinates = meshfile
    else:
        coordinates = None
    if coordinates is not None:
        return make_mesh_from_coordinates(coordinates, name)

    tolerance = kwargs.get("tolerance", 1.0)

    utils._init()

    # We don't need to worry about using a user comm in these cases as
    # they all immediately call a petsc4py which in turn uses a PETSc
    # internal comm
    geometric_dim = kwargs.get("dim", None)
    if isinstance(meshfile, PETSc.DMPlex):
        plex = meshfile
        if MPI.Comm.Compare(user_comm, plex.comm.tompi4py()) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Communicator used to create `plex` must be at least congruent to the communicator used to create the mesh")
    elif netgen and isinstance(meshfile, netgen.libngpy._meshing.Mesh):
        plex = _from_netgen(meshfile, user_comm)
    else:
        basename, ext = os.path.splitext(meshfile)
        if ext.lower() in ['.e', '.exo']:
            plex = _from_exodus(meshfile, user_comm)
        elif ext.lower() == '.cgns':
            plex = _from_cgns(meshfile, user_comm)
        elif ext.lower() == '.msh':
            if geometric_dim is not None:
                opts = {"dm_plex_gmsh_spacedim": geometric_dim}
            else:
                opts = {}
            opts = OptionsManager(opts, "")
            with opts.inserted_options():
                plex = _from_gmsh(meshfile, user_comm)
        elif ext.lower() == '.node':
            plex = _from_triangle(meshfile, geometric_dim, user_comm)
        else:
            raise RuntimeError("Mesh file %s has unknown format '%s'."
                               % (meshfile, ext[1:]))
        plex.setName(_generate_default_mesh_topology_name(name))
    # Create mesh topology
    topology = MeshTopology(plex, name=plex.getName(), reorder=reorder,
                            distribution_parameters=distribution_parameters,
                            distribution_name=kwargs.get("distribution_name"),
                            permutation_name=kwargs.get("permutation_name"),
                            comm=user_comm, tolerance=tolerance)
    mesh = make_mesh_from_mesh_topology(topology, name)
    if netgen and isinstance(meshfile, netgen.libngpy._meshing.Mesh):
        # Adding Netgen mesh and inverse sfBC as attributes
        mesh.netgen_mesh = meshfile
        mesh.sfBCInv = mesh.sfBC.createInverse() if user_comm.Get_size() > 1 else None
        mesh.comm = user_comm
        # Refine Method

        def refine_marked_elements(self, mark):
            with mark.dat.vec as marked:
                marked0 = marked
                getIdx = self._cell_numbering.getOffset
                if self.sfBCInv is not None:
                    getIdx = lambda x: x
                    _, marked0 = self.topology_dm.distributeField(self.sfBCInv,
                                                                  self._cell_numbering,
                                                                  marked)
                if self.comm.Get_rank() == 0:
                    mark = marked0.getArray()
                    for i, el in enumerate(self.netgen_mesh.Elements2D()):
                        if mark[getIdx(i)]:
                            el.refine = True
                        else:
                            el.refine = False
                    self.netgen_mesh.Refine(adaptive=True)
                    return Mesh(self.netgen_mesh)
                else:
                    return Mesh(netgen.libngpy._meshing.Mesh(2))

        setattr(MeshGeometry, "refine_marked_elements", refine_marked_elements)
    return mesh


@PETSc.Log.EventDecorator("CreateExtMesh")
def ExtrudedMesh(mesh, layers, layer_height=None, extrusion_type='uniform', periodic=False, kernel=None, gdim=None, name=None, tolerance=1.0):
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
    :arg periodic:       the flag for periodic extrusion; if True, only constant layer extrusion is allowed.
                         Can be used with any "extrusion_type" to make annulus, torus, etc.
    :arg kernel:         a ``pyop2.Kernel`` to produce coordinates for
                         the extruded mesh. See :func:`~.make_extruded_coords`
                         for more details.
    :arg gdim:           number of spatial dimensions of the
                         resulting mesh (this is only used if a
                         custom kernel is provided)
    :arg name:           optional name for the extruded mesh.
    :kwarg tolerance:    The relative tolerance (i.e. as defined on the
                         reference cell) for the distance a point can be from a
                         cell and still be considered to be in the cell.
                         Note that this tolerance uses an L1
                         distance (aka 'manhatten', 'taxicab' or rectilinear
                         distance) so will scale with the dimension of the
                         mesh.

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
        available in the ``radial_coordinates`` attribute.
    ``"custom"``
        use a custom kernel to generate the extruded coordinates

    For more details see the :doc:`manual section on extruded meshes <extruded-meshes>`.
    """
    import firedrake.functionspace as functionspace
    import firedrake.function as function

    if name is not None and name == mesh.name:
        raise ValueError("Extruded mesh and base mesh can not have the same name")
    name = name if name is not None else mesh.name + "_extruded"
    mesh.init()
    layers = np.asarray(layers, dtype=IntType)
    if layers.shape:
        if periodic:
            raise ValueError("Must provide constant layer for periodic extrusion")
        if layers.shape != (mesh.cell_set.total_size, 2):
            raise ValueError("Must provide single layer number or array of shape (%d, 2), not %s",
                             mesh.cell_set.total_size, layers.shape)
        if layer_height is None:
            raise ValueError("Must provide layer height for variable layers")

        # variable-height layers need to be present for the maximum number
        # of extruded layers
        num_layers = layers.sum(axis=1).max() if mesh.cell_set.total_size else 0
        num_layers = mesh._comm.allreduce(num_layers, op=MPI.MAX)

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

    topology = ExtrudedMeshTopology(mesh.topology, layers, periodic=periodic, tolerance=tolerance)

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
    if periodic:
        velement = ufl.FiniteElement("DP", ufl.interval, 1, variant="equispaced")
    else:
        velement = ufl.FiniteElement("Lagrange", ufl.interval, 1)
    element = ufl.TensorProductElement(helement, velement)

    if gdim is None:
        gdim = mesh.ufl_cell().geometric_dimension() + (extrusion_type == "uniform")
    coordinates_fs = functionspace.VectorFunctionSpace(topology, element, dim=gdim)

    coordinates = function.CoordinatelessFunction(coordinates_fs, name=_generate_default_mesh_coordinates_name(name))

    eutils.make_extruded_coords(topology, mesh._coordinates, coordinates,
                                layer_height, extrusion_type=extrusion_type, kernel=kernel)

    self = make_mesh_from_coordinates(coordinates, name)
    self._base_mesh = mesh

    if extrusion_type == "radial_hedgehog":
        helement = mesh._coordinates.ufl_element().sub_elements()[0].reconstruct(family="CG")
        element = ufl.TensorProductElement(helement, velement)
        fs = functionspace.VectorFunctionSpace(self, element, dim=gdim)
        self.radial_coordinates = function.Function(fs, name=name + "_radial_coordinates")
        eutils.make_extruded_coords(topology, mesh._coordinates, self.radial_coordinates,
                                    layer_height, extrusion_type="radial", kernel=kernel)

    return self


@PETSc.Log.EventDecorator()
def VertexOnlyMesh(mesh, vertexcoords, missing_points_behaviour='error',
                   tolerance=None, redundant=True):
    """
    Create a vertex only mesh, immersed in a given mesh, with vertices defined
    by a list of coordinates.

    :arg mesh: The unstructured mesh in which to immerse the vertex only mesh.
    :arg vertexcoords: A list of coordinate tuples which defines the vertices.
    :kwarg missing_points_behaviour: optional string argument for what to do
        when vertices which are outside of the mesh are discarded. If
        ``'warn'``, will print a warning. If ``'error'`` will raise a
        ValueError.
    :kwarg tolerance: The relative tolerance (i.e. as defined on the reference
        cell) for the distance a point can be from a mesh cell and still be
        considered to be in the cell. Note that this tolerance uses an L1
        distance (aka 'manhatten', 'taxicab' or rectilinear distance) so
        will scale with the dimension of the mesh. The default is the parent
        mesh's ``tolerance`` property. Changing this from default will
        cause the parent mesh's spatial index to be rebuilt which can take some
        time.
    :kwarg redundant: If True, the mesh will be built using just the vertices
        which are specified on rank 0. If False, the mesh will be built using
        the vertices specified by each rank. Care must be taken when using
        ``redundant = False``: see the note below for more information.


    .. note::

        The vertex only mesh uses the same communicator as the input ``mesh``.

    .. note::

        Manifold meshes and extruded meshes with variable extrusion layers are
        not yet supported.

    .. note::
        When running in parallel with ``redundant = False``, ``vertexcoords``
        will redistribute to the mesh partition where they are located. This
        means that if rank A has ``vertexcoords`` {X} that are not found in the
        mesh cells owned by rank A but are found in the mesh cells owned by
        rank B, **and rank B has not been supplied with those**, then they will
        be moved to rank B.

    .. note::
        If the same coordinates are supplied more than once, they are always
        assumed to be a new vertex.

    """

    import firedrake.functionspace as functionspace
    import firedrake.function as function

    if tolerance is None:
        tolerance = mesh.tolerance
    else:
        mesh.tolerance = tolerance

    mesh.init()

    vertexcoords = np.asarray(vertexcoords, dtype=np.double)
    gdim = mesh.geometric_dimension()
    tdim = mesh.topological_dimension()
    _, pdim = vertexcoords.shape

    if not np.isclose(np.sum(abs(vertexcoords.imag)), 0):
        raise ValueError("Point coordinates must have zero imaginary part")

    if gdim != tdim:
        raise NotImplementedError("Immersed manifold meshes are not supported")

    # Bendy meshes require a smarter bounding box algorithm at partition and
    # (especially) cell level. Projecting coordinates to Bernstein may be
    # sufficient.
    if np.any(np.asarray(mesh.coordinates.function_space().ufl_element().degree())) > 1:
        raise NotImplementedError("Only straight edged meshes are supported")

    # Currently we take responsibility for locating the mesh cells in which the
    # vertices lie.
    #
    # In the future we hope to update the coordinates field correctly so that
    # the DMSwarm PIC can immerse itself in the DMPlex. We can also hopefully
    # provide a callback for PETSc to use to find the parent cell id. We would
    # add `DMLocatePoints` as an `op` to `DMShell` types and do
    # `DMSwarmSetCellDM(yourdmshell)` which has `DMLocatePoints_Shell`
    # implemented. Whether one or both of these is needed is unclear.

    if pdim != gdim:
        raise ValueError(f"Mesh geometric dimension {gdim} must match point list dimension {pdim}")

    swarm, n_missing_points = _pic_swarm_in_mesh(
        mesh, vertexcoords, tolerance=tolerance, redundant=redundant
    )

    if missing_points_behaviour:
        if n_missing_points:
            msg = f"{n_missing_points} vertices are outside the mesh and have been removed from the VertexOnlyMesh"
            if missing_points_behaviour == 'error':
                raise ValueError(msg)
            elif missing_points_behaviour == 'warn':
                from warnings import warn
                warn(msg)
            else:
                raise ValueError("missing_points_behaviour must be None, 'error' or 'warn'")

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
    vmesh.reference_coordinates = \
        dmcommon.fill_reference_coordinates_function(function.Function(reference_coordinates_fs))

    return vmesh


def _pic_swarm_in_mesh(parent_mesh, coords, fields=None, tolerance=None, redundant=True):
    """Create a Particle In Cell (PIC) DMSwarm immersed in a Mesh

    This should only by used for meshes with straight edges. If not, the
    particles may be placed in the wrong cells.

    :arg parent_mesh: the :class:`Mesh` within with the DMSwarm should be
        immersed.
    :arg coords: an ``ndarray`` of (npoints, coordsdim) shape.
    :kwarg fields: An optional list of named data which can be stored for each
        point in the DMSwarm. The format should be::

        [(fieldname1, blocksize1, dtype1),
          ...,
         (fieldnameN, blocksizeN, dtypeN)]

        For example, the swarm coordinates themselves are stored in a field
        named ``DMSwarmPIC_coor`` which, were it not created automatically,
        would be initialised with ``fields = [("DMSwarmPIC_coor", coordsdim,
        RealType)]``. All fields must have the same number of points. For more
        information see `the DMSWARM API reference
        <https://www.mcs.anl.gov/petsc/petsc-current/manualpages/DMSWARM/DMSWARM.html>_.
    :kwarg tolerance: The relative tolerance (i.e. as defined on the reference
        cell) for the distance a point can be from a cell and still be
        considered to be in the cell. Note that this tolerance uses an L1
        distance (aka 'manhatten', 'taxicab' or rectilinear distance) so
        will scale with the dimension of the mesh. The default is the parent
        mesh's ``tolerance`` property. Changing this from default will
        cause the parent mesh's spatial index to be rebuilt which can take some
        time.
    :kwarg redundant: If True, the DMSwarm will be created using only the
        points specified on MPI rank 0.
    :return: the immersed DMSwarm

    .. note::

        The created DMSwarm uses the communicator of the input Mesh.

    .. note::

        In complex mode the "DMSwarmPIC_coor" field is still saved as a real
        number unlike the coordinates of a DMPlex which become complex (though
        usually with zeroed imaginary parts).

    .. note::
        When running in parallel with ``redundant = False``, ``coords``
        will redistribute to the mesh partition where they are located. This
        means that if rank A has ``coords`` {X} that are not found in the
        mesh cells owned by rank A but are found in the mesh cells owned by
        rank B, **and rank B has not been supplied with those**, then they will
        be moved to rank B.

    .. note::
        If the same coordinates are supplied more than once, they are always
        assumed to be a new vertex.

    .. note::
        Three DMSwarm fields are created automatically here:

        #. ``parentcellnum`` which contains the firedrake cell number of the
           immersed vertex and
        #. ``refcoord`` which contains the reference coordinate of the immersed
           vertex in the parent mesh cell.
        #. ``globalindex`` which contains a unique ID for each DMSwarm point -
           here this is the index into the ``coords`` array if ``redundant`` is
           ``True``, otherwise it's an index in rank order, so if rank 0 has 10
           points, rank 1 has 20 points, and rank 3 has 5 points, then rank 0's
           points will be numbered 0-9, rank 1's points will be numbered 10-29,
           and rank 3's points will be numbered 30-34. Note that this ought to
           be ``DMSwarmField_pid`` but a bug in petsc4py means that this field
           cannot be set.

        Another three are required for proper functioning of the DMSwarm:

        #. ``DMSwarmPIC_coor`` which contains the coordinates of the point.
        #. ``DMSwarm_cellid`` the DMPlex cell within which the DMSwarm point is
           located.
        #. ``DMSwarm_rank``: the MPI rank which owns the DMSwarm point.
    """

    if tolerance is None:
        tolerance = parent_mesh.tolerance
    else:
        parent_mesh.tolerance = tolerance

    # Check coords
    coords = np.asarray(coords, dtype=RealType)

    plex = parent_mesh.topology.topology_dm
    tdim = parent_mesh.topological_dimension()
    gdim = parent_mesh.geometric_dimension()

    if fields is None:
        fields = []
    fields += [("parentcellnum", 1, IntType), ("refcoord", tdim, RealType), ("globalindex", 1, IntType)]

    (
        coords,
        coords_idxs,
        reference_coords,
        parent_cell_nums,
        ranks,
        missing_coords_idxs,
    ) = _parent_mesh_embedding(parent_mesh, coords, tolerance, redundant, exclude_halos=True)

    n_missing_points = len(missing_coords_idxs)

    if parent_mesh.extruded:
        # need to store the base parent cell number and the height to be able
        # to map point coordinates back to the parent mesh
        if parent_mesh.variable_layers:
            raise NotImplementedError("Cannot create a DMSwarm in an ExtrudedMesh with variable layers.")
        # Extruded mesh parent_cell_nums goes from bottom to top. So for
        # mx = ExtrudedMesh(UnitIntervalMesh(2), 3) we have
        # mx.layers = 4
        # and
        #  -------------------layer 4-------------------
        # | parent_cell_num =  2 | parent_cell_num =  5 |
        # | extrusion_height = 2 | extrusion_height = 2 |
        #  -------------------layer 3-------------------
        # | parent_cell_num =  1 | parent_cell_num =  4 |
        # | extrusion_height = 1 | extrusion_height = 1 |
        #  -------------------layer 2-------------------
        # | parent_cell_num =  0 | parent_cell_num =  3 |
        # | extrusion_height = 0 | extrusion_height = 0 |
        #  -------------------layer 1-------------------
        #   base_cell_num = 0         base_cell_num = 1
        # The base_cell_num is the cell number in the base mesh which, in this
        # case, is a UnitIntervalMesh with two cells.
        fields += [("parentcellbasenum", 1, IntType), ("parentcellextrusionheight", 1, IntType)]
        base_parent_cell_nums = parent_cell_nums // (parent_mesh.layers - 1)
        extrusion_heights = parent_cell_nums % (parent_mesh.layers - 1)
        # mesh.topology.cell_closure[:, -1] maps Firedrake cell numbers to plex numbers.
        plex_parent_cell_nums = parent_mesh.topology.cell_closure[base_parent_cell_nums, -1]
    elif parent_mesh.coordinates.dat.dat_version > 0:
        # The parent mesh coordinates have been modified. The DMSwarm parent
        # mesh plex numbering is now not guaranteed to match up with DMPlex
        # numbering so are set to -1. DMSwarm functions which rely on the
        # DMPlex numbering,such as DMSwarmMigrate() will not work as expected.
        plex_parent_cell_nums = -np.ones_like(parent_cell_nums)
    else:
        # mesh.topology.cell_closure[:, -1] maps Firedrake cell numbers to plex numbers.
        plex_parent_cell_nums = parent_mesh.topology.cell_closure[parent_cell_nums, -1]

    _, coordsdim = coords.shape

    # Create a DMSWARM
    swarm = PETSc.DMSwarm().create(comm=parent_mesh._comm)

    plexdim = plex.getDimension()
    if plexdim != tdim:
        # This is a Firedrake extruded mesh, so we need to use the
        # mesh geometric dimension when we create the swarm. In this
        # case DMSwarmMigate() will not work.
        swarmdim = gdim
    else:
        swarmdim = plexdim

    # Set swarm DM dimension to match DMPlex dimension
    # NB: Unlike a DMPlex, this does not correspond to the topological
    #     dimension of a mesh (which would be 0). In all PETSc examples
    #     the dimension of the DMSwarm is set to match that of the
    #     DMPlex used with swarm.setCellDM. As noted above, for an
    #     extruded mesh this will stop DMSwarmMigrate() from working.
    swarm.setDimension(swarmdim)

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

    num_vertices = len(coords)
    swarm.setLocalSizes(num_vertices, -1)

    # Add point coordinates. This amounts to our own implementation of
    # DMSwarmSetPointCoordinates because Firedrake's mesh coordinate model
    # doesn't always exactly coincide with that of DMPlex: in most cases the
    # plex_parent_cell_nums (DMSwarm_cellid field) and parent_cell_nums
    # (parentcellnum field), the latter being the numbering used by firedrake,
    # refer fundamentally to the same cells. For extruded meshes the DMPlex
    # dimension is based on the topological dimension of the base mesh.

    # NOTE ensure that swarm.restoreField is called for each field too!
    swarm_coords = swarm.getField("DMSwarmPIC_coor").reshape((num_vertices, gdim))
    swarm_parent_cell_nums = swarm.getField("DMSwarm_cellid")
    field_parent_cell_nums = swarm.getField("parentcellnum")
    field_reference_coords = swarm.getField("refcoord").reshape((num_vertices, tdim))
    field_global_index = swarm.getField("globalindex")
    field_rank = swarm.getField("DMSwarm_rank")

    swarm_coords[...] = coords
    swarm_parent_cell_nums[...] = plex_parent_cell_nums
    field_parent_cell_nums[...] = parent_cell_nums
    field_reference_coords[...] = reference_coords
    field_global_index[...] = coords_idxs
    field_rank[...] = ranks

    # have to restore fields once accessed to allow access again
    swarm.restoreField("DMSwarm_rank")
    swarm.restoreField("globalindex")
    swarm.restoreField("refcoord")
    swarm.restoreField("parentcellnum")
    swarm.restoreField("DMSwarmPIC_coor")
    swarm.restoreField("DMSwarm_cellid")

    if parent_mesh.extruded:
        field_base_parent_cell_nums = swarm.getField("parentcellbasenum")
        field_extrusion_heights = swarm.getField("parentcellextrusionheight")
        field_base_parent_cell_nums[...] = base_parent_cell_nums
        field_extrusion_heights[...] = extrusion_heights
        swarm.restoreField("parentcellbasenum")
        swarm.restoreField("parentcellextrusionheight")

    # Set the `SF` graph to advertises no shared points (since the halo
    # is now empty) by setting the leaves to an empty list
    sf = swarm.getPointSF()
    nroots = swarm.getLocalSize()
    sf.setGraph(nroots, None, [])
    swarm.setPointSF(sf)

    return swarm, n_missing_points


def _mpi_array_lexicographic_min(x, y, datatype):
    """MPI operator for lexicographic minimum of arrays.

    This compares two arrays of shape (N, 2) lexicographically, i.e. first
    comparing the two arrays by their first column, returning the element-wise
    minimum, with ties broken by comparing the second column element wise.

    Parameters
    ----------
    x : ``np.ndarray``
        The first array to compare of shape (N, 2).
    y : ``np.ndarray``
        The second array to compare of shape (N, 2).
    datatype : ``MPI.Datatype``
        The datatype of the arrays.

    Returns
    -------
    ``np.ndarray``
        The lexicographically lowest array of shape (N, 2).

    """
    # Check the first column
    min_idxs = np.where(x[:, 0] < y[:, 0])[0]
    result = np.copy(y)
    result[min_idxs, :] = x[min_idxs, :]

    # if necessary, check the second column
    eq_idxs = np.where(x[:, 0] == y[:, 0])[0]
    if len(eq_idxs):
        # We only check where we have equal values to avoid unnecessary work
        min_idxs = np.where(x[eq_idxs, 1] < y[eq_idxs, 1])[0]
        result[eq_idxs[min_idxs], :] = x[eq_idxs[min_idxs], :]
    return result


array_lexicographic_mpi_op = MPI.Op.Create(_mpi_array_lexicographic_min, commute=True)


def _parent_mesh_embedding(parent_mesh, coords, tolerance, redundant, exclude_halos):
    """Find the parent mesh cells containing the given coordinates.

    Parameters
    ----------
    parent_mesh : ``Mesh``
        The parent mesh to embed in.
    coords : ``np.ndarray``
        The coordinates to embed of (npoints, coordsdim) shape.
    tolerance : ``float``
        The relative tolerance (i.e. as defined on the reference cell) for the
        distance a point can be from a cell and still be considered to be in
        the cell. Note that this tolerance uses an L1
        distance (aka 'manhatten', 'taxicab' or rectilinear distance) so
        will scale with the dimension of the mesh. The default is the parent
        mesh's ``tolerance`` property. Changing this from default will
        cause the parent mesh's spatial index to be rebuilt which can take some
        time.
    redundant : ``bool``
        If True, the embedding will be done using only the points specified on
        MPI rank 0.
    exclude_halos : ``bool``
        If True, the embedding will be done using only the points specified on
        the locally owned mesh partition.

    Returns
    -------
    coords : ``np.ndarray``
        The coordinates of the points that were embedded.
    coords_idxs : ``np.ndarray``
        The indices of the points that were embedded.
    reference_coords : ``np.ndarray``
        The reference coordinates of the points that were embedded.
    parent_cell_nums : ``np.ndarray``
        The parent cell indices (as given by ``locate_cell``) of the points
        that were embedded.
    ranks : ``np.ndarray``
        The MPI rank of the process that owns the parent cell of the points.
    missing_coords_idxs : ``np.ndarray``
        The indices of the points in the input coords array that were not
        embedded. See note below.

    Notes
    -----
    When redundant is False, it is assumed that all points given are unique.
    If, however, any detected points are not identified as being in the locally
    owned mesh partition (i.e. not in the halo) then an error is raised. This
    is because we do not have point redistribution implemented yet.
    """

    import firedrake.functionspace as functionspace
    import firedrake.constant as constant
    import firedrake.interpolation as interpolation

    # In parallel, we need to make sure we know which point is which and save
    # it.
    if redundant:
        # rank 0 broadcasts coords to all ranks
        coords_local = parent_mesh._comm.bcast(coords, root=0)
        ncoords_local = coords_local.shape[0]
        coords_global = coords_local
        ncoords_global = coords_global.shape[0]
        coords_idxs = np.arange(coords_global.shape[0])
    else:
        # Here, we have to assume that all points we can see are unique.
        # We therefore gather all points on all ranks in rank order: if rank 0
        # has 10 points, rank 1 has 20 points, and rank 3 has 5 points, then
        # rank 0's points have global numbering 0-9, rank 1's points have
        # global numbering 10-29, and rank 3's points have global numbering
        # 30-34.
        coords_local = coords
        ncoords_local = coords.shape[0]
        ncoords_local_allranks = parent_mesh._comm.allgather(ncoords_local)
        ncoords_global = sum(ncoords_local_allranks)
        # The below code looks complicated but it's just an allgather of the
        # (variable length) coords_local array such that they are concatenated.
        coords_local_size = np.array(coords_local.size)
        coords_local_sizes = np.empty(parent_mesh._comm.size, dtype=int)
        parent_mesh._comm.Allgatherv(coords_local_size, coords_local_sizes)
        coords_global = np.empty(
            (ncoords_global, coords.shape[1]), dtype=coords_local.dtype
        )
        parent_mesh._comm.Allgatherv(coords_local, (coords_global, coords_local_sizes))
        # # ncoords_local_allranks is in rank order so we can just sum up the
        # # previous ranks to get the starting index for the global numbering.
        # # For rank 0 we make use of the fact that sum([]) = 0.
        # startidx = sum(ncoords_local_allranks[:parent_mesh._comm.rank])
        # endidx = startidx + ncoords_local
        # coords_idxs = np.arange(startidx, endidx)
        coords_idxs = np.arange(coords_global.shape[0])

    # Get parent mesh rank ownership information:
    # Interpolating Constant(parent_mesh.comm.rank) into P0DG cleverly creates
    # a Function whose dat contains rank ownership information in an ordering
    # that is accessible using Firedrake's cell numbering. This is because, on
    # each rank, parent_mesh.comm.rank creates a Constant with the local rank
    # number, and halo exchange ensures that this information is visible, as
    # nessesary, to other processes.
    P0DG = functionspace.FunctionSpace(parent_mesh, "DG", 0)
    visible_ranks = interpolation.interpolate(
        constant.Constant(parent_mesh.comm.rank), P0DG
    ).dat.data_ro_with_halos.real

    locally_visible = np.full(ncoords_global, False)
    # See below for why np.inf is used here.
    ranks = np.full(ncoords_global, np.inf)

    (
        parent_cell_nums,
        reference_coords,
        ref_cell_dists_l1,
    ) = parent_mesh.locate_cells_ref_coords_and_dists(coords_global, tolerance)
    assert len(parent_cell_nums) == ncoords_global
    assert len(reference_coords) == ncoords_global
    assert len(ref_cell_dists_l1) == ncoords_global

    locally_visible[:] = parent_cell_nums != -1
    ranks[locally_visible] = visible_ranks[parent_cell_nums[locally_visible]]
    # see below for why np.inf is used here.
    ref_cell_dists_l1[~locally_visible] = np.inf
    ref_cell_dists_l1_and_ranks = np.stack((ref_cell_dists_l1, ranks), axis=1)

    # In parallel there will regularly be disagreements about which cell owns a
    # point when those points are close to mesh partition boundaries.
    # We now have the reference cell l1 distance and ranks being np.inf for any
    # point which is not locally visible. By collectively taking the minimum
    # of the reference cell l1 distance, which is tied to the rank via
    # ref_cell_dists_l1_and_ranks, we both check which cell the coordinate is
    # closest to and find out which rank owns that cell.
    # In cases where the reference cell l1 distance is the same for a
    # particular coordinate, we break the tie by choosing the lowest rank.
    # This turns out to be a lexicographic row-wise minimum of the
    # ref_cell_dists_l1_and_ranks array: we minimise the distance first and
    # break ties by minimising the distance.
    ref_cell_dists_l1_and_ranks = parent_mesh.comm.allreduce(
        ref_cell_dists_l1_and_ranks, op=array_lexicographic_mpi_op
    )

    ranks = ref_cell_dists_l1_and_ranks[:, 1]

    # Any ranks which are still np.inf are not in the mesh
    missing_coords_idxs = np.where(ranks == np.inf)[0]

    off_rank_coords_idxs = np.where(ranks != parent_mesh.comm.rank)[0]
    if exclude_halos:
        locally_visible[off_rank_coords_idxs] = False

    # Drop points which are not locally visible but leave the missing coords
    # indices intact for inspection (so that we can tell how many are lost)
    return (
        np.compress(locally_visible, coords_global, axis=0),
        np.compress(locally_visible, coords_idxs, axis=0),
        np.compress(locally_visible, reference_coords, axis=0),
        np.compress(locally_visible, parent_cell_nums, axis=0),
        np.compress(locally_visible, ranks, axis=0),
        missing_coords_idxs,
    )


def RelabeledMesh(mesh, indicator_functions, subdomain_ids, **kwargs):
    """Construct a new mesh that has new subdomain ids.

    :arg mesh: base :class:`~.MeshGeometry` object using which the
        new one is constructed.
    :arg indicator_functions: list of indicator functions that mark
        selected entities (cells or facets) as 1; must use
        "DP"/"DQ" (degree 0) functions to mark cell entities and
        "P" (degree 1) functions in 1D or "HDiv Trace" (degree 0) functions
        in 2D or 3D to mark facet entities.
    :arg subdomain_ids: list of subdomain ids associated with
        the indicator functions in indicator_functions; thus,
        must have the same length as indicator_functions.
    :kwarg name: optional name of the output mesh object.
    """
    import firedrake.function as function

    if not isinstance(mesh, MeshGeometry):
        raise TypeError(f"mesh must be a MeshGeometry, not a {type(mesh)}")
    tmesh = mesh.topology
    if isinstance(tmesh, VertexOnlyMeshTopology):
        raise NotImplementedError("Currently does not work with VertexOnlyMesh")
    elif isinstance(tmesh, ExtrudedMeshTopology):
        raise NotImplementedError("Currently does not work with ExtrudedMesh; use RelabeledMesh() on the base mesh and then extrude")
    if not isinstance(indicator_functions, Sequence) or \
       not isinstance(subdomain_ids, Sequence):
        raise ValueError("indicator_functions and subdomain_ids must be `list`s or `tuple`s of the same length")
    if len(indicator_functions) != len(subdomain_ids):
        raise ValueError("indicator_functions and subdomain_ids must be `list`s or `tuple`s of the same length")
    if len(indicator_functions) == 0:
        raise RuntimeError("At least one indicator function must be given")
    for f in indicator_functions:
        if not isinstance(f, function.Function):
            raise TypeError(f"indicator functions must be instances of function.Function: got {type(f)}")
        if f.function_space().mesh() is not mesh:
            raise ValueError(f"indicator functions must be defined on {mesh}")
    for subid in subdomain_ids:
        if not isinstance(subid, numbers.Integral):
            raise TypeError(f"subdomain id must be an integer: got {subid}")
    name1 = kwargs.get("name", DEFAULT_MESH_NAME)
    plex = tmesh.topology_dm
    # Clone plex: plex1 will share topology with plex.
    plex1 = plex.clone()
    plex1.setName(_generate_default_mesh_topology_name(name1))
    # Remove pyop2 labels.
    plex1.removeLabel("pyop2_core")
    plex1.removeLabel("pyop2_owned")
    plex1.removeLabel("pyop2_ghost")
    # Do not remove "exterior_facets" and "interior_facets" labels;
    # those should be reused as the mesh has already been distributed (if size > 1).
    for label_name in [dmcommon.CELL_SETS_LABEL, dmcommon.FACE_SETS_LABEL]:
        if not plex1.hasLabel(label_name):
            plex1.createLabel(label_name)
    for f, subid in zip(indicator_functions, subdomain_ids):
        elem = f.topological.function_space().ufl_element()
        if elem.value_shape() != ():
            raise RuntimeError(f"indicator functions must be scalar: got {elem.value_shape()} != ()")
        if elem.family() in {"Discontinuous Lagrange", "DQ"} and elem.degree() == 0:
            # cells
            height = 0
            dmlabel_name = dmcommon.CELL_SETS_LABEL
        elif (elem.family() == "HDiv Trace" and elem.degree() == 0 and mesh.topological_dimension() > 1) or \
                (elem.family() == "Lagrange" and elem.degree() == 1 and mesh.topological_dimension() == 1):
            # facets
            height = 1
            dmlabel_name = dmcommon.FACE_SETS_LABEL
        else:
            raise ValueError(f"indicator functions must be 'DP' or 'DQ' (degree 0) to mark cells and 'P' (degree 1) in 1D or 'HDiv Trace' (degree 0) in 2D or 3D to mark facets: got (family, degree) = ({elem.family()}, {elem.degree()})")
        # Clear label stratum; this is a copy, so safe to change.
        plex1.clearLabelStratum(dmlabel_name, subid)
        dmlabel = plex1.getLabel(dmlabel_name)
        section = f.topological.function_space().dm.getSection()
        dmcommon.mark_points_with_function_array(plex, section, height, f.dat.data_ro_with_halos.real.astype(IntType), dmlabel, subid)
    distribution_parameters_noop = {"partition": False,
                                    "overlap_type": (DistributedMeshOverlapType.NONE, 0)}
    reorder_noop = None
    tmesh1 = MeshTopology(plex1, name=plex1.getName(), reorder=reorder_noop,
                          distribution_parameters=distribution_parameters_noop,
                          perm_is=tmesh._plex_renumbering,
                          distribution_name=tmesh._distribution_name,
                          permutation_name=tmesh._permutation_name,
                          comm=tmesh.comm)
    return make_mesh_from_mesh_topology(tmesh1, name1)


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
    m = extract_unique_domain(geometric_expr)

    # Find selected cells
    fs = functionspace.FunctionSpace(m, 'DG', 0)
    f = projection.project(ufl.conditional(geometric_expr, 1, 0), fs)

    # Create cell subset
    indices, = np.nonzero(f.dat.data_ro_with_halos > 0.5)
    return op2.Subset(m.cell_set, indices)
