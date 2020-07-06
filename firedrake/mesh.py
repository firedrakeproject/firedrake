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
from pyop2.datatypes import IntType, RealType
from pyop2 import op2
from pyop2.base import DataSet
from pyop2.mpi import COMM_WORLD, dup_comm
from pyop2.profiling import timed_function, timed_region
from pyop2.utils import as_tuple, tuplify

import firedrake.cython.dmcommon as dmcommon
import firedrake.expression as expression
import firedrake.cython.extrusion_numbering as extnum
import firedrake.extrusion_utils as eutils
import firedrake.cython.spatialindex as spatialindex
import firedrake.utils as utils
from firedrake.interpolation import interpolate
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


def _from_gmsh(filename, comm=None):
    """Read a Gmsh .msh file from `filename`.

    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    comm = comm or COMM_WORLD
    # Create a read-only PETSc.Viewer
    gmsh_viewer = PETSc.Viewer().create(comm=comm)
    gmsh_viewer.setType("ascii")
    gmsh_viewer.setFileMode("r")
    gmsh_viewer.setFileName(filename)
    gmsh_plex = PETSc.DMPlex().createGmsh(gmsh_viewer, comm=comm)

    return gmsh_plex


def _from_exodus(filename, comm):
    """Read an Exodus .e or .exo file from `filename`.

    :arg comm: communicator to build the mesh on.
    """
    plex = PETSc.DMPlex().createExodusFromFile(filename, comm=comm)

    return plex


def _from_cgns(filename, comm):
    """Read a CGNS .cgns file from `filename`.

    :arg comm: communicator to build the mesh on.
    """
    plex = PETSc.DMPlex().createCGNSFromFile(filename, comm=comm)
    return plex


def _from_triangle(filename, dim, comm):
    """Read a set of triangle mesh files from `filename`.

    :arg dim: The embedding dimension.
    :arg comm: communicator to build the mesh on.
    """
    basename, ext = os.path.splitext(filename)

    if comm.rank == 0:
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
        comm.bcast(tdim, root=0)

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
        tdim = comm.bcast(None, root=0)
        cells = None
        coordinates = None
    plex = _from_cell_list(tdim, cells, coordinates, comm=comm)

    # Apply boundary IDs
    if comm.rank == 0:
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


def _from_cell_list(dim, cells, coords, comm):
    """
    Create a DMPlex from a list of cells and coords.

    :arg dim: The topological dimension of the mesh
    :arg cells: The vertices of each cell
    :arg coords: The coordinates of each vertex
    :arg comm: communicator to build the mesh on.
    """
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
    return plex


class AbstractMeshTopology(object, metaclass=abc.ABCMeta):
    """A representation of an abstract mesh topology without a concrete
        PETSc DM implementation"""

    def __init__(self, name):
        """Initialise an abstract mesh topology.

        :arg name: name of the mesh
        """

        utils._init()

        self.name = name

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

    layers = None
    """No layers on unstructured mesh"""

    variable_layers = False
    """No variable layers on unstructured mesh"""

    @property
    def comm(self):
        pass

    def mpi_comm(self):
        """The MPI communicator this mesh is built on (an mpi4py object)."""
        return self.comm

    @timed_function("CreateMesh")
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
        """Returns a :class:`op2.Dat` that maps from a cell index to the local
        facet types on each cell, including the relevant subdomain markers.

        The `i`-th local facet on a cell with index `c` has data
        `cell_facet[c][i]`. The local facet is exterior if
        `cell_facet[c][i][0] == 0`, and interior if the value is `1`.
        The value `cell_facet[c][i][1]` returns the subdomain marker of the
        facet.
        """
        pass

    def create_section(self, nodes_per_entity, real_tensorproduct=False):
        """Create a PETSc Section describing a function space.

        :arg nodes_per_entity: number of function space nodes per topological entity.
        :returns: a new PETSc Section.
        """
        return dmcommon.create_section(self, nodes_per_entity, on_base=real_tensorproduct)

    def node_classes(self, nodes_per_entity, real_tensorproduct=False):
        """Compute node classes given nodes per entity.

        :arg nodes_per_entity: number of function space nodes per topological entity.
        :returns: the number of nodes in each of core, owned, and ghost classes.
        """
        return tuple(np.dot(nodes_per_entity, self._entity_classes))

    def make_cell_node_list(self, global_numbering, entity_dofs, offsets):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg entity_dofs: FInAT element entity DoFs
        :arg offsets: layer offsets for each entity dof (may be None).
        """
        return dmcommon.get_cell_nodes(self, global_numbering,
                                       entity_dofs, offsets)

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

    def cell_orientations(self):
        """Return the orientation of each cell in the mesh.

        Use :func:`init_cell_orientations` on the mesh *geometry* to initialise."""
        if not hasattr(self, '_cell_orientations'):
            raise RuntimeError("No cell orientations found, did you forget to call init_cell_orientations?")
        return self._cell_orientations

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

         :returns: A :class:`pyop2.Subset` for iteration.
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
                ids = tuple(dmcommon.get_cell_markers(self._topology_dm,
                                                      self._cell_numbering,
                                                      sid)
                            for sid in all_integer_subdomain_ids)
                to_remove = np.unique(np.concatenate(ids))
                indices = np.arange(self.cell_set.total_size, dtype=IntType)
                indices = np.delete(indices, to_remove)
            else:
                indices = dmcommon.get_cell_markers(self._topology_dm,
                                                    self._cell_numbering,
                                                    subdomain_id)
            return self._subsets.setdefault(key, op2.Subset(self.cell_set, indices))

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

         :returns: A :class:`pyop2.Subset` for iteration.
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


class MeshTopology(AbstractMeshTopology):
    """A representation of mesh topology implemented on a PETSc DMPlex."""

    @timed_function("CreateMesh")
    def __init__(self, plex, name, reorder, distribution_parameters):
        """Half-initialise a mesh topology.

        :arg plex: :class:`DMPlex` representing the mesh topology
        :arg name: name of the mesh
        :arg reorder: whether to reorder the mesh (bool)
        :arg distribution_parameters: options controlling mesh
            distribution, see :func:`Mesh` for details.
        """

        super().__init__(name)

        # Do some validation of the input mesh
        distribute = distribution_parameters.get("partition")
        self._distribution_parameters = distribution_parameters.copy()
        if distribute is None:
            distribute = True

        overlap_type, overlap = distribution_parameters.get("overlap_type",
                                                            (DistributedMeshOverlapType.FACET, 1))

        if overlap < 0:
            raise ValueError("Overlap depth must be >= 0")
        if overlap_type == DistributedMeshOverlapType.NONE:
            def add_overlap():
                pass
            if overlap > 0:
                raise ValueError("Can't have NONE overlap with overlap > 0")
        elif overlap_type == DistributedMeshOverlapType.FACET:
            def add_overlap():
                dmcommon.set_adjacency_callback(self._topology_dm)
                self._topology_dm.distributeOverlap(overlap)
                dmcommon.clear_adjacency_callback(self._topology_dm)
                self._grown_halos = True
        elif overlap_type == DistributedMeshOverlapType.VERTEX:
            def add_overlap():
                # Default is FEM (vertex star) adjacency.
                self._topology_dm.distributeOverlap(overlap)
                self._grown_halos = True
        else:
            raise ValueError("Unknown overlap type %r" % overlap_type)

        dmcommon.validate_mesh(plex)
        plex.setFromOptions()

        self._topology_dm = plex
        self._comm = dup_comm(plex.comm.tompi4py())

        # Mark exterior and interior facets
        # Note.  This must come before distribution, because otherwise
        # DMPlex will consider facets on the domain boundary to be
        # exterior, which is wrong.
        label_boundary = (self.comm.size == 1) or distribute
        dmcommon.label_facets(plex, label_boundary=label_boundary)

        # Distribute the dm to all ranks
        if self.comm.size > 1 and distribute:
            # We distribute with overlap zero, in case we're going to
            # refine this mesh in parallel.  Later, when we actually use
            # it, we grow the halo.
            partitioner = plex.getPartitioner()
            if IntType.itemsize == 8:
                # Default to PTSCOTCH on 64bit ints (Chaco is 32 bit int only)
                from firedrake_configuration import get_config
                if get_config().get("options", {}).get("with_parmetis", False):
                    partitioner.setType(partitioner.Type.PARMETIS)
                else:
                    partitioner.setType(partitioner.Type.PTSCOTCH)
            else:
                partitioner.setType(partitioner.Type.CHACO)
            try:
                sizes, points = distribute
                partitioner.setType(partitioner.Type.SHELL)
                partitioner.setShellPartition(self.comm.size, sizes, points)
            except TypeError:
                pass
            partitioner.setFromOptions()
            plex.distribute(overlap=0)

        tdim = plex.getDimension()

        # Allow empty local meshes on a process
        cStart, cEnd = plex.getHeightStratum(0)  # cells
        if cStart == cEnd:
            nfacets = -1
        else:
            nfacets = plex.getConeSize(cStart)

        # TODO: this needs to be updated for mixed-cell meshes.
        nfacets = self.comm.allreduce(nfacets, op=MPI.MAX)

        # Note that the geometric dimension of the cell is not set here
        # despite it being a property of a UFL cell. It will default to
        # equal the topological dimension.
        # Firedrake mesh topologies, by convention, which specifically
        # represent a mesh topology (as here) have geometric dimension
        # equal their topological dimension. This is reflected in the
        # corresponding UFL mesh.
        cell = ufl.Cell(_cells[tdim][nfacets])
        self._ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension()))

        def callback(self):
            """Finish initialisation."""
            del self._callback
            if self.comm.size > 1:
                add_overlap()

            if reorder:
                with timed_region("Mesh: reorder"):
                    old_to_new = self._topology_dm.getOrdering(PETSc.Mat.OrderingType.RCM).indices
                    reordering = np.empty_like(old_to_new)
                    reordering[old_to_new] = np.arange(old_to_new.size, dtype=old_to_new.dtype)
            else:
                # No reordering
                reordering = None
            self._did_reordering = bool(reorder)

            # Mark OP2 entities and derive the resulting Plex renumbering
            with timed_region("Mesh: numbering"):
                dmcommon.mark_entity_classes(self._topology_dm)
                self._entity_classes = dmcommon.get_entity_classes(self._topology_dm).astype(int)
                self._plex_renumbering = dmcommon.plex_renumbering(self._topology_dm,
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
                self._facet_ordering = dmcommon.get_facet_ordering(self._topology_dm, facet_numbering)
        self._callback = callback

    @property
    def comm(self):
        return self._comm

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        plex = self._topology_dm
        tdim = plex.getDimension()

        # Cell numbering and global vertex numbering
        cell_numbering = self._cell_numbering
        vertex_numbering = self._vertex_numbering.createGlobalSection(plex.getPointSF())

        cell = self.ufl_cell()
        assert tdim == cell.topological_dimension()
        if cell.is_simplex():
            import FIAT
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

        else:
            raise NotImplementedError("Cell type '%s' not supported." % cell)

    def _facets(self, kind):
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)

        dm = self._topology_dm
        facets, classes = dmcommon.get_facets_by_class(dm, (kind + "_facets").encode(),
                                                       self._facet_ordering)
        label = dmcommon.FACE_SETS_LABEL
        if dm.hasLabel(label):
            from mpi4py import MPI
            markers = dmcommon.get_facet_markers(dm, facets)
            local_markers = set(dm.getLabelIdIS(label).indices)

            def merge_ids(x, y, datatype):
                return x.union(y)

            op = MPI.Op.Create(merge_ids, commute=True)

            unique_markers = np.asarray(sorted(self.comm.allreduce(local_markers, op=op)),
                                        dtype=IntType)
            op.Free()
        else:
            markers = None
            unique_markers = None

        local_facet_number, facet_cell = \
            dmcommon.facet_numbering(dm, kind, facets,
                                     self._cell_numbering,
                                     self.cell_closure)

        point2facetnumber = np.full(facets.max(initial=0)+1, -1, dtype=IntType)
        point2facetnumber[facets] = np.arange(len(facets), dtype=IntType)
        obj = _Facets(self, classes, kind,
                      facet_cell, local_facet_number,
                      markers, unique_markers=unique_markers)
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
        """Returns a :class:`op2.Dat` that maps from a cell index to the local
        facet types on each cell, including the relevant subdomain markers.

        The `i`-th local facet on a cell with index `c` has data
        `cell_facet[c][i]`. The local facet is exterior if
        `cell_facet[c][i][0] == 0`, and interior if the value is `1`.
        The value `cell_facet[c][i][1]` returns the subdomain marker of the
        facet.
        """
        cell_facets = dmcommon.cell_facet_labeling(self._topology_dm,
                                                   self._cell_numbering,
                                                   self.cell_closure)
        if isinstance(self.cell_set, op2.ExtrudedSet):
            dataset = DataSet(self.cell_set.parent, dim=cell_facets.shape[1:])
        else:
            dataset = DataSet(self.cell_set, dim=cell_facets.shape[1:])
        return op2.Dat(dataset, cell_facets, dtype=cell_facets.dtype,
                       name="cell-to-local-facet-dat")

    def num_cells(self):
        cStart, cEnd = self._topology_dm.getHeightStratum(0)
        return cEnd - cStart

    def num_facets(self):
        fStart, fEnd = self._topology_dm.getHeightStratum(1)
        return fEnd - fStart

    def num_faces(self):
        fStart, fEnd = self._topology_dm.getDepthStratum(2)
        return fEnd - fStart

    def num_edges(self):
        eStart, eEnd = self._topology_dm.getDepthStratum(1)
        return eEnd - eStart

    def num_vertices(self):
        vStart, vEnd = self._topology_dm.getDepthStratum(0)
        return vEnd - vStart

    def num_entities(self, d):
        eStart, eEnd = self._topology_dm.getDepthStratum(d)
        return eEnd - eStart

    @utils.cached_property
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "Cells", comm=self.comm)


class ExtrudedMeshTopology(MeshTopology):
    """Representation of an extruded mesh topology."""

    def __init__(self, mesh, layers):
        """Build an extruded mesh topology from an input mesh topology

        :arg mesh:           the unstructured base mesh topology
        :arg layers:         number of extruded cell layers in the "vertical"
                             direction.
        """

        # TODO: refactor to call super().__init__

        from firedrake_citations import Citations
        Citations().register("McRae2016")
        Citations().register("Bercea2016")
        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)

        if isinstance(mesh.topology, VertexOnlyMeshTopology):
            raise NotImplementedError("Extrusion not implemented for VertexOnlyMeshTopology")

        mesh.init()
        self._base_mesh = mesh
        self._comm = mesh.comm
        # TODO: These attributes are copied so that FunctionSpaceBase can
        # access them directly.  Eventually we would want a better refactoring
        # of responsibilities between mesh and function space.
        self._topology_dm = mesh._topology_dm
        self._plex_renumbering = mesh._plex_renumbering
        self._cell_numbering = mesh._cell_numbering
        self._entity_classes = mesh._entity_classes
        self._subsets = {}
        cell = ufl.TensorProductCell(mesh.ufl_cell(), ufl.interval)
        self._ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension()))
        if layers.shape:
            self.variable_layers = True
            extents = extnum.layer_extents(self._topology_dm,
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
        self.cell_set = op2.ExtrudedSet(mesh.cell_set, layers=layers)

    @property
    def comm(self):
        return self._comm

    @property
    def name(self):
        return self._base_mesh.name

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        return self._base_mesh.cell_closure

    def _facets(self, kind):
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)
        base = getattr(self._base_mesh, "%s_facets" % kind)
        return _Facets(self, base.classes,
                       kind,
                       base.facet_cell,
                       base.local_facet_dat.data_ro_with_halos,
                       markers=base.markers,
                       unique_markers=base.unique_markers)

    def make_cell_node_list(self, global_numbering, entity_dofs, offsets):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg entity_dofs: FInAT element entity DoFs
        :arg offsets: layer offsets for each entity dof.
        """
        entity_dofs = eutils.flat_entity_dofs(entity_dofs)
        return super().make_cell_node_list(global_numbering, entity_dofs, offsets)

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
            nodes_per_entity = sum(nodes[:, i]*(self.layers - i) for i in range(2))
            return super(ExtrudedMeshTopology, self).node_classes(nodes_per_entity)

    def make_offset(self, entity_dofs, ndofs, real_tensorproduct=False):
        """Returns the offset between the neighbouring cells of a
        column for each DoF.

        :arg entity_dofs: FInAT element entity DoFs
        :arg ndofs: number of DoFs in the FInAT element
        """
        entity_offset = [0] * (1 + self._base_mesh.cell_dimension())
        for (b, v), entities in entity_dofs.items():
            entity_offset[b] += len(entities[0])

        dof_offset = np.zeros(ndofs, dtype=IntType)
        if not real_tensorproduct:
            for (b, v), entities in entity_dofs.items():
                for dof_indices in entities.values():
                    for i in dof_indices:
                        dof_offset[i] = entity_offset[b]
        return dof_offset

    @utils.cached_property
    def layers(self):
        """Return the number of layers of the extruded mesh
        represented by the number of occurences of the base mesh."""
        if self.variable_layers:
            raise ValueError("Can't ask for mesh layers with variable layers")
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


# TODO: Could this be merged with MeshTopology given that dmcommon.pyx
# now covers DMSwarms and DMPlexes?
class VertexOnlyMeshTopology(AbstractMeshTopology):
    """
    Representation of a vertex-only mesh topology immersed within
    another mesh.
    """

    def __init__(self, swarm, parentmesh, name, reorder):
        """
        Half-initialise a mesh topology.

        :arg swarm: Particle In Cell (PIC) :class:`DMSwarm` representing
            vertices immersed within a :class:`DMPlex` stored in the
            `parentmesh`
        :arg parentmesh: the mesh within which the vertex-only mesh
            topology is immersed.
        :arg name: name of the mesh
        :arg reorder: whether to reorder the mesh (bool)
        """

        super().__init__(name)

        # TODO: As a performance optimisation, we should renumber the
        # swarm to in parent-cell order so that we traverse efficiently.
        if reorder:
            raise NotImplementedError("Mesh reordering not implemented for vertex only meshes yet.")

        dmcommon.validate_mesh(swarm)
        swarm.setFromOptions()

        self._parent_mesh = parentmesh
        self._topology_dm = swarm
        self._comm = dup_comm(swarm.comm.tompi4py())

        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)

        # Cell subsets for integration over subregions
        self._subsets = {}

        tdim = 0

        cell = ufl.Cell("vertex")
        self._ufl_mesh = ufl.Mesh(ufl.VectorElement("DG", cell, 0, dim=cell.topological_dimension()))

        def callback(self):
            """Finish initialisation."""
            del self._callback

            # Mark OP2 entities and derive the resulting Swarm numbering
            with timed_region("Mesh: numbering"):
                dmcommon.mark_entity_classes(self._topology_dm)
                self._entity_classes = dmcommon.get_entity_classes(self._topology_dm).astype(int)

                # Derive a cell numbering from the Swarm numbering
                entity_dofs = np.zeros(tdim+1, dtype=IntType)
                entity_dofs[-1] = 1

                self._cell_numbering = self.create_section(entity_dofs)
                entity_dofs[:] = 0
                entity_dofs[0] = 1
                self._vertex_numbering = self.create_section(entity_dofs)

        self._callback = callback

    @property
    def comm(self):
        return self._comm

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        swarm = self._topology_dm
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

    def _facets(self, kind):
        """Raises an AttributeError since cells in a
        `VertexOnlyMeshTopology` have no facets.
        """
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)
        raise AttributeError("Cells in a VertexOnlyMeshTopology have no facets.")

    @utils.cached_property
    def exterior_facets(self):
        return self._facets("exterior")

    @utils.cached_property
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
        return self._topology_dm.getLocalSize()

    def num_entities(self, d):
        if d > 0:
            return 0
        else:
            return self.num_vertices()

    @utils.cached_property
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "Cells", comm=self.comm)


class MeshGeometry(ufl.Mesh, MeshGeometryMixin):
    """A representation of mesh topology and geometry."""

    def __new__(cls, element):
        """Create mesh geometry object."""
        utils._init()
        mesh = super(MeshGeometry, cls).__new__(cls)
        mesh.uid = utils._new_uid()
        assert isinstance(element, ufl.FiniteElementBase)
        ufl.Mesh.__init__(mesh, element, ufl_id=mesh.uid)
        return mesh

    @MeshGeometryMixin._ad_annotate_init
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
    @MeshGeometryMixin._ad_annotate_coordinates_function
    def _coordinates_function(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        import firedrake.functionspaceimpl as functionspaceimpl
        import firedrake.function as function
        self.init()

        coordinates_fs = self._coordinates.function_space()
        V = functionspaceimpl.WithGeometry(coordinates_fs, self)
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
        coords_min = function.Function(V)
        coords_max = function.Function(V)

        coords_min.dat.data.fill(np.inf)
        coords_max.dat.data.fill(-np.inf)

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
                 {'f': (self.coordinates, READ),
                  'f_min': (coords_min, MIN),
                  'f_max': (coords_max, MAX)},
                 is_loopy_kernel=True)

        # Reorder bounding boxes according to the cell indices we use
        column_list = V.cell_node_list.reshape(-1)
        coords_min = self._order_data_by_cell_index(column_list, coords_min.dat.data_ro_with_halos)
        coords_max = self._order_data_by_cell_index(column_list, coords_max.dat.data_ro_with_halos)

        # Build spatial index
        return spatialindex.from_regions(coords_min, coords_max)

    def locate_cell(self, x, tolerance=None):
        """Locate cell containg given point.

        :arg x: point coordinates
        :kwarg tolerance: for checking if a point is in a cell.
        :returns: cell number (int), or None (if the point is not in the domain)
        """
        if self.variable_layers:
            raise NotImplementedError("Cell location not implemented for variable layers")
        x = np.asarray(x, dtype=utils.ScalarType)
        cell = self._c_locator(tolerance=tolerance)(self.coordinates._ctypes,
                                                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if cell == -1:
            return None
        else:
            return cell

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
    int locator(struct Function *f, double *x)
    {
        struct ReferenceCoords reference_coords;
        return locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &to_reference_coords_xtr, &reference_coords);
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
                                ctypes.POINTER(ctypes.c_double)]
            locator.restype = ctypes.c_int
            return cache.setdefault(tolerance, locator)

    def init_cell_orientations(self, expr):
        """Compute and initialise :attr:`cell_orientations` relative to a specified orientation.

        :arg expr: an :class:`.Expression` evaluated to produce a
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

        if isinstance(expr, expression.Expression):
            if expr.value_shape()[0] != 3:
                raise NotImplementedError('Only implemented for 3-vectors')

            expr = interpolate(expr, functionspace.VectorFunctionSpace(self, 'DG', 0))
        elif isinstance(expr, ufl.classes.Expr):
            if expr.ufl_shape != (3,):
                raise NotImplementedError('Only implemented for 3-vectors')
        else:
            raise TypeError("UFL expression or Expression object expected!")

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


@timed_function("CreateMesh")
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

    geometric_dim = kwargs.get("dim", None)
    reorder = kwargs.get("reorder", None)
    if reorder is None:
        reorder = parameters["reorder_meshes"]

    distribution_parameters = kwargs.get("distribution_parameters", None)
    if distribution_parameters is None:
        distribution_parameters = {}

    if isinstance(meshfile, PETSc.DMPlex):
        name = "plexmesh"
        plex = meshfile
    else:
        comm = kwargs.get("comm", COMM_WORLD)
        name = meshfile
        basename, ext = os.path.splitext(meshfile)

        if ext.lower() in ['.e', '.exo']:
            plex = _from_exodus(meshfile, comm)
        elif ext.lower() == '.cgns':
            plex = _from_cgns(meshfile, comm)
        elif ext.lower() == '.msh':
            if geometric_dim is not None:
                opts = {"dm_plex_gmsh_spacedim": geometric_dim}
            else:
                opts = {}
            opts = OptionsManager(opts, "")
            with opts.inserted_options():
                plex = _from_gmsh(meshfile, comm)
        elif ext.lower() == '.node':
            plex = _from_triangle(meshfile, geometric_dim, comm)
        else:
            raise RuntimeError("Mesh file %s has unknown format '%s'."
                               % (meshfile, ext[1:]))

    # Create mesh topology
    topology = MeshTopology(plex, name=name, reorder=reorder,
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


@timed_function("CreateExtMesh")
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
    :arg layer_height:   the layer height, assuming all layers are evenly
                         spaced. If this is omitted, the value defaults to
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
        # Convert to internal representation
        layers[:, 1] += 1 + layers[:, 0]
    else:
        if layer_height is None:
            # Default to unit
            layer_height = 1 / layers
        # All internal logic works with layers of base mesh (not layers of cells)
        layers = layers + 1

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


def VertexOnlyMesh(mesh, vertexcoords):
    """
    Create a vertex only mesh, immersed in a given mesh, with vertices
    defined by a list of coordinates.

    :arg mesh: The unstructured mesh in which to immerse the vertex only
        mesh.
    :arg vertexcoords: A list of coordinate tuples which defines the vertices.

    .. note::

        The vertex only mesh uses the same communicator as the input `mesh`.

    .. note::

        Meshes created from a coordinates `firedrake.Function` and immersed
        manifold meshes are not yet supported.
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

    swarm = _pic_swarm_in_plex(mesh.topology._topology_dm, vertexcoords, fields=[("parentcellnum", 1, IntType)])

    dmcommon.label_pic_parent_cell_nums(swarm, mesh)

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
    :arg coords: an `ndarray` of (npoints, coordsdim) shape.
    :kwarg fields: An optional list of named data which can be stored
        for each point in the DMSwarm. The format should be::

        [(fieldname1, blocksize1, dtype1),
          ...,
         (fieldnameN, blocksizeN, dtypeN)]

        For example, the swarm coordinates themselves are stored in a
        field named `DMSwarmPIC_coor` which, were it not created
        automatically, would be initialised with
        ``fields = [("DMSwarmPIC_coor", coordsdim, ScalarType)]``.
        All fields must have the same number of points. For more
        information see https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DMSWARM/DMSWARM.html
    :return: the immersed DMSwarm

    .. note::

        The created DMSwarm uses the communicator of the input DMPlex.
    """

    # Check coords
    coords = np.asarray(coords, dtype=RealType)
    npoints, coordsdim = coords.shape
    if coordsdim == 1:
        raise NotImplementedError("You can't yet use a 1D DMPlex as DMSwarm cellDM.")

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


def SubDomainData(geometric_expr):
    """Creates a subdomain data object from a boolean-valued UFL expression.

    The result can be attached as the subdomain_data field of a
    :class:`ufl.Measure`. For example:

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
