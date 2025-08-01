import numpy as np
import ctypes
import os
import sys
import ufl
import finat.ufl
import FIAT
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from ufl.classes import ReferenceGrad
from ufl.domain import extract_unique_domain
import enum
import numbers
import abc
import rtree
from textwrap import dedent
from pathlib import Path

from pyop2 import op2
from pyop2.mpi import (
    MPI, COMM_WORLD, internal_comm, is_pyop2_comm, temp_internal_comm
)
from pyop2.utils import as_tuple
from petsctools import OptionsManager, get_external_packages

import firedrake.cython.dmcommon as dmcommon
from firedrake.cython.dmcommon import DistributedMeshOverlapType
import firedrake.cython.extrusion_numbering as extnum
import firedrake.extrusion_utils as eutils
import firedrake.cython.spatialindex as spatialindex
import firedrake.utils as utils
from firedrake.utils import as_cstr, IntType, RealType
from firedrake.logging import info_red
from firedrake.parameters import parameters
from firedrake.petsc import PETSc, DEFAULT_PARTITIONER
from firedrake.adjoint_utils import MeshGeometryMixin
from pyadjoint import stop_annotating
import gem

try:
    import netgen
except ImportError:
    netgen = None
    ngsPETSc = None
# Only for docstring
import mpi4py  # noqa: F401
from finat.element_factory import as_fiat_cell


__all__ = [
    'Mesh', 'ExtrudedMesh', 'VertexOnlyMesh', 'RelabeledMesh',
    'SubDomainData', 'unmarked', 'DistributedMeshOverlapType',
    'DEFAULT_MESH_NAME', 'MeshGeometry', 'MeshTopology',
    'AbstractMeshTopology', 'ExtrudedMeshTopology', 'VertexOnlyMeshTopology',
    'VertexOnlyMeshMissingPointsError',
    'Submesh'
]


_cells = {
    0: {0: "vertex"},
    1: {2: "interval"},
    2: {3: "triangle", 4: "quadrilateral"},
    3: {4: "tetrahedron", 6: "hexahedron"}
}


_supported_embedded_cell_types_and_gdims = [('interval', 2),
                                            ('triangle', 3),
                                            ("quadrilateral", 3),
                                            ("interval * interval", 3)]


unmarked = -1
"""A mesh marker that selects all entities that are not explicitly marked."""

DEFAULT_MESH_NAME = "_".join(["firedrake", "default"])
"""The default name of the mesh."""


def _generate_default_submesh_name(name):
    """Generate the default submesh name from the mesh name.

    Parameters
    ----------
    name : str
        Name of the parent mesh.

    Returns
    -------
    str
        Default submesh name.

    """
    return "_".join([name, "submesh"])


def _generate_default_mesh_coordinates_name(name):
    """Generate the default mesh coordinates name from the mesh name.

    :arg name: the mesh name.
    :returns: the default mesh coordinates name.
    """
    return "_".join([name, "coordinates"])


def _generate_default_mesh_reference_coordinates_name(name):
    """Generate the default mesh reference coordinates name from the mesh name.

    :arg name: the mesh name.
    :returns: the default mesh reference coordinates name.
    """
    return "_".join([name, "reference_coordinates"])


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

    @utils.cached_property
    def local_facet_orientation_dat(self):
        """Dat for the local facet orientations."""
        dtype = gem.uint_type
        # Make a map from cell to facet orientations.
        fiat_cell = as_fiat_cell(self.mesh.ufl_cell())
        topo = fiat_cell.topology
        num_entities = [0]
        for d in range(len(topo)):
            num_entities.append(len(topo[d]))
        offsets = np.cumsum(num_entities)
        local_facet_start = offsets[-3]
        local_facet_end = offsets[-2]
        map_from_cell_to_facet_orientations = self.mesh.entity_orientations[:, local_facet_start:local_facet_end]
        # Make output data;
        # this is a map from an exterior/interior facet to the corresponding local facet orientation/orientations.
        # Halo data are required by design, but not actually used.
        # -- Reshape as (-1, self._rank) to uniformly handle exterior and interior facets.
        data = np.empty_like(self.local_facet_dat.data_ro_with_halos).reshape((-1, self._rank))
        data.fill(np.iinfo(dtype).max)
        # Set local facet orientations on the block corresponding to the owned facets; i.e., data[:shape[0], :] below.
        local_facets = self.local_facet_dat.data_ro  # do not need halos.
        # -- Reshape as (-1, self._rank) to uniformly handle exterior and interior facets.
        local_facets = local_facets.reshape((-1, self._rank))
        shape = local_facets.shape
        map_from_owned_facet_to_cells = self.facet_cell[:shape[0], :]
        data[:shape[0], :] = np.take_along_axis(
            map_from_cell_to_facet_orientations[map_from_owned_facet_to_cells],
            local_facets.reshape(shape + (1, )),  # reshape as required by take_along_axis.
            axis=2,
        ).reshape(shape)
        return op2.Dat(
            self.local_facet_dat.dataset,
            data,
            dtype,
            f"{self.mesh.name}_{self.kind}_local_facet_orientation"
        )


@PETSc.Log.EventDecorator()
def _from_gmsh(filename, comm=None):
    """Read a Gmsh .msh file from `filename`.

    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    comm = comm or COMM_WORLD
    gmsh_plex = PETSc.DMPlex().createFromFile(filename, comm=comm)

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

    def __init__(self, topology_dm, name, reorder, sfXB, perm_is, distribution_name, permutation_name, comm, submesh_parent=None):
        """Initialise a mesh topology.

        Parameters
        ----------
        topology_dm : PETSc.DMPlex or PETSc.DMSwarm
            `PETSc.DMPlex` or `PETSc.DMSwarm` representing the mesh topology.
        name : str
            Name of the mesh topology.
        reorder : bool
            Whether to reorder the mesh entities.
        sfXB : PETSc.PetscSF
            `PETSc.SF` that pushes forward the global point number
            slab ``[0, NX)`` to input (naive) plex (only significant when
            the mesh topology is loaded from file and only passed from inside
            `~.CheckpointFile`).
        perm_is : PETSc.IS
            `PETSc.IS` that is used as ``_dm_renumbering``; only
            makes sense if we know the exact parallel distribution of ``plex``
            at the time of mesh topology construction like when we load mesh
            along with its distribution. If given, ``reorder`` param will be ignored.
        distribution_name : str
            Name of the parallel distribution; if `None`, automatically generated.
        permutation_name : str
            Name of the entity permutation (reordering); if `None`, automatically generated.
        comm : mpi4py.MPI.Comm
            Communicator.
        submesh_parent: AbstractMeshTopology
            Submesh parent.

        """
        utils._init()
        dmcommon.validate_mesh(topology_dm)
        topology_dm.setFromOptions()
        self.topology_dm = topology_dm
        r"The PETSc DM representation of the mesh topology."
        self.sfBC = None
        r"The PETSc SF that pushes the input (naive) plex to current (good) plex."
        self.sfXB = sfXB
        r"The PETSc SF that pushes the global point number slab [0, NX) to input (naive) plex."
        self.submesh_parent = submesh_parent
        # User comm
        self.user_comm = comm
        # Internal comm
        self._comm = internal_comm(self.user_comm, self)
        dmcommon.label_facets(self.topology_dm)
        self._distribute()
        self._grown_halos = False
        if self.comm.size > 1:
            self._add_overlap()
        if self.sfXB is not None:
            self.sfXC = sfXB.compose(self.sfBC) if self.sfBC else self.sfXB
        dmcommon.label_facets(self.topology_dm)
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
        # Mark OP2 entities and derive the resulting Plex renumbering
        with PETSc.Log.Event("Mesh: numbering"):
            self._mark_entity_classes()
            self._entity_classes = dmcommon.get_entity_classes(self.topology_dm).astype(int)
            if perm_is:
                self._dm_renumbering = perm_is
            else:
                self._dm_renumbering = self._renumber_entities(reorder)
            self._did_reordering = bool(reorder)
            # Derive a cell numbering from the Plex renumbering
            tdim = dmcommon.get_topological_dimension(self.topology_dm)
            entity_dofs = np.zeros(tdim+1, dtype=IntType)
            entity_dofs[-1] = 1
            self._cell_numbering, _ = self.create_section(entity_dofs)
            if tdim == 0:
                self._vertex_numbering = self._cell_numbering
            else:
                entity_dofs[:] = 0
                entity_dofs[0] = 1
                self._vertex_numbering, _ = self.create_section(entity_dofs)
                entity_dofs[:] = 0
                entity_dofs[-2] = 1
                facet_numbering, _ = self.create_section(entity_dofs)
                self._facet_ordering = dmcommon.get_facet_ordering(self.topology_dm, facet_numbering)
        self.name = name
        # Set/Generate names to be used when checkpointing.
        self._distribution_name = distribution_name or _generate_default_mesh_topology_distribution_name(self.topology_dm.comm.size, self._distribution_parameters)
        self._permutation_name = permutation_name or _generate_default_mesh_topology_permutation_name(reorder)
        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)
        # Cell subsets for integration over subregions
        self._subsets = {}
        # A set of weakrefs to meshes that are explicitly labelled as being
        # parallel-compatible for interpolation/projection/supermeshing
        # To set, do e.g.
        # target_mesh._parallel_compatible = {weakref.ref(source_mesh)}
        self._parallel_compatible = None

    layers = None
    """No layers on unstructured mesh"""

    variable_layers = False
    """No variable layers on unstructured mesh"""

    @abc.abstractmethod
    def _distribute(self):
        """Distribute the mesh toplogy."""
        pass

    @abc.abstractmethod
    def _add_overlap(self):
        """Add overlap."""
        pass

    @abc.abstractmethod
    def _mark_entity_classes(self):
        """Mark entities with pyop2 classes."""
        pass

    @abc.abstractmethod
    def _renumber_entities(self, reorder):
        """Renumber entities."""
        pass

    @property
    def comm(self):
        return self.user_comm

    def mpi_comm(self):
        """The MPI communicator this mesh is built on (an mpi4py object)."""
        return self.comm

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
        return self._ufl_cell

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
    def dm_cell_types(self):
        """All DM.PolytopeTypes of cells in the mesh."""
        pass

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

    def create_section(self, nodes_per_entity, real_tensorproduct=False, block_size=1, boundary_set=None):
        """Create a PETSc Section describing a function space.

        :arg nodes_per_entity: number of function space nodes per topological entity.
        :arg real_tensorproduct: If True, assume extruded space is actually Foo x Real.
        :arg block_size: The integer by which nodes_per_entity is uniformly multiplied
            to get the true data layout.
        :arg boundary_set: A set of boundary markers, indicating the subdomains
            a boundary condition is specified on.
        :returns: a new PETSc Section.
        """
        return dmcommon.create_section(self, nodes_per_entity, on_base=real_tensorproduct, block_size=block_size, boundary_set=boundary_set)

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

    @abc.abstractmethod
    def mark_entities(self, tf, label_value, label_name=None):
        """Mark selected entities.

        :arg tf: The :class:`.CoordinatelessFunction` object that marks
            selected entities as 1. f.function_space().ufl_element()
            must be "DP" or "DQ" (degree 0) to mark cell entities and
            "P" (degree 1) in 1D or "HDiv Trace" (degree 0) in 2D or 3D
            to mark facet entities.
            Can use "Q" (degree 2) functions for 3D hex meshes until
            we support "HDiv Trace" elements on hex.
        :arg lable_value: The value used in the label.
        :arg label_name: The name of the label to store entity selections.

        All entities must live on the same topological dimension. Currently,
        one can only mark cell or facet entities.
        """
        pass

    @utils.cached_property
    def extruded_periodic(self):
        return self.cell_set._extruded_periodic

    # submesh

    @utils.cached_property
    def submesh_ancesters(self):
        """Tuple of submesh ancesters."""
        if self.submesh_parent:
            return (self, ) + self.submesh_parent.submesh_ancesters
        else:
            return (self, )

    def submesh_youngest_common_ancester(self, other):
        """Return the youngest common ancester of self and other.

        Parameters
        ----------
        other : AbstractMeshTopology
            The other mesh.

        Returns
        -------
        AbstractMeshTopology or None
            Youngest common ancester or None if not found.

        """
        # self --- ... --- m --- common --- common --- common
        #                          /
        #       other --- ... --- m
        self_ancesters = list(self.submesh_ancesters)
        other_ancesters = list(other.submesh_ancesters)
        c = None
        while self_ancesters and other_ancesters:
            a = self_ancesters.pop()
            b = other_ancesters.pop()
            if a is b:
                c = a
            else:
                break
        return c

    def submesh_map_child_parent(self, source_integral_type, source_subset_points, reverse=False):
        """Return the map from submesh child entities to submesh parent entities or its reverse.

        Parameters
        ----------
        source_integral_type : str
            Integral type on the source mesh.
        source_subset_points : numpy.ndarray
            Subset points on the source mesh.
        reverse : bool
            If True, return the map from parent entities to child entities.

        Returns
        -------
        tuple
           (map from source to target, integral type on the target mesh, subset points on the target mesh).

        """
        raise NotImplementedError(f"Not implemented for {type(self)}")

    def submesh_map_composed(self, other, other_integral_type, other_subset_points):
        """Create entity-entity map from ``other`` to `self`.

        Parameters
        ----------
        other : AbstractMeshTopology
            Base mesh topology.
        other_integral_type : str
            Integral type on ``other``.
        other_subset_points : numpy.ndarray
            Subset points on ``other``; only used to identify (facet) integral_type on ``self``.

        Returns
        -------
        tuple
            Tuple of `op2.ComposedMap` from other to self, integral_type on self, and points on self.

        """
        common = self.submesh_youngest_common_ancester(other)
        if common is None:
            raise ValueError(f"Unable to create composed map between (sub)meshes: {self} and {other} are unrelated")
        maps = []
        integral_type = other_integral_type
        subset_points = other_subset_points
        aa = other.submesh_ancesters
        for a in aa[:aa.index(common)]:
            m, integral_type, subset_points = a.submesh_map_child_parent(integral_type, subset_points)
            maps.append(m)
        bb = self.submesh_ancesters
        for b in reversed(bb[:bb.index(common)]):
            m, integral_type, subset_points = b.submesh_map_child_parent(integral_type, subset_points, reverse=True)
            maps.append(m)
        return op2.ComposedMap(*reversed(maps)), integral_type, subset_points

    # trans mesh

    def trans_mesh_entity_map(self, base_mesh, base_integral_type, base_subdomain_id, base_all_integer_subdomain_ids):
        """Create entity-entity (composed) map from base_mesh to `self`.

        Parameters
        ----------
        base_mesh : AbstractMeshTopology
            Base mesh topology.
        base_integral_type : str
            Integral type on ``base_mesh``.
        base_subdomain_id : int
            Subdomain ID on ``base_mesh``.
        base_all_integer_subdomain_ids : tuple
            ``all_integer_subdomain_ids`` corresponding to ``base_mesh`` and ``base_integral_type``.

        Returns
        -------
        tuple
            `tuple` of `op2.ComposedMap` from base_mesh to `self` and integral_type on `self`.

        """
        raise NotImplementedError(f"Not implemented for {type(self)}")


class MeshTopology(AbstractMeshTopology):
    """A representation of mesh topology implemented on a PETSc DMPlex."""

    @PETSc.Log.EventDecorator("CreateMesh")
    def __init__(
        self,
        plex,
        name,
        reorder,
        distribution_parameters,
        sfXB=None,
        perm_is=None,
        distribution_name=None,
        permutation_name=None,
        submesh_parent=None,
        comm=COMM_WORLD,
    ):
        """Initialise a mesh topology.

        Parameters
        ----------
        plex : PETSc.DMPlex
            `PETSc.DMPlex` representing the mesh topology.
        name : str
            Name of the mesh topology.
        reorder : bool
            Whether to reorder the mesh entities.
        distribution_parameters : dict
            Options controlling mesh distribution; see `Mesh` for details.
        sfXB : PETSc.PetscSF
            `PETSc.SF` that pushes forward the global point number
            slab ``[0, NX)`` to input (naive) plex (only significant when
            the mesh topology is loaded from file and only passed from inside
            `~.CheckpointFile`).
        perm_is : PETSc.IS
            `PETSc.IS` that is used as ``_dm_renumbering``; only
            makes sense if we know the exact parallel distribution of ``plex``
            at the time of mesh topology construction like when we load mesh
            along with its distribution. If given, ``reorder`` param will be ignored.
        distribution_name : str
            Name of the parallel distribution; if `None`, automatically generated.
        permutation_name : str
            Name of the entity permutation (reordering); if `None`, automatically generated.
        submesh_parent: MeshTopology
            Submesh parent.
        comm : mpi4py.MPI.Comm
            Communicator.

        """
        if distribution_parameters is None:
            distribution_parameters = {}
        self._distribution_parameters = {}
        distribute = distribution_parameters.get("partition")
        if distribute is None:
            distribute = True
        self._distribution_parameters["partition"] = distribute
        partitioner_type = distribution_parameters.get("partitioner_type")
        self._set_partitioner(plex, distribute, partitioner_type)
        self._distribution_parameters["partitioner_type"] = self._get_partitioner(plex).getType()
        self._distribution_parameters["overlap_type"] = distribution_parameters.get("overlap_type",
                                                                                    (DistributedMeshOverlapType.FACET, 1))
        # Disable auto distribution and reordering before setFromOptions is called.
        plex.distributeSetDefault(False)
        plex.reorderSetDefault(PETSc.DMPlex.ReorderDefaultFlag.FALSE)
        super().__init__(plex, name, reorder, sfXB, perm_is, distribution_name, permutation_name, comm, submesh_parent=submesh_parent)

    def _distribute(self):
        # Distribute/redistribute the dm to all ranks
        distribute = self._distribution_parameters["partition"]
        if self.comm.size > 1 and distribute:
            plex = self.topology_dm
            # We distribute with overlap zero, in case we're going to
            # refine this mesh in parallel.  Later, when we actually use
            # it, we grow the halo.
            original_name = plex.getName()
            sfBC = plex.distribute(overlap=0)
            plex.setName(original_name)
            self.sfBC = sfBC
            # plex carries a new dm after distribute, which
            # does not inherit partitioner from the old dm.
            # It probably makes sense as chaco does not work
            # once distributed.

    def _add_overlap(self):
        overlap_type, overlap = self._distribution_parameters["overlap_type"]
        if overlap < 0:
            raise ValueError("Overlap depth must be >= 0")
        if overlap_type == DistributedMeshOverlapType.NONE:
            if overlap > 0:
                raise ValueError("Can't have NONE overlap with overlap > 0")
        elif overlap_type in [DistributedMeshOverlapType.FACET, DistributedMeshOverlapType.RIDGE]:
            dmcommon.set_adjacency_callback(self.topology_dm, overlap_type)
            original_name = self.topology_dm.getName()
            sfBC = self.topology_dm.distributeOverlap(overlap)
            self.topology_dm.setName(original_name)
            self.sfBC = self.sfBC.compose(sfBC) if self.sfBC else sfBC
            dmcommon.clear_adjacency_callback(self.topology_dm)
            self._grown_halos = True
        elif overlap_type == DistributedMeshOverlapType.VERTEX:
            # Default is FEM (vertex star) adjacency.
            original_name = self.topology_dm.getName()
            sfBC = self.topology_dm.distributeOverlap(overlap)
            self.topology_dm.setName(original_name)
            self.sfBC = self.sfBC.compose(sfBC) if self.sfBC else sfBC
            self._grown_halos = True
        else:
            raise ValueError("Unknown overlap type %r" % overlap_type)

    def _mark_entity_classes(self):
        dmcommon.mark_entity_classes(self.topology_dm)

    @utils.cached_property
    def _ufl_cell(self):
        plex = self.topology_dm
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
        return ufl.Cell(_cells[tdim][nfacets])

    @utils.cached_property
    def _ufl_mesh(self):
        cell = self._ufl_cell
        return ufl.Mesh(finat.ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension()))

    @property
    def _default_reordering(self):
        with PETSc.Log.Event("Mesh: reorder"):
            old_to_new = self.topology_dm.getOrdering(PETSc.Mat.OrderingType.RCM).indices
            reordering = np.empty_like(old_to_new)
            reordering[old_to_new] = np.arange(old_to_new.size, dtype=old_to_new.dtype)
        return reordering

    def _renumber_entities(self, reorder):
        if reorder:
            reordering = self._default_reordering
        else:
            # No reordering
            reordering = None
        return dmcommon.plex_renumbering(self.topology_dm, self._entity_classes, reordering)

    @property
    def dm_cell_types(self):
        """All DM.PolytopeTypes of cells in the mesh."""
        return dmcommon.get_dm_cell_types(self.topology_dm)

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
        if self.submesh_parent is not None and \
                not (self.submesh_parent.ufl_cell().cellname() == "hexahedron" and cell.cellname() == "quadrilateral"):
            # Codim-1 submesh of a hex mesh (i.e. a quad submesh) can not
            # inherit cell_closure from the hex mesh as the cell_closure
            # must follow the special orientation restriction. This means
            # that, when the quad submesh works with the parent hex mesh,
            # quadrature points must be permuted (i.e. use the canonical
            # quadrature point ordering based on the cone ordering).
            topology = FIAT.ufc_cell(cell).get_topology()
            entity_per_cell = np.zeros(len(topology), dtype=IntType)
            for d, ents in topology.items():
                entity_per_cell[d] = len(ents)
            return dmcommon.submesh_create_cell_closure(
                plex,
                self.submesh_parent.topology_dm,
                cell_numbering,
                self.submesh_parent._cell_numbering,
                self.submesh_parent.cell_closure,
                entity_per_cell,
            )
        elif cell.is_simplex():
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
    def _set_partitioner(self, plex, distribute, partitioner_type=None):
        """Set partitioner for (re)distributing underlying plex over comm.

        :arg distribute: Boolean or (sizes, points)-tuple.  If (sizes, point)-
            tuple is given, it is used to set shell partition. If Boolean, no-op.
        :kwarg partitioner_type: Partitioner to be used: "chaco", "ptscotch", "parmetis",
            "shell", or `None` (unspecified). Ignored if the distribute parameter
            specifies the distribution.
        """
        if plex.comm.size == 1 or distribute is False:
            return
        partitioner = plex.getPartitioner()
        if distribute is True:
            if partitioner_type:
                if partitioner_type not in ["chaco", "ptscotch", "parmetis", "simple", "shell"]:
                    raise ValueError(
                        f"Unexpected partitioner_type: {partitioner_type}")
                if partitioner_type in ["chaco", "ptscotch", "parmetis"] and \
                        partitioner_type not in get_external_packages():
                    raise ValueError(
                        f"Unable to use {partitioner_type} as PETSc is not "
                        f"installed with {partitioner_type}."
                    )
                if partitioner_type == "chaco" and plex.isDistributed():
                    raise ValueError(
                        "Unable to use 'chaco' mesh partitioner, 'chaco' is a "
                        "serial partitioner, but the mesh is distributed."
                    )
            else:
                partitioner_type = DEFAULT_PARTITIONER

            partitioner.setType({
                "chaco": partitioner.Type.CHACO,
                "ptscotch": partitioner.Type.PTSCOTCH,
                "parmetis": partitioner.Type.PARMETIS,
                "shell": partitioner.Type.SHELL,
                "simple": partitioner.Type.SIMPLE
            }[partitioner_type])
        else:
            sizes, points = distribute
            partitioner.setType(partitioner.Type.SHELL)
            partitioner.setShellPartition(plex.comm.size, sizes, points)
        # Command line option `-petscpartitioner_type <type>` overrides.
        # partitioner.setFromOptions() is called from inside plex.setFromOptions().

    @PETSc.Log.EventDecorator()
    def _get_partitioner(self, plex):
        """Get partitioner actually used for (re)distributing underlying plex over comm."""
        return plex.getPartitioner()

    def mark_entities(self, tf, label_value, label_name=None):
        import firedrake.function as function

        if not isinstance(label_value, numbers.Integral):
            raise TypeError(f"label_value must be an integer: got {label_value}")
        if label_name and not isinstance(label_name, str):
            raise TypeError(f"label_name must be `None` or a string: got {label_name}")
        if label_name in ("depth",
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
        if elem.reference_value_shape != ():
            raise RuntimeError(f"tf must be scalar: {elem.reference_value_shape} != ()")
        if elem.family() in {"Discontinuous Lagrange", "DQ"} and elem.degree() == 0:
            # cells
            height = 0
            label_name = label_name or dmcommon.CELL_SETS_LABEL
        elif (elem.family() == "HDiv Trace" and elem.degree() == 0 and self.cell_dimension() > 1) or \
                (elem.family() == "Lagrange" and elem.degree() == 1 and self.cell_dimension() == 1) or \
                (elem.family() == "Q" and elem.degree() == 2 and self.ufl_cell().cellname() == "hexahedron"):
            # facets
            height = 1
            label_name = label_name or dmcommon.FACE_SETS_LABEL
        else:
            raise ValueError(f"indicator functions must be 'DP' or 'DQ' (degree 0) to mark cells and 'P' (degree 1) in 1D or 'HDiv Trace' (degree 0) in 2D or 3D to mark facets: got (family, degree) = ({elem.family()}, {elem.degree()})")
        plex = self.topology_dm
        if not plex.hasLabel(label_name):
            plex.createLabel(label_name)
        plex.clearLabelStratum(label_name, label_value)
        label = plex.getLabel(label_name)
        section = tV.dm.getSection()
        array = tf.dat.data_ro_with_halos.real.astype(IntType)
        dmcommon.mark_points_with_function_array(plex, section, height, array, label, label_value)

    # submesh

    def _submesh_make_entity_entity_map(self, from_set, to_set, from_points, to_points, child_parent_map):
        assert from_set.total_size == len(from_points)
        assert to_set.total_size == len(to_points)
        with self.topology_dm.getSubpointIS() as subpoints:
            if child_parent_map:
                _, from_indices, to_indices = np.intersect1d(subpoints[from_points], to_points, return_indices=True)
            else:
                _, from_indices, to_indices = np.intersect1d(from_points, subpoints[to_points], return_indices=True)
        values = np.full(from_set.total_size, -1, dtype=IntType)
        values[from_indices] = to_indices
        return op2.Map(from_set, to_set, 1, values.reshape((-1, 1)), f"{self}_submesh_map_{from_set}_{to_set}")

    @utils.cached_property
    def submesh_child_cell_parent_cell_map(self):
        return self._submesh_make_entity_entity_map(self.cell_set, self.submesh_parent.cell_set, self.cell_closure[:, -1], self.submesh_parent.cell_closure[:, -1], True)

    @utils.cached_property
    def submesh_child_exterior_facet_parent_exterior_facet_map(self):
        return self._submesh_make_entity_entity_map(self.exterior_facets.set, self.submesh_parent.exterior_facets.set, self.exterior_facets.facets, self.submesh_parent.exterior_facets.facets, True)

    @utils.cached_property
    def submesh_child_exterior_facet_parent_interior_facet_map(self):
        return self._submesh_make_entity_entity_map(self.exterior_facets.set, self.submesh_parent.interior_facets.set, self.exterior_facets.facets, self.submesh_parent.interior_facets.facets, True)

    @utils.cached_property
    def submesh_child_interior_facet_parent_exterior_facet_map(self):
        raise RuntimeError("Should never happen")

    @utils.cached_property
    def submesh_child_interior_facet_parent_interior_facet_map(self):
        return self._submesh_make_entity_entity_map(self.interior_facets.set, self.submesh_parent.interior_facets.set, self.interior_facets.facets, self.submesh_parent.interior_facets.facets, True)

    @utils.cached_property
    def submesh_parent_cell_child_cell_map(self):
        return self._submesh_make_entity_entity_map(self.submesh_parent.cell_set, self.cell_set, self.submesh_parent.cell_closure[:, -1], self.cell_closure[:, -1], False)

    @utils.cached_property
    def submesh_parent_exterior_facet_child_exterior_facet_map(self):
        return self._submesh_make_entity_entity_map(self.submesh_parent.exterior_facets.set, self.exterior_facets.set, self.submesh_parent.exterior_facets.facets, self.exterior_facets.facets, False)

    @utils.cached_property
    def submesh_parent_exterior_facet_child_interior_facet_map(self):
        raise RuntimeError("Should never happen")

    @utils.cached_property
    def submesh_parent_interior_facet_child_exterior_facet_map(self):
        return self._submesh_make_entity_entity_map(self.submesh_parent.interior_facets.set, self.exterior_facets.set, self.submesh_parent.interior_facets.facets, self.exterior_facets.facets, False)

    @utils.cached_property
    def submesh_parent_interior_facet_child_interior_facet_map(self):
        return self._submesh_make_entity_entity_map(self.submesh_parent.interior_facets.set, self.interior_facets.set, self.submesh_parent.interior_facets.facets, self.interior_facets.facets, False)

    def submesh_map_child_parent(self, source_integral_type, source_subset_points, reverse=False):
        """Return the map from submesh child entities to submesh parent entities or its reverse.

        Parameters
        ----------
        source_integral_type : str
            Integral type on the source mesh.
        source_subset_points : numpy.ndarray
            Subset points on the source mesh.
        reverse : bool
            If True, return the map from parent entities to child entities.

        Returns
        -------
        tuple
           (map from source to target, integral type on the target mesh, subset points on the target mesh).

        """
        if self.submesh_parent is None:
            raise RuntimeError("Must only be called on submesh")
        if reverse:
            source = self.submesh_parent
            target = self
        else:
            source = self
            target = self.submesh_parent
        target_dim = target.topology_dm.getDimension()
        source_dim = source.topology_dm.getDimension()
        if source_dim != target_dim:
            raise NotImplementedError(f"Not implemented for (source_dim, target_dim) == ({source_dim}, {target_dim})")
        if source_integral_type == "cell":
            target_integral_type = "cell"
            target_subset_points = None
        elif source_integral_type in ["interior_facet", "exterior_facet"]:
            with self.topology_dm.getSubpointIS() as subpoints:
                if reverse:
                    _, target_indices_int, source_indices_int = np.intersect1d(subpoints[target.interior_facets.facets], source_subset_points, return_indices=True)
                    _, target_indices_ext, source_indices_ext = np.intersect1d(subpoints[target.exterior_facets.facets], source_subset_points, return_indices=True)
                else:
                    target_subset_points = subpoints[source_subset_points]
                    _, target_indices_int, source_indices_int = np.intersect1d(target.interior_facets.facets, target_subset_points, return_indices=True)
                    _, target_indices_ext, source_indices_ext = np.intersect1d(target.exterior_facets.facets, target_subset_points, return_indices=True)
            n_int = len(source_indices_int)
            n_ext = len(source_indices_ext)
            n_int_max = self._comm.allreduce(n_int, op=MPI.MAX)
            n_ext_max = self._comm.allreduce(n_ext, op=MPI.MAX)
            if n_int_max > 0:
                if n_ext_max != 0:
                    raise RuntimeError(f"integral_type on the target mesh is interior facet, but {n_ext_max} exterior facet entities are also included")
                if n_int > len(source_subset_points):
                    raise RuntimeError("Found inconsistent data")
                target_integral_type = "interior_facet"
            elif n_ext_max > 0:
                if n_int_max != 0:
                    raise RuntimeError(f"integral_type on the target mesh is exterior facet, but {n_int_max} interior facet entities are also included")
                if n_ext > len(source_subset_points):
                    raise RuntimeError("Found inconsistent data")
                target_integral_type = "exterior_facet"
            else:
                raise RuntimeError("Can not find a map from source to target.")
            if reverse:
                if target_integral_type == "interior_facet":
                    target_subset_points = target.interior_facets.facets[target_indices_int]
                elif target_integral_type == "exterior_facet":
                    target_subset_points = target.exterior_facets.facets[target_indices_ext]
        else:
            raise NotImplementedError(f"Not implemented for (source_dim, target_dim, source_integral_type) == ({source_dim}, {target_dim}, {source_integral_type})")
        if reverse:
            map_ = getattr(self, f"submesh_parent_{source_integral_type}_child_{target_integral_type}_map")
        else:
            map_ = getattr(self, f"submesh_child_{source_integral_type}_parent_{target_integral_type}_map")
        return map_, target_integral_type, target_subset_points

    # trans mesh

    def trans_mesh_entity_map(self, base_mesh, base_integral_type, base_subdomain_id, base_all_integer_subdomain_ids):
        """Create entity-entity (composed) map from base_mesh to `self`.

        Parameters
        ----------
        base_mesh : AbstractMeshTopology
            Base mesh topology.
        base_integral_type : str
            Integral type on ``base_mesh``.
        base_subdomain_id : int
            Subdomain ID on ``base_mesh``.
        base_all_integer_subdomain_ids : tuple
            ``all_integer_subdomain_ids`` corresponding to ``base_mesh`` and ``base_integral_type``.

        Returns
        -------
        tuple
            `tuple` of `op2.ComposedMap` from base_mesh to `self` and integral_type on `self`.

        """
        common = self.submesh_youngest_common_ancester(base_mesh)
        if common is None:
            raise NotImplementedError(f"Currently only implemented for (sub)meshes in the same family: got {self} and {base_mesh}")
        elif base_mesh is self:
            raise NotImplementedError("Currenlty can not return identity map")
        else:
            if base_integral_type == "cell":
                base_subset_points = None
            elif base_integral_type in ["interior_facet", "exterior_facet"]:
                base_subset = base_mesh.measure_set(base_integral_type, base_subdomain_id, all_integer_subdomain_ids=base_all_integer_subdomain_ids)
                if base_integral_type == "interior_facet":
                    base_subset_points = base_mesh.interior_facets.facets[base_subset.indices]
                elif base_integral_type == "exterior_facet":
                    base_subset_points = base_mesh.exterior_facets.facets[base_subset.indices]
            else:
                raise NotImplementedError(f"Unknown integration type : {base_integral_type}")
            composed_map, integral_type, _ = self.submesh_map_composed(base_mesh, base_integral_type, base_subset_points)
            return composed_map, integral_type


class ExtrudedMeshTopology(MeshTopology):
    """Representation of an extruded mesh topology."""

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, layers, periodic=False, name=None):
        """Build an extruded mesh topology from an input mesh topology

        :arg mesh:           the unstructured base mesh topology
        :arg layers:         number of occurence of base layer in the "vertical" direction.
        :arg periodic:       the flag for periodic extrusion; if True, only constant layer extrusion is allowed.
        :arg name:           optional name of the extruded mesh topology.
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

        self._base_mesh = mesh
        self.user_comm = mesh.comm
        self._comm = internal_comm(mesh._comm, self)
        if name is not None and name == mesh.name:
            raise ValueError("Extruded mesh topology and base mesh topology can not have the same name")
        self.name = name if name is not None else mesh.name + "_extruded"
        # TODO: These attributes are copied so that FunctionSpaceBase can
        # access them directly.  Eventually we would want a better refactoring
        # of responsibilities between mesh and function space.
        self.topology_dm = mesh.topology_dm
        r"The PETSc DM representation of the mesh topology."
        self._dm_renumbering = mesh._dm_renumbering
        self._cell_numbering = mesh._cell_numbering
        self._entity_classes = mesh._entity_classes
        self._did_reordering = mesh._did_reordering
        self._distribution_parameters = mesh._distribution_parameters
        self._subsets = {}
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
        # submesh
        self.submesh_parent = None

    @utils.cached_property
    def _ufl_cell(self):
        return ufl.TensorProductCell(self._base_mesh.ufl_cell(), ufl.interval)

    @utils.cached_property
    def _ufl_mesh(self):
        cell = self._ufl_cell
        return ufl.Mesh(finat.ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension()))

    @property
    def dm_cell_types(self):
        """All DM.PolytopeTypes of cells in the mesh."""
        raise NotImplementedError("'dm_cell_types' is not implemented for ExtrudedMeshTopology")

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

        # Convert to a tuple of tuples with int (not numpy.intXX) values. This is
        # to give us a string representation like ((0, 1), (2, 3)) instead of
        # ((numpy.int32(0), numpy.int32(1)), (numpy.int32(2), numpy.int32(3))).
        return tuple(
            tuple(int(d_) for d_ in d)
            for d in dofs_per_entity
        )

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

    def mark_entities(self, tf, label_value, label_name=None):
        raise NotImplementedError("Currently not implemented for ExtrudedMesh")


# TODO: Could this be merged with MeshTopology given that dmcommon.pyx
# now covers DMSwarms and DMPlexes?
class VertexOnlyMeshTopology(AbstractMeshTopology):
    """
    Representation of a vertex-only mesh topology immersed within
    another mesh.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, swarm, parentmesh, name, reorder, input_ordering_swarm=None, perm_is=None, distribution_name=None, permutation_name=None):
        """Initialise a mesh topology.

        Parameters
        ----------
        swarm : PETSc.DMSwarm
            `PETSc.DMSwarm` representing Particle In Cell (PIC) vertices
            immersed within a `PETSc.DM` stored in the ``parentmesh``.
        parentmesh : AbstractMeshTopology
            Mesh topology within which the vertex-only mesh topology is immersed.
        name : str
            Name of the mesh topology.
        reorder : bool
            Whether to reorder the mesh entities.
        input_ordering_swarm : PETSc.DMSwarm
            The swarm from which the input-ordering vertex-only mesh is constructed.
        perm_is : PETSc.IS
            `PETSc.IS` that is used as ``_dm_renumbering``; only
            makes sense if we know the exact parallel distribution of ``plex``
            at the time of mesh topology construction like when we load mesh
            along with its distribution. If given, ``reorder`` param will be ignored.
        distribution_name : str
            Name of the parallel distribution; if `None`, automatically generated.
        permutation_name : str
            Name of the entity permutation (reordering); if `None`, automatically generated.

        """
        if MPI.Comm.Compare(parentmesh.comm, swarm.comm.tompi4py()) not in {MPI.CONGRUENT, MPI.IDENT}:
            ValueError("Parent mesh communicator and swarm communicator are not congruent")
        self._distribution_parameters = {"partition": False,
                                         "partitioner_type": None,
                                         "overlap_type": (DistributedMeshOverlapType.NONE, 0)}
        self.input_ordering_swarm = input_ordering_swarm
        self._parent_mesh = parentmesh
        super().__init__(swarm, name, reorder, None, perm_is, distribution_name, permutation_name, parentmesh.comm)

    def _distribute(self):
        pass

    def _add_overlap(self):
        pass

    def _mark_entity_classes(self):
        if self.input_ordering_swarm:
            assert isinstance(self._parent_mesh, MeshTopology)
            dmcommon.mark_entity_classes_using_cell_dm(self.topology_dm)
        else:
            # Have an input-ordering vertex-only mesh. These should mark
            # all entities as pyop2 core, which mark_entity_classes will do.
            assert isinstance(self._parent_mesh, VertexOnlyMeshTopology)
            dmcommon.mark_entity_classes(self.topology_dm)

    @utils.cached_property
    def _ufl_cell(self):
        return ufl.Cell(_cells[0][0])

    @utils.cached_property
    def _ufl_mesh(self):
        cell = self._ufl_cell
        return ufl.Mesh(finat.ufl.VectorElement("DG", cell, 0, dim=cell.topological_dimension()))

    def _renumber_entities(self, reorder):
        if reorder:
            swarm = self.topology_dm
            parent = self._parent_mesh.topology_dm
            cell_id_name = swarm.getCellDMActive().getCellID()
            swarm_parent_cell_nums = swarm.getField(cell_id_name).ravel()
            parent_renum = self._parent_mesh._dm_renumbering.getIndices()
            pStart, _ = parent.getChart()
            parent_renum_inv = np.empty_like(parent_renum)
            parent_renum_inv[parent_renum - pStart] = np.arange(len(parent_renum))
            # Use kind = 'stable' to make the ordering deterministic.
            perm = np.argsort(parent_renum_inv[swarm_parent_cell_nums - pStart], kind='stable').astype(IntType)
            swarm.restoreField(cell_id_name)
            perm_is = PETSc.IS().create(comm=swarm.comm)
            perm_is.setType("general")
            perm_is.setIndices(perm)
            return perm_is
        else:
            return dmcommon.plex_renumbering(self.topology_dm, self._entity_classes, None)

    @property
    def dm_cell_types(self):
        """All DM.PolytopeTypes of cells in the mesh."""
        return (PETSc.DM.PolytopeType.POINT,)

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
        cell_parent_cell_list = np.copy(self.topology_dm.getField("parentcellnum").ravel())
        self.topology_dm.restoreField("parentcellnum")
        return cell_parent_cell_list[self.cell_closure[:, -1]]

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
        cell_parent_base_cell_list = np.copy(self.topology_dm.getField("parentcellbasenum").ravel())
        self.topology_dm.restoreField("parentcellbasenum")
        return cell_parent_base_cell_list[self.cell_closure[:, -1]]

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
        cell_parent_extrusion_height_list = np.copy(self.topology_dm.getField("parentcellextrusionheight").ravel())
        self.topology_dm.restoreField("parentcellextrusionheight")
        return cell_parent_extrusion_height_list[self.cell_closure[:, -1]]

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_extrusion_height_map(self):
        """Return the :class:`pyop2.types.map.Map` from vertex only mesh cells to
        parent mesh extrusion heights.
        """
        if not isinstance(self._parent_mesh, ExtrudedMeshTopology):
            raise AttributeError("Parent mesh is not extruded.")
        return op2.Map(self.cell_set, self._parent_mesh.cell_set, 1,
                       self.cell_parent_extrusion_height_list, "cell_parent_extrusion_height")

    def mark_entities(self, tf, label_value, label_name=None):
        raise NotImplementedError("Currently not implemented for VertexOnlyMesh")

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_global_index(self):
        """Return a list of unique cell IDs in vertex only mesh cell order."""
        cell_global_index = np.copy(self.topology_dm.getField("globalindex").ravel())
        self.topology_dm.restoreField("globalindex")
        return cell_global_index

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def input_ordering(self):
        """
        Return the input ordering of the mesh vertices as a
        :class:`~.VertexOnlyMeshTopology` whilst preserving other information, such as
        the global indices and parent mesh cell information.

        Notes
        -----
        If ``redundant=True`` at mesh creation, all the vertices will
        be returned on rank 0.

        Any points that were not found in the original mesh when it was created
        will still be present here in their originally supplied order.
        """
        if not isinstance(self.topology, VertexOnlyMeshTopology):
            raise AttributeError("Input ordering is only defined for vertex-only meshes.")
        # Make the VOM which uses the original ordering of the points
        if self.input_ordering_swarm:
            return VertexOnlyMeshTopology(
                self.input_ordering_swarm,
                self,
                name=self.input_ordering_swarm.getName(),
                reorder=False,
            )

    @staticmethod
    def _make_input_ordering_sf(swarm, nroots, ilocal):
        # ilocal = None -> leaves are swarm points [0, 1, 2, ...).
        # ilocal can also be Firedrake cell numbers.
        sf = PETSc.SF().create(comm=swarm.comm)
        input_ranks = swarm.getField("inputrank").ravel()
        input_indices = swarm.getField("inputindex").ravel()
        nleaves = len(input_ranks)
        if ilocal is not None and nleaves != len(ilocal):
            swarm.restoreField("inputrank")
            swarm.restoreField("inputindex")
            raise RuntimeError(f"Mismatching leaves: nleaves {nleaves} != len(ilocal) {len(ilocal)}")
        input_ranks_and_idxs = np.empty(2 * nleaves, dtype=IntType)
        input_ranks_and_idxs[0::2] = input_ranks
        input_ranks_and_idxs[1::2] = input_indices
        swarm.restoreField("inputrank")
        swarm.restoreField("inputindex")
        sf.setGraph(nroots, ilocal, input_ranks_and_idxs)
        return sf

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def input_ordering_sf(self):
        """
        Return a PETSc SF which has :func:`~.VertexOnlyMesh` input ordering
        vertices as roots and this mesh's vertices (including any halo cells)
        as leaves.
        """
        if not isinstance(self.topology, VertexOnlyMeshTopology):
            raise AttributeError("Input ordering is only defined for vertex-only meshes.")
        nroots = self.input_ordering.num_cells()
        e_p_map = self.cell_closure[:, -1]  # cell-entity -> swarm-point map
        ilocal = np.empty_like(e_p_map)
        if len(e_p_map) > 0:
            cStart = e_p_map.min()  # smallest swarm point number
            ilocal[e_p_map - cStart] = np.arange(len(e_p_map))
        return VertexOnlyMeshTopology._make_input_ordering_sf(self.topology_dm, nroots, ilocal)

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def input_ordering_without_halos_sf(self):
        """
        Return a PETSc SF which has :func:`~.VertexOnlyMesh` input ordering
        vertices as roots and this mesh's non-halo vertices as leaves.
        """
        # The leaves have been ordered according to the pyop2 classes with non-halo
        # cells first; self.cell_set.size is the number of rank-local non-halo cells.
        return self.input_ordering_sf.createEmbeddedLeafSF(np.arange(self.cell_set.size, dtype=IntType))


class CellOrientationsRuntimeError(RuntimeError):
    """Exception raised when there are problems with cell orientations."""
    pass


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

    def __new__(cls, element, comm):
        """Create mesh geometry object."""
        utils._init()
        mesh = super(MeshGeometry, cls).__new__(cls)
        uid = utils._new_uid(internal_comm(comm, mesh))
        mesh.uid = uid
        cargo = MeshGeometryCargo(uid)
        assert isinstance(element, finat.ufl.FiniteElementBase)
        ufl.Mesh.__init__(mesh, element, ufl_id=mesh.uid, cargo=cargo)
        return mesh

    @MeshGeometryMixin._ad_annotate_init
    def __init__(self, coordinates):
        """Initialise a mesh geometry from coordinates.

        Parameters
        ----------
        coordinates : CoordinatelessFunction
            The `CoordinatelessFunction` containing the coordinates.

        """
        topology = coordinates.function_space().mesh()

        # this is codegen information so we attach it to the MeshGeometry rather than its cargo
        self.extruded = isinstance(topology, ExtrudedMeshTopology)
        self.variable_layers = self.extruded and topology.variable_layers

        # initialise the mesh cargo
        self.ufl_cargo().init(coordinates)

        # Cache mesh object on the coordinateless coordinates function
        coordinates._as_mesh_geometry = weakref.ref(self)

        # submesh
        self.submesh_parent = None

        self._spatial_index = None
        self._saved_coordinate_dat_version = coordinates.dat.dat_version

    def _ufl_signature_data_(self, *args, **kwargs):
        return (type(self), self.extruded, self.variable_layers,
                super()._ufl_signature_data_(*args, **kwargs))

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
        coordinates_fs = functionspace.FunctionSpace(self.topology, self.ufl_coordinate_element())
        coordinates_data = dmcommon.reordered_coords(topology.topology_dm, coordinates_fs.dm.getDefaultSection(),
                                                     (self.num_vertices(), self.geometric_dimension()))
        coordinates = function.CoordinatelessFunction(coordinates_fs,
                                                      val=coordinates_data,
                                                      name=_generate_default_mesh_coordinates_name(self.name))
        self.__init__(coordinates)

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
        """A :class:`~.Function` in the :math:`P^1` space containing the local mesh size.

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
        distance (aka 'manhattan', 'taxicab' or rectilinear distance) so will
        scale with the dimension of the mesh.

        If this property is not set (i.e. set to ``None``) no tolerance is
        added to the bounding box and points deemed at all outside the mesh,
        even by floating point error distances, will be deemed to be outside
        it.

        Notes
        -----
        After changing tolerance any requests for :attr:`spatial_index` will cause
        the spatial index to be rebuilt with the new tolerance which may take some time.
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        if not isinstance(value, numbers.Number):
            raise TypeError("tolerance must be a number")
        if value != self._tolerance:
            self.clear_spatial_index()
            self._tolerance = value

    def clear_spatial_index(self):
        """Reset the :attr:`spatial_index` on this mesh geometry.

        Use this if you move the mesh (for example by reassigning to
        the coordinate field)."""
        self._spatial_index = None

    @property
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

        if (
            self._spatial_index
            and self.coordinates.dat.dat_version == self._saved_coordinate_dat_version
        ):
            return self._spatial_index

        gdim = self.geometric_dimension()
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
                  'f_max': (coords_max, MAX)})

        # Reorder bounding boxes according to the cell indices we use
        column_list = V.cell_node_list.reshape(-1)
        coords_min = self._order_data_by_cell_index(column_list, coords_min.dat.data_ro_with_halos)
        coords_max = self._order_data_by_cell_index(column_list, coords_max.dat.data_ro_with_halos)

        # Change min and max to refer to an n-hypercube, where n is the
        # geometric dimension of the mesh, centred on the midpoint of the
        # bounding box. Its side length is the L1 diameter of the bounding box.
        # This aids point evaluation on immersed manifolds and other cases
        # where points may be just off the mesh but should be evaluated.
        # TODO: This is perhaps unnecessary when we aren't in these special
        # cases.

        # We also push max and min out so we can find points on the boundary
        # within the mesh tolerance.
        # NOTE: getattr doesn't work here due to the inheritance games that are
        # going on in getattr.
        tolerance = self.tolerance if hasattr(self, "tolerance") else 0.0
        coords_mid = (coords_max + coords_min)/2
        d = np.max(coords_max - coords_min, axis=1)[:, None]
        coords_min = coords_mid - (tolerance + 0.5)*d
        coords_max = coords_mid + (tolerance + 0.5)*d

        # Build spatial index
        self._spatial_index = spatialindex.from_regions(coords_min, coords_max)
        self._saved_coordinate_dat_version = self.coordinates.dat.dat_version
        return self._spatial_index

    @PETSc.Log.EventDecorator()
    def locate_cell(self, x, tolerance=None, cell_ignore=None):
        """Locate cell containing a given point.

        :arg x: point coordinates
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the spatial index to be rebuilt which
            can take some time.
        :kwarg cell_ignore: Cell number to ignore in the search.
        :returns: cell number (int), or None (if the point is not
            in the domain)
        """
        return self.locate_cell_and_reference_coordinate(x, tolerance=tolerance, cell_ignore=cell_ignore)[0]

    def locate_reference_coordinate(self, x, tolerance=None, cell_ignore=None):
        """Get reference coordinates of a given point in its cell. Which
        cell the point is in can be queried with the locate_cell method.

        :arg x: point coordinates
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the spatial index to be rebuilt which
            can take some time.
        :kwarg cell_ignore: Cell number to ignore in the search.
        :returns: reference coordinates within cell (numpy array) or
            None (if the point is not in the domain)
        """
        return self.locate_cell_and_reference_coordinate(x, tolerance=tolerance, cell_ignore=cell_ignore)[1]

    def locate_cell_and_reference_coordinate(self, x, tolerance=None, cell_ignore=None):
        """Locate cell containing a given point and the reference
        coordinates of the point within the cell.

        :arg x: point coordinates
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the spatial index to be rebuilt which
            can take some time.
        :kwarg cell_ignore: Cell number to ignore in the search.
        :returns: tuple either
            (cell number, reference coordinates) of type (int, numpy array),
            or, when point is not in the domain, (None, None).
        """
        x = np.asarray(x)
        if x.size != self.geometric_dimension():
            raise ValueError("Point must have the same geometric dimension as the mesh")
        x = x.reshape((1, self.geometric_dimension()))
        cells, ref_coords, _ = self.locate_cells_ref_coords_and_dists(x, tolerance=tolerance, cells_ignore=[[cell_ignore]])
        if cells[0] == -1:
            return None, None
        return cells[0], ref_coords[0]

    def locate_cells_ref_coords_and_dists(self, xs, tolerance=None, cells_ignore=None):
        """Locate cell containing a given point and the reference
        coordinates of the point within the cell.

        :arg xs: 1 or more point coordinates of shape (npoints, gdim)
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the spatial index to be rebuilt which
            can take some time.
        :kwarg cells_ignore: Cell numbers to ignore in the search for each
            point in xs. Shape should be (npoints, n_ignore_pts). Each column
            corresponds to a single coordinate in xs. To not ignore any cells,
            pass None. To ensure a full cell search for any given point, set
            the corresponding entries to -1.
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
        if cells_ignore is None or cells_ignore[0][0] is None:
            cells_ignore = np.full((npoints, 1), -1, dtype=IntType, order="C")
        else:
            cells_ignore = np.asarray(cells_ignore, dtype=IntType, order="C")
        if cells_ignore.shape[0] != npoints:
            raise ValueError("Number of cells to ignore does not match number of points")
        assert cells_ignore.shape == (npoints, cells_ignore.shape[1])
        ref_cell_dists_l1 = np.empty(npoints, dtype=utils.RealType)
        cells = np.empty(npoints, dtype=IntType)
        assert xs.size == npoints * self.geometric_dimension()
        self._c_locator(tolerance=tolerance)(self.coordinates._ctypes,
                                             xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             Xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             ref_cell_dists_l1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             cells.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                             npoints,
                                             cells_ignore.shape[1],
                                             cells_ignore)
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
            IntTypeC = as_cstr(IntType)
            src = pq_utils.src_locate_cell(self, tolerance=tolerance)
            src += dedent(f"""
                int locator(struct Function *f, double *x, double *X, double *ref_cell_dists_l1, {IntTypeC} *cells, {IntTypeC} npoints, size_t ncells_ignore, int* cells_ignore)
                {{
                    {IntTypeC} j = 0;  /* index into x and X */
                    for({IntTypeC} i=0; i<npoints; i++) {{
                        /* i is the index into cells and ref_cell_dists_l1 */

                        /* The type definitions and arguments used here are defined as
                        statics in pointquery_utils.py */
                        struct ReferenceCoords temp_reference_coords, found_reference_coords;

                        /* to_reference_coords and to_reference_coords_xtr are defined in
                        pointquery_utils.py. If they contain python calls, this loop will
                        not run at c-loop speed. */
                        /* cells_ignore has shape (npoints, ncells_ignore) - find the ith row */
                        int *cells_ignore_i = cells_ignore + i*ncells_ignore;
                        cells[i] = locate_cell(f, &x[j], {self.geometric_dimension()}, &to_reference_coords, &to_reference_coords_xtr, &temp_reference_coords, &found_reference_coords, &ref_cell_dists_l1[i], ncells_ignore, cells_ignore_i);

                        for (int k = 0; k < {self.geometric_dimension()}; k++) {{
                            X[j] = found_reference_coords.X[k];
                            j++;
                        }}
                    }}
                    return 0;
                }}
            """)

            libspatialindex_so = Path(rtree.core.rt._name).absolute()
            lsi_runpath = f"-Wl,-rpath,{libspatialindex_so.parent}"
            dll = compilation.load(
                src, "c",
                cppargs=[
                    f"-I{os.path.dirname(__file__)}",
                    f"-I{sys.prefix}/include",
                    f"-I{rtree.finder.get_include()}"
                ] + [f"-I{d}/include" for d in get_petsc_dir()],
                ldargs=[
                    f"-L{sys.prefix}/lib",
                    str(libspatialindex_so),
                    f"-Wl,-rpath,{sys.prefix}/lib",
                    lsi_runpath
                ],
                comm=self.comm
            )
            locator = getattr(dll, "locator")
            locator.argtypes = [ctypes.POINTER(function._CFunction),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_int),
                                ctypes.c_size_t,
                                ctypes.c_size_t,
                                np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
            locator.restype = ctypes.c_int
            return cache.setdefault(tolerance, locator)

    @utils.cached_property  # TODO: Recalculate if mesh moves. Extend this for regular meshes.
    def input_ordering(self):
        """
        Return the input ordering of the mesh vertices as a
        :func:`~.VertexOnlyMesh` whilst preserving other information, such as
        the global indices and parent mesh cell information.

        Notes
        -----
        If ``redundant=True`` at mesh creation, all the vertices will
        be returned on rank 0.

        Any points that were not found in the original mesh when it was created
        will still be present here in their originally supplied order.

        """
        if not isinstance(self.topology, VertexOnlyMeshTopology):
            raise AttributeError("Input ordering is only defined for vertex-only meshes.")
        _input_ordering = make_vom_from_vom_topology(self.topology.input_ordering, self.name + "_input_ordering")
        if _input_ordering:
            _input_ordering._parent_mesh = self
            return _input_ordering

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
            raise CellOrientationsRuntimeError("No cell orientations found, did you forget to call init_cell_orientations?")
        return self._cell_orientations

    @PETSc.Log.EventDecorator()
    def init_cell_orientations(self, expr):
        """Compute and initialise meth:`cell_orientations` relative to a specified orientation.

        :arg expr: a UFL expression evaluated to produce a
             reference normal direction.

        """
        import firedrake.function as function
        import firedrake.functionspace as functionspace

        if (self.ufl_cell().cellname(), self.geometric_dimension()) not in _supported_embedded_cell_types_and_gdims:
            raise NotImplementedError('Only implemented for intervals embedded in 2d and triangles and quadrilaterals embedded in 3d')

        if hasattr(self, '_cell_orientations'):
            raise CellOrientationsRuntimeError("init_cell_orientations already called, did you mean to do so again?")

        if not isinstance(expr, ufl.classes.Expr):
            raise TypeError("UFL expression expected!")

        if expr.ufl_shape != (self.geometric_dimension(), ):
            raise ValueError(f"Mismatching shapes: expr.ufl_shape ({expr.ufl_shape}) != (self.geometric_dimension(), ) (({self.geometric_dimension}, ))")

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

    def mark_entities(self, f, label_value, label_name=None):
        """Mark selected entities.

        :arg f: The :class:`.Function` object that marks
            selected entities as 1. f.function_space().ufl_element()
            must be "DP" or "DQ" (degree 0) to mark cell entities and
            "P" (degree 1) in 1D or "HDiv Trace" (degree 0) in 2D or 3D
            to mark facet entities.
            Can use "Q" (degree 2) functions for 3D hex meshes until
            we support "HDiv Trace" elements on hex.
        :arg lable_value: The value used in the label.
        :arg label_name: The name of the label to store entity selections.

        All entities must live on the same topological dimension. Currently,
        one can only mark cell or facet entities.
        """
        self.topology.mark_entities(f.topological, label_value, label_name)


@PETSc.Log.EventDecorator()
def make_mesh_from_coordinates(coordinates, name, tolerance=0.5):
    """Given a coordinate field build a new mesh, using said coordinate field.

    Parameters
    ----------
    coordinates : CoordinatelessFunction
        The `CoordinatelessFunction` from which mesh is made.
    name : str
        The name of the mesh.
    tolerance : numbers.Number
        The tolerance; see `Mesh`.
    comm: mpi4py.Intracomm
        Communicator.

    Returns
    -------
    MeshGeometry
        The mesh.

    """
    if hasattr(coordinates, '_as_mesh_geometry'):
        mesh = coordinates._as_mesh_geometry()
        if mesh is not None:
            return mesh

    V = coordinates.function_space()
    element = coordinates.ufl_element()
    if V.rank != 1 or len(element.reference_value_shape) != 1:
        raise ValueError("Coordinates must be from a rank-1 FunctionSpace with rank-1 value_shape.")
    assert V.mesh().ufl_cell().topological_dimension() <= V.value_size

    mesh = MeshGeometry.__new__(MeshGeometry, element, coordinates.comm)
    mesh.__init__(coordinates)
    mesh.name = name
    # Mark mesh as being made from coordinates
    mesh._made_from_coordinates = True
    mesh._tolerance = tolerance
    return mesh


def make_mesh_from_mesh_topology(topology, name, tolerance=0.5):
    """Make mesh from tpology.

    Parameters
    ----------
    topology : MeshTopology
        The `MeshTopology` from which mesh is made.
    name : str
        The name of the mesh.
    tolerance : numbers.Number
        The tolerance; see `Mesh`.

    Returns
    -------
    MeshGeometry
        The mesh.

    """
    # Construct coordinate element
    # TODO: meshfile might indicates higher-order coordinate element
    cell = topology.ufl_cell()
    geometric_dim = topology.topology_dm.getCoordinateDim()
    if not topology.topology_dm.getCoordinatesLocalized():
        element = finat.ufl.VectorElement("Lagrange", cell, 1, dim=geometric_dim)
    else:
        element = finat.ufl.VectorElement("DQ" if cell in [ufl.quadrilateral, ufl.hexahedron] else "DG", cell, 1, dim=geometric_dim, variant="equispaced")
    # Create mesh object
    mesh = MeshGeometry.__new__(MeshGeometry, element, topology.comm)
    mesh._init_topology(topology)
    mesh.name = name
    mesh._tolerance = tolerance
    return mesh


def make_vom_from_vom_topology(topology, name, tolerance=0.5):
    """Make `VertexOnlyMesh` from a mesh topology.

    Parameters
    ----------
    topology : VertexOnlyMeshTopology
        The `VertexOnlyMeshTopology`.
    name : str
        The name of the mesh.
    tolerance : numbers.Number
        The tolerance; see `Mesh`.

    Returns
    -------
    MeshGeometry
        The mesh.

    """
    import firedrake.functionspaceimpl as functionspaceimpl
    import firedrake.functionspace as functionspace
    import firedrake.function as function

    gdim = topology.topology_dm.getCoordinateDim()
    cell = topology.ufl_cell()
    element = finat.ufl.VectorElement("DG", cell, 0, dim=gdim)
    vmesh = MeshGeometry.__new__(MeshGeometry, element, topology.comm)
    vmesh._init_topology(topology)
    # Save vertex reference coordinate (within reference cell) in function
    parent_tdim = topology._parent_mesh.ufl_cell().topological_dimension()
    if parent_tdim > 0:
        reference_coordinates_fs = functionspace.VectorFunctionSpace(topology, "DG", 0, dim=parent_tdim)
        reference_coordinates_data = dmcommon.reordered_coords(topology.topology_dm, reference_coordinates_fs.dm.getDefaultSection(),
                                                               (topology.num_vertices(), parent_tdim),
                                                               reference_coord=True)
        reference_coordinates = function.CoordinatelessFunction(reference_coordinates_fs,
                                                                val=reference_coordinates_data,
                                                                name=_generate_default_mesh_reference_coordinates_name(name))
        refCoordV = functionspaceimpl.WithGeometry.create(reference_coordinates_fs, vmesh)
        vmesh.reference_coordinates = function.Function(refCoordV, val=reference_coordinates)
    else:
        # We can't do this in 0D so leave it undefined.
        vmesh.reference_coordinates = None
    vmesh.name = name
    vmesh._tolerance = tolerance
    return vmesh


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
           considered to be in the cell. Defaults to 0.5. Increase
           this if point at mesh boundaries (either rank local or global) are
           reported as being outside the mesh, for example when creating a
           :class:`VertexOnlyMesh`. Note that this tolerance uses an L1
           distance (aka 'manhattan', 'taxicab' or rectilinear distance) so
           will scale with the dimension of the mesh.

    :param netgen_flags: The dictionary of flags to be passed to ngsPETSc.

    When the mesh is read from a file the following mesh formats
    are supported (determined, case insensitively, from the
    filename extension):

    * GMSH: with extension `.msh`
    * Exodus: with extension `.e`, `.exo`
    * CGNS: with extension `.cgns`
    * Triangle: with extension `.node`
    * HDF5: with extension `.h5`, `.hdf5`
      (Can only load HDF5 files created by
      :meth:`~.CheckpointFile.save_mesh` method.)

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

    tolerance = kwargs.get("tolerance", 0.5)

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
        try:
            from ngsPETSc import FiredrakeMesh
        except ImportError:
            raise ImportError("Unable to import ngsPETSc. Please ensure that ngsolve is installed and available to Firedrake.")
        from firedrake_citations import Citations
        Citations().register("Betteridge2024")
        netgen_flags = kwargs.get("netgen_flags", {"quad": False, "transform": None, "purify_to_tets": False})
        netgen_firedrake_mesh = FiredrakeMesh(meshfile, netgen_flags, user_comm)
        plex = netgen_firedrake_mesh.meshMap.petscPlex
        plex.setName(_generate_default_mesh_topology_name(name))
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
    submesh_parent = kwargs.get("submesh_parent", None)
    topology = MeshTopology(plex, name=plex.getName(), reorder=reorder,
                            distribution_parameters=distribution_parameters,
                            distribution_name=kwargs.get("distribution_name"),
                            permutation_name=kwargs.get("permutation_name"),
                            submesh_parent=submesh_parent.topology if submesh_parent else None,
                            comm=user_comm)
    if netgen and isinstance(meshfile, netgen.libngpy._meshing.Mesh):
        netgen_firedrake_mesh.createFromTopology(topology, name=name, comm=user_comm)
        mesh = netgen_firedrake_mesh.firedrakeMesh
    else:
        mesh = make_mesh_from_mesh_topology(topology, name)
    mesh.submesh_parent = submesh_parent
    mesh._tolerance = tolerance
    return mesh


@PETSc.Log.EventDecorator("CreateExtMesh")
def ExtrudedMesh(mesh, layers, layer_height=None, extrusion_type='uniform', periodic=False, kernel=None, gdim=None, name=None, tolerance=0.5):
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
                         distance (aka 'manhattan', 'taxicab' or rectilinear
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

    topology = ExtrudedMeshTopology(mesh.topology, layers, periodic=periodic)

    if extrusion_type == "uniform":
        pass
    elif extrusion_type in ("radial", "radial_hedgehog"):
        # do not allow radial extrusion if tdim = gdim
        if mesh.geometric_dimension() == mesh.topological_dimension():
            raise RuntimeError("Cannot radially-extrude a mesh with equal geometric and topological dimension")
    else:
        # check for kernel
        if kernel is None:
            raise RuntimeError("If the custom extrusion_type is used, a kernel must be provided")
        # otherwise, use the gdim that was passed in
        if gdim is None:
            raise RuntimeError("The geometric dimension of the mesh must be specified if a custom extrusion kernel is used")

    helement = mesh._coordinates.ufl_element().sub_elements[0]
    if extrusion_type == 'radial_hedgehog':
        helement = helement.reconstruct(family="DG", variant="equispaced")
    if periodic:
        velement = finat.ufl.FiniteElement("DP", ufl.interval, 1, variant="equispaced")
    else:
        velement = finat.ufl.FiniteElement("Lagrange", ufl.interval, 1)
    element = finat.ufl.TensorProductElement(helement, velement)

    if gdim is None:
        gdim = mesh.geometric_dimension() + (extrusion_type == "uniform")
    coordinates_fs = functionspace.VectorFunctionSpace(topology, element, dim=gdim)

    coordinates = function.CoordinatelessFunction(coordinates_fs, name=_generate_default_mesh_coordinates_name(name))

    eutils.make_extruded_coords(topology, mesh._coordinates, coordinates,
                                layer_height, extrusion_type=extrusion_type, kernel=kernel)

    self = make_mesh_from_coordinates(coordinates, name)
    self._base_mesh = mesh

    if extrusion_type == "radial_hedgehog":
        helement = mesh._coordinates.ufl_element().sub_elements[0].reconstruct(family="CG")
        element = finat.ufl.TensorProductElement(helement, velement)
        fs = functionspace.VectorFunctionSpace(self, element, dim=gdim)
        self.radial_coordinates = function.Function(fs, name=name + "_radial_coordinates")
        eutils.make_extruded_coords(topology, mesh._coordinates, self.radial_coordinates,
                                    layer_height, extrusion_type="radial", kernel=kernel)
    self._tolerance = tolerance
    return self


class MissingPointsBehaviour(enum.Enum):
    IGNORE = None
    ERROR = "error"
    WARN = "warn"


class VertexOnlyMeshMissingPointsError(Exception):
    """Exception raised when 1 or more points are not found by a
    :func:`~.VertexOnlyMesh` in its parent mesh.

    Attributes
    ----------
    n_missing_points : int
        The number of points which were not found in the parent mesh.
    """

    def __init__(self, n_missing_points):
        self.n_missing_points = n_missing_points

    def __str__(self):
        return (
            f"{self.n_missing_points} vertices are outside the mesh and have "
            "been removed from the VertexOnlyMesh."
        )


@PETSc.Log.EventDecorator()
def VertexOnlyMesh(mesh, vertexcoords, reorder=None, missing_points_behaviour='error',
                   tolerance=None, redundant=True, name=None):
    """
    Create a vertex only mesh, immersed in a given mesh, with vertices defined
    by a list of coordinates.

    :arg mesh: The unstructured mesh in which to immerse the vertex only mesh.
    :arg vertexcoords: A list of coordinate tuples which defines the vertices.
    :kwarg reorder: optional flag indicating whether to reorder
           meshes for better cache locality.  If not supplied the
           default value in ``parameters["reorder_meshes"]``
           is used.
    :kwarg missing_points_behaviour: optional string argument for what to do
        when vertices which are outside of the mesh are discarded. If
        ``'warn'``, will print a warning. If ``'error'`` will raise a
        :class:`~.VertexOnlyMeshMissingPointsError`.
    :kwarg tolerance: The relative tolerance (i.e. as defined on the reference
        cell) for the distance a point can be from a mesh cell and still be
        considered to be in the cell. Note that this tolerance uses an L1
        distance (aka 'manhattan', 'taxicab' or rectilinear distance) so
        will scale with the dimension of the mesh. The default is the parent
        mesh's ``tolerance`` property. Changing this from default will
        cause the parent mesh's spatial index to be rebuilt which can take some
        time.
    :kwarg redundant: If True, the mesh will be built using just the vertices
        which are specified on rank 0. If False, the mesh will be built using
        the vertices specified by each rank. Care must be taken when using
        ``redundant = False``: see the note below for more information.
    :kwarg name: Optional name for the new ``VertexOnlyMesh``. If none is
        specified a name will be generated from the parent mesh name.

    .. note::

        The vertex only mesh uses the same communicator as the input ``mesh``.

    .. note::

        Extruded meshes with variable extrusion layers are not yet supported.
        See note below about ``VertexOnlyMesh`` as input.

    .. note::
        When running in parallel with ``redundant = False``, ``vertexcoords``
        will redistribute to the mesh partition where they are located. This
        means that if rank A has ``vertexcoords`` {X} that are not found in the
        mesh cells owned by rank A but are found in the mesh cells owned by
        rank B, then they will be moved to rank B.

    .. note::
        If the same coordinates are supplied more than once, they are always
        assumed to be a new vertex.

    """
    from firedrake_citations import Citations
    Citations().register("nixonhill2023consistent")

    if tolerance is None:
        tolerance = mesh.tolerance
    else:
        mesh.tolerance = tolerance
    vertexcoords = np.asarray(vertexcoords, dtype=RealType)
    if reorder is None:
        reorder = parameters["reorder_meshes"]
    gdim = mesh.geometric_dimension()
    _, pdim = vertexcoords.shape
    if not np.isclose(np.sum(abs(vertexcoords.imag)), 0):
        raise ValueError("Point coordinates must have zero imaginary part")
    # Bendy meshes require a smarter bounding box algorithm at partition and
    # (especially) cell level. Projecting coordinates to Bernstein may be
    # sufficient.
    if np.any(np.asarray(mesh.coordinates.function_space().ufl_element().degree()) > 1):
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
    swarm, input_ordering_swarm, n_missing_points = _pic_swarm_in_mesh(
        mesh, vertexcoords, tolerance=tolerance, redundant=redundant, exclude_halos=False
    )
    missing_points_behaviour = MissingPointsBehaviour(missing_points_behaviour)
    if missing_points_behaviour != MissingPointsBehaviour.IGNORE:
        if n_missing_points:
            error = VertexOnlyMeshMissingPointsError(n_missing_points)
            if missing_points_behaviour == MissingPointsBehaviour.ERROR:
                raise error
            elif missing_points_behaviour == MissingPointsBehaviour.WARN:
                from warnings import warn
                warn(str(error))
            else:
                raise ValueError("missing_points_behaviour must be IGNORE, ERROR or WARN")
    name = name if name is not None else mesh.name + "_immersed_vom"
    swarm.setName(_generate_default_mesh_topology_name(name))
    input_ordering_swarm.setName(_generate_default_mesh_topology_name(name) + "_input_ordering")
    topology = VertexOnlyMeshTopology(
        swarm,
        mesh.topology,
        name=swarm.getName(),
        reorder=reorder,
        input_ordering_swarm=input_ordering_swarm,
    )
    vmesh_out = make_vom_from_vom_topology(topology, name, tolerance)
    vmesh_out._parent_mesh = mesh
    return vmesh_out


class FiredrakeDMSwarm(PETSc.DMSwarm):
    """A DMSwarm with a saved list of added fields"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fields = None
        self._default_fields = None
        self._default_extra_fields = None
        self._other_fields = None

    @property
    def fields(self):
        return self._fields

    @fields.setter
    def fields(self, fields):
        if self._fields:
            raise ValueError("Fields have already been set")
        self._fields = fields

    @property
    def default_fields(self):
        return self._default_fields

    @default_fields.setter
    def default_fields(self, fields):
        if self._default_fields:
            raise ValueError("Default fields have already been set")
        self._default_fields = fields

    @property
    def default_extra_fields(self):
        return self._default_extra_fields

    @default_extra_fields.setter
    def default_extra_fields(self, fields):
        if self._default_extra_fields:
            raise ValueError("Default extra fields have already been set")
        self._default_extra_fields = fields

    @property
    def other_fields(self):
        return self._other_fields

    @other_fields.setter
    def other_fields(self, fields):
        if self._other_fields:
            raise ValueError("Other fields have already been set")
        self._other_fields = fields


def _pic_swarm_in_mesh(
    parent_mesh,
    coords,
    fields=None,
    tolerance=None,
    redundant=True,
    exclude_halos=True,
):
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
        <https://petsc.org/release/manualpages/DMSwarm/DMSWARM/>_.
    :kwarg tolerance: The relative tolerance (i.e. as defined on the reference
        cell) for the distance a point can be from a cell and still be
        considered to be in the cell. Note that this tolerance uses an L1
        distance (aka 'manhattan', 'taxicab' or rectilinear distance) so
        will scale with the dimension of the mesh. The default is the parent
        mesh's ``tolerance`` property. Changing this from default will
        cause the parent mesh's spatial index to be rebuilt which can take some
        time.
    :kwarg redundant: If True, the DMSwarm will be created using only the
        points specified on MPI rank 0.
    :kwarg exclude_halos: If True, the DMSwarm will not contain any points in
        the mesh halos. If False, it will but the global index of the points
        in the halos will match a global index of a point which is not in the
        halo.
    :returns: (swarm, input_ordering_swarm, n_missing_points)
        - swarm: the immersed DMSwarm
        - input_ordering_swarm: a DMSwarm with points in the same order and with the
            same rank decomposition as the supplied ``coords`` argument. This
            includes any points which are not found in the parent mesh! Note
            that if ``redundant=True``, all points in the generated DMSwarm
            will be found on rank 0 since that was where they were taken from.
        - n_missing_points: the number of points in the supplied ``coords``
            argument which were not found in the parent mesh.

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
        #. ``inputrank`` which contains the MPI rank at which the ``coords``
           argument was specified. For ``redundant=True`` this is always 0.
        #. ``inputindex`` which contains the index of the point in the
           originally supplied ``coords`` array after it has been redistributed
           to the correct rank. For ``redundant=True`` this is always the same
           as ``globalindex`` since we only use the points on rank 0.

        If the parent mesh is extruded, two more fields are created:

        #. ``parentcellbasenum`` which contains the firedrake cell number of the
            base cell of the immersed vertex and
        #. ``parentcellextrusionheight`` which contains the extrusion height of
            the immersed vertex in the parent mesh cell.

        Another two are required for proper functioning of the DMSwarm:

        #. ``DMSwarmPIC_coor`` which contains the coordinates of the point.
        #. ``DMSwarm_rank``: the MPI rank which owns the DMSwarm point.

    .. note::
        All PIC DMSwarm have an associated "Cell DM", if one wishes to interact
        directly with PETSc's DMSwarm API. For the ``swarm`` output, this is
        the parent mesh's topology DM (in most cases a DMPlex). For the
        ``input_ordering_swarm`` output, this is the ``swarm`` itself.

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

    (
        coords_local,
        global_idxs_local,
        reference_coords_local,
        parent_cell_nums_local,
        owned_ranks_local,
        input_ranks_local,
        input_coords_idxs_local,
        missing_global_idxs,
    ) = _parent_mesh_embedding(
        parent_mesh,
        coords,
        tolerance,
        redundant,
        exclude_halos,
        remove_missing_points=False,
    )
    visible_idxs = parent_cell_nums_local != -1
    if parent_mesh.extruded:
        # need to store the base parent cell number and the height to be able
        # to map point coordinates back to the parent mesh
        if parent_mesh.variable_layers:
            raise NotImplementedError(
                "Cannot create a DMSwarm in an ExtrudedMesh with variable layers."
            )
        base_parent_cell_nums, extrusion_heights = _parent_extrusion_numbering(
            parent_cell_nums_local, parent_mesh.layers
        )
        # mesh.topology.cell_closure[:, -1] maps Firedrake cell numbers to plex
        # numbers.
        plex_parent_cell_nums = parent_mesh.topology.cell_closure[
            base_parent_cell_nums, -1
        ]
        base_parent_cell_nums_visible = base_parent_cell_nums[visible_idxs]
        extrusion_heights_visible = extrusion_heights[visible_idxs]
    else:
        plex_parent_cell_nums = parent_mesh.topology.cell_closure[
            parent_cell_nums_local, -1
        ]
        base_parent_cell_nums_visible = None
        extrusion_heights_visible = None
    n_missing_points = len(missing_global_idxs)

    # Exclude the invisible points at this stage
    swarm = _dmswarm_create(
        fields,
        parent_mesh.comm,
        plex,
        coords_local[visible_idxs],
        plex_parent_cell_nums[visible_idxs],
        global_idxs_local[visible_idxs],
        reference_coords_local[visible_idxs],
        parent_cell_nums_local[visible_idxs],
        owned_ranks_local[visible_idxs],
        input_ranks_local[visible_idxs],
        input_coords_idxs_local[visible_idxs],
        base_parent_cell_nums_visible,
        extrusion_heights_visible,
        parent_mesh.extruded,
        tdim,
        gdim,
    )
    # Note when getting original ordering for extruded meshes we recalculate
    # the base_parent_cell_nums and extrusion_heights - note this could
    # be an SF operation
    if redundant and parent_mesh.comm.rank != 0:
        original_ordering_swarm_coords = np.empty(shape=(0, coords.shape[1]))
    else:
        original_ordering_swarm_coords = coords
    # Set pointSF
    # In the below, n merely defines the local size of an array, local_points_reduced,
    # that works as "broker". The set of indices of local_points_reduced is the target of
    # inputindex; see _make_input_ordering_sf. All points in local_points are leaves.
    # Then, local_points[halo_indices] = -1, local_points_reduced.fill(-1), and MPI.MAX ensure that local_points_reduced has
    # the swarm local point numbers of the owning ranks after reduce. local_points_reduced[j] = -1
    # if j corresponds to a missing point. Then, broadcast updates
    # local_points[halo_indices] (it also updates local_points[~halo_indices]`, not changing any values there).
    # If some index of local_points_reduced corresponds to a missing point, local_points_reduced[index] is not updated
    # when we reduce and it does not update any leaf data, i.e., local_points, when we bcast.
    owners = swarm.getField("DMSwarm_rank").ravel()
    halo_indices, = np.where(owners != parent_mesh.comm.rank)
    halo_indices = halo_indices.astype(IntType)
    n = coords.shape[0]
    m = owners.shape[0]
    _swarm_input_ordering_sf = VertexOnlyMeshTopology._make_input_ordering_sf(swarm, n, None)  # sf: swarm local point <- (inputrank, inputindex)
    local_points_reduced = np.empty(n, dtype=utils.IntType)
    local_points_reduced.fill(-1)
    local_points = np.arange(m, dtype=utils.IntType)  # swarm local point numbers
    local_points[halo_indices] = -1
    unit = MPI._typedict[np.dtype(utils.IntType).char]
    _swarm_input_ordering_sf.reduceBegin(unit, local_points, local_points_reduced, MPI.MAX)
    _swarm_input_ordering_sf.reduceEnd(unit, local_points, local_points_reduced, MPI.MAX)
    _swarm_input_ordering_sf.bcastBegin(unit, local_points_reduced, local_points, MPI.REPLACE)
    _swarm_input_ordering_sf.bcastEnd(unit, local_points_reduced, local_points, MPI.REPLACE)
    if np.any(local_points < 0):
        raise RuntimeError("Unable to make swarm pointSF due to inconsistent data")
    # Interleave each rank and index into (rank, index) pairs for use as remote
    # in the SF
    remote_ranks_and_idxs = np.empty(2 * len(halo_indices), dtype=IntType)
    remote_ranks_and_idxs[0::2] = owners[halo_indices]
    remote_ranks_and_idxs[1::2] = local_points[halo_indices]
    swarm.restoreField("DMSwarm_rank")
    sf = swarm.getPointSF()
    sf.setGraph(m, halo_indices, remote_ranks_and_idxs)
    swarm.setPointSF(sf)
    original_ordering_swarm = _swarm_original_ordering_preserve(
        parent_mesh.comm,
        swarm,
        original_ordering_swarm_coords,
        plex_parent_cell_nums,
        global_idxs_local,
        reference_coords_local,
        parent_cell_nums_local,
        owned_ranks_local,
        input_ranks_local,  # This is just an array of 0s for redundant, and comm.rank otherwise. But I need to pass it in to get the correct ordering
        input_coords_idxs_local,
        parent_mesh.extruded,
        parent_mesh.layers,
    )

    # no halos here
    sf = original_ordering_swarm.getPointSF()
    nroots = original_ordering_swarm.getLocalSize()
    sf.setGraph(nroots, None, [])
    original_ordering_swarm.setPointSF(sf)

    return swarm, original_ordering_swarm, n_missing_points


def _dmswarm_create(
    fields,
    comm,
    plex,
    coords,
    plex_parent_cell_nums,
    coords_idxs,
    reference_coords,
    parent_cell_nums,
    ranks,
    input_ranks,
    input_coords_idxs,
    base_parent_cell_nums,
    extrusion_heights,
    extruded,
    tdim,
    gdim,
):

    """
    Create a PIC DMSwarm (or DMSwarm that looks like it's a PIC DMSwarm) using
    the given data.

    Parameters
    ----------

    fields : list of tuples
        List of tuples of the form (name, number of components, type) for any
        additional fields to be added to the DMSwarm. The default fields are
        automatically added and do not need to be specified here. Can be an
        empty list if no additional fields are required.
    comm : MPI communicator
        The MPI communicator to use when creating the DMSwarm.
    plex : PETSc DM
        The DM to set as the "CellDM" of the DMSwarm - i.e. the DMPlex or
        DMSwarm of the parent mesh.
    coords : numpy array of RealType with shape (npoints, gdim)
        The coordinates of the particles in the DMSwarm.
    plex_parent_cell_nums : numpy array of IntType with shape (npoints,)
        Array to be used as the "parentcellnum" field of the DMSwarm.
    coords_idxs : numpy array of IntType with shape (npoints,)
        Array to be used as the "globalindex" field of the DMSwarm.
    reference_coords : numpy array of RealType with shape (npoints, tdim)
        Array to be used as the "refcoord" field of the DMSwarm.
    parent_cell_nums : numpy array of IntType with shape (npoints,)
        Array to be used as the "parentcellnum" field of the DMSwarm.
    ranks : numpy array of IntType with shape (npoints,)
        Array to be used as the "DMSwarm_rank" field of the DMSwarm.
    input_ranks : numpy array of IntType with shape (npoints,)
        Array to be used as the "inputrank" field of the DMSwarm.
    input_coords_idxs : numpy array of IntType with shape (npoints,)
        Array to be used as the "inputindex" field of the DMSwarm.
    base_parent_cell_nums : numpy array of IntType with shape (npoints,) (or None)
        Optional array to be used as the "parentcellbasenum" field of the
        DMSwarm. Must be provided if extruded=True.
    extrusion_heights : numpy array of IntType with shape (npoints,) (or None)
        Optional array to be used as the "parentcellextrusionheight" field of
        the DMSwarm. Must be provided if extruded=True.
    extruded : bool
        Whether the parent mesh is extruded.
    tdim : int
        The topological dimension of the parent mesh.
    gdim : int
        The geometric dimension of the parent mesh.

    Returns
    -------
    swarm : PETSc DMSwarm
        The created DMSwarm.

    Notes
    -----
    When the `plex` is a DMSwarm, the created DMSwarm isn't actually a PIC
    DMSwarm, but it has all the associated fields of a PIC DMSwarm. This is
    because PIC DMSwarms cannot have their "CellDM" set to a DMSwarm, so we
    fake it!
    """

    # These are created by default for a PIC DMSwarm
    default_fields = [
        ("DMSwarmPIC_coor", gdim, RealType),
        ("DMSwarm_rank", 1, IntType),
    ]

    default_extra_fields = [
        ("parentcellnum", 1, IntType),
        ("refcoord", tdim, RealType),
        ("globalindex", 1, IntType),
        ("inputrank", 1, IntType),
        ("inputindex", 1, IntType),
    ]

    if extruded:
        default_extra_fields += [
            ("parentcellbasenum", 1, IntType),
            ("parentcellextrusionheight", 1, IntType),
        ]

    other_fields = fields
    if other_fields is None:
        other_fields = []

    _, coordsdim = coords.shape

    # Create a DMSWARM
    swarm = FiredrakeDMSwarm().create(comm=comm)

    # save the fields on the swarm
    swarm.fields = default_fields + default_extra_fields + other_fields
    swarm.default_fields = default_fields
    swarm.default_extra_fields = default_extra_fields
    swarm.other_fields = other_fields

    plexdim = plex.getDimension()
    if plexdim != tdim or plexdim != gdim:
        # This is a Firedrake extruded or immersed mesh, so we need to use the
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
    if not isinstance(plex, PETSc.DMSwarm):
        swarm.setType(PETSc.DMSwarm.Type.PIC)

    # Register any fields
    for name, size, dtype in swarm.default_extra_fields + swarm.other_fields:
        swarm.registerField(name, size, dtype=dtype)
    swarm.finalizeFieldRegister()
    # Note that no new fields can now be associated with the DMSWARM.

    num_vertices = len(coords)
    swarm.setLocalSizes(num_vertices, -1)

    # Add point coordinates. This amounts to our own implementation of
    # DMSwarmSetPointCoordinates because Firedrake's mesh coordinate model
    # doesn't always exactly coincide with that of DMPlex: in most cases the
    # plex_parent_cell_nums and parent_cell_nums (parentcellnum field), the
    # latter being the numbering used by firedrake, refer fundamentally to the
    # same cells. For extruded meshes the DMPlex dimension is based on the
    # topological dimension of the base mesh.

    # NOTE ensure that swarm.restoreField is called for each field too!
    swarm_coords = swarm.getField("DMSwarmPIC_coor").reshape((num_vertices, gdim))
    cell_id_name = swarm.getCellDMActive().getCellID()
    swarm_parent_cell_nums = swarm.getField(cell_id_name).ravel()
    field_parent_cell_nums = swarm.getField("parentcellnum").ravel()
    field_reference_coords = swarm.getField("refcoord").reshape((num_vertices, tdim))
    field_global_index = swarm.getField("globalindex").ravel()
    field_rank = swarm.getField("DMSwarm_rank").ravel()
    field_input_rank = swarm.getField("inputrank").ravel()
    field_input_index = swarm.getField("inputindex").ravel()
    swarm_coords[...] = coords
    swarm_parent_cell_nums[...] = plex_parent_cell_nums
    field_parent_cell_nums[...] = parent_cell_nums
    field_reference_coords[...] = reference_coords
    field_global_index[...] = coords_idxs
    field_rank[...] = ranks
    field_input_rank[...] = input_ranks
    field_input_index[...] = input_coords_idxs

    # have to restore fields once accessed to allow access again
    swarm.restoreField("inputindex")
    swarm.restoreField("inputrank")
    swarm.restoreField("DMSwarm_rank")
    swarm.restoreField("globalindex")
    swarm.restoreField("refcoord")
    swarm.restoreField("parentcellnum")
    swarm.restoreField("DMSwarmPIC_coor")
    swarm.restoreField(cell_id_name)

    if extruded:
        field_base_parent_cell_nums = swarm.getField("parentcellbasenum").ravel()
        field_extrusion_heights = swarm.getField("parentcellextrusionheight").ravel()
        field_base_parent_cell_nums[...] = base_parent_cell_nums
        field_extrusion_heights[...] = extrusion_heights
        swarm.restoreField("parentcellbasenum")
        swarm.restoreField("parentcellextrusionheight")

    return swarm


def _parent_extrusion_numbering(parent_cell_nums, parent_layers):
    """
    Given a list of Firedrake cell numbers (e.g. from mesh.locate_cell) and
    number of layers, get the base parent cell numbers and extrusion heights.

    Parameters
    ----------

    parent_cell_nums : ``np.ndarray``
        Firedrake cell numbers (e.g. from mesh.locate_cell)
    parent_layers : ``int``
        Number of layers in the extruded mesh

    Returns
    -------
    base_parent_cell_nums : ``np.ndarray``
        The base parent cell numbers
    extrusion_heights : ``np.ndarray``
        The extrusion heights

    Notes
    -----
    Only works for meshes without variable layers.
    """
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
    base_parent_cell_nums = parent_cell_nums // (parent_layers - 1)
    extrusion_heights = parent_cell_nums % (parent_layers - 1)
    return base_parent_cell_nums, extrusion_heights


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


def _parent_mesh_embedding(
    parent_mesh, coords, tolerance, redundant, exclude_halos, remove_missing_points
):
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
        distance (aka 'manhattan', 'taxicab' or rectilinear distance) so
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
    remove_missing_points : ``bool``
        If True, any points which are not found in the mesh will be removed
        from the output arrays. If False, they will be kept on the MPI rank
        which owns them but will be marked as not being not found in the mesh
        by setting their associated cell numbers to -1 and their reference
        coordinates to NaNs. This does not effect the behaviour of
        ``missing_global_idxs``.

    Returns
    -------
    coords_embedded : ``np.ndarray``
        The coordinates of the points that were embedded on this rank. If
        ``remove_missing_points`` is False then this will include points that
        were specified on this rank but not found in the mesh.
    global_idxs : ``np.ndarray``
        The global indices of the points on this rank.
    reference_coords : ``np.ndarray``
        The reference coordinates of the points that were embedded as given by
        the local mesh partition. If ``remove_missing_points`` is False then
        the missing point reference coordinates will be NaNs.
    parent_cell_nums : ``np.ndarray``
        The parent cell indices (as given by ``locate_cell``) of the global
        coordinates that were embedded in the local mesh partition. If
        ``remove_missing_points`` is False then the missing point numbers
        will be -1.
    owned_ranks : ``np.ndarray``
        The MPI rank of the process that owns the parent cell of each point.
        By "owns" we mean the mesh partition where the parent cell is not in
        the halo. If a point is not found in the mesh then the rank is
        ``parent_mesh.comm.size + 1``.
    input_ranks : ``np.ndarray``
        The MPI rank of the process that specified the input ``coords``.
    input_coords_idx : ``np.ndarray``
        The indices of the points in the input ``coords`` array that were
        embedded. If ``remove_missing_points`` is False then this will include
        points that were specified on this rank but not found in the mesh.
    missing_global_idxs : ``np.ndarray``
        The indices of the points in the input coords array that were not
        embedded on any rank.

    .. note::
        Where we have ``exclude_halos == True`` and ``remove_missing_points ==
        False``, and we run in parallel, the points are ordered such that the
        halo points follow the owned points. Any missing points will be at the
        end of the array. This is to ensure that dat views work as expected -
        in general it is always assumed that halo points follow owned points.

    """

    if isinstance(parent_mesh.topology, VertexOnlyMeshTopology):
        raise NotImplementedError(
            "VertexOnlyMeshes don't have a working locate_cells_ref_coords_and_dists method"
        )

    import firedrake.functionspace as functionspace
    import firedrake.constant as constant
    import firedrake.interpolation as interpolation
    import firedrake.assemble as assemble

    # In parallel, we need to make sure we know which point is which and save
    # it.
    if redundant:
        # rank 0 broadcasts coords to all ranks
        coords_local = parent_mesh._comm.bcast(coords, root=0)
        ncoords_local = coords_local.shape[0]
        coords_global = coords_local
        ncoords_global = coords_global.shape[0]
        global_idxs_global = np.arange(coords_global.shape[0])
        input_coords_idxs_local = np.arange(ncoords_local)
        input_coords_idxs_global = input_coords_idxs_local
        input_ranks_local = np.zeros(ncoords_local, dtype=int)
        input_ranks_global = input_ranks_local
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
        # global_idxs_global = np.arange(startidx, endidx)
        global_idxs_global = np.arange(coords_global.shape[0])
        input_coords_idxs_local = np.arange(ncoords_local)
        input_coords_idxs_global = np.empty(ncoords_global, dtype=int)
        parent_mesh._comm.Allgatherv(
            input_coords_idxs_local, (input_coords_idxs_global, ncoords_local_allranks)
        )
        input_ranks_local = np.full(ncoords_local, parent_mesh._comm.rank, dtype=int)
        input_ranks_global = np.empty(ncoords_global, dtype=int)
        parent_mesh._comm.Allgatherv(
            input_ranks_local, (input_ranks_global, ncoords_local_allranks)
        )

    # Get parent mesh rank ownership information:
    # Interpolating Constant(parent_mesh.comm.rank) into P0DG cleverly creates
    # a Function whose dat contains rank ownership information in an ordering
    # that is accessible using Firedrake's cell numbering. This is because, on
    # each rank, parent_mesh.comm.rank creates a Constant with the local rank
    # number, and halo exchange ensures that this information is visible, as
    # nessesary, to other processes.
    P0DG = functionspace.FunctionSpace(parent_mesh, "DG", 0)
    with stop_annotating():
        visible_ranks = interpolation.Interpolate(
            constant.Constant(parent_mesh.comm.rank), P0DG
        )
        visible_ranks = assemble(visible_ranks).dat.data_ro_with_halos.real

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

    if parent_mesh.geometric_dimension() > parent_mesh.topological_dimension():
        # The reference coordinates contain an extra unnecessary dimension
        # which we can safely delete
        reference_coords = reference_coords[:, : parent_mesh.topological_dimension()]

    locally_visible[:] = parent_cell_nums != -1
    ranks[locally_visible] = visible_ranks[parent_cell_nums[locally_visible]]
    # see below for why np.inf is used here.
    ref_cell_dists_l1[~locally_visible] = np.inf

    # ensure that points which a rank thinks it owns are always chosen in a tie
    # break by setting the rank to be negative. If multiple ranks think they
    # own a point then the one with the highest rank will be chosen.
    on_this_rank = ranks == parent_mesh.comm.rank
    ranks[on_this_rank] = -parent_mesh.comm.rank
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
    # break ties by choosing the lowest rank.
    owned_ref_cell_dists_l1_and_ranks = parent_mesh.comm.allreduce(
        ref_cell_dists_l1_and_ranks, op=array_lexicographic_mpi_op
    )

    # switch ranks back to positive
    owned_ref_cell_dists_l1_and_ranks[:, 1] = np.abs(
        owned_ref_cell_dists_l1_and_ranks[:, 1]
    )
    ref_cell_dists_l1_and_ranks[:, 1] = np.abs(ref_cell_dists_l1_and_ranks[:, 1])
    ranks = np.abs(ranks)

    owned_ref_cell_dists_l1 = owned_ref_cell_dists_l1_and_ranks[:, 0]
    owned_ranks = owned_ref_cell_dists_l1_and_ranks[:, 1]

    changed_ref_cell_dists_l1 = owned_ref_cell_dists_l1 != ref_cell_dists_l1
    changed_ranks = owned_ranks != ranks

    # If distance has changed the the point is not in local mesh partition
    # since some other cell on another rank is closer.
    locally_visible[changed_ref_cell_dists_l1] = False
    parent_cell_nums[changed_ref_cell_dists_l1] = -1
    # If the rank has changed but the distance hasn't then there was a tie
    # break and we need to search for the point again, this time disallowing
    # the previously identified cell: if we match the identified owned_rank AND
    # the distance is the same then we have found the correct cell. If we
    # cannot make a match to owned_rank and distance then we can't see the
    # point.
    changed_ranks_tied = changed_ranks & ~changed_ref_cell_dists_l1
    if any(changed_ranks_tied):
        cells_ignore_T = np.asarray([np.copy(parent_cell_nums)])
        while any(changed_ranks_tied):
            (
                parent_cell_nums[changed_ranks_tied],
                new_reference_coords,
                ref_cell_dists_l1[changed_ranks_tied],
            ) = parent_mesh.locate_cells_ref_coords_and_dists(
                coords_global[changed_ranks_tied],
                tolerance,
                cells_ignore=cells_ignore_T.T[changed_ranks_tied, :],
            )
            # delete extra dimension if necessary
            if parent_mesh.geometric_dimension() > parent_mesh.topological_dimension():
                new_reference_coords = new_reference_coords[:, : parent_mesh.topological_dimension()]
            reference_coords[changed_ranks_tied, :] = new_reference_coords
            # remove newly lost points
            locally_visible[changed_ranks_tied] = (
                parent_cell_nums[changed_ranks_tied] != -1
            )
            changed_ranks_tied &= locally_visible
            # if new ref_cell_dists_l1 > owned_ref_cell_dists_l1 then we should
            # disregard the point.
            locally_visible[changed_ranks_tied] &= (
                ref_cell_dists_l1[changed_ranks_tied]
                <= owned_ref_cell_dists_l1[changed_ranks_tied]
            )
            changed_ranks_tied &= locally_visible
            # update the identified rank
            ranks[changed_ranks_tied] = visible_ranks[
                parent_cell_nums[changed_ranks_tied]
            ]
            # if the rank now matches then we have found the correct cell
            locally_visible[changed_ranks_tied] &= (
                owned_ranks[changed_ranks_tied] == ranks[changed_ranks_tied]
            )
            # remove these rank matches from changed_ranks_tied
            changed_ranks_tied &= ~locally_visible
            # add more cells to ignore
            cells_ignore_T = np.vstack((
                cells_ignore_T,
                parent_cell_nums)
            )

    # Any ranks which are still np.inf are not in the mesh
    missing_global_idxs = np.where(owned_ranks == np.inf)[0]

    if not remove_missing_points:
        missing_coords_idxs_on_rank = np.where(
            (owned_ranks == np.inf) & (input_ranks_global == parent_mesh.comm.rank)
        )[0]
        locally_visible[missing_coords_idxs_on_rank] = True
        parent_cell_nums[missing_coords_idxs_on_rank] = -1
        reference_coords[missing_coords_idxs_on_rank, :] = np.nan
        owned_ranks[missing_coords_idxs_on_rank] = parent_mesh.comm.size + 1

    if exclude_halos and parent_mesh.comm.size > 1:
        off_rank_coords_idxs = np.where(
            (owned_ranks != parent_mesh.comm.rank)
            & (owned_ranks != parent_mesh.comm.size + 1)
        )[0]
        locally_visible[off_rank_coords_idxs] = False

    coords_embedded = np.compress(locally_visible, coords_global, axis=0)
    global_idxs = np.compress(locally_visible, global_idxs_global, axis=0)
    reference_coords = np.compress(locally_visible, reference_coords, axis=0)
    parent_cell_nums = np.compress(locally_visible, parent_cell_nums, axis=0)
    owned_ranks = np.compress(locally_visible, owned_ranks, axis=0).astype(int)
    input_ranks = np.compress(locally_visible, input_ranks_global, axis=0)
    input_coords_idxs = np.compress(locally_visible, input_coords_idxs_global, axis=0)

    return (
        coords_embedded,
        global_idxs,
        reference_coords,
        parent_cell_nums,
        owned_ranks,
        input_ranks,
        input_coords_idxs,
        missing_global_idxs,
    )


def _swarm_original_ordering_preserve(
    comm,
    swarm,
    original_ordering_coords_local,
    plex_parent_cell_nums_local,
    global_idxs_local,
    reference_coords_local,
    parent_cell_nums_local,
    ranks_local,
    input_ranks_local,
    input_idxs_local,
    extruded,
    layers,
):
    """
    Create a DMSwarm with the original ordering of the coordinates in a vertex
    only mesh embedded using ``_parent_mesh_embedding`` whilst preserving the
    values of all other DMSwarm fields except any added fields.
    """
    ncoords_local = len(reference_coords_local)
    gdim = original_ordering_coords_local.shape[1]
    tdim = reference_coords_local.shape[1]

    # Gather everything except original_ordering_coords_local from all mpi
    # ranks
    ncoords_local_allranks = comm.allgather(ncoords_local)
    ncoords_global = sum(ncoords_local_allranks)

    parent_cell_nums_global = np.empty(
        ncoords_global, dtype=parent_cell_nums_local.dtype
    )
    comm.Allgatherv(
        parent_cell_nums_local, (parent_cell_nums_global, ncoords_local_allranks)
    )

    plex_parent_cell_nums_global = np.empty(
        ncoords_global, dtype=plex_parent_cell_nums_local.dtype
    )
    comm.Allgatherv(
        plex_parent_cell_nums_local,
        (plex_parent_cell_nums_global, ncoords_local_allranks),
    )

    reference_coords_local_size = np.array(reference_coords_local.size)
    reference_coords_local_sizes = np.empty(comm.size, dtype=int)
    comm.Allgatherv(reference_coords_local_size, reference_coords_local_sizes)
    reference_coords_global = np.empty(
        (ncoords_global, reference_coords_local.shape[1]),
        dtype=reference_coords_local.dtype,
    )
    comm.Allgatherv(
        reference_coords_local, (reference_coords_global, reference_coords_local_sizes)
    )

    global_idxs_global = np.empty(ncoords_global, dtype=global_idxs_local.dtype)
    comm.Allgatherv(global_idxs_local, (global_idxs_global, ncoords_local_allranks))

    ranks_global = np.empty(ncoords_global, dtype=ranks_local.dtype)
    comm.Allgatherv(ranks_local, (ranks_global, ncoords_local_allranks))

    input_ranks_global = np.empty(ncoords_global, dtype=input_ranks_local.dtype)
    comm.Allgatherv(input_ranks_local, (input_ranks_global, ncoords_local_allranks))

    input_idxs_global = np.empty(ncoords_global, dtype=input_idxs_local.dtype)
    comm.Allgatherv(input_idxs_local, (input_idxs_global, ncoords_local_allranks))

    # Sort by global index, which will be in rank order (they probably already
    # are but we can't rely on that)
    global_idxs_global_order = np.argsort(global_idxs_global)
    sorted_parent_cell_nums_global = parent_cell_nums_global[global_idxs_global_order]
    sorted_plex_parent_cell_nums_global = plex_parent_cell_nums_global[
        global_idxs_global_order
    ]
    sorted_reference_coords_global = reference_coords_global[
        global_idxs_global_order, :
    ]
    sorted_global_idxs_global = global_idxs_global[global_idxs_global_order]
    sorted_ranks_global = ranks_global[global_idxs_global_order]
    sorted_input_ranks_global = input_ranks_global[global_idxs_global_order]
    sorted_input_idxs_global = input_idxs_global[global_idxs_global_order]
    # Check order is correct - we can probably remove this eventually since it's
    # quite expensive
    if not np.all(sorted_input_ranks_global[1:] >= sorted_input_ranks_global[:-1]):
        raise ValueError("Global indexing has not ordered the ranks as expected")

    # get rid of any duplicated global indices (i.e. points in halos)
    unique_global_idxs, unique_idxs = np.unique(
        sorted_global_idxs_global, return_index=True
    )
    unique_parent_cell_nums_global = sorted_parent_cell_nums_global[unique_idxs]
    unique_plex_parent_cell_nums_global = sorted_plex_parent_cell_nums_global[
        unique_idxs
    ]
    unique_reference_coords_global = sorted_reference_coords_global[unique_idxs, :]
    unique_ranks_global = sorted_ranks_global[unique_idxs]
    unique_input_ranks_global = sorted_input_ranks_global[unique_idxs]
    unique_input_idxs_global = sorted_input_idxs_global[unique_idxs]

    # save the points on this rank which match the input rank ready for output
    input_ranks_match = unique_input_ranks_global == comm.rank
    output_global_idxs = unique_global_idxs[input_ranks_match]
    output_parent_cell_nums = unique_parent_cell_nums_global[input_ranks_match]
    output_plex_parent_cell_nums = unique_plex_parent_cell_nums_global[
        input_ranks_match
    ]
    output_reference_coords = unique_reference_coords_global[input_ranks_match, :]
    output_ranks = unique_ranks_global[input_ranks_match]
    output_input_ranks = unique_input_ranks_global[input_ranks_match]
    output_input_idxs = unique_input_idxs_global[input_ranks_match]
    if extruded:
        (
            output_base_parent_cell_nums,
            output_extrusion_heights,
        ) = _parent_extrusion_numbering(output_parent_cell_nums, layers)
    else:
        output_base_parent_cell_nums = None
        output_extrusion_heights = None

    # check if the input indices are in order from zero - this can also probably
    # be removed eventually because, again, it's expensive.
    if not np.array_equal(output_input_idxs, np.arange(output_input_idxs.size)):
        raise ValueError(
            "Global indexing has not ordered the input indices as expected."
        )
    if len(output_global_idxs) != len(original_ordering_coords_local):
        raise ValueError(
            "The number of local global indices which will be used to make the swarm do not match the input number of original ordering coordinates."
        )
    if len(output_parent_cell_nums) != len(original_ordering_coords_local):
        raise ValueError(
            "The number of local parent cell numbers which will be used to make the swarm do not match the input number of original ordering coordinates."
        )
    if len(output_plex_parent_cell_nums) != len(original_ordering_coords_local):
        raise ValueError(
            "The number of local plex parent cell numbers which will be used to make the swarm do not match the input number of original ordering coordinates."
        )
    if len(output_reference_coords) != len(original_ordering_coords_local):
        raise ValueError(
            "The number of local reference coordinates which will be used to make the swarm do not match the input number of original ordering coordinates."
        )
    if len(output_ranks) != len(original_ordering_coords_local):
        raise ValueError(
            "The number of local rank numbers which will be used to make the swarm do not match the input number of original ordering coordinates."
        )
    if len(output_input_ranks) != len(original_ordering_coords_local):
        raise ValueError(
            "The number of local input rank numbers which will be used to make the swarm do not match the input number of original ordering coordinates."
        )
    if len(output_input_idxs) != len(original_ordering_coords_local):
        raise ValueError(
            "The number of local input indices which will be used to make the swarm do not match the input number of original ordering coordinates."
        )
    if extruded:
        if len(output_base_parent_cell_nums) != len(original_ordering_coords_local):
            raise ValueError(
                "The number of local base parent cell numbers which will be used to make the swarm do not match the input number of original ordering coordinates."
            )
        if len(output_extrusion_heights) != len(original_ordering_coords_local):
            raise ValueError(
                "The number of local extrusion heights which will be used to make the swarm do not match the input number of original ordering coordinates."
            )

    return _dmswarm_create(
        [],
        comm,
        swarm,
        original_ordering_coords_local,
        output_plex_parent_cell_nums,
        output_global_idxs,
        output_reference_coords,
        output_parent_cell_nums,
        output_ranks,
        output_input_ranks,
        output_input_idxs,
        output_base_parent_cell_nums,
        output_extrusion_heights,
        extruded,
        tdim,
        gdim,
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
        Can use "Q" (degree 2) functions for 3D hex meshes until
        we support "HDiv Trace" elements on hex.
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
        if elem.reference_value_shape != ():
            raise RuntimeError(f"indicator functions must be scalar: got {elem.reference_value_shape} != ()")
        if elem.family() in {"Discontinuous Lagrange", "DQ"} and elem.degree() == 0:
            # cells
            height = 0
            dmlabel_name = dmcommon.CELL_SETS_LABEL
        elif (elem.family() == "HDiv Trace" and elem.degree() == 0 and mesh.topological_dimension() > 1) or \
                (elem.family() == "Lagrange" and elem.degree() == 1 and mesh.topological_dimension() == 1) or \
                (elem.family() == "Q" and elem.degree() == 2 and mesh.topology.ufl_cell().cellname() == "hexahedron"):
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
                          perm_is=tmesh._dm_renumbering,
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


def Submesh(mesh, subdim, subdomain_id, label_name=None, name=None):
    """Construct a submesh from a given mesh.

    Parameters
    ----------
    mesh : MeshGeometry
        Parent mesh (`MeshGeometry`).
    subdim : int
        Topological dimension of the submesh.
    subdomain_id : int
        Subdomain ID representing the submesh.
    label_name : str
        Name of the label to search ``subdomain_id`` in.
    name : str
        Name of the submesh.

    Returns
    -------
    MeshGeometry
        Submesh.

    Notes
    -----
    Currently, one can only make submeshes of co-dimension 0 or 1.

    To make a submesh of co-dimension 1, the parent mesh must have
    been overlapped with :class:`DistributedMeshOverlapType` of
    {``None``, `VERTEX``, ``RIDGE``}; see ``distribution_parameters``
    kwarg of :func:`~.Mesh`.

    To use interior facet integration on a submesh of co-dimension 1,
    the parent mesh must have been overlapped with
    ``DistributedMeshOverlapType`` of {`VERTEX``, ``RIDGE``}, and the
    facets of the parent mesh must have been labeled such that the
    ridges (entities of co-dim 2) to be contained in the submesh are
    shared by at most two facets.

    Currently, to make a quadrilateral submesh from a hexahedral mesh,
    the facets of the hex mesh must have been labeled such that the
    ridges to be contained in the quad mesh are shared by at most two
    facets to make the quad mesh orientation algorithm work.

    Examples
    --------

    .. code-block:: python3

        dim = 2
        mesh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
        x, y = SpatialCoordinate(mesh)
        DQ0 = FunctionSpace(mesh, "DQ", 0)
        indicator_function = Function(DQ0).interpolate(conditional(x > 1., 1, 0))
        mesh.mark_entities(indicator_function, 999)
        mesh = RelabeledMesh(mesh, [indicator_function], [999])
        subm = Submesh(mesh, dim, 999)
        V0 = FunctionSpace(mesh, "CG", 1)
        V1 = FunctionSpace(subm, "CG", 1)
        V = V0 * V1
        u = TrialFunction(V)
        v = TestFunction(V)
        u0, u1 = split(u)
        v0, v1 = split(v)
        dx0 = Measure("dx", domain=mesh)
        dx1 = Measure("dx", domain=subm)
        a = inner(u1, v0) * dx0(999) + inner(u0, v1) * dx1
        A = assemble(a)

    """
    if not isinstance(mesh, MeshGeometry):
        raise TypeError("Parent mesh must be a `MeshGeometry`")
    if isinstance(mesh.topology, ExtrudedMeshTopology):
        raise NotImplementedError("Can not create a submesh of an ``ExtrudedMesh``")
    elif isinstance(mesh.topology, VertexOnlyMeshTopology):
        raise NotImplementedError("Can not create a submesh of a ``VertexOnlyMesh``")
    plex = mesh.topology_dm
    dim = plex.getDimension()
    if subdim not in [dim, dim - 1]:
        raise NotImplementedError(f"Found submesh dim ({subdim}) and parent dim ({dim})")
    if label_name is None:
        if subdim == dim:
            label_name = dmcommon.CELL_SETS_LABEL
        elif subdim == dim - 1:
            label_name = dmcommon.FACE_SETS_LABEL
    name = name or _generate_default_submesh_name(mesh.name)
    subplex = dmcommon.submesh_create(plex, subdim, label_name, subdomain_id, False)
    subplex.setName(_generate_default_mesh_topology_name(name))
    if subplex.getDimension() != subdim:
        raise RuntimeError(f"Found subplex dim ({subplex.getDimension()}) != expected ({subdim})")
    submesh = Mesh(
        subplex,
        submesh_parent=mesh,
        name=name,
        distribution_parameters={
            "partition": False,
            "overlap_type": (DistributedMeshOverlapType.NONE, 0),
        },
    )
    return submesh
