import dataclasses
import numpy as np
import ctypes
from contextlib import contextmanager
import os
import sys
import ufl
import finat.ufl
import FIAT
import weakref
from typing import Tuple
from collections import OrderedDict, defaultdict
from collections.abc import Sequence, Generator
from ufl.classes import ReferenceGrad
from ufl.cell import CellSequence
from ufl.domain import extract_unique_domain
import enum
import numbers
import abc
import firedrake_rtree
from textwrap import dedent
from pathlib import Path
import typing
import warnings

from pyop2 import op2
from pyop2.mpi import (
    MPI, COMM_WORLD, temp_internal_comm
)
from functools import cached_property
from pyop2.utils import as_tuple
import petsctools
from petsctools import OptionsManager, get_external_packages

import firedrake.cython.dmcommon as dmcommon
from firedrake.cython.dmcommon import DistributedMeshOverlapType
import firedrake.cython.extrusion_numbering as extnum
import firedrake.extrusion_utils as eutils
import firedrake.cython.rtree as rtree
import firedrake.utils as utils
from firedrake.utils import IntType, IntType_c, RealType, RealType_c, as_ctypes, cached_property_until
from firedrake.logging import logger
from firedrake.parameters import parameters
from firedrake.petsc import PETSc, DEFAULT_PARTITIONER
from firedrake.adjoint_utils import MeshGeometryMixin
from firedrake.exceptions import VertexOnlyMeshMissingPointsError, NonUniqueMeshSequenceError
import gem

try:
    import netgen
except ImportError:
    netgen = None
    ngsPETSc = None
# Only for docstring
import mpi4py  # noqa: F401
from finat.element_factory import as_fiat_cell


if typing.TYPE_CHECKING:
    from firedrake import CoordinatelessFunction, Function


__all__ = [
    'Mesh', 'ExtrudedMesh', 'VertexOnlyMesh', 'RelabeledMesh',
    'SubDomainData', 'UNMARKED', 'DistributedMeshOverlapType',
    'DEFAULT_MESH_NAME', 'MeshGeometry', 'MeshTopology',
    'AbstractMeshTopology', 'ExtrudedMeshTopology', 'VertexOnlyMeshTopology',
    'MeshSequenceGeometry', 'MeshSequenceTopology',
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


UNMARKED = -1
"""A mesh marker that selects all entities that are not explicitly marked."""

DEFAULT_MESH_NAME = "_".join(["firedrake", "default"])
"""The default name of the mesh."""

DISTRIBUTION_PARAMETERS_NOOP = {
    "partition": False,
    "overlap_type": (DistributedMeshOverlapType.NONE, 0),
}
"""Distribution parameters for derived meshes (RelabeledMesh/Submesh)."""


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
    def __init__(self, mesh, facets, classes, set_, kind, facet_cell, local_facet_number,
                 unique_markers=None):

        self.mesh = mesh
        self.facets = facets
        self.classes = classes
        self.set = set_

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
        valid_markers = set([UNMARKED]).union(self.unique_markers)
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
                if i == UNMARKED:
                    _markers = self.mesh.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL).indices
                    # Can exclude points labeled with i\in markers here,
                    # as they will be included in the below anyway.
                    marked_points_list.append(self._collect_unmarked_points([_i for _i in _markers if _i not in markers]))
                else:
                    if self.mesh.topology_dm.getStratumSize(dmcommon.FACE_SETS_LABEL, i):
                        marked_points_list.append(self.mesh.topology_dm.getStratumIS(dmcommon.FACE_SETS_LABEL, i).indices)
            if marked_points_list:
                _, indices, _ = np.intersect1d(self.facets, np.concatenate(marked_points_list), return_indices=True)
            else:
                indices = np.empty(0, dtype=IntType)

            with temp_internal_comm(self.mesh.comm) as icomm:
                num_global_indices = icomm.reduce(len(indices), MPI.SUM, root=0)
                if num_global_indices == 0 and icomm.rank == 0:
                    logger.warn(f"Subdomain {markers} is empty. This is likely an error. "
                                "Did you choose the right label?")

            return self._subsets.setdefault(markers, op2.Subset(self.set, indices))

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

    @cached_property
    def facet_cell_map(self):
        """Map from facets to cells."""
        return op2.Map(self.set, self.mesh.cell_set, self._rank, self.facet_cell,
                       "facet_to_cell_map")

    @cached_property
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
        # this is a map from an exterior/interior facet to the corresponding
        # local facet orientation/orientations.
        # The local facet orientation/orientations of a halo facet is/are also
        # used in some submesh problems.
        #
        #  Example:
        #
        #         +-------+-------+
        #         |       |       |
        #  meshA  |   g   g   o   |
        #         |       |       |
        #         +-------+-------+
        #                 +-------+
        #                 |       |
        #  meshB          o   o   |    o: owned
        #                 |       |    g: ghost
        #                 +-------+
        #
        #  form = FacetNormal(meshA)[0] * ds(meshB, interface)
        #
        # Reshape local_facets as (-1, self._rank) to uniformly handle exterior and interior facets.
        local_facets = self.local_facet_dat.data_ro_with_halos.reshape((-1, self._rank))
        # Make slice for masking out rows for which orientations are not needed.
        slice_ = (self.facet_cell != -1).all(axis=1)
        data = np.full_like(local_facets, np.iinfo(dtype).max)
        data[slice_, :] = np.take_along_axis(
            map_from_cell_to_facet_orientations[self.facet_cell[slice_, :]],
            local_facets.reshape(local_facets.shape + (1, ))[slice_, :, :],  # reshape as required by take_along_axis.
            axis=2,
        ).reshape((-1, self._rank))
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
        plex = plex_from_cell_list(tdim, cells, coordinates, comm)

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
    # These types are /correct/, DMPlexCreateFromCellList wants int
    # and double (not PetscInt, PetscReal).
    with temp_internal_comm(comm) as icomm:
        if comm.rank == 0:
            cells = np.asarray(cells, dtype=np.int32)
            coords = np.asarray(coords, dtype=np.double)
            icomm.bcast(cells.shape, root=0)
            icomm.bcast(coords.shape, root=0)
            # Provide the actual data on rank 0.
            plex = PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=comm)
        else:
            cell_shape = list(icomm.bcast(None, root=0))
            coord_shape = list(icomm.bcast(None, root=0))
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
        self.sfBC_orig = None
        # User comm
        self.user_comm = comm
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

    @property
    @abc.abstractmethod
    def local_cell_orientation_dat(self):
        """Local cell orientation dat."""
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
        return self.ufl_cell().topological_dimension

    def facet_dimension(self):
        """Returns the facet dimension."""
        # Facets have co-dimension 1
        return self.ufl_cell().topological_dimension - 1

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

    @cached_property
    def extruded_periodic(self):
        return self.cell_set._extruded_periodic

    def __iter__(self):
        yield self

    def unique(self):
        return self

    # submesh

    @cached_property
    def submesh_ancestors(self):
        """Tuple of submesh ancestors."""
        if self.submesh_parent:
            return (self, ) + self.submesh_parent.submesh_ancestors
        else:
            return (self, )

    def submesh_youngest_common_ancestor(self, other):
        """Return the youngest common ancestor of self and other.

        Parameters
        ----------
        other : AbstractMeshTopology
            The other mesh.

        Returns
        -------
        AbstractMeshTopology or None
            Youngest common ancestor or None if not found.

        """
        # self --- ... --- m --- common --- common --- common
        #                          /
        #       other --- ... --- m
        self_ancestors = list(self.submesh_ancestors)
        other_ancestors = list(other.submesh_ancestors)
        c = None
        while self_ancestors and other_ancestors:
            a = self_ancestors.pop()
            b = other_ancestors.pop()
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
        common = self.submesh_youngest_common_ancestor(other)
        if common is None:
            raise ValueError(f"Unable to create composed map between (sub)meshes: {self} and {other} are unrelated")
        maps = []
        integral_type = other_integral_type
        subset_points = other_subset_points
        aa = other.submesh_ancestors
        for a in aa[:aa.index(common)]:
            m, integral_type, subset_points = a.submesh_map_child_parent(integral_type, subset_points)
            maps.append(m)
        bb = self.submesh_ancestors
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
            self.sfBC_orig = sfBC
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

    @cached_property
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
        with temp_internal_comm(self.comm) as icomm:
            nfacets = icomm.allreduce(nfacets, op=MPI.MAX)

        # Note that the geometric dimension of the cell is not set here
        # despite it being a property of a UFL cell. It will default to
        # equal the topological dimension.
        # Firedrake mesh topologies, by convention, which specifically
        # represent a mesh topology (as here) have geometric dimension
        # equal their topological dimension. This is reflected in the
        # corresponding UFL mesh.
        return ufl.Cell(_cells[tdim][nfacets])

    @cached_property
    def _ufl_mesh(self):
        cell = self._ufl_cell
        return ufl.Mesh(finat.ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension))

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

    @cached_property
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
        assert tdim == cell.topological_dimension
        if self.submesh_parent is not None and \
                not (self.submesh_parent.ufl_cell().cellname == "hexahedron" and cell.cellname == "quadrilateral") and \
                len(self.submesh_parent.dm_cell_types) == 1:
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
        elif cell.is_simplex:
            topology = FIAT.ufc_cell(cell).get_topology()
            entity_per_cell = np.zeros(len(topology), dtype=IntType)
            for d, ents in topology.items():
                entity_per_cell[d] = len(ents)

            return dmcommon.closure_ordering(plex, vertex_numbering,
                                             cell_numbering, entity_per_cell)

        elif cell.cellname == "quadrilateral":
            petsctools.cite("Homolya2016")
            petsctools.cite("McRae2016")
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
        elif cell.cellname == "hexahedron":
            # TODO: Should change and use create_cell_closure() for all cell types.
            topology = FIAT.ufc_cell(cell).get_topology()
            closureSize = sum([len(ents) for _, ents in topology.items()])
            return dmcommon.create_cell_closure(plex, cell_numbering, closureSize)
        else:
            raise NotImplementedError("Cell type '%s' not supported." % cell)

    @cached_property
    def entity_orientations(self):
        return dmcommon.entity_orientations(self, self.cell_closure)

    @cached_property
    def local_cell_orientation_dat(self):
        """Local cell orientation dat."""
        return op2.Dat(
            op2.DataSet(self.cell_set, 1),
            self.entity_orientations[:, [-1]],
            gem.uint_type,
            f"{self.name}_local_cell_orientation"
        )

    @PETSc.Log.EventDecorator()
    def _facets(self, kind):
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)

        dm = self.topology_dm
        facets, classes, set_ = getattr(self, "_" + kind + "_facet_numbers_classes_set")
        label = dmcommon.FACE_SETS_LABEL
        if dm.hasLabel(label):
            from mpi4py import MPI
            local_markers = set(dm.getLabelIdIS(label).indices)

            def merge_ids(x, y, datatype):
                return x.union(y)

            op = MPI.Op.Create(merge_ids, commute=True)

            with temp_internal_comm(self.comm) as icomm:
                unique_markers = np.asarray(sorted(icomm.allreduce(local_markers, op=op)),
                                            dtype=IntType)
            op.Free()
        else:
            unique_markers = None

        local_facet_number, facet_cell = \
            dmcommon.facet_numbering(dm, kind, facets,
                                     self._cell_numbering,
                                     self.cell_closure)

        _, pEnd = dm.getChart()
        point2facetnumber = np.full(pEnd, -1, dtype=IntType)
        point2facetnumber[facets] = np.arange(len(facets), dtype=IntType)
        obj = _Facets(self, facets, classes, set_, kind,
                      facet_cell, local_facet_number,
                      unique_markers=unique_markers)
        obj.point2facetnumber = point2facetnumber
        return obj

    @cached_property
    def exterior_facets(self):
        return self._facets("exterior")

    @cached_property
    def interior_facets(self):
        return self._facets("interior")

    def _facet_numbers_classes_set(self, kind):
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)
        # Can not call target.{interior, exterior}_facets.facets
        # if target is a mixed cell mesh (cell_closure etc. can not be defined),
        # so directly call dmcommon.get_facets_by_class.
        _numbers, _classes = dmcommon.get_facets_by_class(self.topology_dm, (kind + "_facets"), self._facet_ordering)
        _classes = as_tuple(_classes, int, 3)
        _set = op2.Set(_classes, f"{kind.capitalize()[:3]}Facets", comm=self.comm)
        return _numbers, _classes, _set

    @cached_property
    def _exterior_facet_numbers_classes_set(self):
        return self._facet_numbers_classes_set("exterior")

    @cached_property
    def _interior_facet_numbers_classes_set(self):
        return self._facet_numbers_classes_set("interior")

    @cached_property
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

    @cached_property
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "Cells", comm=self.comm)

    @staticmethod
    @PETSc.Log.EventDecorator()
    def _set_partitioner(plex, distribute, partitioner_type=None):
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
                (elem.family() == "Q" and elem.degree() == 2 and self.ufl_cell().cellname == "hexahedron"):
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

    @cached_property
    def submesh_child_cell_parent_cell_map(self):
        return self._submesh_make_entity_entity_map(self.cell_set, self.submesh_parent.cell_set, self.cell_closure[:, -1], self.submesh_parent.cell_closure[:, -1], True)

    @cached_property
    def submesh_child_exterior_facet_parent_exterior_facet_map(self):
        _self_numbers, _, _self_set = self._exterior_facet_numbers_classes_set
        _parent_numbers, _, _parent_set = self.submesh_parent._exterior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(_self_set, _parent_set, _self_numbers, _parent_numbers, True)

    @cached_property
    def submesh_child_exterior_facet_parent_interior_facet_map(self):
        _self_numbers, _, _self_set = self._exterior_facet_numbers_classes_set
        _parent_numbers, _, _parent_set = self.submesh_parent._interior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(_self_set, _parent_set, _self_numbers, _parent_numbers, True)

    @cached_property
    def submesh_child_interior_facet_parent_exterior_facet_map(self):
        raise RuntimeError("Should never happen")

    @cached_property
    def submesh_child_interior_facet_parent_interior_facet_map(self):
        _self_numbers, _, _self_set = self._interior_facet_numbers_classes_set
        _parent_numbers, _, _parent_set = self.submesh_parent._interior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(_self_set, _parent_set, _self_numbers, _parent_numbers, True)

    @cached_property
    def submesh_child_cell_parent_interior_facet_map(self):
        _parent_numbers, _, _parent_set = self.submesh_parent._interior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(self.cell_set, _parent_set, self.cell_closure[:, -1], _parent_numbers, True)

    @cached_property
    def submesh_child_cell_parent_exterior_facet_map(self):
        _parent_numbers, _, _parent_set = self.submesh_parent._exterior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(self.cell_set, _parent_set, self.cell_closure[:, -1], _parent_numbers, True)

    @cached_property
    def submesh_parent_cell_child_cell_map(self):
        return self._submesh_make_entity_entity_map(self.submesh_parent.cell_set, self.cell_set, self.submesh_parent.cell_closure[:, -1], self.cell_closure[:, -1], False)

    @cached_property
    def submesh_parent_exterior_facet_child_exterior_facet_map(self):
        _self_numbers, _, _self_set = self._exterior_facet_numbers_classes_set
        _parent_numbers, _, _parent_set = self.submesh_parent._exterior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(_parent_set, _self_set, _parent_numbers, _self_numbers, False)

    @cached_property
    def submesh_parent_exterior_facet_child_interior_facet_map(self):
        raise RuntimeError("Should never happen")

    @cached_property
    def submesh_parent_interior_facet_child_exterior_facet_map(self):
        _self_numbers, _, _self_set = self._exterior_facet_numbers_classes_set
        _parent_numbers, _, _parent_set = self.submesh_parent._interior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(_parent_set, _self_set, _parent_numbers, _self_numbers, False)

    @cached_property
    def submesh_parent_interior_facet_child_interior_facet_map(self):
        _self_numbers, _, _self_set = self._interior_facet_numbers_classes_set
        _parent_numbers, _, _parent_set = self.submesh_parent._interior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(_parent_set, _self_set, _parent_numbers, _self_numbers, False)

    @cached_property
    def submesh_parent_exterior_facet_child_cell_map(self):
        _parent_numbers, _, _parent_set = self.submesh_parent._exterior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(_parent_set, self.cell_set, _parent_numbers, self.cell_closure[:, -1], False)

    @cached_property
    def submesh_parent_interior_facet_child_cell_map(self):
        _parent_numbers, _, _parent_set = self.submesh_parent._interior_facet_numbers_classes_set
        return self._submesh_make_entity_entity_map(_parent_set, self.cell_set, _parent_numbers, self.cell_closure[:, -1], False)

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
        if target_dim == source_dim:
            if source_integral_type == "cell":
                target_integral_type_temp = "cell"
            elif source_integral_type in ["interior_facet", "exterior_facet"]:
                target_integral_type_temp = "facet"
            else:
                raise NotImplementedError("Unsupported combination")
        elif target_dim - 1 == source_dim:
            if source_integral_type == "cell":
                target_integral_type_temp = "facet"
            else:
                raise NotImplementedError("Unsupported combination")
        elif target_dim == source_dim - 1:
            if source_integral_type in ["interior_facet", "exterior_facet"]:
                target_integral_type_temp = "cell"
            else:
                raise NotImplementedError("Unsupported combination")
        else:
            raise NotImplementedError("Unsupported combination")
        if target_integral_type_temp == "cell":
            _cell_numbers = target.cell_closure[:, -1]
            with self.topology_dm.getSubpointIS() as subpoints:
                if reverse:
                    _, target_indices_cell, source_indices_cell = np.intersect1d(subpoints[_cell_numbers], source_subset_points, return_indices=True)
                else:
                    target_subset_points = subpoints[source_subset_points]
                    _, target_indices_cell, source_indices_cell = np.intersect1d(_cell_numbers, target_subset_points, return_indices=True)
            n_cell = len(source_indices_cell)
            with temp_internal_comm(self.comm) as icomm:
                n_cell_max = icomm.allreduce(n_cell, op=MPI.MAX)
            if n_cell_max > 0:
                if n_cell > len(source_subset_points):
                    raise RuntimeError("Found inconsistent data")
            target_integral_type = "cell"
            if reverse:
                target_subset_points = _cell_numbers[target_indices_cell]
        elif target_integral_type_temp == "facet":
            _exterior_facet_numbers, _, _ = target._exterior_facet_numbers_classes_set
            _interior_facet_numbers, _, _ = target._interior_facet_numbers_classes_set
            with self.topology_dm.getSubpointIS() as subpoints:
                if reverse:
                    _, target_indices_int, source_indices_int = np.intersect1d(subpoints[_interior_facet_numbers], source_subset_points, return_indices=True)
                    _, target_indices_ext, source_indices_ext = np.intersect1d(subpoints[_exterior_facet_numbers], source_subset_points, return_indices=True)
                else:
                    target_subset_points = subpoints[source_subset_points]
                    _, target_indices_int, source_indices_int = np.intersect1d(_interior_facet_numbers, target_subset_points, return_indices=True)
                    _, target_indices_ext, source_indices_ext = np.intersect1d(_exterior_facet_numbers, target_subset_points, return_indices=True)
            n_int = len(source_indices_int)
            n_ext = len(source_indices_ext)
            with temp_internal_comm(self.comm) as icomm:
                n_int_max = icomm.allreduce(n_int, op=MPI.MAX)
                n_ext_max = icomm.allreduce(n_ext, op=MPI.MAX)
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
                    target_subset_points = _interior_facet_numbers[target_indices_int]
                elif target_integral_type == "exterior_facet":
                    target_subset_points = _exterior_facet_numbers[target_indices_ext]
        else:
            raise NotImplementedError
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
        common = self.submesh_youngest_common_ancestor(base_mesh)
        if common is None:
            raise NotImplementedError(f"Currently only implemented for (sub)meshes in the same family: got {self} and {base_mesh}")
        elif base_mesh is self:
            raise NotImplementedError("Currenlty can not return identity map")
        else:
            if base_integral_type == "cell":
                base_subset = base_mesh.measure_set(base_integral_type, base_subdomain_id, all_integer_subdomain_ids=base_all_integer_subdomain_ids)
                base_subset_points = base_mesh.cell_closure[:, -1][base_subset.indices]
            elif base_integral_type in ["interior_facet", "exterior_facet"]:
                base_subset = base_mesh.measure_set(base_integral_type, base_subdomain_id, all_integer_subdomain_ids=base_all_integer_subdomain_ids)
                if base_integral_type == "interior_facet":
                    _interior_facet_numbers, _, _ = base_mesh._interior_facet_numbers_classes_set
                    base_subset_points = _interior_facet_numbers[base_subset.indices]
                elif base_integral_type == "exterior_facet":
                    _exterior_facet_numbers, _, _ = base_mesh._exterior_facet_numbers_classes_set
                    base_subset_points = _exterior_facet_numbers[base_subset.indices]
            else:
                raise NotImplementedError(f"Unknown integration type : {base_integral_type}")
            composed_map, integral_type, _ = self.submesh_map_composed(base_mesh, base_integral_type, base_subset_points)
            return composed_map, integral_type

    @cached_property
    def _visible_ranks(self):
        # Get parent mesh rank ownership information.
        visible_ranks = np.empty(self.cell_set.total_size, dtype=IntType)
        visible_ranks[:self.cell_set.size] = self.comm.rank
        visible_ranks[self.cell_set.size:] = -1
        # Halo exchange the visible ranks so that each rank knows which ranks can see each cell.
        dmcommon.exchange_cell_orientations(
            self.topology_dm, self._cell_numbering, visible_ranks
        )
        return visible_ranks


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

        petsctools.cite("McRae2016")
        petsctools.cite("Bercea2016")
        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)

        if isinstance(mesh.topology, VertexOnlyMeshTopology):
            raise NotImplementedError("Extrusion not implemented for VertexOnlyMeshTopology")
        if layers.shape and periodic:
            raise ValueError("Must provide constant layer for periodic extrusion")

        self._base_mesh = mesh
        self.user_comm = mesh.comm
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

    @cached_property
    def _ufl_cell(self):
        return ufl.TensorProductCell(self._base_mesh.ufl_cell(), ufl.interval)

    @cached_property
    def _ufl_mesh(self):
        cell = self._ufl_cell
        return ufl.Mesh(finat.ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension))

    @property
    def dm_cell_types(self):
        """All DM.PolytopeTypes of cells in the mesh."""
        raise NotImplementedError("'dm_cell_types' is not implemented for ExtrudedMeshTopology")

    @cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        return self._base_mesh.cell_closure

    @cached_property
    def entity_orientations(self):
        return self._base_mesh.entity_orientations

    @cached_property
    def local_cell_orientation_dat(self):
        """Local cell orientation dat."""
        return self._base_mesh.local_cell_orientation_dat

    def _facets(self, kind):
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)
        label = f"{kind}_facets"
        base = getattr(self._base_mesh, label)
        layers = self.entity_layers(1, label)
        set_ = op2.ExtrudedSet(base.set, layers=layers)
        return _Facets(self, base.facets, base.classes, set_,
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

    @cached_property
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
        swarm : FiredrakeDMSwarm
            DMSwarm representing particle-in-cell vertices immersed within a
            PETSc DM stored in ``parentmesh``.
        parentmesh : AbstractMeshTopology
            Mesh topology within which the vertex-only mesh topology is immersed.
        name : str
            Name of the mesh topology.
        reorder : bool
            Whether to reorder the mesh entities.
        input_ordering_swarm : FiredrakeDMSwarm
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
        if MPI.Comm.Compare(parentmesh.comm, swarm.dm.comm.tompi4py()) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Parent mesh communicator and swarm communicator are not congruent")
        self._distribution_parameters = {"partition": False,
                                         "partitioner_type": None,
                                         "overlap_type": (DistributedMeshOverlapType.NONE, 0)}
        self.swarm = swarm
        self.input_ordering_swarm = input_ordering_swarm
        self._parent_mesh = parentmesh

        super().__init__(swarm.dm, name, reorder, None, perm_is, distribution_name, permutation_name, parentmesh.comm)
        self._init_particle_ids()

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

    def _init_particle_ids(self):
        from firedrake.functionspace import FunctionSpace
        from firedrake.function import CoordinatelessFunction

        # Attach persistent IDs to VOM points
        P0 = FunctionSpace(self, "DG", 0)
        pid = CoordinatelessFunction(P0, dtype=IntType, name="firedrake_particle_ids")
        n_owned = self.cell_set.size
        offset = self.comm.scan(n_owned) - n_owned
        pid.dat.data_wo[:] = np.arange(offset, offset + n_owned, dtype=IntType)
        self._particle_ids = pid

    @cached_property
    def _ufl_cell(self):
        return ufl.Cell(_cells[0][0])

    @cached_property
    def _ufl_mesh(self):
        cell = self._ufl_cell
        return ufl.Mesh(finat.ufl.VectorElement("DG", cell, 0, dim=cell.topological_dimension))

    def _renumber_entities(self, reorder):
        if reorder:
            swarm = self.swarm
            parent = self._parent_mesh.topology_dm
            cell_id_name = swarm.dm.getCellDMActive().getCellID()
            parent_renum = self._parent_mesh._dm_renumbering.getIndices()
            pStart, _ = parent.getChart()
            parent_renum_inv = np.empty_like(parent_renum)
            parent_renum_inv[parent_renum - pStart] = np.arange(len(parent_renum))
            with (
                swarm.field(cell_id_name) as swarm_parent_cell_nums,
                swarm.field("globalindex") as swarm_global_indices,
            ):
                parent_order = parent_renum_inv[swarm_parent_cell_nums.ravel() - pStart]
                # sort by parent cell order, with ties broken by point global index
                perm = np.lexsort((swarm_global_indices.ravel(), parent_order)).astype(IntType)
            perm_is = PETSc.IS().create(comm=swarm.dm.comm)
            perm_is.setType("general")
            perm_is.setIndices(perm)
            return perm_is
        else:
            return dmcommon.plex_renumbering(self.topology_dm, self._entity_classes, None)

    @property
    def dm_cell_types(self):
        """All DM.PolytopeTypes of cells in the mesh."""
        return (PETSc.DM.PolytopeType.POINT,)

    @cached_property  # TODO: Recalculate if mesh moves
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
        assert tdim == cell.topological_dimension
        assert cell.is_simplex

        import FIAT
        topology = FIAT.ufc_cell(cell).get_topology()
        entity_per_cell = np.zeros(len(topology), dtype=IntType)
        for d, ents in topology.items():
            entity_per_cell[d] = len(ents)

        return dmcommon.closure_ordering(swarm, vertex_numbering,
                                         cell_numbering, entity_per_cell)

    entity_orientations = None

    @property
    def local_cell_orientation_dat(self):
        """Local cell orientation dat."""
        raise NotImplementedError("Not implemented for VertexOnlyMeshTopology")

    def _facets(self, kind):
        """Raises an AttributeError since cells in a
        `VertexOnlyMeshTopology` have no facets.
        """
        if kind not in ["interior", "exterior"]:
            raise ValueError("Unknown facet type '%s'" % kind)
        raise AttributeError("Cells in a VertexOnlyMeshTopology have no facets.")

    @cached_property  # TODO: Recalculate if mesh moves
    def exterior_facets(self):
        return self._facets("exterior")

    @cached_property  # TODO: Recalculate if mesh moves
    def interior_facets(self):
        return self._facets("interior")

    @cached_property
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

    @cached_property  # TODO: Recalculate if mesh moves
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "Cells", comm=self.comm)

    @cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_cell_list(self):
        """Return a list of parent mesh cells numbers in vertex only
        mesh cell order.
        """
        with self.swarm.field("parentcellnum") as parentcellnum_field:
            cell_parent_cell_list = parentcellnum_field.ravel().copy()
        return cell_parent_cell_list[self.cell_closure[:, -1]]

    @cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_cell_map(self):
        """Return the :class:`pyop2.types.map.Map` from vertex only mesh cells to
        parent mesh cells.
        """
        return op2.Map(self.cell_set, self._parent_mesh.cell_set, 1,
                       self.cell_parent_cell_list, "cell_parent_cell")

    @cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_base_cell_list(self):
        """Return a list of parent mesh base cells numbers in vertex only
        mesh cell order.
        """
        if not isinstance(self._parent_mesh, ExtrudedMeshTopology):
            raise AttributeError("Parent mesh is not extruded")
        with self.swarm.field("parentcellbasenum") as parentcellbasenum_field:
            cell_parent_base_cell_list = parentcellbasenum_field.ravel().copy()
        return cell_parent_base_cell_list[self.cell_closure[:, -1]]

    @cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_base_cell_map(self):
        """Return the :class:`pyop2.types.map.Map` from vertex only mesh cells to
        parent mesh base cells.
        """
        if not isinstance(self._parent_mesh, ExtrudedMeshTopology):
            raise AttributeError("Parent mesh is not extruded.")
        return op2.Map(self.cell_set, self._parent_mesh.cell_set, 1,
                       self.cell_parent_base_cell_list, "cell_parent_base_cell")

    @cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_extrusion_height_list(self):
        """Return a list of parent mesh extrusion heights in vertex only
        mesh cell order.
        """
        if not isinstance(self._parent_mesh, ExtrudedMeshTopology):
            raise AttributeError("Parent mesh is not extruded.")
        with self.swarm.field("parentcellextrusionheight") as parentcellextrusionheight_field:
            cell_parent_extrusion_height_list = parentcellextrusionheight_field.ravel().copy()
        return cell_parent_extrusion_height_list[self.cell_closure[:, -1]]

    @cached_property  # TODO: Recalculate if mesh moves
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

    @cached_property  # TODO: Recalculate if mesh moves
    def cell_global_index(self):
        """Return a list of unique cell IDs in vertex only mesh cell order."""
        with self.swarm.field("globalindex") as globalindex_field:
            cell_global_index = globalindex_field.ravel().copy()
        return cell_global_index

    @cached_property  # TODO: Recalculate if mesh moves
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
                name=self.input_ordering_swarm.dm.getName(),
                reorder=False,
            )

    @staticmethod
    def _make_input_ordering_sf(swarm, nroots, ilocal):
        # ilocal = None -> leaves are swarm points [0, 1, 2, ...).
        # ilocal can also be Firedrake cell numbers.
        sf = PETSc.SF().create(comm=swarm.dm.comm)
        with (
            swarm.field("inputrank") as input_ranks,
            swarm.field("inputindex") as input_indices,
        ):
            input_ranks = input_ranks.ravel()
            input_indices = input_indices.ravel()
            nleaves = len(input_ranks)
            if ilocal is not None and nleaves != len(ilocal):
                raise RuntimeError(f"Mismatching leaves: nleaves {nleaves} != len(ilocal) {len(ilocal)}")
            input_ranks_and_idxs = np.empty(2 * nleaves, dtype=IntType)
            input_ranks_and_idxs[0::2] = input_ranks
            input_ranks_and_idxs[1::2] = input_indices
        sf.setGraph(nroots, ilocal, input_ranks_and_idxs)
        return sf

    @cached_property  # TODO: Recalculate if mesh moves
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
        return VertexOnlyMeshTopology._make_input_ordering_sf(self.swarm, nroots, ilocal)

    @cached_property  # TODO: Recalculate if mesh moves
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


@dataclasses.dataclass(frozen=True)
class _MultiCellTypeDummyCoordinates:
    """Placeholder object for the coordinates of a mesh with >1 cell types."""
    topology: AbstractMeshTopology
    _ufl_element: finat.ufl.FiniteElementBase

    def ufl_element(self) -> finat.ufl.FiniteElementBase:
        return self._ufl_element

    @property
    def comm(self) -> MPI.Comm:
        return self.topology.comm


class MeshGeometry(ufl.Mesh, MeshGeometryMixin):
    """A representation of mesh topology and geometry."""

    @MeshGeometryMixin._ad_annotate_init
    def __init__(self, coordinates):
        """Initialise a mesh geometry from coordinates.

        Parameters
        ----------
        coordinates : CoordinatelessFunction
            The `CoordinatelessFunction` containing the coordinates.

        """
        import firedrake.functionspaceimpl as functionspaceimpl
        import firedrake.function as function

        utils._init()

        element = coordinates.ufl_element()
        uid = utils._new_uid(coordinates.comm)
        super().__init__(element, ufl_id=uid)

        if isinstance(coordinates, _MultiCellTypeDummyCoordinates):
            topology = coordinates.topology
        else:
            topology = coordinates.function_space().mesh()

        # this is codegen information so we attach it to the MeshGeometry rather than its cargo
        self.extruded = isinstance(topology, ExtrudedMeshTopology)
        self.variable_layers = self.extruded and topology.variable_layers
        self._base_mesh = None  # this is set by extruded meshes in a later step
        # these are set by firedrake.adapt.refine_marked_elements
        self.adaptive_parent = None
        self.adaptive_cell_maps = None

        self.topology = topology
        self.geometric_shared_data_cache = defaultdict(dict)

        # A lot of the infrastructure of MeshGeometry does not work for meshes
        # with multiple cell types
        if isinstance(coordinates, _MultiCellTypeDummyCoordinates):
            return

        # submesh
        self.submesh_parent = None

        # Cache mesh object on the coordinateless coordinates function
        coordinates._as_mesh_geometry = weakref.ref(self)

        # Save the coordinates as a 'CoordinatelessFunction' and as a 'Function'
        self._coordinates = coordinates
        V = functionspaceimpl.WithGeometry(coordinates.function_space(), self)
        self._coordinates_function = function.Function(V, val=coordinates)

    def _ufl_signature_data_(self, *args, **kwargs):
        return (type(self), self.extruded, self.variable_layers,
                super()._ufl_signature_data_(*args, **kwargs))

    @property
    def topological(self):
        """Alias of topology.

        This is to ensure consistent naming for some multigrid codes."""
        return self.topology

    @property
    def coordinates(self) -> "Function":
        """The coordinates of the mesh."""
        return self._coordinates_function

    @coordinates.setter
    def coordinates(self, value):
        if value is self.coordinates:
            return
        message = """Cannot re-assign the coordinates.

You are free to change the coordinate values, but if you need a
different coordinate function space, use Mesh(f) to create a new mesh
object whose coordinates are f's values.  (This will not copy the
values from f.)"""

        raise AttributeError(message)

    @cached_property
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
        After changing tolerance any requests for :attr:`rtree` or
        :attr:`distributed_rtree` will cause the tree to be rebuilt with the
        new tolerance which may take some time.
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        if not isinstance(value, numbers.Number):
            raise TypeError("tolerance must be a number")
        if value != self._tolerance:
            self._tolerance = value

    def clear_rtree(self):
        """Reset the :attr:`rtree` on this mesh geometry.

        Use this if you move the mesh (for example by reassigning to
        the coordinate field)."""
        warnings.warn(
            "The ``clear_rtree`` method is deprecated and will be removed in a future release. "
            "There is no need to manually clear the rtree after changing the mesh coordinates;"
            "the rtree will be automatically rebuilt.", FutureWarning
        )
        # `cached_property_until` stores the cached rtree in self._rtree_cache
        # setting it to None will force the rtree to be rebuilt on next access.
        self._rtree_cache = None

    @cached_property_until(lambda self: self.coordinates.dat.dat_version)
    @PETSc.Log.EventDecorator()
    def bounding_box_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates bounding boxes for the mesh rtree.

        Returns
        -------
        Tuple of arrays of shape (num_cells, gdim) containing
        the minimum and maximum coordinates of each cell's bounding box.

        Notes
        -----
        If we have a higher-order (bendy) mesh we project the mesh coordinates into
        a Bernstein finite element space. Functions on a Bernstein element are
        Bezier curves and are completely contained in the convex hull of the mesh nodes.
        Hence the bounding box will contain the entire element.
        """
        from firedrake import function, functionspace
        from firedrake.parloops import par_loop, READ, MIN, MAX

        coord_element = self.ufl_coordinate_element()
        coord_degree = coord_element.degree()
        if np.all(np.asarray(coord_degree) == 1):
            mesh = self
        elif coord_element.family() == "Bernstein":
            # Already have Bernstein coordinates, no need to project
            mesh = self
        else:
            # For bendy meshes we project the coordinate function onto Bernstein
            if self.extruded:
                bernstein_fs = functionspace.VectorFunctionSpace(
                    self, "Bernstein", coord_degree[0], vfamily="Bernstein", vdegree=coord_degree[1]
                )
            else:
                bernstein_fs = functionspace.VectorFunctionSpace(self, "Bernstein", coord_degree)
            f = function.Function(bernstein_fs)
            f.interpolate(self.coordinates)
            mesh = Mesh(f)

        if utils.complex_mode:
            if not np.allclose(mesh.coordinates.dat.data_ro.imag, 0):
                raise ValueError("Coordinate field has non-zero imaginary part")
            coords = function.Function(mesh.coordinates.function_space(),
                                       val=mesh.coordinates.dat.data_ro_with_halos.real.copy(),
                                       dtype=RealType)
        else:
            coords = mesh.coordinates

        cell_node_list = mesh.coordinates.function_space().cell_node_list
        if not mesh.extruded:
            all_coords = coords.dat.data_ro_with_halos[cell_node_list]
            return np.min(all_coords, axis=1), np.max(all_coords, axis=1)

        # Extruded case: calculate the bounding boxes for all cells by running a kernel
        V = functionspace.VectorFunctionSpace(mesh, "DG", 0, dim=self.geometric_dimension)
        coords_min = function.Function(V, dtype=RealType)
        coords_max = function.Function(V, dtype=RealType)

        coords_min.dat.data.fill(np.inf)
        coords_max.dat.data.fill(-np.inf)

        _, nodes_per_cell = cell_node_list.shape

        domain = f"{{[d, i]: 0 <= d < {self.geometric_dimension} and 0 <= i < {nodes_per_cell}}}"
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
        coords_min = mesh._order_data_by_cell_index(column_list, coords_min.dat.data_ro_with_halos)
        coords_max = mesh._order_data_by_cell_index(column_list, coords_max.dat.data_ro_with_halos)
        return coords_min, coords_max

    @cached_property_until(lambda self: (self.coordinates.dat.dat_version, self.tolerance))
    @PETSc.Log.EventDecorator()
    def rtree(self):
        """Builds an rtree from bounding box coordinates, expanding
        the bounding boxes by the mesh tolerance.

        Returns
        -------
        :class:`~.rtree.RTree`

        Notes
        -----
        If this mesh has a :attr:`tolerance` property, which
        should be a float, this tolerance is added to the extrema of the
        rtree so that points just outside the mesh, within tolerance,
        can be found.

        """
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
        coords_min, coords_max = self.bounding_box_coords
        if self.geometric_dimension == 1:
            coords_min = coords_min.reshape(-1, 1)
            coords_max = coords_max.reshape(-1, 1)

        tolerance = self.tolerance if hasattr(self, "tolerance") else 0.0
        if self.topological_dimension < self.geometric_dimension:
            # Immersed manifold case: Change min and max to refer to an n-hypercube,
            # where n is the geometric dimension of the mesh, centred on the midpoint of the
            # bounding box. Its side length is the L1 diameter of the bounding box.
            # This aids point evaluation where points may be just off the mesh but should be evaluated.
            # We also push max and min out so we can find points on the boundary
            # within the mesh tolerance.
            coords_mid = (coords_max + coords_min) / 2
            d = np.max(coords_max - coords_min, axis=1)[:, None]
            coords_min = coords_mid - (tolerance + 0.5) * d
            coords_max = coords_mid + (tolerance + 0.5) * d
        else:
            coords_extent = coords_max - coords_min
            coords_min = coords_min - tolerance * coords_extent
            coords_max = coords_max + tolerance * coords_extent
        with PETSc.Log.Event("rtree_build"):
            self._rtree = rtree.build_from_aabb(coords_min, coords_max)
        return self._rtree

    @PETSc.Log.EventDecorator()
    def bounding_boxes_total_volume(self, bounding_boxes: np.ndarray):
        side_lengths = bounding_boxes[:, 1, :] - bounding_boxes[:, 0, :]
        return np.prod(side_lengths, axis=1).sum()

    @cached_property_until(lambda self: (self.coordinates.dat.dat_version, self.tolerance))
    @PETSc.Log.EventDecorator()
    def _box_ratio_heuristic(self):
        """Return partition bounding boxes at some 'optimal' Rtree level.

        Descends the local Rtree top-down breadth-first, stopping when the total
        bounding box volume stops decreasing (ratio of next level to current
        level >= 1).

        Returns
        -------
        numpy.ndarray
            Array of shape `(n_boxes, 2, gdim)` containing bounding boxes
        """
        tree_depth = rtree.tree_depth(self.rtree)
        if tree_depth == 0:
            # This indicates an empty tree, which can happen if the mesh has no cells on this rank.
            return np.empty((0, 2, self.geometric_dimension), dtype=utils.RealType)
        gdim = self.geometric_dimension
        prev_bboxes = rtree.bounding_boxes_at_level(self.rtree, 0, gdim)
        prev_vol = self.bounding_boxes_total_volume(prev_bboxes)

        for level in range(1, tree_depth):
            next_bboxes = rtree.bounding_boxes_at_level(self.rtree, level, gdim)
            next_vol = self.bounding_boxes_total_volume(next_bboxes)

            if next_vol >= prev_vol:
                break

            prev_bboxes = next_bboxes
            prev_vol = next_vol
        return prev_bboxes

    @cached_property_until(lambda self: (self.coordinates.dat.dat_version, self.tolerance))
    @PETSc.Log.EventDecorator()
    def distributed_rtree(self):
        """Build a global Rtree from all ranks' partition bounding boxes.

        Each rank contributes bounding boxes chosen by `box_ratio_heuristic`.
        The boxes are gathered from all ranks and a single Rtree is built
        on every rank. The owning MPI rank is stored as the id of each leaf,
        so querying the tree with a point will return a list of candidate ranks
        who may have a cell containing that point.

        Returns
        -------
        :class:`~firedrake.cython.rtree.RTree`
            A global Rtree whose leaf ids are MPI rank numbers.
        """
        gdim = self.geometric_dimension
        comm = self.comm

        local_bboxes = self._box_ratio_heuristic  # (n_local, 2, gdim)
        n_local = local_bboxes.shape[0]

        # Allgather per-rank box counts
        counts = np.empty(comm.size, dtype=IntType)
        comm.Allgather(np.array([n_local], dtype=IntType), counts)
        n_total = int(counts.sum())

        # Allgatherv the bbox data
        all_bboxes_flat = np.empty(n_total * 2 * gdim, dtype=RealType)
        comm.Allgatherv(sendbuf=local_bboxes.ravel(), recvbuf=(all_bboxes_flat, counts * 2 * gdim))

        # Reshape to (n_total, 2, gdim) and split into lo/hi corner arrays.
        all_bboxes = all_bboxes_flat.reshape(n_total, 2, gdim)
        regions_lo = np.ascontiguousarray(all_bboxes[:, 0, :])  # (n_total, gdim)
        regions_hi = np.ascontiguousarray(all_bboxes[:, 1, :])  # (n_total, gdim)

        # Set the owning rank as the leaf id so queries return rank numbers.
        ids = np.repeat(np.arange(comm.size, dtype=np.int64), counts)

        return rtree.build_from_aabb(regions_lo, regions_hi, ids)

    @PETSc.Log.EventDecorator()
    def locate_cell(self, x, tolerance=None, cell_ignore=None):
        """Locate cell containing a given point.

        :arg x: point coordinates
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the rtree to be rebuilt which
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
            this from default will cause the rtree to be rebuilt which
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
            this from default will cause the rtree to be rebuilt which
            can take some time.
        :kwarg cell_ignore: Cell number to ignore in the search.
        :returns: tuple either
            (cell number, reference coordinates) of type (int, numpy array),
            or, when point is not in the domain, (None, None).
        """
        x = np.asarray(x)
        if x.size != self.geometric_dimension:
            raise ValueError("Point must have the same geometric dimension as the mesh")
        x = x.reshape((1, self.geometric_dimension))
        cells, ref_coords, _ = self.locate_cells_ref_coords_and_dists(x, tolerance=tolerance, cells_ignore=[[cell_ignore]])
        if cells[0] == -1:
            return None, None
        return cells[0], ref_coords[0]

    @PETSc.Log.EventDecorator()
    def locate_cells_ref_coords_and_dists(self, xs, tolerance=None, cells_ignore=None):
        """Locate cell containing a given point and the reference
        coordinates of the point within the cell.

        :arg xs: 1 or more point coordinates of shape (npoints, gdim)
        :kwarg tolerance: Tolerance for checking if a point is in a cell.
            Default is this mesh's :attr:`tolerance` property. Changing
            this from default will cause the rtree to be rebuilt which
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
        cells, ref_coords, ref_cell_dists_l1, _ = self._locate_cells_ref_coords_dists_and_owners(
            xs, tolerance=tolerance, cells_ignore=cells_ignore
        )
        return cells, ref_coords, ref_cell_dists_l1

    def _locate_cells_ref_coords_dists_and_owners(
        self,
        xs: np.ndarray,
        tolerance: float | None = None,
        cells_ignore: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Locate cells and their owner ranks for an array of points.

        Parameters
        ----------
        xs : numpy.ndarray
            Point coordinates with shape ``(npoints, gdim)``.
        tolerance : float, optional
            Reference-cell tolerance used to accept nearby cells.
        cells_ignore : numpy.ndarray, optional
            Cell numbers to exclude for each point.

        Returns
        -------
        cells : numpy.ndarray
            Located Firedrake cell numbers, or ``-1`` for missing points.
        reference_coordinates : numpy.ndarray
            Reference coordinates in the located cells.
        distances : numpy.ndarray
            L1 distances from the reference cells.
        owner_ranks : numpy.ndarray
            Owner rank of each located cell, or ``-1`` for missing points.
        """
        if self.variable_layers:
            raise NotImplementedError("Cell location not implemented for variable layers")
        if tolerance is None:
            tolerance = self.tolerance
        else:
            self.tolerance = tolerance
        # `xs` are the physical coordinates we query the rtree with.
        # libspatialindex requires these to be of type double
        xs = np.asarray(xs).real.astype(np.float64, order="C")
        if xs.shape[1] != self.geometric_dimension:
            raise ValueError("Point coordinate dimension does not match mesh geometric dimension")
        Xs = np.empty_like(xs, dtype=RealType)
        npoints = len(xs)
        if cells_ignore is None or cells_ignore[0][0] is None:
            cells_ignore = np.full((npoints, 1), -1, dtype=IntType, order="C")
        else:
            cells_ignore = np.asarray(cells_ignore, dtype=IntType, order="C")
        if cells_ignore.shape[0] != npoints:
            raise ValueError("Number of cells to ignore does not match number of points")
        assert cells_ignore.shape == (npoints, cells_ignore.shape[1])
        ref_cell_dists_l1 = np.empty(npoints, dtype=RealType)
        cells = np.empty(npoints, dtype=IntType)
        owner_ranks = np.empty(npoints, dtype=IntType)
        cell_owner_ranks = np.ascontiguousarray(self._visible_ranks, dtype=IntType)
        assert xs.size == npoints * self.geometric_dimension
        run_c = self._c_locator(tolerance=tolerance)
        cells_data = cells.ctypes.data_as(ctypes.POINTER(as_ctypes(IntType)))
        owner_ranks_data = owner_ranks.ctypes.data_as(ctypes.POINTER(as_ctypes(IntType)))
        ref_cells_dists = ref_cell_dists_l1.ctypes.data_as(ctypes.POINTER(as_ctypes(RealType)))
        xs_data = xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        Xs_data = Xs.ctypes.data_as(ctypes.POINTER(as_ctypes(RealType)))
        with PETSc.Log.Event("c_locator_run"):
            err = run_c(
                self.coordinates._ctypes,
                xs_data,
                Xs_data,
                ref_cells_dists,
                cells_data,
                owner_ranks_data,
                npoints,
                cells_ignore.shape[1],
                cells_ignore,
                cell_owner_ranks,
            )
        if err != 0:
            raise RuntimeError(f"C locator failed with error code {err}")
        return cells, Xs, ref_cell_dists_l1, owner_ranks

    @PETSc.Log.EventDecorator()
    def _c_locator(self, tolerance=None):
        """Generates C code to compute containing cells and reference coordinates for a set of points.

        First, the rtree is queried to find candidate cells for each point. Then, for each point, we locate
        the single owning cell from the candidates. This owning cell is the one which is closest to the point
        in the L1 norm in reference coordinates, breaking equal-distance ties in favour of the highest owner
        rank.
        """
        from pyop2 import compilation
        import firedrake.function as function
        import firedrake.pointquery_utils as pq_utils

        cache = self.__dict__.setdefault("_c_locator_cache", {})
        try:
            return cache[tolerance]
        except KeyError:
            src = pq_utils.src_locate_cell(self, tolerance=tolerance)
            src += dedent(f"""
                PetscErrorCode locator(struct Function *f,
                                       double *x,
                                       {RealType_c} *X,
                                       {RealType_c} *ref_cell_dists_l1,
                                       {IntType_c} *cells,
                                       {IntType_c} *owners,
                                       size_t npoints,
                                       size_t ncells_ignore,
                                       {IntType_c} *cells_ignore,
                                       const {IntType_c} *cell_owner_ranks)
                {{
                    PetscErrorCode locate_err = PETSC_SUCCESS;
                    int64_t *candidate_ids = NULL;
                    size_t *candidate_offsets = NULL;

                    RTreeError rtree_err = rtree_locate_all_at_points(
                        (const struct RTreeH *)f->rtree, x, npoints, &candidate_ids, &candidate_offsets);
                    if (rtree_err != Success) {{
                        fputs("ERROR: rtree_locate_all_at_points failed.\\n", stderr);
                        return PETSC_ERR_LIB;
                    }}

                    size_t j = 0;  /* index into x and X */
                    for(size_t i=0; i<npoints; i++) {{
                        /* i is the index into cells and ref_cell_dists_l1 */

                        /* The type definitions and arguments used here are defined as
                        statics in pointquery_utils.py */
                        struct ReferenceCoords temp_reference_coords, found_reference_coords;

                        size_t nids_i = candidate_offsets[i + 1] - candidate_offsets[i];
                        int64_t *ids_i = candidate_ids + candidate_offsets[i];

                        /* to_reference_coords and to_reference_coords_xtr are defined in
                        pointquery_utils.py. If they contain python calls, this loop will
                        not run at c-loop speed. */
                        /* cells_ignore has shape (npoints, ncells_ignore) - find the ith row */
                        {IntType_c} *cells_ignore_i = cells_ignore + i*ncells_ignore;

                        locate_err = locate_cell_from_candidates(
                            f, &x[j], &to_reference_coords, &to_reference_coords_xtr,
                            &temp_reference_coords, &found_reference_coords,
                            &ref_cell_dists_l1[i], nids_i, ids_i,
                            ncells_ignore, cells_ignore_i, cell_owner_ranks,
                            &cells[i], &owners[i]);

                        if (locate_err != PETSC_SUCCESS) {{
                            break;
                        }}

                        for (int k = 0; k < {self.geometric_dimension}; k++) {{
                            X[j] = found_reference_coords.X[k];
                            j++;
                        }}
                    }}
                    rtree_free_ids(candidate_ids, candidate_offsets[npoints]);
                    rtree_free_offsets(candidate_offsets, npoints + 1);
                    return locate_err;
                }}
            """)

            dll = compilation.load(
                src, "c",
                cppargs=[
                    f"-I{os.path.dirname(__file__)}",
                    f"-I{sys.prefix}/include",
                    f"-I{firedrake_rtree.get_include()}",
                    *petsctools.get_petsc_dirs(prefix="-I", subdir="include"),
                ],
                ldargs=[
                    f"-L{sys.prefix}/lib",
                    firedrake_rtree.get_lib_filename(),
                    f"-Wl,-rpath,{sys.prefix}/lib",
                    f"-Wl,-rpath,{firedrake_rtree.get_lib()}"
                ],
                comm=self.comm
            )
            locator = getattr(dll, "locator")
            locator.argtypes = [ctypes.POINTER(function._CFunction),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(as_ctypes(RealType)),
                                ctypes.POINTER(as_ctypes(RealType)),
                                ctypes.POINTER(as_ctypes(IntType)),
                                ctypes.POINTER(as_ctypes(IntType)),
                                ctypes.c_size_t,
                                ctypes.c_size_t,
                                np.ctypeslib.ndpointer(as_ctypes(IntType), flags="C_CONTIGUOUS"),
                                np.ctypeslib.ndpointer(as_ctypes(IntType), flags="C_CONTIGUOUS")]
            locator.restype = ctypes.c_int
            return cache.setdefault(tolerance, locator)

    @cached_property  # TODO: Recalculate if mesh moves. Extend this for regular meshes.
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

        if (self.ufl_cell().cellname, self.geometric_dimension) not in _supported_embedded_cell_types_and_gdims:
            raise NotImplementedError('Only implemented for intervals embedded in 2d and triangles and quadrilaterals embedded in 3d')

        if hasattr(self, '_cell_orientations'):
            raise CellOrientationsRuntimeError("init_cell_orientations already called, did you mean to do so again?")

        if not isinstance(expr, ufl.classes.Expr):
            raise TypeError("UFL expression expected!")

        if expr.ufl_shape != (self.geometric_dimension, ):
            raise ValueError(f"Mismatching shapes: expr.ufl_shape ({expr.ufl_shape}) != (self.geometric_dimension, ) (({self.geometric_dimension}, ))")

        fs = functionspace.FunctionSpace(self, 'DG', 0)
        x = ufl.SpatialCoordinate(self)
        f = function.Function(fs)

        if self.topological_dimension == 1:
            normal = ufl.as_vector((-ReferenceGrad(x)[1, 0], ReferenceGrad(x)[0, 0]))
        else:  # self.topological_dimension == 2
            normal = ufl.cross(ReferenceGrad(x)[:, 0], ReferenceGrad(x)[:, 1])

        f.interpolate(ufl.dot(expr, normal))

        cell_orientations = function.Function(fs, name="cell_orientations", dtype=np.int32)
        cell_orientations.dat.data[:] = (f.dat.data_ro < 0)
        self._cell_orientations = cell_orientations.topological

    def __getattr__(self, name):
        return getattr(self.topology, name)

    def __dir__(self):
        current = super(MeshGeometry, self).__dir__()
        return list(OrderedDict.fromkeys(dir(self.topology) + current))

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

    def __iter__(self):
        yield self

    def unique(self):
        return self

    @PETSc.Log.EventDecorator()
    def refine_marked_elements(self, mark):
        """Adaptively refine a mesh using a DG0 marking function.

        Parameters
        ----------
        mark
            A DG0 `~firedrake.function.Function` on this mesh: cells
            with a positive value ``n`` are refined ``n`` times.

        Returns
        -------
        MeshGeometry
            The adaptively refined mesh, recording this mesh as its
            ``adaptive_parent`` and the cell maps relative to it as its
            ``adaptive_cell_maps``, ready to be passed to
            :meth:`~firedrake.mg.mesh.HierarchyBase.add_mesh`.
        """
        from firedrake.adapt import refine_marked_elements
        return refine_marked_elements(self, mark)

    @PETSc.Log.EventDecorator()
    def curve_field(self, order, permutation_tol=None, cg_field=None):
        '''Return a function containing the curved coordinates of the mesh.

        This method requires that the mesh has been constructed from a
        netgen mesh.

        :arg order: the order of the curved mesh.
        :arg permutation_tol: ignored.
        :arg cg_field: return a CG function field representing the mesh, as opposed to a DG field.
            Defaults to the continuity of the coordinates of the original mesh.

        '''
        utils.check_netgen_installed()
        from firedrake.netgen import find_permutation, netgen_distribute
        from firedrake.functionspace import FunctionSpace
        from firedrake.function import Function

        if not hasattr(self, "netgen_mesh"):
            raise ValueError("Cannot curve a mesh that has not been generated by netgen.")
        if permutation_tol is not None:
            warnings.warn(
                "permutation_tol is no longer required to obtain the curved coordinates. "
                "This kwarg will be removed in a future release.",
                FutureWarning,
            )

        if cg_field is None:
            cg_field = not self.coordinates.function_space().finat_element.is_dg()

        # Check if the mesh is a surface mesh or two dimensional mesh
        if self.topological_dimension == 2:
            ng_element = self.netgen_mesh.Elements2D()
        else:
            ng_element = self.netgen_mesh.Elements3D()
        ng_dimension = len(ng_element)

        # Construct the coordinates as a Firedrake function
        coords_space = self.coordinates.function_space().reconstruct(degree=order)
        broken_space = coords_space.broken_space()
        if not cg_field:
            coords_space = broken_space
        new_coordinates = Function(coords_space).interpolate(self.coordinates)

        # Compute reference points using fiat
        fiat_element = new_coordinates.function_space().finat_element.fiat_equivalent
        nodes = fiat_element.dual_basis()
        ref_pts = []
        entity_ids = fiat_element.entity_dofs()
        for dim in sorted(entity_ids):
            for entity in sorted(entity_ids[dim]):
                for i in entity_ids[dim][entity]:
                    # Assert singleton point for each node.
                    pt, = nodes[i].get_point_dict().keys()
                    ref_pts.append(pt)
        reference_points = np.array(ref_pts)

        # Construct numpy arrays for physical domain data
        physical_points = np.zeros(
            (ng_dimension, reference_points.shape[0], self.geometric_dimension)
        )
        curved_points = np.zeros(
            (ng_dimension, reference_points.shape[0], self.geometric_dimension)
        )
        self.netgen_mesh.Curve(1)
        self.netgen_mesh.CalcElementMapping(reference_points, physical_points)
        self.netgen_mesh.Curve(order)
        self.netgen_mesh.CalcElementMapping(reference_points, curved_points)
        curved = ng_element.NumPy()["curved"]

        # Distribute curved cell data
        cell_node_map = new_coordinates.cell_node_map()
        num_cells = cell_node_map.values.shape[0]
        DG0 = FunctionSpace(self, "DG", 0)
        own_curved = netgen_distribute(DG0, curved)
        own_curved = np.flatnonzero(own_curved[:num_cells])

        # Distribute coordinate data
        own_curved_points = netgen_distribute(broken_space, curved_points)[own_curved]
        own_physical_points = netgen_distribute(broken_space, physical_points)[own_curved]

        # Get broken indices
        cstart, cend = self.topology_dm.getHeightStratum(0)
        cellNum = np.array(list(map(self._cell_numbering.getOffset, range(cstart, cend))))
        broken_indices = cell_node_map.values[cellNum[own_curved]]

        # Find the correct coordinate permutation for each cell
        permutation = find_permutation(
            own_physical_points,
            new_coordinates.dat.data_ro_with_halos[broken_indices].real,
        )
        self.comm.Barrier()
        # Apply the permutation to each cell in turn
        for i in range(own_curved_points.shape[0]):
            own_curved_points[i] = own_curved_points[i, permutation[i]]

        # Assign the curved coordinates to the dat
        new_coordinates.dat.data_wo_with_halos[broken_indices] = own_curved_points
        return new_coordinates


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
    orig_mesh = V.mesh()
    assert orig_mesh.ufl_cell().topological_dimension <= V.value_size

    mesh = MeshGeometry(coordinates)
    mesh.name = name
    # Mark mesh as being made from coordinates
    mesh._made_from_coordinates = True
    mesh._tolerance = tolerance
    mesh._did_reordering = orig_mesh._did_reordering
    mesh._distribution_parameters = orig_mesh._distribution_parameters
    return mesh


def _fully_localize_coordinates(dm):
    """Expand sparsely localized coordinates to cover all cells.

    For file-based periodic meshes (e.g. Gmsh), PETSc only creates
    cell-local (DG) coordinates for cells touching the periodic
    boundary. This fills in the remaining cells using CG vertex
    coordinates via ``vecGetClosure``.
    """
    gdim = dm.getCoordinateDim()
    cStart, cEnd = dm.getHeightStratum(0)
    cell_sec = dm.getCellCoordinateSection()
    coord_sec = dm.getCoordinateSection()
    coord_vec = dm.getCoordinatesLocal()
    old_cell_vec = dm.getCellCoordinatesLocal()

    # Find dofs_per_cell from an existing cell entry
    dofs_per_cell = None
    for c in range(cStart, cEnd):
        dof = cell_sec.getDof(c)
        if dof > 0:
            dofs_per_cell = dof
            break
    if dofs_per_cell is None:
        return

    # Build new section and vector covering all cells
    new_sec = PETSc.Section().create(comm=PETSc.COMM_SELF)
    new_sec.setNumFields(1)
    new_sec.setFieldComponents(0, gdim)
    new_sec.setChart(cStart, cEnd)
    for c in range(cStart, cEnd):
        new_sec.setDof(c, dofs_per_cell)
        new_sec.setFieldDof(c, 0, dofs_per_cell)
    new_sec.setUp()

    new_vec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
    new_vec.setSizes((new_sec.getStorageSize(), PETSc.DETERMINE), gdim)
    new_vec.setType(coord_vec.getType())

    arr = new_vec.array
    old_arr = old_cell_vec.array
    for c in range(cStart, cEnd):
        off = new_sec.getOffset(c)
        old_dof = cell_sec.getDof(c)
        if old_dof > 0:
            old_off = cell_sec.getOffset(c)
            arr[off:off + dofs_per_cell] = old_arr[old_off:old_off + old_dof]
        else:
            arr[off:off + dofs_per_cell] = dm.vecGetClosure(
                coord_sec, coord_vec, c)[:dofs_per_cell]

    coord_dm = dm.getCoordinateDM()
    dm.setCellCoordinateDM(coord_dm.clone())
    dm.setCellCoordinateSection(gdim, new_sec)
    dm.setCellCoordinatesLocal(new_vec)


def make_mesh_from_mesh_topology(topology, name, tolerance=0.5):
    """Make mesh from topology.

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
    dm = topology.topology_dm
    geometric_dim = dm.getCoordinateDim()
    # For periodic meshes loaded from file (e.g. Gmsh), PETSc creates
    # cell-local (DG) coordinates only for cells touching the periodic
    # boundary (sparse localization). Firedrake needs every cell to
    # have an entry, so we expand to full localization.
    if dm.getCoordinatesLocalized():
        _fully_localize_coordinates(dm)
    if not dm.getCoordinatesLocalized():
        element = finat.ufl.VectorElement("Lagrange", cell, 1, dim=geometric_dim)
    else:
        element = finat.ufl.VectorElement("DQ" if cell in [ufl.quadrilateral, ufl.hexahedron] else "DG", cell, 1, dim=geometric_dim, variant="equispaced")

    coords = coordinates_from_topology(topology, element)
    mesh = MeshGeometry(coords)
    mesh.name = name
    mesh._tolerance = tolerance
    return mesh


@PETSc.Log.EventDecorator()
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
    coords = coordinates_from_topology(topology, element)
    vmesh = MeshGeometry(coords)
    vmesh.name = name
    vmesh._tolerance = tolerance

    # Save vertex reference coordinate (within reference cell) in function
    parent_tdim = topology._parent_mesh.ufl_cell().topological_dimension
    if parent_tdim > 0:
        reference_coordinates_fs = functionspace.VectorFunctionSpace(topology, "DG", 0, dim=parent_tdim)
        reference_coordinates_data = dmcommon.reordered_coords(topology.topology_dm, reference_coordinates_fs.dm.getDefaultSection(),
                                                               (topology.num_vertices(), parent_tdim),
                                                               reference_coord=True)
        reference_coordinates = function.CoordinatelessFunction(reference_coordinates_fs,
                                                                val=reference_coordinates_data,
                                                                name=_generate_default_mesh_reference_coordinates_name(name))
        refCoordV = functionspaceimpl.WithGeometry(reference_coordinates_fs, vmesh)
        vmesh.reference_coordinates = function.Function(refCoordV, val=reference_coordinates)
    else:
        # We can't do this in 0D so leave it undefined.
        vmesh.reference_coordinates = None
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
    if isinstance(meshfile, Path):
        meshfile = str(meshfile)
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

    from_netgen = netgen and isinstance(meshfile, netgen.libngpy._meshing.Mesh)

    # We don't need to worry about using a user comm in these cases as
    # they all immediately call a petsc4py which in turn uses a PETSc
    # internal comm
    geometric_dim = kwargs.get("dim", None)
    if isinstance(meshfile, PETSc.DMPlex):
        plex = meshfile
        if MPI.Comm.Compare(user_comm, plex.comm.tompi4py()) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Communicator used to create `plex` must be at least congruent to the communicator used to create the mesh")
    elif from_netgen:
        from firedrake.netgen import FiredrakeMesh

        petsctools.cite("Betteridge2024")
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
    mesh = make_mesh_from_mesh_topology(topology, name)

    if from_netgen:
        mesh.netgen_mesh = netgen_firedrake_mesh.meshMap.ngMesh
        mesh.netgen_flags = netgen_flags

        # Curve the mesh, if requested
        degree = netgen_flags.get("degree", 1)
        if degree != 1:
            permutation_tol = netgen_flags.get("permutation_tol", None)
            cg = netgen_flags.get("cg", None)
            coordinates = mesh.curve_field(
                order=degree,
                permutation_tol=permutation_tol,
                cg_field=cg,
            )
            # Do not redistribute the mesh
            reorder_noop = None
            temp = Mesh(coordinates,
                        reorder=reorder_noop,
                        perm_is=mesh._dm_renumbering,
                        distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
                        comm=mesh.comm)
            temp.netgen_mesh = mesh.netgen_mesh
            temp.netgen_flags = mesh.netgen_flags
            temp.sfBC = mesh.sfBC
            temp.sfBC_orig = mesh.sfBC_orig
            temp._distribution_parameters = mesh._distribution_parameters
            temp._did_reordering = mesh._did_reordering
            mesh = temp

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
                         of layers (deprecated).  In this case, each entry is a pair
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
        warnings.warn(
            "Variable layer extrusion is deprecated and will be removed "
            "in the 2026.10.0 release. If possible we recommend using "
            "Submesh instead. Please get in touch if this is a critical "
            "issue for you.",
            FutureWarning,
        )
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
        with temp_internal_comm(mesh.comm) as icomm:
            num_layers = icomm.allreduce(num_layers, op=MPI.MAX)

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
        if mesh.geometric_dimension == mesh.topological_dimension:
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
        gdim = mesh.geometric_dimension + (extrusion_type == "uniform")
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
    IGNORE = "ignore"
    ERROR = "error"
    WARN = "warn"


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
        :class:`~.VertexOnlyMeshMissingPointsError`. If ``'ignore'``, will do
        nothing. Default is ``'error'``.
    :kwarg tolerance: The relative tolerance (i.e. as defined on the reference
        cell) for the distance a point can be from a mesh cell and still be
        considered to be in the cell. Note that this tolerance uses an L1
        distance (aka 'manhattan', 'taxicab' or rectilinear distance) so
        will scale with the dimension of the mesh. The default is the parent
        mesh's ``tolerance`` property. Changing this from default will
        cause the parent mesh's rtree to be rebuilt which can take some
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
    petsctools.cite("nixonhill2023consistent")

    if tolerance is None:
        tolerance = mesh.tolerance
    else:
        mesh.tolerance = tolerance

    vertexcoords = np.asarray(vertexcoords, dtype=RealType)

    if reorder is None:
        reorder = parameters["reorder_meshes"]

    gdim = mesh.geometric_dimension
    _, pdim = vertexcoords.shape

    if not np.isclose(np.sum(abs(vertexcoords.imag)), 0):
        raise ValueError("Point coordinates must have zero imaginary part")
    if pdim != gdim:
        raise ValueError(f"Mesh geometric dimension {gdim} must match point list dimension {pdim}")

    swarm, input_ordering_swarm, n_missing_points = _pic_swarm_in_mesh(
        mesh, vertexcoords, tolerance=tolerance, redundant=redundant, exclude_halos=False
    )

    missing_points_behaviour = MissingPointsBehaviour(missing_points_behaviour)
    if missing_points_behaviour != MissingPointsBehaviour.IGNORE:
        n_missing_points_global = mesh.comm.allreduce(n_missing_points, op=MPI.SUM)
        if n_missing_points_global:
            error = VertexOnlyMeshMissingPointsError(n_missing_points_global)
            if missing_points_behaviour == MissingPointsBehaviour.ERROR:
                raise error
            elif missing_points_behaviour == MissingPointsBehaviour.WARN:
                from warnings import warn
                warn(str(error))
            else:
                raise ValueError("missing_points_behaviour must be IGNORE, ERROR or WARN")

    name = name if name is not None else mesh.name + "_immersed_vom"
    swarm.dm.setName(_generate_default_mesh_topology_name(name))
    input_ordering_swarm.dm.setName(_generate_default_mesh_topology_name(name) + "_input_ordering")

    topology = VertexOnlyMeshTopology(
        swarm,
        mesh.topology,
        name=swarm.dm.getName(),
        reorder=reorder,
        input_ordering_swarm=input_ordering_swarm,
    )
    vmesh_out = make_vom_from_vom_topology(topology, name, tolerance)
    vmesh_out._parent_mesh = mesh

    return vmesh_out


class VertexOnlyMeshSF:
    """A PETSc.SF to use for VertexOnlyMesh"""

    def __init__(self, sf: PETSc.SF) -> None:
        if not isinstance(sf, PETSc.SF):
            raise TypeError(f"`sf` must be a `PETSc.SF`, not a {type(sf).__name__}")

        nroots, leaf_indices, remote = sf.getGraph()

        leaf_indices.setflags(write=False)
        remote.setflags(write=False)

        self.sf = sf
        self.nroots = nroots
        self.nleaves = len(leaf_indices)
        self.leaf_indices = leaf_indices
        self.remote = remote
        self.input_ranks = remote[:, 0]
        self.input_indices = remote[:, 1]
        self.leaf_buffer_size = (
            0 if len(leaf_indices) == 0 else int(leaf_indices.max()) + 1
        )

    @classmethod
    @PETSc.Log.EventDecorator()
    def discover(cls, parent_mesh: MeshGeometry, root_coordinates: np.ndarray) -> "VertexOnlyMeshSF":
        root_coordinates = np.asarray(
            root_coordinates.real,
            dtype=np.float64,
            order="C",
        )

        with temp_internal_comm(parent_mesh.comm) as comm:
            remote = rtree.discover_remote_roots(
                parent_mesh.distributed_rtree,
                root_coordinates,
                comm,
            )
        sf = PETSc.SF().create(comm=parent_mesh.comm)
        sf.setGraph(len(root_coordinates), None, remote)

        return cls(sf)

    @contextmanager
    def _mpi_unit(self, values: np.ndarray):
        item_count = np.prod(values.shape[1:])

        try:
            base_type = MPI._typedict[values.dtype.char]
        except KeyError:
            base_type = MPI.BYTE
            item_count *= values.dtype.itemsize

        if item_count == 1:
            # No need to create contiguous unit
            # freeing is handled automatically
            yield base_type
            return

        unit = base_type.Create_contiguous(item_count)
        unit.Commit()
        try:
            yield unit
        finally:
            unit.Free()

    def _check_arrays(
        self,
        root_values: np.ndarray,
        leaf_values: np.ndarray,
    ) -> None:
        # TODO: make these collective
        if root_values.shape[0] != self.nroots:
            raise ValueError("Number of root values does not match number of roots in the SF.")
        if leaf_values.shape[0] < self.leaf_buffer_size:
            raise ValueError("Leaf array is too small for the SF leaf indices.")
        if leaf_values.shape[1:] != root_values.shape[1:]:
            raise ValueError("`leaf_values` shape does not match `root_values`.")
        if leaf_values.dtype != root_values.dtype:
            raise TypeError("`leaf_values` dtype does not match `root_values`.")

    def broadcast(
        self,
        root_values: np.ndarray,
        leaf_values: np.ndarray | None = None,
        op: MPI.Op = MPI.REPLACE,
    ) -> np.ndarray:
        if leaf_values is None:
            leaf_shape = (self.leaf_buffer_size,) + root_values.shape[1:]
            leaf_values = np.empty(leaf_shape, dtype=root_values.dtype)

        self._check_arrays(root_values, leaf_values)

        with self._mpi_unit(root_values) as unit:
            self.sf.bcastBegin(unit, root_values, leaf_values, op)
            self.sf.bcastEnd(unit, root_values, leaf_values, op)
        return leaf_values

    def reduce(
        self,
        leaf_values: np.ndarray,
        root_values: np.ndarray,
        op: MPI.Op = MPI.REPLACE,
    ) -> np.ndarray:
        self._check_arrays(root_values, leaf_values)

        with self._mpi_unit(root_values) as unit:
            self.sf.reduceBegin(unit, leaf_values, root_values, op)
            self.sf.reduceEnd(unit, leaf_values, root_values, op)
        return root_values

    def create_embedded_leaf_sf(
        self,
        mask: np.ndarray,
    ) -> "VertexOnlyMeshSF":
        if mask.shape != (self.nleaves,):
            raise ValueError("mask must contain one entry per leaf")
        selected_leaf_indices = self.leaf_indices[mask]
        return type(self)(self.sf.createEmbeddedLeafSF(selected_leaf_indices))


class FiredrakeDMSwarm:
    """A DMSwarm for use with :func:`VertexOnlyMesh`."""

    def __init__(self, dm: PETSc.DMSwarm, extruded: bool = False):
        """Initialize a Firedrake DMSwarm.

        Parameters
        ----------
        dm : PETSc.DMSwarm
            The underlying PETSc DMSwarm.
        extruded : bool
            Whether the swarm is embedded in an extruded mesh.
        """
        if not isinstance(dm, PETSc.DMSwarm):
            raise TypeError(f"`dm` must be a `PETSc.DMSwarm`, not a {type(dm).__name__}")
        self.dm = dm
        self.extruded = extruded

    @classmethod
    def create(
        cls,
        cell_dm: PETSc.DM,
        tdim: int,
        gdim: int,
        extruded: bool,
        extra_fields: Sequence[tuple] = (),
    ) -> "FiredrakeDMSwarm":
        """Create an empty Firedrake DMSwarm.

        Parameters
        ----------
        cell_dm : PETSc.DM
            The PETSc DM containing the cells in which the swarm is embedded.
        tdim : int
            The topological dimension of the embedding mesh.
        gdim : int
            The geometric dimension of the embedding mesh.
        extruded : bool
            Whether the parent mesh is extruded.
        extra_fields : sequence of tuple
            Additional ``(name, block_size, dtype)`` fields to register.

        Returns
        -------
        FiredrakeDMSwarm
            The empty swarm with all fields registered.
        """
        dm = PETSc.DMSwarm().create(comm=cell_dm.comm)
        dm.setDimension(gdim)
        dm.setCoordinateDim(gdim)
        dm.setCellDM(cell_dm)
        if not isinstance(cell_dm, PETSc.DMSwarm):
            dm.setType(PETSc.DMSwarm.Type.PIC)

        dm.registerField("parentcellnum", 1, dtype=IntType)
        dm.registerField("refcoord", tdim, dtype=RealType)
        dm.registerField("globalindex", 1, dtype=IntType)
        dm.registerField("inputrank", 1, dtype=IntType)
        dm.registerField("inputindex", 1, dtype=IntType)
        if extruded:
            dm.registerField("parentcellbasenum", 1, dtype=IntType)
            dm.registerField("parentcellextrusionheight", 1, dtype=IntType)

        for name, size, dtype in extra_fields:
            dm.registerField(name, size, dtype=dtype)

        dm.finalizeFieldRegister()
        return cls(dm, extruded=extruded)

    def set_halo_sf(
        self,
        n_owned: int,
        owner_ranks: np.ndarray,
        owner_indices: np.ndarray,
    ) -> None:
        """Set the point SF connecting halo points to their owners.

        Parameters
        ----------
        n_owned : int
            Number of owned points, which precede halo points in the swarm.
        owner_ranks : numpy.ndarray
            Owning MPI rank for each halo point.
        owner_indices : numpy.ndarray
            Local index on the owning rank for each halo point.
        """
        npoints = self.dm.getLocalSize()
        local = np.arange(n_owned, npoints, dtype=IntType)
        remote = np.empty((len(owner_ranks), 2), dtype=IntType)
        remote[:, 0] = owner_ranks
        remote[:, 1] = owner_indices

        sf = self.dm.getPointSF()
        sf.setGraph(npoints, local, remote)
        self.dm.setPointSF(sf)

    @contextmanager
    def field(self, name: str) -> Generator[np.ndarray]:
        """Context manager to access a field on the DMSwarm.

        Parameters
        ----------
        name : str
            The name of the field to access.

        Yields
        ------
        numpy.ndarray
            The field as a NumPy array. The array and views derived from it
            must not be used after leaving the context.
        """
        # petsc4py will error if you try to access an active field without first restoring it.
        values = self.dm.getField(name)
        try:
            yield values
        finally:
            self.dm.restoreField(name)

    def set_field(self, name: str, values: np.ndarray) -> None:
        """Set the values of a DMSwarm field.

        Parameters
        ----------
        name : str
            Name of the field.
        values : numpy.ndarray
            Values to copy into the field.
        """
        with self.field(name) as field:
            field[...] = np.asarray(values).reshape(field.shape)


@PETSc.Log.EventDecorator()
def _pic_swarm_in_mesh(
    parent_mesh: MeshGeometry,
    coords: np.ndarray,
    fields: Sequence[tuple] | None = None,
    tolerance: float | None = None,
    redundant: bool = True,
    exclude_halos: bool = True,
) -> tuple[FiredrakeDMSwarm, FiredrakeDMSwarm, int]:
    """Create a particle-in-cell DMSwarm immersed in a mesh.

    Parameters
    ----------
    parent_mesh : MeshGeometry
        The mesh in which to immerse the points.
    coords : numpy.ndarray
        Point coordinates with shape ``(npoints, gdim)``.
    fields : sequence of tuple, optional
        Additional ``(name, block_size, dtype)`` fields to register on the
        distributed swarm.
    tolerance : float, optional
        Reference-cell tolerance used when locating points.
    redundant : bool
        If true, use only the coordinates supplied on MPI rank zero.
    exclude_halos : bool
        If true, exclude points in halo cells.

    Returns
    -------
    FiredrakeDMSwarm
        The swarm distributed according to the parent mesh.
    FiredrakeDMSwarm
        A swarm preserving the input rank and point ordering, including
        points not found in the parent mesh.
    int
        Number of input points not found in the parent mesh on this rank.

    Notes
    -----
    The input-ordering swarm uses the distributed swarm's PETSc DM as its
    CellDM. Its cell IDs are therefore local indices into the distributed
    swarm rather than parent-mesh cell numbers.
    """

    if tolerance is None:
        tolerance = parent_mesh.tolerance
    else:
        parent_mesh.tolerance = tolerance

    if parent_mesh.extruded and parent_mesh.variable_layers:
        raise NotImplementedError(
            "Cannot create a DMSwarm in an ExtrudedMesh with variable layers."
        )

    # in the redundant=True case we discard all the points not on rank zero
    if redundant and parent_mesh.comm.rank != 0:
        coords = np.empty((0, parent_mesh.geometric_dimension), dtype=RealType)

    (
        embedded_sf,
        winner_cells,
        winner_ref_coords,
        winner_ranks,
        parent_cell_nums,
        reference_coords,
        owner_ranks,
        physical_coords,
    ) = _parent_mesh_embedding(
        parent_mesh,
        coords,
        tolerance,
        exclude_halos=exclude_halos,
    )

    nroots = len(winner_cells)
    missing_roots = winner_ranks == -1
    n_missing_points = int(np.count_nonzero(missing_roots))
    input_owner_ranks = np.where(
        missing_roots, parent_mesh.comm.size + 1, winner_ranks
    )

    start_idx = parent_mesh.comm.exscan(nroots) or 0
    global_idxs = start_idx + np.arange(nroots, dtype=IntType)
    global_idxs_leaves = embedded_sf.broadcast(global_idxs)[embedded_sf.leaf_indices]

    owned_indices = np.flatnonzero(owner_ranks == parent_mesh.comm.rank)
    halo_indices = np.flatnonzero(owner_ranks != parent_mesh.comm.rank)
    n_owned = len(owned_indices)
    swarm_indices = np.concatenate([owned_indices, halo_indices])
    swarm_parent_cells = parent_cell_nums[swarm_indices]
    if parent_mesh.extruded:
        swarm_base_cells, swarm_extrusion_heights = _parent_extrusion_numbering(
            swarm_parent_cells, parent_mesh.layers
        )
        cell_numbers = swarm_base_cells
    else:
        cell_numbers = swarm_parent_cells
    cell_ids = parent_mesh.topology.cell_closure[cell_numbers, -1]

    swarm = FiredrakeDMSwarm.create(
        parent_mesh.topology.topology_dm,
        parent_mesh.topological_dimension,
        parent_mesh.geometric_dimension,
        parent_mesh.extruded,
        extra_fields=() if fields is None else fields,
    )
    swarm.dm.setLocalSizes(len(swarm_indices), -1)
    cell_id_name = swarm.dm.getCellDMActive().getCellID()
    swarm.set_field("DMSwarmPIC_coor", physical_coords[swarm_indices])
    swarm.set_field(cell_id_name, cell_ids)
    swarm.set_field("parentcellnum", swarm_parent_cells)
    swarm.set_field("refcoord", reference_coords[swarm_indices])
    swarm.set_field("globalindex", global_idxs_leaves[swarm_indices])
    swarm.set_field("DMSwarm_rank", owner_ranks[swarm_indices])
    swarm.set_field("inputrank", embedded_sf.input_ranks[swarm_indices].astype(IntType))
    swarm.set_field("inputindex", embedded_sf.input_indices[swarm_indices].astype(IntType))
    if parent_mesh.extruded:
        swarm.set_field("parentcellbasenum", swarm_base_cells)
        swarm.set_field("parentcellextrusionheight", swarm_extrusion_heights)

    owner_swarm_idx_buf = np.full(embedded_sf.leaf_buffer_size, -1, dtype=IntType)
    owner_swarm_idx_buf[embedded_sf.leaf_indices[owned_indices]] = np.arange(n_owned, dtype=IntType)

    owner_swarm_idx_roots = np.full(nroots, -1, dtype=IntType)
    embedded_sf.reduce(owner_swarm_idx_buf, owner_swarm_idx_roots, op=MPI.MAX)

    owner_swarm_idxs = embedded_sf.broadcast(owner_swarm_idx_roots)[embedded_sf.leaf_indices]
    swarm.set_halo_sf(
        n_owned,
        owner_ranks[halo_indices],
        owner_swarm_idxs[halo_indices],
    )

    # Now we create the corresponding input-ordering swarm.
    original_ordering_swarm = FiredrakeDMSwarm.create(
        swarm.dm,
        parent_mesh.topological_dimension,
        parent_mesh.geometric_dimension,
        parent_mesh.extruded,
    )
    original_ordering_swarm.dm.setLocalSizes(nroots, -1)
    cell_id_name = original_ordering_swarm.dm.getCellDMActive().getCellID()
    original_ordering_swarm.set_field("DMSwarmPIC_coor", coords)
    original_ordering_swarm.set_field(cell_id_name, owner_swarm_idx_roots.astype(IntType))
    original_ordering_swarm.set_field("parentcellnum", winner_cells)
    original_ordering_swarm.set_field("refcoord", winner_ref_coords)
    original_ordering_swarm.set_field("globalindex", global_idxs)
    original_ordering_swarm.set_field("DMSwarm_rank", input_owner_ranks)
    original_ordering_swarm.set_field("inputrank", np.full(nroots, parent_mesh.comm.rank, dtype=IntType))
    original_ordering_swarm.set_field("inputindex", np.arange(nroots, dtype=IntType))
    if parent_mesh.extruded:
        base_cells, extrusion_heights = _parent_extrusion_numbering(winner_cells, parent_mesh.layers)
        original_ordering_swarm.set_field("parentcellbasenum", base_cells)
        original_ordering_swarm.set_field("parentcellextrusionheight", extrusion_heights)

    # no halos in input-ordering swarm
    empty = np.empty(0, dtype=IntType)
    original_ordering_swarm.set_halo_sf(nroots, empty, empty)

    return swarm, original_ordering_swarm, n_missing_points


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


@PETSc.Log.EventDecorator()
def _parent_mesh_embedding(
    parent_mesh,
    coords,
    tolerance,
    exclude_halos=False,
):
    """Find the parent mesh cells containing the given coordinates.

    Uses a distributed R-tree to identify candidate ranks for each point,
    then assigns owning cells using sparse communication.

    Parameters
    ----------
    parent_mesh : Mesh
        The parent mesh to embed in.
    coords : np.ndarray
        The array coordinates to embed, of shape `(npoints, dim)`.
    tolerance : float
        The relative tolerance (i.e. as defined on the reference cell) for the
        distance a point can be from a cell and still be considered to be in
        the cell. Note that this tolerance uses an L1
        distance (aka 'manhattan', 'taxicab' or rectilinear distance) so
        will scale with the dimension of the mesh. The default is the parent
        mesh's `tolerance` property. Changing this from default will
        cause the parent mesh's rtree to be rebuilt which can take some
        time.
    exclude_halos : bool
        If True, the embedded SF excludes halo leaves and contains only
        winning owned leaves.

    Returns
    -------
    embedded_sf : VertexOnlyMeshSF
        The star forest connecting root points to the 'winning' leaf point(s).
        Each root may be connected to multiple leaves if halos are included.
    winner_cells : np.ndarray
        An array of shape `(nroots,)` containing the Firedrake cell number on
        the winner rank for each root point. -1 for missing points.
    winner_ref_coords : np.ndarray
        An array of shape `(nroots, ref_dim)`, containing the reference
        coordinates inside the winner cell of each point. NaN for missing points.
    winner_ranks : np.ndarray
        An array of shape `(nroots,)` containing the MPI ranks that own the winning
        cells for each point. -1 for missing points.
    parent_cell_nums : np.ndarray
        Firedrake parent cell numbers for the embedded leaves.
    reference_coords : np.ndarray
        Reference coordinates for the embedded leaves.
    owner_ranks : np.ndarray
        Parent cell owner ranks for the embedded leaves.
    physical_coords : np.ndarray
        Physical coordinates for the embedded leaves.
    """
    if isinstance(parent_mesh.topology, VertexOnlyMeshTopology):
        raise NotImplementedError(
            "VertexOnlyMeshes don't have a working locate_cells_ref_coords_and_dists method"
        )
    # `candidate_sf` is a star forest where each root is an input point,
    # and its leaves are candidate points on ranks which may own the point
    candidate_sf = VertexOnlyMeshSF.discover(parent_mesh, coords)
    nroots = candidate_sf.nroots  # nroots == coords.shape[0]

    # send coords to the candidates, and locate each candidate point
    coords = candidate_sf.broadcast(coords)
    parent_cell_nums, ref_coords, ref_cell_dists, owning_ranks = (
        parent_mesh._locate_cells_ref_coords_dists_and_owners(coords, tolerance)
    )
    # Immersed manifold case: the reference coords have an extra dimension we can safely drop
    if parent_mesh.geometric_dimension > parent_mesh.topological_dimension:
        ref_coords = ref_coords[:, :parent_mesh.topological_dimension]

    # `keep` is a mask of candidate points we want to keep
    # keep only points which are visible on this rank (they were found in a cell)
    keep = parent_cell_nums != -1

    # TODO: try packing these next two reduction into (distance, -owner_rank) and reduce with MPI.MINLOC
    # don't think PETSc has the fast pack/unpack operations in SF for this, so we'd
    # have to create our own numpy dtype to do this...

    # keep points which attain the minimum L1 distance out of all candidates
    root_distance_min = np.full(nroots, np.inf, dtype=RealType)
    candidate_sf.reduce(
        np.where(keep, ref_cell_dists, np.inf),
        root_distance_min,
        op=MPI.MIN,
    )
    keep &= ref_cell_dists == candidate_sf.broadcast(root_distance_min)

    # multiple ranks may claim the minimum L1 distance. Break ties
    # by choosing the highest numbered rank.
    root_owner_max = np.full(nroots, -1, dtype=IntType)
    candidate_sf.reduce(
        np.where(keep, owning_ranks, -1),
        root_owner_max,
        op=MPI.MAX,
    )
    keep &= owning_ranks == candidate_sf.broadcast(root_owner_max)

    # Points in halo cells will be assigned to the rank owning that cell
    not_in_halo = owning_ranks == parent_mesh.comm.rank

    # this SF maps roots to their winning candidate leaf
    winner_sf = candidate_sf.create_embedded_leaf_sf(keep & not_in_halo)

    # Try packing these two reductions and do a single reduction

    # send winning cell number and ref coords to roots
    winner_cells = np.full(nroots, -1, dtype=IntType)
    winner_sf.reduce(parent_cell_nums, winner_cells)

    winner_ref_coords = np.full((nroots, ref_coords.shape[1]), np.nan, dtype=RealType)
    winner_sf.reduce(ref_coords, winner_ref_coords)

    embedded_sf = winner_sf if exclude_halos else candidate_sf.create_embedded_leaf_sf(keep)

    return (
        embedded_sf,
        winner_cells,
        winner_ref_coords,
        root_owner_max,
        parent_cell_nums[embedded_sf.leaf_indices],
        ref_coords[embedded_sf.leaf_indices],
        owning_ranks[embedded_sf.leaf_indices],
        coords[embedded_sf.leaf_indices],
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
        elif (elem.family() == "HDiv Trace" and elem.degree() == 0 and mesh.topological_dimension > 1) or \
                (elem.family() == "Lagrange" and elem.degree() == 1 and mesh.topological_dimension == 1) or \
                (elem.family() == "Q" and elem.degree() == 2 and mesh.topology.ufl_cell().cellname == "hexahedron"):
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
    reorder_noop = None
    tmesh1 = MeshTopology(plex1, name=plex1.getName(), reorder=reorder_noop,
                          distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
                          perm_is=tmesh._dm_renumbering,
                          distribution_name=tmesh._distribution_name,
                          permutation_name=tmesh._permutation_name,
                          comm=tmesh.comm)

    # Create a new coordinates function with the same values as before but
    # living on the new topology
    coordinates_fs = mesh.coordinates.function_space().reconstruct(mesh=tmesh1)
    relabeled_coordinates = function.CoordinatelessFunction(
        coordinates_fs,
        val=mesh.coordinates.dat.data_ro_with_halos,
        name=_generate_default_mesh_coordinates_name(tmesh1.name),
    )
    rmesh = MeshGeometry(relabeled_coordinates)
    rmesh.name = name1
    rmesh._tolerance = mesh.tolerance

    # Tag the relabeled mesh with the original distribution parameters
    rmesh._distribution_parameters = mesh._distribution_parameters
    rmesh._did_reordering = mesh._did_reordering
    return rmesh


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


def Submesh(mesh, subdim=None, subdomain_id=None, label_name=None, name=None, ignore_halo=False, reorder=None, comm=None):
    """Construct a submesh from a given mesh.

    Parameters
    ----------
    mesh : MeshGeometry
        Parent mesh (`MeshGeometry`).
    subdim : int | None
        Topological dimension of the submesh.
        Defaults to ``mesh.topological_dimension``.
    subdomain_id : int | Sequence | None
        Subdomain ID representing the submesh.
        If multiple subdomain IDs are provided, their union is taken.
        If `None` the submesh will cover the entire domain,
        this is useful to obtain a codim-1 submesh over all facets or
        a submesh over a different communicator.
    label_name : str | None
        Name of the label to search ``subdomain_id`` in.
        Defaults to ``'Cell Sets'`` or ``'Face Sets'`` depending on ``subdim``.
    name : str |  None
        Name of the submesh.
        Defaults to ``mesh.name + "_submesh"``·
    ignore_halo : bool
        Whether to exclude the halo from the submesh.
    reorder : bool | None
        Whether to reorder the mesh entities. By default,
        the submesh will be reordered if the parent mesh was reordered.
    comm : PETSc.Comm | None
        An optional sub-communicator to define the submesh.
        By default, the submesh is defined on `mesh.comm`.

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
    >>> mesh = UnitSquareMesh(4, 4)
    >>> x, y = SpatialCoordinate(mesh)
    >>> DG = FunctionSpace(mesh, "DG", 0)
    >>> DGT = FunctionSpace(mesh, "DGT", 0)

    Mark a cell subdomain and construct a codim-0 submesh from all cells in the subdomain

    >>> cell_marker = assemble(interpolate(conditional(lt(x, 0.5), 1, 0), DG))
    >>> mesh.mark_entities(cell_marker, 111)
    >>> submesh = Submesh(mesh, subdomain_id=111)

    Mark a facet subdomain and construct a codim-1 submesh from all facets in the subdomain

    >>> facet_marker = assemble(interpolate(conditional(lt(abs(x-0.5), 1E-12), 1, 0), DGT))
    >>> mesh.mark_entities(facet_marker, 222)
    >>> submesh = Submesh(mesh, subdim=mesh.topological_dimension-1, subdomain_id=222)

    Construct a codim-0 submesh of the union of multiple subdomains by passing a list

    >>> mesh.mark_entities(assemble(interpolate(conditional(lt(x, 0.5), 1, 0), DG)), 1)
    >>> mesh.mark_entities(assemble(interpolate(conditional(lt(y, 0.5), 1, 0), DG)), 2)
    >>> submesh = Submesh(mesh, subdomain_id=[1, 2])

    Construct a codim-1 submesh of all the facets (the skeleton mesh)

    >>> submesh = Submesh(mesh, subdim=1)

    Construct a codim-1 submesh of the entire boundary

    >>> submesh = Submesh(mesh, subdomain_id="on_boundary")

    Construct a codim-1 submesh of the union of multiple boundaries

    >>> submesh = Submesh(mesh, subdim=mesh.topological_dimension-1, subdomain_id=[1, 2, 3])

    Construct a codim-0 submesh of the part of the mesh owned by each MPI rank

    >>> submesh = Submesh(mesh, ignore_halo=True, comm=COMM_SELF)

    """
    if not isinstance(mesh, MeshGeometry):
        raise TypeError("Parent mesh must be a `MeshGeometry`")
    if isinstance(mesh.topology, ExtrudedMeshTopology):
        raise NotImplementedError("Can not create a submesh of an ``ExtrudedMesh``")
    elif isinstance(mesh.topology, VertexOnlyMeshTopology):
        raise NotImplementedError("Can not create a submesh of a ``VertexOnlyMesh``")

    if subdomain_id == "on_boundary":
        if subdim is None:
            subdim = mesh.topological_dimension - 1
        elif subdim != mesh.topological_dimension - 1:
            raise ValueError('subdomain_id="on_boundary" requires subdim=dim-1')
        if label_name is None:
            label_name = "exterior_facets"
        elif label_name != "exterior_facets":
            raise ValueError('subdomain_id="on_boundary" requires label_name="exterior_facets"')
        subdomain_id = 1

    if subdim is None:
        subdim = mesh.topological_dimension
    plex = mesh.topology_dm
    dim = plex.getDimension()
    if subdim not in {dim, dim - 1}:
        raise NotImplementedError(f"Found submesh dim ({subdim}) and parent dim ({dim})")
    if subdomain_id is None:
        if label_name is not None:
            raise ValueError("subdomain_id=None requires label_name=None.")
        # Select all entities
        label_name = "depth"
        subdomain_id = subdim
    elif label_name is None:
        if subdim == dim:
            label_name = dmcommon.CELL_SETS_LABEL
        elif subdim == dim - 1:
            label_name = dmcommon.FACE_SETS_LABEL
    subplex = dmcommon.submesh_create(plex, subdim, label_name, subdomain_id, ignore_halo, comm=comm)

    comm = comm or mesh.comm
    name = name or _generate_default_submesh_name(mesh.name)
    subplex.setName(_generate_default_mesh_topology_name(name))
    if subplex.getDimension() != subdim:
        raise RuntimeError(f"Found subplex dim ({subplex.getDimension()}) != expected ({subdim})")
    if reorder is None:
        # Ideally we should set perm_is = mesh._dm_renumbering[label_indices]
        reorder = mesh._did_reordering

    submesh = Mesh(
        subplex,
        submesh_parent=mesh,
        name=name,
        comm=comm,
        reorder=reorder,
        distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
    )
    # Tag the relabeled mesh with the original distribution parameters
    submesh._distribution_parameters = mesh._distribution_parameters
    return submesh


def coordinates_from_topology(topology: AbstractMeshTopology, element: finat.ufl.FiniteElement) -> "CoordinatelessFunction":
    """Convert DMPlex coordinates into Firedrake coordinates.

    Parameters
    ----------
    topology :
        The mesh topology.
    element :
        The finite element defining the coordinate function space.

    Returns
    -------
    CoordinatelessFunction :
        The coordinates of the DMPlex reordered to agree with Firedrake's
        element numbering.

    """
    import firedrake.functionspace as functionspace
    import firedrake.function as function

    if not isinstance(topology, ExtrudedMeshTopology) and len(topology.dm_cell_types) > 1:
        return _MultiCellTypeDummyCoordinates(topology, element)

    (gdim,) = element.reference_value_shape
    coordinates_fs = functionspace.FunctionSpace(topology, element)
    coordinates_data = dmcommon.reordered_coords(topology.topology_dm, coordinates_fs.dm.getDefaultSection(),
                                                 (topology.num_vertices(), gdim))
    return function.CoordinatelessFunction(coordinates_fs,
                                           val=coordinates_data,
                                           name=_generate_default_mesh_coordinates_name(topology.name))


class MeshSequenceGeometry(ufl.MeshSequence):
    """A representation of mixed mesh geometry."""

    def __init__(self, meshes, set_hierarchy=True):
        """Initialise.

        Parameters
        ----------
        meshes : tuple or list
            `MeshGeometry`s to make `MeshSequenceGeometry` with.
        set_hierarchy : bool
            Flag for making hierarchy.

        """
        for m in meshes:
            if not isinstance(m, MeshGeometry):
                raise ValueError(f"Got {type(m)}")
        super().__init__(meshes)
        self.comm = meshes[0].comm
        # Only set hierarchy at top level.
        if set_hierarchy:
            self.set_hierarchy()

    @cached_property
    def topology(self):
        return MeshSequenceTopology([m.topology for m in self._meshes])

    @property
    def topological(self):
        """Alias of topology.

        This is to ensure consistent naming for some multigrid codes."""
        return self.topology

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if len(other) != len(self):
            return False
        for o, s in zip(other, self):
            if o is not s:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._meshes)

    def __len__(self):
        return len(self._meshes)

    def __iter__(self):
        return iter(self._meshes)

    def __getitem__(self, i):
        return self._meshes[i]

    @cached_property
    def extruded(self):
        m = self.unique()
        return m.extruded

    def unique(self):
        """Return a single component or raise exception."""
        if len(set(self._meshes)) > 1:
            raise NonUniqueMeshSequenceError(f"Found multiple meshes in {self} where a single mesh is expected")
        m, = set(self._meshes)
        return m

    def set_hierarchy(self):
        """Set mesh hierarchy if needed."""
        from firedrake.mg.utils import set_level, get_level, has_level

        # TODO: Think harder on how mesh hierarchy should work with mixed meshes.
        if all(not has_level(m) for m in self._meshes):
            return
        else:
            if not all(has_level(m) for m in self._meshes):
                raise RuntimeError("Found inconsistent component meshes")
        hierarchy_list = []
        level_list = []
        for m in self:
            hierarchy, level = get_level(m)
            hierarchy_list.append(hierarchy)
            level_list.append(level)
        nlevels, = set(len(hierarchy) for hierarchy in hierarchy_list)
        level, = set(level_list)
        result = []
        for ilevel in range(nlevels):
            if ilevel == level:
                result.append(self)
            else:
                result.append(MeshSequenceGeometry([hierarchy[ilevel] for hierarchy in hierarchy_list], set_hierarchy=False))
        result = tuple(result)
        for i, m in enumerate(result):
            set_level(m, result, i)


class MeshSequenceTopology:
    """A representation of mixed mesh topology."""

    def __init__(self, meshes):
        """Initialise.

        Parameters
        ----------
        meshes : tuple or list
            `MeshTopology`s to make `MeshSequenceTopology` with.

        """
        for m in meshes:
            if not isinstance(m, AbstractMeshTopology):
                raise ValueError(f"Got {type(m)}")
        self._meshes = tuple(meshes)
        self.comm = meshes[0].comm

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
        return CellSequence([m.ufl_cell() for m in self._meshes])

    def ufl_mesh(self):
        dim = self.ufl_cell().topological_dimension
        return ufl.MeshSequence(
            [ufl.Mesh(finat.ufl.VectorElement("Lagrange", cell, 1, dim=dim))
             for cell in self.ufl_cell().cells]
        )

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if len(other) != len(self):
            return False
        for o, s in zip(other, self):
            if o is not s:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._meshes)

    def __len__(self):
        return len(self._meshes)

    def __iter__(self):
        return iter(self._meshes)

    def __getitem__(self, i):
        return self._meshes[i]

    @cached_property
    def extruded(self):
        m = self.unique()
        return m.extruded

    def unique(self):
        """Return a single component or raise exception."""
        if len(set(self._meshes)) > 1:
            raise NonUniqueMeshSequenceError(f"Found multiple meshes in {self} where a single mesh is expected")
        m, = set(self._meshes)
        return m
