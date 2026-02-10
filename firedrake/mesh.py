from __future__ import annotations

import dataclasses
import numpy as np
import collections
import ctypes
import functools
import os
import sys
from pyop3.cache import cached_on, serial_cache
import ufl
import finat.ufl
import FIAT
import weakref
from typing import Hashable, Literal, NoReturn, Tuple
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from ufl.classes import ReferenceGrad
from ufl.cell import CellSequence
from ufl.domain import extract_unique_domain, extract_domains
import enum
import numbers
from functools import cache, cached_property
import abc
from immutabledict import immutabledict as idict
import rtree
from textwrap import dedent
from pathlib import Path
from typing import Iterable, Optional, Union

from cachetools import cachedmethod
from pyop3.mpi import (
    MPI, COMM_WORLD, temp_internal_comm, collective
)
from pyop3.cache import memory_cache
from pyop3.pyop2_utils import as_tuple, tuplify
import pyop3 as op3
from pyop3.utils import pairwise, steps, debug_assert, just_one, single_valued, readonly
from finat.element_factory import as_fiat_cell
import petsctools
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
from firedrake.exceptions import VertexOnlyMeshMissingPointsError, NonUniqueMeshSequenceError
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


# TODO: use these
_FLAT_MESH_AXIS_LABEL_SUFFIX = "points"
_STRATIFIED_MESH_AXIS_LABEL_SUFFIX = "strata"


UNMARKED = -1
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


class ClosureOrdering(enum.Enum):
    PLEX = "plex"
    FIAT = "fiat"


class AbstractMeshTopology(abc.ABC):
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
        dmcommon.label_facets(self.topology_dm)
        mylabel = self.topology_dm.getLabel("exterior_facets")
        self._distribute()
        self._grown_halos = False

        self.name = name

        if self.comm.size > 1:
            self._add_overlap()
        if self.sfXB is not None:
            self.sfXC = sfXB.compose(self.sfBC) if self.sfBC else self.sfXB
        dmcommon.label_facets(self.topology_dm)  # this is there twice, why?
        dmcommon.complete_facet_labels(self.topology_dm)

        # TODO: Allow users to set distribution name if they want to save
        #       conceptually the same mesh but with different distributions,
        #       e.g., those generated by different partitioners.
        #       This currently does not make sense since those mesh instances
        #       of different distributions in general have different global
        #       point numbers (so they must be saved under different mesh names
        #       even though they are conceptually the same).
        # The name set here almost uniquely identifies a distribution, but
        # there is no guarantee that it really does or it continues to do so
        # there are lots of parameters that can change distributions.
        # Thus, when using CheckpointFile, it is recommended that the user set
        # distribution_name explicitly.

        dmcommon.mark_owned_points(self.topology_dm)

        if perm_is:
            self._old_to_new_point_renumbering = perm_is.invertPermutation()
            self._new_to_old_point_renumbering = perm_is
        else:
            with PETSc.Log.Event("Renumber mesh topology"):
                if isinstance(self.topology_dm, PETSc.DMPlex):
                    if reorder:
                        # Create an IS mapping from new to old cell numbers. This
                        # is unfortunately fairly involved, hopefully my choice of
                        # variable names is sufficient to explain things.
                        old_to_new_rcm_point_numbering_is = PETSc.IS().createGeneral(
                            self.topology_dm.getOrdering(PETSc.Mat.OrderingType.RCM).indices,
                            comm=MPI.COMM_SELF,
                        )
                        new_to_old_rcm_point_numbering_is = \
                            old_to_new_rcm_point_numbering_is.invertPermutation()
                        cell_is = PETSc.IS().createStride(self.num_cells, comm=MPI.COMM_SELF)
                        old_to_new_rcm_cell_numbering_section = dmcommon.entity_numbering(
                            cell_is, new_to_old_rcm_point_numbering_is, self.comm
                        )
                        old_to_new_rcm_cell_numbering_is = dmcommon.section_offsets(
                            old_to_new_rcm_cell_numbering_section, cell_is
                        )
                        new_to_old_rcm_cell_numbering_is = \
                            old_to_new_rcm_cell_numbering_is.invertPermutation()
                    else:
                        new_to_old_rcm_cell_numbering_is = None
                    new_to_old_point_numbering = dmcommon.compute_dm_renumbering(
                        self, new_to_old_rcm_cell_numbering_is
                    )
                    # Now take this renumbering and partition owned and ghost points, this
                    # is the part that pyop3 should ultimately be able to handle.
                    # NOTE: probably shouldn't do this for a VoM
                    new_to_old_point_numbering = dmcommon.partition_renumbering(
                        self.topology_dm, new_to_old_point_numbering
                    )

                else:
                    assert isinstance(self.topology_dm, PETSc.DMSwarm)
                    if reorder:
                        swarm = self.topology_dm
                        parent = self._parent_mesh.topology_dm
                        cell_id_name = swarm.getCellDMActive().getCellID()
                        swarm_parent_cell_nums = swarm.getField(cell_id_name).flatten()
                        old_to_new_parent_cell_indices = \
                            self._parent_mesh._old_to_new_point_renumbering.indices[swarm_parent_cell_nums]
                        swarm.restoreField(cell_id_name)
                        new_to_old_point_indices = \
                            np.argsort(old_to_new_parent_cell_indices, stable=True).astype(IntType)
                        new_to_old_point_numbering = \
                            PETSc.IS().createGeneral(new_to_old_point_indices, comm=MPI.COMM_SELF)
                    else:
                        new_to_old_point_numbering = dmcommon.compute_dm_renumbering(self, None)
                        # NOTE: probably shouldn't do this for a VoM
                        new_to_old_point_numbering = dmcommon.partition_renumbering(
                            self.topology_dm, new_to_old_point_numbering
                        )

            # TODO: replace "renumbering" with "numbering"
            self._new_to_old_point_renumbering = new_to_old_point_numbering
            self._old_to_new_point_renumbering = new_to_old_point_numbering.invertPermutation()

        # Set/Generate names to be used when checkpointing.
        self._distribution_name = distribution_name or _generate_default_mesh_topology_distribution_name(self.topology_dm.comm.size, self._distribution_parameters)
        self._permutation_name = permutation_name or _generate_default_mesh_topology_permutation_name(reorder)
        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)
        self._cache = defaultdict(dict)
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

    @property
    @abc.abstractmethod
    def dimension(self):
        pass

    @abc.abstractmethod
    def _distribute(self):
        """Distribute the mesh toplogy."""
        pass

    @abc.abstractmethod
    def _add_overlap(self):
        """Add overlap."""
        pass

    @cached_property
    def flat_points(self):
        # NOTE: In serial the point SF isn't set up in a valid state so we do this. It
        # would be nice to avoid this branch.
        if self.comm.size > 1:
            point_sf = self.topology_dm.getPointSF()
        else:
            point_sf = op3.local_sf(self.num_points, self.comm).sf

        point_sf_renum = op3.sf.renumber_petsc_sf(point_sf, self._new_to_old_point_renumbering)
        point_sf_renum = op3.StarForest(point_sf_renum, self.comm)


        # TODO: Allow the label here to be None
        return op3.Axis(
            [op3.AxisComponent(self.num_points, "mylabel", sf=point_sf_renum)],
            label="mesh",
        )

    @property
    @utils.deprecated("_new_to_old_point_renumbering")
    def _dm_renumbering(self):
        return self._new_to_old_point_renumbering

    @property
    def _is_renumbered(self) -> bool:
        return utils.strictly_all(
            map(bool, [self._old_to_new_point_renumbering, self._new_to_old_point_renumbering])
        )

    @cached_property
    def _cell_plex_indices(self) -> PETSc.IS:
        return PETSc.IS().createStride(self.num_cells, comm=MPI.COMM_SELF)

    @cached_property
    def _exterior_facet_plex_indices(self) -> PETSc.IS:
        return PETSc.IS().createGeneral(
            dmcommon.facets_with_label(self, "exterior_facets"), comm=MPI.COMM_SELF
        )

    @cached_property
    def _interior_facet_plex_indices(self) -> PETSc.IS:
        return PETSc.IS().createGeneral(
            dmcommon.facets_with_label(self, "interior_facets"), comm=MPI.COMM_SELF
        )

    @cached_property
    def exterior_facet_local_facet_indices(self) -> op3.Dat:
        local_facet_index = dmcommon.local_facet_number(self, "exterior")
        axis_tree = op3.AxisTree.from_iterable([self.exterior_facets.as_axis(), 1])
        return op3.Dat(axis_tree, data=local_facet_index.flatten())

    @cached_property
    def interior_facet_local_facet_indices(self) -> op3.Dat:
        local_facet_index = dmcommon.local_facet_number(self, "interior")
        axis_tree = op3.AxisTree.from_iterable([self.interior_facets.as_axis(), 2])
        return op3.Dat(axis_tree, data=local_facet_index.flatten())

    @cached_property
    def exterior_facet_vert_local_facet_indices(self) -> op3.Dat:
        local_facet_index = dmcommon.local_facet_number(self, "exterior_vert")
        axis_tree = op3.AxisTree.from_iterable([self.exterior_facets_vert.as_axis(), 1])
        return op3.Dat(axis_tree, data=local_facet_index.flatten())

    @cached_property
    def interior_facet_vert_local_facet_indices(self) -> op3.Dat:
        local_facet_index = dmcommon.local_facet_number(self, "interior_vert")
        axis_tree = op3.AxisTree.from_iterable([self.interior_facets_vert.as_axis(), 2])
        return op3.Dat(axis_tree, data=local_facet_index.flatten())

    @cached_property
    def _exterior_facet_local_numbers_dat(self):
        return self._local_facet_numbers_dat("exterior")

    @cached_property
    def _interior_facet_local_numbers_dat(self):
        return self._local_facet_numbers_dat("interior")

    # TODO: Make a standalone function
    def _local_facet_numbers_dat(self, facet_type: Literal["exterior"] | Literal["interior"]) -> op3.Dat:
        if facet_type == "exterior":
            facet_axes = self.exterior_facets
            arity = 1
        else:
            assert facet_type == "interior"
            facet_axes = self.interior_facets
            arity = 2

        local_facet_numbers = dmcommon.local_facet_number(self, facet_type)
        owned_local_facet_numbers = local_facet_numbers[:facet_axes.owned.local_size]

        # only ghost facets can have negative entries
        utils.debug_assert(lambda: (owned_local_facet_numbers >= 0).all())

        # FIXME: cast dtype, should be avoidable
        owned_local_facet_numbers = owned_local_facet_numbers.astype(np.uint32)

        axes = op3.AxisTree.from_iterable([facet_axes.owned.as_axis(), arity])
        return op3.Dat(axes, data=owned_local_facet_numbers.flatten())

    @cached_property
    def _exterior_facet_local_orientation_dat(self) -> op3.Dat:
        return self._local_facet_orientation_dat("exterior")

    @cached_property
    def _interior_facet_local_orientation_dat(self) -> op3.Dat:
        return self._local_facet_orientation_dat("interior")

    # TODO: make a standalone function
    def _local_facet_orientation_dat(self, facet_type: Literal["exterior", "interior"]) -> op3.Dat:
        if facet_type == "exterior":
            local_facet_numbers_dat = self._exterior_facet_local_numbers_dat
            arity = 1
            facet_to_cell_map = self._facet_support_dat("exterior").data_ro
        else:
            assert facet_type == "interior"
            local_facet_numbers_dat = self._interior_facet_local_numbers_dat
            arity = 2
            facet_to_cell_map = self._facet_support_dat("interior").data_ro

        facet_to_cell_map = facet_to_cell_map.reshape((-1, arity))

        dtype = gem.uint_type
        # Make a map from cell to facet orientations.
        fiat_cell = as_fiat_cell(self.ufl_cell())
        topo = fiat_cell.topology
        num_entities = [0]
        for d in range(len(topo)):
            num_entities.append(len(topo[d]))
        offsets = np.cumsum(num_entities)
        local_facet_start = offsets[-3]
        local_facet_end = offsets[-2]
        map_from_cell_to_facet_orientations = self.entity_orientations[:, local_facet_start:local_facet_end]

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
        local_facets = local_facet_numbers_dat.data_ro.reshape((-1, arity))
        # Make slice for masking out rows for which orientations are not needed.
        slice_ = (facet_to_cell_map != -1).all(axis=1)
        data = np.full_like(local_facets, np.iinfo(dtype).max)
        data[slice_, :] = np.take_along_axis(
            map_from_cell_to_facet_orientations[facet_to_cell_map[slice_, :]],
            local_facets.reshape(local_facets.shape + (1, ))[slice_, :, :],  # reshape as required by take_along_axis.
            axis=2,
        ).reshape(local_facets.shape)
        return op3.Dat(
            local_facet_numbers_dat.axes, data=data.flatten(),
            name=f"{self.name}_{facet_type}_local_facet_orientation"
        )

    @property
    @abc.abstractmethod
    def _strata_slice(self):  # or strata_axis?
        pass

    @property
    def _plex_strata_ordering(self):
        if self.dimension == 0:
            return (0,)
        elif self.dimension == 1:
            return (1, 0)
        elif self.dimension == 2:
            return (2, 0, 1)
        else:
            assert self.dimension == 3
            return (3, 0, 2, 1)  # I think, 1 and 2 might need swapping

    @utils.cached_property
    def points(self):
        return self.flat_points[self._strata_slice]

    @cached_property
    def _old_to_new_cell_numbering(self) -> PETSc.Section:
        return self._plex_to_entity_numbering(self.dimension)

    @cached_property
    def _old_to_new_cell_numbering_is(self) -> PETSc.IS:
        cell_indices = PETSc.IS().createStride(self.num_cells, comm=MPI.COMM_SELF)
        return dmcommon.section_offsets(self._old_to_new_cell_numbering, cell_indices)

    @cached_property
    def _new_to_old_cell_numbering(self) -> np.ndarray:
        cell_indices = PETSc.IS().createStride(self.num_cells, comm=MPI.COMM_SELF)
        renumbering_is = dmcommon.section_offsets(self._old_to_new_cell_numbering, cell_indices)
        return renumbering_is.invertPermutation().indices

    @cached_property
    def _new_to_old_interior_facet_numbering_is(self) -> PETSc.IS:
        old_to_new_numbering_is = dmcommon.section_offsets(
            self._old_to_new_interior_facet_numbering, self._interior_facet_plex_indices
        )
        return old_to_new_numbering_is.invertPermutation()

    @property
    def _new_to_old_interior_facet_numbering(self) -> np.ndarray[IntType]:
        return self._new_to_old_interior_facet_numbering_is.indices

    @cached_property
    def _new_to_old_exterior_facet_numbering_is(self) -> PETSc.IS:
        old_to_new_numbering_is = dmcommon.section_offsets(
            self._old_to_new_exterior_facet_numbering, self._exterior_facet_plex_indices
        )
        return old_to_new_numbering_is.invertPermutation()

    @property
    def _new_to_old_exterior_facet_numbering(self) -> np.ndarray[IntType]:
        return self._new_to_old_exterior_facet_numbering_is.indices

    @cached_property
    def _old_to_new_facet_numbering(self) -> PETSc.Section:
        return self._plex_to_entity_numbering(self.dimension-1)

    @cached_property
    def _old_to_new_exterior_facet_numbering(self):
        return dmcommon.entity_numbering(self._exterior_facet_plex_indices, self._new_to_old_point_renumbering, self.comm)

    @cached_property
    def _old_to_new_interior_facet_numbering(self):
        return dmcommon.entity_numbering(self._interior_facet_plex_indices, self._new_to_old_point_renumbering, self.comm)

    @cached_property
    def _old_to_new_vertex_numbering(self) -> PETSc.Section:
        return self._plex_to_entity_numbering(0)

    # IMPORTANT: This used to return a mapping from point numbering to entity numbering
    # but now returns entity numbering to entity numbering
    @cachedmethod(lambda self: self._cache["_entity_numbering"])
    def _plex_to_entity_numbering(self, dim):
        p_start, p_end = self.topology_dm.getDepthStratum(dim)
        plex_indices = PETSc.IS().createStride(size=p_end-p_start, first=p_start, comm=MPI.COMM_SELF)
        return dmcommon.entity_numbering(plex_indices, self._new_to_old_point_renumbering, self.comm)

    @cached_property
    def _global_old_to_new_vertex_numbering(self) -> PETSc.Section:
        # NOTE: This will return negative entries for ghosts
        return self._old_to_new_vertex_numbering.createGlobalSection(self.topology_dm.getPointSF())

    @property
    def comm(self):
        return self.user_comm

    def mpi_comm(self):
        """The MPI communicator this mesh is built on (an mpi4py object)."""
        return self.comm

    @cached_property
    def point_sf(self) -> op3.StarForest:
        petsc_sf = self.topology_dm.getPointSF()
        return op3.StarForest(petsc_sf, self.num_points)

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
    def exterior_facets(self) -> op3.IndexedAxisTree:
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

    @utils.cached_property
    def _strata_slice(self):
        if self.dimension == 0:
            return op3.Slice("mesh", [op3.AffineSliceComponent("mylabel", 0, None, label=0)], label=self.name)

        subsets = []
        if self._is_renumbered:
            for dim in self._plex_strata_ordering:
                indices = op3.ArrayBuffer(self._entity_indices[dim], ordered=True)
                subset_axes = op3.Axis({dim: op3.Scalar(indices.size)}, self.name)
                subset_array = op3.Dat(subset_axes, buffer=indices)
                subset = op3.Subset("mylabel", subset_array, label=dim)
                subsets.append(subset)
        else:
            raise NotImplementedError("TODO")
            for dim in self._plex_strata_ordering:
                start, end = self.topology_dm.getDepthStratum(dim)
                slice_component = op3.AffineSliceComponent("mylabel", start, end, label=str(dim))
                subsets.append(slice_component)

        return op3.Slice("mesh", subsets, label=self.name)

    @property
    @abc.abstractmethod
    def _entity_indices(self):
        pass

    # @property
    # @abc.abstractmethod
    # def _plex_strata_ordering(self):
    #     """Map from entity dimension to ordering in the DMPlex numbering.
    #
    #     For example, 3D meshes begin by numbering cells from 0, then vertices,
    #     then faces and lastly edges.
    #
    #     """

    def closure(self, index, ordering: ClosureOrdering | str = ClosureOrdering.FIAT):
        if ordering == ClosureOrdering.PLEX:
            return self._plex_closure(index)
        elif ordering == ClosureOrdering.FIAT:
            # target_paths = tuple(index.iterset.target_paths.values())
            # if len(target_paths) != 1 or target_paths[0] != {self.name: self.cell_label}:
            #     raise ValueError("FIAT closure ordering is only valid for cell closures")
            return self._fiat_closure(index)

    @cached_property
    def _closure_sizes(self) -> idict[idict]:
        """
        Examples
        --------
        UFCInterval:
            return idict({
                # the closure of a vertex is just the vertex
                0: {0: 1, 1: 0},
                # the closure of a cell is the cell and two vertices
                1: {0: 2, 1: 1},
            })
        """
        fiat_cell = as_fiat_cell(self.ufl_cell())

        # This just counts the number of entries to figure out how many
        # entities are in the cell. For example, a UFCInterval has topology
        #
        #     {0: {0: (0,), 1: (1,)}, 1: {0: (0, 1)}}
        #
        # from which we can infer that there are 2 vertices from
        # len(topology[0]) and a single cell from len(topology[1]).

        # TODO: This only works for cell closures, in principle it should be
        # possible to do this for sub-dimensions too.
        return idict({
            self.cell_label: {
                dim: len(dim_topology)
                for dim, dim_topology in fiat_cell.get_topology().items()
            }
        })

    @cached_property
    def _plex_closure(self):
        return self._closure_map(ClosureOrdering.PLEX)

    @cached_property
    def _fiat_closure(self):
        return self._closure_map(ClosureOrdering.FIAT)

    # TODO: remove _fiat_closure and _plex_closure and just cache this method
    def _closure_map(self, ordering):
        # if ordering is a string (e.g. "fiat") then convert to an enum
        ordering = ClosureOrdering(ordering)
        if ordering == ClosureOrdering.PLEX:
            closure_arrayss = dict(enumerate(self._plex_closures_localized))
        elif ordering == ClosureOrdering.FIAT:
            # FIAT ordering is only valid for cell closures
            closure_arrayss = {self.cell_label: self._fiat_cell_closures_localized}
        else:
            raise ValueError(f"'{ordering}' is not a recognised closure ordering option")

        # NOTE: This is very similar to what we do for supports
        closures = {}
        for from_dim, closure_arrays in closure_arrayss.items():
            iterset = self.points[op3.as_slice(from_dim)]

            full_map_components = []
            owned_map_components = []
            for to_dim, closure_array in closure_arrays.items():
                # Ideally this should be fine to not have, makes the logic more complicated
                # _, size = clos.shape
                # if size == 0:
                #     continue

                # target_axis = self.name
                # target_dim = map_dim

                # NOTE: currently we must label the innermost axis of the map to be the same as the resulting
                # indexed axis tree. I don't yet know whether to raise an error if this is not upheld or to
                # fix automatically internally via additional replace() arguments.
                closure_axes = op3.AxisTree.from_iterable([
                    iterset.as_axis(), op3.Axis({to_dim: self._closure_sizes[from_dim][to_dim]}, "closure")
                ])
                closure_dat = op3.Dat(closure_axes, data=closure_array.flatten())
                owned_closure_dat = op3.Dat(closure_axes.owned.materialize(), data=closure_dat.data_ro)

                full_map_component = op3.TabulatedMapComponent(
                    self.points.as_axis().label, to_dim, closure_dat, label=to_dim
                )
                owned_map_component = op3.TabulatedMapComponent(
                    self.points.as_axis().label, to_dim, owned_closure_dat, label=to_dim
                )

                full_map_components.append(full_map_component)
                owned_map_components.append(owned_map_component)

            full_from_path = idict({iterset.as_axis().label: iterset.as_axis().component.label})
            owned_from_path  = idict({iterset.owned.as_axis().label: iterset.owned.as_axis().component.label})

            closures[full_from_path] = [full_map_components]
            closures[owned_from_path] = [owned_map_components]
        return op3.Map(closures, name="closure")

    # NOTE: Probably better to cache the 'everything' case and then drop as necessary when k is given
    @cachedmethod(lambda self: self._cache["MeshTopology._star"])
    def _star(self, *, k: int) -> op3.Map:
        def star_func(pt):
            return self.topology_dm.getTransitiveClosure(pt, useCone=False)[0]

        stars = {}
        for dim in range(self.dimension+1):
            map_plex_pts, sizes = self._memoize_map(star_func, dim)

            # Now renumber the points. Note that this transforms us from 'plex' numbering to 'stratum' numbering.
            map_strata_pts_renum = tuple(
                self._renumber_map(dim, d, map_plex_pts[d], sizes[d]) for d in range(self.dimension+1)
            )

            map_components = []
            for map_dim, (map_stratum_pts, sizes) in enumerate(map_strata_pts_renum):
                if k is not None and k != map_dim:
                    continue

                outer_axis = self.points[str(dim)].root
                # NOTE: This is technically constant-sized, so we want to invalidate writes, but we don't
                # want to inject into the kernel!
                size_dat = op3.Dat(outer_axis, data=sizes, max_value=max(sizes), prefix="size")
                inner_axis = op3.Axis({str(map_dim): size_dat}, "star")
                map_axes = op3.AxisTree.from_nest(
                    {outer_axis: inner_axis}
                )
                map_dat = op3.Dat(map_axes, data=map_stratum_pts, prefix="map")
                map_components.append(
                    op3.TabulatedMapComponent(self.name, str(map_dim), map_dat)
                )
            # 1-tuple here because in theory star(cell) could map to other valid things (like points)
            stars[idict({self.name: str(dim)})] = (tuple(map_components),)

        return op3.Map(stars, name="star")

    def _reorder_closure_fiat_simplex(self, closure_data):
        return dmcommon.closure_ordering(self, closure_data)

    def _reorder_closure_fiat_quad(self, closure_data):
        petsctools.cite("Homolya2016")
        petsctools.cite("McRae2016")

        cell_ranks = dmcommon.get_cell_remote_ranks(self.topology_dm)
        facet_orientations = dmcommon.quadrilateral_facet_orientations(self.topology_dm, self._global_old_to_new_vertex_numbering, cell_ranks)
        cell_orientations = dmcommon.orientations_facet2cell(
            self, cell_ranks, facet_orientations,
        )
        dmcommon.exchange_cell_orientations(self, self._old_to_new_cell_numbering, cell_orientations)

        return dmcommon.quadrilateral_closure_ordering(self, cell_orientations)

    def _reorder_closure_fiat_hex(self, plex_closures):
        return dmcommon.create_cell_closure(plex_closures)

    def star(self, index, *, k=None):
        return self._star(k=k)(index)

    # NOTE: I think I duplicated this somewhere...
    def _renumber_map(self, map_pts, src_dim, dest_dim, sizes=None, *, src_mesh = None):
        """
        sizes :
            If `None` implies non-ragged
        """
        # debug
        if src_mesh is None:
            src_mesh = self

        src_renumbering = src_mesh._plex_to_entity_numbering(src_dim)
        dest_renumbering = self._plex_to_entity_numbering(dest_dim)

        src_start, src_end = src_mesh.topology_dm.getDepthStratum(src_dim)
        dest_start, dest_end = self.topology_dm.getDepthStratum(dest_dim)

        map_pts_renum = np.empty_like(map_pts)

        if sizes is None:  # fixed size
            for src_pt, map_data_per_pt in enumerate(map_pts):
                src_pt_renum = src_renumbering.getOffset(src_pt+src_start)
                for i, dest_pt in enumerate(map_data_per_pt):
                    dest_pt_renum = dest_renumbering.getOffset(dest_pt)
                    map_pts_renum[src_pt_renum, i] = dest_pt_renum
            return readonly(map_pts_renum)
        else:
            sizes_renum = np.empty_like(sizes)
            offsets = utils.steps(sizes)
            for src_stratum_pt, src_plex_pt in enumerate(range(src_start, src_end)):
                src_stratum_pt_renum = src_renumbering.getOffset(src_plex_pt)
                sizes_renum[src_stratum_pt_renum] = sizes[src_stratum_pt]

            offsets_renum = utils.steps(sizes_renum)
            map_pts_renum = np.empty_like(map_pts)
            for src_stratum_pt, src_plex_pt in enumerate(range(src_start, src_end)):
                src_stratum_pt_renum = src_renumbering.getOffset(src_plex_pt)
                for i in range(sizes[src_stratum_pt]):
                    dest_pt = map_pts[offsets[src_stratum_pt]+i]
                    dest_stratum_pt_renum = dest_renumbering.getOffset(dest_pt)
                    map_pts_renum[offsets_renum[src_stratum_pt_renum]+i] = dest_stratum_pt_renum

            return readonly(map_pts_renum), readonly(sizes_renum)

    def support(self, index):
        return self._support(index)

    @cached_property
    def _support(self):
        supports = {}

        # 1-tuple here because in theory support(facet) could map to other valid things (like points)
        exterior_facets_axis = self.exterior_facets.owned.as_axis()
        supports[idict({exterior_facets_axis.label: exterior_facets_axis.component.label})] = (
            (
                op3.TabulatedMapComponent(
                    self.name,
                    self.cell_label,
                    self._facet_support_dat("exterior"),
                    label=0,
                ),
            ),
        )

        interior_facets_axis = self.interior_facets.owned.as_axis()
        supports[idict({interior_facets_axis.label: interior_facets_axis.component.label})] = (
            (
                op3.TabulatedMapComponent(
                    self.name,
                    self.cell_label,
                    self._facet_support_dat("interior"),
                    label=0,
                ),
            ),
        )

        return op3.Map(supports, name="support")

    # TODO: Redesign all this, this sucks for extruded meshes
    @cached_property
    def _support_dats(self):
        def support_func(pt):
            return self.topology_dm.getSupport(pt)

        supports = []
        for from_dim in range(self.dimension+1):
            # cells have no support
            if from_dim == self.dimension:
                supports.append({})
                continue

            map_data, sizes = self._memoize_map(support_func, from_dim)

            # renumber it
            for to_dim, size in sizes.items():
                map_data[to_dim], sizes[to_dim] = self._renumber_map(
                    map_data[to_dim],
                    from_dim,
                    to_dim,
                    size,
                )

            # only the next dimension has entries
            map_dim = from_dim + 1
            size = sizes[map_dim]
            data = map_data[map_dim]

            # supports should only target a single dimension
            op3.utils.debug_assert(
                lambda: all(
                    (s == 0).all() for d, s in sizes.items() if d != map_dim
                )
            )

            iterset_axis = self.points[from_dim].as_axis()
            size_dat = op3.Dat(iterset_axis, data=size, prefix="size")
            support_axes = op3.AxisTree.from_iterable([
                iterset_axis, op3.Axis(size_dat)
            ])
            support_dat = op3.Dat(support_axes, data=data, prefix="support")
            owned_support_dat = op3.Dat(
                support_axes.owned.materialize(), data=support_dat.data_ro, prefix="support"
            )


            supports.append({map_dim: (support_dat, owned_support_dat)})
        return tuple(supports)

    # this is almost completely pointless
    def _facet_support_dat(self, facet_type: Literal["exterior"] | Literal["interior"]) -> op3.Dat:
        assert facet_type in {"exterior", "interior"}

        # Get the support map for *all* facets in the mesh, not just the
        # exterior/interior ones. We have to filter it. Note that these
        # dats are ragged because support sizes are not consistent.
        _, facet_support_dat = self._support_dats[self.facet_label][self.cell_label]

        if facet_type == "exterior":
            facet_axis = self.exterior_facets.owned.as_axis()
            selected_facets_is = dmcommon.section_offsets(
                self._old_to_new_facet_numbering, self._exterior_facet_plex_indices, sort=True
            )
            arity = 1
        else:
            facet_axis = self.interior_facets.owned.as_axis()
            selected_facets_is = dmcommon.section_offsets(
                self._old_to_new_facet_numbering, self._interior_facet_plex_indices, sort=True
            )
            arity = 2

        # Remove ghost indices
        new_selected_facets_is = dmcommon.filter_is(selected_facets_is, 0, self.facets.owned.local_size)
        selected_facets = new_selected_facets_is.indices
        assert selected_facets.size == facet_axis.local_size

        mysubset = op3.Slice(
            facet_support_dat.axes.root.label,
            [
                op3.Subset(
                    facet_support_dat.axes.root.component.label,
                    op3.Dat.from_array(selected_facets),
                    label=facet_axis.component.label,
                )
            ],
            label=facet_axis.label,
        )

        *others, (leaf_axis_label, leaf_component_label) = facet_support_dat.axes.leaf_path.items()
        myslice = op3.Slice(leaf_axis_label, [op3.AffineSliceComponent(leaf_component_label, stop=arity)], label="support")

        # TODO: This should ideally work
        # return facet_support_dat[mysubset, slice(arity)]
        specialized_by_type_facet_support_dat = facet_support_dat[mysubset, myslice]
        assert specialized_by_type_facet_support_dat.axes.local_size == facet_axis.local_size * arity
        return specialized_by_type_facet_support_dat


    # delete?
    def create_section(self, nodes_per_entity, real_tensorproduct=False, block_size=1):
        """Create a PETSc Section describing a function space.

        :arg nodes_per_entity: number of function space nodes per topological entity.
        :arg real_tensorproduct: If True, assume extruded space is actually Foo x Real.
        :arg block_size: The integer by which nodes_per_entity is uniformly multiplied
            to get the true data layout.
        :arg boundary_set: A set of boundary markers, indicating the subdomains
            a boundary condition is specified on.
        :returns: a new PETSc Section.
        """
        return dmcommon.create_section(self, nodes_per_entity, on_base=real_tensorproduct, block_size=block_size)

    # delete?
    def node_classes(self, nodes_per_entity, real_tensorproduct=False):
        """Compute node classes given nodes per entity.

        :arg nodes_per_entity: number of function space nodes per topological entity.
        :returns: the number of nodes in each of core, owned, and ghost classes.
        """
        return tuple(np.dot(nodes_per_entity, self._entity_classes))

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
        assert False, "old code"
        return cell_data[column_list]

    @property
    @abc.abstractmethod
    def num_points(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def num_cells(self):
        pass

    @property
    @abc.abstractmethod
    def num_facets(self):
        pass

    @property
    @abc.abstractmethod
    def num_faces(self):
        pass

    @property
    @abc.abstractmethod
    def num_edges(self):
        pass

    @property
    @abc.abstractmethod
    def num_vertices(self):
        pass

    @abc.abstractmethod
    def entity_count(self, dim):
        pass

    def size(self, depth):
        return self.num_entities(depth)

    def cell_dimension(self):
        """Returns the cell dimension."""
        return self.ufl_cell().topological_dimension

    def facet_dimension(self):
        """Returns the facet dimension."""
        # Facets have co-dimension 1
        return self.ufl_cell().topological_dimension - 1

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
        return self.periodic

    @cached_property
    def _plex_closures(self) -> dict[Any, np.ndarray]:
        # TODO: Provide more detail about the return type
        """Memoized DMPlex point closures with default numbering.

        Returns
        -------
        tuple :
            Closure data per dimension.

        """
        # TODO: make memoize_closures nicer to reuse code
        # NOTE: At the moment this only works for cells because I don't know how to
        # compute closure sizes elsewise
        return {self.dimension: self._memoize_closures(self.dimension)}

    @cached_property
    def _plex_closures_renumbered(self) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    @cached_property
    def _plex_closures_localized(self) -> tuple[tuple[np.ndarray, ...], ...]:
        raise NotImplementedError

    @cached_property
    def _fiat_cell_closures(self) -> np.ndarray:
        """

        Reorders verts -> cell from cell -> verts

        """
        plex_closures = self._plex_closures[self.dimension]

        if (
            self.submesh_parent is not None
            and not (
                self.submesh_parent.ufl_cell().cellname == "hexahedron"
                and self.ufl_cell().cellname == "quadrilateral"
            )
            and len(self.submesh_parent.dm_cell_types) == 1
        ):
            # Codim-1 submesh of a hex mesh (i.e. a quad submesh) can not
            # inherit cell_closure from the hex mesh as the cell_closure
            # must follow the special orientation restriction. This means
            # that, when the quad submesh works with the parent hex mesh,
            # quadrature points must be permuted (i.e. use the canonical
            # quadrature point ordering based on the cone ordering).
            topology = FIAT.ufc_cell(self.ufl_cell()).get_topology()
            entity_per_cell = np.zeros(len(topology), dtype=IntType)
            for d, ents in topology.items():
                entity_per_cell[d] = len(ents)
            return dmcommon.submesh_create_cell_closure(
                self.topology_dm,
                self.submesh_parent.topology_dm,
                self._old_to_new_cell_numbering,  # not used
                self.submesh_parent._old_to_new_cell_numbering,  # not used
                self.submesh_parent._fiat_cell_closures,
                entity_per_cell,
            )

        elif self.ufl_cell().is_simplex:
            return self._reorder_closure_fiat_simplex(plex_closures)

        elif self.ufl_cell() == ufl.quadrilateral:
            return self._reorder_closure_fiat_quad(plex_closures)

        else:
            assert self.ufl_cell() == ufl.hexahedron
            return self._reorder_closure_fiat_hex(plex_closures)

    @cached_property
    def _fiat_cell_closures_renumbered(self) -> np.ndarray:
        renumbered_closures = np.empty_like(self._fiat_cell_closures)
        from_dim = self.dimension
        offset = 0
        for to_dim, size in self._closure_sizes[from_dim].items():
            start = offset
            stop = offset + size
            renumbered_closures[:, start:stop] = self._renumber_map(
                self._fiat_cell_closures[:, start:stop],
                from_dim,
                to_dim,
            )
            offset += size
        return renumbered_closures

    @property
    def _fiat_cell_closures_localized(self):
        # NOTE: Now a bad name, this doesn't localize but it does put into a dict
        localized_closures = {}
        from_dim = self.dimension
        offset = 0
        for to_dim, size in self._closure_sizes[from_dim].items():
            localized_closures[to_dim] = self._fiat_cell_closures_renumbered[:, offset:offset+size]
            offset += size
        return idict(localized_closures)

    def _memoize_closures(self, dim) -> np.ndarray:
        def closure_func(_pt):
            return self.topology_dm.getTransitiveClosure(_pt)[0]

        p_start, p_end = self.topology_dm.getDepthStratum(dim)
        npoints = p_end - p_start
        closure_size = sum(self._closure_sizes[dim].values())
        closure_data = np.empty((npoints, closure_size), dtype=IntType)

        for i, pt in enumerate(range(p_start, p_end)):
            closure_data[i] = closure_func(pt)

        return utils.readonly(closure_data)

    def __iter__(self):
        yield self

    def unique(self):
        return self

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

        return tuple(maps), integral_type, subset_points

    # trans mesh

    def trans_mesh_entity_map(self, iteration_spec):
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

    @cached_property
    def _entity_indices(self):
        indices = []
        renumbering = self._old_to_new_point_renumbering.indices
        for dim in range(self.dimension+1):
            p_start, p_end = self.topology_dm.getDepthStratum(dim)
            indices.append(readonly(np.sort(renumbering[p_start:p_end])))
        return tuple(indices)

    # @cached_property
    # def _closure_sizes(self) -> dict:
    #     # Determine the closure size for the given dimension. For triangles
    #     # this would be:
    #     #
    #     #     (1, 0, 0) if dim == 0 (vertex)
    #     #     (2, 1, 0) if dim == 1 (edge)
    #     #     (3, 3, 1) if dim == 2 (cell)
    #     sizes = collections.defaultdict(list)
    #     for dim in range(self.dimension+1):
    #         cell_connectivity = as_fiat_cell(self.ufl_cell()).connectivity
    #         for d in range(dim+1):
    #             # This tells us the points with dimension d that lie in the closure
    #             # of the different points with dimension dim. We just want to know
    #             # how many there are (e.g. each edge is connected to 2 vertices).
    #             closures = cell_connectivity[dim, d]
    #             sizes[dim].append(single_valued(map(len, closures)))
    #     return sizes


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

    # @property
    # def cell_label(self) -> int:
    #     return self.dimension
    #
    # # should error
    # @property
    # def facet_label(self):
    #     return str(self.dimension - 1)
    #
    # # should error
    # @property
    # def edge_label(self):
    #     return "1"
    #
    # # TODO I prefer "vertex_label"
    # @property
    # def vert_label(self):
    #     return "0"

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

    @utils.cached_property
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
    def strata_offsets(self) -> tuple[int, ...]:
        return tuple(
            self.topology_dm.getDepthStratum(dim)[0]
            for dim in range(self.dimension+1)
        )

    @cached_property
    def entity_orientations(self):
        return dmcommon.entity_orientations(self, self._fiat_cell_closures)[self._new_to_old_cell_numbering]

    @cached_property
    def entity_orientations_dat(self):
        # FIXME: the following does not work because the labels change
        cell_axis = self.cells.root
        # # so instead we do
        # cell_axis = op3.Axis([self.points.root.components[0]], self.points.root.label)

        # TODO: This is quite a funky way of getting this. We should be able to get
        # it without calling the map.
        closure_axis = self.closure(self.cells.iter()).axes.root
        axis_tree = op3.AxisTree.from_nest({cell_axis: [closure_axis]})
        assert axis_tree.local_size == self.entity_orientations.size
        return op3.Dat(axis_tree, data=self.entity_orientations.flatten(), prefix="orientations")

    @cached_property
    def local_cell_orientation_dat(self):
        return self.entity_orientations_dat[:, op3.as_slice(self.cell_label)]
        # return op2.Dat(
        #     op2.DataSet(self.cell_set, 1),
        #     self.entity_orientations[:, [-1]],
        #     gem.uint_type,
        #     f"{self.name}_local_cell_orientation"
        # )

    def _memoize_map(self, map_func, dim, sizes=None):
        if sizes is not None:
            return self._memoize_map_fixed(map_func, dim, sizes), sizes
        else:
            return _memoize_map_ragged(self.topology_dm, dim, map_func)

    def _memoize_map_fixed(self, map_func, dim, sizes):
        pstart, pend = self.topology_dm.getDepthStratum(dim)
        npoints = pend - pstart

        map_data = tuple(
            np.empty((npoints, sizes[d]), dtype=IntType)
            for d in range(self.dimension+1)
        )

        for pt in range(pstart, pend):
            stratum_pt = pt - pstart

            map_pts = iter(map_func(pt))
            for map_dim in reversed(range(self.dimension+1)):
                for i in range(sizes[map_dim]):
                    map_pt = next(map_pts)
                    map_data[map_dim][stratum_pt, i] = map_pt
            utils.assert_empty(map_pts)
        return map_data

    @cached_property
    @collective
    def facet_markers(self) -> np.ndarray[IntType, ...]:
        # The IS returned by 'getLabelIdIS' exists on COMM_SELF so if we want
        # a collective IS we must convert to COMM_WORLD before calling 'allGather'.
        local_facet_markers_is = self.topology_dm.getLabelIdIS(dmcommon.FACE_SETS_LABEL)
        global_facet_markers_is = PETSc.IS().createGeneral(
            local_facet_markers_is.indices, comm=MPI.COMM_WORLD
        ).allGather()
        return utils.readonly(np.unique(np.sort(global_facet_markers_is.indices)))

    @cached_property
    def exterior_facets(self) -> op3.IndexedAxisTree:
        subset = self._facet_subset(self._exterior_facet_plex_indices, self._old_to_new_facet_numbering, self.facet_label)
        return self.points[subset]

    @cached_property
    def interior_facets(self) -> op3.IndexedAxisTree:
        subset = self._facet_subset(self._interior_facet_plex_indices, self._old_to_new_facet_numbering, self.facet_label)
        return self.points[subset]

    # TODO: typing for component_label
    # Maybe doesn't have to be a method either
    def _facet_subset(self, plex_indices_is: PETSc.IS, component_renumbering: PETSc.Section, component_label) -> op3.Slice:
        subset_indices = dmcommon.section_offsets(component_renumbering, plex_indices_is, sort=True)
        subset_dat = op3.Dat.from_array(subset_indices.indices)
        return op3.Slice(self.name, [op3.Subset(component_label, subset_dat)])

    @cached_property
    def _exterior_facet_strata_indices_plex(self) -> np.ndarray[IntType]:
        return self._facet_strata_indices_plex("exterior")

    @cached_property
    def _interior_facet_strata_indices_plex(self) -> np.ndarray[IntType]:
        return self._facet_strata_indices_plex("interior")

    def _facet_strata_indices_plex(self,facet_type: Literal["exterior"] | Literal["interior"]) -> np.ndarray[IntType]:
        if facet_type == "exterior":
            label_value = "exterior_facets"
        else:
            assert facet_type == "interior"
            label_value = "interior_facets"
        indices_plex = dmcommon.facets_with_label(self, label_value)
        f_start, _ = self.topology_dm.getDepthStratum(self.dimension-1)
        return utils.readonly(indices_plex - f_start)

    # def _facet_numbers_classes_set(self, kind):
    #     if kind not in ["interior", "exterior"]:
    #         raise ValueError("Unknown facet type '%s'" % kind)
    #     # Can not call target.{interior, exterior}_facets.facets
    #     # if target is a mixed cell mesh (cell_closure etc. can not be defined),
    #     # so directly call dmcommon.get_facets_by_class.
    #     _numbers, _classes = dmcommon.get_facets_by_class(self.topology_dm, (kind + "_facets"), self._facet_ordering)
    #     _classes = as_tuple(_classes, int, 3)
    #     _set = op2.Set(_classes, f"{kind.capitalize()[:3]}Facets", comm=self.comm)
    #     return _numbers, _classes, _set

    # @cached_property
    # def _exterior_facet_numbers_classes_set(self):
    #     return self._facet_numbers_classes_set("exterior")
    #
    # @cached_property
    # def _interior_facet_numbers_classes_set(self):
    #     return self._facet_numbers_classes_set("interior")
    #
    # @cached_property
    # def _facet_ordering(self):
    #     return dmcommon.get_facet_ordering(self.topology_dm, self._old_to_new_facet_numbering)

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
                                                   self._old_to_new_cell_numbering,
                                                   self.cell_closure)
        axes = op3.AxisTree.from_iterable([self.cells.root, *cell_facets.shape[1:]])
        return op3.Dat(axes, data=cell_facets, name="cell-to-local-facet-dat")

    @cached_property
    def cell_closure(self):
        # old attribute, keeping around for now
        return self._fiat_cell_closures[self._new_to_old_cell_numbering]

    @property
    def dimension(self):
        return self.topology_dm.getDimension()

    @property
    def num_points(self) -> int:
        start, end = self.topology_dm.getChart()
        assert start == 0
        return end

    @property
    def num_owned_points(self) -> int:
        return self.num_points - self.num_ghost_points

    @property
    def num_ghost_points(self) -> int:
        return self.topology_dm.getLabel("firedrake_is_ghost").getStratumSize(1)

    @property
    def num_cells(self) -> int:
        return self.entity_count(self.dimension)

    @property
    def num_facets(self) -> int:
        return self.entity_count(self.dimension - 1)

    @property
    def num_faces(self):
        return self.entity_count(2)

    @property
    def num_edges(self):
        return self.entity_count(1)

    @property
    def num_vertices(self):
        return self.entity_count(0)

    def entity_count(self, dim):
        p_start, p_end = self.topology_dm.getDepthStratum(dim)
        num_points = p_end - p_start

        # if not include_ghost_points:
        #     ghost_label = self.topology_dm.getLabel("firedrake_is_ghost")
        #     ghost_indices = ghost_label.getStratumIS(1).indices
        #     # TODO: This is what ISGeneralFilter() does, but that is not exposed in petsc4py
        #     # https://petsc.org/release/manualpages/IS/ISGeneralFilter/
        #     num_ghost_points = sum((p_start <= ghost_indices) & (ghost_indices < p_end))
        #     num_points -= num_ghost_points

        return num_points

    @property
    def cell_label(self) -> int:
        return self.dimension

    @property
    def facet_label(self) -> int:
        return self.dimension - 1

    @property
    def edge_label(self) -> int:
        return 1

    @property
    def vertex_label(self) -> int:
        return 0

    @cached_property
    def cells(self) -> op3.IndexedAxisTree:
        # TODO: Implement and use 'FullComponentSlice' (or similar)
        cell_slice = op3.Slice(self.name, [op3.AffineSliceComponent(self.cell_label, label=self.cell_label)], label=self.name)
        return self.points[cell_slice]

    @cached_property
    def facets(self):
        return self.points[self.facet_label]

    @cached_property
    def vertices(self):
        return self.points[self.vertex_label]

    @property
    @utils.deprecated("cells.owned")
    def cell_set(self):
        return self.cells.owned

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

    def _submesh_make_entity_entity_map(self, from_set, to_set, from_points, to_points, from_numbering, to_numbering, child_parent_map):
        assert from_set.local_size == len(from_points)
        assert to_set.local_size == len(to_points)
        # this always maps from child plex point to parent plex point
        if child_parent_map:
            # this is a dense map from the child points to the parent points
            plex_index_map = self._submesh_to_parent_plex_index_map
        else:
            plex_index_map = self._parent_to_submesh_plex_index_map

        subpoints = plex_index_map[from_points]
        values = dmcommon.renumber_map_fixed(
            from_points,
            subpoints[:, np.newaxis],  # arity 1 map between plex points
            from_numbering,
            to_numbering,
        )
        map_name = f"{self.name}_submesh_map_{from_set.root.label}_{to_set.root.label}"
        to_label = to_set.as_axis().component.label
        map_axes = op3.AxisTree.from_iterable([
            from_set.as_axis(), op3.Axis(1, map_name)
        ])
        map_dat = op3.Dat(map_axes, data=values.flatten())
        return op3.Map(
            {
                from_set.leaf_path: [[
                    op3.TabulatedMapComponent(to_set.as_axis().label, to_label, map_dat, label=to_label),
                ]],
            },
            name=map_name,
        )

    @cached_property
    def submesh_child_cell_parent_cell_map(self):
        return self._submesh_make_entity_entity_map(
            self.cells,
            self.submesh_parent.cells,
            PETSc.IS().createStride(self.num_cells, comm=MPI.COMM_SELF).indices,
            PETSc.IS().createStride(self.submesh_parent.num_cells, comm=MPI.COMM_SELF).indices,
            self._old_to_new_cell_numbering,
            self.submesh_parent._old_to_new_cell_numbering,
            True,
        )

    @cached_property
    def submesh_child_exterior_facet_parent_exterior_facet_map(self):
        return self._submesh_make_entity_entity_map(
            self.exterior_facets,
            self.submesh_parent.exterior_facets,
            self._exterior_facet_plex_indices.indices,
            self.submesh_parent._exterior_facet_plex_indices.indices,
            self._old_to_new_exterior_facet_numbering,
            self.submesh_parent._old_to_new_exterior_facet_numbering,
            True,
        )

    @cached_property
    def submesh_child_exterior_facet_parent_interior_facet_map(self):
        return self._submesh_make_entity_entity_map(
            self.exterior_facets,
            self.submesh_parent.interior_facets,
            self._exterior_facet_plex_indices.indices,
            self.submesh_parent._interior_facet_plex_indices.indices,
            self._old_to_new_exterior_facet_numbering,
            self.submesh_parent._old_to_new_interior_facet_numbering,
            True,
        )

    @cached_property
    def submesh_child_interior_facet_parent_exterior_facet_map(self):
        raise RuntimeError("Should never happen")

    @cached_property
    def submesh_child_interior_facet_parent_interior_facet_map(self):
        return self._submesh_make_entity_entity_map(
            self.interior_facets,
            self.submesh_parent.interior_facets,
            self._interior_facet_plex_indices.indices,
            self.submesh_parent._interior_facet_plex_indices.indices,
            self._old_to_new_interior_facet_numbering,
            self.submesh_parent._old_to_new_interior_facet_numbering,
            True,
        )

    @cached_property
    def submesh_child_cell_parent_interior_facet_map(self):
        return self._submesh_make_entity_entity_map(
            self.cells,
            self.submesh_parent.interior_facets,
            PETSc.IS().createStride(self.num_cells, comm=MPI.COMM_SELF).indices,
            self.submesh_parent._interior_facet_plex_indices.indices,
            self._old_to_new_cell_numbering,
            self.submesh_parent._old_to_new_interior_facet_numbering,
            True,
        )

    @cached_property
    def submesh_child_cell_parent_exterior_facet_map(self):
        return self._submesh_make_entity_entity_map(
            self.cells,
            self.submesh_parent.exterior_facets,
            self._new_to_old_cell_numbering,
            self.submesh_parent._exterior_facet_plex_indices.indices,
            True,
        )

    @cached_property
    def submesh_parent_cell_child_cell_map(self):
        return self._submesh_make_entity_entity_map(
            self.submesh_parent.cells,
            self.cells,
            PETSc.IS().createStride(self.submesh_parent.num_cells, comm=MPI.COMM_SELF).indices,
            PETSc.IS().createStride(self.num_cells, comm=MPI.COMM_SELF).indices,
            self.submesh_parent._old_to_new_cell_numbering,
            self._old_to_new_cell_numbering,
            False,
        )

    @cached_property
    def submesh_parent_exterior_facet_child_exterior_facet_map(self):
        return self._submesh_make_entity_entity_map(
            self.submesh_parent.exterior_facets,
            self.exterior_facets,
            self.submesh_parent._exterior_facet_plex_indices.indices,
            self._exterior_facet_plex_indices.indices,
            self.submesh_parent._old_to_new_exterior_facet_numbering,
            self._old_to_new_exterior_facet_numbering,
            False,
        )

    @cached_property
    def submesh_parent_exterior_facet_child_interior_facet_map(self):
        raise RuntimeError("Should never happen")

    @cached_property
    def submesh_parent_interior_facet_child_exterior_facet_map(self):
        return self._submesh_make_entity_entity_map(
            self.submesh_parent.interior_facets,
            self.exterior_facets,
            self.submesh_parent._interior_facet_plex_indices.indices,
            self._exterior_facet_plex_indices.indices,
            self.submesh_parent._old_to_new_interior_facet_numbering,
            self._old_to_new_exterior_facet_numbering,
            False,
        )

    @cached_property
    def submesh_parent_interior_facet_child_interior_facet_map(self):
        return self._submesh_make_entity_entity_map(
            self.submesh_parent.interior_facets,
            self.interior_facets,
            self.submesh_parent._interior_facet_plex_indices.indices,
            self._interior_facet_plex_indices.indices,
            self.submesh_parent._old_to_new_interior_facet_numbering,
            self._old_to_new_interior_facet_numbering,
            False,
        )

    @cached_property
    def submesh_parent_exterior_facet_child_cell_map(self):
        return self._submesh_make_entity_entity_map(
            self.submesh_parent.exterior_facets,
            self.cells,
            self.submesh_parent._exterior_facet_plex_indices.indices,
            PETSc.IS().createStride(self.num_cells, comm=MPI.COMM_SELF).indices,
            self.submesh_parent._old_to_new_exterior_facet_numbering,
            self._old_to_new_cell_numbering,
            False,
        )

    @cached_property
    def submesh_parent_interior_facet_child_cell_map(self):
        return self._submesh_make_entity_entity_map(
            self.submesh_parent.interior_facets,
            self.cells,
            self.submesh_parent._interior_facet_plex_indices.indices,
            PETSc.IS().createStride(self.num_cells, comm=MPI.COMM_SELF).indices,
            self.submesh_parent._old_to_new_interior_facet_numbering,
            self._old_to_new_cell_numbering,
            False,
        )

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
            # NOTE: we don't really use target_subset_points at all...
            if reverse:
                target_subset_points = self._parent_to_submesh_plex_index_map[source_subset_points]
            else:
                target_subset_points = self._submesh_to_parent_plex_index_map[source_subset_points]
            target_integral_type = "cell"

        elif target_integral_type_temp == "facet":
            if reverse:
                target_subset_points = self._parent_to_submesh_plex_index_map[source_subset_points]
            else:
                target_subset_points = self._submesh_to_parent_plex_index_map[source_subset_points]

            # It is possible for an exterior facet integral on the submesh to correspond to
            # and interior facet integral on the parent mesh (but never the other way around
            # or to a mix of facet types).
            # We don't know a priori what this type is so we instead detect it here.
            target_exterior_facets = dmcommon.intersect_is(
                PETSc.IS().createGeneral(target_subset_points),
                target._exterior_facet_plex_indices,
            )
            with temp_internal_comm(self.comm) as icomm:
                includes_exterior_facets = icomm.allreduce(
                    target_exterior_facets.size>0, MPI.LOR
                )

            target_interior_facets = dmcommon.intersect_is(
                PETSc.IS().createGeneral(target_subset_points),
                target._interior_facet_plex_indices,
            )
            with temp_internal_comm(self.comm) as icomm:
                includes_interior_facets = icomm.allreduce(
                    target_interior_facets.size>0, MPI.LOR
                )

            if includes_exterior_facets and includes_interior_facets:
                raise RuntimeError(f"Attempting to target a mix of interior and exterior facets")
            elif includes_exterior_facets:
                target_integral_type = "exterior_facet"
            elif includes_interior_facets:
                target_integral_type = "interior_facet"
            else:
                # should this ever happen? and could we just continue with an empty set if so?
                raise RuntimeError("Can not find a map from source to target.")
        else:
            raise NotImplementedError
        if reverse:
            map_ = getattr(self, f"submesh_parent_{source_integral_type}_child_{target_integral_type}_map")
        else:
            map_ = getattr(self, f"submesh_child_{source_integral_type}_parent_{target_integral_type}_map")
        return map_, target_integral_type, target_subset_points

    # trans mesh

    @cached_property
    def _submesh_to_parent_plex_index_map(self) -> np.ndarray[IntType]:
        return self.topology_dm.getSubpointIS().indices

    @cached_property
    def _parent_to_submesh_plex_index_map(self) -> np.ndarray[IntType]:
        """

        Points that are not present in ``self`` are given as '-1's.

        """
        submesh_to_parent_map = self._submesh_to_parent_plex_index_map
        parent_to_submesh_map = np.full(self.submesh_parent.num_points, -1, dtype=IntType)
        parent_to_submesh_map[submesh_to_parent_map] = np.arange(submesh_to_parent_map.size, dtype=IntType)
        return parent_to_submesh_map

    def trans_mesh_entity_map(self, iter_spec):
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
        base_mesh = iter_spec.mesh
        base_integral_type = iter_spec.integral_type

        if plex_indices_is := iter_spec.plex_indices:
            base_plex_points = plex_indices_is.indices
        else:
            base_plex_points = np.arange(iter_spec.iterset.local_size, dtype=IntType)

        common = self.submesh_youngest_common_ancester(base_mesh)
        if common is None:
            raise NotImplementedError(f"Currently only implemented for (sub)meshes in the same family: got {self} and {base_mesh}")
        elif base_mesh is self:
            raise NotImplementedError("Currently cannot return identity map")
        # else:
        #     if base_integral_type == "cell":
        #         base_subset_points = base_mesh._new_to_old_cell_numbering[base_indices]
        #     elif base_integral_type in ["interior_facet", "exterior_facet"]:
        #         if base_integral_type == "interior_facet":
        #             # IMPORTANT: This probably makes a renumbering error!
        #             # _interior_facet_numbers, _, _ = base_mesh._interior_facet_numbers_classes_set
        #             # base_subset_points = base_mesh._new_to_old_interior_facet_numbering[base_indices]
        #             base_subset_points = base_mesh._interior_facet_plex_indices.indices[base_indices]
        #             # base_subset_points = _interior_facet_numbers[base_indices]
        #         else:
        #             assert base_integral_type == "exterior_facet"
        #             base_subset_points = base_mesh._exterior_facet_plex_indices.indices[base_indices]
        #             # _exterior_facet_numbers, _, _ = base_mesh._exterior_facet_numbers_classes_set
        #             # base_subset_points = _exterior_facet_numbers[base_indices]
        #     else:
        #         raise NotImplementedError(f"Unknown integration type : {base_integral_type}")
        composed_map, integral_type, _ = self.submesh_map_composed(base_mesh, base_integral_type, base_plex_points)
        # poor man's reduce
        # return self_map(reduce(operator.call, composed_map, iteration_spec.loop_index))
        map_ = iter_spec.loop_index
        for map2 in composed_map:
            map_ = map2(map_)
        return map_, integral_type


# NOTE: I don't think that we need an extra class here. The public API is exactly the same as 'MeshTopology'.
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
        self._cache = self._shared_data_cache  # alias, yuck

        if not isinstance(layers, numbers.Integral):
            raise TypeError("Variable layer extrusion is no longer supported")

        if isinstance(mesh.topology, VertexOnlyMeshTopology):
            raise NotImplementedError("Extrusion not implemented for VertexOnlyMeshTopology")

        self._base_mesh = mesh
        self.layers = layers
        self.user_comm = mesh.comm
        if name is not None and name == mesh.name:
            raise ValueError("Extruded mesh topology and base mesh topology can not have the same name")
        self.name = name if name is not None else mesh.name + "_extruded"

        # TODO: These attributes are copied so that FunctionSpaceBase can
        # access them directly.  Eventually we would want a better refactoring
        # of responsibilities between mesh and function space.
        # self.topology_dm = mesh.topology_dm
        base_dm = mesh.topology_dm.clone()
        # base_dm.removeLabel(dmcommon.FACE_SETS_LABEL)
        # base_dm.removeLabel("exterior_facets")
        # base_dm.removeLabel("interior_facets")
        self.topology_dm = dmcommon.extrude_mesh(base_dm, layers-1, 666, periodic=periodic)
        self.topology_dm.getLabel("exterior_facets").setName("base_exterior_facets")
        self.topology_dm.getLabel("interior_facets").setName("base_interior_facets")
        r"The PETSc DM representation of the mesh topology."

        self._distribution_parameters = mesh._distribution_parameters
        self._subsets = {}
        self.periodic = periodic
        # submesh
        self.submesh_parent = None

    @utils.cached_property
    def _ufl_cell(self):
        return ufl.TensorProductCell(self._base_mesh.ufl_cell(), ufl.interval)

    @utils.cached_property
    def _ufl_mesh(self):
        cell = self._ufl_cell
        return ufl.Mesh(finat.ufl.VectorElement("Lagrange", cell, 1, dim=cell.topological_dimension))

    @property
    def dm_cell_types(self):
        """All DM.PolytopeTypes of cells in the mesh."""
        raise NotImplementedError("'dm_cell_types' is not implemented for ExtrudedMeshTopology")

    @utils.cached_property
    def flat_points(self):
        n_extr_cells = int(self.layers) - 1

        column_height = 2 * n_extr_cells
        if not self.periodic:
            column_height += 1

        base_mesh_axis = self._base_mesh.flat_points
        npoints = base_mesh_axis.component.local_size * column_height

        # NOTE: In serial the point SF isn't set up in a valid state so we do this. It
        # would be nice to avoid this branch.
        if self.comm.size > 1:
            point_sf = self.topology_dm.getPointSF()
        else:
            point_sf = op3.local_sf(self.num_points, self.comm).sf

        point_sf_renum = op3.sf.renumber_petsc_sf(point_sf, self._new_to_old_point_renumbering)
        point_sf_renum = op3.StarForest(point_sf_renum, self.comm)

        return op3.Axis(
            [op3.AxisComponent(npoints, "mylabel", sf=point_sf_renum)],
            label="mesh",
        )

    @property
    def cell_label(self):
        return (self._base_mesh.cell_label, 1)

    @property
    def facet_label(self):
        raise TypeError("Extruded meshes do not have a unique facet label")

    @property
    def facet_horiz_label(self):
        return (self._base_mesh.cell_label, 0)

    @property
    def facet_vert_label(self):
        return (self._base_mesh.facet_label, 1)

    @property
    def edge_label(self):
        raise NotImplementedError

    @property
    def vert_label(self) -> tuple:
        return (self._base_mesh.vert_label, 0)

    @property
    def num_cells(self) -> int:
        nlayers = int(self.layers) - 1
        return self._base_mesh.num_cells * nlayers

    @property
    def num_facets(self) -> int:
        assert False, "hard"

    @property
    def num_faces(self):
        assert False, "hard"

    @property
    def num_edges(self):
        assert False, "hard"

    @property
    def num_vertices(self):
        nlayers = int(self.layers) - 1
        return self._base_mesh.num_vertices * (nlayers+1)

    @cached_property
    def _new_to_old_point_renumbering(self) -> PETSc.IS:
        return self._old_to_new_point_renumbering.invertPermutation()

    @utils.cached_property
    def _old_to_new_point_renumbering(self) -> PETSc.IS:
        """
        Consider

              x-----x-----x
              2  0  3  1  4
             (1  0  2  3  4)   going to

        When we extrude it will have the following numbering:

              5--2--8-11-14
              |     |     |
              4  1  7 10 13
              |     |     |
              3--0--6--9-12

        whilst the DMPlex will think it is:

              3--9--5-11--7
              |     |     |
             12  0 13  1 14
              |     |     |
              2--8--4-10--6

        (To see this recall that points are numbered cells then vertices then edges.)

        """
        n_extr_cells = int(self.layers) - 1

        # we always have 2n+1 entities when we extrude
        base_indices = self._base_mesh._old_to_new_point_renumbering.indices
        base_point_label = self.topology_dm.getLabel("base_point")

        column_height = 2*n_extr_cells
        if not self.periodic:
            column_height += 1
        indices = np.empty(base_indices.size * column_height, dtype=base_indices.dtype)
        for base_dim in range(self._base_mesh.topology_dm.getDimension()+1):
            cell_stratum = self.topology_dm.getDepthStratum(base_dim+1)
            vert_stratum = self.topology_dm.getDepthStratum(base_dim)
            for base_pt in range(*self._base_mesh.topology_dm.getDepthStratum(base_dim)):
                extruded_points = base_point_label.getStratumIS(base_pt)
                extruded_cells = dmcommon.filter_is(extruded_points, *cell_stratum)
                extruded_verts = dmcommon.filter_is(extruded_points, *vert_stratum)
                if self.periodic:
                    assert extruded_verts.size == extruded_cells.size
                else:
                    assert extruded_verts.size == extruded_cells.size + 1

                for i, ec in enumerate(extruded_cells.indices):
                    indices[ec] = base_indices[base_pt] * column_height + (2*i+1)

                for i, ev in enumerate(extruded_verts.indices):
                    indices[ev] = base_indices[base_pt] * column_height + 2*i

        return PETSc.IS().createGeneral(indices, comm=MPI.COMM_SELF)

    @cached_property
    def _entity_indices(self):
        # First get the indices of the right entity type. This is more complicated
        # for extruded meshes because the different facet types are not natively
        # distinguished.
        indices = {}
        base_dim_label = self.topology_dm.getLabel("base_dim")
        for base_dim in range(self._base_mesh.dimension+1):
            # Get all points that were originally a vertex, say
            matching_base_dim_extruded_points = base_dim_label.getStratumIS(base_dim)
            matching_base_dim_extruded_points.toGeneral()

            for extr_dim in range(2):
                # Filter out the extruded dimension that we don't want
                matching_extruded_points = dmcommon.filter_is(
                    matching_base_dim_extruded_points,
                    *self.topology_dm.getDepthStratum(base_dim+extr_dim),
                )
                # Finally do the renumbering
                indices[(base_dim, extr_dim)] = utils.readonly(
                    np.sort(self._old_to_new_point_renumbering.indices[matching_extruded_points.indices])
                )
        return indices

    # TODO: I don't think that the specific ordering actually matters here...
    @property
    def _plex_strata_ordering(self):
        return tuple(
            (base_dim, extr_dim)
            for base_dim in self._base_mesh._plex_strata_ordering
            for extr_dim in range(2)
        )

    @cached_property
    def entity_orientations(self):
        # As an example, consider extruding a single-cell interval mesh:
        #
        #     x-----x-----x
        #    o1    o3    o2
        #
        # where 'o1', 'o2', and 'o3' are the orientations of the points in the
        # cell closure. Note that we are ignoring the fact that vertices only
        # have a single orientation.
        #
        # If we extrude this mesh once then we have a new cell with the following
        # orientations:
        #
        #    o1    o3    o2
        #     x-----------x
        #     |           |
        #  o1 |    o3     | o2
        #     |           |
        #     x-----------x
        #    o1    o3    o2
        #
        # The base mesh here has 'entity_orientations' as [o1, o2, o3] but we
        # need the extruded counterpart which looks like:
        #
        #     [ o1, o1, o2, o2 | o1, o2 | o3, o3 |  o3 ]
        #           (0, 0)       (0, 1)   (1, 0)  (1, 1)
        orientationss = []
        base_closure_sizes = self._base_mesh._closure_sizes[self._base_mesh.cell_label]
        base_orientations = self._base_mesh.entity_orientations
        start = 0
        for base_dim, closure_size in base_closure_sizes.items():
            base_entity_selector = slice(start, start+closure_size)

            vert_orientations = (
                np.repeat(base_orientations[:, base_entity_selector], 2).reshape((-1, closure_size*2))
            )
            edge_orientations = base_orientations[:, base_entity_selector]
            orientationss.extend([vert_orientations, edge_orientations])

            start += closure_size
        orientationss = np.concatenate(orientationss, axis=1)

        # We now have the orientation for a single extruded cell, now blow this
        # up for the whole column
        return np.repeat(orientationss, self.layers-1, axis=0)

    # {{{ facet iteration

    @cached_property
    def exterior_facets(self) -> NoReturn:
        raise TypeError(
            "Cannot use 'exterior_facets' for extruded meshes, use 'exterior_facets_vert', "
            "'exterior_facets_top' or 'exterior_facets_bottom' instead"
        )

    @cached_property
    def interior_facets(self) -> NoReturn:
        raise TypeError(
            "Cannot use 'interior_facets' for extruded meshes, use 'interior_facets_vert' "
            "or 'interior_facets_horiz instead"
        )

    @cached_property
    def exterior_facets_top(self) -> op3.IndexedAxisTree:
        subset = self._facet_subset(
            self._exterior_facet_top_plex_indices,
            self._old_to_new_facet_horiz_numbering,
            self.facet_horiz_label,
        )
        return self.points[subset]

    @cached_property
    def exterior_facets_bottom(self) -> op3.IndexedAxisTree:
        subset = self._facet_subset(
            self._exterior_facet_bottom_plex_indices,
            self._old_to_new_facet_horiz_numbering,
            self.facet_horiz_label,
        )
        return self.points[subset]

    @cached_property
    def exterior_facets_vert(self) -> op3.IndexedAxisTree:
        subset = self._facet_subset(
            self._exterior_facet_vert_plex_indices,
            self._old_to_new_facet_vert_numbering,
            self.facet_vert_label,
        )
        return self.points[subset]

    @cached_property
    def interior_facets_horiz(self) -> op3.IndexedAxisTree:
        subset = self._facet_subset(
            self._interior_facet_horiz_plex_indices,
            self._old_to_new_facet_horiz_numbering,
            self.facet_horiz_label,
        )
        return self.points[subset]

    @cached_property
    def interior_facets_vert(self) -> op3.IndexedAxisTree:
        subset = self._facet_subset(
            self._interior_facet_vert_plex_indices,
            self._old_to_new_facet_vert_numbering,
            self.facet_vert_label,
        )
        return self.points[subset]

    @cached_property
    def _exterior_facet_vert_plex_indices(self) -> PETSc.IS:
        # Consider extruding the following interval mesh:
        #
        #     E-----I-----E
        #
        # to
        #
        #     x--E--x--E--x
        #     |     |     |
        #     E     I     E
        #     |     |     |
        #     x--I--x--I--x
        #     |     |     |
        #     E     I     E
        #     |     |     |
        #     x--E--x--E--x
        #
        # The vertical exterior facets are simply given by all the points coming
        # from exterior facets in the base mesh.
        exterior_vert_plex_indices = self.topology_dm.getLabel("base_exterior_facets").getStratumIS(1)

        # Drop non-facet indices (i.e. the extruded vertices)
        return dmcommon.filter_is(
            exterior_vert_plex_indices,
            *self.topology_dm.getDepthStratum(self.dimension-1),
        )

    @cached_property
    def _facet_horiz_plex_indices(self) -> PETSc.IS:
        # Consider extruding the following interval mesh:
        #
        #     x-----x-----x
        #
        # to
        #
        #     x--H--x--H--x
        #     |     |     |
        #     |     |     |
        #     |     |     |
        #     x--H--x--H--x
        #     |     |     |
        #     |     |     |
        #     |     |     |
        #     x--H--x--H--x
        #
        # The horizontal facets are those generated from base cells.
        base_cell_plex_indices = self.topology_dm.getLabel("base_dim").getStratumIS(self._base_mesh.dimension)

        # Drop non-facet indices (i.e. the extruded cells)
        return dmcommon.filter_is(
            base_cell_plex_indices,
            *self.topology_dm.getDepthStratum(self.dimension-1),
        )

    @cached_property
    def _facet_vert_plex_indices(self) -> PETSc.IS:
        # Consider extruding the following interval mesh:
        #
        #     x-----x-----x
        #
        # to
        #
        #     x-----x-----x
        #     |     |     |
        #     V     V     V
        #     |     |     |
        #     x-----x-----x
        #     |     |     |
        #     V     V     V
        #     |     |     |
        #     x-----x-----x
        #
        # The vertical facets are those generated from base facets.
        base_facet_plex_indices = self.topology_dm.getLabel("base_dim").getStratumIS(self._base_mesh.dimension-1)

        # Drop non-facet indices (i.e. the extruded vertices)
        return dmcommon.filter_is(
            base_facet_plex_indices,
            *self.topology_dm.getDepthStratum(self.dimension-1),
        )

    # TODO: Prefer '_is' over '_indices'
    @cached_property
    def _exterior_facet_top_plex_indices(self) -> PETSc.IS:
        return self._exterior_facet_horiz_plex_indices_is("top")

    @cached_property
    def _exterior_facet_bottom_plex_indices(self) -> PETSc.IS:
        return self._exterior_facet_horiz_plex_indices_is("bottom")

    def _exterior_facet_horiz_plex_indices_is(self, facet_type: Literal["top"] | Literal["bottom"]) -> PETSc.IS:
        # Consider extruding the following interval mesh:
        #
        #     x-----x-----x
        #
        # to
        #
        #     x--E--x--E--x
        #     |     |     |
        #     |     |     |
        #     |     |     |
        #     x--I--x--I--x
        #     |     |     |
        #     |     |     |
        #     |     |     |
        #     x--E--x--E--x
        #
        # The external horizontal facets are the first and last horizontal
        # facets in each column. Since we know DMPlex numbers edges contiguously
        # up the column we can just slice them out.
        if facet_type == "top":
            take_index = -1
        else:
            assert facet_type == "bottom"
            take_index = 0
        exterior_facet_horiz_indices = (
            self._facet_horiz_plex_indices.indices
            .reshape((-1, self.layers))[:, take_index]
            .flatten()
        )
        return PETSc.IS().createGeneral(exterior_facet_horiz_indices, comm=MPI.COMM_SELF)

    @cached_property
    def _interior_facet_horiz_plex_indices(self) -> PETSc.IS:
        return (
            self._facet_horiz_plex_indices
            .difference(self._exterior_facet_top_plex_indices)
            .difference(self._exterior_facet_bottom_plex_indices)
        )

    @cached_property
    def _interior_facet_vert_plex_indices(self) -> PETSc.IS:
        # Consider extruding the following interval mesh:
        #
        #     E-----I-----E
        #
        # to
        #
        #     x--E--x--E--x
        #     |     |     |
        #     E     I     E
        #     |     |     |
        #     x--I--x--I--x
        #     |     |     |
        #     E     I     E
        #     |     |     |
        #     x--E--x--E--x
        #
        # The vertical interior facets are simply given by all the points coming
        # from interior facets in the base mesh.
        interior_vert_plex_indices_is = utils.safe_is(
            self.topology_dm.getLabel("base_interior_facets").getStratumIS(1)
        )

        # Drop non-facet indices (i.e. the extruded vertices)
        return dmcommon.filter_is(
            interior_vert_plex_indices_is,
            *self.topology_dm.getDepthStratum(self.dimension-1),
        )

    @cached_property
    def _old_to_new_facet_horiz_numbering(self) -> PETSc.IS:
        return dmcommon.entity_numbering(self._facet_horiz_plex_indices, self._new_to_old_point_renumbering, self.comm)

    @cached_property
    def _old_to_new_facet_vert_numbering(self) -> PETSc.IS:
        return dmcommon.entity_numbering(self._facet_vert_plex_indices, self._new_to_old_point_renumbering, self.comm)

    # Maybe this is better as a match-case thing, instead of lots and lots of properties (cached on the mesh)
    # TODO: This is a bad name, needs to point out that we map between entities here, not plex points
    @cached_property
    def _old_to_new_exterior_facet_top_numbering(self) -> PETSc.IS:
        return dmcommon.entity_numbering(self._exterior_facet_top_plex_indices, self._new_to_old_point_renumbering, self.comm)

    @cached_property
    def _old_to_new_exterior_facet_bottom_numbering(self) -> PETSc.IS:
        return dmcommon.entity_numbering(self._exterior_facet_bottom_plex_indices, self._new_to_old_point_renumbering, self.comm)

    @cached_property
    def _old_to_new_exterior_facet_vert_numbering(self) -> PETSc.IS:
        return dmcommon.entity_numbering(self._exterior_facet_vert_plex_indices, self._new_to_old_point_renumbering, self.comm)

    # Maybe this is better as a match-case thing, instead of lots and lots of properties (cached on the mesh)
    @cached_property
    def _old_to_new_interior_facet_horiz_numbering(self) -> PETSc.IS:
        return dmcommon.entity_numbering(self._interior_facet_horiz_plex_indices, self._new_to_old_point_renumbering, self.comm)

    @cached_property
    def _old_to_new_interior_facet_vert_numbering(self) -> PETSc.IS:
        return dmcommon.entity_numbering(self._interior_facet_vert_plex_indices, self._new_to_old_point_renumbering, self.comm)

    @cached_property
    def _exterior_facet_top_support_dat(self) -> op3.Dat:
        return _memoize_facet_supports(
            self.topology_dm,
            self.exterior_facets_top.owned,
            self._exterior_facet_top_plex_indices,
            self._old_to_new_exterior_facet_top_numbering,
            self._old_to_new_cell_numbering,
            "exterior",
        )

    @cached_property
    def _exterior_facet_bottom_support_dat(self) -> op3.Dat:
        return _memoize_facet_supports(
            self.topology_dm,
            self.exterior_facets_bottom.owned,
            self._exterior_facet_bottom_plex_indices,
            self._old_to_new_exterior_facet_bottom_numbering,
            self._old_to_new_cell_numbering,
            "exterior",
        )

    @cached_property
    def _exterior_facet_vert_support_dat(self) -> op3.Dat:
        return _memoize_facet_supports(
            self.topology_dm,
            self.exterior_facets_vert.owned,
            self._exterior_facet_vert_plex_indices,
            self._old_to_new_exterior_facet_vert_numbering,
            self._old_to_new_cell_numbering,
            "exterior",
        )

    @cached_property
    def _interior_facet_horiz_support_dat(self) -> op3.Dat:
        return _memoize_facet_supports(
            self.topology_dm,
            self.interior_facets_horiz.owned,
            self._interior_facet_horiz_plex_indices,
            self._old_to_new_interior_facet_horiz_numbering,
            self._old_to_new_cell_numbering,
            "interior",
        )

    @cached_property
    def _interior_facet_vert_support_dat(self) -> op3.Dat:
        return _memoize_facet_supports(
            self.topology_dm,
            self.interior_facets_vert.owned,
            self._interior_facet_vert_plex_indices,
            self._old_to_new_interior_facet_vert_numbering,
            self._old_to_new_cell_numbering,
            "interior",
        )

    # }}}


    # @utils.cached_property
    # def cell_closure(self):
    #     """2D array of ordered cell closures
    #
    #     Each row contains ordered cell entities for a cell, one row per cell.
    #     """
    #     return self._base_mesh.cell_closure

    @cached_property
    def _plex_closures(self) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    @cached_property
    def _plex_closures_renumbered(self) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    @cached_property
    def _plex_closures_localized(self) -> tuple[tuple[np.ndarray, ...], ...]:
        raise NotImplementedError

    @cached_property
    def _fiat_cell_closures(self) -> np.ndarray:
        assert False, "not needed for extruded meshes"

    @cached_property
    def _fiat_cell_closures_renumbered(self) -> np.ndarray:
        assert False, "not needed for extruded meshes"

    # TODO: I think I should be able to avoid a lot of this if the base closure ordering can be expressed as a permutation
    @cached_property
    def _fiat_cell_closures_localized(self) -> tuple[np.ndarray, ...]:
        nlayers = int(self.layers) - 1

        closures = {}
        for base_dest_dim in range(self._base_mesh.dimension+1):
            base_closures = self._base_mesh._fiat_cell_closures_localized[base_dest_dim]
            for extr_dest_dim in range(2):
                dest_dim = (base_dest_dim, extr_dest_dim)
                closure_size = self._closure_sizes[self.cell_label][dest_dim]

                n_base_cells = base_closures.shape[0]  # always the same
                idxs = np.empty((n_base_cells, nlayers, closure_size), dtype=base_closures.dtype)

                num_extr_pts = nlayers+1 if extr_dest_dim == 0 else nlayers
                if self.periodic and extr_dest_dim == 0:
                    real_column_height = num_extr_pts - 1
                else:
                    real_column_height = num_extr_pts

                if extr_dest_dim == 0:
                    # 'vertex' extrusion, twice as many points in the closure
                    for ci in range(n_base_cells):
                        for j in range(nlayers):
                            for k in range(base_closures.shape[1]):
                                idxs[ci, j, 2*k] = base_closures[ci, k] * real_column_height + j
                                idxs[ci, j, 2*k+1] = base_closures[ci, k] * real_column_height + ((j + 1) % real_column_height)
                else:
                    # 'edge' extrusion, only one point in the closure
                    for ci in range(n_base_cells):
                        for j in range(nlayers):
                            # for k in range(closure_size):
                            for k in range(base_closures.shape[1]):
                                idxs[ci, j, k] = base_closures[ci, k] * num_extr_pts + j
                closures[dest_dim] = idxs.reshape((-1, closure_size))
        return closures

    @cached_property
    def _base_fiat_cell_closure_data_localized(self):
        if self.layers.shape:
            raise NotImplementedError
        else:
            n_extr_cells = int(self.layers) - 1

        closures = {}
        for base_dest_dim in range(self._base_mesh.dimension+1):
            base_closures = self._base_mesh._fiat_cell_closures_localized[base_dest_dim]
            # for extr_dest_dim in range(2):
            #     dest_dim = (base_dest_dim, extr_dest_dim)
            #     closure_size = self._base_mesh._closure_sizes[self._base_mesh.dimension][base_dest_dim]
            #
            #     n_base_cells = self._base_mesh.num_cells
            #     idxs = np.empty((n_base_cells, nlayers, closure_size), dtype=base_closures.dtype)
            #     num_extr_pts = nlayers+1 if extr_dest_dim == 0 else nlayers
            #
            #     for ci in range(n_base_cells):
            #         for j in range(nlayers):
            #             for k in range(closure_size):
            #                 idxs[ci, j, k] = base_closures[ci, k]
            #
            #     closures[dest_dim] = idxs.reshape((-1, closure_size))

            dest_dim = base_dest_dim
            closure_size = self._base_mesh._closure_sizes[self._base_mesh.dimension][base_dest_dim]

            n_base_cells = self._base_mesh.num_cells
            idxs = np.empty((n_base_cells, n_extr_cells, closure_size), dtype=base_closures.dtype)

            for ci in range(n_base_cells):
                for j in range(n_extr_cells):
                    for k in range(closure_size):
                        idxs[ci, j, k] = base_closures[ci, k]

            closures[dest_dim] = idxs.reshape((-1, closure_size))
        return closures

    # NOTE: This is very similar to the other closure stuff that we do.
    @cached_property
    def base_mesh_closure(self):
        # map from extruded entities to flat ones.
        closures = {}

        dim = self.cell_label
        closure_data = self._base_fiat_cell_closure_data_localized

        map_components = []
        for map_dim, map_data in closure_data.items():
            *_, size = map_data.shape
            if size == 0:
                continue

            # target_axis = self.name
            # target_dim = map_dim
            target_axis = self._base_mesh.name
            target_dim = map_dim

            # Discard any parallel information, the maps are purely local
            outer_axis = self.points.root.linearize(dim).localize()

            # NOTE: currently we must label the innermost axis of the map to be the same as the resulting
            # indexed axis tree. I don't yet know whether to raise an error if this is not upheld or to
            # fix automatically internally via additional replace() arguments.
            map_axes = op3.AxisTree.from_nest(
                {outer_axis: op3.Axis({target_dim: size}, "closure")}
            )
            map_dat = op3.Dat(
                map_axes, data=map_data.flatten(), prefix="closure"
            )
            map_components.append(
                op3.TabulatedMapComponent(target_axis, target_dim, map_dat, label=map_dim)
            )

            # 1-tuple here because in theory closure(cell) could map to other valid things (like points)
            closures[idict({self.name: dim})] =  (tuple(map_components),)

        return op3.Map(closures, name="closure")

    @cached_property
    def _support(self) -> op3.Map:
        supports = {}
        supported_supports = (
            (self.exterior_facets_top, self._exterior_facet_top_support_dat),
            (self.exterior_facets_bottom, self._exterior_facet_bottom_support_dat),
            (self.exterior_facets_vert, self._exterior_facet_vert_support_dat),
            (self.interior_facets_horiz, self._interior_facet_horiz_support_dat),
            (self.interior_facets_vert, self._interior_facet_vert_support_dat),
        )
        for iterset, support_dat in supported_supports:
            axis = iterset.owned.as_axis()
            from_path = idict({axis.label: axis.component.label})
            supports[from_path] = [[
                op3.TabulatedMapComponent(self.name, self.cell_label, support_dat, label=None),
            ]]
        return op3.Map(supports, name="support")

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

    # @utils.cached_property
    # def layers(self):
    #     """Return the layers parameter used to construct the mesh topology,
    #     which is the number of layers represented by the number of occurences
    #     of the base mesh for non-variable layer mesh and an array of size
    #     (num_cells, 2), each row representing the
    #     (first layer index, last layer index + 1) pair for the associated cell,
    #     for variable layer mesh."""
    #     if self.variable_layers:
    #         return self.cell_set.layers_array
    #     else:
    #         return self.cell_set.layers

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
        assert False, "old code"
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

    @property
    def dimension(self):
        return 0

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
        return ufl.Mesh(finat.ufl.VectorElement("DG", cell, 0, dim=cell.topological_dimension))

    def _renumber_entities(self, reorder):
        assert False, "old code"
        if reorder:
            swarm = self.topology_dm
            parent = self._parent_mesh.topology_dm
            cell_id_name = swarm.getCellDMActive().getCellID()
            swarm_parent_cell_nums = swarm.getField(cell_id_name).ravel()
            parent_renum = self._parent_mesh._new_to_old_point_renumbering.getIndices()
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

    entity_orientations = None

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

    @property
    def num_points(self) -> int:
        return self.num_vertices

    @property
    def num_cells(self) -> int:
        return self.num_vertices

    # TODO I reckon that these should error instead
    @property
    def num_facets(self):
        return 0

    # TODO I reckon that these should error instead
    @property
    def num_faces(self):
        return 0

    # TODO I reckon that these should error instead
    @property
    def num_edges(self):
        return 0

    @property
    def num_vertices(self):
        return self.topology_dm.getLocalSize()

    def num_entities(self, d):
        if d > 0:
            return 0
        else:
            return self.num_vertices

    # TODO: Clean this all up
    def entity_count(self, dim):
        if dim == 0:
            return self.num_vertices
        else:
            return 0

    @cached_property
    def cells(self):
        # Need to be more verbose as we don't want to consume the axis
        # return self.points[self.cell_label]
        # This may no longer be needed
        cell_slice = op3.Slice(self.name, [op3.AffineSliceComponent(self.cell_label)])
        return self.points[cell_slice]

    @cached_property  # TODO: Recalculate if mesh moves
    @utils.deprecated("cells.owned")
    def cell_set(self):
        return self.cells.owned

    @property
    def cell_label(self) -> int:
        return self.dimension

    @property
    def facet_label(self):
        raise RuntimeError

    @property
    def edge_label(self):
        raise RuntimeError

    # TODO I prefer "vertex_label"
    @property
    def vert_label(self):
        return 0

    @cached_property
    def _entity_indices(self):
        raise NotImplementedError

    @cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_cell_list(self):
        """Return a list of parent mesh cells numbers in vertex only
        mesh cell order.
        """
        cell_parent_cell_list = np.copy(self.topology_dm.getField("parentcellnum").ravel())
        self.topology_dm.restoreField("parentcellnum")
        return cell_parent_cell_list[self._old_to_new_cell_numbering_is.invertPermutation().indices]

    @cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_cell_map(self):
        """Return the :class:`pyop2.types.map.Map` from vertex only mesh cells to
        parent mesh cells.
        """
        dest_axis = self._parent_mesh.name
        dest_stratum = self._parent_mesh.cell_label

        map_axes = op3.AxisTree.from_iterable([self.points.root, op3.Axis(1, "cell_parent_cell")])
        dat = op3.Dat(map_axes, data=self.cell_parent_cell_list)

        return op3.Map(
            {
                idict({self.name: self.cell_label}): [[
                    op3.TabulatedMapComponent(dest_axis, dest_stratum, dat, label=None),
                ]]
            },
            name="cell_parent_cell",
        )

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_base_cell_list(self):
        """Return a list of parent mesh base cells numbers in vertex only
        mesh cell order.
        """
        if not isinstance(self._parent_mesh, ExtrudedMeshTopology):
            raise AttributeError("Parent mesh is not extruded")
        cell_parent_base_cell_list = np.copy(self.topology_dm.getField("parentcellbasenum").ravel())
        self.topology_dm.restoreField("parentcellbasenum")
        return cell_parent_base_cell_list[self._new_to_old_cell_numbering]

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_base_cell_map(self):
        """Return the :class:`pyop2.types.map.Map` from vertex only mesh cells to
        parent mesh base cells.
        """
        raise NotImplementedError
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
        return cell_parent_extrusion_height_list[self._new_to_old_cell_numbering]

    @utils.cached_property  # TODO: Recalculate if mesh moves
    def cell_parent_extrusion_height_map(self):
        """Return the :class:`pyop2.types.map.Map` from vertex only mesh cells to
        parent mesh extrusion heights.
        """
        raise NotImplementedError
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
        nroots = self.input_ordering.num_cells
        e_p_map = self._new_to_old_cell_numbering  # cell-entity -> swarm-point map
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
        return self.input_ordering_sf.createEmbeddedLeafSF(np.arange(self.cells.owned.local_size, dtype=IntType))


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
        mesh = super(MeshGeometry, cls).__new__(cls)
        uid = utils._new_uid(comm)
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
        self._base_mesh = None  # this is set by extruded meshes in a later step

        # initialise the mesh cargo
        self.ufl_cargo().init(coordinates)

        # Cache mesh object on the coordinateless coordinates function
        coordinates._as_mesh_geometry = weakref.ref(self)

        # submesh
        self.submesh_parent = None

        self._bounding_box_coords = None
        self._spatial_index = None
        self._saved_coordinate_dat_version = coordinates.dat.buffer.state

        self._cache = {}

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
        if len(topology.dm_cell_types) > 1:
            return
        coordinates_fs = functionspace.FunctionSpace(self.topology, self.ufl_coordinate_element())
        coordinates_data = dmcommon.reordered_coords(topology.topology_dm, coordinates_fs.dm.getLocalSection(),
                                                     (self.num_vertices, self.geometric_dimension))
        coordinates = function.CoordinatelessFunction(coordinates_fs,
                                                      val=coordinates_data,
                                                      name=_generate_default_mesh_coordinates_name(self.name))
        self.__init__(coordinates)

    @property
    def comm(self):
        return self.topology.comm

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

    # @property
    # def num_cells(self):
    #     return self.topology.num_cells
    #
    # @property
    # def num_facets(self):
    #     return self.topology.num_facets
    #
    # @property
    # def num_edges(self):
    #     return self.topology.num_edges
    #
    # @property
    # def num_vertices(self):
    #     return self.topology.num_vertices

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

    @cached_property
    def bounding_box_coords(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Calculates bounding boxes for spatial indexing.

        Returns
        -------
        Tuple of arrays of shape (num_cells, gdim) containing
        the minimum and maximum coordinates of each cell's bounding box.

        None if the geometric dimension is 1, since libspatialindex
        does not support 1D.

        Notes
        -----
        If we have a higher-order (bendy) mesh we project the mesh coordinates into
        a Bernstein finite element space. Functions on a Bernstein element are
        Bezier curves and are completely contained in the convex hull of the mesh nodes.
        Hence the bounding box will contain the entire element.
        """
        from firedrake import function, functionspace
        from firedrake.parloops import par_loop, READ

        gdim = self.geometric_dimension
        if gdim <= 1:
            info_red("libspatialindex does not support 1-dimension, falling back on brute force.")
            return None

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

        # Calculate the bounding boxes for all cells by running a kernel
        V = functionspace.VectorFunctionSpace(mesh, "DG", 0, dim=gdim)
        coords_min = function.Function(V, dtype=RealType)
        coords_max = function.Function(V, dtype=RealType)

        coords_min.dat.data_wo.fill(np.inf)
        coords_max.dat.data_wo.fill(-np.inf)

        if utils.complex_mode:
            if not np.allclose(mesh.coordinates.dat.data_ro.imag, 0):
                raise ValueError("Coordinate field has non-zero imaginary part")
            coords = function.Function(mesh.coordinates.function_space(),
                                       val=mesh.coordinates.dat.data_ro_with_halos.real.copy(),
                                       dtype=RealType)
        else:
            coords = mesh.coordinates

        cell_node_list = mesh.coordinates.function_space().cell_node_list
        _, nodes_per_cell = cell_node_list.shape

        domain = f"{{[d, i]: 0 <= d < {gdim} and 0 <= i < {nodes_per_cell}}}"
        instructions = """
        for d, i
            f_min[0, d] = fmin(f_min[0, d], f[i, d])
            f_max[0, d] = fmax(f_max[0, d], f[i, d])
        end
        """
        par_loop((domain, instructions), ufl.dx,
                 {'f': (coords, op3.READ),
                  'f_min': (coords_min, op3.RW),
                  'f_max': (coords_max, op3.RW)})

        # Reorder bounding boxes according to the cell indices we use
        column_list = V.cell_node_list.flatten()

        return coords_min.dat.data_ro[column_list], coords_max.dat.data_ro[column_list]

    @property
    def spatial_index(self):
        """Builds spatial index from bounding box coordinates, expanding
        the bounding box by the mesh tolerance.

        Returns
        -------
        :class:`~.spatialindex.SpatialIndex` or None if the mesh is
        one-dimensional.

        Notes
        -----
        If this mesh has a :attr:`tolerance` property, which
        should be a float, this tolerance is added to the extrema of the
        spatial index so that points just outside the mesh, within tolerance,
        can be found.

        """
        if self.coordinates.dat.dat_version != self._saved_coordinate_dat_version:
            if "bounding_box_coords" in self.__dict__:
                del self.bounding_box_coords
        else:
            if self._spatial_index:
                return self._spatial_index
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
        if self.bounding_box_coords is None:
            # This happens in 1D meshes
            return None
        else:
            coords_min, coords_max = self.bounding_box_coords
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
        if x.size != self.geometric_dimension:
            raise ValueError("Point must have the same geometric dimension as the mesh")
        x = x.reshape((1, self.geometric_dimension))
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
        if tolerance is None:
            tolerance = self.tolerance
        else:
            self.tolerance = tolerance
        xs = np.asarray(xs, dtype=utils.ScalarType)
        xs = xs.real.copy()
        if xs.shape[1] != self.geometric_dimension:
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
        assert xs.size == npoints * self.geometric_dimension
        locator = self._c_locator(tolerance=tolerance)
        locator(self.coordinates._ctypes,
                xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                Xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ref_cell_dists_l1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                cells.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                npoints,
                cells_ignore.shape[1],
                cells_ignore)
        return cells, Xs, ref_cell_dists_l1

    def _c_locator(self, tolerance=None):
        from pyop3 import compile as compilation
        from pyop3.pyop2_utils import get_petsc_dir
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

                        /* to_reference_coords is defined in
                        pointquery_utils.py. If they contain python calls, this loop will
                        not run at c-loop speed. */
                        /* cells_ignore has shape (npoints, ncells_ignore) - find the ith row */
                        int *cells_ignore_i = cells_ignore + i*ncells_ignore;
                        cells[i] = locate_cell(f, &x[j], {self.geometric_dimension}, &to_reference_coords, &temp_reference_coords, &found_reference_coords, &ref_cell_dists_l1[i], ncells_ignore, cells_ignore_i);

                        for (int k = 0; k < {self.geometric_dimension}; k++) {{
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
        return getattr(self._topology, name)

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

    def __iter__(self):
        yield self

    def unique(self):
        return self

    def refine_marked_elements(self, mark, netgen_flags=None):
        """Refine a mesh using a DG0 marking function.

        This method requires that the mesh has been constructed from a
        netgen mesh.

        :arg mark: the marking function which is a Firedrake DG0 function.
        :arg netgen_flags: the dictionary of flags to be passed to ngsPETSc.

        It includes the option:
            - refine_faces, which is a boolean specifying if you want to refine faces.

        """
        import firedrake as fd

        utils.check_netgen_installed()

        if netgen_flags is None:
            netgen_flags = {}
        DistParams = self._distribution_parameters
        els = {2: self.netgen_mesh.Elements2D, 3: self.netgen_mesh.Elements3D}
        dim = self.geometric_dimension
        refine_faces = netgen_flags.get("refine_faces", False)
        if dim in [2, 3]:
            with mark.dat.vec as marked:
                marked0 = marked
                getIdx = self._cell_numbering.getOffset
                if self.sfBC is not None:
                    sfBCInv = self.sfBC.createInverse()
                    getIdx = lambda x: x
                    _, marked0 = self.topology_dm.distributeField(sfBCInv,
                                                                  self._cell_numbering,
                                                                  marked)
                if self.comm.Get_rank() == 0:
                    mark = marked0.getArray()
                    max_refs = np.max(mark)
                    for _ in range(int(max_refs)):
                        for i, el in enumerate(els[dim]()):
                            if mark[getIdx(i)] > 0:
                                el.refine = True
                            else:
                                el.refine = False
                        if not refine_faces and dim == 3:
                            self.netgen_mesh.Elements2D().NumPy()["refine"] = 0
                        self.netgen_mesh.Refine(adaptive=True)
                        mark = mark-np.ones(mark.shape)
                    return fd.Mesh(self.netgen_mesh, distribution_parameters=DistParams, comm=self.comm)
                return fd.Mesh(netgen.libngpy._meshing.Mesh(dim),
                               distribution_parameters=DistParams, comm=self.comm)
        else:
            raise NotImplementedError("No implementation for dimension other than 2 and 3.")

    @PETSc.Log.EventDecorator()
    def curve_field(self, order, permutation_tol=1e-8, location_tol=1e-1, cg_field=False):
        '''Return a function containing the curved coordinates of the mesh.

        This method requires that the mesh has been constructed from a
        netgen mesh.

        :arg order: the order of the curved mesh.
        :arg permutation_tol: tolerance used to construct the permutation of the reference element.
        :arg location_tol: tolerance used to locate the cell a point belongs to.
        :arg cg_field: return a CG function field representing the mesh, rather than the
                       default DG field.

        '''
        import firedrake as fd

        utils.check_netgen_installed()

        from firedrake.netgen import find_permutation

        # Check if the mesh is a surface mesh or two dimensional mesh
        if len(self.netgen_mesh.Elements3D()) == 0:
            ng_element = self.netgen_mesh.Elements2D
        else:
            ng_element = self.netgen_mesh.Elements3D
        ng_dimension = len(ng_element())
        geom_dim = self.geometric_dimension

        # Construct the mesh as a Firedrake function
        if cg_field:
            firedrake_space = fd.VectorFunctionSpace(self, "CG", order)
        else:
            low_order_element = self.coordinates.function_space().ufl_element().sub_elements[0]
            ufl_element = low_order_element.reconstruct(degree=order)
            firedrake_space = fd.VectorFunctionSpace(self, fd.BrokenElement(ufl_element))
        new_coordinates = fd.assemble(fd.interpolate(self.coordinates, firedrake_space))

        # Compute reference points using fiat
        fiat_element = new_coordinates.function_space().finat_element.fiat_equivalent
        entity_ids = fiat_element.entity_dofs()
        nodes = fiat_element.dual_basis()
        ref = []
        for dim in entity_ids:
            for entity in entity_ids[dim]:
                for dof in entity_ids[dim][entity]:
                    # Assert singleton point for each node.
                    pt, = nodes[dof].get_point_dict().keys()
                    ref.append(pt)
        reference_space_points = np.array(ref)

        # Curve the mesh on rank 0 only
        if self.comm.rank == 0:
            # Construct numpy arrays for physical domain data
            physical_space_points = np.zeros(
                (ng_dimension, reference_space_points.shape[0], geom_dim)
            )
            curved_space_points = np.zeros(
                (ng_dimension, reference_space_points.shape[0], geom_dim)
            )
            self.netgen_mesh.CalcElementMapping(reference_space_points, physical_space_points)
            # NOTE: This will segfault!
            self.netgen_mesh.Curve(order)
            self.netgen_mesh.CalcElementMapping(reference_space_points, curved_space_points)
            curved = ng_element().NumPy()["curved"]
            # Broadcast a boolean array identifying curved cells
            curved = self.comm.bcast(curved, root=0)
            physical_space_points = physical_space_points[curved]
            curved_space_points = curved_space_points[curved]
        else:
            curved = self.comm.bcast(None, root=0)
            # Construct numpy arrays as buffers to receive physical domain data
            ncurved = np.sum(curved)
            physical_space_points = np.zeros(
                (ncurved, reference_space_points.shape[0], geom_dim)
            )
            curved_space_points = np.zeros(
                (ncurved, reference_space_points.shape[0], geom_dim)
            )

        # Broadcast curved cell point data
        self.comm.Bcast(physical_space_points, root=0)
        self.comm.Bcast(curved_space_points, root=0)
        cell_node_map = new_coordinates.cell_node_map()

        # Select only the points in curved cells
        barycentres = np.average(physical_space_points, axis=1)
        ng_index = [*map(lambda x: self.locate_cell(x, tolerance=location_tol), barycentres)]

        # Select only the indices of points owned by this rank
        owned = [(0 <= ii < len(cell_node_map.values)) if ii is not None else False for ii in ng_index]

        # Select only the points owned by this rank
        physical_space_points = physical_space_points[owned]
        curved_space_points = curved_space_points[owned]
        barycentres = barycentres[owned]
        ng_index = [idx for idx, o in zip(ng_index, owned) if o]

        # Get the PyOP2 indices corresponding to the netgen indices
        pyop2_index = []
        for ngidx in ng_index:
            pyop2_index.extend(cell_node_map.values[ngidx])

        # Find the correct coordinate permutation for each cell
        permutation = find_permutation(
            physical_space_points,
            new_coordinates.dat.data[pyop2_index].real.reshape(
                physical_space_points.shape
            ),
            tol=permutation_tol
        )

        # Apply the permutation to each cell in turn
        for ii, p in enumerate(curved_space_points):
            curved_space_points[ii] = p[permutation[ii]]

        # Assign the curved coordinates to the dat
        new_coordinates.dat.data[pyop2_index] = curved_space_points.reshape(-1, geom_dim)

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
    assert V.mesh().ufl_cell().topological_dimension <= V.value_size

    mesh = MeshGeometry.__new__(MeshGeometry, element, coordinates.comm)
    mesh.__init__(coordinates)
    mesh.name = name
    # Mark mesh as being made from coordinates
    mesh._made_from_coordinates = True
    mesh._tolerance = tolerance
    return mesh


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
    parent_tdim = topology._parent_mesh.ufl_cell().topological_dimension
    if parent_tdim > 0:
        reference_coordinates_fs = functionspace.VectorFunctionSpace(topology, "DG", 0, dim=parent_tdim)
        reference_coordinates_data = dmcommon.reordered_coords(topology.topology_dm, reference_coordinates_fs.dm.getDefaultSection(),
                                                               (topology.num_vertices, parent_tdim),
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

    def getDepthStratum(self, dimension: int) -> tuple[int, int]:
        assert dimension == 0
        return (0, self.getLocalSize())

    def getHeightStratum(self, dimension: int) -> tuple[int, int]:
        return self.getDepthStratum(dimension)

    def getTransitiveClosure(self, point: int) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray([point], dtype=IntType), [0])

    def getChart(self):
        return self.getDepthStratum(0)


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
    tdim = parent_mesh.topological_dimension
    gdim = parent_mesh.geometric_dimension

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
    plex_parent_cell_nums = parent_mesh._new_to_old_cell_numbering[parent_cell_nums_local]
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
        parent_mesh.topology.layers,
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

    # if extruded:
    if False:
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

    with temp_internal_comm(parent_mesh.comm) as icomm:
        # In parallel, we need to make sure we know which point is which and save
        # it.
        if redundant:
            # rank 0 broadcasts coords to all ranks
            coords_local = icomm.bcast(coords, root=0)
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
            ncoords_local_allranks = icomm.allgather(ncoords_local)
            ncoords_global = sum(ncoords_local_allranks)
            # The below code looks complicated but it's just an allgather of the
            # (variable length) coords_local array such that they are concatenated.
            coords_local_size = np.array(coords_local.size)
            coords_local_sizes = np.empty(parent_mesh.comm.size, dtype=int)
            icomm.Allgatherv(coords_local_size, coords_local_sizes)
            coords_global = np.empty(
                (ncoords_global, coords.shape[1]), dtype=coords_local.dtype
            )
            icomm.Allgatherv(coords_local, (coords_global, coords_local_sizes))
            # # ncoords_local_allranks is in rank order so we can just sum up the
            # # previous ranks to get the starting index for the global numbering.
            # # For rank 0 we make use of the fact that sum([]) = 0.
            # startidx = sum(ncoords_local_allranks[:parent_mesh.comm.rank])
            # endidx = startidx + ncoords_local
            # global_idxs_global = np.arange(startidx, endidx)
            global_idxs_global = np.arange(coords_global.shape[0])
            input_coords_idxs_local = np.arange(ncoords_local)
            input_coords_idxs_global = np.empty(ncoords_global, dtype=int)
            icomm.Allgatherv(
                input_coords_idxs_local, (input_coords_idxs_global, ncoords_local_allranks)
            )
            input_ranks_local = np.full(ncoords_local, icomm.rank, dtype=int)
            input_ranks_global = np.empty(ncoords_global, dtype=int)
            icomm.Allgatherv(
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
        visible_ranks = interpolation.interpolate(
            constant.Constant(parent_mesh.comm.rank), P0DG
        )
        visible_ranks = assemble(visible_ranks).dat.buffer._data.real

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

    if parent_mesh.geometric_dimension > parent_mesh.topological_dimension:
        # The reference coordinates contain an extra unnecessary dimension
        # which we can safely delete
        reference_coords = reference_coords[:, : parent_mesh.topological_dimension]

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
            if parent_mesh.geometric_dimension > parent_mesh.topological_dimension:
                new_reference_coords = new_reference_coords[:, : parent_mesh.topological_dimension]
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
    plex1.removeLabel("firedrake_is_ghost")
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
        section = f.topological.function_space().local_section
        dmcommon.mark_points_with_function_array(plex, section, height, f.dat.data_ro_with_halos.real.astype(IntType), dmlabel, subid)
    distribution_parameters_noop = {"partition": False,
                                    "overlap_type": (DistributedMeshOverlapType.NONE, 0)}
    reorder_noop = None
    tmesh1 = MeshTopology(plex1, name=plex1.getName(), reorder=reorder_noop,
                          distribution_parameters=distribution_parameters_noop,
                          perm_is=tmesh._new_to_old_point_renumbering,
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
    raise NotImplementedError
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


def Submesh(mesh, subdim, subdomain_id, label_name=None, name=None, ignore_halo=False, reorder=True, comm=None):
    """Construct a submesh from a given mesh.

    Parameters
    ----------
    mesh : MeshGeometry
        Parent mesh (`MeshGeometry`).
    subdim : int
        Topological dimension of the submesh.
    subdomain_id : int | None
        Subdomain ID representing the submesh.
        `None` defines the submesh owned by the sub-communicator.
    label_name : str | None
        Name of the label to search ``subdomain_id`` in.
    name : str |  None
        Name of the submesh.
    ignore_halo : bool
        Whether to exclude the halo from the submesh.
    reorder : bool
        Whether to reorder the mesh entities.
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
    if subdomain_id is None:
        # Filter the plex with PETSc's default label (cells owned by comm)
        if label_name != dmcommon.CELL_SETS_LABEL:
            raise ValueError("subdomain_id == None requires label_name == CELL_SETS_LABEL.")
        subplex, sf = plex.filter(sanitizeSubMesh=True, ignoreHalo=ignore_halo, comm=comm)
        dmcommon.submesh_update_facet_labels(plex, subplex)
        dmcommon.submesh_correct_entity_classes(plex, subplex, sf)
    else:
        subplex = dmcommon.submesh_create(plex, subdim, label_name, subdomain_id, ignore_halo, comm=comm)

    comm = comm or mesh.comm
    name = name or _generate_default_submesh_name(mesh.name)
    subplex.setName(_generate_default_mesh_topology_name(name))
    if subplex.getDimension() != subdim:
        raise RuntimeError(f"Found subplex dim ({subplex.getDimension()}) != expected ({subdim})")
    submesh = Mesh(
        subplex,
        submesh_parent=mesh,
        name=name,
        comm=comm,
        reorder=reorder,
        distribution_parameters={
            "partition": False,
            "overlap_type": (DistributedMeshOverlapType.NONE, 0),
        },
    )
    return submesh


@dataclasses.dataclass(frozen=True)
class IterationSpec:
    mesh: MeshGeometry
    integral_type: str
    iterset: op3.IndexedAxisTree
    plex_indices: PETSc.IS | None
    old_to_new_numbering: PETSc.Section

    @cached_property
    def loop_index(self) -> op3.LoopIndex:
        return self.iterset[self.subset].iter()

    @cached_property
    def subset(self) -> op3.Slice | Ellipsis:
        if self.indices is None:
            return Ellipsis
        else:
            iterset_axis = self.iterset.as_axis()
            # TODO: Ideally should be able to avoid creating these here and just index
            # with the array
            subset_dat = op3.Dat.from_array(self.indices.indices, prefix="subset")
            return op3.Slice(iterset_axis.label, [op3.Subset(iterset_axis.component.label, subset_dat)])

    @cached_property
    def indices(self) -> PETSc.IS | None:
        if self.plex_indices is None:
            return None
        # We now have the correct set of indices represented in DMPlex numbering, now
        # we have to convert this to a numbering specific to the iteration set (e.g.
        # map point 12 to interior facet 3).
        localized_indices = dmcommon.section_offsets(self.old_to_new_numbering, self.plex_indices, sort=True)

        # Remove ghost points
        localized_indices = dmcommon.filter_is(localized_indices, 0, self.iterset.local_size)
        return localized_indices


def _get_iteration_spec_get_obj(mesh, *args, **kwargs):
    return mesh.topology


def _get_iteration_spec_get_key(mesh, *args, **kwargs) -> Hashable:
    return utils.freeze((args, kwargs))


@cached_on(_get_iteration_spec_get_obj, _get_iteration_spec_get_key)
def get_iteration_spec(
    mesh: MeshGeometry,
    integral_type: str,
    subdomain_id: int | tuple[int, ...] | Literal["everywhere"] | Literal["otherwise"] = "everywhere",
    *,
    all_integer_subdomain_ids: Iterable[int] | None = None,
) -> IterationSpec:
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
    match integral_type:
        case "cell":
            iterset = mesh.cells.owned
            dmlabel_name = dmcommon.CELL_SETS_LABEL
            valid_plex_indices = mesh._cell_plex_indices
            old_to_new_entity_numbering  = mesh._old_to_new_cell_numbering
        case "exterior_facet":
            iterset = mesh.exterior_facets.owned
            dmlabel_name = dmcommon.FACE_SETS_LABEL
            valid_plex_indices = mesh._exterior_facet_plex_indices
            old_to_new_entity_numbering  = mesh._old_to_new_exterior_facet_numbering
        case "interior_facet":
            iterset = mesh.interior_facets.owned
            dmlabel_name = dmcommon.FACE_SETS_LABEL
            valid_plex_indices = mesh._interior_facet_plex_indices
            old_to_new_entity_numbering = mesh._old_to_new_interior_facet_numbering
        case "exterior_facet_top":
            iterset = mesh.exterior_facets_top.owned
            dmlabel_name = dmcommon.FACE_SETS_LABEL
            valid_plex_indices = mesh._exterior_facet_top_plex_indices
            old_to_new_entity_numbering = mesh._old_to_new_exterior_facet_top_numbering
        case "exterior_facet_bottom":
            iterset = mesh.exterior_facets_bottom.owned
            dmlabel_name = dmcommon.FACE_SETS_LABEL
            valid_plex_indices = mesh._exterior_facet_bottom_plex_indices
            old_to_new_entity_numbering = mesh._old_to_new_exterior_facet_bottom_numbering
        case "exterior_facet_vert":
            iterset = mesh.exterior_facets_vert.owned
            dmlabel_name = dmcommon.FACE_SETS_LABEL
            valid_plex_indices = mesh._exterior_facet_vert_plex_indices
            old_to_new_entity_numbering = mesh._old_to_new_exterior_facet_vert_numbering
        case "interior_facet_horiz":
            iterset = mesh.interior_facets_horiz.owned
            dmlabel_name = dmcommon.FACE_SETS_LABEL
            valid_plex_indices = mesh._interior_facet_horiz_plex_indices
            old_to_new_entity_numbering = mesh._old_to_new_interior_facet_horiz_numbering
        case "interior_facet_vert":
            iterset = mesh.interior_facets_vert.owned
            dmlabel_name = dmcommon.FACE_SETS_LABEL
            valid_plex_indices = mesh._interior_facet_vert_plex_indices
            old_to_new_entity_numbering = mesh._old_to_new_interior_facet_vert_numbering
        case _:
            raise AssertionError(f"Integral type {integral_type} not recognised")

    if subdomain_id == "everywhere":
        plex_indices = None
    else:
        if subdomain_id == "otherwise":
            subdomain_ids = (all_integer_subdomain_ids or {}).get(integral_type, ())
            complement = True
        else:
            subdomain_ids = utils.as_tuple(subdomain_id)
            complement = False

        # Get all points labelled with the subdomain ID
        plex_indices = PETSc.IS().createGeneral(np.empty(0, dtype=IntType), MPI.COMM_SELF)
        for subdomain_id in subdomain_ids:
            if subdomain_id == UNMARKED:  # NOTE: This is a constant, but it's very unclear
                plex_indices_to_exclude = PETSc.IS().createGeneral(np.empty(0, dtype=IntType), MPI.COMM_SELF)
                # NOTE: This is different to all_integer_subdomain_ids because that comes from the integral
                all_plex_subdomain_ids = mesh.topology_dm.getLabelIdIS(dmlabel_name).indices
                for subdomain_id_ in all_plex_subdomain_ids:
                    plex_indices_to_exclude = plex_indices_to_exclude.union(
                        utils.safe_is(mesh.topology_dm.getStratumIS(dmlabel_name, subdomain_id_))
                    )
                matching_indices = valid_plex_indices.difference(plex_indices_to_exclude)
            else:
                matching_indices = utils.safe_is(mesh.topology_dm.getStratumIS(dmlabel_name, subdomain_id))
            plex_indices = plex_indices.union(matching_indices)

        # Restrict to indices that exist within the iterset (e.g. drop exterior facets
        # from an interior facet integral)
        plex_indices = dmcommon.intersect_is(plex_indices, valid_plex_indices)

        # If the 'subdomain_id' is 'otherwise' then we now have a list of the
        # indices that we *do not* want
        if complement:
            plex_indices = valid_plex_indices.difference(plex_indices)

        # NOTE: Should we sort plex indices?

    # Use a weakref for the mesh here because otherwise we would store a
    # reference to the mesh in the cache and, since the lifetime of the cache
    # is tied to the mesh, things will never be cleaned up.
    mesh_ref = weakref.proxy(mesh)

    return IterationSpec(mesh_ref, integral_type, iterset, plex_indices, old_to_new_entity_numbering)


# NOTE: This is a bit of an abuse of 'cachedmethod' (this isn't a method) but I think
# it's still a good general approach.
# @cachedmethod(cache=lambda plex: getattr(plex, "_firedrake_cache"))
# TODO: Make this return an IS
def memoize_supports(plex: PETSc.DMPlex, dim: int):
    return _memoize_map_ragged(plex, dim, plex.getSupport)


def _memoize_map_ragged(plex: PETSc.DMPlex, dim, map_func):
    strata = tuple(plex.getDepthStratum(d) for d in range(plex.getDimension()+1))
    def get_dim(_pt):
        for _d, (_start, _end) in enumerate(strata):
            if _start <= _pt < _end:
                return _d
        assert False

    p_start, p_end = plex.getDepthStratum(dim)
    npoints = p_end - p_start

    # Store arities
    sizes = {to_dim: np.zeros(npoints, dtype=IntType) for to_dim in range(plex.getDimension()+1)}
    for stratum_pt, pt in enumerate(range(p_start, p_end)):
        for map_pt in map_func(pt):
            map_dim = get_dim(map_pt)
            sizes[map_dim][stratum_pt] += 1

    # Now store map data
    map_pts = {to_dim: np.full(sum(sizes[to_dim]), -1, dtype=IntType) for to_dim in range(plex.getDimension()+1)}
    offsets = tuple(op3.utils.steps(sizes[d]) for d in range(plex.getDimension()+1))
    plex_pt_offsets = np.empty(plex.getDimension()+1, dtype=IntType)
    for stratum_pt, plex_pt in enumerate(range(p_start, p_end)):
        plex_pt_offsets[...] = 0
        for map_pt in map_func(plex_pt):
            map_dim = get_dim(map_pt)
            map_pts[map_dim][offsets[map_dim][stratum_pt] + plex_pt_offsets[map_dim]] = map_pt
            plex_pt_offsets[map_dim] += 1
    return map_pts, sizes


# def memoize_supports_new(plex: PETSc.DMPlex, dim: int) -> tuple[PETSc.IS, PETSc.Section]:
#     return _memoize_map_ragged_new(plex, dim, plex.getSupport)
#
#
# def _memoize_map_ragged_new(plex: PETSc.DMPlex, dim, map_func) -> tuple[PETSc.IS, PETSc.Section]:
#     strata = tuple(plex.getDepthStratum(d) for d in range(plex.dimension+1))
#     def get_dim(_pt):
#         for _d, (_start, _end) in enumerate(strata):
#             if _start <= _pt < _end:
#                 return _d
#         assert False
#
#     p_start, p_end = plex.getDepthStratum(dim)
#     npoints = p_end - p_start
#
#     # Store arities
#     sizes = {to_dim: np.zeros(npoints, dtype=IntType) for to_dim in range(plex.dimension+1)}
#     for stratum_pt, pt in enumerate(range(p_start, p_end)):
#         for map_pt in map_func(pt):
#             map_dim = get_dim(map_pt)
#             sizes[map_dim][stratum_pt] += 1
#
#     # Now store map data
#     map_pts = {to_dim: np.full(sum(sizes[to_dim]), -1, dtype=IntType) for to_dim in range(plex.dimension+1)}
#     offsets = tuple(op3.utils.steps(sizes[d]) for d in range(plex.dimension+1))
#     plex_pt_offsets = np.empty(plex.dimension+1, dtype=IntType)
#     for stratum_pt, plex_pt in enumerate(range(p_start, p_end)):
#         plex_pt_offsets[...] = 0
#         for map_pt in map_func(plex_pt):
#             map_dim = get_dim(map_pt)
#             map_pts[map_dim][offsets[map_dim][stratum_pt] + plex_pt_offsets[map_dim]] = map_pt
#             plex_pt_offsets[map_dim] += 1
#     return map_pts, sizes


def _memoize_facet_supports(
    plex: PETSc.DMPlex,
    iterset: op3.AbstractAxisTree,
    facet_plex_indices: PETSc.IS,
    facet_numbering: PETSc.Section,
    cell_numbering: PETSc.Section,
    facet_type: Literal["exterior"] | Literal["interior"],
) -> op3.Dat:
    if facet_type == "exterior":
        support_size = 1
    else:
        assert facet_type == "interior"
        # Note that this is only true for owned facets
        support_size = 2

    support_cells_renum = np.empty((iterset.local_size, support_size), dtype=IntType)
    for facet_plex in facet_plex_indices.indices:
        facet_renum = facet_numbering.getOffset(facet_plex)
        for i, support_cell_plex in enumerate(plex.getSupport(facet_plex)):
            support_cell_renum = cell_numbering.getOffset(support_cell_plex)
            support_cells_renum[facet_renum, i] = support_cell_renum

    # TODO: Ideally only pass an integer as the subaxis size
    axes = op3.AxisTree.from_iterable([iterset.as_axis(), op3.Axis(support_size, "support")])
    return op3.Dat(axes, data=support_cells_renum.flatten())


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

    @utils.cached_property
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

    @utils.cached_property
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


class MeshSequenceTopology(object):
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

    @utils.cached_property
    def extruded(self):
        m = self.unique()
        return m.extruded

    def unique(self):
        """Return a single component or raise exception."""
        if len(set(self._meshes)) > 1:
            raise NonUniqueMeshSequenceError(f"Found multiple meshes in {self} where a single mesh is expected")
        m, = set(self._meshes)
        return m


def get_mesh_topologies(expr) -> frozenset[AbstractMeshTopology]:
    """Return all `AbstractMeshTopology` objects associated with the expression.

    This valuable as we often like to use the mesh topologies as 'heavy' caches.

    """
    # FIXME: This isn't valid for certain inputs (e.g. ZeroBaseForm) but this
    # is a very heavy-handed way to fix that
    try:
        return frozenset({d.topology for d in extract_domains(expr)})
    except:
        return frozenset()
