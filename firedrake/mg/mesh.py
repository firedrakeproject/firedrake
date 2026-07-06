import numpy as np
from fractions import Fraction
from collections import defaultdict
from collections.abc import Sequence

from pyop2.datatypes import IntType

import petsctools
import firedrake
from functools import cached_property

from firedrake.mesh import DISTRIBUTION_PARAMETERS_NOOP
from firedrake import utils
from firedrake.cython import mgimpl as impl
from .utils import (RedistributedMeshTransfer, dm_has_empty_rank, make_unoverlapped_dm,
                    redistribute_dm, set_level, set_dm_refine_level)

__all__ = ("HierarchyBase", "MeshHierarchy", "ExtrudedMeshHierarchy", "NonNestedHierarchy",
           "SemiCoarsenedExtrudedHierarchy", "SubmeshHierarchy")


class HierarchyBase(object):
    """Create an encapsulation of an hierarchy of meshes.

    :arg meshes: list of meshes (coarse to fine)
    :arg coarse_to_fine_cells: list of numpy arrays for each level
       pair, mapping each coarse cell into fine cells it intersects.
    :arg fine_to_coarse_cells: list of numpy arrays for each level
       pair, mapping each fine cell into coarse cells it intersects.
    :arg refinements_per_level: number of mesh refinements each
       multigrid level should "see".
    :arg nested: Is this mesh hierarchy nested?

    .. note::

       Most of the time, you do not need to create this object
       yourself, instead using :func:`MeshHierarchy`,
       :func:`ExtrudedMeshHierarchy`, or :func:`NonNestedHierarchy`.
    """
    def __init__(self, meshes, coarse_to_fine_cells, fine_to_coarse_cells,
                 refinements_per_level=1, nested=False):
        petsctools.cite("Mitchell2016")
        self._meshes = tuple(meshes)
        self.meshes = tuple(meshes[::refinements_per_level])
        self.coarse_to_fine_cells = coarse_to_fine_cells
        self.fine_to_coarse_cells = fine_to_coarse_cells
        self.refinements_per_level = refinements_per_level
        self.nested = nested
        for level, m in enumerate(meshes):
            set_level(m, self, Fraction(level, refinements_per_level))
        for level, m in enumerate(self):
            set_level(m, self, level)
        self._shared_data_cache = defaultdict(dict)

    @cached_property
    def comm(self):
        comm = self[0].comm
        if not all(m.comm == comm for m in self):
            raise NotImplementedError("All meshes in hierarchy must be on same communicator")
        return comm

    def __iter__(self):
        """Iterate over the hierarchy of meshes from coarsest to finest"""
        for m in self.meshes:
            yield m

    def __len__(self):
        """Return the size of hierarchy"""
        return len(self.meshes)

    def __getitem__(self, idx):
        """Return a mesh in the hierarchy

        :arg idx: The :func:`~.Mesh` to return"""
        return self.meshes[idx]


def MeshHierarchy(mesh, refinement_levels,
                  refinements_per_level=1,
                  netgen_flags=False,
                  reorder=None,
                  distribution_parameters=None, callbacks=None,
                  mesh_builder=firedrake.Mesh,
                  redistribute=True):
    """Build a hierarchy of meshes by uniformly refining a coarse mesh.

    Parameters
    ----------
    mesh : MeshGeometry
        the coarse mesh to refine
    refinement_levels : int
        the number of levels of refinement
    refinements_per_level : int
        the number of refinements for each level in the hierarchy.
    netgen_flags : bool, dict
        either a bool or a dictionary containing options for Netgen.
        If not False the hierachy is constructed using ngsPETSc, if
        None hierarchy constructed in a standard manner.
    distribution_parameters : dict
        options controlling mesh distribution, see :py:func:`.Mesh`
        for details.  If ``None``, use the same distribution
        parameters as were used to distribute the coarse mesh,
        otherwise, these options override the default.
    redistribute : bool
        If ``True``, redistribute refined meshes when this is needed to
        avoid empty ranks.  Transfer operators use an internal
        parent-owned mesh before moving data to or from the redistributed
        mesh.
    reorder : bool
        optional flag indicating whether to reorder the
        refined meshes.
    callbacks : tuple
        A 2-tuple of callbacks to call before and
        after refinement of the DM.  The before callback receives
        the DM to be refined (and the current level), the after
        callback receives the refined DM (and the current level).
    mesh_builder
        Function to turn a DM into a ``Mesh``. Used by pyadjoint.
    Returns
    -------
    A :py:class:`HierarchyBase` object representing the
    mesh hierarchy.
    """

    if (isinstance(netgen_flags, bool) and netgen_flags) or isinstance(netgen_flags, dict):
        utils.check_netgen_installed()
        from firedrake.mg.netgen import NetgenHierarchy

        if hasattr(mesh, "netgen_mesh"):
            return NetgenHierarchy(mesh, refinement_levels, flags=netgen_flags)
        else:
            raise RuntimeError("Cannot create a NetgenHierarchy from a mesh that has not been generated by\
                                Netgen.")

    parameters = {}
    if distribution_parameters is not None:
        parameters.update(distribution_parameters)
    else:
        parameters.update(mesh._distribution_parameters)

    if callbacks is not None:
        before, after = callbacks
    else:
        before = after = lambda dm, i: None

    mesh_parameters = dict(parameters)
    mesh_parameters["partition"] = False

    redist_parameters = dict(parameters)
    redist_parameters["partition"] = True

    meshes = [mesh]
    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    cmesh = mesh
    for i in range(refinement_levels*refinements_per_level):
        cdm = make_unoverlapped_dm(cmesh)
        clgmaps = (impl.create_lgmap(cdm), impl.create_lgmap(cmesh.topology_dm))

        if i % refinements_per_level == 0:
            before(cdm, i)
        rdm = cdm.refine()
        if i % refinements_per_level == 0:
            after(rdm, i)
        rdm.removeLabel("pyop2_core")
        rdm.removeLabel("pyop2_owned")
        rdm.removeLabel("pyop2_ghost")
        # Fix up coords if refining embedded circle or sphere
        if hasattr(mesh, '_radius'):
            # FIXME, really we need some CAD-like representation
            # of the boundary we're trying to conform to.  This
            # doesn't DTRT really for cubed sphere meshes (the
            # refined meshes are no longer gnonomic).
            coords = cdm.getCoordinatesLocal().array.reshape(-1, mesh.geometric_dimension)
            scale = mesh._radius / np.linalg.norm(coords, axis=1).reshape(-1, 1)
            coords *= scale

        fine_lgmap_without_overlap = impl.create_lgmap(rdm)
        needs_redist = (redistribute and mesh.comm.size > 1
                        and dm_has_empty_rank(rdm))
        if needs_redist:
            transfer_mesh = mesh_builder(
                rdm,
                dim=mesh.geometric_dimension,
                distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
                reorder=reorder,
                comm=mesh.comm,
            )

            rdm_redist = rdm.clone()
            _, point_sf = redistribute_dm(rdm_redist, redist_parameters)
            fine_mesh = mesh_builder(
                rdm_redist,
                dim=mesh.geometric_dimension,
                distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
                reorder=reorder,
                comm=mesh.comm,
            )
            fine_mesh._distribution_parameters = parameters
            fine_mesh.redist = RedistributedMeshTransfer(transfer_mesh, fine_mesh, point_sf)
        else:
            fine_mesh = mesh_builder(
                rdm,
                dim=mesh.geometric_dimension,
                distribution_parameters=mesh_parameters,
                reorder=reorder,
                comm=mesh.comm,
            )
            transfer_mesh = fine_mesh

        flgmaps = (
            fine_lgmap_without_overlap,
            impl.create_lgmap(transfer_mesh.topology_dm),
        )

        c2f, f2c = impl.coarse_to_fine_cells(cmesh, transfer_mesh,
                                             clgmaps, flgmaps)
        coarse_to_fine_cells.append(c2f)
        fine_to_coarse_cells.append(f2c)
        meshes.append(fine_mesh)
        cmesh = fine_mesh

    for i, m in enumerate(meshes):
        set_dm_refine_level(m, i)

    coarse_to_fine_cells = dict((Fraction(i, refinements_per_level), c2f)
                                for i, c2f in enumerate(coarse_to_fine_cells))
    fine_to_coarse_cells = dict((Fraction(i, refinements_per_level), f2c)
                                for i, f2c in enumerate(fine_to_coarse_cells))
    return HierarchyBase(meshes, coarse_to_fine_cells, fine_to_coarse_cells,
                         refinements_per_level, nested=True)


def ExtrudedMeshHierarchy(base_hierarchy, height, base_layer=-1, refinement_ratio=2, layers=None,
                          kernel=None, extrusion_type='uniform', gdim=None,
                          mesh_builder=firedrake.ExtrudedMesh):
    """Build a hierarchy of extruded meshes by extruding a hierarchy of meshes.

    :arg base_hierarchy: the unextruded base mesh hierarchy to extrude.
    :arg height: the height of the domain to extrude to. This is in contrast
       to the extrusion routines, which take in layer_height, the height of
       an individual layer. This is because when refining in the extruded
       dimension, the height of an individual layer will vary.
    :arg base_layer: the number of layers to use the extrusion of the coarsest
       grid.
    :arg refinement_ratio: the ratio by which base_layer should be increased
       on every refinement. refinement_ratio = 2 means standard uniform
       refinement. refinement_ratio = 1 means to not refine in the extruded
       dimension, i.e. the multigrid hierarchy will use semicoarsening.
    :arg layers: as an alternative to specifying base_layer and refinement_ratio,
       one may specify directly the number of layers to be used by each level
       in the extruded hierarchy. This option cannot be combined with base_layer
       and refinement_ratio. Note that the ratio of successive entries in this
       iterable must be an integer for the multigrid transfer operators to work.
    :arg mesh_builder: function used to turn a ``Mesh`` into an
       extruded mesh. Used by pyadjoint.

    See :func:`~.ExtrudedMesh` for the meaning of the remaining parameters.
    """
    if not isinstance(base_hierarchy, HierarchyBase):
        raise ValueError("Expecting a HierarchyBase, not a %r" % type(base_hierarchy))
    if any(m.cell_set._extruded for m in base_hierarchy):
        raise ValueError("Meshes in base hierarchy must not be extruded")

    if layers is None:
        if base_layer == -1:
            raise ValueError("Must specify number of layers for coarsest grid with base_layer=N")
        layers = [base_layer * refinement_ratio**idx for idx in range(len(base_hierarchy._meshes))]
    else:
        if base_layer != -1:
            raise ValueError("Can't specify both layers and base_layer")

    meshes = [mesh_builder(m, layer, kernel=kernel,
                           layer_height=height/layer,
                           extrusion_type=extrusion_type,
                           gdim=gdim)
              for (m, layer) in zip(base_hierarchy._meshes, layers)]

    return HierarchyBase(meshes,
                         base_hierarchy.coarse_to_fine_cells,
                         base_hierarchy.fine_to_coarse_cells,
                         refinements_per_level=base_hierarchy.refinements_per_level,
                         nested=base_hierarchy.nested)


def SemiCoarsenedExtrudedHierarchy(base_mesh, height, nref=1, base_layer=-1, refinement_ratio=2, layers=None,
                                   kernel=None, extrusion_type='uniform', gdim=None,
                                   mesh_builder=firedrake.ExtrudedMesh):
    """Build a hierarchy of extruded meshes with refinement only in the extruded dimension.

    :arg base_mesh: the unextruded base mesh to extrude.
    :arg nref: Number of refinements.
    :arg height: the height of the domain to extrude to. This is in contrast
       to the extrusion routines, which take in layer_height, the height of
       an individual layer. This is because when refining in the extruded
       dimension, the height of an individual layer will vary.
    :arg base_layer: the number of layers to use the extrusion of the coarsest
       grid.
    :arg refinement_ratio: the ratio by which base_layer should be increased
       on every refinement. refinement_ratio = 2 means standard uniform
       refinement. refinement_ratio = 1 means to not refine in the extruded
       dimension, i.e. the multigrid hierarchy will use semicoarsening.
    :arg layers: as an alternative to specifying base_layer and refinement_ratio,
       one may specify directly the number of layers to be used by each level
       in the extruded hierarchy. This option cannot be combined with base_layer
       and refinement_ratio. Note that the ratio of successive entries in this
       iterable must be an integer for the multigrid transfer operators to work.
    :arg mesh_builder: function used to turn a ``Mesh`` into an
       extruded mesh. Used by pyadjoint.

    See :func:`~.ExtrudedMesh` for the meaning of the remaining parameters.

    See also :func:`~.ExtrudedMeshHierarchy` if you want to extruded a
    hierarchy of unstructured meshes.
    """
    if not isinstance(base_mesh, firedrake.mesh.MeshGeometry):
        raise ValueError(f"Can only extruded a mesh, not a {type(base_mesh)}")
    if base_mesh.cell_set._extruded:
        raise ValueError("Base mesh must not be extruded")
    if layers is None:
        if base_layer == -1:
            raise ValueError("Must specify number of layers for coarsest grid with base_layer=N")
        layers = [base_layer * refinement_ratio**idx for idx in range(nref+1)]
    else:
        if base_layer != -1:
            raise ValueError("Can't specify both layers and base_layer")
        if len(layers) == nref+1:
            raise ValueError("Need to provide a number of layers for every refined mesh. "
                             f"Got {len(layers)}, needed {nref+1}")

    meshes = [mesh_builder(base_mesh, layer, kernel=kernel,
                           layer_height=height/layer,
                           extrusion_type=extrusion_type,
                           gdim=gdim)
              for layer in layers]
    refinements_per_level = 1
    identity = np.arange(base_mesh.cell_set.size, dtype=IntType).reshape(-1, 1)
    coarse_to_fine_cells = dict((Fraction(i, refinements_per_level), identity)
                                for i in range(nref))
    fine_to_coarse_cells = dict((Fraction(i+1, refinements_per_level), identity)
                                for i in range(nref))
    return HierarchyBase(meshes, coarse_to_fine_cells, fine_to_coarse_cells,
                         refinements_per_level=refinements_per_level,
                         nested=True)


def NonNestedHierarchy(*meshes):
    return HierarchyBase(meshes, [None for _ in meshes], [None for _ in meshes],
                         nested=False)


def SubmeshHierarchy(parent_hierarchy: HierarchyBase,
                     subdim: int | None = None,
                     subdomain_id: int | Sequence | None = None,
                     label_name: str | None = None,
                     name: str | None = None,
                     ignore_halo: bool = False,
                     reorder: bool | None = None,
                     comm=None):
    """Build a hierarchy of submeshes from an existing mesh hierarchy.

    Parameters
    ----------
    parent_hierarchy :
        Parent mesh hierarchy.
    subdim :
        Topological dimension of the submesh.
        Defaults to ``mesh.topological_dimension``.
    subdomain_id :
        Subdomain ID representing the submesh.
        If `None` the submesh will cover the entire domain.
        This is useful to obtain a codim-1 submesh over all facets or
        a submesh over a different communicator.
    label_name :
        Name of the label to search ``subdomain_id`` in.
        Defaults to ``'Cell Sets'`` or ``'Face Sets'`` depending on ``subdim``.
    name : str |  None
        Name of the submesh.
        Defaults to ``mesh.name + "_submesh"``·
    ignore_halo :
        Whether to exclude the halo from the submesh.
    reorder :
        Whether to reorder the mesh entities. By default,
        the submesh will be reordered if the parent mesh was reordered.
    comm : PETSc.Comm | None
        An optional sub-communicator to define the submesh.
        By default, the submesh is defined on `mesh.comm`.

    Returns
    -------
    HierarchyBase
        The submesh hierarchy.

    """
    meshes = [firedrake.Submesh(mesh,
                                subdim=subdim, subdomain_id=subdomain_id,
                                label_name=label_name, name=name,
                                ignore_halo=ignore_halo, reorder=reorder,
                                comm=comm)
              for mesh in parent_hierarchy._meshes]

    lgmaps_with_overlap = []
    for i, m in enumerate(meshes):
        lgmaps_with_overlap.append(impl.create_lgmap(m.topology_dm))
        m.topology_dm.setRefineLevel(i)
    lgmaps_without_overlap = lgmaps_with_overlap
    lgmaps = [
        (no, o) for no, o in zip(lgmaps_without_overlap, lgmaps_with_overlap)
    ]
    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    for (coarse, fine), (clgmaps, flgmaps) in zip(zip(meshes[:-1], meshes[1:]),
                                                  zip(lgmaps[:-1], lgmaps[1:])):
        c2f, f2c = impl.coarse_to_fine_cells(coarse, fine, clgmaps, flgmaps)
        coarse_to_fine_cells.append(c2f)
        fine_to_coarse_cells.append(f2c)

    refinements_per_level = parent_hierarchy.refinements_per_level
    coarse_to_fine_cells = dict((Fraction(i, refinements_per_level), c2f)
                                for i, c2f in enumerate(coarse_to_fine_cells))
    fine_to_coarse_cells = dict((Fraction(i, refinements_per_level), f2c)
                                for i, f2c in enumerate(fine_to_coarse_cells))
    return HierarchyBase(meshes, coarse_to_fine_cells, fine_to_coarse_cells,
                         refinements_per_level=refinements_per_level,
                         nested=parent_hierarchy.nested)
