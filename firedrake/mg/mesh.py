import numpy as np
from fractions import Fraction
from collections import defaultdict

from pyop2.datatypes import IntType
import firedrake
from firedrake.utils import cached_property
from firedrake.halo import _get_mtype as get_mpi_type
from firedrake.halo import MPI
from firedrake.cython import mgimpl as impl
from .utils import set_level
from firedrake.cython.mgimpl import (build_section_migration_sf,
                                     get_entity_renumbering)


__all__ = ("HierarchyBase", "MeshHierarchy", "ExtrudedMeshHierarchy", "NonNestedHierarchy",
           "SemiCoarsenedExtrudedHierarchy", "RedistMeshHierarchy")


class RedistMesh:
    def __init__(self, orig, pointmigrationsf):
        self.orig = orig
        self.pointmigrationsf = pointmigrationsf

    def orig2redist(self, source, target):
        # TODO: cache keyed on UFL element
        source_section = source.function_space().dm.getDefaultSection()
        target_section = target.function_space().dm.getDefaultSection()
        secmigrationsf = build_section_migration_sf(
            self.pointmigrationsf, source_section, target_section
        )
        dtype, _ = get_mpi_type(source.dat)
        secmigrationsf.bcastBegin(dtype,
                                  source.dat.data_ro_with_halos,
                                  target.dat.data_ro_with_halos,
                                  MPI.REPLACE)
        secmigrationsf.bcastEnd(dtype,
                                source.dat.data_ro_with_halos,
                                target.dat.data_ro_with_halos,
                                MPI.REPLACE)

    def redist2orig(self, target, source):
        # TODO: cache keyed on UFL element
        source_section = source.function_space().dm.getDefaultSection()
        target_section = target.function_space().dm.getDefaultSection()
        secmigrationsf = build_section_migration_sf(
            self.pointmigrationsf, source_section, target_section
        )
        dtype, _ = get_mpi_type(source.dat)
        secmigrationsf.reduceBegin(dtype,
                                   target.dat.data_ro_with_halos,
                                   source.dat.data_ro_with_halos,
                                   MPI.REPLACE)
        secmigrationsf.reduceEnd(dtype,
                                 target.dat.data_ro_with_halos,
                                 source.dat.data_ro_with_halos,
                                 MPI.REPLACE)


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
        from firedrake_citations import Citations
        Citations().register("Mitchell2016")
        self._meshes = tuple(meshes)
        self.meshes = tuple(meshes[::refinements_per_level])
        self.coarse_to_fine_cells = coarse_to_fine_cells
        self.fine_to_coarse_cells = fine_to_coarse_cells
        self.refinements_per_level = refinements_per_level
        self.nested = nested
        self._shared_data_cache = defaultdict(dict)
        for level, m in enumerate(meshes):
            set_level(m, self, level)
            if hasattr(m, "redist"):
                set_level(m.redist.orig, self, level)

    @cached_property
    def comm(self):
        comm = self[0].comm
        if not all(m.comm == comm for m in self):
            raise NotImplementedError("All meshes in hierarchy must be on same communicator")
        return comm

    @cached_property
    def _comm(self):
        _comm = self[0]._comm
        if not all(m._comm == _comm for m in self):
            raise NotImplementedError("All meshes in hierarchy must be on same communicator")
        return _comm

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


def RedistMeshHierarchy(cmesh, refinement_levels, refinements_per_level=1,
                        callbacks=None,
                        distribution_parameters=None):
    from firedrake_configuration import get_config
    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    meshes = [cmesh]
    if callbacks is not None:
        before, after = callbacks
    else:
        before = after = lambda dm, i: None
    if distribution_parameters is None:
        distribution_parameters = {}

    for i in range(refinement_levels*refinements_per_level):
        cmesh.init()
        cdm = cmesh.topology_dm

        if i % refinements_per_level == 0:
            before(cdm, i)

        cdm.setRefinementUniform(True)
        _, n2oc = get_entity_renumbering(cdm, cmesh._cell_numbering, "cell")
        rdm = cdm.refine()
        rmesh = firedrake.Mesh(rdm,
                               distribution_parameters={
                                   "partition": False,
                                   "overlap_type": (firedrake.DistributedMeshOverlapType.NONE, 0)
                               })
        rmesh.init()

        o2nf, _ = get_entity_renumbering(rdm, rmesh._cell_numbering, "cell")

        fine_to_coarse = np.empty((rmesh.cell_set.size, 1), dtype=np.int32)
        coarse_to_fine = np.empty((cmesh.cell_set.size, 4), dtype=np.int32)
        for coarse_cell in range(cmesh.cell_set.size):
            c = n2oc[coarse_cell]
            assert 0 <= c < cmesh.cell_set.size
            for i in range(4):
                f = 4 * c + i
                assert 0 <= f < rmesh.cell_set.size
                f = o2nf[f]
                assert 0 <= f < rmesh.cell_set.size
                coarse_to_fine[coarse_cell, i] = f
                fine_to_coarse[f, 0] = coarse_cell

        if rmesh.comm.size == 1:
            rmeshredist = rmesh
            if i % refinements_per_level == 0:
                after(rdm, i)
        else:
            rdmredist = rdm.clone()
            if i % refinements_per_level == 0:
                after(rdmredist, i)
            partitioner_type = distribution_parameters.get("partitioner_type")
            if partitioner_type is None:
                if IntType.itemsize == 8 or rmesh.topology_dm.isDistributed():
                    if get_config().get("options", {}).get("with_parmetis", False):
                        partitioner_type = "parmetis"
                    else:
                        partitioner_type = "ptscotch"
                else:
                    partitioner_type = "chaco"
            part = rdmredist.getPartitioner()
            part.setType({"chaco": part.Type.CHACO,
                          "ptscotch": part.Type.PTSCOTCH,
                          "parmetis": part.Type.PARMETIS}[partitioner_type])
            part.setFromOptions()
            rdmredist.removeLabel("pyop2_ghost")
            rdmredist.removeLabel("pyop2_owned")
            rdmredist.removeLabel("pyop2_core")
            pointmigrationsf = rdmredist.distribute(overlap=1)
            rmeshredist = firedrake.Mesh(
                rdmredist,
                distribution_parameters={
                    "partition": False,
                    "overlap_type": (firedrake.DistributedMeshOverlapType.NONE, 0),
                },
            )
            if pointmigrationsf is None:
                assert rmesh.comm.size == 1
            else:
                rmeshredist.redist = RedistMesh(rmesh, pointmigrationsf)
        meshes.append(rmeshredist)
        coarse_to_fine_cells.append(coarse_to_fine)
        fine_to_coarse_cells.append(fine_to_coarse)
        cmesh = rmeshredist

    coarse_to_fine_cells = dict((Fraction(i, 1), c2f)
                                for i, c2f in enumerate(coarse_to_fine_cells))
    fine_to_coarse_cells = dict((Fraction(i, 1), f2c)
                                for i, f2c in enumerate(fine_to_coarse_cells))

    return HierarchyBase(meshes,
                         coarse_to_fine_cells=coarse_to_fine_cells,
                         fine_to_coarse_cells=fine_to_coarse_cells,
                         refinements_per_level=1,
                         nested=True)


def MeshHierarchy(mesh, refinement_levels,
                  refinements_per_level=1,
                  reorder=None,
                  distribution_parameters=None, callbacks=None,
                  mesh_builder=firedrake.Mesh):
    """Build a hierarchy of meshes by uniformly refining a coarse mesh.

    :arg mesh: the coarse :func:`~.Mesh` to refine
    :arg refinement_levels: the number of levels of refinement
    :arg refinements_per_level: the number of refinements for each
        level in the hierarchy.
    :arg distribution_parameters: options controlling mesh
        distribution, see :func:`~.Mesh` for details.  If ``None``,
        use the same distribution parameters as were used to
        distribute the coarse mesh, otherwise, these options override
        the default.
    :arg reorder: optional flag indicating whether to reorder the
         refined meshes.
    :arg callbacks: A 2-tuple of callbacks to call before and
        after refinement of the DM.  The before callback receives
        the DM to be refined (and the current level), the after
        callback receives the refined DM (and the current level).
    """
    cdm = mesh.topology_dm
    cdm.setRefinementUniform(True)
    dms = []
    if mesh.comm.size > 1 and mesh._grown_halos:
        raise RuntimeError("Cannot refine parallel overlapped meshes "
                           "(make sure the MeshHierarchy is built immediately after the Mesh)")
    parameters = {}
    if distribution_parameters is not None:
        parameters.update(distribution_parameters)
    else:
        parameters.update(mesh._distribution_parameters)

    parameters["partition"] = False
    distribution_parameters = parameters

    if callbacks is not None:
        before, after = callbacks
    else:
        before = after = lambda dm, i: None

    for i in range(refinement_levels*refinements_per_level):
        if i % refinements_per_level == 0:
            before(cdm, i)
        rdm = cdm.refine()
        if i % refinements_per_level == 0:
            after(rdm, i)
        rdm.removeLabel("pyop2_core")
        rdm.removeLabel("pyop2_owned")
        rdm.removeLabel("pyop2_ghost")

        dms.append(rdm)
        cdm = rdm
        # Fix up coords if refining embedded circle or sphere
        if hasattr(mesh, '_radius'):
            # FIXME, really we need some CAD-like representation
            # of the boundary we're trying to conform to.  This
            # doesn't DTRT really for cubed sphere meshes (the
            # refined meshes are no longer gnonomic).
            coords = cdm.getCoordinatesLocal().array.reshape(-1, mesh.geometric_dimension())
            scale = mesh._radius / np.linalg.norm(coords, axis=1).reshape(-1, 1)
            coords *= scale

    meshes = [mesh] + [mesh_builder(dm, dim=mesh.ufl_cell().geometric_dimension(),
                                    distribution_parameters=distribution_parameters,
                                    reorder=reorder, comm=mesh.comm)
                       for dm in dms]

    lgmaps = []
    for i, m in enumerate(meshes):
        no = impl.create_lgmap(m.topology_dm)
        m.init()
        o = impl.create_lgmap(m.topology_dm)
        m.topology_dm.setRefineLevel(i)
        lgmaps.append((no, o))

    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    for (coarse, fine), (clgmaps, flgmaps) in zip(zip(meshes[:-1], meshes[1:]),
                                                  zip(lgmaps[:-1], lgmaps[1:])):
        c2f, f2c = impl.coarse_to_fine_cells(coarse, fine, clgmaps, flgmaps)
        coarse_to_fine_cells.append(c2f)
        fine_to_coarse_cells.append(f2c)

    coarse_to_fine_cells = dict((Fraction(i, refinements_per_level), c2f)
                                for i, c2f in enumerate(coarse_to_fine_cells))
    fine_to_coarse_cells = dict((Fraction(i, refinements_per_level), f2c)
                                for i, f2c in enumerate(fine_to_coarse_cells))
    return HierarchyBase(meshes, coarse_to_fine_cells, fine_to_coarse_cells,
                         refinements_per_level, nested=True)


def ExtrudedMeshHierarchy(base_hierarchy, height, base_layer=-1, refinement_ratio=2, layers=None,
                          kernel=None, extrusion_type='uniform', gdim=None):
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

    layer_height = height/base_layer
    meshes = []
    for m, layer in zip(base_hierarchy._meshes, layers):
        ext = firedrake.ExtrudedMesh(m, layer, kernel=kernel,
                                     layer_height=layer_height,
                                     extrusion_type=extrusion_type,
                                     gdim=gdim)
        meshes.append(ext)
        if hasattr(m, "redist"):
            ext_orig = firedrake.ExtrudedMesh(m.redist.orig, layer, kernel=kernel,
                                              layer_height=layer_height,
                                              extrusion_type=extrusion_type,
                                              gdim=gdim)
            pointmigrationsf = m.redist.pointmigrationsf
            ext.redist = RedistMesh(ext_orig, pointmigrationsf)
        layer_height /= refinement_ratio
        try:
            len(layer_height)
            layer_height = np.repeat(layer_height, refinement_ratio)
        except TypeError:
            pass

    return HierarchyBase(meshes,
                         base_hierarchy.coarse_to_fine_cells,
                         base_hierarchy.fine_to_coarse_cells,
                         refinements_per_level=base_hierarchy.refinements_per_level,
                         nested=base_hierarchy.nested)


def SemiCoarsenedExtrudedHierarchy(base_mesh, height, nref=1, base_layer=-1, refinement_ratio=2, layers=None,
                                   kernel=None, extrusion_type='uniform', gdim=None):
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

    See :func:`~.ExtrudedMesh` for the meaning of the remaining parameters.

    See also :func:`~.ExtrudedMeshHierarchy` if you want to extruded a
    hierarchy of unstructured meshes.
    """
    if not isinstance(base_mesh, firedrake.mesh.MeshGeometry):
        raise ValueError(f"Can only extruded a mesh, not a {type(base_mesh)}")
    base_mesh.init()
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

    meshes = [firedrake.ExtrudedMesh(base_mesh, layer, kernel=kernel,
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
