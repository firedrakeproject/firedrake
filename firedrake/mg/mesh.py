import numpy as np
from collections import defaultdict

import firedrake
from firedrake.utils import cached_property
from . import impl
from .utils import set_level


__all__ = ["MeshHierarchy", "ExtrudedMeshHierarchy", "NonNestedHierarchy"]


class HierarchyBase(object):
    def __init__(self, meshes, coarse_to_fine_cells, fine_to_coarse_cells):
        from firedrake_citations import Citations
        Citations().register("Mitchell2016")
        self.meshes = tuple(meshes)
        self.coarse_to_fine_cells = tuple(coarse_to_fine_cells)
        self.fine_to_coarse_cells = tuple(fine_to_coarse_cells)
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


def MeshHierarchy(mesh, refinement_levels, reorder=None,
                  distribution_parameters=None, callbacks=None):
    """Build a hierarchy of meshes by uniformly refining a coarse mesh.

    :arg mesh: the coarse :func:`~.Mesh` to refine
    :arg refinement_levels: the number of levels of refinement
    :arg distribution_parameters: options controlling mesh
        distribution, see :func:`~.Mesh` for details.
    :arg reorder: optional flag indicating whether to reorder the
         refined meshes.
    :arg callbacks: A 2-tuple of callbacks to call before and
        after refinement of the DM.  The before callback receives
        the DM to be refined (and the current level), the after
        callback receives the refined DM (and the current level).
    """
    cdm = mesh._plex
    cdm.setRefinementUniform(True)
    dms = []
    if mesh.comm.size > 1 and mesh._grown_halos:
        raise RuntimeError("Cannot refine parallel overlapped meshes "
                           "(make sure the MeshHierarchy is built immediately after the Mesh)")
    if distribution_parameters is None:
        distribution_parameters = {}
    distribution_parameters.update({"partition": False})

    if callbacks is not None:
        before, after = callbacks
    else:
        before = after = lambda dm, i: None

    for i in range(refinement_levels):
        before(cdm, i)
        rdm = cdm.refine()
        after(rdm, i)
        # Remove interior facet label (re-construct from
        # complement of exterior facets).  Necessary because the
        # refinement just marks points "underneath" the refined
        # facet with the appropriate label.  This works for
        # exterior, but not marked interior facets
        rdm.removeLabel("interior_facets")
        # Remove vertex (and edge) points from labels on exterior
        # facets.  Interior facets will be relabeled in Mesh
        # construction below.
        impl.filter_exterior_facet_labels(rdm)
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

    meshes = [mesh] + [firedrake.Mesh(dm, dim=mesh.ufl_cell().geometric_dimension(),
                                      distribution_parameters=distribution_parameters,
                                      reorder=reorder)
                       for dm in dms]

    lgmaps = []
    for i, m in enumerate(meshes):
        no = impl.create_lgmap(m._plex)
        m.init()
        o = impl.create_lgmap(m._plex)
        m._plex.setRefineLevel(i)
        lgmaps.append((no, o))

    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    for (coarse, fine), (clgmaps, flgmaps) in zip(zip(meshes[:-1], meshes[1:]),
                                                  zip(lgmaps[:-1], lgmaps[1:])):
        c2f, f2c = impl.coarse_to_fine_cells(coarse, fine, clgmaps, flgmaps)
        coarse_to_fine_cells.append(c2f)
        fine_to_coarse_cells.append(f2c)

    return HierarchyBase(meshes, coarse_to_fine_cells, fine_to_coarse_cells)


def ExtrudedMeshHierarchy(base_hierarchy, layers, kernel=None, layer_height=None,
                          extrusion_type='uniform', gdim=None):
    """Build a hierarchy of extruded meshes by extruded a hierarchy of meshes.

    :arg base_hierarchy: the unextruded base mesh hierarchy to extrude.

    See :func:`~.ExtrudedMesh` for the meaning of the remaining parameters.
    """
    if not isinstance(base_hierarchy, HierarchyBase):
        raise ValueError("Expecting a HierarchyBase, not a %r" % type(base_hierarchy))
    if any(m.cell_set._extruded for m in base_hierarchy):
        raise ValueError("Meshes in base hierarchy must not be extruded")

    meshes = [firedrake.ExtrudedMesh(m, layers, kernel=kernel,
                                     layer_height=layer_height,
                                     extrusion_type=extrusion_type,
                                     gdim=gdim)
              for m in base_hierarchy]

    return HierarchyBase(meshes,
                         base_hierarchy.coarse_to_fine_cells,
                         base_hierarchy.fine_to_coarse_cells)


def NonNestedHierarchy(*meshes):
    return HierarchyBase(meshes, [None for _ in meshes], [None for _ in meshes])
