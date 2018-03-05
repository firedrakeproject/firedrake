
import numpy as np
from fractions import Fraction

from firedrake import functionspace
from firedrake import mesh
from . import impl
from .utils import set_level


__all__ = ["MeshHierarchy", "ExtrudedMeshHierarchy"]


class MeshHierarchy(object):
    def __init__(self, m, refinement_levels, refinements_per_level=1, reorder=None,
                 distribution_parameters=None):
        """Build a hierarchy of meshes by uniformly refining a coarse mesh.

        :arg m: the coarse :func:`~.Mesh` to refine
        :arg refinement_levels: the number of levels of refinement
        :arg refinements_per_level: Optional number of refinements per
            level in the resulting hierarchy.  Note that the
            intermediate meshes are still kept, but iteration over the
            mesh hierarchy skips them.
        :arg distribution_parameters: options controlling mesh
            distribution, see :func:`~.Mesh` for details.
        :arg reorder: optional flag indicating whether to reorder the
             refined meshes.
        """
        from firedrake_citations import Citations
        Citations().register("Mitchell2016")
        if m.ufl_cell().cellname() not in ["triangle", "interval"]:
            raise NotImplementedError("Only supported on intervals and triangles")
        if refinements_per_level < 1:
            raise ValueError("Must provide positive number of refinements per level")
        m._plex.setRefinementUniform(True)
        dm_hierarchy = []

        cdm = m._plex
        self.comm = m.comm
        fpoint_ises = []
        if m.comm.size > 1 and m._grown_halos:
            raise RuntimeError("Cannot refine parallel overlapped meshes (make sure the MeshHierarchy is built immediately after the Mesh)")
        if distribution_parameters is None:
            distribution_parameters = {}
        distribution_parameters.update({"partition": False})

        for i in range(refinement_levels*refinements_per_level):
            rdm = cdm.refine()
            fpoint_ises.append(cdm.createCoarsePointIS())
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

            dm_hierarchy.append(rdm)
            cdm = rdm
            # Fix up coords if refining embedded circle or sphere
            if hasattr(m, '_radius'):
                # FIXME, really we need some CAD-like representation
                # of the boundary we're trying to conform to.  This
                # doesn't DTRT really for cubed sphere meshes (the
                # refined meshes are no longer gnonomic).
                coords = cdm.getCoordinatesLocal().array.reshape(-1, m.geometric_dimension())
                scale = m._radius / np.linalg.norm(coords, axis=1).reshape(-1, 1)
                coords *= scale

        hierarchy = [m] + [mesh.Mesh(dm, dim=m.ufl_cell().geometric_dimension(),
                                     distribution_parameters=distribution_parameters,
                                     reorder=reorder)
                           for i, dm in enumerate(dm_hierarchy)]
        for m in hierarchy:
            m._non_overlapped_lgmap = impl.create_lgmap(m._plex)
            m._non_overlapped_nent = []
            for d in range(m._plex.getDimension()+1):
                m._non_overlapped_nent.append(m._plex.getDepthStratum(d))
            m.init()
            m._overlapped_lgmap = impl.create_lgmap(m._plex)

        # On coarse mesh n, a map of consistent cell orientations and
        # vertex permutations for the fine cells on each coarse cell.
        self._cells_vperm = []

        for mc, mf, fpointis in zip(hierarchy[:-1],
                                    hierarchy[1:],
                                    fpoint_ises):
            mc._fpointIS = fpointis
            c2f = impl.coarse_to_fine_cells(mc, mf)
            P1c = functionspace.FunctionSpace(mc, 'CG', 1)
            P1f = functionspace.FunctionSpace(mf, 'CG', 1)
            self._cells_vperm.append(impl.compute_orientations(P1c, P1f, c2f))

        self._hierarchy = tuple(hierarchy[::refinements_per_level])
        self._unskipped_hierarchy = tuple(hierarchy)
        for level, m in enumerate(self):
            set_level(m, self, level)
        # Attach fractional levels to skipped parts
        # This allows us to do magic under the hood so that multigrid
        # on skipping hierarchies still works!
        for level, m in enumerate(hierarchy):
            if level % refinements_per_level == 0:
                continue
            set_level(m, self, Fraction(level, refinements_per_level))
        self.refinements_per_level = refinements_per_level

    def __iter__(self):
        """Iterate over the hierarchy of meshes from coarsest to finest"""
        for m in self._hierarchy:
            yield m

    def __len__(self):
        """Return the size of hierarchy"""
        return len(self._hierarchy)

    def __getitem__(self, idx):
        """Return a mesh in the hierarchy

        :arg idx: The :func:`~.Mesh` to return"""
        return self._hierarchy[idx]


class ExtrudedMeshHierarchy(MeshHierarchy):
    def __init__(self, mesh_hierarchy, layers, kernel=None, layer_height=None,
                 extrusion_type='uniform', gdim=None):
        """Build a hierarchy of extruded meshes by extruded a hierarchy of meshes.

        :arg mesh_hierarchy: the :class:`MeshHierarchy` to extruded

        See :func:`~.ExtrudedMesh` for the meaning of the remaining parameters.
        """
        self.comm = mesh_hierarchy.comm
        self._base_hierarchy = mesh_hierarchy
        hierarchy = [mesh.ExtrudedMesh(m, layers, kernel=kernel,
                                       layer_height=layer_height,
                                       extrusion_type=extrusion_type,
                                       gdim=gdim)
                     for m in mesh_hierarchy._unskipped_hierarchy]
        self._unskipped_hierarchy = tuple(hierarchy)
        self._hierarchy = tuple(hierarchy[::mesh_hierarchy.refinements_per_level])
        self._cells_vperm = self._base_hierarchy._cells_vperm
        for level, m in enumerate(self):
            set_level(m, self, level)
        # Attach fractional levels to skipped parts
        for level, m in enumerate(hierarchy):
            if level % mesh_hierarchy.refinements_per_level == 0:
                continue
            set_level(m, self, Fraction(level, mesh_hierarchy.refinements_per_level))
        self.refinements_per_level = mesh_hierarchy.refinements_per_level
