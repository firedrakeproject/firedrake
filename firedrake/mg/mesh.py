from __future__ import absolute_import

import numpy as np
from pyop2.mpi import MPI

from firedrake import functionspace
from firedrake import mesh
from . import impl
from .utils import set_level


__all__ = ["MeshHierarchy", "ExtrudedMeshHierarchy"]


class MeshHierarchy(object):
    def __init__(self, m, refinement_levels, reorder=None):
        """Build a hierarchy of meshes by uniformly refining a coarse mesh.

        :arg m: the coarse :class:`~.Mesh` to refine
        :arg refinement_levels: the number of levels of refinement
        :arg reorder: optional flag indicating whether to reorder the
             refined meshes.
        """
        if m.ufl_cell().cellname() not in ["triangle", "interval"]:
            raise NotImplementedError("Only supported on intervals and triangles")
        m._plex.setRefinementUniform(True)
        dm_hierarchy = []

        cdm = m._plex
        fpoint_ises = []
        if MPI.comm.size > 1 and m._grown_halos:
            raise RuntimeError("Cannot refine parallel overlapped meshes (make sure the MeshHierarchy is built immediately after the Mesh)")
        for i in range(refinement_levels):
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
            rdm.removeLabel("op2_core")
            rdm.removeLabel("op2_non_core")
            rdm.removeLabel("op2_exec_halo")
            rdm.removeLabel("op2_non_exec_halo")

            dm_hierarchy.append(rdm)
            cdm = rdm
            # Fix up coords if refining embedded circle or sphere
            if hasattr(m, '_circle_manifold'):
                coords = cdm.getCoordinatesLocal().array.reshape(-1, 2)
                scale = m._circle_manifold / np.linalg.norm(coords, axis=1).reshape(-1, 1)
                coords *= scale
            elif hasattr(m, '_icosahedral_sphere'):
                coords = cdm.getCoordinatesLocal().array.reshape(-1, 3)
                scale = m._icosahedral_sphere / np.linalg.norm(coords, axis=1).reshape(-1, 1)
                coords *= scale

        hierarchy = [m] + [mesh.Mesh(dm, dim=m.ufl_cell().geometric_dimension(),
                                     distribute=False, reorder=reorder)
                           for i, dm in enumerate(dm_hierarchy)]
        self._hierarchy = tuple([set_level(o, self, lvl)
                                 for lvl, o in enumerate(hierarchy)])

        for m in self:
            m._non_overlapped_lgmap = impl.create_lgmap(m._plex)
            m._non_overlapped_nent = []
            for d in range(m._plex.getDimension()+1):
                m._non_overlapped_nent.append(m._plex.getDepthStratum(d))
            m.init()
            m._overlapped_lgmap = impl.create_lgmap(m._plex)

        # On coarse mesh n, a map of consistent cell orientations and
        # vertex permutations for the fine cells on each coarse cell.
        self._cells_vperm = []

        for mc, mf, fpointis in zip(self._hierarchy[:-1],
                                    self._hierarchy[1:],
                                    fpoint_ises):
            mc._fpointIS = fpointis
            c2f = impl.coarse_to_fine_cells(mc, mf)
            P1c = functionspace.FunctionSpace(mc, 'CG', 1)
            P1f = functionspace.FunctionSpace(mf, 'CG', 1)
            self._cells_vperm.append(impl.compute_orientations(P1c, P1f, c2f))

    def __iter__(self):
        """Iterate over the hierarchy of meshes from coarsest to finest"""
        for m in self._hierarchy:
            yield m

    def __len__(self):
        """Return the size of hierarchy"""
        return len(self._hierarchy)

    def __getitem__(self, idx):
        """Return a mesh in the hierarchy

        :arg idx: The :class:`~.Mesh` to return"""
        return self._hierarchy[idx]


class ExtrudedMeshHierarchy(MeshHierarchy):
    def __init__(self, mesh_hierarchy, layers, kernel=None, layer_height=None,
                 extrusion_type='uniform', gdim=None):
        """Build a hierarchy of extruded meshes by extruded a hierarchy of meshes.

        :arg mesh_hierarchy: the :class:`MeshHierarchy` to extruded

        See :class:`~.ExtrudedMesh` for the meaning of the remaining parameters.
        """
        self._base_hierarchy = mesh_hierarchy
        hierarchy = [set_level(mesh.ExtrudedMesh(m, layers, kernel=kernel,
                                                 layer_height=layer_height,
                                                 extrusion_type=extrusion_type,
                                                 gdim=gdim),
                               self, lvl)
                     for lvl, m in enumerate(mesh_hierarchy)]
        self._hierarchy = tuple(hierarchy)
        self._cells_vperm = self._base_hierarchy._cells_vperm
