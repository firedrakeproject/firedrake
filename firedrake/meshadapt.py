from abc import ABCMeta, abstractmethod
import numpy as np

import firedrake.cython.dmcommon as dmcommon
import firedrake.function as func
import firedrake.functionspace as fs
import firedrake.mesh as fmesh
from firedrake.petsc import PETSc
import firedrake.utils as utils
import ufl


__all__ = []


class Metric(object):
    """
    Abstract class that defines the API for all metrics.
    """

    __metaclass__ = ABCMeta

    def __init__(self, mesh, metric_parameters={}):
        """
        :arg mesh: mesh upon which to build the metric
        """
        mesh.init()
        self.dim = mesh.topological_dimension()
        if self.dim not in (2, 3):
            raise ValueError(f"Mesh must be 2D or 3D, not {self.dim}")
        self.mesh = mesh
        self.update_plex_coordinates()
        self.set_metric_parameters(metric_parameters)

    # TODO: This will be redundant at some point
    def update_plex_coordinates(self):
        """
        Ensure that the coordinates of the Firedrake mesh and
        the underlying DMPlex are consistent.
        """
        self.plex = self.mesh.topology_dm
        entity_dofs = np.zeros(self.dim + 1, dtype=np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        coord_section = self.mesh.create_section(entity_dofs)
        # FIXME: section doesn't have any fields, but PETSc assumes it to have one
        coord_dm = self.plex.getCoordinateDM()
        coord_dm.setDefaultSection(coord_section)
        coords_local = coord_dm.createLocalVec()
        coords_local.array[:] = np.reshape(
            self.mesh.coordinates.dat.data_ro_with_halos, coords_local.array.shape
        )
        self.plex.setCoordinatesLocal(coords_local)

    @abstractmethod
    def set_parameters(self, metric_parameters={}):
        """
        Set the :attr:`metric_parameters` so that they can be
        used to drive the mesh adaptation routine.
        """
        pass
