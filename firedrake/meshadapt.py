from __future__ import absolute_import, print_function, division

import numpy as np

from firedrake.mesh import Mesh
import firedrake.dmplex as dmplex


__all__ = ['adapt']


class AAdaptation(object):
    """
    Object that performs anisotropic mesh adaptation
    """

    def __init__(self, mesh, metric):
        """
        """
        self.mesh = mesh
        self.metric = metric
        self.meshnew = None

    def adapt(self):
        """
        Adapt the current mesh to the metric field.

        :return: the new mesh adapted to the metric
        """
        plex = self.mesh.topology._plex
        dim = self.mesh._topological_dimension
        entity_dofs = np.zeros(dim+1, dtype=np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        coordSection = plex.createSection([1], entity_dofs, perm=self.mesh.topology._plex_renumbering)
        dmCoords = plex.getCoordinateDM()
        dmCoords.setDefaultSection(coordSection)

        with self.mesh.coordinates.dat.vec_ro as coords:
            plex.setCoordinatesLocal(coords)
        with self.metric.dat.vec as vec:
            reordered_metric = dmplex.reorder_metric(plex, vec, coordSection)
        newplex = plex.adapt(reordered_metric)

        self.newmesh = Mesh(newplex)
        return self.newmesh

    def transfer_solution(self, f, fnew):
        """
        Transfers a solution field from the old mesh to the new mesh

        :arg f: function defined on the old mesh that one wants to transfer
        :arg fnew: tranfered function defined on the new mesh
        """
        if self.newmesh is None:
            raise("Cannot transfer solution before generating adapted mesh")
        # TODO many checks
        fnew.dat.data[:] = f.at(self.meshnew.coordinates.dat.data)


def adapt(mesh, metric):
    """
    Adapt the mesh to a prescribed metric field.

    :arg mesh: the base mesh to adapt
    :arg metric: a metric tensor field (a Function of a TensorFunctionSpace)

    :return: a new mesh adapted to the metric
    """
    adaptor = AAdaptation(mesh, metric)
    newmesh = adaptor.adapt()
    return newmesh
