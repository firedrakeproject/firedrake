from __future__ import absolute_import, print_function, division
from six import with_metaclass
import abc

import numpy as np

from firedrake.mesh import Mesh
import firedrake.dmplex as dmplex
import firedrake.utils as utils
import firedrake.functionspace as functionspace
import firedrake.function as function


__all__ = ['adapt']


class BaseAdaptation(with_metaclass(abc.ABCMeta)):
    """
    Abstract top-level class for mesh adaptation
    """

    def __init__(self, mesh):
        """
        """
        self.mesh = mesh

    @abc.abstractproperty
    def newmesh(self):
        pass

    @abc.abstractmethod
    def transfer_solution(self, f, **kwargs):
        pass


class AAdaptation(BaseAdaptation):
    """
    Object that performs anisotropic mesh adaptation
    """

    def __init__(self, mesh, metric):
        """
        """
        # TODO checks on P1, etc should prob be done here
        super(AAdaptation, self).__init__(mesh)
        self.metric = metric

    @utils.cached_property
    def newmesh(self):
        """
        Generates the adapted mesh wrt the metric
        """
        plex = self.mesh.topology._plex
        dim = self.mesh._topological_dimension
        # PETSc adapt routine expects that the right coordinate section is set
        # hence the following bloc of code
        entity_dofs = np.zeros(dim+1, dtype=np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        coordSection = plex.createSection([1], entity_dofs, perm=self.mesh.topology._plex_renumbering)
        dmCoords = plex.getCoordinateDM()
        dmCoords.setDefaultSection(coordSection)

        with self.mesh.coordinates.dat.vec_ro as coords:
            plex.setCoordinatesLocal(coords)
        with self.metric.dat.vec_ro as vec:
            reordered_metric = dmplex.to_petsc_numbering(vec, self.metric.function_space())

        newplex = plex.adapt(reordered_metric)
        new_mesh = Mesh(newplex)
        return new_mesh

    def transfer_solution(self, f, method=None):
        """
        Transfers a solution field from the old mesh to the new mesh

        :arg f: function defined on the old mesh that one wants to transfer
        """
        # TODO many checks
        Vnew = functionspace.FunctionSpace(self.newmesh, f.function_space().ufl_element())
        fnew = function.Function(Vnew)
        fnew.dat.data[:] = f.at(self.newmesh.coordinates.dat.data)
        return fnew


def adapt(mesh, metric):
    """
    Adapt the mesh to a prescribed metric field.

    :arg mesh: the base mesh to adapt
    :arg metric: a metric tensor field (a Function of a TensorFunctionSpace)

    :return: a new mesh adapted to the metric
    """
    adaptor = AAdaptation(mesh, metric)
    return adaptor.newmesh
