from __future__ import absolute_import, print_function, division
from six import with_metaclass
import abc

import numpy as np

from firedrake.mesh import Mesh
import firedrake.dmplex as dmplex
import firedrake.utils as utils
import firedrake.functionspace as functionspace
import firedrake.function as function
import firedrake.mesh as fmesh


__all__ = ['adapt', 'AAdaptation']


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
        if isinstance(mesh.topology, fmesh.ExtrudedMeshTopology):
            raise NotImplementedError("Cannot adapt extruded meshes")
        if mesh.coordinates.ufl_element().family() != 'Lagrange' \
           or mesh.coordinates.ufl_element().degree() != 1:
            raise NotImplementedError("Mesh coordinates must be P1.")
        if metric.ufl_element().family() != 'Lagrange' \
           or metric.ufl_element().degree() != 1:
            raise ValueError("Metric should be a P1 field.")
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

        newplex = plex.adapt(reordered_metric, "boundary_ids")
        new_mesh = Mesh(newplex)
        return new_mesh

    def transfer_solution(self, *fields, **kwargs):
        """
        Transfers a solution field from the old mesh to the new mesh

        :arg fields: tuple of functions defined on the old mesh that one wants to transfer
        """
        method = kwargs.get('method')  # only way I can see to make it work for now. With python 3 I can put it back in the parameters
        fields_new = ()
        for f in fields:
            # TODO many other checks ??
            Vnew = functionspace.FunctionSpace(self.newmesh, f.function_space().ufl_element())
            fnew = function.Function(Vnew)

            if f.ufl_element().family() == 'Lagrange' and f.ufl_element().degree() == 1:
                fnew.dat.data[:] = f.at(self.newmesh.coordinates.dat.data)
            elif f.ufl_element().family() == 'Lagrange':
                degree = f.ufl_element().degree()
                C = functionspace.VectorFunctionSpace(self.newmesh, 'CG', degree)
                interp_coordinates = function.Function(C)
                interp_coordinates.interpolate(self.newmesh.coordinates)
                fnew.dat.data[:] = f.at(interp_coordinates.dat.data)
            else:
                raise NotImplementedError("Can only interpolate CG fields")
            fields_new += (fnew,)
        return fields_new


def adapt(mesh, metric):
    """
    Adapt the mesh to a prescribed metric field.

    :arg mesh: the base mesh to adapt
    :arg metric: a metric tensor field (a Function of a TensorFunctionSpace)

    :return: a new mesh adapted to the metric
    """
    adaptor = AAdaptation(mesh, metric)
    return adaptor.newmesh
