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


__all__ = ['adapt', 'AnisotropicAdaptation']


class AdaptationBase(with_metaclass(abc.ABCMeta)):

    def __init__(self, mesh):
        """
        Abstract top-level class for mesh adaptation
        """
        self.mesh = mesh

    @abc.abstractproperty
    def adapted_mesh(self):
        pass

    @abc.abstractmethod
    def transfer_solution(self, f, **kwargs):
        pass


class AnisotropicAdaptation(AdaptationBase):

    def __init__(self, mesh, metric):
        """
        Object that performs anisotropic mesh adaptation
        """
        if isinstance(mesh.topology, fmesh.ExtrudedMeshTopology):
            raise NotImplementedError("Cannot adapt extruded meshes")
        if mesh.coordinates.ufl_element().family() != 'Lagrange' \
           or mesh.coordinates.ufl_element().degree() != 1:
            raise NotImplementedError("Mesh coordinates must be P1.")
        if metric.ufl_element().family() != 'Lagrange' \
           or metric.ufl_element().degree() != 1:
            raise ValueError("Metric should be a P1 field.")
        super(AnisotropicAdaptation, self).__init__(mesh)
        self.metric = metric

    @utils.cached_property
    def adapted_mesh(self):
        """
        Generates the adapted mesh wrt the metric
        """
        plex = self.mesh.topology._plex
        dim = self.mesh._topological_dimension
        # PETSc adapt routine expects that the right coordinate section is set
        # hence the following bloc of code
        entity_dofs = np.zeros(dim+1, dtype=np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        coordSection = dmplex.create_section(self.mesh, entity_dofs)
        dmCoords = plex.getCoordinateDM()
        dmCoords.setDefaultSection(coordSection)
        coords_local = dmCoords.createLocalVec()
        coords_local.array[:] = np.reshape(self.mesh.coordinates.dat.data_ro_with_halos, coords_local.array.shape)
        plex.setCoordinatesLocal(coords_local)

        dmMetric = dmCoords.clone()
        entity_dofs = np.zeros(dim+1, dtype=np.int32)
        entity_dofs[0] = dim*dim
        msection = dmplex.create_section(self.mesh, entity_dofs)
        dmMetric.setDefaultSection(msection)
        metric_local = dmMetric.createLocalVec()
        metric_local.array[:] = np.reshape(self.metric.dat.data_ro_with_halos, metric_local.array.shape)
        reordered_metric = dmplex.to_petsc_local_numbering(metric_local, self.metric.function_space())

        # TODO inner facets tags will be lost. Do we want a test and/or a warning ?

        new_plex = plex.adaptMetric(reordered_metric, "Face Sets")
        new_mesh = Mesh(new_plex, distribute=False)
        return new_mesh

    def transfer_solution(self, *fields, **kwargs):
        """
        Transfers a solution field from the old mesh to the new mesh

        :arg fields: tuple of functions defined on the old mesh that one wants to transfer
        """
        # method = kwargs.get('method')  # only way I can see to make it work for now. With python 3 I can put it back in the parameters
        fields_new = ()
        for f in fields:
            # TODO other checks ?
            V_new = functionspace.FunctionSpace(self.adapted_mesh, f.function_space().ufl_element())
            f_new = function.Function(V_new)

            if f.ufl_element().family() == 'Lagrange' and f.ufl_element().degree() == 1:
                f_new.dat.data[:] = f.at(self.adapted_mesh.coordinates.dat.data)
            elif f.ufl_element().family() == 'Lagrange':
                degree = f.ufl_element().degree()
                C = functionspace.VectorFunctionSpace(self.adapted_mesh, 'CG', degree)
                interp_coordinates = function.Function(C)
                interp_coordinates.interpolate(self.adapted_mesh.coordinates)
                f_new.dat.data[:] = f.at(interp_coordinates.dat.data)
            else:
                raise NotImplementedError("Can only interpolate CG fields")
            fields_new += (f_new,)
        return fields_new


def adapt(mesh, metric):
    """
    Adapt the mesh to a prescribed metric field.

    :arg mesh: the base mesh to adapt
    :arg metric: a metric tensor field (a Function of a TensorFunctionSpace)

    :return: a new mesh adapted to the metric
    """
    adaptor = AnisotropicAdaptation(mesh, metric)
    return adaptor.adapted_mesh
