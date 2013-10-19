# A module implementing strong (Dirichlet) boundary conditions.
import utils
import numpy as np


class DirichletBC(object):
    '''Implementation of a strong Dirichlet boundary condition.

    :arg V: the :class:`FunctionSpace` on which the boundary condition
        should be applied.
    :arg g: the :class:`GenericFunction` defining the boundary condition
        values.
    :arg sub_domain: the integer id of the boundary region over which the
        boundary condition should be applied.
    '''

    def __init__(self, V, g, sub_domain):

        self._function_space = V
        self.function_arg = g
        self.sub_domain = sub_domain

    def function_space(self):
        '''The :class:`FunctionSpace` on which this boundary condition should
        be applied.'''

        return self._function_space

    @utils.cached_property
    def nodes(self):
        '''The list of nodes at which this boundary condition applies.'''

        return reduce(np.unique,
                      self._function_space.exterior_facet_node_map().values,
                      self._function_space._mesh.exterior_facets.subset(self.sub_domain).indices)
