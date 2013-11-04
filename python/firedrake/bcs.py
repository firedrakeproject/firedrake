# A module implementing strong (Dirichlet) boundary conditions.
import utils
import numpy as np
import pyop2 as op2


class DirichletBC(object):
    '''Implementation of a strong Dirichlet boundary condition.

    :arg V: the :class:`FunctionSpace` on which the boundary condition
        should be applied.
    :arg g: the boundary condition values. This can be a :class:`Function` on V,
        or an expression (such as a literal constant) which can be pointwise
        evaluated at the nodes of V.
    :arg sub_domain: the integer id of the boundary region over which the
        boundary condition should be applied.
    '''

    def __init__(self, V, g, sub_domain):

        self._function_space = V
        self.function_arg = g
        self._original_arg = g
        self.sub_domain = sub_domain

    def function_space(self):
        '''The :class:`FunctionSpace` on which this boundary condition should
        be applied.'''

        return self._function_space

    def homogenize(self):
        '''Convert this boundary condition into a homogeneous one.

        Set the value to zero.

        '''
        self.function_arg = 0

    def restore(self):
        '''Restore the original value of this boundary condition.

        This uses the value passed on instantiation of the object.'''
        self.function_arg = self._original_arg

    def set_value(self, val):
        '''Set the value of this boundary condition.

        :arg val: The boundary condition values.  See
            :class:`DirichletBC` for valid values.
        '''
        self.function_arg = val

    @utils.cached_property
    def nodes(self):
        '''The list of nodes at which this boundary condition applies.'''

        fs = self._function_space

        return np.unique(
            fs.exterior_facet_boundary_node_map.values_with_halo.take(
                fs._mesh.exterior_facets.subset(self.sub_domain).indices,
                axis=0))

    @utils.cached_property
    def node_set(self):
        '''The subset corresponding to the nodes at which this
        boundary condition applies.'''

        return op2.Subset(self._function_space.node_set, self.nodes)

    def apply(self, r, u=None):
        """Apply this boundary condition to the :class:`Function` ``r``.

        If a current state, ``u``, is supplied then ``r`` is taken to be a
        residual and the boundary condition nodes are set to the value
        ``u-bc``.

        If ``u`` is absent, then the boundary condition nodes of ``r`` are set
        to the boundary condition values."""

        if u:
            r.assign(u-self.function_arg, subset=self.node_set)
        else:
            r.assign(self.function_arg, subset=self.node_set)
