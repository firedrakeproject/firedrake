# A module implementing strong (Dirichlet) boundary conditions.
import utils
import numpy as np
from ufl import as_ufl, UFLException
import types
from core_types import Function
from expression import Expression, to_expression
from projection import project
import pyop2 as op2


class DirichletBC(object):
    '''Implementation of a strong Dirichlet boundary condition.

    :arg V: the :class:`.FunctionSpace` on which the boundary condition
        should be applied.
    :arg g: the boundary condition values. This can be a :class:`.Function` on
        ``V``, an :class:`.Expression`, an iterable of literal constants
        (converted to an :class:`.Expression`), or a literal constant
        which can be pointwise evaluated at the nodes of
        ``V``. :class:`.Expression`\s are projected onto ``V`` if it
        does not support pointwise evaluation.
    :arg sub_domain: the integer id of the boundary region over which the
        boundary condition should be applied. In the case of extrusion
        the ``top`` and ``bottom`` strings are used to flag the bcs application on
        the top and bottom boundaries of the extruded mesh respectively.
    '''

    def __init__(self, V, g, sub_domain):
        if not isinstance(g, Expression):
            try:
                # Bare constant?
                as_ufl(g)
            except UFLException:
                try:
                    # List of bare constants? Convert to Expression
                    g = to_expression(g)
                except:
                    raise ValueError("%r is not a valid DirichletBC expression" % (g,))
        if isinstance(g, Expression):
            try:
                g = Function(V).interpolate(g)
            # Not a point evaluation space, need to project onto V
            except NotImplementedError:
                g = project(g, V)
        self._function_space = V
        self.function_arg = g
        self._original_arg = g
        self.sub_domain = sub_domain

    def function_space(self):
        '''The :class:`.FunctionSpace` on which this boundary condition should
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
            :class:`.DirichletBC` for valid values.
        '''
        self.function_arg = val

    @utils.cached_property
    def nodes(self):
        '''The list of nodes at which this boundary condition applies.'''

        fs = self._function_space
        if self.sub_domain == "bottom":
            return fs.bottom_nodes()
        elif self.sub_domain == "top":
            return fs.top_nodes()
        else:
            if fs.extruded:
                base_maps = fs.exterior_facet_boundary_node_map.values_with_halo.take(
                    fs._mesh._old_mesh.exterior_facets.subset(self.sub_domain).indices,
                    axis=0)
                facet_offset = fs.exterior_facet_boundary_node_map.offset
                return np.unique(np.concatenate([base_maps + i * facet_offset
                                                 for i in range(fs._mesh.layers - 1)]))
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
        """Apply this boundary condition to ``r``.

        :arg r: a :class:`.Function` or :class:`.Matrix` to which the
            boundary condition should be applied.

        :arg u: an optional current state.  If ``u`` is supplied then
            ``r`` is taken to be a residual and the boundary condition
            nodes are set to the value ``u-bc``.  Supplying ``u`` has
            no effect if ``r`` is a :class:`.Matrix` rather than a
            :class:`.Function`. If ``u`` is absent, then the boundary
            condition nodes of ``r`` are set to the boundary condition
            values.


        If ``r`` is a :class:`.Matrix`, it will be assembled with a 1
        on diagonals where the boundary condition applies and 0 in the
        corresponding rows and columns.

        """

        if isinstance(r, types.Matrix):
            r.add_bc(self)
            return
        # If this BC is defined on a subspace of a mixed function space, make
        # sure we only apply to the appropriate subspace of the Function r
        fs = self._function_space
        if fs.index is not None:
            r = Function(self._function_space, r.dat[fs.index])
        if u:
            if fs.index is not None:
                u = Function(fs, u.dat[fs.index])
            r.assign(u - self.function_arg, subset=self.node_set)
        else:
            r.assign(self.function_arg, subset=self.node_set)

    def zero(self, r):
        """Zero the boundary condition nodes on ``r``.

        :arg r: a :class:`.Function` to which the
            boundary condition should be applied.

        """
        if isinstance(r, types.Matrix):
            raise NotImplementedError("Zeroing bcs on a Matrix is not supported")

        self.homogenize()
        self.apply(r)
        self.restore()
