# A module implementing strong (Dirichlet) boundary conditions.
import numpy as np

import functools
import itertools

import ufl
from ufl import as_ufl, SpatialCoordinate, UFLException, as_tensor
from ufl.algorithms.analysis import has_type
import finat

import pyop2 as op2
from pyop2.profiling import timed_function
from pyop2 import exceptions
from pyop2.utils import as_tuple

import firedrake.expression as expression
import firedrake.function as function
import firedrake.matrix as matrix
import firedrake.projection as projection
import firedrake.utils as utils
from firedrake import ufl_expr
from firedrake import slate


__all__ = ['DirichletBC', 'homogenize', 'EquationBC']


class BCBase(object):
    r'''Implementation of a base class of Dirichlet-like boundary conditions.

    :arg V: the :class:`.FunctionSpace` on which the boundary condition
        should be applied.
    :arg sub_domain: the integer id(s) of the boundary region over which the
        boundary condition should be applied. The string "on_boundary" may be used
        to indicate all of the boundaries of the domain. In the case of extrusion
        the ``top`` and ``bottom`` strings are used to flag the bcs application on
        the top and bottom boundaries of the extruded mesh respectively.
    :arg method: the method for determining boundary nodes. The default is
        "topological", indicating that nodes topologically associated with a
        boundary facet will be included. The alternative value is "geometric",
        which indicates that nodes associated with basis functions which do not
        vanish on the boundary will be included. This can be used to impose
        strong boundary conditions on DG spaces, or no-slip conditions on HDiv spaces.
    '''
    def __init__(self, V, sub_domain, method="topological"):

        # First, we bail out on zany elements.  We don't know how to do BC's for them.
        if isinstance(V.finat_element, (finat.Argyris, finat.Morley, finat.Bell)) or \
           (isinstance(V.finat_element, finat.Hermite) and V.mesh().topological_dimension() > 1):
            raise NotImplementedError("Strong BCs not implemented for element %r, use Nitsche-type methods until we figure this out" % V.finat_element)
        self._function_space = V
        self.comm = V.comm
        self.sub_domain = sub_domain
        if method not in ["topological", "geometric"]:
            raise ValueError("Unknown boundary condition method %s" % method)
        self.method = method
        if V.extruded and V.component is not None:
            raise NotImplementedError("Indexed VFS bcs not implemented on extruded meshes")
        # If this BC is defined on a subspace (IndexedFunctionSpace or
        # ComponentFunctionSpace, possibly recursively), pull out the appropriate
        # indices.
        components = []
        indexing = []
        indices = []
        fs = self._function_space
        while True:
            # Add index to indices if found
            if fs.index is not None:
                indexing.append(fs.index)
                indices.append(fs.index)
            if fs.component is not None:
                components.append(fs.component)
                indices.append(fs.component)
            # Now try the parent
            if fs.parent is not None:
                fs = fs.parent
            else:
                # All done
                break
        # Used for indexing functions passed in.
        self._indices = tuple(reversed(indices))
        # Used for finding local to global maps with boundary conditions applied
        self._cache_key = (self.domain_args, (self.method, tuple(indexing), tuple(components)))
        self.bcs = []

    def __iter__(self):
        yield self
        yield from itertools.chain(*self.bcs)

    def function_space(self):
        '''The :class:`.FunctionSpace` on which this boundary condition should
        be applied.'''

        return self._function_space

    @utils.cached_property
    def domain_args(self):
        r"""The sub_domain the BC applies to."""
        # Define facet, edge, vertex using tuples:
        # Ex in 3D:
        #           user input                                                         returned keys
        # facet  = ((1, ), )                                  ->     ((2, ((1, ), )), (1, ()),         (0, ()))
        # edge   = ((1, 2), )                                 ->     ((2, ()),        (1, ((1, 2), )), (0, ()))
        # vertex = ((1, 2, 4), )                              ->     ((2, ()),        (1, ()),         (0, ((1, 2, 4), ))
        #
        # Multiple facets:
        # (1, 2, 4) := ((1, ), (2, ), (4,))                   ->     ((2, ((1, ), (2, ), (4, ))), (1, ()), (0, ()))
        #
        # One facet and two edges:
        # ((1,), (1, 3), (1, 4))                              ->     ((2, ((1,),)), (1, ((1,3), (1, 4))), (0, ()))
        #

        sub_d = self.sub_domain
        # if string, return
        if isinstance(sub_d, str):
            return (sub_d, )
        # convert: i -> (i, )
        sub_d = as_tuple(sub_d)
        # convert: (i, j, (k, l)) -> ((i, ), (j, ), (k, l))
        sub_d = [as_tuple(i) for i in sub_d]

        ndim = self.function_space().mesh()._plex.getDimension()
        sd = [[] for _ in range(ndim)]
        for i in sub_d:
            sd[ndim - len(i)].append(i)
        s = []
        for i in range(ndim):
            s.append((ndim - 1 - i, as_tuple(sd[i])))
        return as_tuple(s)

    @utils.cached_property
    def nodes(self):
        '''The list of nodes at which this boundary condition applies.'''

        def hermite_stride(bcnodes):
            if isinstance(self._function_space.finat_element, finat.Hermite) and \
               self._function_space.mesh().topological_dimension() == 1:
                return bcnodes[::2]  # every second dof is the vertex value
            else:
                return bcnodes

        sub_d = self.sub_domain
        if isinstance(sub_d, str):
            return hermite_stride(self._function_space.boundary_nodes(sub_d, self.method))
        else:
            sub_d = as_tuple(sub_d)
            sub_d = [as_tuple(s) for s in sub_d]
            bcnodes = []
            for s in sub_d:
                # s is of one of the following formats:
                # facet: (i, )
                # edge: (i, j)
                # vertex: (i, j, k)

                # take intersection of facet nodes, and add it to bcnodes
                bcnodes1 = []
                if len(s) > 1 and not isinstance(self._function_space.finat_element, finat.Lagrange):
                    raise TypeError("Currently, edge conditions have only been tested with Lagrange elements")
                for ss in s:
                    # intersection of facets
                    # Edge conditions have only been tested with Lagrange elements.
                    # Need to expand the list.
                    bcnodes1.append(hermite_stride(self._function_space.boundary_nodes(ss, self.method)))
                bcnodes1 = functools.reduce(np.intersect1d, bcnodes1)
                bcnodes.append(bcnodes1)
            return np.concatenate(tuple(bcnodes))

    @utils.cached_property
    def node_set(self):
        '''The subset corresponding to the nodes at which this
        boundary condition applies.'''

        return op2.Subset(self._function_space.node_set, self.nodes)

    def zero(self, r):
        r"""Zero the boundary condition nodes on ``r``.

        :arg r: a :class:`.Function` to which the
            boundary condition should be applied.

        """
        if isinstance(r, matrix.MatrixBase):
            raise NotImplementedError("Zeroing bcs on a Matrix is not supported")

        for idx in self._indices:
            r = r.sub(idx)
        try:
            r.dat.zero(subset=self.node_set)
        except exceptions.MapValueError:
            raise RuntimeError("%r defined on incompatible FunctionSpace!" % r)

    def set(self, r, val):
        r"""Set the boundary nodes to a prescribed (external) value.
        :arg r: the :class:`Function` to which the value should be applied.
        :arg val: the prescribed value.
        """
        for idx in self._indices:
            r = r.sub(idx)
            val = val.sub(idx)
        r.assign(val, subset=self.node_set)

    def integrals(self):
        raise NotImplementedError("integrals() method has to be overwritten")


class DirichletBC(BCBase):
    r'''Implementation of a strong Dirichlet boundary condition.

    :arg V: the :class:`.FunctionSpace` on which the boundary condition
        should be applied.
    :arg g: the boundary condition values. This can be a :class:`.Function` on
        ``V``, a :class:`.Constant`, an :class:`.Expression`, an
        iterable of literal constants (converted to an
        :class:`.Expression`), or a literal constant which can be
        pointwise evaluated at the nodes of
        ``V``. :class:`.Expression`\s are projected onto ``V`` if it
        does not support pointwise evaluation.
    :arg sub_domain: the integer id(s) of the boundary region over which the
        boundary condition should be applied. The string "on_boundary" may be used
        to indicate all of the boundaries of the domain. In the case of extrusion
        the ``top`` and ``bottom`` strings are used to flag the bcs application on
        the top and bottom boundaries of the extruded mesh respectively.
    :arg method: the method for determining boundary nodes. The default is
        "topological", indicating that nodes topologically associated with a
        boundary facet will be included. The alternative value is "geometric",
        which indicates that nodes associated with basis functions which do not
        vanish on the boundary will be included. This can be used to impose
        strong boundary conditions on DG spaces, or no-slip conditions on HDiv spaces.
    '''

    def __init__(self, V, g, sub_domain, method="topological"):
        super().__init__(V, sub_domain, method=method)
        # Save the original value the user passed in.  If the user
        # passed in an Expression that has user-defined variables in
        # it, we need to remember it so that we can re-interpolate it
        # onto the function_arg if its state has changed.  Note that
        # the function_arg assignment is actually a property setter
        # which in the case of expressions interpolates it onto a
        # function and then throws the expression away.
        self._original_val = g
        self.function_arg = g
        self._original_arg = self.function_arg
        self._currently_zeroed = False
        self.is_linear = True
        self.Jp_eq_J = True

    @property
    def function_arg(self):
        '''The value of this boundary condition.'''
        if isinstance(self._original_val, expression.Expression):
            if not self._currently_zeroed and \
               self._original_val._state != self._expression_state:
                # Expression values have changed, need to reinterpolate
                self.function_arg = self._original_val
                # Remember "new" value of original arg, to work with zero/restore pair.
                self._original_arg = self.function_arg
        return self._function_arg

    def reconstruct(self, *, V=None, g=None, sub_domain=None, method=None):
        if V is None:
            V = self.function_space()
        if g is None:
            g = self._original_arg
        if sub_domain is None:
            sub_domain = self.sub_domain
        if method is None:
            method = self.method
        if V == self.function_space() and g == self._original_arg and \
           sub_domain == self.sub_domain and method == self.method:
            return self
        return type(self)(V, g, sub_domain, method=method)

    @function_arg.setter
    def function_arg(self, g):
        '''Set the value of this boundary condition.'''
        if isinstance(g, function.Function) and g.function_space() != self._function_space:
            raise RuntimeError("%r is defined on incompatible FunctionSpace!" % g)
        if not isinstance(g, expression.Expression):
            try:
                # Bare constant?
                as_ufl(g)
            except UFLException:
                try:
                    # List of bare constants? Convert to UFL expression
                    g = as_ufl(as_tensor(g))
                    if g.ufl_shape != self._function_space.shape:
                        raise ValueError("%r doesn't match the shape of the function space." % (g,))
                except UFLException:
                    raise ValueError("%r is not a valid DirichletBC expression" % (g,))
        if isinstance(g, expression.Expression) or has_type(as_ufl(g), SpatialCoordinate):
            if isinstance(g, expression.Expression):
                self._expression_state = g._state
            try:
                g = function.Function(self._function_space).interpolate(g)
            # Not a point evaluation space, need to project onto V
            except NotImplementedError:
                g = projection.project(g, self._function_space)
        self._function_arg = g
        self._currently_zeroed = False

    def homogenize(self):
        '''Convert this boundary condition into a homogeneous one.

        Set the value to zero.

        '''
        self.function_arg = 0
        self._currently_zeroed = True

    def restore(self):
        '''Restore the original value of this boundary condition.

        This uses the value passed on instantiation of the object.'''
        self._function_arg = self._original_arg
        self._currently_zeroed = False

    def set_value(self, val):
        '''Set the value of this boundary condition.

        :arg val: The boundary condition values.  See
            :class:`.DirichletBC` for valid values.
        '''
        self.function_arg = val
        self._original_arg = self.function_arg
        self._original_val = val

    @timed_function('ApplyBC')
    def apply(self, r, u=None):
        r"""Apply this boundary condition to ``r``.

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

        if isinstance(r, matrix.MatrixBase):
            raise NotImplementedError("Capability to delay bc application has been dropped. Use assemble(a, bcs=bcs, ...) to obtain a fully assembled matrix")

        fs = self._function_space

        # Check that u matches r if supplied
        if u and u.function_space() != r.function_space():
            raise RuntimeError("Mismatching spaces for %s and %s" % (r, u))

        # Check that r's function space matches the BC's function
        # space. Run up through parents (IndexedFunctionSpace or
        # ComponentFunctionSpace) until we either match the function space, or
        # else we have a mismatch and raise an error.
        while True:
            if fs == r.function_space():
                break
            elif fs.parent is not None:
                fs = fs.parent
            else:
                raise RuntimeError("%r defined on incompatible FunctionSpace!" % r)

        # Apply the indexing to r (and u if supplied)
        for idx in self._indices:
            r = r.sub(idx)
            if u:
                u = u.sub(idx)
        if u:
            r.assign(u - self.function_arg, subset=self.node_set)
        else:
            r.assign(self.function_arg, subset=self.node_set)

    def integrals(self):
        return []


class EquationBC(BCBase):
    r'''Implementation of an equation boundary condition.

    :param eq: the linear/nonlinear form equation
    :param u: the :class:`.Function` to solve for
    :arg sub_domain: see :class:`.DirichletBC`.
    :arg bcs: a list of :class:`.DirichletBC`s and/or :class:`.EquationBC`s
        to be applied to this boundary condition equation (optional)
    :param J: the Jacobian for this boundary equation (optional)
    :param Jp: a form used for preconditioning the linear system,
        optional, if not supplied then the Jacobian itself
        will be used.
    :arg method: see :class:`.DirichletBC` (optional)
    :arg sub_space_index: the sub_space index for the function space
        on which the equation boundary condition is applied
    '''

    def __init__(self, eq, u, sub_domain, bcs=None, J=None, Jp=None, method="topological", V=None, is_linear=False):

        if V is None:
            V = eq.lhs.arguments()[0].function_space()
        super().__init__(V, sub_domain, method=method)
        # u is always the total solution just as in "solve(F==0, u, ...)"
        self.u = u
        # This nested structure will enable recursive application of boundary conditions.
        #
        # def _assemble(..., bcs, ...)
        #     ...
        #     for bc in bcs:
        #         # boundary conditions for boundary conditions for boun...
        #         ... _assemble(bc.f, bc.bcs, ...)
        #     ...
        self.bcs = bcs or []

        self.Jp_eq_J = Jp is None
        self.is_linear = is_linear

        from firedrake.variational_solver import check_pde_args
        # linear
        if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):
            self.J = eq.lhs
            self.Jp = Jp or self.J
            if eq.rhs == 0:
                self.F = ufl_expr.action(self.J, self.u)
            else:
                if not isinstance(eq.rhs, (ufl.Form, slate.slate.TensorBase)):
                    raise TypeError("Provided BC RHS is a '%s', not a Form or Slate Tensor" % type(eq.rhs).__name__)
                if len(eq.rhs.arguments()) != 1:
                    raise ValueError("Provided BC RHS is not a linear form")
                self.F = ufl_expr.action(self.J, self.u) - eq.rhs
            self.is_linear = self.is_linear or True
        # nonlinear
        else:
            if eq.rhs != 0:
                raise TypeError("RHS of a nonlinear form equation has to be 0")
            self.F = eq.lhs
            self.J = J or ufl_expr.derivative(self.F, self.u)
            self.Jp = Jp or self.J
            # Argument checking
            check_pde_args(self)
            self.is_linear = self.is_linear or False


class EquationBCSplit(BCBase):
    def __init__(self, ebc, form, bcs=None, V=None):
        if not isinstance(ebc, (EquationBC, EquationBCSplit)):
            raise TypeError("EquationBCSplit constructor is expecting an instance of EquationBC/EquationBCSplit")
        if V is None:
            V = ebc._function_space
        super(EquationBCSplit, self).__init__(V, ebc.sub_domain, method=ebc.method)
        self.f = form
        self.bcs = bcs or []

    def integrals(self):
        return self.f.integrals()

    def add(self, bc):
        if not isinstance(bc, (DirichletBC, EquationBCSplit)):
            raise TypeError("EquationBCSplit.add expects an instance of DirichletBC or EquationBCSplit.")
        self.bcs.append(bc)


def homogenize(bc):
    r"""Create a homogeneous version of a :class:`.DirichletBC` object and return it. If
    ``bc`` is an iterable containing one or more :class:`.DirichletBC` objects,
    then return a list of the homogeneous versions of those :class:`.DirichletBC`\s.

    :arg bc: a :class:`.DirichletBC`, or iterable object comprising :class:`.DirichletBC`\(s).
    """
    if isinstance(bc, (tuple, list)):
        lst = []
        for i in bc:
            if not isinstance(i, DirichletBC):
                raise TypeError("homogenize only makes sense for DirichletBCs")
            lst.append(DirichletBC(i.function_space(), 0, i.sub_domain))
        return lst
    elif isinstance(bc, DirichletBC):
        return DirichletBC(bc.function_space(), 0, bc.sub_domain)
    else:
        raise TypeError("homogenize only takes a DirichletBC or a list/tuple of DirichletBCs")
