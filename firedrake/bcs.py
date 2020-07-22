# A module implementing strong (Dirichlet) boundary conditions.
import numpy as np

import functools
import itertools

import ufl
from ufl import as_ufl, SpatialCoordinate, UFLException, as_tensor, VectorElement
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
from firedrake.formmanipulation import ExtractSubBlock
from firedrake import replace
from firedrake import solving
from firedrake.adjoint.dirichletbc import DirichletBCMixin

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
        # init bcs
        self.bcs = []
        # Remember the depth of the bc
        self._bc_depth = 0

    def __iter__(self):
        yield self
        yield from itertools.chain(*self.bcs)

    def function_space(self):
        '''The :class:`.FunctionSpace` on which this boundary condition should
        be applied.'''

        return self._function_space

    def function_space_index(self):
        fs = self._function_space
        if fs.component is not None:
            fs = fs.parent
        if fs.index is None:
            raise RuntimeError("This function should only be called when function space is indexed")
        return fs.index

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

        ndim = self.function_space().mesh()._topology_dm.getDimension()
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

    def as_subspace(self, field, V, use_split):
        fs = self._function_space
        if fs.parent is not None and isinstance(fs.parent.ufl_element(), VectorElement):
            index = fs.parent.index
        else:
            index = fs.index
        cmpt = fs.component
        # TODO: need to test this logic
        field_renumbering = dict([f, i] for i, f in enumerate(field))
        if index in field:
            if len(field) == 1:
                W = V
            else:
                W = V.split()[field_renumbering[index]] if use_split else V.sub(field_renumbering[index])
            if cmpt is not None:
                W = W.sub(cmpt)
            return W

    def sorted_equation_bcs(self):
        return []

    def increment_bc_depth(self):
        # Increment _bc_depth by 1
        self._bc_depth += 1
        for bc in itertools.chain(*self.bcs):
            bc._bc_depth += 1

    def extract_forms(self, form_type):
        # Return boundary condition objects actually used in assembly.
        raise NotImplementedError("Method to extract form objects not implemented.")


class DirichletBC(BCBase, DirichletBCMixin):
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

    @DirichletBCMixin._ad_annotate_init
    def __init__(self, V, g, sub_domain, method="topological"):
        super().__init__(V, sub_domain, method=method)
        if len(V) > 1:
            raise ValueError("Cannot apply boundary conditions on mixed spaces directly.\n"
                             "Apply to the components by indexing the space with .sub(...)")
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

    def dirichlet_bcs(self):
        yield self

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

    def reconstruct(self, field=None, V=None, g=None, sub_domain=None, method=None, use_split=False):
        fs = self.function_space()
        if V is None:
            V = fs
        if g is None:
            g = self._original_arg
        if sub_domain is None:
            sub_domain = self.sub_domain
        if method is None:
            method = self.method
        if field is not None:
            assert V is not None, "`V` can not be `None` when `field` is not `None`"
            V = self.as_subspace(field, V, use_split)
            if V is None:
                return
        if V == fs and \
           V.parent == fs.parent and \
           V.index == fs.index and \
           (V.parent is None or V.parent.parent == fs.parent.parent) and \
           (V.parent is None or V.parent.index == fs.parent.index) and \
           g == self._original_arg and \
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
    @DirichletBCMixin._ad_annotate_apply
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

    def extract_form(self, form_type):
        # DirichletBC is directly used in assembly.
        return self


class EquationBC(object):
    r'''Construct and store EquationBCSplit objects (for `F`, `J`, and `Jp`).

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
    :arg V: the :class:`.FunctionSpace` on which
        the equation boundary condition is applied (optional)
    :arg is_linear: this flag is used only with the `reconstruct` method
    :arg Jp_eq_J: this flag is used only with the `reconstruct` method
    '''

    def __init__(self, *args, bcs=None, J=None, Jp=None, method="topological", V=None, is_linear=False, Jp_eq_J=False):
        from firedrake.variational_solver import check_pde_args, is_form_consistent
        if isinstance(args[0], ufl.classes.Equation):
            # initial construction from equation
            eq = args[0]
            u = args[1]
            sub_domain = args[2]
            if V is None:
                V = eq.lhs.arguments()[0].function_space()
            bcs = solving._extract_bcs(bcs)
            # Jp_eq_J is progressively evaluated as the tree is constructed
            self.Jp_eq_J = Jp is None and all([bc.Jp_eq_J for bc in bcs])

            # linear
            if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):
                J = eq.lhs
                Jp = Jp or J
                if eq.rhs == 0:
                    F = ufl_expr.action(J, u)
                else:
                    if not isinstance(eq.rhs, (ufl.Form, slate.slate.TensorBase)):
                        raise TypeError("Provided BC RHS is a '%s', not a Form or Slate Tensor" % type(eq.rhs).__name__)
                    if len(eq.rhs.arguments()) != 1:
                        raise ValueError("Provided BC RHS is not a linear form")
                    F = ufl_expr.action(J, u) - eq.rhs
                self.is_linear = True
            # nonlinear
            else:
                if eq.rhs != 0:
                    raise TypeError("RHS of a nonlinear form equation has to be 0")
                F = eq.lhs
                J = J or ufl_expr.derivative(F, u)
                Jp = Jp or J
                self.is_linear = False
            # Check form style consistency
            is_form_consistent(self.is_linear, bcs)
            # Argument checking
            check_pde_args(F, J, Jp)
            # EquationBCSplit objects for `F`, `J`, and `Jp`
            self._F = EquationBCSplit(F, u, sub_domain, bcs=[bc if isinstance(bc, DirichletBC) else bc._F for bc in bcs], method=method, V=V)
            self._J = EquationBCSplit(J, u, sub_domain, bcs=[bc if isinstance(bc, DirichletBC) else bc._J for bc in bcs], method=method, V=V)
            self._Jp = EquationBCSplit(Jp, u, sub_domain, bcs=[bc if isinstance(bc, DirichletBC) else bc._Jp for bc in bcs], method=method, V=V)
        elif all(isinstance(args[i], EquationBCSplit) for i in range(3)):
            # reconstruction for splitting `solving_utils.split`
            self.Jp_eq_J = Jp_eq_J
            self.is_linear = is_linear
            self._F = args[0]
            self._J = args[1]
            self._Jp = args[2]
        else:
            raise TypeError("Wrong EquationBC arguments")

    def __iter__(self):
        yield from itertools.chain(self._F)

    def dirichlet_bcs(self):
        # _F, _J, and _Jp all have the same DirichletBCs
        yield from self._F.dirichlet_bcs()

    def extract_form(self, form_type):
        r"""Return :class:`.EquationBCSplit` associated with the given 'form_type'.

        :arg form_type: Form to extract; 'F', 'J', or 'Jp'.
        """
        if form_type not in {"F", "J", "Jp"}:
            raise ValueError("Unknown form_type: 'form_type' must be 'F', 'J', or 'Jp'.")
        else:
            return getattr(self, f"_{form_type}")

    def reconstruct(self, V, subu, u, field):
        _F = self._F.reconstruct(field=field, V=V, subu=subu, u=u)
        _J = self._J.reconstruct(field=field, V=V, subu=subu, u=u)
        _Jp = self._Jp.reconstruct(field=field, V=V, subu=subu, u=u)
        if all([_F is not None, _J is not None, _Jp is not None]):
            return EquationBC(_F, _J, _Jp, Jp_eq_J=self.Jp_eq_J, is_linear=self.is_linear)


class EquationBCSplit(BCBase):
    r'''Class for a BC tree that stores/manipulates either `F`, `J`, or `Jp`.

    :param form: the linear/nonlinear form: `F`, `J`, or `Jp`.
    :param u: the :class:`.Function` to solve for
    :arg sub_domain: see :class:`.DirichletBC`.
    :arg bcs: a list of :class:`.DirichletBC`s and/or :class:`.EquationBC`s
        to be applied to this boundary condition equation (optional)
    :arg method: see :class:`.DirichletBC` (optional)
    :arg V: the :class:`.FunctionSpace` on which
        the equation boundary condition is applied (optional)
    '''

    def __init__(self, form, u, sub_domain, bcs=None, method="topological", V=None):
        # This nested structure will enable recursive application of boundary conditions.
        #
        # def _assemble(..., bcs, ...)
        #     ...
        #     for bc in bcs:
        #         # boundary conditions for boundary conditions for boun...
        #         ... _assemble(bc.f, bc.bcs, ...)
        #     ...
        self.f = form
        self.u = u
        if V is None:
            V = self.f.arguments()[0].function_space()
        super(EquationBCSplit, self).__init__(V, sub_domain, method=method)
        # overwrite bcs
        self.bcs = bcs or []
        for bc in self.bcs:
            bc.increment_bc_depth()

    def dirichlet_bcs(self):
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    def sorted_equation_bcs(self):
        # Create a list of EquationBCSplit objects
        sorted_ebcs = list(bc for bc in itertools.chain(*self.bcs) if isinstance(bc, EquationBCSplit))
        # Sort this list to avoid `self._y` copys in matrix_free/operators.py
        sorted_ebcs.sort(key=lambda bc: bc._bc_depth, reverse=True)
        return sorted_ebcs

    def integrals(self):
        return self.f.integrals()

    def add(self, bc):
        if not isinstance(bc, (DirichletBC, EquationBCSplit)):
            raise TypeError("EquationBCSplit.add expects an instance of DirichletBC or EquationBCSplit.")
        bc.increment_bc_depth()
        self.bcs.append(bc)

    def reconstruct(self, field=None, V=None, subu=None, u=None, row_field=None, col_field=None, action_x=None, use_split=False):
        subu = subu or self.u
        row_field = row_field or field
        col_field = col_field or field
        # define W and form
        if field is None:
            # Returns self
            W = self._function_space
            form = self.f
        else:
            assert V is not None, "`V` can not be `None` when `field` is not `None`"
            W = self.as_subspace(field, V, use_split)
            if W is None:
                return
            rank = len(self.f.arguments())
            splitter = ExtractSubBlock()
            if rank == 1:
                form = splitter.split(self.f, argument_indices=(row_field, ))
            elif rank == 2:
                form = splitter.split(self.f, argument_indices=(row_field, col_field))
            if u is not None:
                form = replace(form, {self.u: u})
        if action_x is not None:
            assert len(form.arguments()) == 2, "rank of self.f must be 2 when using action_x parameter"
            form = ufl_expr.action(form, action_x)
        ebc = EquationBCSplit(form, subu, self.sub_domain, method=self.method, V=W)
        for bc in self.bcs:
            if isinstance(bc, DirichletBC):
                ebc.add(bc.reconstruct(V=W, g=bc.function_arg, sub_domain=bc.sub_domain, method=bc.method, use_split=use_split))
            elif isinstance(bc, EquationBCSplit):
                bc_temp = bc.reconstruct(field=field, V=V, subu=subu, u=u, row_field=row_field, col_field=col_field, action_x=action_x, use_split=use_split)
                # Due to the "if index", bc_temp can be None
                if bc_temp is not None:
                    ebc.add(bc_temp)
        return ebc


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
