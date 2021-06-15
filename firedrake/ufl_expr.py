import ufl
import ufl.argument
from ufl.assertions import ufl_assert
from ufl.split_functions import split
from ufl.corealg.map_dag import map_expr_dag
from ufl.algorithms import extract_arguments, extract_coefficients, compute_form_action
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import GradRuleset, ReferenceGradRuleset, VariableRuleset, GateauxDerivativeRuleset, DerivativeRuleDispatcher
from ufl.form import as_form
from ufl.classes import Zero

import firedrake
from firedrake import utils
from firedrake.projected import FiredrakeProjected
from firedrake.petsc import PETSc


__all__ = ['Argument', 'TestFunction', 'TrialFunction',
           'TestFunctions', 'TrialFunctions',
           'derivative', 'adjoint',
           'action', 'CellSize', 'FacetNormal']


class Argument(ufl.argument.Argument):
    """Representation of the argument to a form.

    :arg function_space: the :class:`.FunctionSpace` the argument
        corresponds to.
    :arg number: the number of the argument being constructed.
    :kwarg part: optional index (mostly ignored).

    .. note::

       an :class:`Argument` with a number of ``0`` is used as a
       :func:`TestFunction`, with a number of ``1`` it is used as
       a :func:`TrialFunction`.
    """
    def __init__(self, function_space, number, part=None):
        super(Argument, self).__init__(function_space.ufl_function_space(),
                                       number, part=part)
        self._function_space = function_space

    @utils.cached_property
    def cell_node_map(self):
        return self.function_space().cell_node_map

    @utils.cached_property
    def interior_facet_node_map(self):
        return self.function_space().interior_facet_node_map

    @utils.cached_property
    def exterior_facet_node_map(self):
        return self.function_space().exterior_facet_node_map

    def function_space(self):
        return self._function_space

    def make_dat(self):
        return self.function_space().make_dat()

    def reconstruct(self, function_space=None,
                    number=None, part=None):
        if function_space is None or function_space == self.function_space():
            function_space = self.function_space()
        if number is None or number == self._number:
            number = self._number
        if part is None or part == self._part:
            part = self._part
        if number is self._number and part is self._part \
           and function_space is self.function_space():
            return self
        ufl_assert(isinstance(number, int),
                   "Expecting an int, not %s" % number)
        ufl_assert(function_space.ufl_element().value_shape()
                   == self.ufl_element().value_shape(),
                   "Cannot reconstruct an Argument with a different value shape.")
        return Argument(function_space, number, part=part)


@PETSc.Log.EventDecorator()
def TestFunction(function_space, part=None):
    """Build a test function on the specified function space.

    :arg function_space: the :class:`.FunctionSpace` to build the test
         function on.
    :kwarg part: optional index (mostly ignored)."""
    return Argument(function_space, 0, part=part)


@PETSc.Log.EventDecorator()
def TrialFunction(function_space, part=None):
    """Build a trial function on the specified function space.

    :arg function_space: the :class:`.FunctionSpace` to build the trial
         function on.
    :kwarg part: optional index (mostly ignored)."""
    return Argument(function_space, 1, part=None)


def TestFunctions(function_space):
    """Return a tuple of test functions on the specified function space.

    :arg function_space: the :class:`.FunctionSpace` to build the test
         functions on.

    This returns ``len(function_space)`` test functions, which, if the
    function space is a :class:`.MixedFunctionSpace`, are indexed
    appropriately.
    """
    return split(TestFunction(function_space))


def TrialFunctions(function_space):
    """Return a tuple of trial functions on the specified function space.

    :arg function_space: the :class:`.FunctionSpace` to build the trial
         functions on.

    This returns ``len(function_space)`` trial functions, which, if the
    function space is a :class:`.MixedFunctionSpace`, are indexed
    appropriately.
    """
    return split(TrialFunction(function_space))


@PETSc.Log.EventDecorator()
def derivative(form, u, du=None, coefficient_derivatives=None):
    """Compute the derivative of a form.

    Given a form, this computes its linearization with respect to the
    provided :class:`.Function`.  The resulting form has one
    additional :class:`Argument` in the same finite element space as
    the Function.

    :arg form: a :class:`~ufl.classes.Form` to compute the derivative of.
    :arg u: a :class:`.Function` to compute the derivative with
         respect to.
    :arg du: an optional :class:`Argument` to use as the replacement
         in the new form (constructed automatically if not provided).
    :arg coefficient_derivatives: an optional :class:`dict` to
         provide the derivative of a coefficient function.

    :raises ValueError: If any of the coefficients in ``form`` were
        obtained from ``u.split()``.  UFL doesn't notice that these
        are related to ``u`` and so therefore the derivative is
        wrong (instead one should have written ``split(u)``).

    See also :func:`ufl.derivative`.
    """
    # TODO: What about Constant?
    u_is_x = isinstance(u, ufl.SpatialCoordinate)
    uc, = (u,) if u_is_x else extract_coefficients(u)
    if not u_is_x and len(uc.split()) > 1 and set(extract_coefficients(form)) & set(uc.split()):
        raise ValueError("Taking derivative of form wrt u, but form contains coefficients from u.split()."
                         "\nYou probably meant to write split(u) when defining your form.")

    mesh = form.ufl_domain()
    if not mesh:
        raise ValueError("Expression to be differentiated has no ufl domain."
                         "\nDo you need to add a domain to your Constant?")
    is_dX = u_is_x or u is mesh.coordinates
    try:
        args = form.arguments()
    except AttributeError:
        args = extract_arguments(form)

    def argument(V):
        if du is None:
            n = max(a.number() for a in args) if args else -1
            return Argument(V, n + 1)
        else:
            return du

    if is_dX:
        coords = mesh.coordinates
        u = ufl.SpatialCoordinate(mesh)
        V = coords.function_space()
        du = argument(V)
        cds = {coords: du}
        if coefficient_derivatives is not None:
            cds.update(coefficient_derivatives)
        coefficient_derivatives = cds
    elif isinstance(uc, firedrake.Function):
        V = uc.function_space()
        du = argument(V)
    elif isinstance(uc, firedrake.Constant):
        if uc.ufl_shape != ():
            raise ValueError("Real function space of vector elements not supported")
        V = firedrake.FunctionSpace(mesh, "Real", 0)
        du = argument(V)
    else:
        raise RuntimeError("Can't compute derivative for form")

    if u.ufl_shape != du.ufl_shape:
        raise ValueError("Shapes of u and du do not match.\n"
                         "If you passed an indexed part of split(u) into "
                         "derivative, you need to provide an appropriate du as well.")
    return ufl.derivative(form, u, du, coefficient_derivatives)


@PETSc.Log.EventDecorator()
def action(form, coefficient):
    """Compute the action of a form on a coefficient.

    :arg form: A UFL form, or a Slate tensor.
    :arg coefficient: The :class:`~.Function` to act on.
    :returns: a symbolic expression for the action.
    """
    if isinstance(form, firedrake.slate.TensorBase):
        if form.rank == 0:
            raise ValueError("Can't take action of rank-0 tensor")
        return form * firedrake.AssembledVector(coefficient)
    else:
        form = as_form(form)
        form = apply_algebra_lowering(form)
        form = apply_derivatives(form)
        return compute_form_action(form, coefficient)


@PETSc.Log.EventDecorator()
def adjoint(form, reordered_arguments=None):
    """Compute the adjoint of a form.

    :arg form: A UFL form, or a Slate tensor.
    :arg reordered_arguments: arguments to use when creating the
       adjoint.  Ignored if form is a Slate tensor.

    If the form is a slate tensor, this just returns its transpose.
    Otherwise, given a bilinear form, compute the adjoint form by
    changing the ordering (number) of the test and trial functions.

    By default, new Argument objects will be created with opposite
    ordering. However, if the adjoint form is to be added to other
    forms later, their arguments must match.  In that case, the user
    must provide a tuple reordered_arguments=(u2,v2).
    """
    if isinstance(form, firedrake.slate.TensorBase):
        if reordered_arguments is not None:
            firedrake.warning("Ignoring arguments for adjoint of Slate tensor.")
        if form.rank != 2:
            raise ValueError("Expecting rank-2 tensor")
        return form.T
    else:
        if len(form.arguments()) != 2:
            raise ValueError("Expecting bilinear form")
        # ufl.adjoint creates new Arguments if no reordered_arguments is
        # given.  To avoid that, always pass reordered_arguments with
        # firedrake.Argument objects.
        if reordered_arguments is None:
            v, u = extract_arguments(form)
            reordered_arguments = (Argument(u.function_space(),
                                            number=v.number(),
                                            part=v.part()),
                                   Argument(v.function_space(),
                                            number=u.number(),
                                            part=u.part()))
        return ufl.adjoint(form, reordered_arguments)


@PETSc.Log.EventDecorator()
def CellSize(mesh):
    """A symbolic representation of the cell size of a mesh.

    :arg mesh: the mesh for which to calculate the cell size.
    """
    mesh.init()
    return 2.0 * ufl.Circumradius(mesh)


@PETSc.Log.EventDecorator()
def FacetNormal(mesh):
    """A symbolic representation of the facet normal on a cell in a mesh.

    :arg mesh: the mesh over which the normal should be represented.
    """
    mesh.init()
    return ufl.FacetNormal(mesh)


# apply_derivative with FiredrakeProjected.


class FiredrakeDerivativeMixin(object):
    def firedrake_projected(self, o, Ap):
        # Propagate zeros
        if isinstance(Ap, Zero):
            return self.independent_operator(o)
        return FiredrakeProjected(Ap, o.subspace())


class FiredrakeGradRuleset(GradRuleset, FiredrakeDerivativeMixin):
    def __init__(self, geometric_dimension):
        GradRuleset.__init__(self, geometric_dimension)


class FiredrakeReferenceGradRuleset(ReferenceGradRuleset, FiredrakeDerivativeMixin):
    def __init__(self, topological_dimension):
        ReferenceGradRuleset.__init__(self, topological_dimension)


class FiredrakeVariableRuleset(VariableRuleset, FiredrakeDerivativeMixin):
    def __init__(self, var):
        VariableRuleset.__init__(self, var)


class FiredrakeGateauxDerivativeRuleset(GateauxDerivativeRuleset, FiredrakeDerivativeMixin):
    def __init__(self, coefficients, arguments, coefficient_derivatives):
        GateauxDerivativeRuleset.__init__(self, coefficients, arguments, coefficient_derivatives)


class FiredrakeDerivativeRuleDispatcher(DerivativeRuleDispatcher):
    def __init__(self):
        DerivativeRuleDispatcher.__init__(self)

    def grad(self, o, f):
        rules = FiredrakeGradRuleset(o.ufl_shape[-1])
        return map_expr_dag(rules, f)

    def reference_grad(self, o, f):
        rules = FiredrakeReferenceGradRuleset(o.ufl_shape[-1])  # FIXME: Look over this and test better.
        return map_expr_dag(rules, f)

    def variable_derivative(self, o, f, dummy_v):
        rules = FiredrakeVariableRuleset(o.ufl_operands[1])
        return map_expr_dag(rules, f)

    def coefficient_derivative(self, o, f, dummy_w, dummy_v, dummy_cd):
        dummy, w, v, cd = o.ufl_operands
        rules = FiredrakeGateauxDerivativeRuleset(w, v, cd)
        return map_expr_dag(rules, f)


def apply_derivatives(expression):
    """Apply derivatives.

    `ufl.algorithms.apply_derivatives.GenericDerivativeRuleset`
    does not define default rule for expr, so we need to
    reconstruct apply_derivative function with rule set for
    FiredrakeProjected.
    """
    rules = FiredrakeDerivativeRuleDispatcher()
    return map_integrand_dags(rules, expression)
