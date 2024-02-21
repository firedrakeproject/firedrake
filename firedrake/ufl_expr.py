import ufl
import ufl.argument
from ufl.duals import is_dual
from ufl.core.base_form_operator import BaseFormOperator
from ufl.split_functions import split
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.domain import as_domain

import firedrake
from firedrake import utils, function, cofunction
from firedrake.constant import Constant
from firedrake.petsc import PETSc


__all__ = ['Argument', 'Coargument', 'TestFunction', 'TrialFunction',
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

    def __new__(cls, *args, **kwargs):
        if args[0] and is_dual(args[0]):
            return Coargument(*args, **kwargs)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, function_space, number, part=None):
        if function_space.ufl_element().family() == "Real" and function_space.shape != ():
            raise NotImplementedError(f"{type(self).__name__} on a vector-valued Real space is not supported.")
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
        if not isinstance(number, int):
            raise TypeError(f"Expecting an int, not {number}")
        if function_space.ufl_element().value_shape != self.ufl_element().value_shape:
            raise ValueError("Cannot reconstruct an Argument with a different value shape.")
        return Argument(function_space, number, part=part)


class Coargument(ufl.argument.Coargument):
    """Representation of an argument to a form in a dual space.

    :arg function_space: the :class:`.FunctionSpace` the argument
        corresponds to.
    :arg number: the number of the argument being constructed.
    :kwarg part: optional index (mostly ignored).
    """

    def __init__(self, function_space, number, part=None):
        if function_space.ufl_element().family() == "Real" and function_space.shape != ():
            raise NotImplementedError(f"{type(self).__name__} on a vector-valued Real space is not supported.")
        super(Coargument, self).__init__(function_space.ufl_function_space(),
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

    def _analyze_form_arguments(self, outer_form=None):
        # Returns the argument found in the Coargument object
        self._coefficients = ()
        # Coarguments map from V* to V*, i.e. V* -> V*, or equivalently V* x V -> R.
        # So they have one argument in the primal space and one in the dual space.
        # However, when they are composed with linear forms with dual arguments, such as BaseFormOperators,
        # the primal argument is discarded when analysing the argument as Coarguments.
        if not outer_form:
            self._arguments = (Argument(self.function_space().dual(), 0), self)
        else:
            self._arguments = (self,)

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
        if not isinstance(number, int):
            raise TypeError(f"Expecting an int, not {number}")
        if function_space.ufl_element().value_shape != self.ufl_element().value_shape:
            raise ValueError("Cannot reconstruct an Coargument with a different value shape.")
        return Coargument(function_space, number, part=part)

    def equals(self, other):
        if type(other) is not Coargument:
            return False
        if self is other:
            return True
        return (self._function_space == other._function_space
                and self._number == other._number and self._part == other._part)


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
        obtained from ``u.subfunctions``.  UFL doesn't notice that these
        are related to ``u`` and so therefore the derivative is
        wrong (instead one should have written ``split(u)``).

    See also :func:`ufl.derivative`.
    """
    if isinstance(form, firedrake.slate.TensorBase):
        raise TypeError(
            f"Cannot take the derivative of a {type(form).__name__}"
        )
    u_is_x = isinstance(u, ufl.SpatialCoordinate)
    if u_is_x or isinstance(u, (Constant, BaseFormOperator)):
        uc = u
    else:
        uc, = extract_coefficients(u)
    if not (u_is_x or isinstance(u, BaseFormOperator)) and len(uc.subfunctions) > 1 and set(extract_coefficients(form)) & set(uc.subfunctions):
        raise ValueError("Taking derivative of form wrt u, but form contains coefficients from u.subfunctions."
                         "\nYou probably meant to write split(u) when defining your form.")

    mesh = as_domain(form)
    if not mesh:
        raise ValueError("Expression to be differentiated has no ufl domain."
                         "\nDo you need to add a domain to your Constant?")
    is_dX = u_is_x or u is mesh.coordinates

    try:
        args = form.arguments()
    except AttributeError:
        args = extract_arguments(form)
    # UFL arguments need unique indices within a form
    n = max(a.number() for a in args) if args else -1

    if is_dX:
        coords = mesh.coordinates
        u = ufl.SpatialCoordinate(mesh)
        V = coords.function_space()
    elif isinstance(uc, (firedrake.Function, firedrake.Cofunction, BaseFormOperator)):
        V = uc.function_space()
    elif isinstance(uc, firedrake.Constant):
        if uc.ufl_shape != ():
            raise ValueError("Real function space of vector elements not supported")
        # Replace instances of the constant with a new argument ``x``
        # and differentiate wrt ``x``.
        V = firedrake.FunctionSpace(mesh, "Real", 0)
        x = ufl.Coefficient(V, n + 1)
        n += 1
        # TODO: Update this line when https://github.com/FEniCS/ufl/issues/171 is fixed
        form = ufl.replace(form, {u: x})
        u = x
    else:
        raise RuntimeError("Can't compute derivative for form")

    if du is None:
        du = Argument(V, n + 1)

    if is_dX:
        internal_coefficient_derivatives = {coords: du}
    else:
        internal_coefficient_derivatives = {}
    if coefficient_derivatives:
        internal_coefficient_derivatives.update(coefficient_derivatives)

    if u.ufl_shape != du.ufl_shape:
        raise ValueError("Shapes of u and du do not match.\n"
                         "If you passed an indexed part of split(u) into "
                         "derivative, you need to provide an appropriate du as well.")
    return ufl.derivative(form, u, du, internal_coefficient_derivatives)


@PETSc.Log.EventDecorator()
def action(form, coefficient, derivatives_expanded=None):
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
        return ufl.action(form, coefficient, derivatives_expanded=derivatives_expanded)


@PETSc.Log.EventDecorator()
def adjoint(form, reordered_arguments=None, derivatives_expanded=None):
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
        return ufl.adjoint(form, reordered_arguments, derivatives_expanded=derivatives_expanded)


@PETSc.Log.EventDecorator()
def CellSize(mesh):
    """A symbolic representation of the cell size of a mesh.

    :arg mesh: the mesh for which to calculate the cell size.
    """
    mesh.init()
    return ufl.CellDiameter(mesh)


@PETSc.Log.EventDecorator()
def FacetNormal(mesh):
    """A symbolic representation of the facet normal on a cell in a mesh.

    :arg mesh: the mesh over which the normal should be represented.
    """
    mesh.init()
    return ufl.FacetNormal(mesh)


def extract_domains(func):
    """Extract the domain from `func`.

    Parameters
    ----------
    x : firedrake.function.Function, firedrake.cofunction.Cofunction, or firedrake.constant.Constant
        The function to extract the domain from.

    Returns
    -------
    list of firedrake.mesh.MeshGeometry
        Extracted domains.
    """
    if isinstance(func, (function.Function, cofunction.Cofunction)):
        return [func.function_space().mesh()]
    else:
        return ufl.domain.extract_domains(func)


def extract_unique_domain(func):
    """Extract the single unique domain `func` is defined on.

    Parameters
    ----------
    x : firedrake.function.Function, firedrake.cofunction.Cofunction, or firedrake.constant.Constant
        The function to extract the domain from.

    Returns
    -------
    list of firedrake.mesh.MeshGeometry
        Extracted domains.
    """
    if isinstance(func, (function.Function, cofunction.Cofunction)):
        return func.function_space().mesh()
    else:
        return ufl.domain.extract_unique_domain(func)
