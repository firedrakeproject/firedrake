"""Utilities for preprocessing UFL objects."""

from functools import singledispatch

import ufl
from ufl import replace
from ufl.algorithms import compute_form_data as ufl_compute_form_data
from ufl.algorithms import estimate_total_polynomial_degree
from ufl.algorithms.analysis import extract_arguments, extract_coefficients, extract_type
from ufl.algorithms.apply_function_pullbacks import (
    apply_function_pullbacks,
    apply_interpolate_pullbacks,
    apply_inverse_pullback,
)
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms.apply_restrictions import apply_restrictions
from ufl.algorithms.cancel_jacobian_products import cancel_jacobian_products
from ufl.algorithms.remove_component_tensors import remove_component_tensors
from ufl.algorithms.comparison_checker import do_comparison_check
from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
from ufl.algorithms.signature import compute_expression_signature
from ufl.corealg.multifunction import MultiFunction
from ufl.geometry import QuadratureWeight
from ufl.geometry import Jacobian, JacobianDeterminant, JacobianInverse
from ufl.classes import (Abs, Argument, CellOrientation,
                         Expr, FloatValue, Division,
                         Product,
                         ScalarValue, Sqrt, Zero, CellVolume, FacetArea)
from ufl.utils.sorting import sorted_by_count
from ufl.domain import extract_domains

import gem
from gem.node import MemoizerArg

from finat.element_factory import as_fiat_cell
from finat.point_set import UnknownPointSet
from finat.quadrature import QuadratureRule
from finat.ufl import FiniteElement, TensorElement

from tsfc.modified_terminals import is_modified_terminal, analyse_modified_terminal


preserve_geometry_types = (CellVolume, FacetArea)

# Prefix that forces TSFC to do runtime tabulation for a gem.Variable.
RUNTIME_VARIABLE_PREFIX = "rt_"


def runtime_quadrature_element(domain, ufl_element, rt_var_name=RUNTIME_VARIABLE_PREFIX + "X"):
    """Construct a Quadrature FiniteElement for interpolation onto a runtime
    point, e.g. a VertexOnlyMesh point known only at run time.

    Parameters
    ----------
    domain : ufl.AbstractDomain
        The source domain.
    ufl_element : finat.ufl.finiteelement.FiniteElement
        The UFL element of the target FunctionSpace.
    rt_var_name : str
        Name of the gem.Variable holding the point, prefixed with
        `RUNTIME_VARIABLE_PREFIX` to force TSFC to tabulate it at run time.
    """
    assert rt_var_name.startswith(RUNTIME_VARIABLE_PREFIX)

    cell = domain.ufl_cell()
    point_expr = gem.Variable(rt_var_name, (1, cell.topological_dimension))
    point_set = UnknownPointSet(point_expr)
    rule = QuadratureRule(point_set, weights=[1.0], ref_el=as_fiat_cell(cell))

    shape = ufl_element.pullback.physical_value_shape(ufl_element, domain)
    rt_element = FiniteElement("Quadrature", cell=cell, degree=0, quad_scheme=rule)
    if shape:
        symmetry = None if len(shape) < 2 else ufl_element.symmetry()
        rt_element = TensorElement(rt_element, shape=shape, symmetry=symmetry)
    return rt_element


def preprocess_interpolate(expression, element, domain, complex_mode=False):
    """Prepare a standalone interpolation for TSFC.

    Parameters
    ----------
    expression : ufl.Interpolate
        The interpolation to preprocess.
    element : finat.ufl.finiteelement.FiniteElement
        The UFL element of the interpolation target.
    domain : ufl.AbstractDomain
        The domain the operand is evaluated on.
    complex_mode : bool
        Is the scalar type complex?

    Returns
    -------
    ufl.Interpolate
        The interpolation with its operand in ``element``'s reference frame.

    Notes
    -----
    A standalone interpolation never reaches `compute_form_data`, so the operand
    gets the scalar preprocessing here.  Interpolations inside a form are lowered
    by `ufl.algorithms.apply_interpolate_pullbacks`.
    """
    dual_arg, operand = expression.argument_slots()
    operand = apply_inverse_pullback(operand, element, domain)
    operand = preprocess_expression(operand, complex_mode=complex_mode)
    operand = simplify_abs(operand, complex_mode)
    # Build the UFL node directly: the operand is now in the reference frame,
    # so it no longer matches the physical shape a Firedrake Interpolate checks.
    return ufl.Interpolate(operand, dual_arg)


def compute_form_data(form,
                      do_apply_function_pullbacks=True,
                      do_apply_integral_scaling=True,
                      do_apply_geometry_lowering=True,
                      preserve_geometry_types=preserve_geometry_types,
                      do_cancel_jacobian_products=True,
                      do_apply_default_restrictions=True,
                      do_apply_restrictions=True,
                      do_estimate_degrees=True,
                      do_replace_functions=True,
                      coefficients_to_split=None,
                      complex_mode=False):
    """Preprocess UFL form in a format suitable for TSFC. Return
    form data.

    This is merely a wrapper to UFL compute_form_data with default
    kwargs overriden in the way TSFC needs it and is provided for
    other form compilers based on TSFC.
    """
    # Multidomain problems require further index simplifications to ensure
    # that unwanted quantities do not appear inside single-domain integrals.
    do_remove_component_tensors = len(extract_domains(form)) > 1
    fd = ufl_compute_form_data(
        form,
        do_apply_function_pullbacks=do_apply_function_pullbacks,
        do_apply_integral_scaling=do_apply_integral_scaling,
        do_apply_geometry_lowering=do_apply_geometry_lowering,
        preserve_geometry_types=preserve_geometry_types,
        do_cancel_jacobian_products=do_cancel_jacobian_products,
        do_apply_default_restrictions=do_apply_default_restrictions,
        do_apply_restrictions=do_apply_restrictions,
        do_estimate_degrees=do_estimate_degrees,
        do_replace_functions=do_replace_functions,
        coefficients_to_split=coefficients_to_split,
        complex_mode=complex_mode,
        do_remove_component_tensors=do_remove_component_tensors,
    )
    constants = extract_firedrake_constants(form)
    fd.constants = constants
    return fd


def extract_firedrake_constants(a):
    """Build a sorted list of all constants in a"""
    return sorted_by_count(extract_type(a, TSFCConstantMixin))


def one_times(measure):
    # Workaround for UFL issue #80:
    # https://bitbucket.org/fenics-project/ufl/issues/80
    form = 1 * measure
    fd = compute_form_data(form, do_estimate_degrees=False)
    itg_data, = fd.integral_data
    integral, = itg_data.integrals
    integrand = integral.integrand()

    # UFL considers QuadratureWeight a geometric quantity, and the
    # general handler for geometric quantities estimates the degree of
    # the coordinate element.  This would unnecessarily increase the
    # estimated degree, so we drop QuadratureWeight instead.
    expression = replace(integrand, {QuadratureWeight(itg_data.domain): 1})

    # Now estimate degree for the preprocessed form
    degree = estimate_total_polynomial_degree(expression)

    return integrand, degree


def entity_avg(integrand, measure, argument_multiindices):
    arguments = extract_arguments(integrand)
    if len(arguments) == 1:
        a, = arguments
        integrand = ufl.replace(integrand, {a: ufl.Argument(a.function_space(),
                                                            number=0,
                                                            part=a.part())})
        argument_multiindices = (argument_multiindices[a.number()], )

    degree = estimate_total_polynomial_degree(integrand)
    form = integrand * measure
    fd = compute_form_data(form, do_estimate_degrees=False,
                           do_apply_function_pullbacks=False,
                           do_replace_functions=False,
                           )
    itg_data, = fd.integral_data
    integral, = itg_data.integrals
    integrand = integral.integrand()
    return integrand, degree, argument_multiindices


def preprocess_expression(expression, complex_mode=False,
                          do_apply_restrictions=False):
    """Imitates the compute_form_data processing pipeline.

    :arg complex_mode: Are we in complex UFL mode?
    :arg do_apply_restrictions: Propogate restrictions to terminals?

    Useful, for example, to preprocess non-scalar expressions, which
    are not and cannot be forms.
    """
    expression = apply_interpolate_pullbacks(expression)
    if complex_mode:
        expression = do_comparison_check(expression)
    else:
        expression = remove_complex_nodes(expression)
    jacobian_types = (Jacobian, JacobianInverse, JacobianDeterminant)
    lowering_preserve_types = preserve_geometry_types + jacobian_types
    expression = apply_algebra_lowering(expression)
    expression = apply_derivatives(expression)
    expression = apply_function_pullbacks(expression)
    expression = apply_geometry_lowering(expression, lowering_preserve_types)
    expression = apply_derivatives(expression)
    expression = apply_geometry_lowering(expression, lowering_preserve_types)
    expression = apply_derivatives(expression)
    # Cancel contractions of the Jacobian with its inverse before
    # expanding the inverse into individual matrix entries
    expression = remove_component_tensors(expression)
    expression = cancel_jacobian_products(expression)
    expression = apply_geometry_lowering(expression, preserve_geometry_types)
    expression = apply_derivatives(expression)
    if not complex_mode:
        expression = remove_complex_nodes(expression)
    if do_apply_restrictions:
        expression = apply_restrictions(expression)
    return expression


class ModifiedTerminalMixin(object):
    """Mixin to use with MultiFunctions that operate on modified
    terminals."""

    def unexpected(self, o):
        assert False, "Not expected %r at this stage." % o

    # global derivates should have been pulled back
    grad = unexpected
    div = unexpected
    curl = unexpected

    # div and curl should have been algebraically lowered
    reference_div = unexpected
    reference_curl = unexpected

    def _modified_terminal(self, o):
        assert is_modified_terminal(o)
        return self.modified_terminal(o)

    # Unlike UFL, we do not regard Indexed as a terminal modifier.
    # indexed = _modified_terminal

    positive_restricted = _modified_terminal
    negative_restricted = _modified_terminal
    single_value_restricted = _modified_terminal
    to_be_restricted = _modified_terminal

    reference_grad = _modified_terminal
    reference_value = _modified_terminal

    terminal = _modified_terminal


class PickRestriction(MultiFunction, ModifiedTerminalMixin):
    """Pick out parts of an expression with specified restrictions on
    the arguments.

    :arg test: The restriction on the test function.
    :arg trial:  The restriction on the trial function.

    Returns those parts of the expression that have the requested
    restrictions, or else :class:`ufl.classes.Zero` if no such part
    exists.
    """
    def __init__(self, test=None, trial=None):
        self.restrictions = {0: test, 1: trial}
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        return o

    def modified_terminal(self, o):
        mt = analyse_modified_terminal(o)
        t = mt.terminal
        r = mt.restriction
        if isinstance(t, Argument) and r in ['+', '-']:
            if r == self.restrictions[t.number()]:
                return o
            else:
                return Zero(o.ufl_shape, o.ufl_free_indices, o.ufl_index_dimensions)
        else:
            return o


def ufl_reuse_if_untouched(o, *ops):
    """Reuse object if operands are the same objects."""
    if all(a is b for a, b in zip(o.ufl_operands, ops)):
        return o
    else:
        return o._ufl_expr_reconstruct_(*ops)


@singledispatch
def _simplify_abs(o, self, in_abs):
    """Single-dispatch function to simplify absolute values.

    :arg o: UFL node
    :arg self: Callback handler for recursion
    :arg in_abs: Is ``o`` inside an absolute value?

    When ``in_abs`` we must return a non-negative value, potentially
    by wrapping the returned node with ``Abs``.
    """
    raise AssertionError("UFL node expected, not %s" % type(o))


@_simplify_abs.register(Expr)
def _simplify_abs_expr(o, self, in_abs):
    # General case, only wrap the outer expression (if necessary)
    operands = [self(op, False) for op in o.ufl_operands]
    result = ufl_reuse_if_untouched(o, *operands)
    if in_abs:
        result = Abs(result)
    return result


@_simplify_abs.register(Sqrt)
def _simplify_abs_sqrt(o, self, in_abs):
    result = ufl_reuse_if_untouched(o, self(o.ufl_operands[0], False))
    if self.complex_mode and in_abs:
        return Abs(result)
    else:
        return result


@_simplify_abs.register(ScalarValue)
def _simplify_abs_(o, self, in_abs):
    if not in_abs:
        return o
    # Inline abs(constant)
    return ufl.as_ufl(abs(o._value))


@_simplify_abs.register(CellOrientation)
def _simplify_abs_cellorientation(o, self, in_abs):
    if not in_abs:
        return o
    # Cell orientation is +-1
    return FloatValue(1)


@_simplify_abs.register(Division)
@_simplify_abs.register(Product)
def _simplify_abs_product(o, self, in_abs):
    if not in_abs:
        # Just reconstruct
        ops = [self(op, False) for op in o.ufl_operands]
        return ufl_reuse_if_untouched(o, *ops)

    # Visit children, distributing Abs
    ops = [self(op, True) for op in o.ufl_operands]

    # Strip Abs off again (we will put it outside now)
    stripped = False
    strip_ops = []
    for op in ops:
        if isinstance(op, Abs):
            stripped = True
            strip_ops.append(op.ufl_operands[0])
        else:
            strip_ops.append(op)

    # Rebuild, and wrap with Abs if necessary
    result = ufl_reuse_if_untouched(o, *strip_ops)
    if stripped:
        result = Abs(result)
    return result


@_simplify_abs.register(Abs)
def _simplify_abs_abs(o, self, in_abs):
    return self(o.ufl_operands[0], True)


def simplify_abs(expression, complex_mode):
    """Simplify absolute values in a UFL expression.  Its primary
    purpose is to "neutralise" CellOrientation nodes that are
    surrounded by absolute values and thus not at all necessary."""
    mapper = MemoizerArg(_simplify_abs)
    mapper.complex_mode = complex_mode
    return mapper(expression, False)


class TSFCConstantMixin:
    """ Mixin class to identify Constants """

    def __init__(self):
        pass


def hash_expr(expr: ufl.core.expr.Expr) -> str:
    """Return a numbering-invariant hash of a UFL expression.

    Parameters
    ----------
    expr :
        A UFL expression.

    Returns
    -------
    str :
        A numbering-invariant hash for the expression.
    """
    domain_numbering = {d: i for i, d in enumerate(ufl.domain.extract_domains(expr))}
    coefficient_numbering = {c: i for i, c in enumerate(extract_coefficients(expr))}
    constant_numbering = {c: i for i, c in enumerate(extract_firedrake_constants(expr))}
    return compute_expression_signature(
        expr, {**domain_numbering, **coefficient_numbering, **constant_numbering}
    )
