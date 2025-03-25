"""Utilities for preprocessing UFL objects."""

from functools import singledispatch

import numpy

import ufl
from ufl import as_tensor, indices, replace
from ufl.algorithms import compute_form_data as ufl_compute_form_data
from ufl.algorithms import estimate_total_polynomial_degree
from ufl.algorithms.analysis import extract_arguments, extract_coefficients, extract_type
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms.apply_restrictions import apply_restrictions
from ufl.algorithms.comparison_checker import do_comparison_check
from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
from ufl.algorithms.signature import compute_expression_signature
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.geometry import QuadratureWeight
from ufl.geometry import Jacobian, JacobianDeterminant, JacobianInverse
from ufl.classes import (Abs, Argument, CellOrientation, Coefficient,
                         ComponentTensor, Expr, FloatValue, Division,
                         MultiIndex, Product,
                         ScalarValue, Sqrt, Zero, CellVolume, FacetArea)
from ufl.domain import extract_unique_domain
from finat.ufl import MixedElement
from ufl.utils.sorting import sorted_by_count

from gem.node import MemoizerArg

from tsfc.modified_terminals import (is_modified_terminal,
                                     analyse_modified_terminal,
                                     construct_modified_terminal)


preserve_geometry_types = (CellVolume, FacetArea)


def compute_form_data(form,
                      do_apply_function_pullbacks=True,
                      do_apply_integral_scaling=True,
                      do_apply_geometry_lowering=True,
                      preserve_geometry_types=preserve_geometry_types,
                      do_apply_restrictions=True,
                      do_estimate_degrees=True,
                      complex_mode=False):
    """Preprocess UFL form in a format suitable for TSFC. Return
    form data.

    This is merely a wrapper to UFL compute_form_data with default
    kwargs overriden in the way TSFC needs it and is provided for
    other form compilers based on TSFC.
    """
    fd = ufl_compute_form_data(
        form,
        do_apply_function_pullbacks=do_apply_function_pullbacks,
        do_apply_integral_scaling=do_apply_integral_scaling,
        do_apply_geometry_lowering=do_apply_geometry_lowering,
        preserve_geometry_types=preserve_geometry_types,
        do_apply_restrictions=do_apply_restrictions,
        do_estimate_degrees=do_estimate_degrees,
        complex_mode=complex_mode
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
                           do_apply_function_pullbacks=False)
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
    if complex_mode:
        expression = do_comparison_check(expression)
    else:
        expression = remove_complex_nodes(expression)
    expression = apply_algebra_lowering(expression)
    expression = apply_derivatives(expression)
    expression = apply_function_pullbacks(expression)
    expression = apply_geometry_lowering(expression, preserve_geometry_types)
    expression = apply_derivatives(expression)
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

    reference_grad = _modified_terminal
    reference_value = _modified_terminal

    terminal = _modified_terminal


class CoefficientSplitter(MultiFunction, ModifiedTerminalMixin):
    def __init__(self, split):
        MultiFunction.__init__(self)
        self._split = split

    expr = MultiFunction.reuse_if_untouched

    def modified_terminal(self, o):
        mt = analyse_modified_terminal(o)
        terminal = mt.terminal

        if not isinstance(terminal, Coefficient):
            # Only split coefficients
            return o

        if type(terminal.ufl_element()) != MixedElement:
            # Only split mixed coefficients
            return o

        # Reference value expected
        assert mt.reference_value

        # Derivative indices
        beta = indices(mt.local_derivatives)

        components = []
        for subcoeff in self._split[terminal]:
            # Apply terminal modifiers onto the subcoefficient
            component = construct_modified_terminal(mt, subcoeff)
            # Collect components of the subcoefficient
            for alpha in numpy.ndindex(subcoeff.ufl_element().reference_value_shape):
                # New modified terminal: component[alpha + beta]
                components.append(component[alpha + beta])
        # Repack derivative indices to shape
        c, = indices(1)
        return ComponentTensor(as_tensor(components)[c], MultiIndex((c,) + beta))


def split_coefficients(expression, split):
    """Split mixed coefficients, so mixed elements need not be
    implemented.

    :arg split: A :py:class:`dict` mapping each mixed coefficient to a
                sequence of subcoefficients.  If None, calling this
                function is a no-op.
    """
    if split is None:
        return expression

    splitter = CoefficientSplitter(split)
    return map_expr_dag(splitter, expression)


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
        if isinstance(t, Argument) and r != self.restrictions[t.number()]:
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


def apply_mapping(expression, element, domain):
    """Apply the inverse of the pullback for element to an expression.

    :arg expression: An expression in physical space
    :arg element: The element we're going to interpolate into, whose
         value_shape must match the shape of the expression, and will
         advertise the pullback to apply.
    :arg domain: Optional domain to provide in case expression does
         not contain a domain (used for constructing geometric quantities).
    :returns: A new UFL expression with shape element.reference_value_shape
    :raises NotImplementedError: If we don't know how to apply the
        inverse of the pullback.
    :raises ValueError: If we get shape mismatches.

    The following is borrowed from the UFC documentation:

    Let g be a field defined on a physical domain T with physical
    coordinates x. Let T_0 be a reference domain with coordinates
    X. Assume that F: T_0 -> T such that

      x = F(X)

    Let J be the Jacobian of F, i.e J = dx/dX and let K denote the
    inverse of the Jacobian K = J^{-1}. Then we (currently) have the
    following four types of mappings:

    'identity' mapping for g:

      G(X) = g(x)

    For vector fields g:

    'contravariant piola' mapping for g:

      G(X) = det(J) K g(x)   i.e  G_i(X) = det(J) K_ij g_j(x)

    'covariant piola' mapping for g:

      G(X) = J^T g(x)          i.e  G_i(X) = J^T_ij g(x) = J_ji g_j(x)

    'double covariant piola' mapping for g:

      G(X) = J^T g(x) J     i.e. G_il(X) = J_ji g_jk(x) J_kl

    'double contravariant piola' mapping for g:

      G(X) = det(J)^2 K g(x) K^T  i.e. G_il(X)=(detJ)^2 K_ij g_jk K_lk

    'covariant contravariant piola' mapping for g:

      G(X) = det(J) J^T g(x) K^T     i.e. G_il(X) = det(J) J_ji g_jk(x) K_lk

    If 'contravariant piola' or 'covariant piola' (or their double
    variants) are applied to a matrix-valued function, the appropriate
    mappings are applied row-by-row.
    """
    mesh = extract_unique_domain(expression)
    if mesh is None:
        mesh = domain
    if domain is not None and mesh != domain:
        raise NotImplementedError("Multiple domains not supported")
    pvs = element.pullback.physical_value_shape(element, mesh)
    if expression.ufl_shape != pvs:
        raise ValueError(f"Mismatching shapes, got {expression.ufl_shape}, expected {pvs}")
    mapping = element.mapping().lower()
    if mapping == "identity":
        rexpression = expression
    elif mapping == "covariant piola":
        J = Jacobian(mesh)
        *k, i, j = indices(len(expression.ufl_shape) + 1)
        kj = (*k, j)
        rexpression = as_tensor(J[j, i] * expression[kj], (*k, i))
    elif mapping == "l2 piola":
        detJ = JacobianDeterminant(mesh)
        rexpression = expression * detJ
    elif mapping == "contravariant piola":
        K = JacobianInverse(mesh)
        detJ = JacobianDeterminant(mesh)
        *k, i, j = indices(len(expression.ufl_shape) + 1)
        kj = (*k, j)
        rexpression = as_tensor(detJ * K[i, j] * expression[kj], (*k, i))
    elif mapping == "double covariant piola":
        J = Jacobian(mesh)
        *k, i, j, m, n = indices(len(expression.ufl_shape) + 2)
        kmn = (*k, m, n)
        rexpression = as_tensor(J[m, i] * expression[kmn] * J[n, j], (*k, i, j))
    elif mapping == "double contravariant piola":
        K = JacobianInverse(mesh)
        detJ = JacobianDeterminant(mesh)
        *k, i, j, m, n = indices(len(expression.ufl_shape) + 2)
        kmn = (*k, m, n)
        rexpression = as_tensor(detJ**2 * K[i, m] * expression[kmn] * K[j, n], (*k, i, j))
    elif mapping == "covariant contravariant piola":
        J = Jacobian(mesh)
        K = JacobianInverse(mesh)
        detJ = JacobianDeterminant(mesh)
        *k, i, j, m, n = indices(len(expression.ufl_shape) + 2)
        kmn = (*k, m, n)
        rexpression = as_tensor(detJ * J[m, i] * expression[kmn] * K[j, n], (*k, i, j))
    elif mapping == "symmetries":
        # This tells us how to get from the pieces of the reference
        # space expression to the physical space one.
        # We're going to apply the inverse of the physical to
        # reference space mapping.
        fcm = element.flattened_sub_element_mapping()
        sub_elem = element.sub_elements[0]
        shape = expression.ufl_shape
        flat = ufl.as_vector([expression[i] for i in numpy.ndindex(shape)])
        vs = sub_elem.pullback.physical_value_shape(sub_elem, mesh)
        rvs = sub_elem.reference_value_shape
        seen = set()
        rpieces = []
        gm = int(numpy.prod(vs, dtype=int))
        for gi, ri in enumerate(fcm):
            # For each unique piece in reference space
            if ri in seen:
                continue
            seen.add(ri)
            # Get the physical space piece
            piece = [flat[gm*gi + j] for j in range(gm)]
            piece = as_tensor(numpy.asarray(piece).reshape(vs))
            # get into reference space
            piece = apply_mapping(piece, sub_elem, mesh)
            assert piece.ufl_shape == rvs
            # Concatenate with the other pieces
            rpieces.extend([piece[idx] for idx in numpy.ndindex(rvs)])
        # And reshape
        rexpression = as_tensor(numpy.asarray(rpieces).reshape(element.reference_value_shape))
    else:
        raise NotImplementedError(f"Don't know how to handle mapping type {mapping} for expression of rank {ufl.FunctionSpace(mesh, element).value_shape}")
    if rexpression.ufl_shape != element.reference_value_shape:
        raise ValueError(f"Mismatching reference shapes, got {rexpression.ufl_shape} expected {element.reference_value_shape}")
    return rexpression


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
