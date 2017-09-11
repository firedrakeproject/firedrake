"""Utilities for preprocessing UFL objects."""

from __future__ import absolute_import, print_function, division

import numpy
from singledispatch import singledispatch

import ufl
from ufl import as_tensor, indices, replace
from ufl.algorithms import compute_form_data as ufl_compute_form_data
from ufl.algorithms import estimate_total_polynomial_degree
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.geometry import QuadratureWeight
from ufl.classes import (Abs, Argument, CellOrientation, Coefficient,
                         ComponentTensor, Expr, FloatValue, Division,
                         MixedElement, MultiIndex, Product,
                         ScalarValue, Sqrt, Zero, CellVolume,
                         FacetArea)

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
                      do_estimate_degrees=True):
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
    )
    return fd


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


def preprocess_expression(expression):
    """Imitates the compute_form_data processing pipeline.

    Useful, for example, to preprocess non-scalar expressions, which
    are not and cannot be forms.
    """
    expression = apply_algebra_lowering(expression)
    expression = apply_derivatives(expression)
    expression = apply_function_pullbacks(expression)
    expression = apply_geometry_lowering(expression, preserve_geometry_types)
    expression = apply_derivatives(expression)
    expression = apply_geometry_lowering(expression, preserve_geometry_types)
    expression = apply_derivatives(expression)
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

    cell_avg = _modified_terminal
    facet_avg = _modified_terminal

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
            for alpha in numpy.ndindex(subcoeff.ufl_element().reference_value_shape()):
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
    # Square root is always non-negative
    return ufl_reuse_if_untouched(o, self(o.ufl_operands[0], False))


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


def simplify_abs(expression):
    """Simplify absolute values in a UFL expression.  Its primary
    purpose is to "neutralise" CellOrientation nodes that are
    surrounded by absolute values and thus not at all necessary."""
    return MemoizerArg(_simplify_abs)(expression, False)
