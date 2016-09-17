"""Utilities for preprocessing UFL objects."""

from __future__ import absolute_import

import numpy
from singledispatch import singledispatch

import ufl
from ufl import indices, as_tensor
from ufl.algorithms import compute_form_data as ufl_compute_form_data
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Abs, Argument, CellOrientation, Coefficient,
                         ComponentTensor, Expr, FloatValue, Division,
                         MixedElement, MultiIndex, Product,
                         ReferenceValue, ScalarValue, Sqrt, Zero,
                         CellVolume, FacetArea)

from gem.node import MemoizerArg

from tsfc.modified_terminals import (is_modified_terminal,
                                     analyse_modified_terminal,
                                     construct_modified_terminal)


def compute_form_data(form,
                      do_apply_function_pullbacks=True,
                      do_apply_integral_scaling=True,
                      do_apply_geometry_lowering=True,
                      preserve_geometry_types=(CellVolume, FacetArea),
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


def is_element_affine(ufl_element):
    """Tells if a UFL element is affine."""
    affine_cells = ["interval", "triangle", "tetrahedron"]
    return ufl_element.cell().cellname() in affine_cells and ufl_element.degree() == 1


class SpatialCoordinateReplacer(MultiFunction):
    """Replace SpatialCoordinate nodes with the ReferenceValue of a
    Coefficient.  Assumes that the coordinate element only needs
    affine mapping.

    :arg coordinates: the coefficient to replace spatial coordinates with
    """
    def __init__(self, coordinates):
        self.coordinates = coordinates
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        return t

    def spatial_coordinate(self, o):
        assert o.ufl_domain().ufl_coordinate_element().mapping() == "identity"
        return ReferenceValue(self.coordinates)


def replace_coordinates(integrand, coordinate_coefficient):
    """Replace SpatialCoordinate nodes with Coefficients."""
    return map_expr_dag(SpatialCoordinateReplacer(coordinate_coefficient), integrand)


def coordinate_coefficient(domain):
    """Create a fake coordinate coefficient for a domain."""
    return ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))


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
    implemented."""
    splitter = CoefficientSplitter(split)
    return map_expr_dag(splitter, expression)


class CollectModifiedTerminals(MultiFunction, ModifiedTerminalMixin):
    """Collect the modified terminals in a UFL expression.

    :arg return_list: modified terminals will be appended to this list
    """
    def __init__(self, return_list):
        MultiFunction.__init__(self)
        self.return_list = return_list

    def expr(self, o, *ops):
        pass  # operands visited

    def indexed(self, o, *ops):
        pass  # not a terminal modifier

    def multi_index(self, o):
        pass  # ignore

    def modified_terminal(self, o):
        self.return_list.append(o)


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


def _spanning_degree(cell, degree):
    if cell is None:
        assert degree == 0
        return degree
    elif cell.cellname() in ["interval", "triangle", "tetrahedron"]:
        return degree
    elif cell.cellname() == "quadrilateral":
        # TODO: Tensor-product space assumed
        return 2 * degree
    elif isinstance(cell, ufl.TensorProductCell):
        try:
            # A component cell might be a quadrilateral, so recurse.
            return sum(_spanning_degree(sub_cell, d)
                       for sub_cell, d in zip(cell.sub_cells(), degree))
        except TypeError:
            assert degree == 0
            return 0
    else:
        raise ValueError("Unknown cell %s" % cell.cellname())


def spanning_degree(element):
    """Determine the degree of the polynomial space spanning an element.

    :arg element: The element to determine the degree of.

    .. warning::

       For non-simplex elements, this assumes a tensor-product
       space.
    """
    return _spanning_degree(element.cell(), element.degree())


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
