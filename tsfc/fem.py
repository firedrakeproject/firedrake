from __future__ import absolute_import

import collections
import itertools

import numpy
from singledispatch import singledispatch

import ufl
from ufl.corealg.map_dag import map_expr_dag, map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, CellEdgeVectors, CellFacetJacobian,
                         CellOrientation, Coefficient, FormArgument,
                         QuadratureWeight, ReferenceCellVolume,
                         ReferenceNormal, ReferenceValue, ScalarValue,
                         Zero)
from ufl.domain import find_geometric_dimension

from tsfc.fiatinterface import create_element, as_fiat_cell

from tsfc.modified_terminals import is_modified_terminal, analyse_modified_terminal
from tsfc.constants import NUMPY_TYPE, PRECISION
from tsfc import gem as ein
from tsfc import ufl2gem


# FFC uses one less digits for rounding than for printing
epsilon = eval("1e-%d" % (PRECISION - 1))


class ReplaceSpatialCoordinates(MultiFunction):

    def __init__(self, coordinates):
        self.coordinates = coordinates
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        return t

    def spatial_coordinate(self, o):
        return ReferenceValue(self.coordinates)


class ModifiedTerminalMixin(object):

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


class CollectModifiedTerminals(MultiFunction, ModifiedTerminalMixin):

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


class FindPolynomialDegree(MultiFunction):

    """Simple-minded degree estimator.

    Attempt to estimate the polynomial degree of an expression.  Used
    to determine whether something we're taking a gradient of is
    cellwise constant.  Returns either the degree of the expression,
    or else ``None`` if the degree could not be determined.

    To do this properly, we'd need to carry around a tensor-valued
    degree object such that we can determine when (say) d^2/dx^2 is
    zero but d^2/dxdy is not.

    """
    def _spanning_degree(self, element):
        """Determine the degree of the polynomial space spanning an element.

        :arg element: The element to determine the degree of.

        .. warning::

           For non-simplex elements, this assumes a tensor-product
           space.
        """
        cell = element.cell()
        if cell is None:
            return element.degree()
        if cell.cellname() in ("interval", "triangle", "tetrahedron"):
            return element.degree()
        elif cell.cellname() == "quadrilateral":
            # TODO: Tensor-product space assumed
            return 2*element.degree()
        elif cell.cellname() == "OuterProductCell":
            try:
                return sum(element.degree())
            except TypeError:
                assert element.degree() == 0
                return 0
        else:
            raise ValueError("Unknown cell %s" % cell.cellname())

    def quadrature_weight(self, o):
        return 0

    def multi_index(self, o):
        return 0

    # Default handler, no estimation.
    def expr(self, o):
        return None

    # Coefficient-like things, compute degree of spanning polynomial space
    def spatial_coordinate(self, o):
        return self._spanning_degree(o.ufl_domain().ufl_coordinate_element())

    def form_argument(self, o):
        return self._spanning_degree(o.ufl_element())

    # Index-like operations, return degree of operand
    def component_tensor(self, o, op, idx):
        return op

    def indexed(self, o, op, idx):
        return op

    def index_sum(self, o, op, idx):
        return op

    def list_tensor(self, o, *ops):
        if any(ops is None for op in ops):
            return None
        return max(*ops)

    # No change
    def reference_value(self, o, op):
        return op

    def restricted(self, o, op):
        return op

    # Constants are constant
    def constant_value(self, o):
        return 0

    # Multiplication adds degrees
    def product(self, o, a, b):
        if a is None or b is None:
            return None
        return a + b

    # If the degree of the exponent is zero, use degree of operand,
    # otherwise don't guess.
    def power(self, o, a, b):
        if b == 0:
            return a
        return None

    # Pick maximal degree
    def conditional(self, o, test, a, b):
        if a is None or b is None:
            return None
        return max(a, b)

    def min_value(self, o, a, b):
        if a is None or b is None:
            return None
        return max(a, b)

    def max_value(self, o, a, b):
        if a is None or b is None:
            return None
        return max(a, b)

    def sum(self, o, a, b):
        if a is None or b is None:
            return None
        return max(a, b)

    # If denominator is constant, use degree of numerator, otherwise
    # don't guess
    def division(self, o, a, b):
        if b == 0:
            return a
        return None

    def abs(self, o, a):
        if a == 0:
            return a
        return None

    # If operand is constant, return 0, otherwise don't guess.
    def math_function(self, o, op):
        if op == 0:
            return 0
        return None

    # Reduce degrees!
    def reference_grad(self, o, degree):
        if degree is None:
            return None
        return max(degree - 1, 0)


class SimplifyExpr(MultiFunction):
    """Apply some simplification passes to an expression."""

    def __init__(self):
        MultiFunction.__init__(self)
        self.mapper = FindPolynomialDegree()

    expr = MultiFunction.reuse_if_untouched

    def reference_grad(self, o):
        """Try and zero-simplify ``RGrad(expr)`` where the degree of
        ``expr`` can be determined.

        Uses :class:`FindPolynomialDegree` to determine the degree of
        ``expr``."""
        # Find degree of operand
        degree = map_expr_dag(self.mapper, o.ufl_operands[0])
        # Either we have non-constant, or we didn't know, in which
        # case return ourselves.
        if degree is None or degree > 0:
            return o
        # We are RGrad(constant-function), return Zero of appropriate shape
        op = o.ufl_operands[0]
        gdim = find_geometric_dimension(op)
        return ufl.classes.Zero(op.ufl_shape + (gdim, ),
                                op.ufl_free_indices,
                                op.ufl_index_dimensions)

    def abs(self, o, op):
        """Convert Abs(CellOrientation * ...) -> Abs(...)"""
        if isinstance(op, ufl.classes.CellOrientation):
            # Cell orientation is +-1
            return ufl.classes.FloatValue(1)
        if isinstance(op, ufl.classes.ScalarValue):
            # Inline abs(constant)
            return self.expr(op, abs(op._value))
        if isinstance(op, (ufl.classes.Division, ufl.classes.Product)):
            # Visit children, distributing Abs
            ops = tuple(map_expr_dag(self, ufl.classes.Abs(_))
                        for _ in op.ufl_operands)
            new_ops = []
            # Strip Abs off again (we'll put it outside the product now)
            for _ in ops:
                if isinstance(_, ufl.classes.Abs):
                    new_ops.append(_.ufl_operands[0])
                else:
                    new_ops.append(_)
            # Rebuild product
            new_prod = self.expr(op, *new_ops)
            # Rebuild Abs
            return self.expr(o, new_prod)
        return self.expr(o, op)


class NumericTabulator(object):

    def __init__(self, points):
        self.points = points
        self.tables = {}

    def tabulate(self, ufl_element, max_deriv):
        element = create_element(ufl_element)
        phi = element.space_dimension()
        C = ufl_element.reference_value_size() - len(ufl_element.symmetry())
        q = len(self.points)
        for D, fiat_table in element.tabulate(max_deriv, self.points).iteritems():
            reordered_table = fiat_table.reshape(phi, C, q).transpose(1, 2, 0)  # (C, q, phi)
            for c, table in enumerate(reordered_table):
                # Copied from FFC (ffc/quadrature/quadratureutils.py)
                table[abs(table) < epsilon] = 0
                table[abs(table - 1.0) < epsilon] = 1.0
                table[abs(table + 1.0) < epsilon] = -1.0
                table[abs(table - 0.5) < epsilon] = 0.5
                table[abs(table + 0.5) < epsilon] = -0.5
                self.tables[(ufl_element, c, D)] = table

    def __getitem__(self, key):
        return self.tables[key]


class TabulationManager(object):

    def __init__(self, integral_type, cell, points):
        self.integral_type = integral_type
        self.cell = cell
        self.points = points

        self.tabulators = []
        self.tables = {}

        if integral_type == 'cell':
            self.tabulators.append(NumericTabulator(points))

        elif integral_type in ['exterior_facet', 'interior_facet']:
            # TODO: handle and test integration on facets of intervals

            for entity in range(cell.num_facets()):
                t = as_fiat_cell(cell).get_facet_transform(entity)
                self.tabulators.append(NumericTabulator(numpy.asarray(map(t, points))))

        elif integral_type in ['exterior_facet_bottom', 'exterior_facet_top', 'interior_facet_horiz']:
            for entity in range(2):  # top and bottom
                t = as_fiat_cell(cell).get_horiz_facet_transform(entity)
                self.tabulators.append(NumericTabulator(numpy.asarray(map(t, points))))

        elif integral_type in ['exterior_facet_vert', 'interior_facet_vert']:
            for entity in range(cell._A.num_facets()):  # "base cell" facets
                t = as_fiat_cell(cell).get_vert_facet_transform(entity)
                self.tabulators.append(NumericTabulator(numpy.asarray(map(t, points))))

        else:
            raise NotImplementedError("integral type %s not supported" % integral_type)

        if integral_type in ['exterior_facet', 'exterior_facet_vert']:
            self.facet = {None: ein.VariableIndex('facet[0]')}
        elif integral_type in ['interior_facet', 'interior_facet_vert']:
            self.facet = {'+': ein.VariableIndex('facet[0]'),
                          '-': ein.VariableIndex('facet[1]')}
        elif integral_type == 'exterior_facet_bottom':
            self.facet = {None: 0}
        elif integral_type == 'exterior_facet_top':
            self.facet = {None: 1}
        elif integral_type == 'interior_facet_horiz':
            self.facet = {'+': 1, '-': 0}
        else:
            self.facet = None

    def tabulate(self, ufl_element, max_deriv):
        for tabulator in self.tabulators:
            tabulator.tabulate(ufl_element, max_deriv)

    def get(self, key, restriction, cellwise_constant=False):
        try:
            table = self.tables[(key, cellwise_constant)]
        except KeyError:
            tables = [tabulator[key] for tabulator in self.tabulators]
            if cellwise_constant:
                tables = [table[0] for table in tables]

            if self.integral_type == 'cell':
                table, = tables
            else:
                table = numpy.array(tables)

            self.tables[(key, cellwise_constant)] = table

        if self.integral_type == 'cell':
            return ein.Literal(table)
        else:
            f = self.facet[restriction]

            indices = tuple(ein.Index() for i in range(len(table.shape)-1))
            return ein.ComponentTensor(
                ein.Indexed(
                    ein.Literal(table),
                    (f,) + indices),
                indices)


class Translator(MultiFunction, ModifiedTerminalMixin, ufl2gem.Mixin):

    def __init__(self, weights, quadrature_index, argument_indices, tabulation_manager, coefficient_map):
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)
        self.weights = ein.Literal(weights)
        self.quadrature_index = quadrature_index
        self.argument_indices = argument_indices
        self.tabulation_manager = tabulation_manager
        self.coefficient_map = coefficient_map
        self.cell_orientations = False
        self.facet = tabulation_manager.facet

    def modified_terminal(self, o):
        mt = analyse_modified_terminal(o)
        return translate(mt.terminal, o, mt, self)


def table_keys(ufl_element, local_derivatives):
    # TODO:
    # Consider potential duplicate calculation due to second
    # derivatives and symmetries.

    size = ufl_element.reference_value_size()
    dim = ufl_element.cell().topological_dimension()

    def flat_index(ordered_deriv):
        result = [0] * dim
        for i in ordered_deriv:
            result[i] += 1
        return tuple(result)

    ordered_derivs = itertools.product(range(dim), repeat=local_derivatives)
    flat_derivs = map(flat_index, ordered_derivs)

    return [(ufl_element, c, flat_deriv)
            for c in xrange(size)
            for flat_deriv in flat_derivs]


@singledispatch
def translate(terminal, e, mt, params):
    raise AssertionError("Cannot handle terminal type: %s" % type(terminal))


@translate.register(Zero)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    assert False


@translate.register(ScalarValue)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    assert False


@translate.register(QuadratureWeight)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    return ein.Indexed(params.weights, (params.quadrature_index,))


@translate.register(Argument)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    argument_index = params.argument_indices[terminal.number()]

    result = numpy.zeros(e.ufl_shape, dtype=object)
    for multiindex, key in zip(numpy.ndindex(e.ufl_shape),
                               table_keys(terminal.ufl_element(),
                                          mt.local_derivatives)):
        table = params.tabulation_manager.get(key, mt.restriction)
        result[multiindex] = ein.Indexed(table, (params.quadrature_index, argument_index))

    if result.shape:
        return ein.ListTensor(result)
    else:
        return result[()]


@translate.register(Coefficient)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    degree = map_expr_dag(FindPolynomialDegree(), e)
    cellwise_constant = not (degree is None or degree > 0)

    def evaluate_at(params, key):
        table = params.tabulation_manager.get(key, mt.restriction, cellwise_constant)
        kernel_argument = params.coefficient_map[terminal]

        q = ein.Index()
        r = ein.Index()

        if mt.restriction is None:
            kar = ein.Indexed(kernel_argument, (r,))
        elif mt.restriction is '+':
            kar = ein.Indexed(kernel_argument, (0, r))
        elif mt.restriction is '-':
            kar = ein.Indexed(kernel_argument, (1, r))
        else:
            assert False

        if cellwise_constant:
            return ein.IndexSum(ein.Product(ein.Indexed(table, (r,)), kar), r)
        else:
            return ein.Indexed(
                ein.ComponentTensor(
                    ein.IndexSum(
                        ein.Product(ein.Indexed(table, (q, r)),
                                    kar),
                        r),
                    (q,)),
                (params.quadrature_index,))

    if terminal.ufl_element().family() == 'Real':
        assert mt.local_derivatives == 0
        return params.coefficient_map[terminal]

    result = numpy.zeros(e.ufl_shape, dtype=object)
    for multiindex, key in zip(numpy.ndindex(e.ufl_shape),
                               table_keys(terminal.ufl_element(),
                                          mt.local_derivatives)):
        result[multiindex] = evaluate_at(params, key)

    if result.shape:
        return ein.ListTensor(result)
    else:
        return result[()]


@translate.register(CellFacetJacobian)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    i = ein.Index()
    j = ein.Index()
    f = params.facet[mt.restriction]
    table = make_cell_facet_jacobian(terminal)
    if params.tabulation_manager.integral_type in ["exterior_facet_bottom",
                                                   "exterior_facet_top",
                                                   "interior_facet_horiz"]:
        table = table[:2]
    elif params.tabulation_manager.integral_type in ["exterior_facet_vert", "interior_facet_vert"]:
        table = table[2:]
    return ein.ComponentTensor(
        ein.Indexed(
            ein.Literal(table),
            (f, i, j)),
        (i, j))


@translate.register(ReferenceNormal)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    i = ein.Index()
    f = params.facet[mt.restriction]
    table = make_reference_normal(terminal)
    if params.tabulation_manager.integral_type in ["exterior_facet_bottom",
                                                   "exterior_facet_top",
                                                   "interior_facet_horiz"]:
        table = table[:2]
    elif params.tabulation_manager.integral_type in ["exterior_facet_vert", "interior_facet_vert"]:
        table = table[2:]
    return ein.ComponentTensor(
        ein.Indexed(
            ein.Literal(table),
            (f, i,)),
        (i,))


@translate.register(CellOrientation)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    if mt.restriction == '+' or mt.restriction is None:
        f = 0
    elif mt.restriction == '-':
        f = 1
    else:
        assert False
    params.cell_orientations = True
    raw = ein.Indexed(ein.Variable("cell_orientations", (2, 1)), (f, 0))  # TODO: (2, 1) and (1, 1)
    return ein.Conditional(ein.Comparison("==", raw, ein.Literal(1)),
                           ein.Literal(-1),
                           ein.Conditional(ein.Comparison("==", raw, ein.Zero()),
                                           ein.Literal(1),
                                           ein.Literal(numpy.nan)))


@translate.register(ReferenceCellVolume)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    cell = terminal.ufl_domain().ufl_cell()
    volume = {ufl.Cell("interval"): 1.0,
              ufl.Cell("triangle"): 1.0/2.0,
              ufl.Cell("quadrilateral"): 1.0,
              ufl.Cell("tetrahedron"): 1.0/6.0,
              ufl.OuterProductCell(ufl.Cell("interval"), ufl.Cell("interval")): 1.0,
              ufl.OuterProductCell(ufl.Cell("triangle"), ufl.Cell("interval")): 1.0/2.0,
              ufl.OuterProductCell(ufl.Cell("quadrilateral"), ufl.Cell("interval")): 1.0}
    return ein.Literal(volume[cell])


@translate.register(CellEdgeVectors)  # noqa: Not actually redefinition
def _(terminal, e, mt, params):
    return ein.Literal(make_cell_edge_vectors(terminal))


def coordinate_coefficient(domain):
    return ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))


def replace_coordinates(integrand, coordinate_coefficient):
    # Replace SpatialCoordinate nodes with Coefficients
    return map_expr_dag(ReplaceSpatialCoordinates(coordinate_coefficient), integrand)


def process(integral_type, integrand, tabulation_manager, quadrature_weights, quadrature_index, argument_indices, coefficient_map):
    # Abs-simplification
    integrand = map_expr_dag(SimplifyExpr(), integrand)

    # Collect modified terminals
    modified_terminals = []
    map_expr_dag(CollectModifiedTerminals(modified_terminals), integrand)

    # Collect maximal derivatives that needs tabulation
    max_derivs = collections.defaultdict(int)

    for mt in map(analyse_modified_terminal, modified_terminals):
        if isinstance(mt.terminal, FormArgument):
            ufl_element = mt.terminal.ufl_element()
            max_derivs[ufl_element] = max(mt.local_derivatives, max_derivs[ufl_element])

    # Collect tabulations for all components and derivatives
    for ufl_element, max_deriv in max_derivs.items():
        if ufl_element.family() != 'Real':
            tabulation_manager.tabulate(ufl_element, max_deriv)

    if integral_type.startswith("interior_facet"):
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(argument_indices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), integrand))
    else:
        expressions = [integrand]

    # Translate UFL to Einstein's notation,
    # lowering finite element specific nodes
    translator = Translator(quadrature_weights, quadrature_index,
                            argument_indices, tabulation_manager,
                            coefficient_map)
    return map_expr_dags(translator, expressions), translator.cell_orientations


def make_cell_facet_jacobian(terminal):

    interval = numpy.array([[1.0],
                            [1.0]], dtype=NUMPY_TYPE)

    triangle = numpy.array([[-1.0, 1.0],
                            [0.0, 1.0],
                            [1.0, 0.0]], dtype=NUMPY_TYPE)

    tetrahedron = numpy.array([[-1.0, -1.0, 1.0, 0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                               [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=NUMPY_TYPE)

    quadrilateral = numpy.array([[0.0, 1.0],
                                 [0.0, 1.0],
                                 [1.0, 0.0],
                                 [1.0, 0.0]], dtype=NUMPY_TYPE)

    # Outer product cells
    # Convention is:
    # Bottom facet, top facet, then the extruded facets in the order
    # of the base cell
    interval_x_interval = numpy.array([[1.0, 0.0],
                                       [1.0, 0.0],
                                       [0.0, 1.0],
                                       [0.0, 1.0]], dtype=NUMPY_TYPE)

    triangle_x_interval = numpy.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                       [-1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                       [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                       [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=NUMPY_TYPE)

    quadrilateral_x_interval = numpy.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                                           dtype=NUMPY_TYPE)

    cell = terminal.ufl_domain().ufl_cell()
    cell = cell.reconstruct(geometric_dimension=cell.topological_dimension())

    cell_to_table = {ufl.Cell("interval"): interval,
                     ufl.Cell("triangle"): triangle,
                     ufl.Cell("quadrilateral"): quadrilateral,
                     ufl.Cell("tetrahedron"): tetrahedron,
                     ufl.OuterProductCell(ufl.Cell("interval"), ufl.Cell("interval")): interval_x_interval,
                     ufl.OuterProductCell(ufl.Cell("triangle"), ufl.Cell("interval")): triangle_x_interval,
                     ufl.OuterProductCell(ufl.Cell("quadrilateral"), ufl.Cell("interval")): quadrilateral_x_interval}

    table = cell_to_table[cell]

    shape = table.shape[:1] + terminal.ufl_shape
    return table.reshape(shape)


def make_reference_normal(terminal):
    interval = numpy.array([[-1.0],
                            [1.0]], dtype=NUMPY_TYPE)

    triangle = numpy.array([[1.0, 1.0],
                            [-1.0, 0.0],
                            [0.0, -1.]], dtype=NUMPY_TYPE)

    tetrahedron = numpy.array([[1.0, 1.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [0.0, 0.0, -1.0]], dtype=NUMPY_TYPE)

    quadrilateral = numpy.array([[-1.0, 0.0],
                                 [1.0, 0.0],
                                 [0.0, -1.0],
                                 [0.0, 1.0]], dtype=NUMPY_TYPE)

    interval_x_interval = numpy.array([[0.0, -1.0],
                                       [0.0, 1.0],
                                       [-1.0, 0.0],
                                       [1.0, 0.0]], dtype=NUMPY_TYPE)

    triangle_x_interval = numpy.array([[0.0, 0.0, -1.0],
                                       [0.0, 0.0, 1.0],
                                       [1.0, 1.0, 0.0],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, -1.0, 0.0]], dtype=NUMPY_TYPE)

    quadrilateral_x_interval = numpy.array([[0.0, 0.0, -1.0],
                                            [0.0, 0.0, 1.0],
                                            [-1.0, 0.0, 0.0],
                                            [1.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0],
                                            [0.0, 1.0, 0.0]], dtype=NUMPY_TYPE)

    cell = terminal.ufl_domain().ufl_cell()
    cell = cell.reconstruct(geometric_dimension=cell.topological_dimension())

    cell_to_table = {ufl.Cell("interval"): interval,
                     ufl.Cell("triangle"): triangle,
                     ufl.Cell("quadrilateral"): quadrilateral,
                     ufl.Cell("tetrahedron"): tetrahedron,
                     ufl.OuterProductCell(ufl.Cell("interval"), ufl.Cell("interval")): interval_x_interval,
                     ufl.OuterProductCell(ufl.Cell("triangle"), ufl.Cell("interval")): triangle_x_interval,
                     ufl.OuterProductCell(ufl.Cell("quadrilateral"), ufl.Cell("interval")): quadrilateral_x_interval}

    table = cell_to_table[cell]

    shape = table.shape[:1] + terminal.ufl_shape
    return table.reshape(shape)


def make_cell_edge_vectors(terminal):
    from FIAT.reference_element import two_product_cell
    shape = terminal.ufl_shape
    cell = as_fiat_cell(terminal.ufl_domain().ufl_cell())
    if isinstance(cell, two_product_cell):
        raise NotImplementedError("CEV not implemented on OPEs yet")
    nedge = len(cell.get_topology()[1])
    vecs = numpy.vstack(tuple(cell.compute_edge_tangent(i) for i in range(nedge))).astype(NUMPY_TYPE)

    assert vecs.shape == shape
    return vecs
