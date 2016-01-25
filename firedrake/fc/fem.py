from __future__ import absolute_import

import collections
import itertools

import numpy
from singledispatch import singledispatch

import ufl
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, Coefficient, FormArgument,
                         QuadratureWeight, ReferenceValue,
                         ScalarValue, Zero, CellFacetJacobian,
                         ReferenceNormal)

from ffc.fiatinterface import create_element, reference_cell

from firedrake.fc.modified_terminals import is_modified_terminal, analyse_modified_terminal
from firedrake.fc.constants import NUMPY_TYPE, PRECISION
from firedrake.fc import einstein as ein
from firedrake.fc.einstein import FromUFLMixin


epsilon = eval("1e-%d" % PRECISION)


class ReplaceSpatialCoordinates(MultiFunction):

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        return t

    def spatial_coordinate(self, o):
        # Firedrake-specific
        mesh = o.ufl_domain()
        return ReferenceValue(mesh.coordinates)


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
            reordered_table = fiat_table.reshape(phi, C, q).transpose(1, 2, 0)  # (C, phi, q)
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
                t = reference_cell(cell).get_facet_transform(entity)
                self.tabulators.append(NumericTabulator(numpy.asarray(map(t, points))))

        elif integral_type in ['exterior_facet_bottom', 'exterior_facet_top', 'interior_facet_horiz']:
            for entity in range(2):  # top and bottom
                t = reference_cell(cell).get_horiz_facet_transform(entity)
                self.tabulators.append(NumericTabulator(numpy.asarray(map(t, points))))

        elif integral_type in ['exterior_facet_vert', 'interior_facet_vert']:
            for entity in range(cell._A.num_facets()):  # "base cell" facets
                t = reference_cell(cell).get_vert_facet_transform(entity)
                self.tabulators.append(NumericTabulator(numpy.asarray(map(t, points))))

        else:
            raise NotImplementedError("integral type %s not supported" % integral_type)

    def tabulate(self, ufl_element, max_deriv):
        for tabulator in self.tabulators:
            tabulator.tabulate(ufl_element, max_deriv)

    def get(self, key, restriction):
        try:
            table = self.tables[key]
        except KeyError:
            tables = [tabulator[key] for tabulator in self.tabulators]

            if self.integral_type == 'cell':
                table, = tables
            else:
                table = numpy.array(tables)

            self.tables[key] = table

        if self.integral_type == 'cell':
            return ein.ListTensor(table)
        else:
            if restriction == '+' or restriction is None:
                f = ein.VariableIndex('facet[0]')
            elif restriction == '-':
                f = ein.VariableIndex('facet[1]')
            else:
                assert False

            i = ein.Index()
            j = ein.Index()
            return ein.ComponentTensor(
                ein.Indexed(
                    ein.ListTensor(table),
                    (f, i, j)),
                (i, j))


class Translator(MultiFunction, ModifiedTerminalMixin, FromUFLMixin):

    def __init__(self, weights, quadrature_index, argument_indices, tabulation_manager, coefficient_map):
        MultiFunction.__init__(self)
        FromUFLMixin.__init__(self)
        self.weights = ein.ListTensor(weights)
        self.quadrature_index = quadrature_index
        self.argument_indices = argument_indices
        self.tabulation_manager = tabulation_manager
        self.coefficient_map = coefficient_map

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


@translate.register(Zero)
def _(terminal, e, mt, params):
    assert False


@translate.register(ScalarValue)
def _(terminal, e, mt, params):
    assert False


@translate.register(QuadratureWeight)
def _(terminal, e, mt, params):
    return ein.Indexed(params.weights, (params.quadrature_index,))


@translate.register(Argument)
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
        return result.item()


@translate.register(Coefficient)
def _(terminal, e, mt, params):
    def evaluate(table, kernel_argument):
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

        return ein.ComponentTensor(
            ein.IndexSum(
                ein.Product(ein.Indexed(table, (q, r)),
                            kar),
                r),
            (q,))

    result = numpy.zeros(e.ufl_shape, dtype=object)
    for multiindex, key in zip(numpy.ndindex(e.ufl_shape),
                               table_keys(terminal.ufl_element(),
                                          mt.local_derivatives)):
        evaluated = evaluate(params.tabulation_manager.get(key, mt.restriction), params.coefficient_map[terminal])
        result[multiindex] = ein.Indexed(evaluated, (params.quadrature_index,))

    if result.shape:
        return ein.ListTensor(result)
    else:
        return result.item()


@translate.register(CellFacetJacobian)
def _(terminal, e, mt, params):
    i = ein.Index()
    j = ein.Index()
    f = ein.VariableIndex('facet[0]')
    return ein.ComponentTensor(
        ein.Indexed(
            ein.ListTensor(make_cell_facet_jacobian(terminal)),
            (f, i, j)),
        (i, j))


@translate.register(ReferenceNormal)
def _(terminal, e, mt, params):
    i = ein.Index()
    f = ein.VariableIndex('facet[0]')
    return ein.ComponentTensor(
        ein.Indexed(
            ein.ListTensor(make_reference_normal(terminal)),
            (f, i,)),
        (i,))


def process(integral_type, integrand, tabulation_manager, quadrature_weights, argument_indices, coefficient_map):
    # Replace SpatialCoordinate nodes with Coefficients
    integrand = map_expr_dag(ReplaceSpatialCoordinates(), integrand)

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
        tabulation_manager.tabulate(ufl_element, max_deriv)

    if integral_type.startswith("interior_facet"):
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(argument_indices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), integrand))
    else:
        expressions = [integrand]

    # Translate UFL to Einstein's notation,
    # lowering finite element specific nodes
    quadrature_index = ein.Index()

    translator = Translator(quadrature_weights, quadrature_index,
                            argument_indices, tabulation_manager,
                            coefficient_map)
    return quadrature_index, [map_expr_dag(translator, e) for e in expressions]


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
