from __future__ import absolute_import

import collections
import itertools

import numpy
from singledispatch import singledispatch

from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, Coefficient, FormArgument,
                         QuadratureWeight, ReferenceValue,
                         ScalarValue, Zero)

from ffc.fiatinterface import create_element

from firedrake.fc.modified_terminals import is_modified_terminal, analyse_modified_terminal
from firedrake.fc.constants import PRECISION
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


class Translator(MultiFunction, ModifiedTerminalMixin, FromUFLMixin):

    def __init__(self, weights, quadrature_index, argument_indices, tables, coefficient_map):
        MultiFunction.__init__(self)
        FromUFLMixin.__init__(self)
        self.weights = ein.ListTensor(weights)
        self.quadrature_index = quadrature_index
        self.argument_indices = argument_indices
        self.tables = tables
        self.coefficient_map = coefficient_map

    def modified_terminal(self, o):
        mt = analyse_modified_terminal(o)
        return translate(mt.terminal, o, mt, self)


def tabulate(return_map, ufl_element, max_deriv, points):
    element = create_element(ufl_element)
    phi = element.space_dimension()
    C = ufl_element.value_size() - len(ufl_element.symmetry())
    q = len(points)
    for D, fiat_table in element.tabulate(max_deriv, points).iteritems():
        reordered_table = fiat_table.reshape(phi, C, q).transpose(1, 2, 0)  # (C, phi, q)
        for c, table in enumerate(reordered_table):
            # Copied from FFC (ffc/quadrature/quadratureutils.py)
            table[abs(table) < epsilon] = 0
            table[abs(table - 1.0) < epsilon] = 1.0
            table[abs(table + 1.0) < epsilon] = -1.0
            table[abs(table - 0.5) < epsilon] = 0.5
            table[abs(table + 0.5) < epsilon] = -0.5
            return_map[(ufl_element, c, D)] = table


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
        table = ein.ListTensor(params.tables[key])
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
        return ein.ComponentTensor(
            ein.IndexSum(
                ein.Product(ein.Indexed(ein.ListTensor(table), (q, r)),
                            ein.Indexed(kernel_argument, (r,))),
                r),
            (q,))

    result = numpy.zeros(e.ufl_shape, dtype=object)
    for multiindex, key in zip(numpy.ndindex(e.ufl_shape),
                               table_keys(terminal.ufl_element(),
                                          mt.local_derivatives)):
        evaluated = evaluate(params.tables[key], params.coefficient_map[terminal])
        result[multiindex] = ein.Indexed(evaluated, (params.quadrature_index,))

    if result.shape:
        return ein.ListTensor(result)
    else:
        return result.item()


def process(integrand, quadrature_points, quadrature_weights, argument_indices, coefficient_map):
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

    # Collect tabulation matrices for all components and derivatives
    tables = {}

    for ufl_element, max_deriv in max_derivs.items():
        tabulate(tables, ufl_element, max_deriv, quadrature_points)

    # Translate UFL to Einstein's notation,
    # lowering finite element specific nodes
    quadrature_index = ein.Index()

    translator = Translator(quadrature_weights, quadrature_index, argument_indices, tables, coefficient_map)
    return quadrature_index, map_expr_dag(translator, integrand)
