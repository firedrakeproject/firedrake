from __future__ import absolute_import, print_function, division
from six import iteritems
from six.moves import map, range

import collections
import itertools
from functools import reduce

import numpy
from singledispatch import singledispatch

from ufl.corealg.map_dag import map_expr_dag, map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, Coefficient, CellVolume,
                         FacetArea, FormArgument,
                         GeometricQuantity, QuadratureWeight)

import gem

from tsfc.constants import PRECISION
from tsfc.fiatinterface import create_element, create_quadrature, as_fiat_cell
from tsfc.modified_terminals import analyse_modified_terminal
from tsfc import compat
from tsfc import ufl2gem
from tsfc import geometric
from tsfc.ufl_utils import (CollectModifiedTerminals,
                            ModifiedTerminalMixin, PickRestriction,
                            spanning_degree, simplify_abs)


# FFC uses one less digits for rounding than for printing
epsilon = eval("1e-%d" % (PRECISION - 1))


def _tabulate(ufl_element, order, points):
    """Ask FIAT to tabulate ``points`` up to order ``order``, then
    rearranges the result into a series of ``(c, D, table)`` tuples,
    where:

    c: component index (for vector-valued and tensor-valued elements)
    D: derivative tuple (e.g. (1, 2) means d/dx d^2/dy^2)
    table: tabulation matrix for the given component and derivative.
           shape: len(points) x space_dimension

    :arg ufl_element: element to tabulate
    :arg order: FIAT gives all derivatives up to this order
    :arg points: points to tabulate the element on
    """
    element = create_element(ufl_element)
    phi = element.space_dimension()
    C = ufl_element.reference_value_size()
    q = len(points)
    for D, fiat_table in iteritems(element.tabulate(order, points)):
        reordered_table = fiat_table.reshape(phi, C, q).transpose(1, 2, 0)  # (C, q, phi)
        for c, table in enumerate(reordered_table):
            yield c, D, table


def tabulate(ufl_element, order, points):
    """Same as the above, but also applies FFC rounding and recognises
    cellwise constantness.  Cellwise constantness is determined
    symbolically, but we also check the numerics to be safe."""
    for c, D, table in _tabulate(ufl_element, order, points):
        # Copied from FFC (ffc/quadrature/quadratureutils.py)
        table[abs(table) < epsilon] = 0
        table[abs(table - 1.0) < epsilon] = 1.0
        table[abs(table + 1.0) < epsilon] = -1.0
        table[abs(table - 0.5) < epsilon] = 0.5
        table[abs(table + 0.5) < epsilon] = -0.5

        if spanning_degree(ufl_element) <= sum(D):
            assert compat.allclose(table, table.mean(axis=0, keepdims=True), equal_nan=True)
            table = table[0]

        yield c, D, table


def make_tabulator(points):
    """Creates a tabulator for an array of points."""
    return lambda elem, order: tabulate(elem, order, points)


class TabulationManager(object):
    """Manages the generation of tabulation matrices for the different
    integral types."""

    def __init__(self, entity_points):
        """Constructs a TabulationManager.

        :arg entity_points: An array of points in cell coordinates for
                            each integration entity, i.e. an iterable
                            of arrays of points.
        """
        self.tabulators = list(map(make_tabulator, entity_points))
        self.tables = {}

    def tabulate(self, ufl_element, max_deriv):
        """Prepare the tabulations of a finite element up to a given
        derivative order.

        :arg ufl_element: UFL element to tabulate
        :arg max_deriv: tabulate derivatives up this order
        """
        store = collections.defaultdict(list)
        for tabulator in self.tabulators:
            for c, D, table in tabulator(ufl_element, max_deriv):
                store[(ufl_element, c, D)].append(table)

        for key, tables in iteritems(store):
            table = numpy.array(tables)
            if len(table.shape) == 2:
                # Cellwise constant; must not depend on the facet
                assert compat.allclose(table, table.mean(axis=0, keepdims=True), equal_nan=True)
                table = table[0]
            self.tables[key] = table

    def __getitem__(self, key):
        return self.tables[key]


# FIXME: copy-paste from PyOP2
class cached_property(object):
    """A read-only @property that is only evaluated once. The value is cached
    on the object itself rather than the function or class; this should prevent
    memory leakage."""
    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        obj.__dict__[self.__name__] = result = self.fget(obj)
        return result


class Parameters(object):
    keywords = ('cell',
                'fiat_cell',
                'integration_dim',
                'entity_ids',
                'quadrature_degree',
                'quadrature_rule',
                'points',
                'weights',
                'point_index',
                'argument_indices',
                'coefficient',
                'cell_orientation',
                'facet_number',
                'cellvolume',
                'facetarea',
                'index_cache')

    def __init__(self, **kwargs):
        invalid_keywords = set(kwargs.keys()) - set(Parameters.keywords)
        if invalid_keywords:
            raise ValueError("unexpected keyword argument '{0}'".format(invalid_keywords.pop()))
        self.__dict__.update(kwargs)

    @cached_property
    def fiat_cell(self):
        return as_fiat_cell(self.cell)

    @cached_property
    def integration_dim(self):
        return self.fiat_cell.get_dimension()

    entity_ids = [0]

    @cached_property
    def quadrature_rule(self):
        integration_cell = self.fiat_cell.construct_subelement(self.integration_dim)
        return create_quadrature(integration_cell, self.quadrature_degree)

    @cached_property
    def points(self):
        return self.quadrature_rule.get_points()

    @cached_property
    def weights(self):
        return self.quadrature_rule.get_weights()

    @cached_property
    def entity_points(self):
        """An array of points in cell coordinates for each entity,
        i.e. a list of arrays of points."""
        result = []
        for entity_id in self.entity_ids:
            t = self.fiat_cell.get_entity_transform(self.integration_dim, entity_id)
            result.append(numpy.asarray(list(map(t, self.points))))
        return result

    def _selector(self, callback, opts, restriction):
        """Helper function for selecting code for the correct entity
        at run-time."""
        if len(opts) == 1:
            return callback(opts[0])
        else:
            results = gem.ListTensor(list(map(callback, opts)))
            f = self.facet_number(restriction)
            return gem.partial_indexed(results, (f,))

    def entity_selector(self, callback, restriction):
        """Selects code for the correct entity at run-time.  Callback
        generates code for a specified entity.

        This function passes ``callback`` the entity number.

        :arg callback: A function to be called with an entity number
                       that generates code for that entity.
        :arg restriction: Restriction of the modified terminal, used
                          for entity selection.
        """
        return self._selector(callback, self.entity_ids, restriction)

    def index_selector(self, callback, restriction):
        """Selects code for the correct entity at run-time.  Callback
        generates code for a specified entity.

        This function passes ``callback`` an index of the entity
        numbers array.

        :arg callback: A function to be called with an entity index
                       that generates code for that entity.
        :arg restriction: Restriction of the modified terminal, used
                          for entity selection.
        """
        return self._selector(callback, list(range(len(self.entity_ids))), restriction)

    argument_indices = ()

    @cached_property
    def index_cache(self):
        return collections.defaultdict(gem.Index)


class Translator(MultiFunction, ModifiedTerminalMixin, ufl2gem.Mixin):
    """Contains all the context necessary to translate UFL into GEM."""

    def __init__(self, tabulation_manager, parameters):
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)

        parameters.tabulation_manager = tabulation_manager
        self.parameters = parameters

    def modified_terminal(self, o):
        """Overrides the modified terminal handler from
        :class:`ModifiedTerminalMixin`."""
        mt = analyse_modified_terminal(o)
        return translate(mt.terminal, mt, self.parameters)


def iterate_shape(mt, callback):
    """Iterates through the components of a modified terminal, and
    calls ``callback`` with ``(ufl_element, c, D)`` keys which are
    used to look up tabulation matrix for that component.  Then
    assembles the result into a GEM tensor (if tensor-valued)
    corresponding to the modified terminal.

    :arg mt: analysed modified terminal
    :arg callback: callback to get the GEM translation of a component
    :returns: GEM translation of the modified terminal

    This is a helper for translating Arguments and Coefficients.
    """
    ufl_element = mt.terminal.ufl_element()
    dim = ufl_element.cell().topological_dimension()

    def flat_index(ordered_deriv):
        return tuple((numpy.asarray(ordered_deriv) == d).sum() for d in range(dim))

    ordered_derivs = itertools.product(range(dim), repeat=mt.local_derivatives)
    flat_derivs = list(map(flat_index, ordered_derivs))

    result = []
    for c in range(ufl_element.reference_value_size()):
        for flat_deriv in flat_derivs:
            result.append(callback((ufl_element, c, flat_deriv)))

    shape = mt.expr.ufl_shape
    assert len(result) == numpy.prod(shape)

    if shape:
        return gem.ListTensor(numpy.asarray(result).reshape(shape))
    else:
        return result[0]


@singledispatch
def translate(terminal, mt, params):
    """Translates modified terminals into GEM.

    :arg terminal: terminal, for dispatching
    :arg mt: analysed modified terminal
    :arg params: translator context
    :returns: GEM translation of the modified terminal
    """
    raise AssertionError("Cannot handle terminal type: %s" % type(terminal))


@translate.register(QuadratureWeight)
def translate_quadratureweight(terminal, mt, params):
    return gem.Indexed(gem.Literal(params.weights), (params.point_index,))


@translate.register(GeometricQuantity)
def translate_geometricquantity(terminal, mt, params):
    return geometric.translate(terminal, mt, params)


@translate.register(CellVolume)
def translate_cellvolume(terminal, mt, params):
    return params.cellvolume(mt.restriction)


@translate.register(FacetArea)
def translate_facetarea(terminal, mt, params):
    return params.facetarea()


@translate.register(Argument)
def translate_argument(terminal, mt, params):
    argument_index = params.argument_indices[terminal.number()]

    def callback(key):
        table = params.tabulation_manager[key]
        if len(table.shape) == 1:
            # Cellwise constant
            row = gem.Literal(table)
        else:
            table = params.index_selector(lambda i: gem.Literal(table[i]), mt.restriction)
            row = gem.partial_indexed(table, (params.point_index,))
        return gem.Indexed(row, (argument_index,))

    return iterate_shape(mt, callback)


@translate.register(Coefficient)
def translate_coefficient(terminal, mt, params):
    vec = params.coefficient(terminal, mt.restriction)

    if terminal.ufl_element().family() == 'Real':
        assert mt.local_derivatives == 0
        return vec

    def callback(key):
        table = params.tabulation_manager[key]
        if len(table.shape) == 1:
            # Cellwise constant
            row = gem.Literal(table)
            if numpy.count_nonzero(table) <= 2:
                assert row.shape == vec.shape
                return reduce(gem.Sum,
                              [gem.Product(gem.Indexed(row, (i,)), gem.Indexed(vec, (i,)))
                               for i in range(row.shape[0])],
                              gem.Zero())
        else:
            table = params.index_selector(lambda i: gem.Literal(table[i]), mt.restriction)
            row = gem.partial_indexed(table, (params.point_index,))

        r = params.index_cache[terminal.ufl_element()]
        return gem.IndexSum(gem.Product(gem.Indexed(row, (r,)),
                                        gem.Indexed(vec, (r,))), r)

    return iterate_shape(mt, callback)


def compile_ufl(expression, interior_facet=False, **kwargs):
    params = Parameters(**kwargs)

    # Abs-simplification
    expression = simplify_abs(expression)

    # Collect modified terminals
    modified_terminals = []
    map_expr_dag(CollectModifiedTerminals(modified_terminals), expression)

    # Collect maximal derivatives that needs tabulation
    max_derivs = collections.defaultdict(int)

    for mt in map(analyse_modified_terminal, modified_terminals):
        if isinstance(mt.terminal, FormArgument):
            ufl_element = mt.terminal.ufl_element()
            max_derivs[ufl_element] = max(mt.local_derivatives, max_derivs[ufl_element])

    # Collect tabulations for all components and derivatives
    tabulation_manager = TabulationManager(params.entity_points)
    for ufl_element, max_deriv in max_derivs.items():
        if ufl_element.family() != 'Real':
            tabulation_manager.tabulate(ufl_element, max_deriv)

    if interior_facet:
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(params.argument_indices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), expression))
    else:
        expressions = [expression]

    # Translate UFL to GEM, lowering finite element specific nodes
    translator = Translator(tabulation_manager, params)
    return map_expr_dags(translator, expressions)
