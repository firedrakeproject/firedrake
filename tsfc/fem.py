from __future__ import absolute_import

import collections
import itertools

import numpy
from singledispatch import singledispatch

from ufl.corealg.map_dag import map_expr_dag, map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, Coefficient, CellVolume, FacetArea,
                         GeometricQuantity, QuadratureWeight)

import gem

from finat.quadrature import QuadratureRule, CollapsedGaussJacobiQuadrature

from tsfc.constants import PRECISION
from tsfc.fiatinterface import create_quadrature
from tsfc.finatinterface import create_element, as_fiat_cell
from tsfc.modified_terminals import analyse_modified_terminal
from tsfc import ufl2gem
from tsfc import geometric
from tsfc.ufl_utils import ModifiedTerminalMixin, PickRestriction, simplify_abs


# FFC uses one less digits for rounding than for printing
epsilon = eval("1e-%d" % (PRECISION - 1))


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
        quad_rule = create_quadrature(integration_cell, self.quadrature_degree)
        quad_rule = QuadratureRule(integration_cell, quad_rule.get_points(), quad_rule.get_weights())
        quad_rule.__class__ = CollapsedGaussJacobiQuadrature
        return quad_rule

    @cached_property
    def points(self):
        return self.quadrature_rule.points

    @cached_property
    def weights(self):
        return self.quadrature_rule.weights

    @cached_property
    def entity_points(self):
        """An array of points in cell coordinates for each entity,
        i.e. a list of arrays of points."""
        result = []
        for entity_id in self.entity_ids:
            t = self.fiat_cell.get_entity_transform(self.integration_dim, entity_id)
            result.append(numpy.asarray(map(t, self.points)))
        return result

    def _selector(self, callback, opts, restriction):
        """Helper function for selecting code for the correct entity
        at run-time."""
        if len(opts) == 1:
            return callback(opts[0])
        else:
            f = self.facet_number(restriction)
            return gem.select_expression(map(callback, opts), f)

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
        return self._selector(callback, range(len(self.entity_ids)), restriction)

    argument_indices = ()

    @cached_property
    def index_cache(self):
        return collections.defaultdict(gem.Index)


class Translator(MultiFunction, ModifiedTerminalMixin, ufl2gem.Mixin):
    """Contains all the context necessary to translate UFL into GEM."""

    def __init__(self, parameters):
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)

        self.parameters = parameters

    def modified_terminal(self, o):
        """Overrides the modified terminal handler from
        :class:`ModifiedTerminalMixin`."""
        mt = analyse_modified_terminal(o)
        return translate(mt.terminal, mt, self.parameters)


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
    element = create_element(terminal.ufl_element())

    def callback(entity_index):
        quad_rule = QuadratureRule(params.fiat_cell, params.entity_points[entity_index], params.weights)
        quad_rule.__class__ = CollapsedGaussJacobiQuadrature
        return element.basis_evaluation(quad_rule, derivative=mt.local_derivatives)
    M = params.index_selector(callback, mt.restriction)
    vi = tuple(gem.Index(extent=d) for d in mt.expr.ufl_shape)
    argument_index = params.argument_indices[terminal.number()]
    result = gem.Indexed(M, (params.point_index,) + argument_index + vi)
    if vi:
        return gem.ComponentTensor(result, vi)
    else:
        return result


@translate.register(Coefficient)
def translate_coefficient(terminal, mt, params):
    vec = params.coefficient(terminal, mt.restriction)

    if terminal.ufl_element().family() == 'Real':
        assert mt.local_derivatives == 0
        return vec

    element = create_element(terminal.ufl_element())

    def callback(entity_index):
        quad_rule = QuadratureRule(params.fiat_cell, params.entity_points[entity_index], params.weights)
        quad_rule.__class__ = CollapsedGaussJacobiQuadrature
        return element.basis_evaluation(quad_rule, derivative=mt.local_derivatives)
    M = params.index_selector(callback, mt.restriction)

    alpha = element.get_indices()
    vi = tuple(gem.Index(extent=d) for d in mt.expr.ufl_shape)
    result = gem.Product(gem.Indexed(M, (params.point_index,) + alpha + vi),
                         gem.Indexed(vec, alpha))
    for i in alpha:
        result = gem.IndexSum(result, i)
    if vi:
        return gem.ComponentTensor(result, vi)
    else:
        return result


def compile_ufl(expression, interior_facet=False, **kwargs):
    params = Parameters(**kwargs)

    # Abs-simplification
    expression = simplify_abs(expression)

    if interior_facet:
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(params.argument_indices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), expression))
    else:
        expressions = [expression]

    # Translate UFL to GEM, lowering finite element specific nodes
    translator = Translator(params)
    return map_expr_dags(translator, expressions)
