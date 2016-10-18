from __future__ import absolute_import, print_function, division
from six.moves import map, range

import collections
import itertools

import numpy
from singledispatch import singledispatch

from ufl.corealg.map_dag import map_expr_dag, map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, Coefficient, CellVolume, FacetArea,
                         GeometricQuantity, QuadratureWeight)

import gem
from gem.utils import cached_property

from finat.quadrature import make_quadrature

from tsfc import ufl2gem, geometric
from tsfc.finatinterface import create_element, as_fiat_cell
from tsfc.kernel_interface import ProxyKernelInterface
from tsfc.modified_terminals import analyse_modified_terminal
from tsfc.parameters import PARAMETERS
from tsfc.ufl_utils import ModifiedTerminalMixin, PickRestriction, simplify_abs


class Context(ProxyKernelInterface):
    keywords = ('ufl_cell',
                'fiat_cell',
                'integration_dim',
                'entity_ids',
                'quadrature_degree',
                'quadrature_rule',
                'points',
                'weights',
                'precision',
                'point_index',
                'argument_indices',
                'cellvolume',
                'facetarea',
                'index_cache')

    def __init__(self, interface, **kwargs):
        ProxyKernelInterface.__init__(self, interface)

        invalid_keywords = set(kwargs.keys()) - set(Context.keywords)
        if invalid_keywords:
            raise ValueError("unexpected keyword argument '{0}'".format(invalid_keywords.pop()))
        self.__dict__.update(kwargs)

    @cached_property
    def fiat_cell(self):
        return as_fiat_cell(self.ufl_cell)

    @cached_property
    def integration_dim(self):
        return self.fiat_cell.get_dimension()

    entity_ids = [0]

    @cached_property
    def quadrature_rule(self):
        integration_cell = self.fiat_cell.construct_subelement(self.integration_dim)
        return make_quadrature(integration_cell, self.quadrature_degree)

    @cached_property
    def points(self):
        return self.quadrature_rule.points

    @cached_property
    def weights(self):
        return self.quadrature_rule.weights

    precision = PARAMETERS["precision"]

    @cached_property
    def epsilon(self):
        # Rounding tolerance mimicking FFC
        return 10.0 * eval("1e-%d" % self.precision)

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
            f = self.facet_number(restriction)
            return gem.select_expression(list(map(callback, opts)), f)

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

    def __init__(self, context):
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)

        self.context = context

    def modified_terminal(self, o):
        """Overrides the modified terminal handler from
        :class:`ModifiedTerminalMixin`."""
        mt = analyse_modified_terminal(o)
        return translate(mt.terminal, mt, self.context)


@singledispatch
def translate(terminal, mt, ctx):
    """Translates modified terminals into GEM.

    :arg terminal: terminal, for dispatching
    :arg mt: analysed modified terminal
    :arg ctx: translator context
    :returns: GEM translation of the modified terminal
    """
    raise AssertionError("Cannot handle terminal type: %s" % type(terminal))


@translate.register(QuadratureWeight)
def translate_quadratureweight(terminal, mt, ctx):
    return ctx.quadrature_rule.weight_expression


@translate.register(GeometricQuantity)
def translate_geometricquantity(terminal, mt, ctx):
    return geometric.translate(terminal, mt, ctx)


@translate.register(CellVolume)
def translate_cellvolume(terminal, mt, ctx):
    return ctx.cellvolume(mt.restriction)


@translate.register(FacetArea)
def translate_facetarea(terminal, mt, ctx):
    return ctx.facetarea()


@translate.register(Argument)
def translate_argument(terminal, mt, ctx):
    element = create_element(terminal.ufl_element())

    def callback(entity_id):
        return element.basis_evaluation(ctx.quadrature_rule.point_set,
                                        derivative=mt.local_derivatives,
                                        entity=(ctx.integration_dim, entity_id))
    M = ctx.entity_selector(callback, mt.restriction)
    vi = tuple(gem.Index(extent=d) for d in mt.expr.ufl_shape)
    argument_index = ctx.argument_indices[terminal.number()]
    result = gem.Indexed(M, ctx.point_index + argument_index + vi)
    if vi:
        return gem.ComponentTensor(result, vi)
    else:
        return result


@translate.register(Coefficient)
def translate_coefficient(terminal, mt, ctx):
    vec = ctx.coefficient(terminal, mt.restriction)

    if terminal.ufl_element().family() == 'Real':
        assert mt.local_derivatives == 0
        return vec

    element = create_element(terminal.ufl_element())

    def callback(entity_id):
        return element.basis_evaluation(ctx.quadrature_rule.point_set,
                                        derivative=mt.local_derivatives,
                                        entity=(ctx.integration_dim, entity_id))
    M = ctx.entity_selector(callback, mt.restriction)

    alpha = element.get_indices()
    vi = tuple(gem.Index(extent=d) for d in mt.expr.ufl_shape)
    result = gem.Product(gem.Indexed(M, ctx.point_index + alpha + vi),
                         gem.Indexed(vec, alpha))
    for i in alpha:
        result = gem.IndexSum(result, i)
    if vi:
        return gem.ComponentTensor(result, vi)
    else:
        return result


def compile_ufl(expression, interior_facet=False, **kwargs):
    context = Context(**kwargs)

    # Abs-simplification
    expression = simplify_abs(expression)
    if interior_facet:
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(context.argument_indices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), expression))
    else:
        expressions = [expression]

    # Translate UFL to GEM, lowering finite element specific nodes
    translator = Translator(context)
    return map_expr_dags(translator, expressions)
