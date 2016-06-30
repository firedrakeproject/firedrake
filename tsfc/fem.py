from __future__ import absolute_import

import collections
import itertools

import numpy
from singledispatch import singledispatch

from ufl.corealg.map_dag import map_expr_dag, map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, Coefficient, CellVolume,
                         ConstantValue, FacetArea, GeometricQuantity,
                         QuadratureWeight)

import gem

from tsfc.constants import PRECISION
from tsfc.finatinterface import create_element, as_fiat_cell
from tsfc.modified_terminals import analyse_modified_terminal
from tsfc.quadrature import create_quadrature
from tsfc import ufl2gem
from tsfc import geometric
from tsfc.ufl_utils import ModifiedTerminalMixin, PickRestriction, simplify_abs


# FFC uses one less digits for rounding than for printing
epsilon = eval("1e-%d" % (PRECISION - 1))


class FacetManager(object):
    """Collection of utilities for facet integrals."""

    def __init__(self, integral_type, ufl_cell):
        """Constructs a FacetManager.

        :arg integral_type: integral type
        :arg ufl_cell: UFL cell
        """
        self.integral_type = integral_type
        self.ufl_cell = ufl_cell

        if integral_type in ['exterior_facet', 'exterior_facet_vert']:
            self.facet = {None: gem.VariableIndex(gem.Indexed(gem.Variable('facet', (1,)), (0,)))}
        elif integral_type in ['interior_facet', 'interior_facet_vert']:
            self.facet = {'+': gem.VariableIndex(gem.Indexed(gem.Variable('facet', (2,)), (0,))),
                          '-': gem.VariableIndex(gem.Indexed(gem.Variable('facet', (2,)), (1,)))}
        elif integral_type == 'exterior_facet_bottom':
            self.facet = {None: 0}
        elif integral_type == 'exterior_facet_top':
            self.facet = {None: 1}
        elif integral_type == 'interior_facet_horiz':
            self.facet = {'+': 1, '-': 0}
        else:
            self.facet = None

    def facet_transform(self, points):
        """Generator function that transforms points in integration cell
        coordinates to cell coordinates for each facet.

        :arg points: points in integration cell coordinates
        """
        if self.integral_type == 'cell':
            yield points

        elif self.integral_type in ['exterior_facet', 'interior_facet']:
            for entity in range(self.ufl_cell.num_facets()):
                t = as_fiat_cell(self.ufl_cell).get_facet_transform(entity)
                yield numpy.asarray(map(t, points))

        elif self.integral_type in ['exterior_facet_bottom', 'exterior_facet_top', 'interior_facet_horiz']:
            for entity in range(2):  # top and bottom
                t = as_fiat_cell(self.ufl_cell).get_horiz_facet_transform(entity)
                yield numpy.asarray(map(t, points))

        elif self.integral_type in ['exterior_facet_vert', 'interior_facet_vert']:
            for entity in range(self.ufl_cell.sub_cells()[0].num_facets()):  # "base cell" facets
                t = as_fiat_cell(self.ufl_cell).get_vert_facet_transform(entity)
                yield numpy.asarray(map(t, points))

        else:
            raise NotImplementedError("integral type %s not supported" % self.integral_type)

    def select_facet(self, tensor, restriction):
        """Applies facet selection on a GEM tensor if necessary.

        :arg tensor: GEM tensor
        :arg restriction: restriction on the modified terminal
        :returns: another GEM tensor
        """
        if self.integral_type == 'cell':
            return tensor
        else:
            f = self.facet[restriction]
            return gem.partial_indexed(tensor, (f,))


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
    keywords = ('integral_type',
                'cell',
                'quadrature_degree',
                'quadrature_rule',
                'points',
                'weights',
                'point_index',
                'argument_indices',
                'coefficient_mapper',
                'cellvolume',
                'facetarea',
                'index_cache')

    def __init__(self, **kwargs):
        invalid_keywords = set(kwargs.keys()) - set(Parameters.keywords)
        if invalid_keywords:
            raise ValueError("unexpected keyword argument '{0}'".format(invalid_keywords.pop()))
        self.__dict__.update(kwargs)

    # Defaults
    integral_type = 'cell'

    @cached_property
    def quadrature_rule(self):
        from finat.quadrature import QuadratureRule, CollapsedGaussJacobiQuadrature
        quad_rule = create_quadrature(self.cell,
                                      self.integral_type,
                                      self.quadrature_degree)
        quad_rule = QuadratureRule(self.cell, quad_rule.points, quad_rule.weights)
        quad_rule.__class__ = CollapsedGaussJacobiQuadrature
        return quad_rule

    @cached_property
    def points(self):
        return self.quadrature_rule.points

    @cached_property
    def weights(self):
        return self.quadrature_rule.weights

    argument_indices = ()

    @cached_property
    def index_cache(self):
        return collections.defaultdict(gem.Index)


class Translator(MultiFunction, ModifiedTerminalMixin, ufl2gem.Mixin):
    """Contains all the context necessary to translate UFL into GEM."""

    def __init__(self, facet_manager, parameters):
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)

        if parameters.integral_type.startswith("interior_facet"):
            parameters.cell_orientations = gem.Variable("cell_orientations", (2, 1))
        else:
            parameters.cell_orientations = gem.Variable("cell_orientations", (1, 1))

        parameters.facet_manager = facet_manager
        parameters.select_facet = facet_manager.select_facet
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
    argument_index = params.argument_indices[terminal.number()]

    element = create_element(terminal.ufl_element())
    if params.integral_type == 'cell':
        M = element.basis_evaluation(params.quadrature_rule, derivative=mt.local_derivatives)
    else:
        from finat.quadrature import QuadratureRule, CollapsedGaussJacobiQuadrature
        Ms = []
        for points in params.facet_manager.facet_transform(params.points):
            quad_rule = QuadratureRule(params.facet_manager.ufl_cell, points, params.weights)
            quad_rule.__class__ = CollapsedGaussJacobiQuadrature
            Ms.append(element.basis_evaluation(quad_rule, derivative=mt.local_derivatives))
        M = params.facet_manager.select_facet(gem.ListTensor(Ms), mt.restriction)
    vi = tuple(gem.Index(extent=d) for d in mt.expr.ufl_shape)
    result = gem.Indexed(M, (params.point_index,) + argument_index + vi)
    if vi:
        return gem.ComponentTensor(result, vi)
    else:
        return result


@translate.register(Coefficient)
def translate_coefficient(terminal, mt, params):
    kernel_arg = params.coefficient_mapper(terminal)

    if terminal.ufl_element().family() == 'Real':
        assert mt.local_derivatives == 0
        return kernel_arg

    ka = gem.partial_indexed(kernel_arg, {None: (), '+': (0,), '-': (1,)}[mt.restriction])

    element = create_element(terminal.ufl_element())
    if params.integral_type == 'cell':
        M = element.basis_evaluation(params.quadrature_rule, derivative=mt.local_derivatives)
    else:
        from finat.quadrature import QuadratureRule, CollapsedGaussJacobiQuadrature
        Ms = []
        for points in params.facet_manager.facet_transform(params.points):
            quad_rule = QuadratureRule(params.facet_manager.ufl_cell, points, params.weights)
            quad_rule.__class__ = CollapsedGaussJacobiQuadrature
            Ms.append(element.basis_evaluation(quad_rule, derivative=mt.local_derivatives))
        M = params.facet_manager.select_facet(gem.ListTensor(Ms), mt.restriction)

    alpha = element.get_indices()
    vi = tuple(gem.Index(extent=d) for d in mt.expr.ufl_shape)
    result = gem.Product(gem.Indexed(M, (params.point_index,) + alpha + vi),
                         gem.Indexed(ka, alpha))
    for i in alpha:
        result = gem.IndexSum(result, i)
    if vi:
        return gem.ComponentTensor(result, vi)
    else:
        return result


@translate.register(ConstantValue)
def _translate_constantvalue(terminal, mt, params):
    # Literal in a modified terminal
    # Terminal modifiers have no effect, just translate the terminal.
    return params(terminal)


def compile_ufl(expression, **kwargs):
    params = Parameters(**kwargs)

    # Abs-simplification
    expression = simplify_abs(expression)

    if params.integral_type.startswith("interior_facet"):
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(params.argument_indices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), expression))
    else:
        expressions = [expression]

    # Translate UFL to GEM, lowering finite element specific nodes
    facet_manager = FacetManager(params.integral_type, params.cell)
    translator = Translator(facet_manager, params)
    return map_expr_dags(translator, expressions)
