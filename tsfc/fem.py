from __future__ import absolute_import

import itertools

import numpy
from singledispatch import singledispatch

from ufl.corealg.map_dag import map_expr_dag, map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import Argument, Coefficient, GeometricQuantity, QuadratureWeight

import gem

from tsfc.constants import PRECISION
from tsfc.finatinterface import create_element, as_fiat_cell
from tsfc.modified_terminals import analyse_modified_terminal
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


class Translator(MultiFunction, ModifiedTerminalMixin, ufl2gem.Mixin):
    """Contains all the context necessary to translate UFL into GEM."""

    def __init__(self, integral_type, facet_manager, quad_rule, quadrature_index,
                 argument_indices, coefficient_mapper, index_cache):
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)
        self.integral_type = integral_type
        self.quad_rule = quad_rule
        self.weights = gem.Literal(quad_rule.weights)
        self.quadrature_index = quadrature_index
        self.argument_indices = argument_indices
        self.coefficient_mapper = coefficient_mapper
        self.index_cache = index_cache
        self.facet_manager = facet_manager
        self.select_facet = facet_manager.select_facet

        if self.integral_type.startswith("interior_facet"):
            self.cell_orientations = gem.Variable("cell_orientations", (2, 1))
        else:
            self.cell_orientations = gem.Variable("cell_orientations", (1, 1))

    def modified_terminal(self, o):
        """Overrides the modified terminal handler from
        :class:`ModifiedTerminalMixin`."""
        mt = analyse_modified_terminal(o)
        return translate(mt.terminal, mt, self)


@singledispatch
def translate(terminal, mt, params):
    """Translates modified terminals into GEM.

    :arg terminal: terminal, for dispatching
    :arg mt: analysed modified terminal
    :arg params: translator context
    :returns: GEM translation of the modified terminal
    """
    raise AssertionError("Cannot handle terminal type: %s" % type(terminal))


@translate.register(QuadratureWeight)  # noqa: Not actually redefinition
def _(terminal, mt, params):
    return gem.Indexed(params.weights, (params.quadrature_index,))


@translate.register(GeometricQuantity)  # noqa: Not actually redefinition
def _(terminal, mt, params):
    return geometric.translate(terminal, mt, params)


@translate.register(Argument)  # noqa: Not actually redefinition
def _(terminal, mt, params):
    argument_index = params.argument_indices[terminal.number()]

    element = create_element(terminal.ufl_element())
    M = element.basis_evaluation(params.quad_rule, derivative=mt.local_derivatives)
    vi = tuple(gem.Index(extent=d) for d in mt.expr.ufl_shape)
    result = gem.Indexed(M, (params.quadrature_index,) + argument_index + vi)
    if vi:
        return gem.ComponentTensor(result, vi)
    else:
        return result


@translate.register(Coefficient)  # noqa: Not actually redefinition
def _(terminal, mt, params):
    kernel_arg = params.coefficient_mapper(terminal)

    if terminal.ufl_element().family() == 'Real':
        assert mt.local_derivatives == 0
        return kernel_arg

    ka = gem.partial_indexed(kernel_arg, {None: (), '+': (0,), '-': (1,)}[mt.restriction])

    element = create_element(terminal.ufl_element())
    M = element.basis_evaluation(params.quad_rule, derivative=mt.local_derivatives)
    alpha = element.get_indices()
    vi = tuple(gem.Index(extent=d) for d in mt.expr.ufl_shape)
    result = gem.Product(gem.Indexed(M, (params.quadrature_index,) + alpha + vi),
                         gem.Indexed(ka, alpha))
    for i in alpha:
        result = gem.IndexSum(result, i)
    if vi:
        return gem.ComponentTensor(result, vi)
    else:
        return result


def process(integral_type, cell, quad_rule, quadrature_index,
            argument_indices, integrand, coefficient_mapper, index_cache):
    # Abs-simplification
    integrand = simplify_abs(integrand)

    if integral_type.startswith("interior_facet"):
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(argument_indices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), integrand))
    else:
        expressions = [integrand]

    # Translate UFL to Einstein's notation,
    # lowering finite element specific nodes
    facet_manager = FacetManager(integral_type, cell)
    translator = Translator(integral_type, facet_manager, quad_rule,
                            quadrature_index, argument_indices,
                            coefficient_mapper, index_cache)
    return map_expr_dags(translator, expressions)
