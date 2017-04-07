"""Functions to translate UFL finite element objects and reference
geometric quantities into GEM expressions."""

from __future__ import absolute_import, print_function, division
from six import iterkeys, iteritems, itervalues
from six.moves import map, range, zip

import collections
import itertools

import numpy
from singledispatch import singledispatch

from ufl.corealg.map_dag import map_expr_dag, map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, CellCoordinate, CellEdgeVectors,
                         CellFacetJacobian, CellOrientation,
                         CellVolume, Coefficient, FacetArea,
                         FacetCoordinate, GeometricQuantity,
                         QuadratureWeight, ReferenceCellVolume,
                         ReferenceFacetVolume, ReferenceNormal)

from FIAT.reference_element import make_affine_mapping

import gem
from gem.node import traversal
from gem.optimise import ffc_rounding
from gem.utils import cached_property

from finat.quadrature import make_quadrature

from tsfc import ufl2gem
from tsfc.finatinterface import create_element, as_fiat_cell
from tsfc.kernel_interface import ProxyKernelInterface
from tsfc.modified_terminals import analyse_modified_terminal
from tsfc.parameters import NUMPY_TYPE, PARAMETERS
from tsfc.ufl_utils import ModifiedTerminalMixin, PickRestriction, simplify_abs


class Context(ProxyKernelInterface):
    keywords = ('ufl_cell',
                'fiat_cell',
                'integration_dim',
                'entity_ids',
                'quadrature_degree',
                'quadrature_rule',
                'point_set',
                'weight_expr',
                'precision',
                'argument_multiindices',
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
    def point_set(self):
        return self.quadrature_rule.point_set

    @cached_property
    def weight_expr(self):
        return self.quadrature_rule.weight_expression

    precision = PARAMETERS["precision"]

    @cached_property
    def epsilon(self):
        # Rounding tolerance mimicking FFC
        return 10.0 * eval("1e-%d" % self.precision)

    def entity_selector(self, callback, restriction):
        """Selects code for the correct entity at run-time.  Callback
        generates code for a specified entity.

        This function passes ``callback`` the entity number.

        :arg callback: A function to be called with an entity number
                       that generates code for that entity.
        :arg restriction: Restriction of the modified terminal, used
                          for entity selection.
        """
        if len(self.entity_ids) == 1:
            return callback(self.entity_ids[0])
        else:
            f = self.entity_number(restriction)
            return gem.select_expression(list(map(callback, self.entity_ids)), f)

    argument_multiindices = ()

    @cached_property
    def index_cache(self):
        return {}


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
    return ctx.weight_expr


@translate.register(GeometricQuantity)
def translate_geometricquantity(terminal, mt, ctx):
    raise NotImplementedError("Cannot handle geometric quantity type: %s" % type(terminal))


@translate.register(CellOrientation)
def translate_cell_orientation(terminal, mt, ctx):
    return ctx.cell_orientation(mt.restriction)


@translate.register(ReferenceCellVolume)
def translate_reference_cell_volume(terminal, mt, ctx):
    return gem.Literal(ctx.fiat_cell.volume())


@translate.register(ReferenceFacetVolume)
def translate_reference_facet_volume(terminal, mt, ctx):
    # FIXME: simplex only code path
    dim = ctx.fiat_cell.get_spatial_dimension()
    facet_cell = ctx.fiat_cell.construct_subelement(dim - 1)
    return gem.Literal(facet_cell.volume())


@translate.register(CellFacetJacobian)
def translate_cell_facet_jacobian(terminal, mt, ctx):
    cell = ctx.fiat_cell
    facet_dim = ctx.integration_dim
    assert facet_dim != cell.get_dimension()

    def callback(entity_id):
        return gem.Literal(make_cell_facet_jacobian(cell, facet_dim, entity_id))
    return ctx.entity_selector(callback, mt.restriction)


def make_cell_facet_jacobian(cell, facet_dim, facet_i):
    facet_cell = cell.construct_subelement(facet_dim)
    xs = facet_cell.get_vertices()
    ys = cell.get_vertices_of_subcomplex(cell.get_topology()[facet_dim][facet_i])

    # Use first 'dim' points to make an affine mapping
    dim = cell.get_spatial_dimension()
    A, b = make_affine_mapping(xs[:dim], ys[:dim])

    for x, y in zip(xs[dim:], ys[dim:]):
        # The rest of the points are checked to make sure the
        # mapping really *is* affine.
        assert numpy.allclose(y, A.dot(x) + b)

    return A


@translate.register(ReferenceNormal)
def translate_reference_normal(terminal, mt, ctx):
    def callback(facet_i):
        n = ctx.fiat_cell.compute_reference_normal(ctx.integration_dim, facet_i)
        return gem.Literal(n)
    return ctx.entity_selector(callback, mt.restriction)


@translate.register(CellEdgeVectors)
def translate_cell_edge_vectors(terminal, mt, ctx):
    from FIAT.reference_element import TensorProductCell as fiat_TensorProductCell
    fiat_cell = ctx.fiat_cell
    if isinstance(fiat_cell, fiat_TensorProductCell):
        raise NotImplementedError("CellEdgeVectors not implemented on TensorProductElements yet")

    nedges = len(fiat_cell.get_topology()[1])
    vecs = numpy.vstack(map(fiat_cell.compute_edge_tangent, range(nedges))).astype(NUMPY_TYPE)
    assert vecs.shape == terminal.ufl_shape
    return gem.Literal(vecs)


@translate.register(CellCoordinate)
def translate_cell_coordinate(terminal, mt, ctx):
    ps = ctx.point_set
    if ctx.integration_dim == ctx.fiat_cell.get_dimension():
        return ps.expression

    # This destroys the structure of the quadrature points, but since
    # this code path is only used to implement CellCoordinate in facet
    # integrals, hopefully it does not matter much.
    point_shape = tuple(index.extent for index in ps.indices)

    def callback(entity_id):
        t = ctx.fiat_cell.get_entity_transform(ctx.integration_dim, entity_id)
        data = numpy.asarray(list(map(t, ps.points)))
        return gem.Literal(data.reshape(point_shape + data.shape[1:]))

    return gem.partial_indexed(ctx.entity_selector(callback, mt.restriction),
                               ps.indices)


@translate.register(FacetCoordinate)
def translate_facet_coordinate(terminal, mt, ctx):
    assert ctx.integration_dim != ctx.fiat_cell.get_dimension()
    return ctx.point_set.expression


@translate.register(CellVolume)
def translate_cellvolume(terminal, mt, ctx):
    return ctx.cellvolume(mt.restriction)


@translate.register(FacetArea)
def translate_facetarea(terminal, mt, ctx):
    return ctx.facetarea()


def fiat_to_ufl(fiat_dict, order):
    # All derivative multiindices must be of the same dimension.
    dimension, = list(set(len(alpha) for alpha in iterkeys(fiat_dict)))

    # All derivative tables must have the same shape.
    shape, = list(set(table.shape for table in itervalues(fiat_dict)))
    sigma = tuple(gem.Index(extent=extent) for extent in shape)

    # Convert from FIAT to UFL format
    eye = numpy.eye(dimension, dtype=int)
    tensor = numpy.empty((dimension,) * order, dtype=object)
    for multiindex in numpy.ndindex(tensor.shape):
        alpha = tuple(eye[multiindex, :].sum(axis=0))
        tensor[multiindex] = gem.Indexed(fiat_dict[alpha], sigma)
    delta = tuple(gem.Index(extent=dimension) for _ in range(order))
    if order > 0:
        tensor = gem.Indexed(gem.ListTensor(tensor), delta)
    else:
        tensor = tensor[()]
    return gem.ComponentTensor(tensor, sigma + delta)


@translate.register(Argument)
def translate_argument(terminal, mt, ctx):
    argument_multiindex = ctx.argument_multiindices[terminal.number()]
    sigma = tuple(gem.Index(extent=d) for d in mt.expr.ufl_shape)
    element = create_element(terminal.ufl_element())

    def callback(entity_id):
        finat_dict = element.basis_evaluation(mt.local_derivatives,
                                              ctx.point_set,
                                              (ctx.integration_dim, entity_id))
        # Filter out irrelevant derivatives
        filtered_dict = {alpha: table
                         for alpha, table in iteritems(finat_dict)
                         if sum(alpha) == mt.local_derivatives}

        # Change from FIAT to UFL arrangement
        square = fiat_to_ufl(filtered_dict, mt.local_derivatives)

        # A numerical hack that FFC used to apply on FIAT tables still
        # lives on after ditching FFC and switching to FInAT.
        return ffc_rounding(square, ctx.epsilon)
    table = ctx.entity_selector(callback, mt.restriction)
    return gem.ComponentTensor(gem.Indexed(table, argument_multiindex + sigma), sigma)


@translate.register(Coefficient)
def translate_coefficient(terminal, mt, ctx):
    vec = ctx.coefficient(terminal, mt.restriction)

    if terminal.ufl_element().family() == 'Real':
        assert mt.local_derivatives == 0
        return vec

    element = create_element(terminal.ufl_element())

    # Collect FInAT tabulation for all entities
    per_derivative = collections.defaultdict(list)
    for entity_id in ctx.entity_ids:
        finat_dict = element.basis_evaluation(mt.local_derivatives,
                                              ctx.point_set,
                                              (ctx.integration_dim, entity_id))
        for alpha, table in iteritems(finat_dict):
            # Filter out irrelevant derivatives
            if sum(alpha) == mt.local_derivatives:
                # A numerical hack that FFC used to apply on FIAT
                # tables still lives on after ditching FFC and
                # switching to FInAT.
                table = ffc_rounding(table, ctx.epsilon)
                per_derivative[alpha].append(table)

    # Merge entity tabulations for each derivative
    if len(ctx.entity_ids) == 1:
        def take_singleton(xs):
            x, = xs  # asserts singleton
            return x
        per_derivative = {alpha: take_singleton(tables)
                          for alpha, tables in iteritems(per_derivative)}
    else:
        f = ctx.entity_number(mt.restriction)
        per_derivative = {alpha: gem.select_expression(tables, f)
                          for alpha, tables in iteritems(per_derivative)}

    # Coefficient evaluation
    ctx.index_cache.setdefault(terminal.ufl_element(), element.get_indices())
    beta = ctx.index_cache[terminal.ufl_element()]
    zeta = element.get_value_indices()
    value_dict = {}
    for alpha, table in iteritems(per_derivative):
        value = gem.IndexSum(gem.Product(gem.Indexed(table, beta + zeta),
                                         gem.Indexed(vec, beta)),
                             beta)
        optimised_value = gem.optimise.contraction(value)
        value_dict[alpha] = gem.ComponentTensor(optimised_value, zeta)

    # Change from FIAT to UFL arrangement
    result = fiat_to_ufl(value_dict, mt.local_derivatives)
    assert result.shape == mt.expr.ufl_shape
    assert set(result.free_indices) <= set(ctx.point_set.indices)

    # Detect Jacobian of affine cells
    if not result.free_indices and all(numpy.count_nonzero(node.array) <= 2
                                       for node in traversal((result,))
                                       if isinstance(node, gem.Literal)):
        result = gem.optimise.aggressive_unroll(result)
    return result


def compile_ufl(expression, interior_facet=False, point_sum=False, **kwargs):
    context = Context(**kwargs)

    # Abs-simplification
    expression = simplify_abs(expression)
    if interior_facet:
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(context.argument_multiindices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), expression))
    else:
        expressions = [expression]

    # Translate UFL to GEM, lowering finite element specific nodes
    translator = Translator(context)
    result = map_expr_dags(translator, expressions)
    if point_sum:
        result = [gem.index_sum(expr, context.point_set.indices) for expr in result]
    return result
