"""Functions to translate UFL finite element objects and reference
geometric quantities into GEM expressions."""

import collections
import itertools
from functools import singledispatch

import numpy

import ufl
from ufl.corealg.map_dag import map_expr_dag, map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, CellCoordinate, CellEdgeVectors,
                         CellFacetJacobian, CellOrientation,
                         CellOrigin, CellVertices, CellVolume,
                         Coefficient, FacetArea, FacetCoordinate,
                         GeometricQuantity, Jacobian, JacobianDeterminant,
                         NegativeRestricted, QuadratureWeight,
                         PositiveRestricted, ReferenceCellVolume,
                         ReferenceCellEdgeVectors,
                         ReferenceFacetVolume, ReferenceNormal,
                         SpatialCoordinate)


from FIAT.reference_element import make_affine_mapping
from FIAT.reference_element import UFCSimplex

import gem
from gem.node import traversal
from gem.optimise import ffc_rounding
from gem.unconcatenate import unconcatenate
from gem.utils import cached_property

from finat.physically_mapped import PhysicalGeometry, NeedsCoordinateMappingElement
from finat.point_set import PointSet, PointSingleton
from finat.quadrature import make_quadrature

from tsfc import ufl2gem
from tsfc.finatinterface import as_fiat_cell, create_element
from tsfc.kernel_interface import ProxyKernelInterface
from tsfc.modified_terminals import (analyse_modified_terminal,
                                     construct_modified_terminal)
from tsfc.parameters import is_complex
from tsfc.ufl_utils import (ModifiedTerminalMixin, PickRestriction,
                            entity_avg, one_times, simplify_abs,
                            preprocess_expression)


class ContextBase(ProxyKernelInterface):
    """Common UFL -> GEM translation context."""

    keywords = ('ufl_cell',
                'fiat_cell',
                'integral_type',
                'integration_dim',
                'entity_ids',
                'argument_multiindices',
                'facetarea',
                'index_cache',
                'scalar_type')

    def __init__(self, interface, **kwargs):
        ProxyKernelInterface.__init__(self, interface)

        invalid_keywords = set(kwargs.keys()) - set(self.keywords)
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
    def epsilon(self):
        return numpy.finfo(self.scalar_type).resolution

    @cached_property
    def complex_mode(self):
        return is_complex(self.scalar_type)

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

    @cached_property
    def translator(self):
        # NOTE: reference cycle!
        return Translator(self)


class CoordinateMapping(PhysicalGeometry):
    """Callback class that provides physical geometry to FInAT elements.

    Required for elements whose basis transformation requires physical
    geometry such as Argyris and Hermite.

    :arg mt: The modified terminal whose element will be tabulated.
    :arg interface: The interface carrying information (generally a
        :class:`PointSetContext`).
    """
    def __init__(self, mt, interface):
        super().__init__()
        self.mt = mt
        self.interface = interface

    def preprocess(self, expr, context):
        """Preprocess a UFL expression for translation.

        :arg expr: A UFL expression
        :arg context: The translation context.
        :returns: A new UFL expression
        """
        ifacet = self.interface.integral_type.startswith("interior_facet")
        return preprocess_expression(expr, complex_mode=context.complex_mode,
                                     do_apply_restrictions=ifacet)

    @property
    def config(self):
        config = {name: getattr(self.interface, name)
                  for name in ["ufl_cell", "index_cache", "scalar_type"]}
        config["interface"] = self.interface
        return config

    def cell_size(self):
        return self.interface.cell_size(self.mt.restriction)

    def jacobian_at(self, point):
        ps = PointSingleton(point)
        expr = Jacobian(self.mt.terminal.ufl_domain())
        assert ps.expression.shape == (expr.ufl_domain().topological_dimension(), )
        if self.mt.restriction == '+':
            expr = PositiveRestricted(expr)
        elif self.mt.restriction == '-':
            expr = NegativeRestricted(expr)
        config = {"point_set": PointSingleton(point)}
        config.update(self.config)
        context = PointSetContext(**config)
        expr = self.preprocess(expr, context)
        return map_expr_dag(context.translator, expr)

    def detJ_at(self, point):
        expr = JacobianDeterminant(self.mt.terminal.ufl_domain())
        if self.mt.restriction == '+':
            expr = PositiveRestricted(expr)
        elif self.mt.restriction == '-':
            expr = NegativeRestricted(expr)
        config = {"point_set": PointSingleton(point)}
        config.update(self.config)
        context = PointSetContext(**config)
        expr = self.preprocess(expr, context)
        return map_expr_dag(context.translator, expr)

    def reference_normals(self):
        if not (isinstance(self.interface.fiat_cell, UFCSimplex) and
                self.interface.fiat_cell.get_spatial_dimension() == 2):
            raise NotImplementedError("Only works for triangles for now")
        return gem.Literal(numpy.asarray([self.interface.fiat_cell.compute_normal(i) for i in range(3)]))

    def reference_edge_tangents(self):
        return gem.Literal(numpy.asarray([self.interface.fiat_cell.compute_edge_tangent(i) for i in range(3)]))

    def physical_tangents(self):
        if not (isinstance(self.interface.fiat_cell, UFCSimplex) and
                self.interface.fiat_cell.get_spatial_dimension() == 2):
            raise NotImplementedError("Only works for triangles for now")

        rts = [self.interface.fiat_cell.compute_tangents(1, f)[0] for f in range(3)]
        jac = self.jacobian_at([1/3, 1/3])

        els = self.physical_edge_lengths()

        return gem.ListTensor([[(jac[0, 0]*rts[i][0] + jac[0, 1]*rts[i][1]) / els[i],
                                (jac[1, 0]*rts[i][0] + jac[1, 1]*rts[i][1]) / els[i]]
                               for i in range(3)])

    def physical_normals(self):
        pts = self.physical_tangents()
        return gem.ListTensor([[pts[i, 1], -1*pts[i, 0]] for i in range(3)])

    def physical_edge_lengths(self):
        expr = ufl.classes.CellEdgeVectors(self.mt.terminal.ufl_domain())
        if self.mt.restriction == '+':
            expr = PositiveRestricted(expr)
        elif self.mt.restriction == '-':
            expr = NegativeRestricted(expr)

        expr = ufl.as_vector([ufl.sqrt(ufl.dot(expr[i, :], expr[i, :])) for i in range(3)])
        config = {"point_set": PointSingleton([1/3, 1/3])}
        config.update(self.config)
        context = PointSetContext(**config)
        expr = self.preprocess(expr, context)
        return map_expr_dag(context.translator, expr)

    def physical_points(self, point_set, entity=None):
        """Converts point_set from reference to physical space"""
        expr = SpatialCoordinate(self.mt.terminal.ufl_domain())
        point_shape, = point_set.expression.shape
        if entity is not None:
            e, _ = entity
            assert point_shape == e
        else:
            assert point_shape == expr.ufl_domain().topological_dimension()
        if self.mt.restriction == '+':
            expr = PositiveRestricted(expr)
        elif self.mt.restriction == '-':
            expr = NegativeRestricted(expr)
        config = {"point_set": point_set}
        config.update(self.config)
        if entity is not None:
            config.update({name: getattr(self.interface, name)
                           for name in ["integration_dim", "entity_ids"]})
        context = PointSetContext(**config)
        expr = self.preprocess(expr, context)
        mapped = map_expr_dag(context.translator, expr)
        indices = tuple(gem.Index() for _ in mapped.shape)
        return gem.ComponentTensor(gem.Indexed(mapped, indices), point_set.indices + indices)

    def physical_vertices(self):
        vs = PointSet(self.interface.fiat_cell.vertices)
        return self.physical_points(vs)


def needs_coordinate_mapping(element):
    """Does this UFL element require a CoordinateMapping for translation?"""
    if element.family() == 'Real':
        return False
    else:
        return isinstance(create_element(element), NeedsCoordinateMappingElement)


class PointSetContext(ContextBase):
    """Context for compile-time known evaluation points."""

    keywords = ContextBase.keywords + (
        'quadrature_degree',
        'quadrature_rule',
        'point_set',
        'weight_expr',
    )

    @cached_property
    def quadrature_rule(self):
        integration_cell = self.fiat_cell.construct_subelement(self.integration_dim)
        return make_quadrature(integration_cell, self.quadrature_degree)

    @cached_property
    def point_set(self):
        return self.quadrature_rule.point_set

    @cached_property
    def point_indices(self):
        return self.point_set.indices

    @cached_property
    def point_expr(self):
        return self.point_set.expression

    @cached_property
    def weight_expr(self):
        return self.quadrature_rule.weight_expression

    def basis_evaluation(self, finat_element, mt, entity_id):
        return finat_element.basis_evaluation(mt.local_derivatives,
                                              self.point_set,
                                              (self.integration_dim, entity_id),
                                              coordinate_mapping=CoordinateMapping(mt, self))


class GemPointContext(ContextBase):
    """Context for evaluation at arbitrary reference points."""

    keywords = ContextBase.keywords + (
        'point_indices',
        'point_expr',
        'weight_expr',
    )

    def basis_evaluation(self, finat_element, mt, entity_id):
        return finat_element.point_evaluation(mt.local_derivatives,
                                              self.point_expr,
                                              (self.integration_dim, entity_id))


class Translator(MultiFunction, ModifiedTerminalMixin, ufl2gem.Mixin):
    """Multifunction for translating UFL -> GEM.  Incorporates ufl2gem.Mixin, and
    dispatches on terminal type when reaching modified terminals."""

    def __init__(self, context):
        # MultiFunction.__init__ does not call further __init__
        # methods, but ufl2gem.Mixin must be initialised.
        # (ModifiedTerminalMixin requires no initialisation.)
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)

        # Need context during translation!
        self.context = context

    # We just use the provided quadrature rule to
    # perform the integration.
    # Can't put these in the ufl2gem mixin, since they (unlike
    # everything else) want access to the translation context.
    def cell_avg(self, o):
        if self.context.integral_type != "cell":
            # Need to create a cell-based quadrature rule and
            # translate the expression using that (c.f. CellVolume
            # below).
            raise NotImplementedError("CellAvg on non-cell integrals not yet implemented")
        integrand, = o.ufl_operands
        domain = o.ufl_domain()
        measure = ufl.Measure(self.context.integral_type, domain=domain)
        integrand, degree, argument_multiindices = entity_avg(integrand / CellVolume(domain), measure, self.context.argument_multiindices)

        config = {name: getattr(self.context, name)
                  for name in ["ufl_cell", "index_cache", "scalar_type"]}
        config.update(quadrature_degree=degree, interface=self.context,
                      argument_multiindices=argument_multiindices)
        expr, = compile_ufl(integrand, point_sum=True, **config)
        return expr

    def facet_avg(self, o):
        if self.context.integral_type == "cell":
            raise ValueError("Can't take FacetAvg in cell integral")
        integrand, = o.ufl_operands
        domain = o.ufl_domain()
        measure = ufl.Measure(self.context.integral_type, domain=domain)
        integrand, degree, argument_multiindices = entity_avg(integrand / FacetArea(domain), measure, self.context.argument_multiindices)

        config = {name: getattr(self.context, name)
                  for name in ["ufl_cell", "index_cache", "scalar_type",
                               "integration_dim", "entity_ids",
                               "integral_type"]}
        config.update(quadrature_degree=degree, interface=self.context,
                      argument_multiindices=argument_multiindices)
        expr, = compile_ufl(integrand, point_sum=True, **config)
        return expr

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
    assert ctx.integral_type != "cell"
    # Sum of quadrature weights is entity volume
    return gem.optimise.aggressive_unroll(gem.index_sum(ctx.weight_expr,
                                                        ctx.point_indices))


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


@translate.register(ReferenceCellEdgeVectors)
def translate_reference_cell_edge_vectors(terminal, mt, ctx):
    from FIAT.reference_element import TensorProductCell as fiat_TensorProductCell
    fiat_cell = ctx.fiat_cell
    if isinstance(fiat_cell, fiat_TensorProductCell):
        raise NotImplementedError("ReferenceCellEdgeVectors not implemented on TensorProductElements yet")

    nedges = len(fiat_cell.get_topology()[1])
    vecs = numpy.vstack(map(fiat_cell.compute_edge_tangent, range(nedges)))
    assert vecs.shape == terminal.ufl_shape
    return gem.Literal(vecs)


@translate.register(CellCoordinate)
def translate_cell_coordinate(terminal, mt, ctx):
    if ctx.integration_dim == ctx.fiat_cell.get_dimension():
        return ctx.point_expr

    # This destroys the structure of the quadrature points, but since
    # this code path is only used to implement CellCoordinate in facet
    # integrals, hopefully it does not matter much.
    ps = ctx.point_set
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
    return ctx.point_expr


@translate.register(SpatialCoordinate)
def translate_spatialcoordinate(terminal, mt, ctx):
    # Replace terminal with a Coefficient
    terminal = ctx.coordinate(terminal.ufl_domain())
    # Get back to reference space
    terminal = preprocess_expression(terminal, complex_mode=ctx.complex_mode)
    # Rebuild modified terminal
    expr = construct_modified_terminal(mt, terminal)
    # Translate replaced UFL snippet
    return ctx.translator(expr)


class CellVolumeKernelInterface(ProxyKernelInterface):
    # Since CellVolume is evaluated as a cell integral, we must ensure
    # that the right restriction is applied when it is used in an
    # interior facet integral.  This proxy diverts coefficient
    # translation to use a specified restriction.

    def __init__(self, wrapee, restriction):
        ProxyKernelInterface.__init__(self, wrapee)
        self.restriction = restriction

    def coefficient(self, ufl_coefficient, r):
        assert r is None
        return self._wrapee.coefficient(ufl_coefficient, self.restriction)


@translate.register(CellVolume)
def translate_cellvolume(terminal, mt, ctx):
    integrand, degree = one_times(ufl.dx(domain=terminal.ufl_domain()))
    interface = CellVolumeKernelInterface(ctx, mt.restriction)

    config = {name: getattr(ctx, name)
              for name in ["ufl_cell", "index_cache", "scalar_type"]}
    config.update(interface=interface, quadrature_degree=degree)
    expr, = compile_ufl(integrand, point_sum=True, **config)
    return expr


@translate.register(FacetArea)
def translate_facetarea(terminal, mt, ctx):
    assert ctx.integral_type != 'cell'
    domain = terminal.ufl_domain()
    integrand, degree = one_times(ufl.Measure(ctx.integral_type, domain=domain))

    config = {name: getattr(ctx, name)
              for name in ["ufl_cell", "integration_dim", "scalar_type",
                           "entity_ids", "index_cache"]}
    config.update(interface=ctx, quadrature_degree=degree)
    expr, = compile_ufl(integrand, point_sum=True, **config)
    return expr


@translate.register(CellOrigin)
def translate_cellorigin(terminal, mt, ctx):
    domain = terminal.ufl_domain()
    coords = SpatialCoordinate(domain)
    expression = construct_modified_terminal(mt, coords)
    point_set = PointSingleton((0.0,) * domain.topological_dimension())

    config = {name: getattr(ctx, name)
              for name in ["ufl_cell", "index_cache", "scalar_type"]}
    config.update(interface=ctx, point_set=point_set)
    context = PointSetContext(**config)
    return context.translator(expression)


@translate.register(CellVertices)
def translate_cell_vertices(terminal, mt, ctx):
    coords = SpatialCoordinate(terminal.ufl_domain())
    ufl_expr = construct_modified_terminal(mt, coords)
    ps = PointSet(numpy.array(ctx.fiat_cell.get_vertices()))

    config = {name: getattr(ctx, name)
              for name in ["ufl_cell", "index_cache", "scalar_type"]}
    config.update(interface=ctx, point_set=ps)
    context = PointSetContext(**config)
    expr = context.translator(ufl_expr)

    # Wrap up point (vertex) index
    c = gem.Index()
    return gem.ComponentTensor(gem.Indexed(expr, (c,)), ps.indices + (c,))


@translate.register(CellEdgeVectors)
def translate_cell_edge_vectors(terminal, mt, ctx):
    # WARNING: Assumes straight edges!
    coords = CellVertices(terminal.ufl_domain())
    ufl_expr = construct_modified_terminal(mt, coords)
    cell_vertices = ctx.translator(ufl_expr)

    e = gem.Index()
    c = gem.Index()
    expr = gem.ListTensor([
        gem.Sum(gem.Indexed(cell_vertices, (u, c)),
                gem.Product(gem.Literal(-1),
                            gem.Indexed(cell_vertices, (v, c))))
        for _, (u, v) in sorted(ctx.fiat_cell.get_topology()[1].items())
    ])
    return gem.ComponentTensor(gem.Indexed(expr, (e,)), (e, c))


def fiat_to_ufl(fiat_dict, order):
    # All derivative multiindices must be of the same dimension.
    dimension, = set(len(alpha) for alpha in fiat_dict.keys())

    # All derivative tables must have the same shape.
    shape, = set(table.shape for table in fiat_dict.values())
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
    element = ctx.create_element(terminal.ufl_element(), restriction=mt.restriction)

    def callback(entity_id):
        finat_dict = ctx.basis_evaluation(element, mt, entity_id)
        # Filter out irrelevant derivatives
        filtered_dict = {alpha: table
                         for alpha, table in finat_dict.items()
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

    element = ctx.create_element(terminal.ufl_element(), restriction=mt.restriction)

    # Collect FInAT tabulation for all entities
    per_derivative = collections.defaultdict(list)
    for entity_id in ctx.entity_ids:
        finat_dict = ctx.basis_evaluation(element, mt, entity_id)
        for alpha, table in finat_dict.items():
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
                          for alpha, tables in per_derivative.items()}
    else:
        f = ctx.entity_number(mt.restriction)
        per_derivative = {alpha: gem.select_expression(tables, f)
                          for alpha, tables in per_derivative.items()}

    # Coefficient evaluation
    ctx.index_cache.setdefault(terminal.ufl_element(), element.get_indices())
    beta = ctx.index_cache[terminal.ufl_element()]
    zeta = element.get_value_indices()
    vec_beta, = gem.optimise.remove_componenttensors([gem.Indexed(vec, beta)])
    value_dict = {}
    for alpha, table in per_derivative.items():
        table_qi = gem.Indexed(table, beta + zeta)
        summands = []
        for var, expr in unconcatenate([(vec_beta, table_qi)], ctx.index_cache):
            indices = tuple(i for i in var.index_ordering() if i not in ctx.unsummed_coefficient_indices)
            value = gem.IndexSum(gem.Product(expr, var), indices)
            summands.append(gem.optimise.contraction(value))
        optimised_value = gem.optimise.make_sum(summands)
        value_dict[alpha] = gem.ComponentTensor(optimised_value, zeta)

    # Change from FIAT to UFL arrangement
    result = fiat_to_ufl(value_dict, mt.local_derivatives)
    assert result.shape == mt.expr.ufl_shape
    assert set(result.free_indices) - ctx.unsummed_coefficient_indices <= set(ctx.point_indices)

    # Detect Jacobian of affine cells
    if not result.free_indices and all(numpy.count_nonzero(node.array) <= 2
                                       for node in traversal((result,))
                                       if isinstance(node, gem.Literal)):
        result = gem.optimise.aggressive_unroll(result)
    return result


def compile_ufl(expression, interior_facet=False, point_sum=False, **kwargs):

    # kwargs must be context kwargs for a PointSetContext or GemPointContext
    point_expr = kwargs.get("point_expr")
    if point_expr:
        context = GemPointContext(**kwargs)
    else:
        context = PointSetContext(**kwargs)

    # Abs-simplification
    expression = simplify_abs(expression, context.complex_mode)
    if interior_facet:
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(context.argument_multiindices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), expression))
    else:
        expressions = [expression]

    # Translate UFL to GEM, lowering finite element specific nodes
    result = map_expr_dags(context.translator, expressions)
    if point_sum:
        result = [gem.index_sum(expr, context.point_indices) for expr in result]
    return result
