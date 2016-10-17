from __future__ import absolute_import, print_function, division
from six.moves import range

import collections
import time
from functools import reduce

from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.classes import Form
from ufl.log import GREEN

import gem
import gem.optimise as opt
import gem.impero_utils as impero_utils

from tsfc import fem, ufl_utils
from tsfc.coffee import generate as generate_coffee
from tsfc.constants import SCALAR_TYPE, default_parameters
from tsfc.fiatinterface import QuadratureRule, as_fiat_cell, create_quadrature
from tsfc.logging import logger

from tsfc.kernel_interface import ProxyKernelInterface
import tsfc.kernel_interface.firedrake as firedrake_interface


def compile_form(form, prefix="form", parameters=None):
    """Compiles a UFL form into a set of assembly kernels.

    :arg form: UFL form
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :returns: list of kernels
    """
    cpu_time = time.time()

    assert isinstance(form, Form)

    fd = ufl_utils.compute_form_data(form)
    logger.info(GREEN % "compute_form_data finished in %g seconds.", time.time() - cpu_time)

    kernels = []
    for integral_data in fd.integral_data:
        start = time.time()
        try:
            kernels.append(compile_integral(integral_data, fd, prefix, parameters))
        except impero_utils.NoopError:
            pass
        logger.info(GREEN % "compile_integral finished in %g seconds.", time.time() - start)

    logger.info(GREEN % "TSFC finished in %g seconds.", time.time() - cpu_time)
    return kernels


def compile_integral(integral_data, form_data, prefix, parameters,
                     interface=firedrake_interface):
    """Compiles a UFL integral into an assembly kernel.

    :arg integral_data: UFL integral data
    :arg form_data: UFL form data
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :arg interface: backend module for the kernel interface
    :returns: a kernel constructed by the kernel interface
    """
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    # Remove these here, they're handled below.
    if parameters.get("quadrature_degree") in ["auto", "default", None, -1, "-1"]:
        del parameters["quadrature_degree"]
    if parameters.get("quadrature_rule") in ["auto", "default", None]:
        del parameters["quadrature_rule"]

    integral_type = integral_data.integral_type
    interior_facet = integral_type.startswith("interior_facet")
    mesh = integral_data.domain
    cell = integral_data.domain.ufl_cell()
    arguments = form_data.preprocessed_form.arguments()

    fiat_cell = as_fiat_cell(cell)
    integration_dim, entity_ids = lower_integral_type(fiat_cell, integral_type)

    argument_indices = tuple(gem.Index(name=name) for arg, name in zip(arguments, ['j', 'k']))
    quadrature_indices = []

    # Dict mapping domains to index in original_form.ufl_domains()
    domain_numbering = form_data.original_form.domain_numbering()
    builder = interface.KernelBuilder(integral_type, integral_data.subdomain_id,
                                      domain_numbering[integral_data.domain])
    return_variables = builder.set_arguments(arguments, argument_indices)

    coordinates = ufl_utils.coordinate_coefficient(mesh)
    if ufl_utils.is_element_affine(mesh.ufl_coordinate_element()):
        # For affine mesh geometries we prefer code generation that
        # composes well with optimisations.
        builder.set_coordinates(coordinates, mode='list_tensor')
    else:
        # Otherwise we use the approach that might be faster (?)
        builder.set_coordinates(coordinates)

    builder.set_coefficients(integral_data, form_data)

    # Map from UFL FiniteElement objects to Index instances.  This is
    # so we reuse Index instances when evaluating the same coefficient
    # multiple times with the same table.  Occurs, for example, if we
    # have multiple integrals here (and the affine coordinate
    # evaluation can be hoisted).
    index_cache = collections.defaultdict(gem.Index)

    kernel_cfg = dict(interface=builder,
                      ufl_cell=cell,
                      precision=parameters["precision"],
                      integration_dim=integration_dim,
                      entity_ids=entity_ids,
                      argument_indices=argument_indices,
                      index_cache=index_cache)

    kernel_cfg["facetarea"] = facetarea_generator(mesh, coordinates, kernel_cfg, integral_type)
    kernel_cfg["cellvolume"] = cellvolume_generator(mesh, coordinates, kernel_cfg)

    irs = []
    for integral in integral_data.integrals:
        params = {}
        # Record per-integral parameters
        params.update(integral.metadata())
        if params.get("quadrature_rule") == "default":
            del params["quadrature_rule"]
        # parameters override per-integral metadata
        params.update(parameters)

        # Check if the integral has a quad degree attached, otherwise use
        # the estimated polynomial degree attached by compute_form_data
        quadrature_degree = params.get("quadrature_degree",
                                       params["estimated_polynomial_degree"])
        integration_cell = fiat_cell.construct_subelement(integration_dim)
        quad_rule = params.get("quadrature_rule",
                               create_quadrature(integration_cell,
                                                 quadrature_degree))

        if not isinstance(quad_rule, QuadratureRule):
            raise ValueError("Expected to find a QuadratureRule object, not a %s" %
                             type(quad_rule))

        integrand = ufl_utils.replace_coordinates(integral.integrand(), coordinates)
        integrand = ufl_utils.split_coefficients(integrand, builder.coefficient_split)
        quadrature_index = gem.Index(name='ip')
        quadrature_indices.append(quadrature_index)
        config = kernel_cfg.copy()
        config.update(quadrature_rule=quad_rule, point_index=quadrature_index)
        ir = fem.compile_ufl(integrand, interior_facet=interior_facet, **config)
        if parameters["unroll_indexsum"]:
            ir = opt.unroll_indexsum(ir, max_extent=parameters["unroll_indexsum"])
        irs.append([(gem.IndexSum(expr, quadrature_index)
                     if quadrature_index in expr.free_indices
                     else expr)
                    for expr in ir])

    # Sum the expressions that are part of the same restriction
    ir = list(reduce(gem.Sum, e, gem.Zero()) for e in zip(*irs))

    # Need optimised roots for COFFEE
    ir = opt.remove_componenttensors(ir)

    # Look for cell orientations in the IR
    if builder.needs_cell_orientations(ir):
        builder.require_cell_orientations()

    impero_c = impero_utils.compile_gem(return_variables, ir,
                                        tuple(quadrature_indices) + argument_indices,
                                        remove_zeros=True)

    # Generate COFFEE
    index_names = [(index, index.name) for index in argument_indices]
    if len(quadrature_indices) == 1:
        index_names.append((quadrature_indices[0], 'ip'))
    else:
        for i, quadrature_index in enumerate(quadrature_indices):
            index_names.append((quadrature_index, 'ip_%d' % i))

    body = generate_coffee(impero_c, index_names, parameters, ir, argument_indices)

    kernel_name = "%s_%s_integral_%s" % (prefix, integral_type, integral_data.subdomain_id)
    return builder.construct_kernel(kernel_name, body)


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


def cellvolume_generator(domain, coordinate_coefficient, kernel_config):
    def cellvolume(restriction):
        from ufl import dx
        form = 1 * dx(domain=domain)
        fd = ufl_utils.compute_form_data(form)
        itg_data, = fd.integral_data
        integral, = itg_data.integrals

        integrand = ufl_utils.replace_coordinates(integral.integrand(),
                                                  coordinate_coefficient)

        # Check if the integral has a quad degree attached, otherwise use
        # the estimated polynomial degree attached by compute_form_data
        quadrature_degree = integral.metadata()["estimated_polynomial_degree"]
        quadrature_index = gem.Index(name='q')

        config = {k: v for k, v in kernel_config.items()
                  if k in ["ufl_cell", "precision", "index_cache"]}
        interface = CellVolumeKernelInterface(kernel_config["interface"], restriction)
        config.update(interface=interface,
                      quadrature_degree=quadrature_degree,
                      point_index=quadrature_index)
        expr, = fem.compile_ufl(integrand, **config)
        if quadrature_index in expr.free_indices:
            expr = gem.IndexSum(expr, quadrature_index)
        return expr
    return cellvolume


def facetarea_generator(domain, coordinate_coefficient, kernel_config, integral_type):
    def facetarea():
        from ufl import Measure
        assert integral_type != 'cell'
        form = 1 * Measure(integral_type, domain=domain)
        fd = ufl_utils.compute_form_data(form)
        itg_data, = fd.integral_data
        integral, = itg_data.integrals

        integrand = ufl_utils.replace_coordinates(integral.integrand(),
                                                  coordinate_coefficient)

        # Check if the integral has a quad degree attached, otherwise use
        # the estimated polynomial degree attached by compute_form_data
        quadrature_degree = integral.metadata()["estimated_polynomial_degree"]
        quadrature_index = gem.Index(name='q')

        config = kernel_config.copy()
        config.update(quadrature_degree=quadrature_degree,
                      point_index=quadrature_index)
        expr, = fem.compile_ufl(integrand, **config)
        if quadrature_index in expr.free_indices:
            expr = gem.IndexSum(expr, quadrature_index)
        return expr
    return facetarea


def compile_expression_at_points(expression, points, coordinates, parameters=None):
    """Compiles a UFL expression to be evaluated at compile-time known
    reference points.  Useful for interpolating UFL expressions onto
    function spaces with only point evaluation nodes.

    :arg expression: UFL expression
    :arg points: reference coordinates of the evaluation points
    :arg coordinates: the coordinate function
    :arg parameters: parameters object
    """
    import coffee.base as ast

    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    # No arguments, please!
    if extract_arguments(expression):
        return ValueError("Cannot interpolate UFL expression with Arguments!")

    # Apply UFL preprocessing
    expression = ufl_utils.preprocess_expression(expression)

    # Replace coordinates (if any)
    domain = expression.ufl_domain()
    if domain:
        assert coordinates.ufl_domain() == domain
        expression = ufl_utils.replace_coordinates(expression, coordinates)

    # Initialise kernel builder
    builder = firedrake_interface.ExpressionKernelBuilder()
    builder.set_coefficients(extract_coefficients(expression))

    # Split mixed coefficients
    expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    point_index = gem.Index(name='p')
    ir, = fem.compile_ufl(expression,
                          interface=builder,
                          ufl_cell=coordinates.ufl_domain().ufl_cell(),
                          precision=parameters["precision"],
                          points=points,
                          point_index=point_index)

    # Deal with non-scalar expressions
    value_shape = ir.shape
    tensor_indices = tuple(gem.Index() for s in value_shape)
    if value_shape:
        ir = gem.Indexed(ir, tensor_indices)

    # Build kernel body
    return_shape = (len(points),) + value_shape
    return_indices = (point_index,) + tensor_indices
    return_var = gem.Variable('A', return_shape)
    return_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('A', rank=return_shape))
    return_expr = gem.Indexed(return_var, return_indices)
    impero_c = impero_utils.compile_gem([return_expr], [ir], return_indices)
    body = generate_coffee(impero_c, {point_index: 'p'}, parameters)

    # Handle cell orientations
    if builder.needs_cell_orientations([ir]):
        builder.require_cell_orientations()

    # Build kernel tuple
    return builder.construct_kernel(return_arg, body)


def lower_integral_type(fiat_cell, integral_type):
    """Lower integral type into the dimension of the integration
    subentity and a list of entity numbers for that dimension.

    :arg fiat_cell: FIAT reference cell
    :arg integral_type: integral type (string)
    """
    dim = fiat_cell.get_dimension()
    if integral_type == 'cell':
        integration_dim = dim
    elif integral_type in ['exterior_facet', 'interior_facet']:
        integration_dim = dim - 1
    else:
        # Extrusion case
        basedim, extrdim = dim
        assert extrdim == 1

        if integral_type in ['exterior_facet_vert', 'interior_facet_vert']:
            integration_dim = (basedim - 1, 1)
        elif integral_type in ['exterior_facet_bottom', 'exterior_facet_top', 'interior_facet_horiz']:
            integration_dim = (basedim, 0)
        else:
            raise NotImplementedError("integral type %s not supported" % integral_type)

    if integral_type == 'exterior_facet_bottom':
        entity_ids = [0]
    elif integral_type == 'exterior_facet_top':
        entity_ids = [1]
    else:
        entity_ids = list(range(len(fiat_cell.get_topology()[integration_dim])))

    return integration_dim, entity_ids
