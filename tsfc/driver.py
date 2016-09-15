from __future__ import absolute_import

import collections
import time

from ufl.classes import Form, CellVolume, FacetArea
from ufl.algorithms import compute_form_data
from ufl.log import GREEN

import gem
import gem.optimise as opt
import gem.impero_utils as impero_utils

from tsfc import fem, ufl_utils
from tsfc.coffee import generate as generate_coffee
from tsfc.constants import default_parameters
from tsfc.fiatinterface import QuadratureRule, as_fiat_cell, create_quadrature
from tsfc.kernel_interface import KernelBuilder, needs_cell_orientations
from tsfc.logging import logger


def compile_form(form, prefix="form", parameters=None):
    """Compiles a UFL form into a set of assembly kernels.

    :arg form: UFL form
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :returns: list of kernels
    """
    cpu_time = time.time()

    assert isinstance(form, Form)

    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    fd = compute_form_data(form,
                           do_apply_function_pullbacks=True,
                           do_apply_integral_scaling=True,
                           do_apply_geometry_lowering=True,
                           do_apply_restrictions=True,
                           preserve_geometry_types=(CellVolume, FacetArea),
                           do_estimate_degrees=True)
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


def compile_integral(integral_data, form_data, prefix, parameters):
    """Compiles a UFL integral into an assembly kernel.

    :arg integral_data: UFL integral data
    :arg form_data: UFL form data
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :returns: a kernel, or None if the integral simplifies to zero
    """
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
    builder = KernelBuilder(integral_type, integral_data.subdomain_id,
                            domain_numbering[integral_data.domain])
    return_variables = builder.set_arguments(arguments, argument_indices)

    coordinates = ufl_utils.coordinate_coefficient(mesh)
    if ufl_utils.is_element_affine(mesh.ufl_coordinate_element()):
        # For affine mesh geometries we prefer code generation that
        # composes well with optimisations.
        #builder.set_coordinates(coordinates, "coords", mode='list_tensor')
        builder.set_coordinates(coordinates, "coordinate_dofs", mode='list_tensor')
    else:
        # Otherwise we use the approach that might be faster (?)
        #builder.set_coordinates(coordinates, "coords")
        builder.set_coordinates(coordinates, "coordinate_dofs")

    builder.set_coefficients(integral_data, form_data)

    builder.set_cell_orientations()
    builder.set_facets()

    # Map from UFL FiniteElement objects to Index instances.  This is
    # so we reuse Index instances when evaluating the same coefficient
    # multiple times with the same table.  Occurs, for example, if we
    # have multiple integrals here (and the affine coordinate
    # evaluation can be hoisted).
    index_cache = collections.defaultdict(gem.Index)

    # TODO: refactor this!
    def cellvolume(restriction):
        from ufl import dx
        form = 1 * dx(domain=mesh)
        fd = compute_form_data(form,
                               do_apply_function_pullbacks=True,
                               do_apply_integral_scaling=True,
                               do_apply_geometry_lowering=True,
                               do_apply_restrictions=True,
                               do_estimate_degrees=True)
        itg_data, = fd.integral_data
        integral, = itg_data.integrals

        # Check if the integral has a quad degree attached, otherwise use
        # the estimated polynomial degree attached by compute_form_data
        quadrature_degree = integral.metadata()["estimated_polynomial_degree"]

        integrand = ufl_utils.replace_coordinates(integral.integrand(), coordinates)
        quadrature_index = gem.Index(name='q')
        if interior_facet:
            def coefficient(ufl_coefficient, r):
                assert r is None
                return builder.coefficient(ufl_coefficient, restriction)
        else:
            assert restriction is None
            coefficient = builder.coefficient
        ir = fem.compile_ufl(integrand,
                             cell=cell,
                             quadrature_degree=quadrature_degree,
                             point_index=quadrature_index,
                             coefficient=coefficient,
                             index_cache=index_cache)
        if parameters["unroll_indexsum"]:
            ir = opt.unroll_indexsum(ir, max_extent=parameters["unroll_indexsum"])
        expr, = ir
        if quadrature_index in expr.free_indices:
            expr = gem.IndexSum(expr, quadrature_index)
        return expr

    # TODO: refactor this!
    def facetarea():
        from ufl import Measure
        assert integral_type != 'cell'
        form = 1 * Measure(integral_type, domain=mesh)
        fd = compute_form_data(form,
                               do_apply_function_pullbacks=True,
                               do_apply_integral_scaling=True,
                               do_apply_geometry_lowering=True,
                               do_apply_restrictions=True,
                               do_estimate_degrees=True)
        itg_data, = fd.integral_data
        integral, = itg_data.integrals

        # Check if the integral has a quad degree attached, otherwise use
        # the estimated polynomial degree attached by compute_form_data
        quadrature_degree = integral.metadata()["estimated_polynomial_degree"]

        integrand = ufl_utils.replace_coordinates(integral.integrand(), coordinates)
        quadrature_index = gem.Index(name='q')
        ir = fem.compile_ufl(integrand,
                             cell=cell,
                             integration_dim=integration_dim,
                             entity_ids=entity_ids,
                             quadrature_degree=quadrature_degree,
                             point_index=quadrature_index,
                             coefficient=builder.coefficient,
                             facet_number=builder.facet_number,
                             index_cache=index_cache)
        if parameters["unroll_indexsum"]:
            ir = opt.unroll_indexsum(ir, max_extent=parameters["unroll_indexsum"])
        expr, = ir
        if quadrature_index in expr.free_indices:
            expr = gem.IndexSum(expr, quadrature_index)
        return expr

    irs = []
    for integral in integral_data.integrals:
        params = {}
        # Record per-integral parameters
        params.update(integral.metadata())
        # parameters override per-integral metadata
        params.update(parameters)

        # Check if the integral has a quad degree attached, otherwise use
        # the estimated polynomial degree attached by compute_form_data
        quad_degree = params.get("quadrature_degree")
        if quad_degree in [None, "auto", "default", -1, "-1"]:
            quad_degree = params["estimated_polynomial_degree"]
        integration_cell = fiat_cell.construct_subelement(integration_dim)
        quad_rule = params.get("quadrature_rule")
        if quad_rule in [None, "auto", "default"]:
            quad_rule = create_quadrature(integration_cell, quad_degree)

        if not isinstance(quad_rule, QuadratureRule):
            raise ValueError("Expected to find a QuadratureRule object, not a %s" %
                             type(quad_rule))

        integrand = ufl_utils.replace_coordinates(integral.integrand(), coordinates)
        integrand = ufl_utils.split_coefficients(integrand, builder.coefficient_split)
        quadrature_index = gem.Index(name='ip')
        quadrature_indices.append(quadrature_index)
        ir = fem.compile_ufl(integrand,
                             interior_facet=interior_facet,
                             cell=cell,
                             integration_dim=integration_dim,
                             entity_ids=entity_ids,
                             quadrature_rule=quad_rule,
                             point_index=quadrature_index,
                             argument_indices=argument_indices,
                             coefficient=builder.coefficient,
                             cell_orientation=builder.cell_orientation,
                             facet_number=builder.facet_number,
                             index_cache=index_cache,
                             cellvolume=cellvolume,
                             facetarea=facetarea)
        if parameters.get("unroll_indexsum"):
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
    if needs_cell_orientations(ir):
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

    body = generate_coffee(impero_c, index_names, ir, argument_indices)

    kernel_name = "%s_%s_integral_%s" % (prefix, integral_type, integral_data.subdomain_id)
    return builder.construct_kernel(kernel_name, body)


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
        entity_ids = range(len(fiat_cell.get_topology()[integration_dim]))

    return integration_dim, entity_ids
