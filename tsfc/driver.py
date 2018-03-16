import collections
import operator
import string
import time
from functools import reduce
from itertools import chain

from numpy import asarray

# import ufl
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.algorithms.analysis import has_type
from ufl.classes import Form, GeometricQuantity
from ufl.log import GREEN
from ufl.utils.sequences import max_degree

import gem
import gem.impero_utils as impero_utils

from FIAT.reference_element import TensorProductCell

from finat.point_set import PointSet
from finat.quadrature import AbstractQuadratureRule, make_quadrature

from tsfc import fem, ufl_utils
from tsfc.coffee import SCALAR_TYPE, generate as generate_coffee
from tsfc.fiatinterface import as_fiat_cell
from tsfc.logging import logger
from tsfc.parameters import default_parameters

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
        kernel = compile_integral(integral_data, fd, prefix, parameters)
        if kernel is not None:
            kernels.append(kernel)
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
    kernel_name = "%s_%s_integral_%s" % (prefix, integral_type, integral_data.subdomain_id)
    # Handle negative subdomain_id
    kernel_name = kernel_name.replace("-", "_")

    fiat_cell = as_fiat_cell(cell)
    integration_dim, entity_ids = lower_integral_type(fiat_cell, integral_type)

    quadrature_indices = []

    # Dict mapping domains to index in original_form.ufl_domains()
    domain_numbering = form_data.original_form.domain_numbering()
    builder = interface.KernelBuilder(integral_type, integral_data.subdomain_id,
                                      domain_numbering[integral_data.domain])
    argument_multiindices = tuple(builder.create_element(arg.ufl_element()).get_indices()
                                  for arg in arguments)
    return_variables = builder.set_arguments(arguments, argument_multiindices)

    builder.set_coordinates(mesh)

    builder.set_coefficients(integral_data, form_data)

    # Map from UFL FiniteElement objects to multiindices.  This is
    # so we reuse Index instances when evaluating the same coefficient
    # multiple times with the same table.
    #
    # We also use the same dict for the unconcatenate index cache,
    # which maps index objects to tuples of multiindices.  These two
    # caches shall never conflict as their keys have different types
    # (UFL finite elements vs. GEM index objects).
    index_cache = {}

    kernel_cfg = dict(interface=builder,
                      ufl_cell=cell,
                      integral_type=integral_type,
                      precision=parameters["precision"],
                      integration_dim=integration_dim,
                      entity_ids=entity_ids,
                      argument_multiindices=argument_multiindices,
                      index_cache=index_cache)

    mode_irs = collections.OrderedDict()
    for integral in integral_data.integrals:
        params = parameters.copy()
        params.update(integral.metadata())  # integral metadata overrides
        if params.get("quadrature_rule") == "default":
            del params["quadrature_rule"]

        mode = pick_mode(params["mode"])
        mode_irs.setdefault(mode, collections.OrderedDict())

        # integrand = ufl.replace(integral.integrand(), form_data.function_replace_map)
        integrand = integral.integrand()
        integrand = ufl_utils.split_coefficients(integrand, builder.coefficient_split)

        # Check if the integral has a quad degree attached, otherwise use
        # the estimated polynomial degree attached by compute_form_data
        quadrature_degree = params.get("quadrature_degree",
                                       params["estimated_polynomial_degree"])
        try:
            quadrature_degree = params["quadrature_degree"]
        except KeyError:
            quadrature_degree = params["estimated_polynomial_degree"]
            functions = list(arguments) + [builder.coordinate(mesh)] + list(integral_data.integral_coefficients)
            function_degrees = [f.ufl_function_space().ufl_element().degree() for f in functions]
            if all((asarray(quadrature_degree) > 10 * asarray(degree)).all()
                   for degree in function_degrees):
                logger.warning("Estimated quadrature degree %s more "
                               "than tenfold greater than any "
                               "argument/coefficient degree (max %s)",
                               quadrature_degree, max_degree(function_degrees))

        try:
            quad_rule = params["quadrature_rule"]
        except KeyError:
            integration_cell = fiat_cell.construct_subelement(integration_dim)
            quad_rule = make_quadrature(integration_cell, quadrature_degree)

        if not isinstance(quad_rule, AbstractQuadratureRule):
            raise ValueError("Expected to find a QuadratureRule object, not a %s" %
                             type(quad_rule))

        quadrature_multiindex = quad_rule.point_set.indices
        quadrature_indices.extend(quadrature_multiindex)

        config = kernel_cfg.copy()
        config.update(quadrature_rule=quad_rule)
        expressions = fem.compile_ufl(integrand,
                                      interior_facet=interior_facet,
                                      **config)
        reps = mode.Integrals(expressions, quadrature_multiindex,
                              argument_multiindices, params)
        for var, rep in zip(return_variables, reps):
            mode_irs[mode].setdefault(var, []).append(rep)

    # Finalise mode representations into a set of assignments
    assignments = []
    for mode, var_reps in mode_irs.items():
        assignments.extend(mode.flatten(var_reps.items(), index_cache))

    if assignments:
        return_variables, expressions = zip(*assignments)
    else:
        return_variables = []
        expressions = []

    # Need optimised roots for COFFEE
    options = dict(reduce(operator.and_,
                          [mode.finalise_options.items()
                           for mode in mode_irs.keys()]))
    expressions = impero_utils.preprocess_gem(expressions, **options)
    assignments = list(zip(return_variables, expressions))

    # Look for cell orientations in the IR
    if builder.needs_cell_orientations(expressions):
        builder.require_cell_orientations()

    # Construct ImperoC
    split_argument_indices = tuple(chain(*[var.index_ordering()
                                           for var in return_variables]))
    index_ordering = tuple(quadrature_indices) + split_argument_indices
    try:
        impero_c = impero_utils.compile_gem(assignments, index_ordering, remove_zeros=True)
    except impero_utils.NoopError:
        # No operations, construct empty kernel
        return builder.construct_empty_kernel(kernel_name)

    # Generate COFFEE
    index_names = []

    def name_index(index, name):
        index_names.append((index, name))
        if index in index_cache:
            for multiindex, suffix in zip(index_cache[index],
                                          string.ascii_lowercase):
                name_multiindex(multiindex, name + suffix)

    def name_multiindex(multiindex, name):
        if len(multiindex) == 1:
            name_index(multiindex[0], name)
        else:
            for i, index in enumerate(multiindex):
                name_index(index, name + str(i))

    name_multiindex(quadrature_indices, 'ip')
    for multiindex, name in zip(argument_multiindices, ['j', 'k']):
        name_multiindex(multiindex, name)

    # Construct kernel
    body = generate_coffee(impero_c, index_names, parameters["precision"], expressions, split_argument_indices)

    return builder.construct_kernel(kernel_name, body)


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

    # Initialise kernel builder
    builder = firedrake_interface.ExpressionKernelBuilder()

    # Replace coordinates (if any)
    domain = expression.ufl_domain()
    if domain:
        assert coordinates.ufl_domain() == domain
        builder.domain_coordinate[domain] = coordinates

    # Collect required coefficients
    coefficients = extract_coefficients(expression)
    if has_type(expression, GeometricQuantity):
        coefficients = [coordinates] + coefficients
    builder.set_coefficients(coefficients)

    # Split mixed coefficients
    expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    point_set = PointSet(points)
    config = dict(interface=builder,
                  ufl_cell=coordinates.ufl_domain().ufl_cell(),
                  precision=parameters["precision"],
                  point_set=point_set)
    ir, = fem.compile_ufl(expression, point_sum=False, **config)

    # Deal with non-scalar expressions
    value_shape = ir.shape
    tensor_indices = tuple(gem.Index() for s in value_shape)
    if value_shape:
        ir = gem.Indexed(ir, tensor_indices)

    # Build kernel body
    return_shape = (len(points),) + value_shape
    return_indices = point_set.indices + tensor_indices
    return_var = gem.Variable('A', return_shape)
    return_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('A', rank=return_shape))
    return_expr = gem.Indexed(return_var, return_indices)
    ir, = impero_utils.preprocess_gem([ir])
    impero_c = impero_utils.compile_gem([(return_expr, ir)], return_indices)
    point_index, = point_set.indices
    body = generate_coffee(impero_c, {point_index: 'p'}, parameters["precision"])

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
    vert_facet_types = ['exterior_facet_vert', 'interior_facet_vert']
    horiz_facet_types = ['exterior_facet_bottom', 'exterior_facet_top', 'interior_facet_horiz']

    dim = fiat_cell.get_dimension()
    if integral_type == 'cell':
        integration_dim = dim
    elif integral_type in ['exterior_facet', 'interior_facet']:
        if isinstance(fiat_cell, TensorProductCell):
            raise ValueError("{} integral cannot be used with a TensorProductCell; need to distinguish between vertical and horizontal contributions.".format(integral_type))
        integration_dim = dim - 1
    elif integral_type == 'vertex':
        integration_dim = 0
    elif integral_type in vert_facet_types + horiz_facet_types:
        # Extrusion case
        if not isinstance(fiat_cell, TensorProductCell):
            raise ValueError("{} integral requires a TensorProductCell.".format(integral_type))
        basedim, extrdim = dim
        assert extrdim == 1

        if integral_type in vert_facet_types:
            integration_dim = (basedim - 1, 1)
        elif integral_type in horiz_facet_types:
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


def pick_mode(mode):
    "Return one of the specialized optimisation modules from a mode string."
    try:
        from firedrake_citations import Citations
        cites = {"vanilla": ("Homolya2017", ),
                 "coffee": ("Luporini2016", "Homolya2017", ),
                 "spectral": ("Luporini2016", "Homolya2017", "Homolya2017a"),
                 "tensor": ("Kirby2006", "Homolya2017", )}
        for c in cites[mode]:
            Citations().register(c)
    except ImportError:
        pass
    if mode == "vanilla":
        import tsfc.vanilla as m
    elif mode == "coffee":
        import tsfc.coffee_mode as m
    elif mode == "spectral":
        import tsfc.spectral as m
    elif mode == "tensor":
        import tsfc.tensor as m
    else:
        raise ValueError("Unknown mode: {}".format(mode))
    return m
