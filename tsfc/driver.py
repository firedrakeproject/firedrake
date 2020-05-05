import collections
import operator
import string
import time
import sys
from functools import reduce
from itertools import chain

from numpy import asarray, allclose

import ufl
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.algorithms.analysis import has_type
from ufl.classes import Form, GeometricQuantity
from ufl.log import GREEN
from ufl.utils.sequences import max_degree

import gem
import gem.impero_utils as impero_utils

import FIAT
from FIAT.reference_element import TensorProductCell
from FIAT.functional import PointEvaluation

from finat.point_set import PointSet
from finat.quadrature import AbstractQuadratureRule, make_quadrature, QuadratureRule

from tsfc import fem, ufl_utils
from tsfc.fiatinterface import as_fiat_cell
from tsfc.logging import logger
from tsfc.parameters import default_parameters, is_complex
from tsfc.ufl_utils import apply_mapping

# To handle big forms. The various transformations might need a deeper stack
sys.setrecursionlimit(3000)


def compile_form(form, prefix="form", parameters=None, interface=None, coffee=True, diagonal=False):
    """Compiles a UFL form into a set of assembly kernels.

    :arg form: UFL form
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :arg coffee: compile coffee kernel instead of loopy kernel
    :arg diagonal: Are we building a kernel for the diagonal of a rank-2 element tensor?
    :returns: list of kernels
    """
    cpu_time = time.time()

    assert isinstance(form, Form)

    # Determine whether in complex mode:
    # complex nodes would break the refactoriser.
    complex_mode = parameters and is_complex(parameters.get("scalar_type"))
    if complex_mode:
        logger.warning("Disabling whole expression optimisations"
                       " in GEM for supporting complex mode.")
        parameters = parameters.copy()
        parameters["mode"] = 'vanilla'

    fd = ufl_utils.compute_form_data(form, complex_mode=complex_mode)
    logger.info(GREEN % "compute_form_data finished in %g seconds.", time.time() - cpu_time)

    kernels = []
    for integral_data in fd.integral_data:
        start = time.time()
        kernel = compile_integral(integral_data, fd, prefix, parameters, interface=interface, coffee=coffee, diagonal=diagonal)
        if kernel is not None:
            kernels.append(kernel)
        logger.info(GREEN % "compile_integral finished in %g seconds.", time.time() - start)

    logger.info(GREEN % "TSFC finished in %g seconds.", time.time() - cpu_time)
    return kernels


def compile_integral(integral_data, form_data, prefix, parameters, interface, coffee, *, diagonal=False):
    """Compiles a UFL integral into an assembly kernel.

    :arg integral_data: UFL integral data
    :arg form_data: UFL form data
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :arg interface: backend module for the kernel interface
    :arg diagonal: Are we building a kernel for the diagonal of a rank-2 element tensor?
    :returns: a kernel constructed by the kernel interface
    """
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _
    if interface is None:
        if coffee:
            import tsfc.kernel_interface.firedrake as firedrake_interface_coffee
            interface = firedrake_interface_coffee.KernelBuilder
        else:
            # Delayed import, loopy is a runtime dependency
            import tsfc.kernel_interface.firedrake_loopy as firedrake_interface_loopy
            interface = firedrake_interface_loopy.KernelBuilder

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
    builder = interface(integral_type, integral_data.subdomain_id,
                        domain_numbering[integral_data.domain],
                        parameters["scalar_type"],
                        diagonal=diagonal)
    argument_multiindices = tuple(builder.create_element(arg.ufl_element()).get_indices()
                                  for arg in arguments)
    if diagonal:
        # Error checking occurs in the builder constructor.
        # Diagonal assembly is obtained by using the test indices for
        # the trial space as well.
        a, _ = argument_multiindices
        argument_multiindices = (a, a)

    return_variables = builder.set_arguments(arguments, argument_multiindices)

    builder.set_coordinates(mesh)
    builder.set_cell_sizes(mesh)

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

        integrand = ufl.replace(integral.integrand(), form_data.function_replace_map)
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

    # Need optimised roots
    options = dict(reduce(operator.and_,
                          [mode.finalise_options.items()
                           for mode in mode_irs.keys()]))
    expressions = impero_utils.preprocess_gem(expressions, **options)
    assignments = list(zip(return_variables, expressions))

    # Let the kernel interface inspect the optimised IR to register
    # what kind of external data is required (e.g., cell orientations,
    # cell sizes, etc.).
    builder.register_requirements(expressions)

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

    return builder.construct_kernel(kernel_name, impero_c, parameters["precision"], index_names, quad_rule)


def compile_expression_dual_evaluation(expression, to_element, coordinates, interface=None,
                                       parameters=None, coffee=False):
    """Compile a UFL expression to be evaluated against a compile-time known reference element's dual basis.

    Useful for interpolating UFL expressions into e.g. N1curl spaces.

    :arg expression: UFL expression
    :arg to_element: A FIAT FiniteElement for the target space
    :arg coordinates: the coordinate function
    :arg interface: backend module for the kernel interface
    :arg parameters: parameters object
    :arg coffee: compile coffee kernel instead of loopy kernel
    """
    import coffee.base as ast
    import loopy as lp
    if any(len(dual.deriv_dict) != 0 for dual in to_element.dual_basis()):
        raise NotImplementedError("Can only interpolate onto dual basis functionals without derivative evaluation, sorry!")

    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    # Determine whether in complex mode
    complex_mode = is_complex(parameters["scalar_type"])

    # Find out which mapping to apply
    try:
        mapping, = set(to_element.mapping())
    except ValueError:
        raise NotImplementedError("Don't know how to interpolate onto zany spaces, sorry")
    expression = apply_mapping(expression, mapping)

    # Apply UFL preprocessing
    expression = ufl_utils.preprocess_expression(expression,
                                                 complex_mode=complex_mode)

    # Initialise kernel builder
    if interface is None:
        if coffee:
            import tsfc.kernel_interface.firedrake as firedrake_interface_coffee
            interface = firedrake_interface_coffee.ExpressionKernelBuilder
        else:
            # Delayed import, loopy is a runtime dependency
            import tsfc.kernel_interface.firedrake_loopy as firedrake_interface_loopy
            interface = firedrake_interface_loopy.ExpressionKernelBuilder

    builder = interface(parameters["scalar_type"])
    arguments = extract_arguments(expression)
    argument_multiindices = tuple(builder.create_element(arg.ufl_element()).get_indices()
                                  for arg in arguments)

    # Replace coordinates (if any)
    domain = expression.ufl_domain()
    if domain:
        assert coordinates.ufl_domain() == domain
        builder.domain_coordinate[domain] = coordinates
        builder.set_cell_sizes(domain)

    # Collect required coefficients
    coefficients = extract_coefficients(expression)
    if has_type(expression, GeometricQuantity) or any(fem.needs_coordinate_mapping(c.ufl_element()) for c in coefficients):
        coefficients = [coordinates] + coefficients
    builder.set_coefficients(coefficients)

    # Split mixed coefficients
    expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    kernel_cfg = dict(interface=builder,
                      ufl_cell=coordinates.ufl_domain().ufl_cell(),
                      precision=parameters["precision"],
                      argument_multiindices=argument_multiindices,
                      index_cache={})

    if all(isinstance(dual, PointEvaluation) for dual in to_element.dual_basis()):
        # This is an optimisation for point-evaluation nodes which
        # should go away once FInAT offers the interface properly
        qpoints = []
        # Everything is just a point evaluation.
        for dual in to_element.dual_basis():
            ptdict = dual.get_point_dict()
            qpoint, = ptdict.keys()
            (qweight, component), = ptdict[qpoint]
            assert allclose(qweight, 1.0)
            assert component == ()
            qpoints.append(qpoint)
        point_set = PointSet(qpoints)
        config = kernel_cfg.copy()
        config.update(point_set=point_set)

        # Allow interpolation onto QuadratureElements to refer to the quadrature
        # rule they represent
        if isinstance(to_element, FIAT.QuadratureElement):
            assert (asarray(qpoints) == asarray(to_element._points)).all()
            quad_rule = QuadratureRule(point_set, to_element._weights)
            config["quadrature_rule"] = quad_rule

        expr, = fem.compile_ufl(expression, **config, point_sum=False)
        shape_indices = tuple(gem.Index() for _ in expr.shape)
        basis_indices = point_set.indices
        ir = gem.Indexed(expr, shape_indices)
    else:
        # This is general code but is more unrolled than necssary.
        dual_expressions = []   # one for each functional
        broadcast_shape = len(expression.ufl_shape) - len(to_element.value_shape())
        shape_indices = tuple(gem.Index() for _ in expression.ufl_shape[:broadcast_shape])
        expr_cache = {}         # Sharing of evaluation of the expression at points
        for dual in to_element.dual_basis():
            pts = tuple(sorted(dual.get_point_dict().keys()))
            try:
                expr, point_set = expr_cache[pts]
            except KeyError:
                point_set = PointSet(pts)
                config = kernel_cfg.copy()
                config.update(point_set=point_set)
                expr, = fem.compile_ufl(expression, **config, point_sum=False)
                expr = gem.partial_indexed(expr, shape_indices)
                expr_cache[pts] = expr, point_set
            weights = collections.defaultdict(list)
            for p in pts:
                for (w, cmp) in dual.get_point_dict()[p]:
                    weights[cmp].append(w)
            qexprs = gem.Zero()
            for cmp in sorted(weights):
                qweights = gem.Literal(weights[cmp])
                qexpr = gem.Indexed(expr, cmp)
                qexpr = gem.index_sum(gem.Indexed(qweights, point_set.indices)*qexpr,
                                      point_set.indices)
                qexprs = gem.Sum(qexprs, qexpr)
            assert qexprs.shape == ()
            assert set(qexprs.free_indices) == set(chain(shape_indices, *argument_multiindices))
            dual_expressions.append(qexprs)
        basis_indices = (gem.Index(), )
        ir = gem.Indexed(gem.ListTensor(dual_expressions), basis_indices)

    # Build kernel body
    return_indices = basis_indices + shape_indices + tuple(chain(*argument_multiindices))
    return_shape = tuple(i.extent for i in return_indices)
    return_var = gem.Variable('A', return_shape)
    if coffee:
        return_arg = ast.Decl(parameters["scalar_type"], ast.Symbol('A', rank=return_shape))
    else:
        return_arg = lp.GlobalArg("A", dtype=parameters["scalar_type"], shape=return_shape)

    return_expr = gem.Indexed(return_var, return_indices)

    # TODO: one should apply some GEM optimisations as in assembly,
    # but we don't for now.
    ir, = impero_utils.preprocess_gem([ir])
    impero_c = impero_utils.compile_gem([(return_expr, ir)], return_indices)
    index_names = dict((idx, "p%d" % i) for (i, idx) in enumerate(basis_indices))
    # Handle kernel interface requirements
    builder.register_requirements([ir])
    # Build kernel tuple
    return builder.construct_kernel(return_arg, impero_c, parameters["precision"], index_names)


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
