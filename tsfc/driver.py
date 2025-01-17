import collections
import time
import sys
from itertools import chain
from finat.physically_mapped import DirectlyDefinedElement, PhysicallyMappedElement

import ufl
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.algorithms.analysis import has_type
from ufl.classes import Form, GeometricQuantity
from ufl.domain import extract_unique_domain

import gem
import gem.impero_utils as impero_utils

import finat

from tsfc import fem, ufl_utils
from tsfc.logging import logger
from tsfc.parameters import default_parameters, is_complex
from tsfc.ufl_utils import apply_mapping, extract_firedrake_constants
import tsfc.kernel_interface.firedrake_loopy as firedrake_interface_loopy

# To handle big forms. The various transformations might need a deeper stack
sys.setrecursionlimit(3000)


TSFCIntegralDataInfo = collections.namedtuple("TSFCIntegralDataInfo",
                                              ["domain", "integral_type", "subdomain_id", "domain_number",
                                               "arguments",
                                               "coefficients", "coefficient_numbers"])
TSFCIntegralDataInfo.__doc__ = """
    Minimal set of objects for kernel builders.

    domain - The mesh.
    integral_type - The type of integral.
    subdomain_id - What is the subdomain id for this kernel.
    domain_number - Which domain number in the original form
        does this kernel correspond to (can be used to index into
        original_form.ufl_domains() to get the correct domain).
    coefficients - A list of coefficients.
    coefficient_numbers - A list of which coefficients from the
        form the kernel needs.

    This is a minimal set of objects that kernel builders need to
    construct a kernel from :attr:`integrals` of :class:`~ufl.IntegralData`.
    """


def compile_form(form, prefix="form", parameters=None, interface=None, diagonal=False):
    """Compiles a UFL form into a set of assembly kernels.

    :arg form: UFL form
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :arg diagonal: Are we building a kernel for the diagonal of a rank-2 element tensor?
    :returns: list of kernels
    """
    cpu_time = time.time()

    assert isinstance(form, Form)

    GREEN = "\033[1;37;32m%s\033[0m"

    # Determine whether in complex mode:
    complex_mode = parameters and is_complex(parameters.get("scalar_type"))
    fd = ufl_utils.compute_form_data(form, complex_mode=complex_mode)
    logger.info(GREEN % "compute_form_data finished in %g seconds.", time.time() - cpu_time)

    kernels = []
    for integral_data in fd.integral_data:
        start = time.time()
        kernel = compile_integral(integral_data, fd, prefix, parameters, interface=interface, diagonal=diagonal)
        if kernel is not None:
            kernels.append(kernel)
        logger.info(GREEN % "compile_integral finished in %g seconds.", time.time() - start)

    logger.info(GREEN % "TSFC finished in %g seconds.", time.time() - cpu_time)
    return kernels


def compile_integral(integral_data, form_data, prefix, parameters, interface, *, diagonal=False):
    """Compiles a UFL integral into an assembly kernel.

    :arg integral_data: UFL integral data
    :arg form_data: UFL form data
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :arg interface: backend module for the kernel interface
    :arg diagonal: Are we building a kernel for the diagonal of a rank-2 element tensor?
    :returns: a kernel constructed by the kernel interface
    """
    parameters = preprocess_parameters(parameters)
    if interface is None:
        interface = firedrake_interface_loopy.KernelBuilder
    scalar_type = parameters["scalar_type"]
    integral_type = integral_data.integral_type
    if integral_type.startswith("interior_facet") and diagonal:
        raise NotImplementedError("Sorry, we can't assemble the diagonal of a form for interior facet integrals")
    mesh = integral_data.domain
    arguments = form_data.preprocessed_form.arguments()
    kernel_name = f"{prefix}_{integral_type}_integral"
    # Dict mapping domains to index in original_form.ufl_domains()
    domain_numbering = form_data.original_form.domain_numbering()
    domain_number = domain_numbering[integral_data.domain]
    coefficients = [form_data.function_replace_map[c] for c in integral_data.integral_coefficients]
    # This is which coefficient in the original form the
    # current coefficient is.
    # Consider f*v*dx + g*v*ds, the full form contains two
    # coefficients, but each integral only requires one.
    coefficient_numbers = tuple(form_data.original_coefficient_positions[i]
                                for i, (_, enabled) in enumerate(zip(form_data.reduced_coefficients, integral_data.enabled_coefficients))
                                if enabled)
    integral_data_info = TSFCIntegralDataInfo(domain=integral_data.domain,
                                              integral_type=integral_data.integral_type,
                                              subdomain_id=integral_data.subdomain_id,
                                              domain_number=domain_number,
                                              arguments=arguments,
                                              coefficients=coefficients,
                                              coefficient_numbers=coefficient_numbers)
    builder = interface(integral_data_info,
                        scalar_type,
                        diagonal=diagonal)
    builder.set_coordinates(mesh)
    builder.set_cell_sizes(mesh)
    builder.set_coefficients(integral_data, form_data)
    # TODO: We do not want pass constants to kernels that do not need them
    # so we should attach the constants to integral data instead
    builder.set_constants(form_data.constants)
    ctx = builder.create_context()
    for integral in integral_data.integrals:
        params = parameters.copy()
        params.update(integral.metadata())  # integral metadata overrides
        integrand = ufl.replace(integral.integrand(), form_data.function_replace_map)
        integrand_exprs = builder.compile_integrand(integrand, params, ctx)
        integral_exprs = builder.construct_integrals(integrand_exprs, params)
        builder.stash_integrals(integral_exprs, params, ctx)
    return builder.construct_kernel(kernel_name, ctx, parameters["add_petsc_events"])


def preprocess_parameters(parameters):
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _
    # Remove these here, they're handled later on.
    if parameters.get("quadrature_degree") in ["auto", "default", None, -1, "-1"]:
        del parameters["quadrature_degree"]
    if parameters.get("quadrature_rule") in ["auto", "default", None]:
        del parameters["quadrature_rule"]
    return parameters


def compile_expression_dual_evaluation(expression, to_element, ufl_element, *,
                                       domain=None, interface=None,
                                       parameters=None):
    """Compile a UFL expression to be evaluated against a compile-time known reference element's dual basis.

    Useful for interpolating UFL expressions into e.g. N1curl spaces.

    :arg expression: UFL expression
    :arg to_element: A FInAT element for the target space
    :arg ufl_element: The UFL element of the target space.
    :arg domain: optional UFL domain the expression is defined on (required when expression contains no domain).
    :arg interface: backend module for the kernel interface
    :arg parameters: parameters object
    :returns: Loopy-based ExpressionKernel object.
    """
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    # Determine whether in complex mode
    complex_mode = is_complex(parameters["scalar_type"])

    if isinstance(to_element, (PhysicallyMappedElement, DirectlyDefinedElement)):
        raise NotImplementedError("Don't know how to interpolate onto zany spaces, sorry")

    orig_expression = expression

    # Map into reference space
    expression = apply_mapping(expression, ufl_element, domain)

    # Apply UFL preprocessing
    expression = ufl_utils.preprocess_expression(expression,
                                                 complex_mode=complex_mode)

    # Initialise kernel builder
    if interface is None:
        # Delayed import, loopy is a runtime dependency
        from tsfc.kernel_interface.firedrake_loopy import ExpressionKernelBuilder as interface

    builder = interface(parameters["scalar_type"])
    arguments = extract_arguments(expression)
    argument_multiindices = tuple(builder.create_element(arg.ufl_element()).get_indices()
                                  for arg in arguments)

    # Replace coordinates (if any) unless otherwise specified by kwarg
    if domain is None:
        domain = extract_unique_domain(expression)
    assert domain is not None

    # Collect required coefficients and determine numbering
    coefficients = extract_coefficients(expression)
    orig_coefficients = extract_coefficients(orig_expression)
    coefficient_numbers = tuple(orig_coefficients.index(c) for c in coefficients)
    builder.set_coefficient_numbers(coefficient_numbers)

    needs_external_coords = False
    if has_type(expression, GeometricQuantity) or any(fem.needs_coordinate_mapping(c.ufl_element()) for c in coefficients):
        # Create a fake coordinate coefficient for a domain.
        coords_coefficient = ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))
        builder.domain_coordinate[domain] = coords_coefficient
        builder.set_cell_sizes(domain)
        coefficients = [coords_coefficient] + coefficients
        needs_external_coords = True
    builder.set_coefficients(coefficients)

    constants = extract_firedrake_constants(expression)
    builder.set_constants(constants)

    # Split mixed coefficients
    expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Set up kernel config for translation of UFL expression to gem
    kernel_cfg = dict(interface=builder,
                      ufl_cell=domain.ufl_cell(),
                      # FIXME: change if we ever implement
                      # interpolation on facets.
                      integral_type="cell",
                      argument_multiindices=argument_multiindices,
                      index_cache={},
                      scalar_type=parameters["scalar_type"])

    # Allow interpolation onto QuadratureElements to refer to the quadrature
    # rule they represent
    if isinstance(to_element, finat.QuadratureElement):
        kernel_cfg["quadrature_rule"] = to_element._rule

    # Create callable for translation of UFL expression to gem
    fn = DualEvaluationCallable(expression, kernel_cfg)

    # Get the gem expression for dual evaluation and corresponding basis
    # indices needed for compilation of the expression
    evaluation, basis_indices = to_element.dual_evaluation(fn)

    # Build kernel body
    return_indices = basis_indices + tuple(chain(*argument_multiindices))
    return_shape = tuple(i.extent for i in return_indices)
    return_var = gem.Variable('A', return_shape)
    return_expr = gem.Indexed(return_var, return_indices)

    # TODO: one should apply some GEM optimisations as in assembly,
    # but we don't for now.
    evaluation, = impero_utils.preprocess_gem([evaluation])
    impero_c = impero_utils.compile_gem([(return_expr, evaluation)], return_indices)
    index_names = dict((idx, "p%d" % i) for (i, idx) in enumerate(basis_indices))
    # Handle kernel interface requirements
    builder.register_requirements([evaluation])
    builder.set_output(return_var)
    # Build kernel tuple
    return builder.construct_kernel(impero_c, index_names, needs_external_coords, parameters["add_petsc_events"])


class DualEvaluationCallable(object):
    """
    Callable representing a function to dual evaluate.

    When called, this takes in a
    :class:`finat.point_set.AbstractPointSet` and returns a GEM
    expression for evaluation of the function at those points.

    :param expression: UFL expression for the function to dual evaluate.
    :param kernel_cfg: A kernel configuration for creation of a
        :class:`GemPointContext` or a :class:`PointSetContext`

    Not intended for use outside of
    :func:`compile_expression_dual_evaluation`.
    """
    def __init__(self, expression, kernel_cfg):
        self.expression = expression
        self.kernel_cfg = kernel_cfg

    def __call__(self, ps):
        """The function to dual evaluate.

        :param ps: The :class:`finat.point_set.AbstractPointSet` for
            evaluating at
        :returns: a gem expression representing the evaluation of the
            input UFL expression at the given point set ``ps``.
            For point set points with some shape ``(*value_shape)``
            (i.e. ``()`` for scalar points ``(x)`` for vector points
            ``(x, y)`` for tensor points etc) then the gem expression
            has shape ``(*value_shape)`` and free indices corresponding
            to the input :class:`finat.point_set.AbstractPointSet`'s
            free indices alongside any input UFL expression free
            indices.
        """

        if not isinstance(ps, finat.point_set.AbstractPointSet):
            raise ValueError("Callable argument not a point set!")

        # Avoid modifying saved kernel config
        kernel_cfg = self.kernel_cfg.copy()

        if isinstance(ps, finat.point_set.UnknownPointSet):
            # Run time known points
            kernel_cfg.update(point_indices=ps.indices, point_expr=ps.expression)
            # GemPointContext's aren't allowed to have quadrature rules
            kernel_cfg.pop("quadrature_rule", None)
            translation_context = fem.GemPointContext(**kernel_cfg)
        else:
            # Compile time known points
            kernel_cfg.update(point_set=ps)
            translation_context = fem.PointSetContext(**kernel_cfg)

        gem_expr, = fem.compile_ufl(self.expression, translation_context, point_sum=False)
        # In some cases ps.indices may be dropped from expr, but nothing
        # new should now appear
        argument_multiindices = kernel_cfg["argument_multiindices"]
        assert set(gem_expr.free_indices) <= set(chain(ps.indices, *argument_multiindices))

        return gem_expr
