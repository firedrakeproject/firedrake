import collections
import time
import sys

import ufl
from ufl.algorithms import extract_coefficients
from ufl.algorithms.analysis import has_type
from ufl.algorithms.apply_coefficient_split import build_coefficient_split
from ufl.classes import Form, GeometricQuantity
from ufl.domain import extract_unique_domain, extract_domains, join_domains

import finat

from tsfc import fem, ufl_utils
from tsfc.logging import logger
from tsfc.parameters import default_parameters, is_complex
from tsfc.ufl_utils import extract_firedrake_constants
import tsfc.kernel_interface.firedrake_loopy as firedrake_interface_loopy
from tsfc.exceptions import MismatchingDomainError


# To handle big forms. The various transformations might need a deeper stack
sys.setrecursionlimit(3000)


TSFCIntegralDataInfo = collections.namedtuple("TSFCIntegralDataInfo",
                                              ["domain", "integral_type", "subdomain_id", "domain_number", "domain_integral_type_map",
                                               "arguments",
                                               "coefficients", "coefficient_split", "coefficient_numbers"])
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


def compile_form(form, prefix="form", parameters=None, dont_split_numbers=(), diagonal=False):
    """Compiles a UFL form into a set of assembly kernels.

    Parameters
    ----------
    form : ufl.classes.Form
        UFL form
    prefix : str
        Kernel name will start with this string
    parameters : dict
        Parameters object
    dont_split_numbers : tuple
        Coefficient numbers of coefficients that are not to be split into components by form compiler.
    diagonal : bool
        Are we building a kernel for the diagonal of a rank-2 element tensor?

    Returns
    -------
    list
        list of kernels

    """
    cpu_time = time.time()

    if isinstance(form, ufl.Interpolate):
        return compile_interpolate(form, prefix=prefix, parameters=parameters)

    assert isinstance(form, Form)

    GREEN = "\033[1;37;32m%s\033[0m"

    # Determine whether in complex mode:
    complex_mode = parameters and is_complex(parameters.get("scalar_type"))
    form_data = ufl_utils.compute_form_data(
        form,
        coefficients_to_split=tuple(
            c
            for i, c in enumerate(form.coefficients())
            if type(c.ufl_element()) == finat.ufl.MixedElement and i not in dont_split_numbers
        ),
        complex_mode=complex_mode,
    )
    logger.info(GREEN % "compute_form_data finished in %g seconds.", time.time() - cpu_time)

    validate_domains(form_data.preprocessed_form)

    # Create local kernels.
    kernels = []
    for integral_data in form_data.integral_data:
        start = time.time()
        if integral_data.integrals:
            kernel = compile_integral(integral_data, form_data, prefix, parameters, diagonal=diagonal)
            if kernel is not None:
                kernels.append(kernel)
        logger.info(GREEN % "compile_integral finished in %g seconds.", time.time() - start)

    logger.info(GREEN % "TSFC finished in %g seconds.", time.time() - cpu_time)
    return kernels


def make_kernel_builder(integral_data_info, constants, parameters, diagonal=False):
    """Create a kernel builder holding every mesh quantity its integral may read."""
    builder = firedrake_interface_loopy.KernelBuilder(integral_data_info, parameters["scalar_type"], diagonal=diagonal)
    domains = tuple(integral_data_info.domain_integral_type_map)
    builder.set_entity_numbers(domains)
    builder.set_entity_orientations(domains)
    builder.set_coordinates(domains)
    builder.set_cell_orientations(domains)
    builder.set_cell_sizes(domains)
    builder.set_coefficients()
    # TODO: We do not want pass constants to kernels that do not need them
    # so we should attach the constants to integral data instead
    builder.set_constants(constants)
    return builder


def compile_integral(integral_data, form_data, prefix, parameters, *, diagonal=False):
    """Compiles a UFL integral into an assembly kernel.

    :arg integral_data: UFL integral data
    :arg form_data: UFL form data
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :arg diagonal: Are we building a kernel for the diagonal of a rank-2 element tensor?
    :returns: a kernel constructed by the kernel interface
    """
    parameters = preprocess_parameters(parameters)
    integral_type = integral_data.integral_type
    arguments = form_data.preprocessed_form.arguments()
    if integral_type.startswith("interior_facet") and diagonal and any(a.function_space().finat_element.is_dg() for a in arguments):
        raise NotImplementedError("Sorry, we can't assemble the diagonal of a form for interior facet integrals")
    kernel_name = f"{prefix}_{integral_type}_integral"
    # This is which coefficient in the original form the
    # current coefficient is.
    # Consider f*v*dx + g*v*ds, the full form contains two
    # coefficients, but each integral only requires one.
    coefficients = []
    coefficient_split = {}
    coefficient_numbers = []
    for i, (coeff_orig, enabled) in enumerate(zip(form_data.reduced_coefficients, integral_data.enabled_coefficients)):
        if enabled:
            coeff = form_data.function_replace_map[coeff_orig]
            coefficients.append(coeff)
            if coeff in form_data.coefficient_split:
                coefficient_split[coeff] = form_data.coefficient_split[coeff]
            coefficient_numbers.append(form_data.original_coefficient_positions[i])
    mesh = integral_data.domain
    all_meshes = extract_domains(form_data.original_form)
    domain_number = all_meshes.index(mesh)

    integral_data_info = TSFCIntegralDataInfo(
        domain=integral_data.domain,
        integral_type=integral_data.integral_type,
        subdomain_id=integral_data.subdomain_id,
        domain_number=domain_number,
        domain_integral_type_map={mesh: integral_data.domain_integral_type_map.get(mesh, None) for mesh in all_meshes},
        arguments=arguments,
        coefficients=coefficients,
        coefficient_split=coefficient_split,
        coefficient_numbers=coefficient_numbers,
    )
    builder = make_kernel_builder(integral_data_info, form_data.constants, parameters, diagonal=diagonal)
    ctx = builder.create_context()
    for integral in integral_data.integrals:
        params = parameters.copy()
        params.update(integral.metadata())  # integral metadata overrides
        integrand_exprs = builder.compile_integrand(integral.integrand(), params, ctx)
        integral_exprs = builder.construct_integrals(integrand_exprs, params)
        builder.stash_integrals(integral_exprs, params, ctx)
    return builder.construct_kernel(kernel_name, ctx, parameters["add_petsc_events"])


def validate_domains(form):
    if len(extract_domains(form)) == 1:
        # Not a multi-domain form, we do not need to keep checking
        return

    for itg in form.integrals():
        # Check that all domains are related to each other
        domain = itg.ufl_domain()
        for other_domain in itg.extra_domain_integral_type_map():
            if domain.submesh_youngest_common_ancestor(other_domain) is None:
                raise MismatchingDomainError("Assembly of forms over unrelated meshes is not supported. "
                                             "Try using Submeshes or cross-mesh interpolation.")

        # Check that all Arguments and Coefficients are defined on the valid domains
        valid_domains = set(itg.extra_domain_integral_type_map())
        valid_domains.add(domain)

        itg_domains = set(extract_domains(itg))
        if len(itg_domains - valid_domains) > 0:
            raise MismatchingDomainError("Argument or Coefficient domain not found in integral. "
                                         "Possibly, the form contains coefficients on different meshes "
                                         "and requires measure intersection, for example: "
                                         'Measure("dx", argument_mesh, intersect_measures=[Measure("dx", coefficient_mesh)]).')


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


def _preprocess_interpolate(expression, target_element, parameters, *, domain=None):
    """Preprocess an interpolation and construct its kernel data."""
    dual_arg, operand = expression.argument_slots()
    target_domains = join_domains([dual_arg.ufl_function_space().ufl_domain()])
    if len(target_domains) != 1:
        raise NotImplementedError("Interpolation onto multiple distinct meshes is not supported")
    target_domain, = target_domains
    source_domain = domain or extract_unique_domain(operand) or target_domain
    if target_domain.topological_dimension == 0 and source_domain.topological_dimension > 0:
        target_element = ufl_utils.runtime_quadrature_element(source_domain, target_element)

    original_coefficients = extract_coefficients(expression)
    expression = ufl_utils.preprocess_interpolate(
        expression, target_element, source_domain, is_complex(parameters["scalar_type"])
    )
    coefficients = extract_coefficients(expression)
    integral_data_info = TSFCIntegralDataInfo(
        domain=source_domain,
        integral_type="cell",
        subdomain_id=("everywhere",),
        domain_number=expression.ufl_domains().index(target_domain),
        domain_integral_type_map={domain: "cell" for domain in extract_domains(operand)},
        arguments=expression.arguments(),
        coefficients=coefficients,
        coefficient_split=build_coefficient_split(
            c for c in coefficients if type(c.ufl_element()) is finat.ufl.MixedElement
        ),
        coefficient_numbers=tuple(map(original_coefficients.index, coefficients)),
    )
    return expression, target_element, integral_data_info


def _lower_interpolate(builder, expression, target_element, parameters):
    """Lower an interpolation through a kernel builder."""
    ctx = builder.create_context()
    reps = builder.compile_interpolate(expression, target_element, parameters, ctx)
    builder.stash_integrals(reps, parameters, ctx)
    return ctx


def compile_expression_dual_evaluation(expression, ufl_element, *,
                                       domain=None, interface=None,
                                       parameters=None, name=None):
    """Compile a UFL expression to be evaluated against a compile-time known reference element's dual basis.

    Useful for interpolating UFL expressions into e.g. N1curl spaces.

    :arg expression: UFL expression
    :arg ufl_element: The UFL element of the target space.
    :arg domain: optional UFL domain the expression is defined on (required when expression contains no domain).
    :arg interface: backend module for the kernel interface
    :arg parameters: parameters object
    :returns: Loopy-based ExpressionKernel object.
    """
    parameters = preprocess_parameters(parameters)
    if not isinstance(expression, ufl.Interpolate):
        V = ufl.FunctionSpace(extract_unique_domain(expression) or domain, ufl_element)
        expression = ufl.Interpolate(expression, V)
    expression, ufl_element, integral_data_info = _preprocess_interpolate(
        expression, ufl_element, parameters, domain=domain
    )
    domain = integral_data_info.domain

    if interface is None:
        # Delayed import, loopy is a runtime dependency
        from tsfc.kernel_interface.firedrake_loopy import ExpressionKernelBuilder as interface

    coefficients = integral_data_info.coefficients
    builder = interface(parameters["scalar_type"], integral_data_info)

    elements = [f.ufl_element() for f in (*coefficients, *integral_data_info.arguments)]
    needs_external_coords = bool(has_type(expression, GeometricQuantity)
                                 or any(map(fem.needs_coordinate_mapping, elements)))
    if needs_external_coords:
        # Create a fake coordinate coefficient for a domain.
        coords_coefficient = ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))
        builder.domain_coordinate[domain] = coords_coefficient
        builder.set_cell_orientations((domain, ))
        builder.set_cell_sizes((domain, ))
        coefficients = [coords_coefficient, *coefficients]
    builder.set_coefficients(coefficients)
    builder.set_constants(extract_firedrake_constants(expression))

    ctx = _lower_interpolate(builder, expression, ufl_element, parameters)
    return builder.construct_kernel(name, ctx, needs_external_coords, parameters["add_petsc_events"])


def compile_interpolate(expression, prefix="interpolate", parameters=None):
    """Compile a UFL interpolation into an assembly kernel.

    Parameters
    ----------
    expression : ufl.Interpolate
        The interpolation to compile.
    prefix : str
        Kernel name will start with this string.
    parameters : dict
        Parameters object.

    Returns
    -------
    list
        A single-element list holding the kernel.

    """
    parameters = preprocess_parameters(parameters)
    target_element = expression.ufl_element()
    expression, target_element, integral_data_info = _preprocess_interpolate(
        expression, target_element, parameters,
    )
    builder = make_kernel_builder(integral_data_info, extract_firedrake_constants(expression), parameters)
    ctx = _lower_interpolate(builder, expression, target_element, parameters)
    return [builder.construct_kernel(f"{prefix}_cell_integral", ctx, parameters["add_petsc_events"])]
