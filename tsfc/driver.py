import collections
import time
import sys
import numpy
from itertools import chain
from finat.physically_mapped import DirectlyDefinedElement, PhysicallyMappedElement

import ufl
from ufl.algorithms import extract_coefficients
from ufl.algorithms.analysis import has_type
from ufl.algorithms.apply_coefficient_split import CoefficientSplitter
from ufl.classes import Form, GeometricQuantity
from ufl.domain import extract_unique_domain, extract_domains

import gem
import gem.impero_utils as impero_utils

import finat
from finat.element_factory import as_fiat_cell

from tsfc import fem, ufl_utils
from tsfc.logging import logger
from tsfc.parameters import default_parameters, is_complex
from tsfc.ufl_utils import apply_mapping, extract_firedrake_constants
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
    scalar_type = parameters["scalar_type"]
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

    builder = firedrake_interface_loopy.KernelBuilder(
        integral_data_info,
        scalar_type,
        diagonal=diagonal,
    )
    builder.set_entity_numbers(all_meshes)
    builder.set_entity_orientations(all_meshes)
    builder.set_coordinates(all_meshes)
    builder.set_cell_orientations(all_meshes)
    builder.set_cell_sizes(all_meshes)
    builder.set_coefficients()
    # TODO: We do not want pass constants to kernels that do not need them
    # so we should attach the constants to integral data instead
    builder.set_constants(form_data.constants)
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
            if domain.submesh_youngest_common_ancester(other_domain) is None:
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

    orig_coefficients = extract_coefficients(expression)
    if isinstance(expression, ufl.Interpolate):
        v, operand = expression.argument_slots()
    else:
        operand = expression
        v = ufl.FunctionSpace(extract_unique_domain(operand), ufl_element)

    # Map into reference space
    operand = apply_mapping(operand, ufl_element, domain)

    # Apply UFL preprocessing
    operand = ufl_utils.preprocess_expression(operand, complex_mode=complex_mode)

    # Reconstructed Interpolate with mapped operand
    expression = ufl.Interpolate(operand, v)

    # Initialise kernel builder
    if interface is None:
        # Delayed import, loopy is a runtime dependency
        from tsfc.kernel_interface.firedrake_loopy import ExpressionKernelBuilder as interface

    builder = interface(parameters["scalar_type"])
    arguments = expression.arguments()
    argument_multiindices = {arg.number(): builder.create_element(arg.ufl_element()).get_indices()
                             for arg in arguments}
    assert len(argument_multiindices) == len(arguments)

    # Replace coordinates (if any) unless otherwise specified by kwarg
    if domain is None:
        domain = extract_unique_domain(expression)
    assert domain is not None
    builder._domain_integral_type_map = {domain: "cell"}
    builder._entity_ids = {domain: (0,)}

    # Collect required coefficients and determine numbering
    coefficients = extract_coefficients(expression)
    coefficient_numbers = tuple(map(orig_coefficients.index, coefficients))
    builder.set_coefficient_numbers(coefficient_numbers)
    # Need this ad-hoc fix for now.
    for c in coefficients:
        d = extract_unique_domain(c)
        builder._domain_integral_type_map[d] = "cell"

    elements = [f.ufl_element() for f in (*coefficients, *arguments)]

    needs_external_coords = False
    if has_type(expression, GeometricQuantity) or any(map(fem.needs_coordinate_mapping, elements)):
        # Create a fake coordinate coefficient for a domain.
        coords_coefficient = ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))
        builder.domain_coordinate[domain] = coords_coefficient
        builder.set_cell_orientations((domain, ))
        builder.set_cell_sizes((domain, ))
        coefficients = [coords_coefficient] + coefficients
        needs_external_coords = True
    builder.set_coefficients(coefficients)

    constants = extract_firedrake_constants(expression)
    builder.set_constants(constants)

    # Split mixed coefficients
    coeff_splitter = CoefficientSplitter(builder.coefficient_split)
    expression = coeff_splitter(expression)

    # Set up kernel config for translation of UFL expression to gem
    kernel_cfg = dict(interface=builder,
                      ufl_cell=domain.ufl_cell(),
                      integration_dim=as_fiat_cell(domain.ufl_cell()).get_dimension(),
                      # FIXME: change if we ever implement
                      # interpolation on facets.
                      argument_multiindices=argument_multiindices,
                      index_cache={},
                      scalar_type=parameters["scalar_type"])

    # Allow interpolation onto QuadratureElements to refer to the quadrature
    # rule they represent
    if isinstance(to_element, finat.QuadratureElement):
        kernel_cfg["quadrature_rule"] = to_element._rule

    dual_arg, operand = expression.argument_slots()

    # Create callable for translation of UFL expression to gem
    fn = DualEvaluationCallable(operand, kernel_cfg)

    # Get the gem expression for dual evaluation and corresponding basis
    # indices needed for compilation of the expression
    evaluation, basis_indices = to_element.dual_evaluation(fn)

    # Compute the action against the dual argument
    if dual_arg in coefficients:
        name = f"w_{coefficients.index(dual_arg)}"
        shape = tuple(i.extent for i in basis_indices)
        size = numpy.prod(shape, dtype=int)
        gem_dual = gem.reshape(gem.Variable(name, shape=(size,)), shape)
        if complex_mode:
            evaluation = gem.MathFunction('conj', evaluation)
        evaluation = gem.IndexSum(evaluation * gem_dual[basis_indices], basis_indices)
        basis_indices = ()
    else:
        argument_multiindices[dual_arg.number()] = basis_indices

    argument_multiindices = dict(sorted(argument_multiindices.items()))

    # Build kernel body
    return_indices = tuple(chain.from_iterable(argument_multiindices.values()))
    return_shape = tuple(i.extent for i in return_indices)
    return_var = gem.Variable('A', return_shape or (1,))
    return_expr = gem.Indexed(return_var, return_indices or (0,))

    # TODO: one should apply some GEM optimisations as in assembly,
    # but we don't for now.
    evaluation, = impero_utils.preprocess_gem([evaluation])
    impero_c = impero_utils.compile_gem([(return_expr, evaluation)], return_indices)
    index_names = {idx: f"p{i}" for (i, idx) in enumerate(basis_indices)}
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
        argument_multiindices = kernel_cfg["argument_multiindices"].values()
        assert set(gem_expr.free_indices) <= set(chain(ps.indices, *argument_multiindices))

        return gem_expr
