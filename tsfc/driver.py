from __future__ import absolute_import

import collections
import time

import numpy

from ufl.classes import Form
from ufl.algorithms import compute_form_data
from ufl.log import GREEN

from tsfc.quadrature import create_quadrature, QuadratureRule

from tsfc import fem, gem, scheduling as sch, optimise as opt, impero_utils
from tsfc.coffee import generate as generate_coffee
from tsfc.constants import default_parameters
from tsfc.node import traversal
from tsfc.kernel_interface import Kernel, Interface


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
                           do_estimate_degrees=True)
    print GREEN % ("compute_form_data finished in %g seconds." % (time.time() - cpu_time))

    kernels = []
    for integral_data in fd.integral_data:
        start = time.time()
        kernel = compile_integral(integral_data, fd, prefix, parameters)
        if kernel is not None:
            kernels.append(kernel)
        print GREEN % ("compile_integral finished in %g seconds." % (time.time() - start))

    print GREEN % ("TSFC finished in %g seconds." % (time.time() - cpu_time))
    return kernels


def compile_integral(integral_data, form_data, prefix, parameters):
    """Compiles a UFL integral into an assembly kernel.

    :arg integral_data: UFL integral data
    :arg form_data: UFL form data
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :returns: a kernel, or None if the integral simplifies to zero
    """
    # Remove these here, they're handled below.
    if parameters.get("quadrature_degree") == "auto":
        del parameters["quadrature_degree"]
    if parameters.get("quadrature_rule") == "auto":
        del parameters["quadrature_rule"]

    integral_type = integral_data.integral_type
    kernel = Kernel(integral_type=integral_type, subdomain_id=integral_data.subdomain_id)
    interface = Interface(integral_type, form_data.preprocessed_form.arguments())

    mesh = integral_data.domain
    coordinates = fem.coordinate_coefficient(mesh)
    if is_mesh_affine(mesh):
        # For affine mesh geometries we prefer code generation that
        # composes well with optimisations.
        interface.preload(coordinates, "coords", mode='list_tensor')
    else:
        # Otherwise we use the approach that might be faster (?)
        interface.preload(coordinates, "coords")

    coefficient_numbers = []
    # enabled_coefficients is a boolean array that indicates which of
    # reduced_coefficients the integral requires.
    for i, on in enumerate(integral_data.enabled_coefficients):
        if not on:
            continue
        coefficient = form_data.reduced_coefficients[i]
        # This is which coefficient in the original form the current
        # coefficient is.
        # Consider f*v*dx + g*v*ds, the full form contains two
        # coefficients, but each integral only requires one.
        coefficient_numbers.append(form_data.original_coefficient_positions[i])
        interface.preload(coefficient, "w_%d" % i)  # TODO: Remove!

    kernel.coefficient_numbers = tuple(coefficient_numbers)

    nonfem_ = []
    quadrature_indices = []
    cell = integral_data.domain.ufl_cell()
    # Map from UFL FiniteElement objects to Index instances.  This is
    # so we reuse Index instances when evaluating the same coefficient
    # multiple times with the same table.  Occurs, for example, if we
    # have multiple integrals here (and the affine coordinate
    # evaluation can be hoisted).
    index_cache = collections.defaultdict(gem.Index)
    for i, integral in enumerate(integral_data.integrals):
        params = {}
        # Record per-integral parameters
        params.update(integral.metadata())
        # parameters override per-integral metadata
        params.update(parameters)

        # Check if the integral has a quad degree attached, otherwise use
        # the estimated polynomial degree attached by compute_form_data
        quadrature_degree = params.get("quadrature_degree",
                                       params["estimated_polynomial_degree"])
        quad_rule = params.get("quadrature_rule",
                               create_quadrature(cell, integral_type,
                                                 quadrature_degree))

        if not isinstance(quad_rule, QuadratureRule):
            raise ValueError("Expected to find a QuadratureRule object, not a %s" %
                             type(quad_rule))

        tabulation_manager = fem.TabulationManager(integral_type, cell,
                                                   quad_rule.points)

        integrand = fem.replace_coordinates(integral.integrand(), coordinates)
        quadrature_index = gem.Index(name="ip%d" % i)
        quadrature_indices.append(quadrature_index)
        nonfem = fem.process(integral_type, integrand,
                             tabulation_manager, quad_rule.weights,
                             quadrature_index, interface.argument_indices(),  # TODO
                             interface.coefficients, index_cache)  # TODO
        if parameters["unroll_indexsum"]:
            nonfem = opt.unroll_indexsum(nonfem, max_extent=parameters["unroll_indexsum"])
        nonfem_.append([(gem.IndexSum(e, quadrature_index) if quadrature_index in e.free_indices else e)
                        for e in nonfem])

    # Sum the expressions that are part of the same restriction
    nonfem = list(reduce(gem.Sum, e, gem.Zero()) for e in zip(*nonfem_))

    index_names = zip(interface.argument_indices(), ['j', 'k'])  # TODO
    if len(quadrature_indices) == 1:
        index_names.append((quadrature_indices[0], 'ip'))
    else:
        for i, quadrature_index in enumerate(quadrature_indices):
            index_names.append((quadrature_index, 'ip_%d' % i))

    body, kernel.oriented = build_kernel_body(interface.return_variables, nonfem,  # TODO
                                              quadrature_indices + list(interface.argument_indices()),  # TODO
                                              coffee_licm=parameters["coffee_licm"],
                                              index_names=index_names)
    if body is None:
        return None

    funname = "%s_%s_integral_%s" % (prefix, integral_type, integral.subdomain_id())
    kernel.ast = interface.construct_kernel_function(funname, body, kernel.oriented)
    return kernel


def build_kernel_body(return_variables, ir, prefix_ordering, coffee_licm=False, index_names=None):
    ir = opt.remove_componenttensors(ir)

    # Look for cell orientations in the simplified GEM
    oriented = False
    for node in traversal(ir):
        if isinstance(node, gem.Variable) and node.name == "cell_orientations":
            oriented = True
            break

    # Collect indices in a deterministic order
    indices = []
    for node in traversal(ir):
        if isinstance(node, gem.Indexed):
            indices.extend(node.multiindex)
    # The next two lines remove duplicate elements from the list, but
    # preserve the ordering, i.e. all elements will appear only once,
    # in the order of their first occurance in the original list.
    _, unique_indices = numpy.unique(indices, return_index=True)
    indices = numpy.asarray(indices)[numpy.sort(unique_indices)]

    # Build ordered index map
    index_ordering = make_prefix_ordering(indices, prefix_ordering)
    apply_ordering = make_index_orderer(index_ordering)

    get_indices = lambda expr: apply_ordering(expr.free_indices)

    # Build operation ordering
    ops = sch.emit_operations(zip(return_variables, ir), get_indices)

    # Zero-simplification occurred
    if len(ops) == 0:
        return None, False

    # Drop unnecessary temporaries
    ops = impero_utils.inline_temporaries(ir, ops, coffee_licm=coffee_licm)

    # Prepare ImperoC (Impero AST + other data for code generation)
    impero_c = impero_utils.process(ops, get_indices)

    # Generate COFFEE
    if index_names is None:
        index_names = {}
    body = generate_coffee(impero_c, index_names)
    body.open_scope = False
    return body, oriented


def is_mesh_affine(mesh):
    """Tells if a mesh geometry is affine."""
    affine_cells = ["interval", "triangle", "tetrahedron"]
    degree = mesh.ufl_coordinate_element().degree()
    return mesh.ufl_cell().cellname() in affine_cells and degree == 1


def make_prefix_ordering(indices, prefix_ordering):
    """Creates an ordering of ``indices`` which starts with those
    indices in ``prefix_ordering``."""
    # Need to return deterministically ordered indices
    return tuple(prefix_ordering) + tuple(k for k in indices if k not in prefix_ordering)


def make_index_orderer(index_ordering):
    """Returns a function which given a set of indices returns those
    indices in the order as they appear in ``index_ordering``."""
    idx2pos = {idx: pos for pos, idx in enumerate(index_ordering)}

    def apply_ordering(indices):
        return tuple(sorted(indices, key=lambda i: idx2pos[i]))
    return apply_ordering
