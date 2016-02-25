from __future__ import absolute_import

import collections
import time

from ufl.classes import Form
from ufl.algorithms import compute_form_data
from ufl.log import GREEN

from tsfc.quadrature import create_quadrature, QuadratureRule

from tsfc import fem, gem, optimise as opt, impero_utils, ufl_utils
from tsfc.coffee import generate as generate_coffee
from tsfc.constants import default_parameters
from tsfc.node import traversal
from tsfc.kernel_interface import Interface


class Kernel(object):
    __slots__ = ("ast", "integral_type", "oriented", "subdomain_id",
                 "coefficient_numbers", "__weakref__")
    """A compiled Kernel object.

    :kwarg ast: The COFFEE ast for the kernel.
    :kwarg integral_type: The type of integral.
    :kwarg oriented: Does the kernel require cell_orientations.
    :kwarg subdomain_id: What is the subdomain id for this kernel.
    :kwarg coefficient_numbers: A list of which coefficients from the
        form the kernel needs.
    """
    def __init__(self, ast=None, integral_type=None, oriented=False,
                 subdomain_id=None, coefficient_numbers=()):
        # Defaults
        self.ast = ast
        self.integral_type = integral_type
        self.oriented = oriented
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        super(Kernel, self).__init__()


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
        try:
            kernels.append(compile_integral(integral_data, fd, prefix, parameters))
        except impero_utils.NoopError:
            pass
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
    coordinates = ufl_utils.coordinate_coefficient(mesh)
    if ufl_utils.is_element_affine(mesh.ufl_coordinate_element()):
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
        interface.preload(coefficient, "w_%d" % i)

    kernel.coefficient_numbers = tuple(coefficient_numbers)

    irs = []
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

        integrand = ufl_utils.replace_coordinates(integral.integrand(), coordinates)
        ir, quadrature_index = fem.process(integral_type, cell,
                                           quad_rule, integrand,
                                           interface, index_cache)
        quadrature_indices.append(quadrature_index)
        if parameters["unroll_indexsum"]:
            ir = opt.unroll_indexsum(ir, max_extent=parameters["unroll_indexsum"])
        irs.append([(gem.IndexSum(e, quadrature_index) if quadrature_index in e.free_indices else e)
                    for e in ir])

    # Sum the expressions that are part of the same restriction
    ir = list(reduce(gem.Sum, e, gem.Zero()) for e in zip(*irs))

    # Look for cell orientations in the IR
    kernel.oriented = False
    for node in traversal(ir):
        if isinstance(node, gem.Variable) and node.name == "cell_orientations":
            kernel.oriented = True
            break

    impero_c = impero_utils.compile_gem(interface.return_variables, ir,  # TODO
                                        quadrature_indices + list(interface.argument_indices()),  # TODO
                                        coffee_licm=parameters["coffee_licm"],
                                        remove_zeros=True)

    # Generate COFFEE
    index_names = zip(interface.argument_indices(), ['j', 'k'])  # TODO
    if len(quadrature_indices) == 1:
        index_names.append((quadrature_indices[0], 'ip'))
    else:
        for i, quadrature_index in enumerate(quadrature_indices):
            index_names.append((quadrature_index, 'ip_%d' % i))

    body = generate_coffee(impero_c, index_names)
    body.open_scope = False

    funname = "%s_%s_integral_%s" % (prefix, integral_type, integral.subdomain_id())
    kernel.ast = interface.construct_kernel_function(funname, body, kernel.oriented)
    return kernel
