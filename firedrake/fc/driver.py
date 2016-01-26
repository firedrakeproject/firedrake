from __future__ import absolute_import

import itertools
import time

import numpy

from ufl.algorithms import compute_form_data

from ffc.log import info_green
from ffc.fiatinterface import create_element
from ffc.mixedelement import MixedElement as ffc_MixedElement
from ffc.representationutils import create_quadrature_points_and_weights

import coffee.base as coffee

from firedrake.fc import fem, einstein as ein, impero as imp, scheduling as sch
from firedrake.fc.coffee import SCALAR_TYPE, generate as generate_coffee


def compile_form(form, prefix="form", parameters=None):
    assert not isinstance(form, (list, tuple))

    fd = compute_form_data(form,
                           do_apply_function_pullbacks=True,
                           do_apply_integral_scaling=True,
                           do_apply_geometry_lowering=True,
                           do_apply_restrictions=True)

    kernels = []
    for integral in fd.preprocessed_form.integrals():
        kernels.append(compile_integral(integral, fd, prefix))
    return kernels


def compile_integral(integral, fd, prefix):
    cpu_time = time.time()

    quadrature_degree = integral.metadata()["quadrature_degree"]
    integral_type = integral.integral_type()
    integrand = integral.integrand()

    arglist = []
    coefficient_map = {}

    arguments = fd.preprocessed_form.arguments()
    output_shape = [create_element(arg.ufl_element()).space_dimension() for arg in arguments]
    # Interior facets, doubled up output size
    if integral_type.startswith("interior_facet"):
        output_shape = [s*2 for s in output_shape]

    output_shape = tuple(output_shape)
    if output_shape == ():
        output_shape = (1,)

    output_symbol = coffee.Symbol("A", rank=output_shape)
    output_tensor = coffee.Decl(SCALAR_TYPE, output_symbol)

    arglist.insert(0, output_tensor)
    argument_indices = tuple(ein.Index() for i in range(len(arguments)))
    output_indices = tuple(ein.Index() for i in range(len(arguments)))
    output_arg = ein.Variable("A", output_shape)
    if arguments:
        output_arg = ein.Indexed(output_arg, output_indices)
    else:
        output_arg = ein.Indexed(output_arg, (0,))

    mesh = fd.preprocessed_form.ufl_domain()
    fiat_element = create_element(mesh.ufl_coordinate_element())
    arglist.append(coffee.Decl("const %s *restrict" % SCALAR_TYPE,
                               coffee.Symbol("coordinate_dofs",
                                             rank=(fiat_element.space_dimension(),))))
    coefficient_map[mesh.coordinates] = make_kernel_argument(fiat_element, "coordinate_dofs", integral_type)

    coefficients = fd.preprocessed_form.coefficients()
    for i, coefficient in enumerate(coefficients):
        if coefficient.ufl_element().family() == "Real":
            rank = coefficient.ufl_shape
            arg = ein.Variable("w_%d" % i, coefficient.ufl_shape + (1,))
            multiindex = tuple(ein.Index() for i in xrange(len(coefficient.ufl_shape)))
            if multiindex:
                coefficient_map[coefficient] = ein.ComponentTensor(ein.Indexed(arg, multiindex + (0,)), multiindex)
            else:
                coefficient_map[coefficient] = ein.Indexed(arg, (0,))
        else:
            fiat_element = create_element(coefficient.ufl_element())
            rank = (fiat_element.space_dimension(),)
            coefficient_map[coefficient] = make_kernel_argument(fiat_element, "w_%d" % i, integral_type)
        decl = coffee.Decl("const %s *restrict" % SCALAR_TYPE, coffee.Symbol("w_%d" % i, rank=rank))
        arglist.append(decl)

    if integral_type.startswith("exterior_facet"):
        decl = coffee.Decl("const unsigned int", coffee.Symbol("facet", rank=(1,)))
        arglist.append(decl)
    elif integral_type.startswith("interior_facet"):
        decl = coffee.Decl("const unsigned int", coffee.Symbol("facet", rank=(2,)))
        arglist.append(decl)

    cell = integrand.ufl_domain().ufl_cell()
    # TODO: Hardcoded "default" quadrature rule!
    quad_points, quad_weights = create_quadrature_points_and_weights(integral_type, cell,
                                                                     quadrature_degree, rule="default")

    tabulation_manager = fem.TabulationManager(integral_type, cell, quad_points)
    quadrature_index, nonfem = fem.process(integral_type, integrand, tabulation_manager,
                                           quad_weights, argument_indices, coefficient_map)
    nonfem = [ein.IndexSum(e, quadrature_index) for e in nonfem]

    assert len(nonfem) == (2**len(arguments) if integral_type.startswith("interior_facet") else 1)
    if integral_type.startswith("interior_facet") and arguments:
        offset = [create_element(arg.ufl_element()).space_dimension() for arg in arguments]
        result = numpy.empty([s*2 for s in offset], dtype=object)
        for i, rs in enumerate(itertools.product((0, 1), repeat=len(arguments))):
            component = ein.ComponentTensor(nonfem[i], argument_indices)
            for mi in numpy.ndindex(tuple(offset)):
                result[tuple(numpy.asarray(mi) + numpy.asarray(offset) * numpy.asarray(rs))] = ein.Indexed(component, mi)

        reorder = []
        element = create_element(arguments[0].ufl_element())
        if isinstance(element, ffc_MixedElement):
            r = numpy.arange(2 * element.space_dimension()).reshape(2, len(element.elements()), -1).transpose(1, 0, 2).reshape(-1)
            reorder += list(r + len(reorder))
        else:
            reorder += range(len(reorder), len(reorder) + 2 * element.space_dimension())
        result = result[reorder]

        if len(arguments) == 2:
            reorder = []
            element = create_element(arguments[1].ufl_element())
            if isinstance(element, ffc_MixedElement):
                r = numpy.arange(2 * element.space_dimension()).reshape(2, len(element.elements()), -1).transpose(1, 0, 2).reshape(-1)
                reorder += list(r + len(reorder))
            else:
                reorder += range(len(reorder), len(reorder) + 2 * element.space_dimension())
            result = result[:, reorder]

        result = ein.ListTensor(result)
        result = ein.Indexed(result, output_indices)
    else:
        result, = nonfem
        result = ein.Indexed(ein.ComponentTensor(result, argument_indices), output_indices)

    simplified = ein.inline_indices(result)

    index_extents = ein.collect_index_extents(simplified)
    for output_index in output_indices:
        assert output_index.extent
        index_extents[output_index] = output_index.extent
    index_ordering = apply_prefix_ordering(index_extents.keys(),
                                           (quadrature_index,) + argument_indices + output_indices)
    apply_ordering = make_index_orderer(index_ordering)

    shape_map = lambda expr: expr.free_indices
    ordered_shape_map = lambda expr: apply_ordering(shape_map(expr))

    indexed_ops = sch.make_ordering(output_arg, simplified, ordered_shape_map)
    temporaries = make_temporaries(op for indices, op in indexed_ops)

    index_names = zip((quadrature_index,) + argument_indices, ['ip', 'j', 'k'])
    body = generate_coffee(indexed_ops, temporaries, shape_map,
                           apply_ordering, index_extents, index_names)

    funname = "%s_%s_integral_%s_%s" % (prefix, integral_type, "0", integral.subdomain_id())
    kernel = coffee.FunDecl("void", funname, arglist, body, pred=["static", "inline"])

    info_green("firedrake.fc finished in %g seconds." % (time.time() - cpu_time))
    return kernel


def make_kernel_argument(fiat_element, name, integral_type):
    if integral_type.startswith("interior_facet"):
        arg = ein.Variable(name, (2 * fiat_element.space_dimension(), 1))

        if isinstance(fiat_element, ffc_MixedElement):
            elements = fiat_element.elements()
        else:
            elements = (fiat_element,)

        facet0 = []
        facet1 = []
        offset = 0
        for element in elements:
            space_dim = element.space_dimension()
            facet0.extend(range(offset, offset + space_dim))
            offset += space_dim
            facet1.extend(range(offset, offset + space_dim))
            offset += space_dim

        return ein.ListTensor([[ein.Indexed(arg, (i, 0)) for i in facet0],
                               [ein.Indexed(arg, (i, 0)) for i in facet1]])

    else:
        arg = ein.Variable(name, (fiat_element.space_dimension(), 1))

        i = ein.Index()
        return ein.ComponentTensor(ein.Indexed(arg, (i, 0)), (i,))


def make_index_orderer(index_ordering):
    idx2pos = {idx: pos for pos, idx in enumerate(index_ordering)}

    def apply_ordering(shape):
        return tuple(sorted(shape, key=lambda i: idx2pos[i]))
    return apply_ordering


def apply_prefix_ordering(indices, prefix_ordering):
    rest = set(indices) - set(prefix_ordering)
    return tuple(prefix_ordering) + tuple(rest)


def make_temporaries(operations):
    # For fast look up
    set_ = set()

    # For ordering
    list = []

    def make_temporary(o):
        if o not in set_:
            set_.add(o)
            list.append(o)

    for op in operations:
        if isinstance(op, (imp.Initialise, imp.Return)):
            pass
        elif isinstance(op, imp.Accumulate):
            make_temporary(op.indexsum)
        elif isinstance(op, imp.Evaluate):
            make_temporary(op.expression)
        else:
            raise AssertionError("unhandled operation: %s" % type(op))

    return list
