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
    prepare = []
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
    funarg, prepare_, expression = prepare_coefficient(integral_type, mesh.coordinates, "coords")

    arglist.append(funarg)
    prepare += prepare_
    coefficient_map[mesh.coordinates] = expression

    for i, coefficient in enumerate(fd.preprocessed_form.coefficients()):
        funarg, prepare_, expression = prepare_coefficient(integral_type, coefficient, "w_%d" % i)

        arglist.append(funarg)
        prepare += prepare_
        coefficient_map[coefficient] = expression

    if integral_type in ["exterior_facet", "exterior_facet_vert"]:
        decl = coffee.Decl("const unsigned int", coffee.Symbol("facet", rank=(1,)))
        arglist.append(decl)
    elif integral_type in ["interior_facet", "interior_facet_vert"]:
        decl = coffee.Decl("const unsigned int", coffee.Symbol("facet", rank=(2,)))
        arglist.append(decl)

    cell = integrand.ufl_domain().ufl_cell()
    # TODO: Hardcoded "default" quadrature rule!
    quad_points, quad_weights = create_quadrature_points_and_weights(integral_type, cell,
                                                                     quadrature_degree, rule="default")

    tabulation_manager = fem.TabulationManager(integral_type, cell, quad_points)
    quadrature_index, nonfem, cell_orientations = \
        fem.process(integral_type, integrand, tabulation_manager, quad_weights, argument_indices, coefficient_map)
    nonfem = [ein.IndexSum(e, quadrature_index) for e in nonfem]

    if cell_orientations:
        decl = coffee.Decl("const int *restrict *restrict", coffee.Symbol("cell_orientations"))
        arglist.insert(2, decl)

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
    body.open_scope = False

    funname = "%s_%s_integral_%s_%s" % (prefix, integral_type, "0", integral.subdomain_id())
    kernel = coffee.FunDecl("void", funname, arglist, coffee.Block(prepare + [body]), pred=["static", "inline"])

    info_green("firedrake.fc finished in %g seconds." % (time.time() - cpu_time))
    return kernel


def prepare_coefficient(integral_type, coefficient, name):

    if coefficient.ufl_element().family() == 'Real':
        # Constant

        shape = coefficient.ufl_shape or (1,)

        funarg = coffee.Decl("const %s" % SCALAR_TYPE, coffee.Symbol(name, rank=shape))
        expression = ein.Variable(name, shape)
        if coefficient.ufl_shape == ():
            expression = ein.Indexed(expression, (0,))

        return funarg, [], expression

    fiat_element = create_element(coefficient.ufl_element())

    if not integral_type.startswith("interior_facet"):
        # Simple case

        shape = (fiat_element.space_dimension(),)
        funarg = coffee.Decl("const %s *restrict" % SCALAR_TYPE, coffee.Symbol(name, rank=shape))

        i = ein.Index()
        expression = ein.ComponentTensor(
            ein.Indexed(ein.Variable(name, shape + (1,)),
                        (i, 0)),
            (i,))

        return funarg, [], expression

    if not isinstance(fiat_element, ffc_MixedElement):
        # Interior facet integral

        shape = (2, fiat_element.space_dimension())

        funarg = coffee.Decl("const %s *restrict" % SCALAR_TYPE, coffee.Symbol(name, rank=shape))
        expression = ein.Variable(name, shape + (1,))

        f, i = ein.Index(), ein.Index()
        expression = ein.ComponentTensor(
            ein.Indexed(ein.Variable(name, shape + (1,)),
                        (f, i, 0)),
            (f, i,))

        return funarg, [], expression

    # Interior facet integral + mixed / vector element
    name_ = name + "_"
    shape = (2, fiat_element.space_dimension())

    funarg = coffee.Decl("const %s *restrict *restrict" % SCALAR_TYPE, coffee.Symbol(name_))
    prepare = [coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=shape))]
    expression = ein.Variable(name, shape)

    offset = 0
    i = coffee.Symbol("i")
    for element in fiat_element.elements():
        space_dim = element.space_dimension()

        loop_body = coffee.Assign(coffee.Symbol(name, rank=(0, coffee.Sum(offset, i))),
                                  coffee.Symbol(name_, rank=(coffee.Sum(2 * offset, i), 0)))
        prepare.append(coffee_for(i, space_dim, loop_body))

        loop_body = coffee.Assign(coffee.Symbol(name, rank=(1, coffee.Sum(offset, i))),
                                  coffee.Symbol(name_, rank=(coffee.Sum(2 * offset + space_dim, i), 0)))
        prepare.append(coffee_for(i, space_dim, loop_body))

        offset += space_dim

    return funarg, prepare, expression


def coffee_for(index, extent, body):
    return coffee.For(coffee.Decl("int", index, init=0),
                      coffee.Less(index, extent),
                      coffee.Incr(index, 1),
                      body)


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
