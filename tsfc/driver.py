from __future__ import absolute_import

import numpy
import time

from ufl.algorithms import compute_form_data
from ufl.log import GREEN

from tsfc.fiatinterface import create_element
from tsfc.mixedelement import MixedElement as ffc_MixedElement
from tsfc.quadrature import create_quadrature, QuadratureRule

import coffee.base as coffee

from tsfc import fem, gem as ein, impero as imp, scheduling as sch, optimise as opt
from tsfc.coffee import SCALAR_TYPE, generate as generate_coffee
from tsfc.constants import default_parameters
from tsfc.node import traversal


def compile_form(form, prefix="form", parameters=None):
    cpu_time = time.time()

    assert not isinstance(form, (list, tuple))

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
    for idata in fd.integral_data:
        # TODO: Merge kernel bodies for multiple integrals with same
        # integral-data (same mesh iteration space).
        start = time.time()
        kernel = compile_integral(idata, fd, prefix, parameters)
        if kernel is not None:
            print GREEN % ("compile_integral finished in %g seconds." % (time.time() - start))
            kernels.append(kernel)

    print GREEN % ("TSFC finished in %g seconds." % (time.time() - cpu_time))
    return kernels


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


def compile_integral(idata, fd, prefix, parameters):
    # Remove these here, they're handled below.
    if parameters.get("quadrature_degree") == "auto":
        del parameters["quadrature_degree"]
    if parameters.get("quadrature_rule") == "auto":
        del parameters["quadrature_rule"]

    integral_type = idata.integral_type
    kernel = Kernel(integral_type=integral_type, subdomain_id=idata.subdomain_id)

    arglist = []
    prepare = []
    coefficient_map = {}

    funarg, prepare_, expressions, finalise = prepare_arguments(integral_type, fd.preprocessed_form.arguments())
    argument_indices = sorted(expressions[0].free_indices, key=lambda index: index.name)

    arglist.append(funarg)
    prepare += prepare_
    argument_indices = tuple(index for index in expressions[0].multiindex if isinstance(index, ein.Index))

    mesh = idata.domain
    coordinates = fem.coordinate_coefficient(mesh)
    funarg, prepare_, expression = prepare_coefficient(integral_type, coordinates, "coords")

    arglist.append(funarg)
    prepare += prepare_
    coefficient_map[coordinates] = expression

    coefficient_numbers = []
    # enabled_coefficients is a boolean array that indicates which of
    # reduced_coefficients the integral requires.
    for i, on in enumerate(idata.enabled_coefficients):
        if not on:
            continue
        coefficient = fd.reduced_coefficients[i]
        # This is which coefficient in the original form the current
        # coefficient is.
        # Consider f*v*dx + g*v*ds, the full form contains two
        # coefficients, but each integral only requires one.
        coefficient_numbers.append(fd.original_coefficient_positions[i])
        funarg, prepare_, expression = prepare_coefficient(integral_type, coefficient, "w_%d" % i)

        arglist.append(funarg)
        prepare += prepare_
        coefficient_map[coefficient] = expression

    kernel.coefficient_numbers = tuple(coefficient_numbers)

    if integral_type in ["exterior_facet", "exterior_facet_vert"]:
        decl = coffee.Decl("unsigned int", coffee.Symbol("facet", rank=(1,)),
                           qualifiers=["const"])
        arglist.append(decl)
    elif integral_type in ["interior_facet", "interior_facet_vert"]:
        decl = coffee.Decl("unsigned int", coffee.Symbol("facet", rank=(2,)),
                           qualifiers=["const"])
        arglist.append(decl)

    nonfem_ = []
    quadrature_indices = []
    cell = idata.domain.ufl_cell()
    # Map from UFL FiniteElement objects to Index instances.  This is
    # so we reuse Index instances when evaluating the same coefficient
    # multiple times with the same table.  Occurs, for example, if we
    # have multiple integrals here (and the affine coordinate
    # evaluation can be hoisted).
    index_cache = {}
    for i, integral in enumerate(idata.integrals):
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
        quadrature_index = ein.Index(name="ip%d" % i)
        quadrature_indices.append(quadrature_index)
        nonfem, cell_orientations = \
            fem.process(integral_type, integrand, tabulation_manager,
                        quad_rule.weights, quadrature_index,
                        argument_indices, coefficient_map, index_cache)
        nonfem_.append([ein.IndexSum(e, quadrature_index) for e in nonfem])

    # Sum the expressions that are part of the same restriction
    nonfem = list(reduce(ein.Sum, e, ein.Zero()) for e in zip(*nonfem_))

    inlining_cache = {}
    simplified = [opt.inline_indices(e, inlining_cache) for e in nonfem]

    simplified = opt.expand_indexsum(simplified, max_extent=3)
    inlining_cache = {}
    simplified = [opt.inline_indices(e, inlining_cache) for e in simplified]

    refcount = sch.count_references(simplified)
    candidates = set()
    for node in traversal(simplified):
        if isinstance(node, ein.IndexSum):
            if refcount[node.children[0]] == 1:
                candidates.add(node.children[0])
        else:
            for child in node.children:
                if set(child.free_indices) == set(node.free_indices) and refcount[child] == 1:
                    if not (isinstance(child, ein.Literal) and child.shape):
                        candidates.add(child)

    if cell_orientations:
        decl = coffee.Decl("int *restrict *restrict", coffee.Symbol("cell_orientations"),
                           qualifiers=["const"])
        arglist.insert(2, decl)
        kernel.oriented = True

    # Need a deterministic ordering for these
    indices = set()
    for node in traversal(simplified):
        indices.update(node.free_indices)
    indices = sorted(indices)

    index_ordering = apply_prefix_ordering(indices, tuple(quadrature_indices) + argument_indices)
    apply_ordering = make_index_orderer(index_ordering)

    shape_map = lambda expr: expr.free_indices
    ordered_shape_map = lambda expr: apply_ordering(shape_map(expr))

    indexed_ops = sch.make_ordering(zip(expressions, simplified), ordered_shape_map)
    indexed_ops = [(multiindex, op)
                   for multiindex, op in indexed_ops
                   if not (isinstance(op, imp.Evaluate) and op.expression in candidates)]

    # Zero-simplification occurred
    if len(indexed_ops) == 0:
        return None
    temporaries = make_temporaries(op for loop_indices, op in indexed_ops)

    body = generate_coffee(indexed_ops, temporaries, shape_map, apply_ordering)
    body.open_scope = False

    funname = "%s_%s_integral_%s" % (prefix, integral_type, integral.subdomain_id())
    ast = coffee.FunDecl("void", funname, arglist, coffee.Block(prepare + [body] + finalise),
                         pred=["static", "inline"])
    kernel.ast = ast

    return kernel


def prepare_coefficient(integral_type, coefficient, name):

    if coefficient.ufl_element().family() == 'Real':
        # Constant

        shape = coefficient.ufl_shape or (1,)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=shape),
                             qualifiers=["const"])
        expression = ein.Variable(name, shape)
        if coefficient.ufl_shape == ():
            expression = ein.Indexed(expression, (0,))

        return funarg, [], expression

    fiat_element = create_element(coefficient.ufl_element())

    if not integral_type.startswith("interior_facet"):
        # Simple case

        shape = (fiat_element.space_dimension(),)
        funarg = coffee.Decl("%s *restrict" % SCALAR_TYPE, coffee.Symbol(name, rank=shape),
                             qualifiers=["const"])

        i = ein.Index()
        expression = ein.ComponentTensor(
            ein.Indexed(ein.Variable(name, shape + (1,)),
                        (i, 0)),
            (i,))

        return funarg, [], expression

    if not isinstance(fiat_element, ffc_MixedElement):
        # Interior facet integral

        shape = (2, fiat_element.space_dimension())

        funarg = coffee.Decl("%s *restrict" % SCALAR_TYPE, coffee.Symbol(name, rank=shape),
                             qualifiers=["const"])
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

    funarg = coffee.Decl("%s *restrict *restrict" % SCALAR_TYPE, coffee.Symbol(name_),
                         qualifiers=["const"])
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


def prepare_arguments(integral_type, arguments):
    from itertools import chain, product

    if len(arguments) == 0:
        # No arguments
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=(1,)))
        expression = ein.Indexed(ein.Variable("A", (1,)), (0,))

        return funarg, [], [expression], []

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)
    indices = tuple(ein.Index(name=name) for i, name in zip(range(len(arguments)), ['j', 'k']))

    if not integral_type.startswith("interior_facet"):
        # Not an interior facet integral
        shape = tuple(element.space_dimension() for element in elements)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        expression = ein.Indexed(ein.Variable("A", shape), indices)

        return funarg, [], [expression], []

    if not any(isinstance(element, ffc_MixedElement) for element in elements):
        # Interior facet integral, but no vector (mixed) arguments
        shape = []
        for element in elements:
            shape += [2, element.space_dimension()]
        shape = tuple(shape)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        varexp = ein.Variable("A", shape)

        expressions = []
        for restrictions in product((0, 1), repeat=len(arguments)):
            is_ = tuple(chain(*zip(restrictions, indices)))
            expressions.append(ein.Indexed(varexp, is_))

        return funarg, [], expressions, []

    # Interior facet integral + vector (mixed) argument(s)
    shape = tuple(element.space_dimension() for element in elements)
    funarg_shape = tuple(s * 2 for s in shape)
    funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=funarg_shape))

    prepare = []
    expressions = []

    references = []
    for restrictions in product((0, 1), repeat=len(arguments)):
        name = "A" + "".join(map(str, restrictions))

        prepare.append(coffee.Decl(SCALAR_TYPE,
                                   coffee.Symbol(name, rank=shape),
                                   init=coffee.ArrayInit(numpy.zeros(1))))
        expressions.append(ein.Indexed(ein.Variable(name, shape), indices))

        for multiindex in numpy.ndindex(shape):
            references.append(coffee.Symbol(name, multiindex))

    restriction_shape = []
    for e in elements:
        if isinstance(e, ffc_MixedElement):
            restriction_shape += [len(e.elements()),
                                  e.elements()[0].space_dimension()]
        else:
            restriction_shape += [1, e.space_dimension()]
    restriction_shape = tuple(restriction_shape)

    references = numpy.array(references)
    if len(arguments) == 1:
        references = references.reshape((2,) + restriction_shape)
        references = references.transpose(1, 0, 2)
    elif len(arguments) == 2:
        references = references.reshape((2, 2) + restriction_shape)
        references = references.transpose(2, 0, 3, 4, 1, 5)
    references = references.reshape(funarg_shape)

    finalise = []
    for multiindex in numpy.ndindex(funarg_shape):
        finalise.append(coffee.Assign(coffee.Symbol("A", rank=multiindex),
                                      references[multiindex]))

    return funarg, prepare, expressions, finalise


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
    # Need to return deterministically ordered indices
    return tuple(prefix_ordering) + tuple(k for k in indices if k in rest)


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
