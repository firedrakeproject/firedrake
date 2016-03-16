from __future__ import absolute_import

import itertools

import numpy

import coffee.base as coffee

from tsfc import gem
from tsfc.fiatinterface import create_element
from tsfc.mixedelement import MixedElement
from tsfc.coffee import SCALAR_TYPE


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


class Interface(object):
    def __init__(self, integral_type, arguments):
        self.integral_type = integral_type
        self.interior_facet = integral_type.startswith("interior_facet")

        funarg, prepare, expressions, finalise = prepare_arguments(self.interior_facet, arguments)

        self.return_funarg = funarg
        self.funargs = []
        self.prepare = prepare
        self.return_variables = expressions  # TODO
        self.finalise = finalise

        self.count = itertools.count()
        self.coefficients = {}

    def argument_indices(self):
        return filter(lambda i: isinstance(i, gem.Index), self.return_variables[0].multiindex)

    def preload(self, coefficient, name=None, mode=None):
        # Do not add the same coefficient twice
        assert coefficient not in self.coefficients

        # Default name with counting
        if name is None:
            name = "w_{0}".format(next(self.count))

        funarg, prepare, expression = prepare_coefficient(self.interior_facet, coefficient, name, mode)
        self.funargs.append(funarg)
        self.prepare.extend(prepare)
        self.coefficients[coefficient] = expression

    def gem(self, coefficient):
        try:
            return self.coefficients[coefficient]
        except KeyError:
            self.preload(coefficient)
            return self.coefficients[coefficient]

    def construct_kernel_function(self, name, body, oriented):
        args = [self.return_funarg] + self.funargs
        if oriented:
            args.insert(2, coffee.Decl("int *restrict *restrict",
                                       coffee.Symbol("cell_orientations"),
                                       qualifiers=["const"]))

        if self.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            args.append(coffee.Decl("unsigned int",
                                    coffee.Symbol("facet", rank=(1,)),
                                    qualifiers=["const"]))
        elif self.integral_type in ["interior_facet", "interior_facet_vert"]:
            args.append(coffee.Decl("unsigned int",
                                    coffee.Symbol("facet", rank=(2,)),
                                    qualifiers=["const"]))

        body_ = coffee.Block(self.prepare + [body] + self.finalise)
        return coffee.FunDecl("void", name, args, body_, pred=["static", "inline"])


def prepare_coefficient(interior_facet, coefficient, name, mode=None):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.  Mixed element Coefficients are rearranged here for
    interior facet integrals.

    :arg interior_facet: interior facet integral?
    :arg coefficient: UFL Coefficient
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg mode: 'manual_loop' or 'list_tensor'; two ways to deal with
               interior facet integrals on mixed elements
    :returns: (funarg, prepare, expression)
         funarg     - :class:`coffee.Decl` function argument
         prepare    - list of COFFEE nodes to be prepended to the
                      kernel body
         expression - GEM expression referring to the Coefficient
                      values
    """
    if mode is None:
        mode = 'manual_loop'

    assert mode in ['manual_loop', 'list_tensor']

    if coefficient.ufl_element().family() == 'Real':
        # Constant

        shape = coefficient.ufl_shape or (1,)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=shape),
                             qualifiers=["const"])
        expression = gem.Variable(name, shape)
        if coefficient.ufl_shape == ():
            expression = gem.Indexed(expression, (0,))

        return funarg, [], expression

    fiat_element = create_element(coefficient.ufl_element())

    if not interior_facet:
        # Simple case

        shape = (fiat_element.space_dimension(),)
        funarg = coffee.Decl("%s *restrict" % SCALAR_TYPE, coffee.Symbol(name, rank=shape),
                             qualifiers=["const"])

        i = gem.Index()
        expression = gem.ComponentTensor(
            gem.Indexed(gem.Variable(name, shape + (1,)),
                        (i, 0)),
            (i,))

        return funarg, [], expression

    if not isinstance(fiat_element, MixedElement):
        # Interior facet integral

        shape = (2, fiat_element.space_dimension())

        funarg = coffee.Decl("%s *restrict" % SCALAR_TYPE, coffee.Symbol(name, rank=shape),
                             qualifiers=["const"])
        expression = gem.Variable(name, shape + (1,))

        f, i = gem.Index(), gem.Index()
        expression = gem.ComponentTensor(
            gem.Indexed(gem.Variable(name, shape + (1,)),
                        (f, i, 0)),
            (f, i,))

        return funarg, [], expression

    # Interior facet integral + mixed / vector element

    # Here we need to reorder the coefficient values.
    #
    # Incoming ordering: E1+ E1- E2+ E2- E3+ E3-
    # Required ordering: E1+ E2+ E3+ E1- E2- E3-
    #
    # Each of E[n]{+,-} is a vector of basis function coefficients for
    # subelement E[n].
    #
    # There are two code generation method to reorder the values.
    # We have not done extensive research yet as to which way yield
    # faster code.

    if mode == 'manual_loop':
        # In this case we generate loops outside the GEM abstraction
        # to reorder the values.  A whole E[n]{+,-} block is copied by
        # a single loop.
        name_ = name + "_"
        shape = (2, fiat_element.space_dimension())

        funarg = coffee.Decl("%s *restrict *restrict" % SCALAR_TYPE, coffee.Symbol(name_),
                             qualifiers=["const"])
        prepare = [coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=shape))]
        expression = gem.Variable(name, shape)

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

    elif mode == 'list_tensor':
        # In this case we generate a gem.ListTensor to do the
        # reordering.  Every single element in a E[n]{+,-} block is
        # referenced separately.
        funarg = coffee.Decl("%s *restrict *restrict" % SCALAR_TYPE, coffee.Symbol(name),
                             qualifiers=["const"])

        variable = gem.Variable(name, (2 * fiat_element.space_dimension(), 1))

        facet_0 = []
        facet_1 = []
        offset = 0
        for element in fiat_element.elements():
            space_dim = element.space_dimension()

            for i in range(offset, offset + space_dim):
                facet_0.append(gem.Indexed(variable, (i, 0)))
            offset += space_dim

            for i in range(offset, offset + space_dim):
                facet_1.append(gem.Indexed(variable, (i, 0)))
            offset += space_dim

        expression = gem.ListTensor(numpy.array([facet_0, facet_1]))
        return funarg, [], expression


def prepare_arguments(interior_facet, arguments):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg interior_facet: interior facet integral?
    :arg arguments: UFL Arguments
    :returns: (funarg, prepare, expression, finalise)
         funarg     - :class:`coffee.Decl` function argument
         prepare    - list of COFFEE nodes to be prepended to the
                      kernel body
         expression - GEM expression referring to the argument tensor
         finalise   - list of COFFEE nodes to be appended to the
                      kernel body
    """
    from itertools import chain, product

    if len(arguments) == 0:
        # No arguments
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=(1,)))
        expression = gem.Indexed(gem.Variable("A", (1,)), (0,))

        return funarg, [], [expression], []

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)
    indices = tuple(gem.Index(name=name) for i, name in zip(range(len(arguments)), ['j', 'k']))

    if not interior_facet:
        # Not an interior facet integral
        shape = tuple(element.space_dimension() for element in elements)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        expression = gem.Indexed(gem.Variable("A", shape), indices)

        return funarg, [], [expression], []

    if not any(isinstance(element, MixedElement) for element in elements):
        # Interior facet integral, but no vector (mixed) arguments
        shape = []
        for element in elements:
            shape += [2, element.space_dimension()]
        shape = tuple(shape)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        varexp = gem.Variable("A", shape)

        expressions = []
        for restrictions in product((0, 1), repeat=len(arguments)):
            is_ = tuple(chain(*zip(restrictions, indices)))
            expressions.append(gem.Indexed(varexp, is_))

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
        expressions.append(gem.Indexed(gem.Variable(name, shape), indices))

        for multiindex in numpy.ndindex(shape):
            references.append(coffee.Symbol(name, multiindex))

    restriction_shape = []
    for e in elements:
        if isinstance(e, MixedElement):
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
    """Helper function to make a COFFEE loop.

    :arg index: :class:`coffee.Symbol` loop index
    :arg extent: loop extent (integer)
    :arg body: loop body (COFFEE node)
    :returns: COFFEE loop
    """
    return coffee.For(coffee.Decl("int", index, init=0),
                      coffee.Less(index, extent),
                      coffee.Incr(index, 1),
                      body)
