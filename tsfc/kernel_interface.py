from __future__ import absolute_import

import operator

import numpy

import coffee.base as coffee

import gem
from gem.node import traversal

from tsfc.finatinterface import create_element
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


class KernelBuilderBase(object):
    """Helper class for building local assembly kernels."""

    def __init__(self, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        assert isinstance(interior_facet, bool)
        self.interior_facet = interior_facet

        self.prepare = []
        self.finalise = []

        self.coefficient_map = {}

    def apply_glue(self, prepare=None, finalise=None):
        """Append glue code for operations that are not handled in the
        GEM abstraction.

        Current uses: mixed interior facet mess

        :arg prepare: code snippets to be prepended to the kernel
        :arg finalise: code snippets to be appended to the kernel
        """
        if prepare is not None:
            self.prepare.extend(prepare)
        if finalise is not None:
            self.finalise.extend(finalise)

    def construct_kernel(self, name, args, body):
        """Construct a COFFEE function declaration with the
        accumulated glue code.

        :arg name: function name
        :arg args: function argument list
        :arg body: function body (:class:`coffee.Block` node)
        :returns: :class:`coffee.FunDecl` object
        """
        assert isinstance(body, coffee.Block)
        body_ = coffee.Block(self.prepare + body.children + self.finalise)
        return coffee.FunDecl("void", name, args, body_, pred=["static", "inline"])

    @property
    def coefficient_mapper(self):
        """A function that maps :class:`ufl.Coefficient`s to GEM
        expressions."""
        return lambda coefficient: self.coefficient_map[coefficient]

    def arguments(self, arguments, indices):
        """Prepare arguments. Adds glue code for the arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg indices: GEM argument indices
        :returns: COFFEE function argument and GEM expression
                  representing the argument tensor
        """
        funarg, prepare, expressions, finalise = prepare_arguments(
            arguments, indices, interior_facet=self.interior_facet)
        self.apply_glue(prepare, finalise)
        return funarg, expressions

    def coefficient(self, coefficient, name, mode=None):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :arg mode: see :func:`prepare_coefficient`
        :returns: COFFEE function argument for the coefficient
        """
        funarg, prepare, expression = prepare_coefficient(
            coefficient, name, mode=mode,
            interior_facet=self.interior_facet)
        self.apply_glue(prepare)
        self.coefficient_map[coefficient] = expression
        return funarg


class KernelBuilder(KernelBuilderBase):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_type, subdomain_id):
        """Initialise a kernel builder."""
        super(KernelBuilder, self).__init__(integral_type.startswith("interior_facet"))

        self.kernel = Kernel(integral_type=integral_type, subdomain_id=subdomain_id)
        self.local_tensor = None
        self.coordinates_arg = None
        self.coefficient_args = []

    def set_arguments(self, arguments, indices):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg indices: GEM argument indices
        :returns: GEM expression representing the return variable
        """
        self.local_tensor, expressions = self.arguments(arguments, indices)
        return expressions

    def set_coordinates(self, coefficient, name, mode=None):
        """Prepare the coordinate field.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coordinate coefficient name
        :arg mode: see :func:`prepare_coefficient`
        """
        self.coordinates_arg = self.coefficient(coefficient, name, mode)

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        coefficient_numbers = []
        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        for i in range(len(integral_data.enabled_coefficients)):
            if integral_data.enabled_coefficients[i]:
                coefficient = form_data.reduced_coefficients[i]
                self.coefficient_args.append(
                    self.coefficient(coefficient, "w_%d" % i))
                # This is which coefficient in the original form the
                # current coefficient is.
                # Consider f*v*dx + g*v*ds, the full form contains two
                # coefficients, but each integral only requires one.
                coefficient_numbers.append(form_data.original_coefficient_positions[i])
        self.kernel.coefficient_numbers = tuple(coefficient_numbers)

    def require_cell_orientations(self):
        """Set that the kernel requires cell orientations."""
        self.kernel.oriented = True

    def construct_kernel(self, name, body):
        """Construct a fully built :class:`Kernel`.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: function name
        :arg body: function body (:class:`coffee.Block` node)
        :returns: :class:`Kernel` object
        """
        args = [self.local_tensor, self.coordinates_arg]
        if self.kernel.oriented:
            args.append(cell_orientations_coffee_arg)
        args.extend(self.coefficient_args)
        if self.kernel.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            args.append(coffee.Decl("unsigned int",
                                    coffee.Symbol("facet", rank=(1,)),
                                    qualifiers=["const"]))
        elif self.kernel.integral_type in ["interior_facet", "interior_facet_vert"]:
            args.append(coffee.Decl("unsigned int",
                                    coffee.Symbol("facet", rank=(2,)),
                                    qualifiers=["const"]))

        self.kernel.ast = KernelBuilderBase.construct_kernel(self, name, args, body)
        return self.kernel


def prepare_coefficient(coefficient, name, mode=None, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.  Mixed element Coefficients are rearranged here for
    interior facet integrals.

    :arg coefficient: UFL Coefficient
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg mode: 'manual_loop' or 'list_tensor'; two ways to deal with
               interior facet integrals on mixed elements
    :arg interior_facet: interior facet integral?
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
    assert isinstance(interior_facet, bool)

    if coefficient.ufl_element().family() == 'Real':
        # Constant

        shape = coefficient.ufl_shape or (1,)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=shape),
                             qualifiers=["const"])
        expression = gem.Variable(name, shape)
        if coefficient.ufl_shape == ():
            expression = gem.Indexed(expression, (0,))

        return funarg, [], expression

    finat_element = create_element(coefficient.ufl_element())

    if not interior_facet:
        # Simple case

        shape = finat_element.index_shape
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=tuple(reversed(shape))),
                             pointers=[("restrict",)],
                             qualifiers=["const"])

        alpha = tuple(gem.Index() for d in shape)
        expression = gem.ComponentTensor(
            gem.Indexed(gem.Variable(name, tuple(reversed(shape)) + (1,)),
                        tuple(reversed(alpha)) + (0,)),
            alpha)

        return funarg, [], expression

    if not isinstance(finat_element, MixedElement):
        # Interior facet integral

        shape = (2,) + finat_element.index_shape

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=shape),
                             pointers=[("restrict",)],
                             qualifiers=["const"])
        expression = gem.Variable(name, shape + (1,))

        f, i = gem.Index(), gem.Index()
        expression = gem.ComponentTensor(
            gem.Indexed(gem.Variable(name, shape + (1,)),
                        (f, i, 0)),
            (f, i,))

        return funarg, [], expression

    assert False, "Deal with FInAT + mixed + interior facet later."
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
        shape = (2,) + fiat_element.index_shape

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name_),
                             pointers=[("restrict",), ("restrict",)],
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
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                             pointers=[("restrict",), ("restrict",)],
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


def prepare_arguments(arguments, indices, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg arguments: UFL Arguments
    :arg indices: Argument indices
    :arg interior_facet: interior facet integral?
    :returns: (funarg, prepare, expression, finalise)
         funarg      - :class:`coffee.Decl` function argument
         prepare     - list of COFFEE nodes to be prepended to the
                       kernel body
         expressions - GEM expressions referring to the argument
                       tensor
         finalise    - list of COFFEE nodes to be appended to the
                       kernel body
    """
    from itertools import chain, product
    assert isinstance(interior_facet, bool)

    if len(arguments) == 0:
        # No arguments
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=(1,)))
        expression = gem.Indexed(gem.Variable("A", (1,)), (0,))

        return funarg, [], [expression], []

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)

    if not interior_facet:
        # Not an interior facet integral
        shape = reduce(operator.add, [element.index_shape for element in elements], ())

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        expression = gem.Indexed(gem.Variable("A", shape), indices)

        return funarg, [], [expression], []

    if not any(isinstance(element, MixedElement) for element in elements):
        # Interior facet integral, but no vector (mixed) arguments
        shape = []
        for element in elements:
            shape += [2] + list(element.index_shape)
        shape = tuple(shape)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        varexp = gem.Variable("A", shape)

        expressions = []
        for restrictions in product((0, 1), repeat=len(arguments)):
            is_ = tuple(chain(*zip(restrictions, indices)))
            expressions.append(gem.Indexed(varexp, is_))

        return funarg, [], expressions, []

    assert False, "Deal with FInAT + vector argument + interior facet later."
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


def needs_cell_orientations(ir):
    """Does a multi-root GEM expression DAG references cell
    orientations?"""
    for node in traversal(ir):
        if isinstance(node, gem.Variable) and node.name == "cell_orientations":
            return True
    return False


cell_orientations_coffee_arg = coffee.Decl("int", coffee.Symbol("cell_orientations"),
                                           pointers=[("restrict",), ("restrict",)],
                                           qualifiers=["const"])
"""COFFEE function argument for cell orientations"""
