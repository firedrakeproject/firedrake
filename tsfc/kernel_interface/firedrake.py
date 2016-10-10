from __future__ import absolute_import, print_function, division

import numpy
from collections import namedtuple
from itertools import product

from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace

import coffee.base as coffee

import gem
from gem.node import traversal

from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase
from tsfc.fiatinterface import create_element
from tsfc.mixedelement import MixedElement
from tsfc.coffee import SCALAR_TYPE


# Expression kernel description type
ExpressionKernel = namedtuple('ExpressionKernel', ['ast', 'oriented', 'coefficients'])


class Kernel(object):
    __slots__ = ("ast", "integral_type", "oriented", "subdomain_id",
                 "domain_number",
                 "coefficient_numbers", "__weakref__")
    """A compiled Kernel object.

    :kwarg ast: The COFFEE ast for the kernel.
    :kwarg integral_type: The type of integral.
    :kwarg oriented: Does the kernel require cell_orientations.
    :kwarg subdomain_id: What is the subdomain id for this kernel.
    :kwarg domain_number: Which domain number in the original form
        does this kernel correspond to (can be used to index into
        original_form.ufl_domains() to get the correct domain).
    :kwarg coefficient_numbers: A list of which coefficients from the
        form the kernel needs.
    """
    def __init__(self, ast=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None,
                 coefficient_numbers=()):
        # Defaults
        self.ast = ast
        self.integral_type = integral_type
        self.oriented = oriented
        self.domain_number = domain_number
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        super(Kernel, self).__init__()


class KernelBuilderBase(_KernelBuilderBase):

    def __init__(self, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        super(KernelBuilderBase, self).__init__(interior_facet=interior_facet)

        # Cell orientation
        if self.interior_facet:
            cell_orientations = gem.Variable("cell_orientations", (2, 1))
            self._cell_orientations = (gem.Indexed(cell_orientations, (0, 0)),
                                       gem.Indexed(cell_orientations, (1, 0)))
        else:
            cell_orientations = gem.Variable("cell_orientations", (1, 1))
            self._cell_orientations = (gem.Indexed(cell_orientations, (0, 0)),)

    def _coefficient(self, coefficient, name, mode=None):
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

    @staticmethod
    def needs_cell_orientations(ir):
        """Does a multi-root GEM expression DAG references cell
        orientations?"""
        for node in traversal(ir):
            if isinstance(node, gem.Variable) and node.name == "cell_orientations":
                return True
        return False


class ExpressionKernelBuilder(KernelBuilderBase):
    """Builds expression kernels for UFL interpolation in Firedrake."""

    def __init__(self):
        super(ExpressionKernelBuilder, self).__init__()
        self.oriented = False

    def set_coefficients(self, coefficients):
        """Prepare the coefficients of the expression.

        :arg coefficients: UFL coefficients from Firedrake
        """
        self.coefficients = []  # Firedrake coefficients for calling the kernel
        self.coefficient_split = {}
        self.kernel_args = []

        for i, coefficient in enumerate(coefficients):
            if type(coefficient.ufl_element()) == ufl_MixedElement:
                subcoeffs = coefficient.split()  # Firedrake-specific
                self.coefficients.extend(subcoeffs)
                self.coefficient_split[coefficient] = subcoeffs
                self.kernel_args += [self._coefficient(subcoeff, "w_%d_%d" % (i, j))
                                     for j, subcoeff in enumerate(subcoeffs)]
            else:
                self.coefficients.append(coefficient)
                self.kernel_args.append(self._coefficient(coefficient, "w_%d" % (i,)))

    def require_cell_orientations(self):
        """Set that the kernel requires cell orientations."""
        self.oriented = True

    def construct_kernel(self, return_arg, body):
        """Constructs an :class:`ExpressionKernel`.

        :arg return_arg: COFFEE argument for the return value
        :arg body: function body (:class:`coffee.Block` node)
        :returns: :class:`ExpressionKernel` object
        """
        args = [return_arg] + self.kernel_args
        if self.oriented:
            args.insert(1, cell_orientations_coffee_arg)

        kernel_code = super(ExpressionKernelBuilder, self).construct_kernel("expression_kernel", args, body)
        return ExpressionKernel(kernel_code, self.oriented, self.coefficients)


class KernelBuilder(KernelBuilderBase):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_type, subdomain_id, domain_number):
        """Initialise a kernel builder."""
        super(KernelBuilder, self).__init__(integral_type.startswith("interior_facet"))

        self.kernel = Kernel(integral_type=integral_type, subdomain_id=subdomain_id,
                             domain_number=domain_number)
        self.local_tensor = None
        self.coordinates_arg = None
        self.coefficient_args = []
        self.coefficient_split = {}

        # Facet number
        if integral_type in ['exterior_facet', 'exterior_facet_vert']:
            facet = gem.Variable('facet', (1,))
            self._facet_number = {None: gem.VariableIndex(gem.Indexed(facet, (0,)))}
        elif integral_type in ['interior_facet', 'interior_facet_vert']:
            facet = gem.Variable('facet', (2,))
            self._facet_number = {
                '+': gem.VariableIndex(gem.Indexed(facet, (0,))),
                '-': gem.VariableIndex(gem.Indexed(facet, (1,)))
            }
        elif integral_type == 'interior_facet_horiz':
            self._facet_number = {'+': 1, '-': 0}

    def set_arguments(self, arguments, indices):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg indices: GEM argument indices
        :returns: GEM expression representing the return variable
        """
        self.local_tensor, prepare, expressions, finalise = prepare_arguments(
            arguments, indices, interior_facet=self.interior_facet)
        self.apply_glue(prepare, finalise)
        return expressions

    def set_coordinates(self, coefficient, mode=None):
        """Prepare the coordinate field.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg mode: see :func:`prepare_coefficient`
        """
        self.coordinates_arg = self._coefficient(coefficient, "coords", mode)

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        coefficients = []
        coefficient_numbers = []
        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        for i in range(len(integral_data.enabled_coefficients)):
            if integral_data.enabled_coefficients[i]:
                coefficient = form_data.reduced_coefficients[i]
                if type(coefficient.ufl_element()) == ufl_MixedElement:
                    split = [Coefficient(FunctionSpace(coefficient.ufl_domain(), element))
                             for element in coefficient.ufl_element().sub_elements()]
                    coefficients.extend(split)
                    self.coefficient_split[coefficient] = split
                else:
                    coefficients.append(coefficient)
                # This is which coefficient in the original form the
                # current coefficient is.
                # Consider f*v*dx + g*v*ds, the full form contains two
                # coefficients, but each integral only requires one.
                coefficient_numbers.append(form_data.original_coefficient_positions[i])
        for i, coefficient in enumerate(coefficients):
            self.coefficient_args.append(
                self._coefficient(coefficient, "w_%d" % i))
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
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                             pointers=[("restrict",)],
                             qualifiers=["const"])

        expression = gem.reshape(gem.Variable(name, (None,)),
                                 coefficient.ufl_shape)

        return funarg, [], expression

    fiat_element = create_element(coefficient.ufl_element())
    size = fiat_element.space_dimension()

    if not interior_facet:
        # Simple case
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                             pointers=[("const", "restrict"), ("restrict",)],
                             qualifiers=["const"])

        expression = gem.reshape(gem.Variable(name, (size, 1)), (size,), ())

        return funarg, [], expression

    if not isinstance(fiat_element, MixedElement):
        # Interior facet integral
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                             pointers=[("const", "restrict"), ("restrict",)],
                             qualifiers=["const"])

        expression = gem.reshape(
            gem.Variable(name, (2 * size, 1)), (2, size), ()
        )

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
        shape = (2, size)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name_),
                             pointers=[("const", "restrict"), ("restrict",)],
                             qualifiers=["const"])
        prepare = [coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=shape))]
        expression = gem.Variable(name, shape)

        offset = 0
        i = coffee.Symbol("i")
        for element in fiat_element.elements():
            space_dim = element.space_dimension()

            loop_body = coffee.Assign(coffee.Symbol(name, rank=(0, "i"),
                                                    offset=((1, 0), (1, offset))),
                                      coffee.Symbol(name_, rank=("i", 0),
                                                    offset=((1, 2 * offset), (1, 0))))
            prepare.append(coffee_for(i, space_dim, loop_body))

            loop_body = coffee.Assign(coffee.Symbol(name, rank=(1, "i"),
                                                    offset=((1, 0), (1, offset))),
                                      coffee.Symbol(name_, rank=("i", 0),
                                                    offset=((1, 2 * offset + space_dim), (1, 0))))
            prepare.append(coffee_for(i, space_dim, loop_body))

            offset += space_dim

        return funarg, prepare, expression

    elif mode == 'list_tensor':
        # In this case we generate a gem.ListTensor to do the
        # reordering.  Every single element in a E[n]{+,-} block is
        # referenced separately.
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                             pointers=[("const", "restrict"), ("restrict",)],
                             qualifiers=["const"])

        variable = gem.Variable(name, (2 * size, 1))

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
    assert isinstance(interior_facet, bool)

    if len(arguments) == 0:
        # No arguments
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=(1,)))
        expression = gem.Indexed(gem.Variable("A", (1,)), (0,))

        return funarg, [], [expression], []

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)

    if not interior_facet:
        # Not an interior facet integral
        shape = tuple(element.space_dimension() for element in elements)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        expression = gem.Indexed(gem.Variable("A", shape), indices)

        return funarg, [], [expression], []

    if not any(isinstance(element, MixedElement) for element in elements):
        # Interior facet integral, but no vector (mixed) arguments
        shape = tuple(2 * element.space_dimension() for element in elements)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        varexp = gem.Variable("A", shape)

        expressions = []
        for restrictions in product((0, 1), repeat=len(arguments)):
            expressions.append(gem.FlexiblyIndexed(
                varexp,
                tuple((r * e.space_dimension(), ((i, e.space_dimension()),))
                      for e, i, r in zip(elements, indices, restrictions))
            ))

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


cell_orientations_coffee_arg = coffee.Decl("int", coffee.Symbol("cell_orientations"),
                                           pointers=[("restrict", "const"), ("restrict",)],
                                           qualifiers=["const"])
"""COFFEE function argument for cell orientations"""
