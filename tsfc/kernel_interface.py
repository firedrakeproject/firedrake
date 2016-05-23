from __future__ import absolute_import

import operator

import coffee.base as coffee

import gem
from gem.node import traversal

from tsfc.finatinterface import create_element
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

        self.coefficient_map = {}

    def construct_kernel(self, name, args, body):
        """Construct a COFFEE function declaration with the
        accumulated glue code.

        :arg name: function name
        :arg args: function argument list
        :arg body: function body (:class:`coffee.Block` node)
        :returns: :class:`coffee.FunDecl` object
        """
        assert isinstance(body, coffee.Block)
        return coffee.FunDecl("void", name, args, body, pred=["static", "inline"])

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
        funarg, expressions = prepare_arguments(arguments, indices, interior_facet=self.interior_facet)
        return funarg, expressions

    def coefficient(self, coefficient, name):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :returns: COFFEE function argument for the coefficient
        """
        funarg, expression = prepare_coefficient(coefficient, name, interior_facet=self.interior_facet)
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
        self.coefficient_split = {}

    def set_arguments(self, arguments, indices):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg indices: GEM argument indices
        :returns: GEM expression representing the return variable
        """
        self.local_tensor, expressions = self.arguments(arguments, indices)
        return expressions

    def set_coordinates(self, coefficient, name):
        """Prepare the coordinate field.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coordinate coefficient name
        """
        self.coordinates_arg = self.coefficient(coefficient, name)

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace
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
                self.coefficient(coefficient, "w_%d" % i))
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


def prepare_coefficient(coefficient, name, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg coefficient: UFL Coefficient
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         funarg     - :class:`coffee.Decl` function argument
         expression - GEM expression referring to the Coefficient
                      values
    """
    assert isinstance(interior_facet, bool)

    if coefficient.ufl_element().family() == 'Real':
        # Constant

        shape = coefficient.ufl_shape or (1,)

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=shape),
                             qualifiers=["const"])
        expression = gem.Variable(name, shape)
        if coefficient.ufl_shape == ():
            expression = gem.Indexed(expression, (0,))

        return funarg, expression

    import ufl
    pyop2_scalar = not isinstance(coefficient.ufl_element(), (ufl.VectorElement, ufl.TensorElement))
    finat_element = create_element(coefficient.ufl_element())

    if not interior_facet:
        # Simple case

        shape = finat_element.index_shape
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=(shape if pyop2_scalar else shape[:-1])),
                             pointers=[("restrict",)],
                             qualifiers=["const"])

        if pyop2_scalar:
            alpha = tuple(gem.Index() for d in shape)
            expression = gem.ComponentTensor(
                gem.Indexed(gem.Variable(name, shape + (1,)),
                            alpha + (0,)),
                alpha)
        else:
            expression = gem.Variable(name, shape)

        return funarg, expression

    # Interior facet integral
    shape = (2,) + finat_element.index_shape

    funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name, rank=(shape if pyop2_scalar else shape[:-1])),
                         pointers=[("restrict",)],
                         qualifiers=["const"])
    if pyop2_scalar:
        alpha = tuple(gem.Index() for d in shape)
        expression = gem.ComponentTensor(
            gem.Indexed(gem.Variable(name, shape + (1,)),
                        alpha + (0,)),
            alpha)
    else:
        expression = gem.Variable(name, shape)

    return funarg, expression


def prepare_arguments(arguments, indices, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg arguments: UFL Arguments
    :arg indices: Argument indices
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         funarg      - :class:`coffee.Decl` function argument
         expressions - GEM expressions referring to the argument
                       tensor
    """
    from itertools import chain, product
    assert isinstance(interior_facet, bool)

    if len(arguments) == 0:
        # No arguments
        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=(1,)))
        expression = gem.Indexed(gem.Variable("A", (1,)), (0,))

        return funarg, [expression]

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)

    if not interior_facet:
        # Not an interior facet integral
        shape = reduce(operator.add, [element.index_shape for element in elements], ())

        funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        expression = gem.Indexed(gem.Variable("A", shape), tuple(chain(*indices)))

        return funarg, [expression]

    # Interior facet integral
    shape = []
    for element in elements:
        shape += [2] + list(element.index_shape)
    shape = tuple(shape)

    funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
    varexp = gem.Variable("A", shape)

    expressions = []
    for restrictions in product(((0,), (1,)), repeat=len(arguments)):
        is_ = tuple(chain(*chain(*zip(restrictions, indices))))
        expressions.append(gem.Indexed(varexp, is_))

    return funarg, expressions


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
