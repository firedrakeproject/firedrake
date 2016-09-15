from __future__ import absolute_import

import os
import six
import numpy
from itertools import chain, product

import coffee.base as coffee

import gem
from gem.node import traversal
from gem.gem import FlexiblyIndexed as gem_FlexiblyIndexed

from tsfc.fiatinterface import create_element
from tsfc.mixedelement import MixedElement
from tsfc.coffee import SCALAR_TYPE


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

        # Coefficients
        self.coefficient_map = {}

    def coefficient(self, ufl_coefficient, restriction):
        """A function that maps :class:`ufl.Coefficient`s to GEM
        expressions."""
        kernel_arg = self.coefficient_map[ufl_coefficient]
        if ufl_coefficient.ufl_element().family() == 'Real':
            return kernel_arg
        else:
            return gem.partial_indexed(kernel_arg, {None: (), '+': (0,), '-': (1,)}[restriction])

    def cell_orientation(self, restriction):
        """Cell orientation as a GEM expression."""
        f = {None: 0, '+': 0, '-': 1}[restriction]
        co_int = self.cell_orientations_mapper[f]
        return gem.Conditional(gem.Comparison("==", co_int, gem.Literal(1)),
                               gem.Literal(-1),
                               gem.Conditional(gem.Comparison("==", co_int, gem.Zero()),
                                               gem.Literal(1),
                                               gem.Literal(numpy.nan)))

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
        return coffee.FunDecl("void", name, args, body_, pred=["virtual"])

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

    def coefficients(self, coefficients, coefficient_numbers, name, mode=None):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: iterable of :class:`ufl.Coefficient`s
        :arg coefficient_numbers: iterable of coefficient indices in the original form
        :arg name: coefficient name
        :arg mode: see :func:`prepare_coefficient`
        :returns: COFFEE function argument for the coefficient
        """
        funarg, prepare, expressions = prepare_coefficients(
            coefficients, coefficient_numbers, name, mode=mode,
            interior_facet=self.interior_facet)
        self.apply_glue(prepare)
        for i, coefficient in enumerate(coefficients):
            self.coefficient_map[coefficient] = expressions[i]
        return funarg

    def coordinates(self, coefficient, name, mode=None):
        """Prepare a coordinates. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :arg mode: see :func:`prepare_coefficient`
        :returns: COFFEE function arguments for the coefficient
        """
        funargs, prepare, expression = prepare_coordinates(
            coefficient, name, mode=mode,
            interior_facet=self.interior_facet)
        self.apply_glue(prepare)
        self.coefficient_map[coefficient] = expression
        return funargs

    def facets(self, integral_type):
        """Prepare facets. Adds glue code for facets
        and stores facet expression.

        :arg integral_type
        :returns: list of COFFEE function arguments for facets
        """
        funargs, prepare, expressions = prepare_facets(integral_type)
        self.apply_glue(prepare)
        self.facet_mapper = expressions
        return funargs

    def cell_orientations(self, integral_type):
        """Prepare cell orientations. Adds glue code for cell orienatations
        and stores cell orientations expression.

        :arg integral_type
        :returns: list of COFFEE function arguments for cell orientations
        """
        funargs, prepare, expressions = prepare_cell_orientations(integral_type)
        self.apply_glue(prepare)
        self.cell_orientations_mapper = expressions
        return funargs


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
        self.cell_orientations_args = []

    def facet_number(self, restriction):
        """Facet number as a GEM index."""
        f = {None: 0, '+': 0, '-': 1}[restriction]
        return self.facet_mapper[f]

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
        self.coordinates_args = self.coordinates(coefficient, name, mode)

    def set_facets(self):
        """Prepare the facets.
        """
        self.facet_args = self.facets(self.kernel.integral_type)

    def set_cell_orientations(self):
        """Prepare the cell orientations.
        """
        self.cell_orientations_args = self.cell_orientations(self.kernel.integral_type)

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
                coefficients.append(coefficient)
                # This is which coefficient in the original form the
                # current coefficient is.
                # Consider f*v*dx + g*v*ds, the full form contains two
                # coefficients, but each integral only requires one.
                coefficient_numbers.append(i)
        self.coefficient_args.append(self.coefficients(coefficients, coefficient_numbers, "w"))
        self.kernel.coefficient_numbers = tuple(coefficient_numbers)

    def require_cell_orientations(self):
        """Set that the kernel requires cell orientations."""
        # FIXME: Don't need this in UFC
        self.kernel.oriented = True

    def construct_kernel(self, name, body):
        """Construct a fully built :class:`Kernel`.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: function name
        :arg body: function body (:class:`coffee.Block` node)
        :returns: :class:`Kernel` object
        """
        args = [self.local_tensor]
        args.extend(self.coefficient_args)
        args.extend(self.coordinates_args)
        args.extend(self.facet_args)
        args.extend(self.cell_orientations_args)

        self.kernel.ast = KernelBuilderBase.construct_kernel(self, name, args, body)
        self.kernel.ast = KernelBuilderBase.construct_kernel(self, name, args, body)
        return self.kernel



def prepare_coefficients(coefficients, coefficient_numbers, name, mode=None,
                         interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.  Mixed element Coefficients are rearranged here for
    interior facet integrals.

    :arg coefficient: iterable of UFL Coefficients
    :arg coefficient_numbers: iterable of coefficient indices in the original form
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg mode: 'manual_loop' or 'list_tensor'; two ways to deal with
               interior facet integrals on mixed elements
    :arg interior_facet: interior facet integral?
    :returns: (funarg, prepare, expressions)
         funarg     - :class:`coffee.Decl` function argument
         prepare    - list of COFFEE nodes to be prepended to the
                      kernel body
         expressions- GEM expressions referring to the Coefficient
                      values
    """
    assert len(coefficients) == len(coefficient_numbers)

    # FIXME: hack; is actual number really needed?
    num_coefficients = max(coefficient_numbers) + 1 if coefficient_numbers else 0
    funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                         pointers=[("const",), ()],
                         qualifiers=["const"])

    # FIXME for interior facets
    expressions = []
    for j, coefficient in enumerate(coefficients):
        if coefficient.ufl_element().family() == 'Real':
            shape = coefficient.ufl_shape
            if shape == ():
                # Scalar constant/real - needs one dummy index
                expression = gem.Indexed(gem.Variable(name, (num_coefficients,) + (1,)),
                                         (coefficient_numbers[j], 0,))
                # FIXME: It seems that Reals are not restricted in gem but are in UFL.
                #if interior_facet:
                #    i = gem.Index()
                #    expression = gem.ComponentTensor(
                #        gem.Indexed(gem.Variable(name, (num_coefficients,) + (2,)),
                #                    (coefficient_numbers[j], i,)),
                #        (i,))
            else:
                # Mixed/vector/tensor constant/real
                # FIXME: Tensor case is incorrect. Gem wants shaped expression, UFC requires flattened.
                indices = tuple(gem.Index() for i in six.moves.xrange(len(shape)))
                expression = gem.ComponentTensor(
                    gem.Indexed(gem.Variable(name, (num_coefficients,) + shape),
                                (coefficient_numbers[j],) + indices),
                    indices)
        else:
            # Everything else
            i = gem.Index()
            fiat_element = create_element(coefficient.ufl_element())
            shape = (fiat_element.space_dimension(),)
            expression = gem.ComponentTensor(
                gem.Indexed(gem.Variable(name, (num_coefficients,) + shape),
                            (coefficient_numbers[j], i)),
                (i,))
            if interior_facet:
                num_dofs = shape[0]
                variable = gem.Variable(name, (num_coefficients, 2*num_dofs))
                # TODO: Seems that this reordering could be done using reinterpret_cast
                expression = gem.ListTensor([[gem.Indexed(variable, (coefficient_numbers[j], i))
                                              for i in six.moves.xrange(       0,   num_dofs)],
                                             [gem.Indexed(variable, (coefficient_numbers[j], i))
                                              for i in six.moves.xrange(num_dofs, 2*num_dofs)]])

        expressions.append(expression)

    return funarg, [], expressions


def prepare_coordinates(coefficient, name, mode=None, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    coordinates.

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
    if not interior_facet:
        funargs = [coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                               pointers=[("",)],
                               qualifiers=["const"])]
    else:
        funargs = [coffee.Decl(SCALAR_TYPE, coffee.Symbol(name+"_0"),
                               pointers=[("",)],
                               qualifiers=["const"]),
                   coffee.Decl(SCALAR_TYPE, coffee.Symbol(name+"_1"),
                               pointers=[("",)],
                               qualifiers=["const"])]

    fiat_element = create_element(coefficient.ufl_element())
    shape = (fiat_element.space_dimension(),)
    gdim = coefficient.ufl_element().cell().geometric_dimension()
    assert len(shape) == 1 and shape[0] % gdim == 0
    num_nodes = shape[0] / gdim

    # Translate coords from XYZXYZXYZXYZ into XXXXYYYYZZZZ
    # NOTE: See dolfin/mesh/Cell.h:get_coordinate_dofs for ordering scheme
    indices = numpy.arange(num_nodes * gdim).reshape(num_nodes, gdim).transpose().flatten()
    if not interior_facet:
        variable = gem.Variable(name, shape)
        expression = gem.ListTensor([gem.Indexed(variable, (i,)) for i in indices])
    else:
        variable0 = gem.Variable(name+"_0", shape)
        variable1 = gem.Variable(name+"_1", shape)
        expression = gem.ListTensor([[gem.Indexed(variable0, (i,)) for i in indices],
                                     [gem.Indexed(variable1, (i,)) for i in indices]])

    return funargs, [], expression


def prepare_facets(integral_type):
    """Bridges the kernel interface and the GEM abstraction for
    facets.

    :arg integral_type
    :returns: (funarg, prepare, expression)
         funargs    - list of :class:`coffee.Decl` function argument
         prepare    - list of COFFEE nodes to be prepended to the
                      kernel body
         expressions- list of GEM expressions referring to facets
    """
    funargs = []
    expressions = []

    if integral_type in ["exterior_facet", "exterior_facet_vert"]:
            funargs.append(coffee.Decl("std::size_t", coffee.Symbol("facet")))
            expressions.append(gem.VariableIndex(gem.Variable("facet", ())))
    elif integral_type in ["interior_facet", "interior_facet_vert"]:
            funargs.append(coffee.Decl("std::size_t", coffee.Symbol("facet_0")))
            funargs.append(coffee.Decl("std::size_t", coffee.Symbol("facet_1")))
            expressions.append(gem.VariableIndex(gem.Variable("facet_0", ())))
            expressions.append(gem.VariableIndex(gem.Variable("facet_1", ())))

    return funargs, [], expressions


def prepare_cell_orientations(integral_type):
    """Bridges the kernel interface and the GEM abstraction for
    cell orientations.

    :arg integral_type
    :returns: (funarg, prepare, expression)
         funargs    - list of :class:`coffee.Decl` function argument
         prepare    - list of COFFEE nodes to be prepended to the
                      kernel body
         expressions- list of GEM expressions referring to facets
    """
    funargs = []
    expressions = []

    if integral_type in ["interior_facet", "interior_facet_vert"]:
            funargs.append(coffee.Decl("int", coffee.Symbol("cell_orientation_0")))
            funargs.append(coffee.Decl("int", coffee.Symbol("cell_orientation_1")))
            expressions.append(gem.Variable("cell_orientation_0", ()))
            expressions.append(gem.Variable("cell_orientation_1", ()))
    else:
            funargs.append(coffee.Decl("int", coffee.Symbol("cell_orientation")))
            expressions.append(gem.Variable("cell_orientation", ()))

    return funargs, [], expressions


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
    funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A"), pointers=[()])

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)
    shape = tuple(element.space_dimension() for element in elements)
    if len(arguments) == 0:
        shape = (1,)
        indices = (0,)
    if interior_facet:
        shape = tuple(j for i in zip(len(shape)*(2,), shape) for j in i)
        indices = tuple(product(*chain(*(((0, 1), (i,)) for i in indices))))
    else:
        indices = (indices,)

    expressions = [gem.Indexed(gem.Variable("AA", shape), i) for i in indices]

    reshape = coffee.Decl(SCALAR_TYPE,
                          coffee.Symbol("(&%s)" % expressions[0].children[0].name,
                                        rank=shape),
                          init="*reinterpret_cast<%s (*)%s>(%s)" %
                              (SCALAR_TYPE,
                               "".join("[%s]"%i for i in shape),
                               funarg.sym.gencode()
                              )
                          )
    zero = coffee.FlatBlock("memset(%s, 0, %d * sizeof(*%s));%s" %
        (funarg.sym.gencode(), numpy.product(shape), funarg.sym.gencode(), os.linesep))
    prepare = [zero, reshape]

    return funarg, prepare, expressions, []



def needs_cell_orientations(ir):
    return True
