from __future__ import absolute_import

import os
import six
import numpy
from itertools import chain, product

import coffee.base as coffee

import gem

from tsfc.kernel_interface import Kernel, KernelBuilderBase
from tsfc.fiatinterface import create_element
from tsfc.coffee import SCALAR_TYPE


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

    def cell_orientations_mapper(self, facet):
        return self._cell_orientations[facet]

    def coefficients(self, coefficients, coefficient_numbers, name, mode=None):
        """Prepare coefficients. Adds glue code for the coefficients
        and adds the coefficients to the coefficient map.

        :arg coefficient: iterable of :class:`ufl.Coefficient`s
        :arg coefficient_numbers: iterable of coefficient indices in the original form
        :arg name: coefficient name
        :arg mode: see :func:`prepare_coefficient`
        :returns: COFFEE function argument for the coefficient
        """
        funarg, prepare, expressions = _prepare_coefficients(
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
        funargs, prepare, expression = _prepare_coordinates(
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
        funargs, prepare, expressions = _prepare_facets(integral_type)
        self.apply_glue(prepare)
        self.facet_mapper = expressions
        return funargs


    def set_arguments(self, arguments, indices):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg indices: GEM argument indices
        :returns: GEM expression representing the return variable
        """
        self.local_tensor, expressions = self.arguments(arguments, indices)
        return expressions

    def set_coordinates(self, coefficient, mode=None):
        """Prepare the coordinate field.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg mode: see :func:`prepare_coefficient`
        """
        self.coordinates_args = self.coordinates(coefficient, "coordinate_dofs", mode)

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
        return self.kernel


    @staticmethod
    def needs_cell_orientations(ir):
        """UFC requires cell orientations argument(s) everytime"""
        return True

    @staticmethod
    def prepare_arguments(arguments, indices, interior_facet=False):
        return _prepare_arguments(arguments, indices, interior_facet=interior_facet)

    @staticmethod
    def prepare_cell_orientations(integral_type):
        return _prepare_cell_orientations(integral_type)


def _prepare_coefficients(coefficients, coefficient_numbers, name, mode=None,
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


def _prepare_coordinates(coefficient, name, mode=None, interior_facet=False):
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


def _prepare_facets(integral_type):
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


def _prepare_cell_orientations(integral_type):
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


def _prepare_arguments(arguments, indices, interior_facet=False):
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
