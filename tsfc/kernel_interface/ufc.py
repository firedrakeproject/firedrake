from __future__ import absolute_import

import numpy
from itertools import product

import coffee.base as coffee

import gem

from tsfc.kernel_interface.common import KernelBuilderBase
from tsfc.fiatinterface import create_element
from tsfc.coffee import SCALAR_TYPE, cumulative_strides


class KernelBuilder(KernelBuilderBase):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_type, subdomain_id, domain_number):
        """Initialise a kernel builder."""
        super(KernelBuilder, self).__init__(integral_type.startswith("interior_facet"))
        self.integral_type = integral_type

        self.local_tensor = None
        self.coordinates_args = None
        self.coefficient_args = None
        self.coefficient_split = None

        if self.interior_facet:
            self._cell_orientations = (gem.Variable("cell_orientation_0", ()),
                                       gem.Variable("cell_orientation_1", ()))
        else:
            self._cell_orientations = (gem.Variable("cell_orientation", ()),)

        if integral_type == "exterior_facet":
            self._facet_number = {None: gem.VariableIndex(gem.Variable("facet", ()))}
        elif integral_type == "interior_facet":
            self._facet_number = {
                '+': gem.VariableIndex(gem.Variable("facet_0", ())),
                '-': gem.VariableIndex(gem.Variable("facet_1", ()))
            }

    def set_arguments(self, arguments, indices):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg indices: GEM argument indices
        :returns: GEM expression representing the return variable
        """
        self.local_tensor, prepare, expressions = prepare_arguments(
            arguments, indices, interior_facet=self.interior_facet)
        self.apply_glue(prepare)
        return expressions

    def set_coordinates(self, coefficient, mode=None):
        """Prepare the coordinate field.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg mode: (ignored)
        """
        self.coordinates_args, expression = prepare_coordinates(
            coefficient, "coordinate_dofs", interior_facet=self.interior_facet)
        self.coefficient_map[coefficient] = expression

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

        funarg, expressions = prepare_coefficients(
            coefficients, coefficient_numbers, "w",
            interior_facet=self.interior_facet)

        self.coefficient_args = [funarg]
        for i, coefficient in enumerate(coefficients):
            self.coefficient_map[coefficient] = expressions[i]

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

        # Facet number(s)
        if self.integral_type == "exterior_facet":
            args.append(coffee.Decl("std::size_t", coffee.Symbol("facet")))
        elif self.integral_type == "interior_facet":
            args.append(coffee.Decl("std::size_t", coffee.Symbol("facet_0")))
            args.append(coffee.Decl("std::size_t", coffee.Symbol("facet_1")))

        # Cell orientation(s)
        if self.interior_facet:
            args.append(coffee.Decl("int", coffee.Symbol("cell_orientation_0")))
            args.append(coffee.Decl("int", coffee.Symbol("cell_orientation_1")))
        else:
            args.append(coffee.Decl("int", coffee.Symbol("cell_orientation")))

        return KernelBuilderBase.construct_kernel(self, name, args, body)

    @staticmethod
    def require_cell_orientations():
        # Nothing to do
        pass

    @staticmethod
    def needs_cell_orientations(ir):
        # UFC tabulate_tensor always have cell orientations
        return True


def prepare_coefficients(coefficients, coefficient_numbers, name, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.  Mixed element Coefficients are rearranged here for
    interior facet integrals.

    :arg coefficient: iterable of UFL Coefficients
    :arg coefficient_numbers: iterable of coefficient indices in the original form
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expressions)
         funarg     - :class:`coffee.Decl` function argument
         expressions- GEM expressions referring to the Coefficient
                      values
    """
    assert len(coefficients) == len(coefficient_numbers)

    funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                         pointers=[("const",), ()],
                         qualifiers=["const"])

    varexp = gem.Variable(name, (None, None))
    expressions = []
    for coefficient_number, coefficient in zip(coefficient_numbers, coefficients):
        if coefficient.ufl_element().family() == 'Real':
            shape = coefficient.ufl_shape
        else:
            fiat_element = create_element(coefficient.ufl_element())
            shape = (fiat_element.space_dimension(),)
            if interior_facet:
                shape = (2,) + shape

        alpha = tuple(gem.Index() for s in shape)
        expressions.append(gem.ComponentTensor(
            gem.FlexiblyIndexed(
                varexp, ((coefficient_number, ()),
                         (0, tuple(zip(alpha, shape))))
            ),
            alpha
        ))

    return funarg, expressions


def prepare_coordinates(coefficient, name, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    coordinates.

    :arg coefficient: UFL Coefficient
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         funarg     - :class:`coffee.Decl` function argument
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

    return funargs, expression


def prepare_arguments(arguments, indices, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg arguments: UFL Arguments
    :arg indices: Argument indices
    :arg interior_facet: interior facet integral?
    :returns: (funarg, prepare, expressions)
         funarg      - :class:`coffee.Decl` function argument
         prepare     - list of COFFEE nodes to be prepended to the
                       kernel body
         expressions - GEM expressions referring to the argument
                       tensor
    """
    funarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A"), pointers=[()])
    varexp = gem.Variable("A", (None,))

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)
    space_dimensions = tuple(element.space_dimension() for element in elements)
    for i, sd in zip(indices, space_dimensions):
        i.set_extent(sd)

    if arguments and interior_facet:
        shape = tuple(2 * element.space_dimension() for element in elements)
        strides = cumulative_strides(shape)

        expressions = []
        for restrictions in product((0, 1), repeat=len(arguments)):
            offset = sum(r * sd * stride
                         for r, sd, stride in zip(restrictions, space_dimensions, strides))
            expressions.append(gem.FlexiblyIndexed(
                varexp,
                ((offset,
                  tuple((i, s) for i, s in zip(indices, shape))),)
            ))
    else:
        shape = space_dimensions
        expressions = [gem.FlexiblyIndexed(varexp, ((0, tuple(zip(indices, space_dimensions))),))]

    zero = coffee.FlatBlock(
        str.format("memset({name}, 0, {size} * sizeof(*{name}));\n",
                   name=funarg.sym.gencode(), size=numpy.product(shape, dtype=int))
    )
    return funarg, [zero], expressions
