from __future__ import absolute_import, print_function, division
from six.moves import range, zip

import numpy
from itertools import product

import coffee.base as coffee

import gem

from tsfc.kernel_interface.common import KernelBuilderBase
from tsfc.finatinterface import create_element
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

    def coefficient(self, ufl_coefficient, restriction):
        """A function that maps :class:`ufl.Coefficient`s to GEM
        expressions."""
        kernel_arg = self.coefficient_map[ufl_coefficient]
        if ufl_coefficient.ufl_element().family() == 'Real':
            return kernel_arg
        elif not isinstance(kernel_arg, tuple):
            return gem.partial_indexed(kernel_arg, {None: (), '+': (0,), '-': (1,)}[restriction])
        elif restriction == '+':
            return kernel_arg[0]
        elif restriction == '-':
            return kernel_arg[1]
        else:
            assert False

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
        cells_shape = ()
        tensor_shape = ()
        if coefficient.ufl_element().family() == 'Real':
            scalar_shape = coefficient.ufl_shape
        else:
            finat_element = create_element(coefficient.ufl_element())
            if hasattr(finat_element, '_base_element'):
                scalar_shape = finat_element._base_element.index_shape
                tensor_shape = finat_element.index_shape[len(scalar_shape):]
            else:
                scalar_shape = finat_element.index_shape

            if interior_facet:
                cells_shape = (2,)

        cells_indices = tuple(gem.Index() for s in cells_shape)
        tensor_indices = tuple(gem.Index() for s in tensor_shape)
        scalar_indices = tuple(gem.Index() for s in scalar_shape)
        shape = cells_shape + tensor_shape + scalar_shape
        alpha = cells_indices + tensor_indices + scalar_indices
        beta = cells_indices + scalar_indices + tensor_indices
        expressions.append(gem.ComponentTensor(
            gem.FlexiblyIndexed(
                varexp, ((coefficient_number, ()),
                         (0, tuple(zip(alpha, shape))))
            ),
            beta
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

    finat_element = create_element(coefficient.ufl_element())
    shape = finat_element.index_shape
    size = numpy.prod(shape, dtype=int)

    if not interior_facet:
        variable = gem.Variable(name, (size,))
        expression = gem.reshape(variable, shape)
    else:
        variable0 = gem.Variable(name+"_0", (size,))
        variable1 = gem.Variable(name+"_1", (size,))
        expression = (gem.reshape(variable0, shape),
                      gem.reshape(variable1, shape))

    return funargs, expression


def prepare_arguments(arguments, multiindices, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg arguments: UFL Arguments
    :arg multiindices: Argument multiindices
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
    ushape = tuple(numpy.prod(element.index_shape, dtype=int) for element in elements)

    # Flat indices and flat shape
    indices = []
    shape = []
    for element, multiindex in zip(elements, multiindices):
        if hasattr(element, '_base_element'):
            scalar_shape = element._base_element.index_shape
            tensor_shape = element.index_shape[len(scalar_shape):]
        else:
            scalar_shape = element.index_shape
            tensor_shape = ()

        haha = tensor_shape + scalar_shape
        if interior_facet and haha:
            haha = (haha[0] * 2,) + haha[1:]
        shape.extend(haha)
        indices.extend(multiindex[len(scalar_shape):] + multiindex[:len(scalar_shape)])
    indices = tuple(indices)
    shape = tuple(shape)

    if arguments and interior_facet:
        result_shape = tuple(2 * sd for sd in ushape)
        strides = cumulative_strides(result_shape)

        expressions = []
        for restrictions in product((0, 1), repeat=len(arguments)):
            offset = sum(r * sd * stride
                         for r, sd, stride in zip(restrictions, ushape, strides))
            expressions.append(gem.FlexiblyIndexed(
                varexp,
                ((offset,
                  tuple((i, s) for i, s in zip(indices, shape))),)
            ))
    else:
        result_shape = ushape
        expressions = [gem.FlexiblyIndexed(varexp, ((0, tuple(zip(indices, shape))),))]

    zero = coffee.FlatBlock(
        str.format("memset({name}, 0, {size} * sizeof(*{name}));\n",
                   name=funarg.sym.gencode(), size=numpy.product(result_shape, dtype=int))
    )
    return funarg, [zero], expressions
