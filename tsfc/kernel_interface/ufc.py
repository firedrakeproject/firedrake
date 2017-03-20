from __future__ import absolute_import, print_function, division
from six.moves import range, zip

import numpy
from itertools import product

from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace

import coffee.base as coffee

import gem
from gem.optimise import remove_componenttensors as prune

from finat import TensorFiniteElement

from tsfc.kernel_interface.common import KernelBuilderBase
from tsfc.finatinterface import create_element
from tsfc.coffee import SCALAR_TYPE


class KernelBuilder(KernelBuilderBase):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_type, subdomain_id, domain_number):
        """Initialise a kernel builder."""
        super(KernelBuilder, self).__init__(integral_type.startswith("interior_facet"))
        self.integral_type = integral_type

        self.local_tensor = None
        self.coordinates_args = None
        self.coefficient_args = None
        self.coefficient_split = {}

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

    def set_arguments(self, arguments, multiindices):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg multiindices: GEM argument multiindices
        :returns: GEM expression representing the return variable
        """
        self.local_tensor, prepare, expressions = prepare_arguments(
            arguments, multiindices, interior_facet=self.interior_facet)
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
        name = "w"
        self.coefficient_args = [
            coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                        pointers=[("const",), ()],
                        qualifiers=["const"])
        ]

        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        for n in range(len(integral_data.enabled_coefficients)):
            if not integral_data.enabled_coefficients[n]:
                continue

            coeff = form_data.reduced_coefficients[n]
            if type(coeff.ufl_element()) == ufl_MixedElement:
                coeffs = [Coefficient(FunctionSpace(coeff.ufl_domain(), element))
                          for element in coeff.ufl_element().sub_elements()]
                self.coefficient_split[coeff] = coeffs
            else:
                coeffs = [coeff]
            expressions = prepare_coefficients(coeffs, n, name, self.interior_facet)
            self.coefficient_map.update(zip(coeffs, expressions))

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


def prepare_coefficients(coefficients, num, name, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg coefficients: split UFL Coefficients
    :arg num: coefficient index in the original form
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: GEM expression referring to the Coefficient value
    """
    varexp = gem.Variable(name, (None, None))

    if len(coefficients) == 1 and coefficients[0].ufl_element().family() == 'Real':
        coefficient, = coefficients
        size = numpy.prod(coefficient.ufl_shape, dtype=int)
        data = gem.view(varexp, slice(num, num + 1), slice(size))
        expression = gem.reshape(data, (), coefficient.ufl_shape)
        return [expression]

    elements = [create_element(coeff.ufl_element()) for coeff in coefficients]
    space_dimensions = [numpy.prod(element.index_shape, dtype=int)
                        for element in elements]
    ends = list(numpy.cumsum(space_dimensions))
    starts = [0] + ends[:-1]
    slices = [slice(start, end) for start, end in zip(starts, ends)]

    transposed_shapes = []
    tensor_ranks = []
    for element in elements:
        if isinstance(element, TensorFiniteElement):
            scalar_shape = element.base_element.index_shape
            tensor_shape = element.index_shape[len(scalar_shape):]
        else:
            scalar_shape = element.index_shape
            tensor_shape = ()

        transposed_shapes.append(tensor_shape + scalar_shape)
        tensor_ranks.append(len(tensor_shape))

    def transpose(expr, rank):
        assert not expr.free_indices
        assert 0 <= rank < len(expr.shape)
        if rank == 0:
            return expr
        else:
            indices = tuple(gem.Index(extent=extent) for extent in expr.shape)
            transposed_indices = indices[rank:] + indices[:rank]
            return gem.ComponentTensor(gem.Indexed(expr, indices),
                                       transposed_indices)

    def expressions(data):
        return prune([transpose(gem.reshape(gem.view(data, slice_), shape), rank)
                      for slice_, shape, rank in zip(slices, transposed_shapes, tensor_ranks)])

    size = sum(space_dimensions)
    if not interior_facet:
        data = gem.view(varexp, slice(num, num + 1), slice(size))
        return expressions(gem.reshape(data, (), (size,)))
    else:
        data_p = gem.view(varexp, slice(num, num + 1), slice(size))
        data_m = gem.view(varexp, slice(num, num + 1), slice(size, 2 * size))
        return list(zip(expressions(gem.reshape(data_p, (), (size,))),
                        expressions(gem.reshape(data_m, (), (size,)))))


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
    finat_element = create_element(coefficient.ufl_element())
    shape = finat_element.index_shape
    size = numpy.prod(shape, dtype=int)

    if not interior_facet:
        funargs = [coffee.Decl(SCALAR_TYPE, coffee.Symbol(name),
                               pointers=[("",)],
                               qualifiers=["const"])]
        variable = gem.Variable(name, (size,))
        expression = gem.reshape(variable, shape)
    else:
        funargs = [coffee.Decl(SCALAR_TYPE, coffee.Symbol(name+"_0"),
                               pointers=[("",)],
                               qualifiers=["const"]),
                   coffee.Decl(SCALAR_TYPE, coffee.Symbol(name+"_1"),
                               pointers=[("",)],
                               qualifiers=["const"])]
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

    if len(arguments) == 0:
        # No arguments
        zero = coffee.FlatBlock(
            "memset({name}, 0, sizeof(*{name}));\n".format(name=funarg.sym.gencode())
        )
        return funarg, [zero], [gem.reshape(varexp, ())]

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)
    transposed_shapes = []
    transposed_indices = []
    for element, multiindex in zip(elements, multiindices):
        if isinstance(element, TensorFiniteElement):
            scalar_shape = element.base_element.index_shape
            tensor_shape = element.index_shape[len(scalar_shape):]
        else:
            scalar_shape = element.index_shape
            tensor_shape = ()

        transposed_shapes.append(tensor_shape + scalar_shape)
        scalar_rank = len(scalar_shape)
        transposed_indices.extend(multiindex[scalar_rank:] + multiindex[:scalar_rank])
    transposed_indices = tuple(transposed_indices)

    def expression(restricted):
        return gem.Indexed(gem.reshape(restricted, *transposed_shapes),
                           transposed_indices)

    u_shape = numpy.array([numpy.prod(element.index_shape, dtype=int)
                           for element in elements])
    if interior_facet:
        c_shape = tuple(2 * u_shape)
        slicez = [[slice(r * s, (r + 1) * s)
                   for r, s in zip(restrictions, u_shape)]
                  for restrictions in product((0, 1), repeat=len(arguments))]
    else:
        c_shape = tuple(u_shape)
        slicez = [[slice(s) for s in u_shape]]

    expressions = [expression(gem.view(gem.reshape(varexp, c_shape), *slices))
                   for slices in slicez]

    zero = coffee.FlatBlock(
        str.format("memset({name}, 0, {size} * sizeof(*{name}));\n",
                   name=funarg.sym.gencode(), size=numpy.product(c_shape, dtype=int))
    )
    return funarg, [zero], prune(expressions)
