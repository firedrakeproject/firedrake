import numpy
import functools
from itertools import chain, product

import coffee.base as coffee

import gem
from gem.optimise import remove_componenttensors as prune

from finat import TensorFiniteElement

import ufl

from tsfc.kernel_interface.common import KernelBuilderBase, KernelBuilderMixin, get_index_names
from tsfc.finatinterface import create_element as _create_element


# UFC DoF ordering for vector/tensor elements is XXXX YYYY ZZZZ.
create_element = functools.partial(_create_element, shape_innermost=False)


class KernelBuilder(KernelBuilderBase, KernelBuilderMixin):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_data_info, scalar_type, diagonal=False):
        """Initialise a kernel builder."""
        integral_type = integral_data_info.integral_type
        if diagonal:
            raise NotImplementedError("Assembly of diagonal not implemented yet, sorry")
        super(KernelBuilder, self).__init__(scalar_type, integral_type.startswith("interior_facet"))
        self.fem_scalar_type = scalar_type
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
            self._entity_number = {None: gem.VariableIndex(gem.Variable("facet", ()))}
        elif integral_type == "interior_facet":
            self._entity_number = {
                '+': gem.VariableIndex(gem.Variable("facet_0", ())),
                '-': gem.VariableIndex(gem.Variable("facet_1", ()))
            }
        elif integral_type == "vertex":
            self._entity_number = {None: gem.VariableIndex(gem.Variable("vertex", ()))}

        self.set_arguments(integral_data_info.arguments)
        self.integral_data_info = integral_data_info

    def set_arguments(self, arguments):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg multiindices: GEM argument multiindices
        :returns: GEM expression representing the return variable
        """
        argument_multiindices = tuple(create_element(arg.ufl_element()).get_indices()
                                      for arg in arguments)
        if self.diagonal:
            # Error checking occurs in the builder constructor.
            # Diagonal assembly is obtained by using the test indices for
            # the trial space as well.
            a, _ = argument_multiindices
            argument_multiindices = (a, a)
        local_tensor, prepare, return_variables = prepare_arguments(arguments,
                                                                    argument_multiindices,
                                                                    self.scalar_type,
                                                                    interior_facet=self.interior_facet)
        self.apply_glue(prepare)
        self.local_tensor = local_tensor
        self.return_variables = return_variables
        self.argument_multiindices = argument_multiindices

    def set_coordinates(self, domain):
        """Prepare the coordinate field.

        :arg domain: :class:`ufl.Domain`
        """
        # Create a fake coordinate coefficient for a domain.
        f = ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))
        self.domain_coordinate[domain] = f
        self.coordinates_args, expression = prepare_coordinates(
            f, "coordinate_dofs", self.scalar_type, interior_facet=self.interior_facet)
        self.coefficient_map[f] = expression

    def set_cell_sizes(self, domain):
        """Prepare cell sizes field.

        :arg domain: :class:`ufl.Domain`
        """
        pass

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        name = "w"
        self.coefficient_args = [
            coffee.Decl(self.scalar_type, coffee.Symbol(name),
                        pointers=[("const",), ()],
                        qualifiers=["const"])
        ]

        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        for n in range(len(integral_data.enabled_coefficients)):
            if not integral_data.enabled_coefficients[n]:
                continue

            coefficient = form_data.function_replace_map[form_data.reduced_coefficients[n]]
            expression = prepare_coefficient(coefficient, n, name, self.interior_facet)
            self.coefficient_map[coefficient] = expression

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        return None, None, None

    def construct_kernel(self, name, ctx):
        """Construct a fully built kernel function.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: kernel name
        :arg ctx: kernel builder context to get impero_c from
        :arg quadrature rule: quadrature rule (not used, stubbed out for Themis integration)
        :returns: a COFFEE function definition object
        """
        from tsfc.coffee import generate as generate_coffee

        impero_c, _, _, _, _ = self.compile_gem(ctx)
        if impero_c is None:
            return self.construct_empty_kernel(name)
        index_names = get_index_names(ctx['quadrature_indices'], self.argument_multiindices, ctx['index_cache'])
        body = generate_coffee(impero_c, index_names, scalar_type=self.scalar_type)
        return self._construct_kernel_from_body(name, body)

    def _construct_kernel_from_body(self, name, body):
        """Construct a fully built kernel function.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: function name
        :arg body: function body (:class:`coffee.Block` node)
        :arg quadrature rule: quadrature rule (ignored)
        :returns: a COFFEE function definition object
        """
        args = [self.local_tensor]
        args.extend(self.coefficient_args)
        args.extend(self.coordinates_args)

        # Facet and vertex number(s)
        if self.integral_type == "exterior_facet":
            args.append(coffee.Decl("std::size_t", coffee.Symbol("facet")))
        elif self.integral_type == "interior_facet":
            args.append(coffee.Decl("std::size_t", coffee.Symbol("facet_0")))
            args.append(coffee.Decl("std::size_t", coffee.Symbol("facet_1")))
        elif self.integral_type == "vertex":
            args.append(coffee.Decl("std::size_t", coffee.Symbol("vertex")))

        # Cell orientation(s)
        if self.interior_facet:
            args.append(coffee.Decl("int", coffee.Symbol("cell_orientation_0")))
            args.append(coffee.Decl("int", coffee.Symbol("cell_orientation_1")))
        else:
            args.append(coffee.Decl("int", coffee.Symbol("cell_orientation")))

        return KernelBuilderBase.construct_kernel(self, name, args, body)

    def construct_empty_kernel(self, name):
        """Construct an empty kernel function.

        Kernel will just zero the return buffer and do nothing else.

        :arg name: function name
        :returns: a COFFEE function definition object
        """
        body = coffee.Block([])  # empty block
        return self._construct_kernel_from_body(name, body)

    def create_element(self, element, **kwargs):
        """Create a FInAT element (suitable for tabulating with) given
        a UFL element."""
        return create_element(element, **kwargs)


def prepare_coefficient(coefficient, num, name, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg coefficient: UFL Coefficient
    :arg num: coefficient index in the original form
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: GEM expression referring to the Coefficient value
    """
    varexp = gem.Variable(name, (None, None))

    if coefficient.ufl_element().family() == 'Real':
        size = numpy.prod(coefficient.ufl_shape, dtype=int)
        data = gem.view(varexp, slice(num, num + 1), slice(size))
        return gem.reshape(data, (), coefficient.ufl_shape)

    element = create_element(coefficient.ufl_element())
    size = numpy.prod(element.index_shape, dtype=int)

    def expression(data):
        result, = prune([gem.reshape(gem.view(data, slice(size)), element.index_shape)])
        return result

    if not interior_facet:
        data = gem.view(varexp, slice(num, num + 1), slice(size))
        return expression(gem.reshape(data, (), (size,)))
    else:
        data_p = gem.view(varexp, slice(num, num + 1), slice(size))
        data_m = gem.view(varexp, slice(num, num + 1), slice(size, 2 * size))
        return (expression(gem.reshape(data_p, (), (size,))),
                expression(gem.reshape(data_m, (), (size,))))


def prepare_coordinates(coefficient, name, scalar_type, interior_facet=False):
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

    assert isinstance(finat_element, TensorFiniteElement)
    scalar_shape = finat_element.base_element.index_shape
    tensor_shape = finat_element._shape
    transposed_shape = scalar_shape + tensor_shape
    scalar_rank = len(scalar_shape)

    def transpose(expr):
        indices = tuple(gem.Index(extent=extent) for extent in expr.shape)
        transposed_indices = indices[scalar_rank:] + indices[:scalar_rank]
        return gem.ComponentTensor(gem.Indexed(expr, indices),
                                   transposed_indices)

    if not interior_facet:
        funargs = [coffee.Decl(scalar_type, coffee.Symbol(name),
                               pointers=[("",)],
                               qualifiers=["const"])]
        variable = gem.Variable(name, (size,))
        expression = transpose(gem.reshape(variable, transposed_shape))
    else:
        funargs = [coffee.Decl(scalar_type, coffee.Symbol(name+"_0"),
                               pointers=[("",)],
                               qualifiers=["const"]),
                   coffee.Decl(scalar_type, coffee.Symbol(name+"_1"),
                               pointers=[("",)],
                               qualifiers=["const"])]
        variable0 = gem.Variable(name+"_0", (size,))
        variable1 = gem.Variable(name+"_1", (size,))
        expression = (transpose(gem.reshape(variable0, transposed_shape)),
                      transpose(gem.reshape(variable1, transposed_shape)))

    return funargs, expression


def prepare_arguments(arguments, multiindices, scalar_type, interior_facet=False):
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
    funarg = coffee.Decl(scalar_type, coffee.Symbol("A"), pointers=[()])
    varexp = gem.Variable("A", (None,))

    if len(arguments) == 0:
        # No arguments
        zero = coffee.FlatBlock(
            "memset({name}, 0, sizeof(*{name}));\n".format(name=funarg.sym.gencode())
        )
        return funarg, [zero], [gem.reshape(varexp, ())]

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)
    shapes = [element.index_shape for element in elements]
    indices = tuple(chain(*multiindices))

    def expression(restricted):
        return gem.Indexed(gem.reshape(restricted, *shapes), indices)

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
