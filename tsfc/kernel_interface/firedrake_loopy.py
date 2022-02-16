import numpy
from collections import namedtuple
from itertools import chain, product
from functools import partial

from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace, FiniteElement

import gem
from gem.flop_count import count_flops
from gem.optimise import remove_componenttensors as prune

import loopy as lp

from tsfc import kernel_args
from tsfc.finatinterface import create_element
from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase, KernelBuilderMixin, get_index_names
from tsfc.kernel_interface.firedrake import check_requirements
from tsfc.loopy import generate as generate_loopy


# Expression kernel description type
ExpressionKernel = namedtuple('ExpressionKernel', ['ast', 'oriented', 'needs_cell_sizes',
                                                   'coefficient_numbers',
                                                   'first_coefficient_fake_coords',
                                                   'tabulations', 'name', 'arguments',
                                                   'flop_count'])


def make_builder(*args, **kwargs):
    return partial(KernelBuilder, *args, **kwargs)


class Kernel:
    __slots__ = ("ast", "arguments", "integral_type", "oriented", "subdomain_id",
                 "domain_number", "needs_cell_sizes", "tabulations",
                 "coefficient_numbers", "name", "flop_count",
                 "__weakref__")
    """A compiled Kernel object.

    :kwarg ast: The loopy kernel object.
    :kwarg integral_type: The type of integral.
    :kwarg oriented: Does the kernel require cell_orientations.
    :kwarg subdomain_id: What is the subdomain id for this kernel.
    :kwarg domain_number: Which domain number in the original form
        does this kernel correspond to (can be used to index into
        original_form.ufl_domains() to get the correct domain).
    :kwarg coefficient_numbers: A list of which coefficients from the
        form the kernel needs.
    :kwarg tabulations: The runtime tabulations this kernel requires
    :kwarg needs_cell_sizes: Does the kernel require cell sizes.
    :kwarg name: The name of this kernel.
    :kwarg flop_count: Estimated total flops for this kernel.
    """
    def __init__(self, ast=None, arguments=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None,
                 coefficient_numbers=(),
                 needs_cell_sizes=False,
                 tabulations=None,
                 flop_count=0,
                 name=None):
        # Defaults
        self.ast = ast
        self.arguments = arguments
        self.integral_type = integral_type
        self.oriented = oriented
        self.domain_number = domain_number
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        self.needs_cell_sizes = needs_cell_sizes
        self.tabulations = tabulations
        self.flop_count = flop_count
        self.name = name


class KernelBuilderBase(_KernelBuilderBase):

    def __init__(self, scalar_type, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        super().__init__(scalar_type=scalar_type, interior_facet=interior_facet)

        # Cell orientation
        if self.interior_facet:
            shape = (2,)
            cell_orientations = gem.Variable("cell_orientations", shape)
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),
                                       gem.Indexed(cell_orientations, (1,)))
        else:
            shape = (1,)
            cell_orientations = gem.Variable("cell_orientations", shape)
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),)
        loopy_arg = lp.GlobalArg("cell_orientations", dtype=numpy.int32, shape=shape)
        self.cell_orientations_arg = kernel_args.CellOrientationsKernelArg(loopy_arg)

    def _coefficient(self, coefficient, name):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :returns: loopy argument for the coefficient
        """
        funarg, expression = prepare_coefficient(coefficient, name, self.scalar_type, interior_facet=self.interior_facet)
        self.coefficient_map[coefficient] = expression
        return funarg

    def set_cell_sizes(self, domain):
        """Setup a fake coefficient for "cell sizes".

        :arg domain: The domain of the integral.

        This is required for scaling of derivative basis functions on
        physically mapped elements (Argyris, Bell, etc...).  We need a
        measure of the mesh size around each vertex (hence this lives
        in P1).

        Should the domain have topological dimension 0 this does
        nothing.
        """
        if domain.ufl_cell().topological_dimension() > 0:
            # Can't create P1 since only P0 is a valid finite element if
            # topological_dimension is 0 and the concept of "cell size"
            # is not useful for a vertex.
            f = Coefficient(FunctionSpace(domain, FiniteElement("P", domain.ufl_cell(), 1)))
            funarg, expression = prepare_coefficient(f, "cell_sizes", self.scalar_type, interior_facet=self.interior_facet)
            self.cell_sizes_arg = kernel_args.CellSizesKernelArg(funarg)
            self._cell_sizes = expression

    def create_element(self, element, **kwargs):
        """Create a FInAT element (suitable for tabulating with) given
        a UFL element."""
        return create_element(element, **kwargs)


class ExpressionKernelBuilder(KernelBuilderBase):
    """Builds expression kernels for UFL interpolation in Firedrake."""

    def __init__(self, scalar_type):
        super(ExpressionKernelBuilder, self).__init__(scalar_type=scalar_type)
        self.oriented = False
        self.cell_sizes = False

    def set_coefficients(self, coefficients):
        """Prepare the coefficients of the expression.

        :arg coefficients: UFL coefficients from Firedrake
        """
        self.coefficient_split = {}
        self.kernel_args = []

        for i, coefficient in enumerate(coefficients):
            if type(coefficient.ufl_element()) == ufl_MixedElement:
                subcoeffs = coefficient.split()  # Firedrake-specific
                self.coefficient_split[coefficient] = subcoeffs
                coeff_loopy_args = [self._coefficient(subcoeff, f"w_{i}_{j}")
                                    for j, subcoeff in enumerate(subcoeffs)]
                self.kernel_args += [kernel_args.CoefficientKernelArg(a)
                                     for a in coeff_loopy_args]
            else:
                coeff_loopy_arg = self._coefficient(coefficient, f"w_{i}")
                self.kernel_args.append(kernel_args.CoefficientKernelArg(coeff_loopy_arg))

    def set_coefficient_numbers(self, coefficient_numbers):
        """Store the coefficient indices of the original form.

        :arg coefficient_numbers: Iterable of indices describing which coefficients
            from the input expression need to be passed in to the kernel.
        """
        self.coefficient_numbers = coefficient_numbers

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        self.oriented, self.cell_sizes, self.tabulations = check_requirements(ir)

    def set_output(self, o):
        """Produce the kernel return argument"""
        loopy_arg = lp.GlobalArg(o.name, dtype=self.scalar_type, shape=o.shape)
        self.output_arg = kernel_args.OutputKernelArg(loopy_arg)

    def construct_kernel(self, impero_c, index_names, first_coefficient_fake_coords):
        """Constructs an :class:`ExpressionKernel`.

        :arg impero_c: gem.ImperoC object that represents the kernel
        :arg index_names: pre-assigned index names
        :arg first_coefficient_fake_coords: If true, the kernel's first
            coefficient is a constructed UFL coordinate field
        :returns: :class:`ExpressionKernel` object
        """
        args = [self.output_arg]
        if self.oriented:
            args.append(self.cell_orientations_arg)
        if self.cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.kernel_args)
        for name_, shape in self.tabulations:
            tab_loopy_arg = lp.GlobalArg(name_, dtype=self.scalar_type, shape=shape)
            args.append(kernel_args.TabulationKernelArg(tab_loopy_arg))

        loopy_args = [arg.loopy_arg for arg in args]

        name = "expression_kernel"
        loopy_kernel = generate_loopy(impero_c, loopy_args, self.scalar_type,
                                      name, index_names)
        return ExpressionKernel(loopy_kernel, self.oriented, self.cell_sizes,
                                self.coefficient_numbers, first_coefficient_fake_coords,
                                self.tabulations, name, args, count_flops(impero_c))


class KernelBuilder(KernelBuilderBase, KernelBuilderMixin):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_data_info, scalar_type,
                 dont_split=(), diagonal=False):
        """Initialise a kernel builder."""
        integral_type = integral_data_info.integral_type
        super(KernelBuilder, self).__init__(scalar_type, integral_type.startswith("interior_facet"))
        self.fem_scalar_type = scalar_type

        self.diagonal = diagonal
        self.local_tensor = None
        self.coordinates_arg = None
        self.coefficient_args = []
        self.coefficient_split = {}
        self.dont_split = frozenset(dont_split)

        # Facet number
        if integral_type in ['exterior_facet', 'exterior_facet_vert']:
            facet = gem.Variable('facet', (1,))
            self._entity_number = {None: gem.VariableIndex(gem.Indexed(facet, (0,)))}
        elif integral_type in ['interior_facet', 'interior_facet_vert']:
            facet = gem.Variable('facet', (2,))
            self._entity_number = {
                '+': gem.VariableIndex(gem.Indexed(facet, (0,))),
                '-': gem.VariableIndex(gem.Indexed(facet, (1,)))
            }
        elif integral_type == 'interior_facet_horiz':
            self._entity_number = {'+': 1, '-': 0}

        self.set_arguments(integral_data_info.arguments)
        self.integral_data_info = integral_data_info

    def set_arguments(self, arguments):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
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
        funarg, return_variables = prepare_arguments(arguments,
                                                     argument_multiindices,
                                                     self.scalar_type,
                                                     interior_facet=self.interior_facet,
                                                     diagonal=self.diagonal)
        self.output_arg = kernel_args.OutputKernelArg(funarg)
        self.return_variables = return_variables
        self.argument_multiindices = argument_multiindices

    def set_coordinates(self, domain):
        """Prepare the coordinate field.

        :arg domain: :class:`ufl.Domain`
        """
        # Create a fake coordinate coefficient for a domain.
        f = Coefficient(FunctionSpace(domain, domain.ufl_coordinate_element()))
        self.domain_coordinate[domain] = f
        coords_loopy_arg = self._coefficient(f, "coords")
        self.coordinates_arg = kernel_args.CoordinatesKernelArg(coords_loopy_arg)

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        coefficients = []
        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        for i in range(len(integral_data.enabled_coefficients)):
            if integral_data.enabled_coefficients[i]:
                original = form_data.reduced_coefficients[i]
                coefficient = form_data.function_replace_map[original]
                if type(coefficient.ufl_element()) == ufl_MixedElement:
                    if original in self.dont_split:
                        coefficients.append(coefficient)
                        self.coefficient_split[coefficient] = [coefficient]
                    else:
                        split = [Coefficient(FunctionSpace(coefficient.ufl_domain(), element))
                                 for element in coefficient.ufl_element().sub_elements()]
                        coefficients.extend(split)
                        self.coefficient_split[coefficient] = split
                else:
                    coefficients.append(coefficient)
        for i, coefficient in enumerate(coefficients):
            coeff_loopy_arg = self._coefficient(coefficient, f"w_{i}")
            self.coefficient_args.append(kernel_args.CoefficientKernelArg(coeff_loopy_arg))

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        return check_requirements(ir)

    def construct_kernel(self, name, ctx):
        """Construct a fully built :class:`Kernel`.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: kernel name
        :arg ctx: kernel builder context to get impero_c from
        :returns: :class:`Kernel` object
        """
        impero_c, oriented, needs_cell_sizes, tabulations = self.compile_gem(ctx)
        if impero_c is None:
            return self.construct_empty_kernel(name)
        info = self.integral_data_info
        args = [self.output_arg, self.coordinates_arg]
        if oriented:
            args.append(self.cell_orientations_arg)
        if needs_cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.coefficient_args)
        if info.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            ext_loopy_arg = lp.GlobalArg("facet", numpy.uint32, shape=(1,))
            args.append(kernel_args.ExteriorFacetKernelArg(ext_loopy_arg))
        elif info.integral_type in ["interior_facet", "interior_facet_vert"]:
            int_loopy_arg = lp.GlobalArg("facet", numpy.uint32, shape=(2,))
            args.append(kernel_args.InteriorFacetKernelArg(int_loopy_arg))
        for name_, shape in tabulations:
            tab_loopy_arg = lp.GlobalArg(name_, dtype=self.scalar_type, shape=shape)
            args.append(kernel_args.TabulationKernelArg(tab_loopy_arg))
        index_names = get_index_names(ctx['quadrature_indices'], self.argument_multiindices, ctx['index_cache'])
        ast = generate_loopy(impero_c, [arg.loopy_arg for arg in args],
                             self.scalar_type, name, index_names)
        flop_count = count_flops(impero_c)  # Estimated total flops for this kernel.
        return Kernel(ast=ast,
                      arguments=tuple(args),
                      integral_type=info.integral_type,
                      subdomain_id=info.subdomain_id,
                      domain_number=info.domain_number,
                      coefficient_numbers=info.coefficient_numbers,
                      oriented=oriented,
                      needs_cell_sizes=needs_cell_sizes,
                      tabulations=tabulations,
                      flop_count=flop_count,
                      name=name)

    def construct_empty_kernel(self, name):
        """Return None, since Firedrake needs no empty kernels.

        :arg name: function name
        :returns: None
        """
        return None


def prepare_coefficient(coefficient, name, scalar_type, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg coefficient: UFL Coefficient
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         funarg     - :class:`loopy.GlobalArg` function argument
         expression - GEM expression referring to the Coefficient
                      values
    """
    assert isinstance(interior_facet, bool)

    if coefficient.ufl_element().family() == 'Real':
        # Constant
        value_size = coefficient.ufl_element().value_size()
        funarg = lp.GlobalArg(name, dtype=scalar_type, shape=(value_size,))
        expression = gem.reshape(gem.Variable(name, (value_size,)),
                                 coefficient.ufl_shape)

        return funarg, expression

    finat_element = create_element(coefficient.ufl_element())

    shape = finat_element.index_shape
    size = numpy.prod(shape, dtype=int)

    if not interior_facet:
        expression = gem.reshape(gem.Variable(name, (size,)), shape)
    else:
        varexp = gem.Variable(name, (2*size,))
        plus = gem.view(varexp, slice(size))
        minus = gem.view(varexp, slice(size, 2*size))
        expression = (gem.reshape(plus, shape), gem.reshape(minus, shape))
        size = size * 2
    funarg = lp.GlobalArg(name, dtype=scalar_type, shape=(size,))
    return funarg, expression


def prepare_arguments(arguments, multiindices, scalar_type, interior_facet=False, diagonal=False):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg arguments: UFL Arguments
    :arg multiindices: Argument multiindices
    :arg interior_facet: interior facet integral?
    :arg diagonal: Are we assembling the diagonal of a rank-2 element tensor?
    :returns: (funarg, expression)
         funarg      - :class:`loopy.GlobalArg` function argument
         expressions - GEM expressions referring to the argument
                       tensor
    """

    assert isinstance(interior_facet, bool)

    if len(arguments) == 0:
        # No arguments
        funarg = lp.GlobalArg("A", dtype=scalar_type, shape=(1,))
        expression = gem.Indexed(gem.Variable("A", (1,)), (0,))

        return funarg, [expression]

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)
    shapes = tuple(element.index_shape for element in elements)

    if diagonal:
        if len(arguments) != 2:
            raise ValueError("Diagonal only for 2-forms")
        try:
            element, = set(elements)
        except ValueError:
            raise ValueError("Diagonal only for diagonal blocks (test and trial spaces the same)")

        elements = (element, )
        shapes = tuple(element.index_shape for element in elements)
        multiindices = multiindices[:1]

    def expression(restricted):
        return gem.Indexed(gem.reshape(restricted, *shapes),
                           tuple(chain(*multiindices)))

    u_shape = numpy.array([numpy.prod(shape, dtype=int) for shape in shapes])
    if interior_facet:
        c_shape = tuple(2 * u_shape)
        slicez = [[slice(r * s, (r + 1) * s)
                   for r, s in zip(restrictions, u_shape)]
                  for restrictions in product((0, 1), repeat=len(arguments))]
    else:
        c_shape = tuple(u_shape)
        slicez = [[slice(s) for s in u_shape]]

    funarg = lp.GlobalArg("A", dtype=scalar_type, shape=c_shape)
    varexp = gem.Variable("A", c_shape)
    expressions = [expression(gem.view(varexp, *slices)) for slices in slicez]
    return funarg, prune(expressions)
