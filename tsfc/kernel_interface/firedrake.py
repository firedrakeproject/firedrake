import numpy
from itertools import chain, product
from functools import partial

from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace, FiniteElement

import coffee.base as coffee

import gem
from gem.flop_count import count_flops
from gem.node import traversal
from gem.optimise import remove_componenttensors as prune

from tsfc import kernel_args
from tsfc.coffee import generate as generate_coffee
from tsfc.finatinterface import create_element
from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase, KernelBuilderMixin


def make_builder(*args, **kwargs):
    return partial(KernelBuilder, *args, **kwargs)


class Kernel(object):
    __slots__ = ("ast", "arguments", "integral_type", "oriented", "subdomain_id",
                 "domain_number", "needs_cell_sizes", "tabulations", "quadrature_rule",
                 "coefficient_numbers", "name", "__weakref__",
                 "flop_count")
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
    :kwarg quadrature_rule: The finat quadrature rule used to generate this kernel
    :kwarg tabulations: The runtime tabulations this kernel requires
    :kwarg needs_cell_sizes: Does the kernel require cell sizes.
    :kwarg flop_count: Estimated total flops for this kernel.
    """
    def __init__(self, ast=None, arguments=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None, quadrature_rule=None,
                 coefficient_numbers=(),
                 needs_cell_sizes=False,
                 flop_count=0):
        # Defaults
        self.ast = ast
        self.arguments = arguments
        self.integral_type = integral_type
        self.oriented = oriented
        self.domain_number = domain_number
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        self.needs_cell_sizes = needs_cell_sizes
        self.flop_count = flop_count
        super(Kernel, self).__init__()


class KernelBuilderBase(_KernelBuilderBase):

    def __init__(self, scalar_type, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        super(KernelBuilderBase, self).__init__(scalar_type=scalar_type,
                                                interior_facet=interior_facet)

        # Cell orientation
        if self.interior_facet:
            cell_orientations = gem.Variable("cell_orientations", (2,))
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),
                                       gem.Indexed(cell_orientations, (1,)))
        else:
            cell_orientations = gem.Variable("cell_orientations", (1,))
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),)

    def _coefficient(self, coefficient, name):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :returns: COFFEE function argument for the coefficient
        """
        funarg, expr = prepare_coefficient(coefficient, name,
                                           self.scalar_type,
                                           interior_facet=self.interior_facet)
        self.coefficient_map[coefficient] = expr
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
            funarg, expr = prepare_coefficient(f, "cell_sizes",
                                               self.scalar_type,
                                               interior_facet=self.interior_facet)
            self.cell_sizes_arg = kernel_args.CellSizesKernelArg(funarg)
            self._cell_sizes = expr

    def create_element(self, element, **kwargs):
        """Create a FInAT element (suitable for tabulating with) given
        a UFL element."""
        return create_element(element, **kwargs)


class KernelBuilder(KernelBuilderBase, KernelBuilderMixin):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_data_info, scalar_type,
                 dont_split=(), diagonal=False):
        """Initialise a kernel builder."""
        integral_type = integral_data_info.integral_type
        subdomain_id = integral_data_info.subdomain_id
        domain_number = integral_data_info.domain_number
        super(KernelBuilder, self).__init__(scalar_type, integral_type.startswith("interior_facet"))

        self.kernel = Kernel(integral_type=integral_type, subdomain_id=subdomain_id,
                             domain_number=domain_number)
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
        coords_coffee_arg = self._coefficient(f, "coords")
        self.coordinates_arg = kernel_args.CoordinatesKernelArg(coords_coffee_arg)

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
            coeff_coffee_arg = self._coefficient(coefficient, f"w_{i}")
            self.coefficient_args.append(kernel_args.CoefficientKernelArg(coeff_coffee_arg))
        self.kernel.coefficient_numbers = tuple(self.integral_data_info.coefficient_numbers)

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        knl = self.kernel
        knl.oriented, knl.needs_cell_sizes, knl.tabulations = check_requirements(ir)

    def construct_kernel(self, name, impero_c, index_names, quadrature_rule):
        """Construct a fully built :class:`Kernel`.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: function name
        :arg impero_c: ImperoC tuple with Impero AST and other data
        :arg index_names: pre-assigned index names
        :arg quadrature rule: quadrature rule
        :returns: :class:`Kernel` object
        """
        args = [self.output_arg, self.coordinates_arg]
        if self.kernel.oriented:
            ori_coffee_arg = coffee.Decl("int", coffee.Symbol("cell_orientations"),
                                         pointers=[("restrict",)],
                                         qualifiers=["const"])
            args.append(kernel_args.CellOrientationsKernelArg(ori_coffee_arg))
        if self.kernel.needs_cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.coefficient_args)
        if self.kernel.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            ext_coffee_arg = coffee.Decl("unsigned int",
                                         coffee.Symbol("facet", rank=(1,)),
                                         qualifiers=["const"])
            args.append(kernel_args.ExteriorFacetKernelArg(ext_coffee_arg))
        elif self.kernel.integral_type in ["interior_facet", "interior_facet_vert"]:
            int_coffee_arg = coffee.Decl("unsigned int",
                                         coffee.Symbol("facet", rank=(2,)),
                                         qualifiers=["const"])
            args.append(kernel_args.InteriorFacetKernelArg(int_coffee_arg))
        for n, shape in self.kernel.tabulations:
            tab_coffee_arg = coffee.Decl(self.scalar_type,
                                         coffee.Symbol(n, rank=shape),
                                         qualifiers=["const"])
            args.append(kernel_args.TabulationKernelArg(tab_coffee_arg))

        coffee_args = [a.coffee_arg for a in args]
        body = generate_coffee(impero_c, index_names, self.scalar_type)

        self.kernel.ast = KernelBuilderBase.construct_kernel(self, name, coffee_args, body)
        self.kernel.arguments = tuple(args)
        self.kernel.quadrature_rule = quadrature_rule
        self.kernel.name = name
        self.kernel.flop_count = count_flops(impero_c)
        return self.kernel

    def construct_empty_kernel(self, name):
        """Return None, since Firedrake needs no empty kernels.

        :arg name: function name
        :returns: None
        """
        return None


def check_requirements(ir):
    """Look for cell orientations, cell sizes, and collect tabulations
    in one pass."""
    cell_orientations = False
    cell_sizes = False
    rt_tabs = {}
    for node in traversal(ir):
        if isinstance(node, gem.Variable):
            if node.name == "cell_orientations":
                cell_orientations = True
            elif node.name == "cell_sizes":
                cell_sizes = True
            elif node.name.startswith("rt_"):
                rt_tabs[node.name] = node.shape
    return cell_orientations, cell_sizes, tuple(sorted(rt_tabs.items()))


def prepare_coefficient(coefficient, name, scalar_type, interior_facet=False):
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
        funarg = coffee.Decl(scalar_type, coffee.Symbol(name),
                             pointers=[("restrict",)],
                             qualifiers=["const"])
        value_size = coefficient.ufl_element().value_size()
        expression = gem.reshape(gem.Variable(name, (value_size,)),
                                 coefficient.ufl_shape)

        return funarg, expression

    finat_element = create_element(coefficient.ufl_element())
    shape = finat_element.index_shape
    size = numpy.prod(shape, dtype=int)

    funarg = coffee.Decl(scalar_type, coffee.Symbol(name),
                         pointers=[("restrict",)],
                         qualifiers=["const"])

    if not interior_facet:
        expression = gem.reshape(gem.Variable(name, (size,)), shape)
    else:
        varexp = gem.Variable(name, (2 * size,))
        plus = gem.view(varexp, slice(size))
        minus = gem.view(varexp, slice(size, 2 * size))
        expression = (gem.reshape(plus, shape),
                      gem.reshape(minus, shape))
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
         funarg      - :class:`coffee.Decl` function argument
         expressions - GEM expressions referring to the argument
                       tensor
    """
    assert isinstance(interior_facet, bool)

    if len(arguments) == 0:
        # No arguments
        funarg = coffee.Decl(scalar_type, coffee.Symbol("A", rank=(1,)))
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

    funarg = coffee.Decl(scalar_type, coffee.Symbol("A", rank=c_shape))
    varexp = gem.Variable("A", c_shape)
    expressions = [expression(gem.view(varexp, *slices)) for slices in slicez]
    return funarg, prune(expressions)
