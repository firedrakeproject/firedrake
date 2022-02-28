from functools import partial

from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace, FiniteElement

import coffee.base as coffee

import gem
from gem.flop_count import count_flops

from tsfc import kernel_args
from tsfc.coffee import generate as generate_coffee
from tsfc.finatinterface import create_element
from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase, KernelBuilderMixin, get_index_names, check_requirements, prepare_coefficient, prepare_arguments


def make_builder(*args, **kwargs):
    return partial(KernelBuilder, *args, **kwargs)


class Kernel(object):
    __slots__ = ("ast", "arguments", "integral_type", "oriented", "subdomain_id",
                 "domain_number", "needs_cell_sizes", "tabulations",
                 "coefficient_numbers", "name", "__weakref__",
                 "flop_count", "event")
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
    :kwarg tabulations: The runtime tabulations this kernel requires
    :kwarg needs_cell_sizes: Does the kernel require cell sizes.
    :kwarg name: The name of this kernel.
    :kwarg flop_count: Estimated total flops for this kernel.
    :kwarg event: name for logging event
    """
    def __init__(self, ast=None, arguments=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None,
                 coefficient_numbers=(),
                 needs_cell_sizes=False,
                 tabulations=None,
                 flop_count=0,
                 name=None,
                 event=None):
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
        self.event = None
        super(Kernel, self).__init__()


class KernelBuilderBase(_KernelBuilderBase):

    def __init__(self, scalar_type, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        super().__init__(scalar_type=scalar_type, interior_facet=interior_facet)

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
        """
        expr = prepare_coefficient(coefficient, name, interior_facet=self.interior_facet)
        self.coefficient_map[coefficient] = expr

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
            expr = prepare_coefficient(f, "cell_sizes", interior_facet=self.interior_facet)
            self._cell_sizes = expr

    def create_element(self, element, **kwargs):
        """Create a FInAT element (suitable for tabulating with) given
        a UFL element."""
        return create_element(element, **kwargs)

    def generate_arg_from_variable(self, var, is_output=False):
        """Generate kernel arg from a :class:`gem.Variable`.

        :arg var: a :class:`gem.Variable`
        :arg is_output: if expr represents the output or not
        :returns: kernel arg
        """
        if is_output:
            return coffee.Decl(self.scalar_type, coffee.Symbol(var.name, rank=var.shape))
        else:
            return coffee.Decl(self.scalar_type, coffee.Symbol(var.name), pointers=[("restrict",)], qualifiers=["const"])

    def generate_arg_from_expression(self, expr, is_output=False):
        """Generate kernel arg from gem expression(s).

        :arg expr: gem expression(s) representing a coefficient or the output tensor
        :arg is_output: if expr represents the output or not
        :returns: kernel arg
        """
        var, = gem.extract_type(expr if isinstance(expr, tuple) else (expr, ), gem.Variable)
        return self.generate_arg_from_variable(var, is_output=is_output)


class KernelBuilder(KernelBuilderBase, KernelBuilderMixin):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_data_info, scalar_type,
                 dont_split=(), diagonal=False):
        """Initialise a kernel builder."""
        integral_type = integral_data_info.integral_type
        super(KernelBuilder, self).__init__(coffee.as_cstr(scalar_type), integral_type.startswith("interior_facet"))
        self.fem_scalar_type = scalar_type

        self.diagonal = diagonal
        self.local_tensor = None
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
        return_variables = prepare_arguments(arguments,
                                             argument_multiindices,
                                             interior_facet=self.interior_facet,
                                             diagonal=self.diagonal)
        self.return_variables = return_variables
        self.argument_multiindices = argument_multiindices

    def set_coordinates(self, domain):
        """Prepare the coordinate field.

        :arg domain: :class:`ufl.Domain`
        """
        # Create a fake coordinate coefficient for a domain.
        f = Coefficient(FunctionSpace(domain, domain.ufl_coordinate_element()))
        self.domain_coordinate[domain] = f
        self._coefficient(f, "coords")

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
            self._coefficient(coefficient, f"w_{i}")

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        return check_requirements(ir)

    def construct_kernel(self, name, ctx, log=False):
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
        # Add return arg
        funarg = self.generate_arg_from_expression(self.return_variables, is_output=True)
        args = [kernel_args.OutputKernelArg(funarg)]
        # Add coordinates arg
        coord = self.domain_coordinate[info.domain]
        expr = self.coefficient_map[coord]
        funarg = self.generate_arg_from_expression(expr)
        args.append(kernel_args.CoordinatesKernelArg(funarg))
        if oriented:
            ori_coffee_arg = coffee.Decl("int", coffee.Symbol("cell_orientations"),
                                         pointers=[("restrict",)],
                                         qualifiers=["const"])
            args.append(kernel_args.CellOrientationsKernelArg(ori_coffee_arg))
        if needs_cell_sizes:
            funarg = self.generate_arg_from_expression(self._cell_sizes)
            args.append(kernel_args.CellSizesKernelArg(funarg))
        for coeff, expr in self.coefficient_map.items():
            if coeff in self.domain_coordinate.values():
                continue
            funarg = self.generate_arg_from_expression(expr)
            args.append(kernel_args.CoefficientKernelArg(funarg))
        if info.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            ext_coffee_arg = coffee.Decl("unsigned int",
                                         coffee.Symbol("facet", rank=(1,)),
                                         qualifiers=["const"])
            args.append(kernel_args.ExteriorFacetKernelArg(ext_coffee_arg))
        elif info.integral_type in ["interior_facet", "interior_facet_vert"]:
            int_coffee_arg = coffee.Decl("unsigned int",
                                         coffee.Symbol("facet", rank=(2,)),
                                         qualifiers=["const"])
            args.append(kernel_args.InteriorFacetKernelArg(int_coffee_arg))
        for name_, shape in tabulations:
            tab_coffee_arg = coffee.Decl(self.scalar_type,
                                         coffee.Symbol(name_, rank=shape),
                                         qualifiers=["const"])
            args.append(kernel_args.TabulationKernelArg(tab_coffee_arg))
        index_names = get_index_names(ctx['quadrature_indices'], self.argument_multiindices, ctx['index_cache'])
        body = generate_coffee(impero_c, index_names, self.scalar_type)
        ast = KernelBuilderBase.construct_kernel(self, name, [a.coffee_arg for a in args], body)
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
