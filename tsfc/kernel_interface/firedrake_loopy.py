import numpy
from collections import namedtuple, OrderedDict
from functools import partial

from ufl import Coefficient, FunctionSpace
from ufl.domain import extract_unique_domain
from finat.ufl import MixedElement as ufl_MixedElement, FiniteElement

import gem
from gem.flop_count import count_flops

import loopy as lp

from tsfc import kernel_args
from finat.element_factory import create_element
from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase, KernelBuilderMixin, get_index_names, check_requirements, prepare_coefficient, prepare_arguments, prepare_constant
from tsfc.loopy import generate as generate_loopy


# Expression kernel description type
ExpressionKernel = namedtuple('ExpressionKernel', ['ast', 'oriented', 'needs_cell_sizes',
                                                   'coefficient_numbers',
                                                   'needs_external_coords',
                                                   'tabulations', 'name', 'arguments',
                                                   'flop_count', 'event'])


def make_builder(*args, **kwargs):
    return partial(KernelBuilder, *args, **kwargs)


class Kernel:
    __slots__ = ("ast", "arguments", "integral_type", "oriented", "subdomain_id",
                 "domain_number", "needs_cell_sizes", "tabulations",
                 "coefficient_numbers", "name", "flop_count", "event",
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
        self.event = event


class KernelBuilderBase(_KernelBuilderBase):

    def __init__(self, scalar_type, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        super().__init__(scalar_type=scalar_type, interior_facet=interior_facet)

        # Cell orientation
        if self.interior_facet:
            cell_orientations = gem.Variable("cell_orientations", (2,), dtype=gem.uint_type)
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),
                                       gem.Indexed(cell_orientations, (1,)))
        else:
            cell_orientations = gem.Variable("cell_orientations", (1,), dtype=gem.uint_type)
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),)

    def _coefficient(self, coefficient, name):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :returns: GEM expression representing the coefficient
        """
        expr = prepare_coefficient(coefficient, name, interior_facet=self.interior_facet)
        self.coefficient_map[coefficient] = expr
        return expr

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

    def generate_arg_from_variable(self, var, dtype=None):
        """Generate kernel arg from a :class:`gem.Variable`.

        :arg var: a :class:`gem.Variable`
        :arg dtype: dtype of the kernel arg
        :returns: kernel arg
        """
        return lp.GlobalArg(var.name, dtype=dtype or self.scalar_type, shape=var.shape)

    def generate_arg_from_expression(self, expr, dtype=None):
        """Generate kernel arg from gem expression(s).

        :arg expr: gem expression(s) representing a coefficient or the output tensor
        :arg dtype: dtype of the kernel arg
        :returns: kernel arg
        """
        var, = gem.extract_type(expr if isinstance(expr, tuple) else (expr, ), gem.Variable)
        return self.generate_arg_from_variable(var, dtype=dtype or self.scalar_type)


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

        for i, coefficient in enumerate(coefficients):
            if type(coefficient.ufl_element()) == ufl_MixedElement:
                subcoeffs = coefficient.subfunctions  # Firedrake-specific
                self.coefficient_split[coefficient] = subcoeffs
                for j, subcoeff in enumerate(subcoeffs):
                    self._coefficient(subcoeff, f"w_{i}_{j}")
            else:
                self._coefficient(coefficient, f"w_{i}")

    def set_constants(self, constants):
        for i, const in enumerate(constants):
            gemexpr = prepare_constant(const, i)
            self.constant_map[const] = gemexpr

    def set_coefficient_numbers(self, coefficient_numbers):
        """Store the coefficient indices of the original form.

        :arg coefficient_numbers: Iterable of indices describing which coefficients
            from the input expression need to be passed in to the kernel.
        """
        self.coefficient_numbers = coefficient_numbers

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        self.oriented, self.cell_sizes, self.tabulations, _ = check_requirements(ir)

    def set_output(self, o):
        """Produce the kernel return argument"""
        loopy_arg = lp.GlobalArg(o.name, dtype=self.scalar_type, shape=o.shape)
        self.output_arg = kernel_args.OutputKernelArg(loopy_arg)

    def construct_kernel(self, impero_c, index_names, needs_external_coords, log=False):
        """Constructs an :class:`ExpressionKernel`.

        :arg impero_c: gem.ImperoC object that represents the kernel
        :arg index_names: pre-assigned index names
        :arg needs_external_coords: If ``True``, the first argument to
            the kernel is an externally provided coordinate field.
        :arg log: bool if the Kernel should be profiled with Log events

        :returns: :class:`ExpressionKernel` object
        """
        args = [self.output_arg]
        if self.oriented:
            funarg = self.generate_arg_from_expression(self._cell_orientations, dtype=numpy.int32)
            args.append(kernel_args.CellOrientationsKernelArg(funarg))
        if self.cell_sizes:
            funarg = self.generate_arg_from_expression(self._cell_sizes)
            args.append(kernel_args.CellSizesKernelArg(funarg))
        for _, expr in self.coefficient_map.items():
            # coefficient_map is OrderedDict.
            funarg = self.generate_arg_from_expression(expr)
            args.append(kernel_args.CoefficientKernelArg(funarg))

        # now constants
        for gemexpr in self.constant_map.values():
            funarg = self.generate_arg_from_expression(gemexpr)
            args.append(kernel_args.ConstantKernelArg(funarg))

        for name_, shape in self.tabulations:
            tab_loopy_arg = lp.GlobalArg(name_, dtype=self.scalar_type, shape=shape)
            args.append(kernel_args.TabulationKernelArg(tab_loopy_arg))

        loopy_args = [arg.loopy_arg for arg in args]

        name = "expression_kernel"
        loopy_kernel, event = generate_loopy(impero_c, loopy_args, self.scalar_type,
                                             name, index_names, log=log)
        return ExpressionKernel(loopy_kernel, self.oriented, self.cell_sizes,
                                self.coefficient_numbers, needs_external_coords,
                                self.tabulations, name, args, count_flops(impero_c), event)


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
        self.coefficient_split = {}
        self.coefficient_number_index_map = OrderedDict()
        self.dont_split = frozenset(dont_split)

        # Facet number
        if integral_type in ['exterior_facet', 'exterior_facet_vert']:
            facet = gem.Variable('facet', (1,), dtype=gem.uint_type)
            self._entity_number = {None: gem.VariableIndex(gem.Indexed(facet, (0,)))}
            facet_orientation = gem.Variable('facet_orientation', (1,), dtype=gem.uint_type)
            self._entity_orientation = {None: gem.OrientationVariableIndex(gem.Indexed(facet_orientation, (0,)))}
        elif integral_type in ['interior_facet', 'interior_facet_vert']:
            facet = gem.Variable('facet', (2,), dtype=gem.uint_type)
            self._entity_number = {
                '+': gem.VariableIndex(gem.Indexed(facet, (0,))),
                '-': gem.VariableIndex(gem.Indexed(facet, (1,)))
            }
            facet_orientation = gem.Variable('facet_orientation', (2,), dtype=gem.uint_type)
            self._entity_orientation = {
                '+': gem.OrientationVariableIndex(gem.Indexed(facet_orientation, (0,))),
                '-': gem.OrientationVariableIndex(gem.Indexed(facet_orientation, (1,)))
            }
        elif integral_type == 'interior_facet_horiz':
            self._entity_number = {'+': 1, '-': 0}
            facet_orientation = gem.Variable('facet_orientation', (1,), dtype=gem.uint_type)  # base mesh entity orientation
            self._entity_orientation = {
                '+': gem.OrientationVariableIndex(gem.Indexed(facet_orientation, (0,))),
                '-': gem.OrientationVariableIndex(gem.Indexed(facet_orientation, (0,)))
            }

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
        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        n, k = 0, 0
        for i in range(len(integral_data.enabled_coefficients)):
            if integral_data.enabled_coefficients[i]:
                original = form_data.reduced_coefficients[i]
                coefficient = form_data.function_replace_map[original]
                if type(coefficient.ufl_element()) == ufl_MixedElement:
                    if original in self.dont_split:
                        self.coefficient_split[coefficient] = [coefficient]
                        self._coefficient(coefficient, f"w_{k}")
                        self.coefficient_number_index_map[coefficient] = (n, 0)
                        k += 1
                    else:
                        self.coefficient_split[coefficient] = []
                        for j, element in enumerate(coefficient.ufl_element().sub_elements):
                            c = Coefficient(FunctionSpace(extract_unique_domain(coefficient), element))
                            self.coefficient_split[coefficient].append(c)
                            self._coefficient(c, f"w_{k}")
                            self.coefficient_number_index_map[c] = (n, j)
                            k += 1
                else:
                    self._coefficient(coefficient, f"w_{k}")
                    self.coefficient_number_index_map[coefficient] = (n, 0)
                    k += 1
                n += 1

    def set_constants(self, constants):
        for i, const in enumerate(constants):
            gemexpr = prepare_constant(const, i)
            self.constant_map[const] = gemexpr

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
        :arg log: bool if the Kernel should be profiled with Log events
        :returns: :class:`Kernel` object
        """
        impero_c, oriented, needs_cell_sizes, tabulations, active_variables, need_facet_orientation = self.compile_gem(ctx)
        if impero_c is None:
            return self.construct_empty_kernel(name)
        info = self.integral_data_info
        # In the following funargs are only generated
        # for gem expressions that are actually used;
        # see `generate_arg_from_expression()` method.
        # Specifically, funargs are not generated for
        # unused components of mixed coefficients.
        # Problem solving environment, such as Firedrake,
        # will know which components have been included
        # in the list of kernel arguments by investigating
        # `Kernel.coefficient_numbers`.
        # Add return arg
        funarg = self.generate_arg_from_expression(self.return_variables)
        args = [kernel_args.OutputKernelArg(funarg)]
        # Add coordinates arg
        coord = self.domain_coordinate[info.domain]
        expr = self.coefficient_map[coord]
        funarg = self.generate_arg_from_expression(expr)
        args.append(kernel_args.CoordinatesKernelArg(funarg))
        if oriented:
            funarg = self.generate_arg_from_expression(self._cell_orientations, dtype=numpy.int32)
            args.append(kernel_args.CellOrientationsKernelArg(funarg))
        if needs_cell_sizes:
            funarg = self.generate_arg_from_expression(self._cell_sizes)
            args.append(kernel_args.CellSizesKernelArg(funarg))
        coefficient_indices = OrderedDict()
        for coeff, (number, index) in self.coefficient_number_index_map.items():
            a = coefficient_indices.setdefault(number, [])
            expr = self.coefficient_map[coeff]
            var, = gem.extract_type(expr if isinstance(expr, tuple) else (expr, ), gem.Variable)
            if var in active_variables:
                funarg = self.generate_arg_from_expression(expr)
                args.append(kernel_args.CoefficientKernelArg(funarg))
                a.append(index)

        # now constants
        for gemexpr in self.constant_map.values():
            funarg = self.generate_arg_from_expression(gemexpr)
            args.append(kernel_args.ConstantKernelArg(funarg))

        coefficient_indices = tuple(tuple(v) for v in coefficient_indices.values())
        assert len(coefficient_indices) == len(info.coefficient_numbers)
        if info.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            ext_loopy_arg = lp.GlobalArg("facet", numpy.uint32, shape=(1,))
            args.append(kernel_args.ExteriorFacetKernelArg(ext_loopy_arg))
        elif info.integral_type in ["interior_facet", "interior_facet_vert"]:
            int_loopy_arg = lp.GlobalArg("facet", numpy.uint32, shape=(2,))
            args.append(kernel_args.InteriorFacetKernelArg(int_loopy_arg))
        # The submesh PR will introduce a robust mechanism to check if a Variable
        # is actually used in the final form of the expression, so there will be
        # no need to get "need_facet_orientation" from self.compile_gem().
        if need_facet_orientation:
            if info.integral_type == "exterior_facet":
                ext_ornt_loopy_arg = lp.GlobalArg("facet_orientation", gem.uint_type, shape=(1,))
                args.append(kernel_args.ExteriorFacetOrientationKernelArg(ext_ornt_loopy_arg))
            elif info.integral_type == "interior_facet":
                int_ornt_loopy_arg = lp.GlobalArg("facet_orientation", gem.uint_type, shape=(2,))
                args.append(kernel_args.InteriorFacetOrientationKernelArg(int_ornt_loopy_arg))
        for name_, shape in tabulations:
            tab_loopy_arg = lp.GlobalArg(name_, dtype=self.scalar_type, shape=shape)
            args.append(kernel_args.TabulationKernelArg(tab_loopy_arg))
        index_names = get_index_names(ctx['quadrature_indices'], self.argument_multiindices, ctx['index_cache'])
        ast, event_name = generate_loopy(impero_c, [arg.loopy_arg for arg in args],
                                         self.scalar_type, name, index_names, log=log)
        flop_count = count_flops(impero_c)  # Estimated total flops for this kernel.
        return Kernel(ast=ast,
                      arguments=tuple(args),
                      integral_type=info.integral_type,
                      subdomain_id=info.subdomain_id,
                      domain_number=info.domain_number,
                      coefficient_numbers=tuple(zip(info.coefficient_numbers, coefficient_indices)),
                      oriented=oriented,
                      needs_cell_sizes=needs_cell_sizes,
                      tabulations=tabulations,
                      flop_count=flop_count,
                      name=name,
                      event=event_name)

    def construct_empty_kernel(self, name):
        """Return None, since Firedrake needs no empty kernels.

        :arg name: function name
        :returns: None
        """
        return None
