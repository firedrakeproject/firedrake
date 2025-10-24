import numpy
from collections import namedtuple, OrderedDict

from ufl import Coefficient, FunctionSpace
from ufl.domain import MeshSequence

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


ActiveDomainNumbers = namedtuple('ActiveDomainNumbers', ['coordinates',
                                                         'cell_orientations',
                                                         'cell_sizes',
                                                         'exterior_facets',
                                                         'interior_facets',
                                                         'orientations_exterior_facet',
                                                         'orientations_interior_facet'])
ActiveDomainNumbers.__doc__ = """
    Active domain numbers collected for each key.

    """


class Kernel:
    __slots__ = ("ast", "arguments", "integral_type", "subdomain_id",
                 "domain_number", "active_domain_numbers", "tabulations",
                 "coefficient_numbers", "name", "flop_count", "event",
                 "__weakref__")
    """A compiled Kernel object.

    :kwarg ast: The loopy kernel object.
    :kwarg integral_type: The type of integral.
    :kwarg subdomain_id: What is the subdomain id for this kernel.
    :kwarg domain_number: Which domain number in the original form
        does this kernel correspond to (can be used to index into
        original_form.ufl_domains() to get the correct domain).
    :kwarg coefficient_numbers: A list of which coefficients from the
        form the kernel needs.
    :kwarg tabulations: The runtime tabulations this kernel requires
    :kwarg name: The name of this kernel.
    :kwarg flop_count: Estimated total flops for this kernel.
    :kwarg event: name for logging event
    """
    def __init__(self, ast=None, arguments=None, integral_type=None,
                 subdomain_id=None, domain_number=None, active_domain_numbers=None,
                 coefficient_numbers=(),
                 tabulations=None,
                 flop_count=0,
                 name=None,
                 event=None):
        # Defaults
        self.ast = ast
        self.arguments = arguments
        self.integral_type = integral_type
        self.domain_number = domain_number
        self.active_domain_numbers = active_domain_numbers
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        self.tabulations = tabulations
        self.flop_count = flop_count
        self.name = name
        self.event = event


class KernelBuilderBase(_KernelBuilderBase):

    def __init__(self, scalar_type):
        """Initialise a kernel builder."""
        super().__init__(scalar_type=scalar_type)

    def _coefficient(self, coefficient, name):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :returns: GEM expression representing the coefficient
        """
        expr = prepare_coefficient(coefficient, name, self._domain_integral_type_map)
        self.coefficient_map[coefficient] = expr
        return expr

    def set_coordinates(self, domains):
        """Set coordinates for each domain.

        Parameters
        ----------
        domains : list or tuple
            All domains in the form.

        """
        # Create a fake coordinate coefficient for a domain.
        for i, domain in enumerate(domains):
            if isinstance(domain, MeshSequence):
                raise RuntimeError("Found a MeshSequence")
            f = Coefficient(FunctionSpace(domain, domain.ufl_coordinate_element()))
            self.domain_coordinate[domain] = f
            self._coefficient(f, f"coords_{i}")

    def set_cell_orientations(self, domains):
        """Set cell orientations for each domain.

        Parameters
        ----------
        domains : list or tuple
            All domains in the form.

        """
        # Cell orientation
        self._cell_orientations = {}
        for i, domain in enumerate(domains):
            integral_type = self._domain_integral_type_map[domain]
            if integral_type is None:
                # See comment in prepare_coefficient.
                self._cell_orientations[domain] = None
            elif integral_type.startswith("interior_facet"):
                cell_orientations = gem.Variable(f"cell_orientations_{i}", (2,), dtype=gem.uint_type)
                self._cell_orientations[domain] = (gem.Indexed(cell_orientations, (0,)),
                                                   gem.Indexed(cell_orientations, (1,)))
            else:
                cell_orientations = gem.Variable(f"cell_orientations_{i}", (1,), dtype=gem.uint_type)
                self._cell_orientations[domain] = (gem.Indexed(cell_orientations, (0,)),)

    def set_cell_sizes(self, domains):
        """Setup a fake coefficient for "cell sizes" for each domain.

        Parameters
        ----------
        domains : list or tuple
            All domains in the form.

        This is required for scaling of derivative basis functions on
        physically mapped elements (Argyris, Bell, etc...).  We need a
        measure of the mesh size around each vertex (hence this lives
        in P1).

        Should the domain have topological dimension 0 this does
        nothing.
        """
        self._cell_sizes = {}
        for i, domain in enumerate(domains):
            if domain.ufl_cell().topological_dimension > 0:
                # Can't create P1 since only P0 is a valid finite element if
                # topological_dimension is 0 and the concept of "cell size"
                # is not useful for a vertex.
                f = Coefficient(FunctionSpace(domain, FiniteElement("P", domain.ufl_cell(), 1)))
                expr = prepare_coefficient(f, f"cell_sizes_{i}", self._domain_integral_type_map)
                self._cell_sizes[domain] = expr

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
        if "coords" in var.name:
            dtype = numpy.complex128
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
        self.oriented, self.cell_sizes, self.tabulations = check_requirements(ir)

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
            cell_orientations, = tuple(self._cell_orientations.values())
            funarg = self.generate_arg_from_expression(cell_orientations, dtype=numpy.int32)
            args.append(kernel_args.CellOrientationsKernelArg(funarg))
        if self.cell_sizes:
            cell_sizes, = tuple(self._cell_sizes.values())
            funarg = self.generate_arg_from_expression(cell_sizes)
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
                 diagonal=False):
        """Initialise a kernel builder."""
        super(KernelBuilder, self).__init__(scalar_type)
        self.fem_scalar_type = scalar_type
        self.diagonal = diagonal
        self.local_tensor = None
        self.coefficient_number_index_map = OrderedDict()
        self.integral_data_info = integral_data_info
        self._domain_integral_type_map = integral_data_info.domain_integral_type_map  # For consistency with ExpressionKernelBuilder.
        self.set_arguments()

    def set_arguments(self):
        """Process arguments."""
        arguments = self.integral_data_info.arguments
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
                                             self.integral_data_info.domain_integral_type_map,
                                             diagonal=self.diagonal)
        self.return_variables = return_variables
        self.argument_multiindices = argument_multiindices

    def set_entity_numbers(self, domains):
        """Set entity numbers for each domain.

        Parameters
        ----------
        domains : list or tuple
            All domains in the form.

        """
        self._entity_numbers = {}
        for i, domain in enumerate(domains):
            # Facet number
            integral_type = self.integral_data_info.domain_integral_type_map[domain]
            if integral_type in ['exterior_facet', 'exterior_facet_vert']:
                facet = gem.Variable(f'facet_{i}', (1,), dtype=gem.uint_type)
                self._entity_numbers[domain] = {None: gem.VariableIndex(gem.Indexed(facet, (0,))), }
            elif integral_type in ['interior_facet', 'interior_facet_vert']:
                facet = gem.Variable(f'facet_{i}', (2,), dtype=gem.uint_type)
                self._entity_numbers[domain] = {
                    '+': gem.VariableIndex(gem.Indexed(facet, (0,))),
                    '-': gem.VariableIndex(gem.Indexed(facet, (1,)))
                }
            elif integral_type == 'interior_facet_horiz':
                self._entity_numbers[domain] = {'+': 1, '-': 0}
            else:
                self._entity_numbers[domain] = {None: None}

    def set_entity_orientations(self, domains):
        """Set entity orientations for each domain.

        Parameters
        ----------
        domains : list or tuple
            All domains in the form.

        """
        self._entity_orientations = {}
        for i, domain in enumerate(domains):
            integral_type = self.integral_data_info.domain_integral_type_map[domain]
            variable_name = f"entity_orientations_{i}"
            if integral_type in ['exterior_facet', 'exterior_facet_vert']:
                o = gem.Variable(variable_name, (1,), dtype=gem.uint_type)
                self._entity_orientations[domain] = {None: gem.OrientationVariableIndex(gem.Indexed(o, (0,))), }
            elif integral_type in ['interior_facet', 'interior_facet_vert']:
                o = gem.Variable(variable_name, (2,), dtype=gem.uint_type)
                self._entity_orientations[domain] = {
                    '+': gem.OrientationVariableIndex(gem.Indexed(o, (0,))),
                    '-': gem.OrientationVariableIndex(gem.Indexed(o, (1,)))
                }
            elif integral_type == 'interior_facet_horiz':
                o = gem.Variable(variable_name, (1,), dtype=gem.uint_type)  # base mesh entity orientation
                self._entity_orientations[domain] = {
                    '+': gem.OrientationVariableIndex(gem.Indexed(o, (0,))),
                    '-': gem.OrientationVariableIndex(gem.Indexed(o, (0,)))
                }
            else:
                self._entity_orientations[domain] = {None: None}

    def set_coefficients(self):
        """Prepare the coefficients of the form."""
        info = self.integral_data_info
        k = 0
        for n, coeff in enumerate(info.coefficients):
            if coeff in info.coefficient_split:
                for i, c in enumerate(info.coefficient_split[coeff]):
                    self.coefficient_number_index_map[c] = (n, i)
                    self._coefficient(c, f"w_{k}")
                    k += 1
            else:
                self.coefficient_number_index_map[coeff] = (n, 0)
                self._coefficient(coeff, f"w_{k}")
                k += 1

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
        impero_c, _, _, tabulations, active_variables = self.compile_gem(ctx)
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
        active_domain_numbers_coordinates, args_ = self.make_active_domain_numbers({d: self.coefficient_map[c] for d, c in self.domain_coordinate.items()},
                                                                                   active_variables,
                                                                                   kernel_args.CoordinatesKernelArg)
        args.extend(args_)
        active_domain_numbers_cell_orientations, args_ = self.make_active_domain_numbers(self._cell_orientations,
                                                                                         active_variables,
                                                                                         kernel_args.CellOrientationsKernelArg,
                                                                                         dtype=numpy.int32)
        args.extend(args_)
        active_domain_numbers_cell_sizes, args_ = self.make_active_domain_numbers(self._cell_sizes,
                                                                                  active_variables,
                                                                                  kernel_args.CellSizesKernelArg)
        args.extend(args_)
        coefficient_indices = OrderedDict()
        for coeff, (number, index) in self.coefficient_number_index_map.items():
            a = coefficient_indices.setdefault(number, [])
            expr = self.coefficient_map[coeff]
            if expr is None:
                # See comment in prepare_coefficient.
                continue
            var, = gem.extract_type(expr if isinstance(expr, tuple) else (expr, ), gem.Variable)
            if var in active_variables:
                funarg = self.generate_arg_from_expression(expr)
                args.append(kernel_args.CoefficientKernelArg(funarg))
                a.append(index)
        for gemexpr in self.constant_map.values():
            funarg = self.generate_arg_from_expression(gemexpr)
            args.append(kernel_args.ConstantKernelArg(funarg))
        coefficient_indices = tuple(tuple(v) for v in coefficient_indices.values())
        assert len(coefficient_indices) == len(info.coefficient_numbers)
        ext_dict = {}
        for domain, expr in self._entity_numbers.items():
            integral_type = info.domain_integral_type_map[domain]
            ext_dict[domain] = expr[None].expression if integral_type in ["exterior_facet", "exterior_facet_vert"] else None
        active_domain_numbers_exterior_facets, args_ = self.make_active_domain_numbers(
            ext_dict,
            active_variables,
            kernel_args.ExteriorFacetKernelArg,
            dtype=numpy.uint32,
        )
        args.extend(args_)
        int_dict = {}
        for domain, expr in self._entity_numbers.items():
            integral_type = info.domain_integral_type_map[domain]
            int_dict[domain] = expr['+'].expression if integral_type in ["interior_facet", "interior_facet_vert"] else None
        active_domain_numbers_interior_facets, args_ = self.make_active_domain_numbers(
            int_dict,
            active_variables,
            kernel_args.InteriorFacetKernelArg,
            dtype=numpy.uint32,
        )
        args.extend(args_)
        ext_dict = {}
        for domain, expr in self._entity_orientations.items():
            integral_type = info.domain_integral_type_map[domain]
            ext_dict[domain] = expr[None].expression if integral_type in ["exterior_facet", "exterior_facet_vert"] else None
        active_domain_numbers_orientations_exterior_facet, args_ = self.make_active_domain_numbers(
            ext_dict,
            active_variables,
            kernel_args.OrientationsExteriorFacetKernelArg,
            dtype=gem.uint_type,
        )
        args.extend(args_)
        int_dict = {}
        for domain, expr in self._entity_orientations.items():
            integral_type = info.domain_integral_type_map[domain]
            int_dict[domain] = expr['+'].expression if integral_type in ["interior_facet", "interior_facet_vert", "interior_facet_horiz"] else None
        active_domain_numbers_orientations_interior_facet, args_ = self.make_active_domain_numbers(
            int_dict,
            active_variables,
            kernel_args.OrientationsInteriorFacetKernelArg,
            dtype=gem.uint_type,
        )
        args.extend(args_)
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
                      active_domain_numbers=ActiveDomainNumbers(
                          coordinates=tuple(active_domain_numbers_coordinates),
                          cell_orientations=tuple(active_domain_numbers_cell_orientations),
                          cell_sizes=tuple(active_domain_numbers_cell_sizes),
                          exterior_facets=tuple(active_domain_numbers_exterior_facets),
                          interior_facets=tuple(active_domain_numbers_interior_facets),
                          orientations_exterior_facet=tuple(active_domain_numbers_orientations_exterior_facet),
                          orientations_interior_facet=tuple(active_domain_numbers_orientations_interior_facet),
                      ),
                      coefficient_numbers=tuple(zip(info.coefficient_numbers, coefficient_indices)),
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

    def make_active_domain_numbers(self, domain_expr_dict, active_variables, kernel_arg_type, dtype=None):
        """Make active domain numbers.

        Parameters
        ----------
        domain_expr_dict : dict
            Map from domains to expressions; must be ordered as extract_domains(form).
        active_variables : tuple
            Active variables in the DAG.
        kernel_arg_type : KernelArg
            Type of `KernelArg`.
        dtype : numpy.dtype
            dtype.

        Returns
        -------
        tuple
            Tuple of active domain numbers and corresponding kernel args.

        """
        active_dns = []
        args = []
        for i, expr in enumerate(domain_expr_dict.values()):
            if expr is None:
                var = None
            else:
                var, = gem.extract_type(expr if isinstance(expr, tuple) else (expr, ), gem.Variable)
            if var in active_variables:
                funarg = self.generate_arg_from_expression(expr, dtype=dtype)
                args.append(kernel_arg_type(funarg))
                active_dns.append(i)
        return tuple(active_dns), tuple(args)
