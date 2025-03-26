import collections
import operator
import string
from functools import cached_property, reduce
from itertools import chain, product

import gem
import gem.impero_utils as impero_utils
import numpy
from FIAT.reference_element import TensorProductCell
from finat.cell_tools import max_complex
from finat.quadrature import AbstractQuadratureRule
from gem.node import traversal
from gem.optimise import constant_fold_zero
from gem.optimise import remove_componenttensors as prune
from numpy import asarray
from tsfc import fem, ufl_utils
from finat.element_factory import as_fiat_cell, create_element
from tsfc.kernel_interface import KernelInterface
from tsfc.logging import logger
from ufl.utils.sequences import max_degree


class KernelBuilderBase(KernelInterface):
    """Helper class for building local assembly kernels."""

    def __init__(self, scalar_type, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        assert isinstance(interior_facet, bool)
        self.scalar_type = scalar_type
        self.interior_facet = interior_facet

        self.prepare = []
        self.finalise = []

        # Coordinates
        self.domain_coordinate = {}

        # Coefficients
        self.coefficient_map = collections.OrderedDict()

        # Constants
        self.constant_map = collections.OrderedDict()

    @cached_property
    def unsummed_coefficient_indices(self):
        return frozenset()

    def coordinate(self, domain):
        return self.domain_coordinate[domain]

    def coefficient(self, ufl_coefficient, restriction):
        """A function that maps :class:`ufl.Coefficient`s to GEM
        expressions."""
        kernel_arg = self.coefficient_map[ufl_coefficient]
        if ufl_coefficient.ufl_element().family() == 'Real':
            return kernel_arg
        elif not self.interior_facet:
            return kernel_arg
        else:
            return kernel_arg[{'+': 0, '-': 1}[restriction]]

    def constant(self, const):
        return self.constant_map[const]

    def cell_orientation(self, restriction):
        """Cell orientation as a GEM expression."""
        f = {None: 0, '+': 0, '-': 1}[restriction]
        # Assume self._cell_orientations tuple is set up at this point.
        co_int = self._cell_orientations[f]
        return gem.Conditional(gem.Comparison("==", co_int, gem.Literal(1)),
                               gem.Literal(-1),
                               gem.Conditional(gem.Comparison("==", co_int, gem.Zero()),
                                               gem.Literal(1),
                                               gem.Literal(numpy.nan)))

    def cell_size(self, restriction):
        if not hasattr(self, "_cell_sizes"):
            raise RuntimeError("Haven't called set_cell_sizes")
        if self.interior_facet:
            return self._cell_sizes[{'+': 0, '-': 1}[restriction]]
        else:
            return self._cell_sizes

    def entity_number(self, restriction):
        """Facet or vertex number as a GEM index."""
        # Assume self._entity_number dict is set up at this point.
        return self._entity_number[restriction]

    def entity_orientation(self, restriction):
        """Facet orientation as a GEM index."""
        # Assume self._entity_orientation dict is set up at this point.
        return self._entity_orientation[restriction]

    def apply_glue(self, prepare=None, finalise=None):
        """Append glue code for operations that are not handled in the
        GEM abstraction.

        Current uses: mixed interior facet mess

        :arg prepare: code snippets to be prepended to the kernel
        :arg finalise: code snippets to be appended to the kernel
        """
        if prepare is not None:
            self.prepare.extend(prepare)
        if finalise is not None:
            self.finalise.extend(finalise)

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface.

        :arg ir: multi-root GEM expression DAG
        """
        # Nothing is required by default
        pass


class KernelBuilderMixin(object):
    """Mixin for KernelBuilder classes."""

    def compile_integrand(self, integrand, params, ctx):
        """Compile UFL integrand.

        :arg integrand: UFL integrand.
        :arg params: a dict containing "quadrature_rule".
        :arg ctx: context created with :meth:`create_context` method.

        See :meth:`create_context` for typical calling sequence.
        """
        # Split Coefficients
        if self.coefficient_split:
            integrand = ufl_utils.split_coefficients(integrand, self.coefficient_split)
        # Compile: ufl -> gem
        info = self.integral_data_info
        functions = [*info.arguments, self.coordinate(info.domain), *info.coefficients]
        set_quad_rule(params, info.domain.ufl_cell(), info.integral_type, functions)
        quad_rule = params["quadrature_rule"]
        config = self.fem_config()
        config['argument_multiindices'] = self.argument_multiindices
        config['quadrature_rule'] = quad_rule
        config['index_cache'] = ctx['index_cache']
        expressions = fem.compile_ufl(integrand,
                                      fem.PointSetContext(**config),
                                      interior_facet=self.interior_facet)
        ctx['quadrature_indices'].extend(quad_rule.point_set.indices)
        return expressions

    def construct_integrals(self, integrand_expressions, params):
        """Construct integrals from integrand expressions.

        :arg integrand_expressions: gem expressions for integrands.
        :arg params: a dict containing "mode" and "quadrature_rule".

        integrand_expressions must be indexed with :attr:`argument_multiindices`;
        these gem expressions are obtained by calling :meth:`compile_integrand`
        method or by modifying the gem expressions returned by
        :meth:`compile_integrand`.

        See :meth:`create_context` for typical calling sequence.
        """
        mode = pick_mode(params["mode"])
        return mode.Integrals(integrand_expressions,
                              params["quadrature_rule"].point_set.indices,
                              self.argument_multiindices,
                              params)

    def stash_integrals(self, reps, params, ctx):
        """Stash integral representations in ctx.

        :arg reps: integral representations.
        :arg params: a dict containing "mode".
        :arg ctx: context in which reps are stored.

        See :meth:`create_context` for typical calling sequence.
        """
        mode = pick_mode(params["mode"])
        mode_irs = ctx['mode_irs']
        mode_irs.setdefault(mode, collections.OrderedDict())
        for var, rep in zip(self.return_variables, reps):
            mode_irs[mode].setdefault(var, []).append(rep)

    def compile_gem(self, ctx):
        """Compile gem representation of integrals to impero_c.

        :arg ctx: the context containing the gem representation of integrals.
        :returns: a tuple of impero_c, oriented, needs_cell_sizes, tabulations
            required to finally construct a kernel in :meth:`construct_kernel`.

        See :meth:`create_context` for typical calling sequence.
        """
        # Finalise mode representations into a set of assignments
        mode_irs = ctx['mode_irs']

        assignments = []
        for mode, var_reps in mode_irs.items():
            assignments.extend(mode.flatten(var_reps.items(), ctx['index_cache']))

        if assignments:
            return_variables, expressions = zip(*assignments)
        else:
            return_variables = []
            expressions = []
        expressions = constant_fold_zero(expressions)

        # Need optimised roots
        options = dict(reduce(operator.and_,
                              [mode.finalise_options.items()
                               for mode in mode_irs.keys()]))
        expressions = impero_utils.preprocess_gem(expressions, **options)

        # Let the kernel interface inspect the optimised IR to register
        # what kind of external data is required (e.g., cell orientations,
        # cell sizes, etc.).
        oriented, needs_cell_sizes, tabulations, need_facet_orientation = self.register_requirements(expressions)

        # Extract Variables that are actually used
        active_variables = gem.extract_type(expressions, gem.Variable)
        # Construct ImperoC
        assignments = list(zip(return_variables, expressions))
        index_ordering = get_index_ordering(ctx['quadrature_indices'], return_variables)
        try:
            impero_c = impero_utils.compile_gem(assignments, index_ordering, remove_zeros=True)
        except impero_utils.NoopError:
            impero_c = None
        return impero_c, oriented, needs_cell_sizes, tabulations, active_variables, need_facet_orientation

    def fem_config(self):
        """Return a dictionary used with fem.compile_ufl.

        One needs to update this dictionary with "argument_multiindices",
        "quadrature_rule", and "index_cache" before using this with
        fem.compile_ufl.
        """
        info = self.integral_data_info
        integral_type = info.integral_type
        cell = info.domain.ufl_cell()
        fiat_cell = as_fiat_cell(cell)
        integration_dim, entity_ids = lower_integral_type(fiat_cell, integral_type)
        return dict(interface=self,
                    ufl_cell=cell,
                    integral_type=integral_type,
                    integration_dim=integration_dim,
                    entity_ids=entity_ids,
                    scalar_type=self.fem_scalar_type)

    def create_context(self):
        """Create builder context.

        *index_cache*

        Map from UFL FiniteElement objects to multiindices.
        This is so we reuse Index instances when evaluating the same
        coefficient multiple times with the same table.

        We also use the same dict for the unconcatenate index cache,
        which maps index objects to tuples of multiindices. These two
        caches shall never conflict as their keys have different types
        (UFL finite elements vs. GEM index objects).

        *quadrature_indices*

        List of quadrature indices used.

        *mode_irs*

        Dict for mode representations.

        For each set of integrals to make a kernel for (i,e.,
        `integral_data.integrals`), one must first create a ctx object by
        calling :meth:`create_context` method.
        This ctx object collects objects associated with the integrals that
        are eventually used to construct the kernel.
        The following is a typical calling sequence:

        .. code-block:: python3

            builder = KernelBuilder(...)
            params = {"mode": "spectral"}
            ctx = builder.create_context()
            for integral in integral_data.integrals:
                integrand = integral.integrand()
                integrand_exprs = builder.compile_integrand(integrand, params, ctx)
                integral_exprs = builder.construct_integrals(integrand_exprs, params)
                builder.stash_integrals(integral_exprs, params, ctx)
            kernel = builder.construct_kernel(kernel_name, ctx)

        """
        return {'index_cache': {},
                'quadrature_indices': [],
                'mode_irs': collections.OrderedDict()}


def set_quad_rule(params, cell, integral_type, functions):
    # Check if the integral has a quad degree attached, otherwise use
    # the estimated polynomial degree attached by compute_form_data
    try:
        quadrature_degree = params["quadrature_degree"]
    except KeyError:
        quadrature_degree = params["estimated_polynomial_degree"]
        function_degrees = [f.ufl_function_space().ufl_element().degree()
                            for f in functions]
        if all((asarray(quadrature_degree) > 10 * asarray(degree)).all()
               for degree in function_degrees):
            logger.warning("Estimated quadrature degree %s more "
                           "than tenfold greater than any "
                           "argument/coefficient degree (max %s)",
                           quadrature_degree, max_degree(function_degrees))
    quad_rule = params.get("quadrature_rule", "default")
    if isinstance(quad_rule, str):
        scheme = quad_rule
        fiat_cell = as_fiat_cell(cell)
        finat_elements = set(create_element(f.ufl_element()) for f in functions
                             if f.ufl_element().family() != "Real")
        fiat_cells = [fiat_cell] + [finat_el.complex for finat_el in finat_elements]
        fiat_cell = max_complex(fiat_cells)

        integration_dim, _ = lower_integral_type(fiat_cell, integral_type)
        quad_rule = fem.get_quadrature_rule(fiat_cell, integration_dim, quadrature_degree, scheme)
        params["quadrature_rule"] = quad_rule

    if not isinstance(quad_rule, AbstractQuadratureRule):
        raise ValueError("Expected to find a QuadratureRule object, not a %s" %
                         type(quad_rule))


def get_index_ordering(quadrature_indices, return_variables):
    split_argument_indices = tuple(chain(*(var.index_ordering()
                                           for var in return_variables)))
    return tuple(quadrature_indices) + split_argument_indices


def get_index_names(quadrature_indices, argument_multiindices, index_cache):
    index_names = []

    def name_index(index, name):
        index_names.append((index, name))
        if index in index_cache:
            for multiindex, suffix in zip(index_cache[index],
                                          string.ascii_lowercase):
                name_multiindex(multiindex, name + suffix)

    def name_multiindex(multiindex, name):
        if len(multiindex) == 1:
            name_index(multiindex[0], name)
        else:
            for i, index in enumerate(multiindex):
                name_index(index, name + str(i))

    name_multiindex(quadrature_indices, 'ip')
    for multiindex, name in zip(argument_multiindices, ['j', 'k']):
        name_multiindex(multiindex, name)
    return index_names


def lower_integral_type(fiat_cell, integral_type):
    """Lower integral type into the dimension of the integration
    subentity and a list of entity numbers for that dimension.

    :arg fiat_cell: FIAT reference cell
    :arg integral_type: integral type (string)
    """
    vert_facet_types = ['exterior_facet_vert', 'interior_facet_vert']
    horiz_facet_types = ['exterior_facet_bottom', 'exterior_facet_top', 'interior_facet_horiz']

    dim = fiat_cell.get_dimension()
    if integral_type == 'cell':
        integration_dim = dim
    elif integral_type in ['exterior_facet', 'interior_facet']:
        if isinstance(fiat_cell, TensorProductCell):
            raise ValueError("{} integral cannot be used with a TensorProductCell; need to distinguish between vertical and horizontal contributions.".format(integral_type))
        integration_dim = dim - 1
    elif integral_type == 'vertex':
        integration_dim = 0
    elif integral_type in vert_facet_types + horiz_facet_types:
        # Extrusion case
        if not isinstance(fiat_cell, TensorProductCell):
            raise ValueError("{} integral requires a TensorProductCell.".format(integral_type))
        basedim, extrdim = dim
        assert extrdim == 1

        if integral_type in vert_facet_types:
            integration_dim = (basedim - 1, 1)
        elif integral_type in horiz_facet_types:
            integration_dim = (basedim, 0)
    else:
        raise NotImplementedError("integral type %s not supported" % integral_type)

    if integral_type == 'exterior_facet_bottom':
        entity_ids = [0]
    elif integral_type == 'exterior_facet_top':
        entity_ids = [1]
    else:
        entity_ids = list(fiat_cell.get_topology()[integration_dim])

    return integration_dim, entity_ids


def pick_mode(mode):
    "Return one of the specialized optimisation modules from a mode string."
    try:
        from firedrake_citations import Citations
        cites = {"vanilla": ("Homolya2017", ),
                 "coffee": ("Luporini2016", "Homolya2017", ),
                 "spectral": ("Luporini2016", "Homolya2017", "Homolya2017a"),
                 "tensor": ("Kirby2006", "Homolya2017", )}
        for c in cites[mode]:
            Citations().register(c)
    except ImportError:
        pass
    if mode == "vanilla":
        import tsfc.vanilla as m
    elif mode == "coffee":
        import tsfc.coffee_mode as m
    elif mode == "spectral":
        import tsfc.spectral as m
    elif mode == "tensor":
        import tsfc.tensor as m
    else:
        raise ValueError("Unknown mode: {}".format(mode))
    return m


def check_requirements(ir):
    """Look for cell orientations, cell sizes, and collect tabulations
    in one pass."""
    cell_orientations = False
    cell_sizes = False
    facet_orientation = False
    rt_tabs = {}
    for node in traversal(ir):
        if isinstance(node, gem.Variable):
            if node.name == "cell_orientations":
                cell_orientations = True
            elif node.name == "cell_sizes":
                cell_sizes = True
            elif node.name.startswith("rt_"):
                rt_tabs[node.name] = node.shape
            elif node.name == "facet_orientation":
                facet_orientation = True
    return cell_orientations, cell_sizes, tuple(sorted(rt_tabs.items())), facet_orientation


def prepare_constant(constant, number):
    """Bridges the kernel interface and the GEM abstraction for
    Constants.

    :arg constant: Firedrake Constant
    :arg number: Value to uniquely identify the constant
    :returns: (funarg, expression)
         expression - GEM expression referring to the Constant value(s)
    """
    value_size = numpy.prod(constant.ufl_shape, dtype=int)
    return gem.reshape(gem.Variable(f"c_{number}", (value_size,)),
                       constant.ufl_shape)


def prepare_coefficient(coefficient, name, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg coefficient: UFL Coefficient
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         expression - GEM expression referring to the Coefficient
                      values
    """
    assert isinstance(interior_facet, bool)

    if coefficient.ufl_element().family() == 'Real':
        # Constant
        value_size = coefficient.ufl_function_space().value_size
        expression = gem.reshape(gem.Variable(name, (value_size,)),
                                 coefficient.ufl_shape)
        return expression

    finat_element = create_element(coefficient.ufl_element())
    shape = finat_element.index_shape
    size = numpy.prod(shape, dtype=int)

    if not interior_facet:
        expression = gem.reshape(gem.Variable(name, (size,)), shape)
    else:
        varexp = gem.Variable(name, (2 * size,))
        plus = gem.view(varexp, slice(size))
        minus = gem.view(varexp, slice(size, 2 * size))
        expression = (gem.reshape(plus, shape), gem.reshape(minus, shape))
    return expression


def prepare_arguments(arguments, multiindices, interior_facet=False, diagonal=False):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg arguments: UFL Arguments
    :arg multiindices: Argument multiindices
    :arg interior_facet: interior facet integral?
    :arg diagonal: Are we assembling the diagonal of a rank-2 element tensor?
    :returns: (funarg, expression)
         expressions - GEM expressions referring to the argument
                       tensor
    """
    assert isinstance(interior_facet, bool)

    if len(arguments) == 0:
        # No arguments
        expression = gem.Indexed(gem.Variable("A", (1,)), (0,))
        return (expression, )

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

    varexp = gem.Variable("A", c_shape)
    expressions = [expression(gem.view(varexp, *slices)) for slices in slicez]
    return tuple(prune(expressions))
