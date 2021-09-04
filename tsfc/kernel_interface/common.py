import collections
import operator
from functools import reduce

import numpy

import coffee.base as coffee

import gem

from gem.utils import cached_property
import gem.impero_utils as impero_utils

from tsfc.driver import lower_integral_type, set_quad_rule, pick_mode, get_index_ordering
from tsfc import fem, ufl_utils
from tsfc.kernel_interface import KernelInterface
from tsfc.finatinterface import as_fiat_cell


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
        self.coefficient_map = {}

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

    def construct_kernel(self, name, args, body):
        """Construct a COFFEE function declaration with the
        accumulated glue code.

        :arg name: function name
        :arg args: function argument list
        :arg body: function body (:class:`coffee.Block` node)
        :returns: :class:`coffee.FunDecl` object
        """
        assert isinstance(body, coffee.Block)
        body_ = coffee.Block(self.prepare + body.children + self.finalise)
        return coffee.FunDecl("void", name, args, body_, pred=["static", "inline"])

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
        functions = list(info.arguments) + [self.coordinate(info.domain)] + list(info.coefficients)
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

        # Need optimised roots
        options = dict(reduce(operator.and_,
                              [mode.finalise_options.items()
                               for mode in mode_irs.keys()]))
        expressions = impero_utils.preprocess_gem(expressions, **options)

        # Let the kernel interface inspect the optimised IR to register
        # what kind of external data is required (e.g., cell orientations,
        # cell sizes, etc.).
        oriented, needs_cell_sizes, tabulations = self.register_requirements(expressions)

        # Construct ImperoC
        assignments = list(zip(return_variables, expressions))
        index_ordering = get_index_ordering(ctx['quadrature_indices'], return_variables)
        try:
            impero_c = impero_utils.compile_gem(assignments, index_ordering, remove_zeros=True)
        except impero_utils.NoopError:
            impero_c = None
        return impero_c, oriented, needs_cell_sizes, tabulations

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
