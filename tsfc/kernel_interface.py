from __future__ import absolute_import

import numpy

import coffee.base as coffee

import gem


class Kernel(object):
    __slots__ = ("ast", "integral_type", "oriented", "subdomain_id",
                 "domain_number",
                 "coefficient_numbers", "__weakref__")
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
    """
    def __init__(self, ast=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None,
                 coefficient_numbers=()):
        # Defaults
        self.ast = ast
        self.integral_type = integral_type
        self.oriented = oriented
        self.domain_number = domain_number
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        super(Kernel, self).__init__()


class KernelBuilderBase(object):
    """Helper class for building local assembly kernels."""

    def __init__(self, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        assert isinstance(interior_facet, bool)
        self.interior_facet = interior_facet

        self.prepare = []
        self.finalise = []

        # Coefficients
        self.coefficient_map = {}

    def coefficient(self, ufl_coefficient, restriction):
        """A function that maps :class:`ufl.Coefficient`s to GEM
        expressions."""
        kernel_arg = self.coefficient_map[ufl_coefficient]
        if ufl_coefficient.ufl_element().family() == 'Real':
            return kernel_arg
        else:
            return gem.partial_indexed(kernel_arg, {None: (), '+': (0,), '-': (1,)}[restriction])

    def cell_orientation(self, restriction):
        """Cell orientation as a GEM expression."""
        f = {None: 0, '+': 0, '-': 1}[restriction]
        co_int = self.cell_orientations_mapper(f)
        return gem.Conditional(gem.Comparison("==", co_int, gem.Literal(1)),
                               gem.Literal(-1),
                               gem.Conditional(gem.Comparison("==", co_int, gem.Zero()),
                                               gem.Literal(1),
                                               gem.Literal(numpy.nan)))

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

    def arguments(self, arguments, indices):
        """Prepare arguments. Adds glue code for the arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg indices: GEM argument indices
        :returns: COFFEE function argument and GEM expression
                  representing the argument tensor
        """
        funarg, prepare, expressions, finalise = self.prepare_arguments(
            arguments, indices, interior_facet=self.interior_facet)
        self.apply_glue(prepare, finalise)
        return funarg, expressions

    def cell_orientations(self, integral_type):
        """Prepare cell orientations. Adds glue code for cell orienatations
        and stores cell orientations expression.

        :arg integral_type
        :returns: list of COFFEE function arguments for cell orientations
        """
        funargs, prepare, expressions = self.prepare_cell_orientations(integral_type)
        self.apply_glue(prepare)
        self._cell_orientations = expressions
        return funargs

    @staticmethod
    def prepare_arguments(arguments, indices, interior_facet=False):
        """Bridges the kernel interface and the GEM abstraction for
        Arguments.

        :arg arguments: UFL Arguments
        :arg indices: Argument indices
        :arg interior_facet: interior facet integral?
        :returns: (funarg, prepare, expression, finalise)
             funarg      - :class:`coffee.Decl` function argument
             prepare     - list of COFFEE nodes to be prepended to the
                           kernel body
             expressions - GEM expressions referring to the argument
                           tensor
             finalise    - list of COFFEE nodes to be appended to the
                           kernel body
        """
        raise NotImplementedError("This class is abstract")

    @staticmethod
    def prepare_cell_orientations(integral_type):
        """Bridges the kernel interface and the GEM abstraction for
        cell orientations.

        :arg integral_type
        :returns: (funarg, prepare, expression)
             funargs    - list of :class:`coffee.Decl` function argument
             prepare    - list of COFFEE nodes to be prepended to the
                          kernel body
             expressions- list of GEM expressions referring to facets
        """
        raise NotImplementedError("This class is abstract")


# Avoid circular import
from tsfc.backends import pyop2, ufc


class KernelBuilder(KernelBuilderBase):
    def __new__(cls, backend, *args, **kwargs):
        if backend == "pyop2":
            return pyop2.KernelBuilder(*args, **kwargs)
        elif backend == "ufc":
            return ufc.KernelBuilder(*args, **kwargs)
        else:
            raise ValueError("Unknown backend '%s'" % backend)
