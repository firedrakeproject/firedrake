from __future__ import absolute_import, print_function, division
from six.moves import filter, map

import collections

from coffee import base as ast

from firedrake.slate.slate import TensorBase, Tensor, UnaryOp, BinaryOp, Action
from firedrake.slate.slac.tsfc_driver import compile_terminal_form
from firedrake.slate.slac.utils import Transformer
from firedrake.utils import cached_property

from ufl import MixedElement


class KernelBuilder(object):
    """A helper class for constructing SLATE kernels.

    This class provides access to all temporaries and subkernels associated
    with a SLATE expression. If the SLATE expression contains nodes that
    require operations on already assembled data (such as the action of a
    slate tensor on a `ufl.Coefficient`), this class provides access to the
    expression which needs special handling.

    Instructions for assembling the full kernel AST of a SLATE expression is
    provided by the method `construct_ast`.
    """
    def __init__(self, expression, tsfc_parameters=None):
        """Constructor for the KernelBuilder class.

        :arg expression: a :class:`TensorBase` object.
        :arg tsfc_parameters: an optional `dict` of parameters to provide to
                              TSFC when constructing subkernels associated
                              with the expression.
        """
        assert isinstance(expression, TensorBase)
        self.expression = expression
        self.tsfc_parameters = tsfc_parameters
        self.needs_cell_facets = False
        self.oriented = False
        self.finalized_ast = None

        # Generate coefficient map (both mixed and non-mixed cases handled)
        self.coefficient_map = prepare_coefficients(expression)

        self.temps, self.aux_exprs = generate_expr_data(expression)

    @property
    def integral_type(self):
        """
        """
        return "cell"

    def require_cell_facets(self):
        """Assigns `self.needs_cell_facets` to be `True` if facet integrals
        are present.
        """
        self.needs_cell_facets = True

    def get_temporary(self, expr):
        """
        """
        if expr not in self.temps:
            raise ValueError("No temporary for the given expression")

        return self.temps[expr]

    def coefficient(self, coefficient):
        """Extracts a coefficient from the coefficient_map. This handles both
        the case when the coefficient is defined on a mixed or non-mixed
        function space.
        """
        if type(coefficient.ufl_element()) == MixedElement:
            sub_coeffs = coefficient.split()
            return tuple(self.coefficient_map[c] for c in sub_coeffs)
        else:
            return (self.coefficient_map[coefficient],)

    @cached_property
    def context_kernels(self):
        """
        """
        cxt_list = gather_context_kernels(self.temps,
                                          self.tsfc_parameters)
        cxt_kernels = [cxt_k for cxt_tuple in cxt_list
                       for cxt_k in cxt_tuple]
        return cxt_kernels

    @cached_property
    def full_kernel_list(self):
        """
        """
        cxt_kernels = self.context_kernels
        splitkernels = [splitkernel for cxt_k in cxt_kernels
                        for splitkernel in cxt_k.tsfc_kernels]
        return splitkernels

    def construct_macro_kernel(self, name, args, statements):
        """Constructs a macro kernel function that calls any subkernels.
        The :class:`Transformer` is used to perform the conversion from
        standard C into the Eigen C++ template library syntax.

        :arg name: a string denoting the name of the macro kernel.
        :arg args: a list of arguments for the macro_kernel.
        :arg statements: a `coffee.base.Block` of instructions, which contains
                         declarations of temporaries, function calls to all
                         subkernels and any auxilliary information needed to
                         evaulate the SLATE expression.
                         E.g. facet integral loops and action loops.
        """
        # all kernel body statements must be wrapped up as a coffee.base.Block
        assert isinstance(statements, ast.Block), (
            "Body statements must be wrapped in an ast.Block"
        )

        macro_kernel = ast.FunDecl("void", name, args,
                                   statements, pred=["static", "inline"])
        return macro_kernel

    def _finalize_kernels_and_update(self):
        """
        """
        kernel_list = []
        transformer = Transformer()
        oriented = self.oriented

        for splitkernel in self.full_kernel_list:
            oriented = oriented or splitkernel.kinfo.oriented
            # TODO: Extend multiple domains support
            assert splitkernel.kinfo.subdomain_id == "otherwise"
            kast = transformer.visit(splitkernel.kinfo.kernel._ast)
            kernel_list.append(kast)

        self.oriented = oriented
        self.finalized_ast = kernel_list

    def construct_ast(self, macro_kernels):
        """
        """
        assert isinstance(macro_kernels, list), (
            "Please wrap all macro kernel functions in a list"
        )
        self._finalize_kernels_and_update()
        kernel_ast = self.finalized_ast
        kernel_ast.extend(macro_kernels)

        return ast.Node(kernel_ast)


def prepare_coefficients(expression):
    """Prepares the coefficient map that maps a `ufl.Coefficient`
    to `coffee.Symbol`.
    """
    coefficient_map = {}

    for i, coefficient in enumerate(expression.coefficients()):
        if type(coefficient.ufl_element()) == MixedElement:
            for j, sub_coeff in enumerate(coefficient.split()):
                coefficient_map[sub_coeff] = ast.Symbol("w_%d_%d" % (i, j))
        else:
            coefficient_map[coefficient] = ast.Symbol("w_%d" % i)

    return coefficient_map


def gather_context_kernels(temps, tsfc_parameters=None):
    """
    """
    cxt_list = [compile_terminal_form(expr, tsfc_parameters)
                for expr in temps]
    return cxt_list


def generate_expr_data(expr, temps=None, aux_exprs=None):
    """
    """
    # Prepare temporaries map and auxiliary expressions list
    if temps is None:
        temps = collections.defaultdict()

    if aux_exprs is None:
        aux_exprs = []

    if isinstance(expr, Tensor):
        if expr not in temps:
            temps[expr] = ast.Symbol("T%d" % len(temps))

    elif isinstance(expr, Action):
        aux_exprs.append(expr)
        # Pass in the acting tensor to extract any necessary temporaries
        generate_expr_data(expr.operands[0], temps=temps, aux_exprs=aux_exprs)

    elif isinstance(expr, (UnaryOp, BinaryOp)):
        map(lambda x: generate_expr_data(x, temps=temps,
                                         aux_exprs=aux_exprs), expr.operands)
    else:
        raise NotImplementedError("Type %s not supported." % type(expr))

    return temps, aux_exprs
