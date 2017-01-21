from __future__ import absolute_import, print_function, division
from six.moves import filter, map

import collections

from coffee import base as ast

from firedrake.slate.slate import TensorBase, Tensor, UnaryOp, BinaryOp, Action
from firedrake.slate.slac.tsfc_manager import TSFCKernelManager
from firedrake.slate.slac.utils import Transformer

from ufl import MixedElement


ExpressionData = collections.namedtuple("ExpressionData",
                                        ["temporaries",
                                         "auxiliary_exprs",
                                         "kernel_managers"])


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
        self.needs_cell_facets = False

        # Generate coefficient map (both mixed and non-mixed cases handled)
        self.coefficient_map = prepare_coefficients(expression)

        # Initialize temporaries, auxiliary expressions and tsfc managers
        temps, aux_exprs = generate_expr_data(expression)
        tsfc_managers = gather_tsfc_managers(temps)
        self.expr_data = ExpressionData(temporaries=temps,
                                        auxiliary_exprs=aux_exprs,
                                        kernel_managers=tsfc_managers)

    def require_cell_facets(self):
        """Assigns `self.needs_cell_facets` to be `True` if facet integrals
        are present.
        """
        self.needs_cell_facets = True

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

    def construct_ast(self, name, args, statements):
        """Constructs the full kernel AST of a given SLATE expression.
        The :class:`Transformer` is used to perform the conversion from
        standard C into the Eigen C++ template library syntax.

        :arg name: a string denoting the name of the macro kernel.
        :arg args: a list of arguments for the macro_kernel.
        :arg statements: a `coffee.base.Block` of instructions, which contains
                         declarations of temporaries, function calls to all
                         subkernels and any auxilliary information needed to
                         evaulate the SLATE expression.
                         E.g. facet integral loops and action loops.

        Returns: the full kernel AST to be converted into a PyOP2 kernel,
                 as well as any orientation information.
        """
        # all kernel body statements must be wrapped up as a coffee.base.Block
        assert isinstance(statements, ast.Block)

        macro_kernel = ast.FunDecl("void", name, args,
                                   statements, pred=["static", "inline"])

        kernel_list = []
        transformer = Transformer()
        oriented = False
        # Assume self.expr_data is populated at this point
        # with tsfc managers already compiled tsfc kernels
        managers = self.expr_data.kernel_managers
        for tsfc_manager in managers.values():
            for kernel_items in tsfc_manager.kernels.values():
                for splitkernel in kernel_items:
                    oriented = oriented or splitkernel.kinfo.oriented
                    # TODO: Extend multiple domains support
                    assert splitkernel.kinfo.subdomain_id == "otherwise"
                    kast = transformer.visit(splitkernel.kinfo.kernel._ast)
                    kernel_list.append(kast)
        kernel_list.append(macro_kernel)

        return ast.Node(kernel_list), oriented


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


def gather_tsfc_managers(temps, tsfc_parameters=None):
    """
    """
    kernel_managers = {}

    for expr in temps.keys():
        kernel_managers[expr] = TSFCKernelManager(tensor=expr,
                                                  parameters=tsfc_parameters)
    return kernel_managers


def generate_expr_data(expr, temps=None, aux_exprs=None):
    """
    """
    # Prepare temporaries map and auxiliary expressions list
    if temps is None:
        temps = {}

    if aux_exprs is None:
        aux_exprs = []

    if isinstance(expr, Tensor):
        if expr not in temps.keys():
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
