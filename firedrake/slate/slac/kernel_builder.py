from __future__ import absolute_import, print_function, division
from six import iteritems

from collections import OrderedDict

from coffee import base as ast

from firedrake.slate.slate import (TensorBase, Tensor, TensorOp,
                                   Action, Inverse, Transpose)
from firedrake.slate.slac.utils import Transformer, collect_reference_count
from firedrake.utils import cached_property

from ufl import MixedElement


class KernelBuilder(object):
    """A helper class for constructing Slate kernels.

    This class provides access to all temporaries and subkernels associated
    with a Slate expression. If the Slate expression contains nodes that
    require operations on already assembled data (such as the action of a
    slate tensor on a `ufl.Coefficient`), this class provides access to the
    expression which needs special handling.

    Instructions for assembling the full kernel AST of a Slate expression is
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
        self.needs_mesh_layers = False
        self.oriented = False
        self.finalized_ast = None
        self._is_finalized = False

        # Generate coefficient map (both mixed and non-mixed cases handled)
        self.coefficient_map = prepare_coefficients(expression)
        # Initialize temporaries and any auxiliary temporaries
        temps, aux_exprs, expr_dags = generate_expr_data(expression)
        # Sort by temporary str: 'T0', 'T1', etc.
        self.temps = OrderedDict(sorted(iteritems(temps),
                                        key=lambda x: str(x[1])))

        # Collect the reference count of operands in auxiliary expressions
        self._ref_counts = collect_reference_count(expr_dags)
        # Since the most complicated expressions get caught first, we
        # reorder by expression complexity to create temporaries for the
        # inner-most instances first (using reference count later to decide
        # if it's worth the extra memory)
        self.aux_temps = OrderedDict.fromkeys(sorted(aux_exprs,
                                                     key=lambda x:
                                                     x._complexity))

    @property
    def integral_type(self):
        """Returns the integral type associated with a Slate kernel.

        Note that Slate kernels are always of type 'cell' since these
        are localized kernels for element-wise linear algebra. This
        may change in the future if we want Slate to be used for
        LDG/CDG finite element discretizations.
        """
        return "cell"

    def require_cell_facets(self):
        """Assigns `self.needs_cell_facets` to be `True` if facet integrals
        are present.
        """
        self.needs_cell_facets = True

    def require_mesh_layers(self):
        """Assigns `self.needs_mesh_layers` to be `True` if mesh levels are
        needed.
        """
        self.needs_mesh_layers = True

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
        """Gathers all `ContextKernel`s containing all TSFC kernels,
        and integral type information.
        """
        from firedrake.slate.slac.tsfc_driver import compile_terminal_form

        cxt_list = [compile_terminal_form(expr, prefix="subkernel%d_" % i,
                                          tsfc_parameters=self.tsfc_parameters)
                    for i, expr in enumerate(self.temps)]

        cxt_kernels = [cxt_k for cxt_tuple in cxt_list
                       for cxt_k in cxt_tuple]
        return cxt_kernels

    def construct_macro_kernel(self, name, args, statements):
        """Constructs a macro kernel function that calls any subkernels.
        The :class:`Transformer` is used to perform the conversion from
        standard C into the Eigen C++ template library syntax.

        :arg name: a string denoting the name of the macro kernel.
        :arg args: a list of arguments for the macro_kernel.
        :arg statements: a `coffee.base.Block` of instructions, which contains
                         declarations of temporaries, function calls to all
                         subkernels and any auxilliary information needed to
                         evaulate the Slate expression.
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
        """Prepares the kernel AST by transforming all outpute/input
        references to Slate tensors with eigen references and updates
        any orientation information.
        """
        kernel_list = []
        transformer = Transformer()
        oriented = self.oriented

        cxt_kernels = self.context_kernels
        splitkernels = [splitkernel for cxt_k in cxt_kernels
                        for splitkernel in cxt_k.tsfc_kernels]

        for splitkernel in splitkernels:
            oriented = oriented or splitkernel.kinfo.oriented
            # TODO: Extend multiple domains support
            assert splitkernel.kinfo.subdomain_id == "otherwise"
            kast = transformer.visit(splitkernel.kinfo.kernel._ast)
            kernel_list.append(kast)

        self.oriented = oriented
        self.finalized_ast = kernel_list
        self._is_finalized = True

    def construct_ast(self, macro_kernels):
        """Constructs the final kernel AST.

        :arg macro_kernels: A `list` of macro kernel functions, which
                            call subkernels and perform elemental
                            linear algebra.

        Returns: The complete kernel AST as a COFFEE `ast.Node`
        """
        assert isinstance(macro_kernels, list), (
            "Please wrap all macro kernel functions in a list"
        )
        assert self._is_finalized, (
            "AST not finalized. Did you forget to call "
            "builder._finalize_kernels_and_update()?"
        )
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


def generate_expr_data(expr, temps=None, aux_exprs=None, expr_dags=None):
    """This function generates a mapping of the form:

       ``temporaries = {node: symbol_name}``

    where `node` objects are :class:`slate.TensorBase` nodes, and
    `symbol_name` are :class:`coffee.base.Symbol` objects. In addition,
    this function will return a list `aux_exprs` of any expressions that
    require special handling in the compiler. This includes expressions
    that require performing operations on already assembled data.

    This mapping is used in the :class:`KernelBuilder` to provide direct
    access to all temporaries associated with a particular slate expression.

    :arg expression: a :class:`slate.TensorBase` object.
    :arg temps: a dictionary that becomes populated recursively and is later
                returned as the temporaries map. This argument is initialized
                as an empty `dict` before recursion starts.
    :arg aux_exprs: a set-like object that becomes populated recursively and
                    is later returned as a unique list of auxiliary expressions
                    that require special handling in Slate's linear algebra
                    compiler.
    :arg expr_dags: a list of collected operands (including the top-level
                    expression) that make up the expression DAG. This is
                    used to compute how many times a particular operand is
                    referenced in a given Slate expression.

    Returns: the arguments temps, aux_temps and expr_dags.
    """
    # Prepare temporaries map and auxiliary expressions list
    # NOTE: Ordering here matters, especially when running
    # Slate in parallel.
    if temps is None:
        temps = OrderedDict()

    if aux_exprs is None:
        aux_exprs = OrderedDict()

    if expr_dags is None:
        expr_dags = []

    if isinstance(expr, Tensor):
        expr_dags.append(expr)
        temps.setdefault(expr, ast.Symbol("T%d" % len(temps)))

    elif isinstance(expr, TensorOp):
        expr_dags.append(expr)
        # We handle Action, Transpose and Inverse nodes by creating
        # separate temporaries for these quantities later
        if isinstance(expr, (Action, Transpose, Inverse)):
            aux_exprs.setdefault(expr)

        # Send operands through recursively
        map(lambda x: generate_expr_data(x, temps=temps,
                                         aux_exprs=aux_exprs), expr.operands)
    else:
        raise NotImplementedError("Type %s not supported." % type(expr))

    return temps, aux_exprs, expr_dags
