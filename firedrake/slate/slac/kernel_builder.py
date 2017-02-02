from __future__ import absolute_import, print_function, division
from six.moves import filter, map

from coffee import base as ast

from firedrake.slate.slate import TensorBase, Tensor, UnaryOp, BinaryOp, Action
from firedrake.slate.slac.utils import RemoveRestrictions, Transformer
from firedrake.tsfc_interface import compile_form

from functools import partial

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl import Form, MixedElement


class KernelBuilder(object):
    """A helper class for constructing SLATE kernels.

    This class provides access to all temporaries and subkernels associated with a SLATE
    expression. If the SLATE expression contains nodes that require operations on already
    assembled data (such as the action of a slate tensor on a `ufl.Coefficient`), this class
    provides access to the expression which needs special handling.

    Instructions for assembling the full kernel AST of a SLATE expression is
    provided by the method `construct_ast`.
    """
    def __init__(self, expression, tsfc_parameters=None):
        """Constructor for the KernelBuilder class.

        :arg expression: a :class:`TensorBase` object.
        :arg tsfc_parameters: an optional `dict` of parameters to provide to TSFC when
                              constructing subkernels associated with the expression.
        """
        assert isinstance(expression, TensorBase)
        self.expression = expression
        self.needs_cell_facets = False

        # Generate coefficient map (both mixed and non-mixed cases handled)
        self.coefficient_map = prepare_coefficients(expression)

        # Initialize temporaries, auxiliary expressions and tsfc kernels
        self.temps, self.aux_exprs = prepare_temps_and_aux_exprs(expression)
        self.kernel_exprs = prepare_tsfc_kernels(self.temps, tsfc_parameters=tsfc_parameters)

    def require_cell_facets(self):
        """Assigns `self.needs_cell_facets` to be `True` if facet integrals are present."""
        self.needs_cell_facets = True

    def extract_coefficient(self, coefficient):
        """Extracts a coefficient from the coefficient_map. This handles both the case when
        the coefficient is defined on a mixed or non-mixed function space.
        """
        if type(coefficient.ufl_element()) == MixedElement:
            sub_coeffs = coefficient.split()
            return tuple(self.coefficient_map[c] for c in sub_coeffs)
        else:
            return (self.coefficient_map[coefficient],)

    def construct_ast(self, name, args, statements):
        """Constructs the full kernel AST of a given SLATE expression. The :class:`Transformer`
        is used to perform the conversion from standard C into the Eigen C++ template library
        syntax.

        :arg name: a string denoting the name of the macro kernel.
        :arg args: a list of arguments for the macro_kernel.
        :arg statements: a `coffee.base.Block` of instructions, which contains declarations
                         of temporaries, function calls to all subkernels and any auxilliary
                         information needed to evaulate the SLATE expression.
                         E.g. facet integral loops and action loops.

        Returns: the full kernel AST to be converted into a PyOP2 kernel, as well as any
                 orientation information.
        """
        # all kernel body statements must be wrapped up as a coffee.base.Block
        assert isinstance(statements, ast.Block)

        macro_kernel = ast.FunDecl("void", name, args, statements, pred=["static", "inline"])

        kernel_list = []
        transformer = Transformer()
        oriented = False
        # Assume self.kernel_exprs is populated at this point
        for kernel_items in self.kernel_exprs.values():
            for ks in kernel_items:
                oriented = oriented or ks.kinfo.oriented
                # TODO: Extend multiple domains support
                assert ks.kinfo.subdomain_id == "otherwise"
                kast = transformer.visit(ks.kinfo.kernel._ast)
                kernel_list.append(kast)
        kernel_list.append(macro_kernel)

        return ast.Node(kernel_list), oriented


def prepare_coefficients(expression):
    """Prepares the coefficient map that maps a `ufl.Coefficient` to `coffee.Symbol`"""
    coefficient_map = {}

    for i, coefficient in enumerate(expression.coefficients()):
        if type(coefficient.ufl_element()) == MixedElement:
            for j, sub_coeff in enumerate(coefficient.split()):
                coefficient_map[sub_coeff] = ast.Symbol("w_%d_%d" % (i, j))
        else:
            coefficient_map[coefficient] = ast.Symbol("w_%d" % i)

    return coefficient_map


def prepare_tsfc_kernels(temps, tsfc_parameters=None):
    """This function generates a mapping of the form:

       ``kernel_exprs = {terminal_node: kernels}``

    where `terminal_node` objects are :class:`slate.Tensor` nodes and `kernels` is an iterable
    of `namedtuple` objects, `SplitKernel`, provided by TSFC.

    This mapping is used in :class:`SlateKernelBuilder` to provide direct access to all
    `SplitKernel` objects associated with a `slate.Tensor` node.

    :arg temps: a mapping of the form ``{terminal_node: symbol_name}``
                (see :meth:`prepare_temporaries`).
    :arg tsfc_parameters: an optional `dict` of parameters to pass onto TSFC.
    """
    kernel_exprs = {}

    for expr in temps.keys():
        integrals = expr.form.integrals()
        mapper = RemoveRestrictions()
        integrals = list(map(partial(map_integrand_dags, mapper), integrals))
        prefix = "subkernel%d_" % len(kernel_exprs)

        # Now we split integrals by type: interior_facet and all other cases
        # First, the interior_facet case:
        interior_facet_intergrals = list(filter(lambda x: x.integral_type() == "interior_facet", integrals))

        # Now we reconstruct all interior_facet integrals to be of type: exterior_facet
        # This is because locally over each cell, SLATE views them as being "exterior"
        # with respect to the cell.
        interior_facet_intergrals = [it.reconstruct(integral_type="exterior_facet")
                                     for it in interior_facet_intergrals]
        # Now for the rest:
        other_integrals = list(filter(lambda x: x.integral_type() != "interior_facet", integrals))

        forms = (Form(interior_facet_intergrals), Form(other_integrals))
        compiled_forms = []
        for form in forms:
            compiled_forms.extend(compile_form(form, prefix, parameters=tsfc_parameters))

        kernel_exprs[expr] = tuple(compiled_forms)

    return kernel_exprs


def prepare_temps_and_aux_exprs(expression, temps=None, aux_exprs=None):
    """This function generates a mapping of the form:

       ``temporaries = {terminal_node: symbol_name}``

    where `terminal_node` objects are :class:`slate.Tensor` nodes, and `symbol_name` are
    :class:`coffee.base.Symbol` objects.

    In addition, this function will return a list `aux_exprs` of any expressions that require
    special handling in the compiler. This includes expressions that require performing
    operations on already assembled data.

    This mapping is used in the :class:`SlateKernelBuilder` to provide direct access to all
    temporaries associated with a particular slate expression.

    :arg expression: a :class:`slate.TensorBase` object.
    :arg temps: a dictionary that becomes populated recursively and is later returned as the
                temporaries map. This argument is initialized as an empty `dict` before
                recursion starts.
    :arg aux_exprs: a list that becomes populated recursively and is later returned as the
                    list of auxiliary expressions that require special handling in SLATE's
                    linear algebra compiler

    Returns: the arguments temps and aux_exprs.
    """
    # Prepare temporaries map and auxiliary expressions list
    if temps is None:
        temps = {}
    else:
        temps = temps

    if aux_exprs is None:
        aux_exprs = []
    else:
        aux_exprs = aux_exprs

    if isinstance(expression, Tensor):
        if expression not in temps.keys():
            temps[expression] = ast.Symbol("T%d" % len(temps))

    elif isinstance(expression, Action):
        # This is a special case where we need to handle this expression separately from the rest
        aux_exprs.append(expression)
        # Pass in the acting tensor to extract any necessary temporaries
        prepare_temps_and_aux_exprs(expression.tensor, temps=temps, aux_exprs=aux_exprs)

    elif isinstance(expression, (UnaryOp, BinaryOp)):
        for operand in expression.operands:
            prepare_temps_and_aux_exprs(operand, temps=temps, aux_exprs=aux_exprs)

    else:
        raise NotImplementedError("Expression of type %s not currently supported." % type(expression))

    return temps, aux_exprs
