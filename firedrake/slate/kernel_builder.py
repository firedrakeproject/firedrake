from __future__ import absolute_import, print_function, division


from coffee import base as ast

from firedrake.slate.slate import TensorBase, Tensor, UnaryOp, BinaryOp, TensorAction
from firedrake.slate.utils import RemoveRestrictions, Transformer
from firedrake.tsfc_interface import compile_form

from functools import partial

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.coefficient import Coefficient
from ufl.form import Form


class SlateKernelBuilder(object):
    """A helper class for constructing SLATE kernels.

    This class provides access to all temporaries and subkernels associated with a SLATE
    expression. If the SLATE expression contains nodes that require operations on already
    assembled data (such as the action of a slate tensor on a `ufl.Coefficient`), this class
    provides access to the necessary components.

    Instructions for assembling the full kernel AST of a SLATE expression is
    provided by the method `construct_ast`.
    """
    def __init__(self, expression, tsfc_parameters=None):
        """Constructor for the SlateKernelBuilder class.

        :arg expression: a :class:`TensorBase` object.
        :arg tsfc_parameters: an optional `dict` of parameters to provide to TSFC when
                              constructing subkernels associated with the expression.
        """
        assert isinstance(expression, TensorBase)
        self.expression = expression

        # Initialize temporaries and tsfc kernels
        self.temps = prepare_temporaries(expression)
        self.kernel_exprs = prepare_tsfc_kernels(self.temps, tsfc_parameters=tsfc_parameters)

        # Gather data for expressions that require special handling
        self.data_exprs = exprs_from_assembled_data(expression)

    def coefficient_map(self):
        """Returns a mapping from `ufl.Coefficient` to its corresponding `coffee.base.Symbol` object."""
        return dict((c, ast.Symbol("w%d" % i))
                    for i, c in enumerate(self.expression.coefficients()))

    def construct_ast(self, name, args, statements):
        """Constructs the full kernel AST of a given SLATE expression. The :class:`Transformer` is used to
        perform the conversion from standard C into the Eigen C++ template library syntax.

        :arg name: a string denoting the name of the macro kernel.
        :arg args: a list of arguments for the macro_kernel.
        :arg statements: a `coffee.base.Block` of instructions, which contains declarations of temporaries,
                         function calls to all subkernels and any auxilliary information needed to evaulate
                         the SLATE expression. E.g. facet integral loops and action loops.

        Returns: the full kernel AST to be converted into a PyOP2 kernel, as well as any orientation
                 information.
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
                # TODO: Is this true for SLATE?
                assert ks.kinfo.subdomain_id == "otherwise"
                kast = transformer.visit(ks.kinfo.kernel._ast)
                kernel_list.append(kast)
        kernel_list.append(macro_kernel)

        return ast.Node(kernel_list), oriented


def prepare_tsfc_kernels(temps, tsfc_parameters=None):
    """This function generates a mapping of the form:

       ``kernel_exprs = {terminal_node: kernels}``

    where `terminal_node` objects are :class:`slate.Tensor` nodes and `kernels` is an iterable
    of `namedtuple` objects, `SplitKernel`, provided by TSFC.

    This mapping is used in :class:`SlateKernelBuilder` to provide direct access to all `SplitKernel`
    objects associated with a `slate.Tensor` node.

    :arg temps: a mapping of the form ``{terminal_node: symbol_name}`` (see :meth:`prepare_temporaries`).
    :arg tsfc_parameters: an optional `dict` of parameters to pass onto TSFC.
    """
    kernel_exprs = {}

    for expr in temps.keys():
        integrals = expr.form.integrals()
        mapper = RemoveRestrictions()
        integrals = map(partial(map_integrand_dags, mapper), integrals)
        prefix = "subkernel%d_" % len(kernel_exprs)

        # Now we split integrals by type: interior_facet and all other cases
        # First, the interior_facet case:
        interior_facet_intergrals = filter(lambda x: x.integral_type() == "interior_facet", integrals)

        # Now we reconstruct all interior_facet integrals to be of type: exterior_facet
        # This is because locally over each cell, SLATE views them as being "exterior"
        # with respect to the cell.
        interior_facet_intergrals = [it.reconstruct(integral_type="exterior_facet")
                                     for it in interior_facet_intergrals]
        # Now for the rest:
        other_integrals = filter(lambda x: x.integral_type() != "interior_facet", integrals)

        forms = (Form(interior_facet_intergrals), Form(other_integrals))
        compiled_forms = []
        for form in forms:
            compiled_forms.extend(compile_form(form, prefix, parameters=tsfc_parameters))

        kernel_exprs[expr] = tuple(compiled_forms)

    return kernel_exprs


def prepare_temporaries(expression, temps=None):
    """This function generates a mapping of the form:

       ``temporaries = {terminal_node: symbol_name}``

    where `terminal_node` objects are :class:`slate.Tensor` nodes, and `symbol_name` are
    :class:`coffee.base.Symbol` objects.

    This mapping is used in the :class:`SlateKernelBuilder` to provide direct access to all
    temporaries associated with a particular slate expression.

    :arg expression: a :class:`slate.TensorBase` object.
    :arg temps: a dictionary that becomes populated recursively and is later returned as the
                temporaries map. This argument is initialized as an empty `dict` before recursion
                starts.
    """
    if temps is None:
        temporaries = {}
    else:
        temporaries = temps

    if isinstance(expression, Tensor):
        if expression not in temporaries.keys():
            temporaries[expression] = ast.Symbol("T%d" % len(temporaries))

    elif isinstance(expression, TensorAction):
        prepare_temporaries(expression.tensor, temps=temporaries)

    elif isinstance(expression, (UnaryOp, BinaryOp)):
        map(lambda expr: prepare_temporaries(expr, temps=temporaries), expression.operands)

    else:
        raise NotImplementedError("Expression of type %s not currently supported." % type(expression))

    return temporaries


def exprs_from_assembled_data(expression, aux_data=None):
    """This provides a mapping from a particular expression needing assembled data to a tuple
    containing a SLATE object and assembled data. That is, a mapping of the form:

       ``data = {expression: (slate_obj, data)}``

    For example, if the `expression` is the action of a SLATE `TensorBase` object on a `ufl.Coefficient`,
    then `data` will have the form:

       ``data = {TensorAction(Tensor, Coefficient): (Tensor, Coefficient)}``
    """
    if aux_data is None:
        data = {}
    else:
        data = aux_data

    if isinstance(expression, Tensor):
        pass

    elif isinstance(expression, (UnaryOp, BinaryOp)):
        map(lambda expr: exprs_from_assembled_data(expr, aux_data=data), expression.operands)

    elif isinstance(expression, TensorAction):
        acting_coefficient = expression._acting_coefficient
        assert isinstance(acting_coefficient, Coefficient)
        data[expression] = (expression.tensor, acting_coefficient)

    else:
        raise NotImplementedError("Expression of type %s not currently supported." % type(expression))

    return data
