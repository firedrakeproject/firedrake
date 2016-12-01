from __future__ import absolute_import, print_function, division


from coffee import base as ast

from firedrake.slate.slate import TensorBase, Tensor, UnaryOp, BinaryOp, TensorAction
from firedrake.slate.utils import RemoveRestrictions, Transformer
from firedrake.tsfc_interface import compile_form

from functools import partial

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.form import Form


class SlateKernelBuilder(object):
    """
    """
    def __init__(self, expression, tsfc_parameters=None):
        assert isinstance(expression, TensorBase)
        self.expression = expression
        self.oriented = False
        self.needs_cell_facets = False

        # Initialize temporaries and tsfc kernels
        self.temps = prepare_temporaries(expression)
        self.kernel_exprs = prepare_tsfc_kernels(self.temps, tsfc_parameters=tsfc_parameters)

    def coefficient_map(self):
        """ """
        return dict((c, ast.Symbol("w%d" % i))
                    for i, c in enumerate(self.expression.coefficients()))

    def require_cell_facets(self):
        """ """
        self.needs_cell_facets = True

    def construct_ast(self, name, args, statements):
        """
        """
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
    """
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
    """
    """
    if temps is None:
        temporaries = {}
    else:
        temporaries = temps

    if isinstance(expression, Tensor):
        if expression not in temporaries.keys():
            temporaries[expression] = ast.Symbol("T%d" % len(temporaries))

    elif isinstance(expression, TensorAction):
        if expression not in temporaries.keys():
            prepare_temporaries(expression.tensor, temps=temporaries)
            temporaries[expression] = ast.Symbol("T%d" % len(temps))

    elif isinstance(expression, (UnaryOp, BinaryOp)):
        map(lambda expr: prepare_temporaries(expr, temps=temporaries), expression.operands)

    else:
        raise NotImplementedError("Expression of type %s not currently supported." % type(expression))

    return temporaries
