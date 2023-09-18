from functools import partial
import types

from ufl.constantvalue import as_ufl

import firedrake.ufl_expr as ufl_expr
from firedrake.assemble import assemble
from firedrake.interpolation import Interpolate
from firedrake.external_operators import AbstractExternalOperator, assemble_method


class PointexprOperator(AbstractExternalOperator):
    r"""A :class:`PointexprOperator` is an implementation of ExternalOperator that is defined through
    a given function f (e.g. a lambda expression) and whose values are defined through the mere evaluation
    of f pointwise.
    """

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=(), operator_data):
        r"""
        :param operands: operands on which act the :class:`PointexrOperator`.
        :param function_space: the :class:`.FunctionSpace`,
        or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        Alternatively, another :class:`Function` may be passed here and its function space
        will be used to build this :class:`Function`.  In this case, the function values are copied.
        :param derivatives: tuple specifiying the derivative multiindex.
        :param operator_data: dictionary containing the function defining how to evaluate the :class:`PointexprOperator`.
        """

        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                          argument_slots=argument_slots, operator_data=operator_data)

        # Check
        if not isinstance(operator_data, types.FunctionType):
            raise TypeError("Expecting a FunctionType pointwise expression")
        expr_shape = operator_data(*operands).ufl_shape
        if expr_shape != function_space.ufl_element().value_shape():
            raise ValueError("The dimension does not match with the dimension of the function space %s" % function_space)

    @property
    def expr(self):
        return self.operator_data

    # --- Evaluation ---

    @assemble_method(0, (0,))
    def assemble_operator(self, *args, **kwargs):
        V = self.function_space()
        expr = as_ufl(self.expr(*self.ufl_operands))
        if len(V) < 2:
            interp = Interpolate(expr, self.function_space())
            return assemble(interp)
        # Interpolation of UFL expressions for mixed functions is not yet supported
        # -> `Function.assign` might be enough in some cases.
        try:
            from firedrake.function import Function
            return Function(V).assign(expr)
        except NotImplementedError:
            raise NotImplementedError("Interpolation of UFL expressions for mixed functions is not yet supported")

    @assemble_method(1, (0, None))
    def assemble_Jacobian_action(self, *args, **kwargs):
        V = self.function_space()
        expr = as_ufl(self.expr(*self.ufl_operands))
        interp = Interpolate(expr, V)

        u, = [e for i, e in enumerate(self.ufl_operands) if self.derivatives[i] == 1]
        w = self.argument_slots()[-1]
        dinterp = ufl_expr.derivative(interp, u)
        return assemble(ufl_expr.action(dinterp, w))

    @assemble_method(1, (0, 1))
    def assemble_Jacobian(self, *args, assembly_opts, **kwargs):
        V = self.function_space()
        expr = as_ufl(self.expr(*self.ufl_operands))
        interp = Interpolate(expr, V)

        u, = [e for i, e in enumerate(self.ufl_operands) if self.derivatives[i] == 1]
        jac = ufl_expr.derivative(interp, u)
        return assemble(jac)

    @assemble_method(1, (1, 0))
    def assemble_Jacobian_adjoint(self, *args, assembly_opts, **kwargs):
        J = self.assemble_Jacobian(*args, assembly_opts=assembly_opts, **kwargs)
        J.petscmat.hermitianTranspose()
        return J

    @assemble_method(1, (None, 0))
    def assemble_Jacobian_adjoint_action(self, *args, **kwargs):
        V = self.function_space()
        expr = as_ufl(self.expr(*self.ufl_operands))
        interp = Interpolate(expr, V)

        u, = [e for i, e in enumerate(self.ufl_operands) if self.derivatives[i] == 1]
        ustar = self.argument_slots()[0]
        jac = ufl_expr.derivative(interp, u)
        return assemble(ufl_expr.action(ufl_expr.adjoint(jac), ustar))


# Helper function #
def point_expr(point_expr, function_space):
    r"""The point_expr function returns the `PointexprOperator` class initialised with :
        - point_expr : a function expression (e.g. lambda expression)
        - function space
     """
    return partial(PointexprOperator, operator_data=point_expr, function_space=function_space)
