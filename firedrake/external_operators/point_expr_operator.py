from functools import partial
import types

from ufl.constantvalue import as_ufl

import firedrake.ufl_expr as ufl_expr
from firedrake.assemble import assemble
from firedrake.interpolation import Interpolate
from firedrake.external_operators import AbstractExternalOperator, assemble_method


class PointexprOperator(AbstractExternalOperator):

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=(), operator_data):
        """External operator representing UFL expressions.

        Parameters
        ----------
        *operands : ufl.core.expr.Expr or ufl.form.BaseForm
                    Operands of the external operator.
        function_space : firedrake.functionspaceimpl.WithGeometryBase
                         The function space the external operator is mapping to.
        derivatives : tuple
                      Tuple specifiying the derivative multiindex.
        *argument_slots : ufl.coefficient.BaseCoefficient or ufl.argument.BaseArgument
                          Tuple containing the arguments of the linear form associated with the external operator,
                          i.e. the arguments with respect to which the external operator is linear. Those arguments
                          can be ufl.Argument objects, as a result of differentiation, or ufl.Coefficient objects,
                          as a result of taking the action on a given function.
        operator_data : dict
                        Dictionary to stash external data specific to the :class:`~.PointexprOperator` class.
                        This dictionary must at least contain the following:
                        - 'func': The function implementing the pointwise expression.

        Notes
        -----
        The :class:`~.PointexprOperator` class mimics the :class:`~.Interpolate` class and is mostly design
        for debugging purposes.
        """
        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                          argument_slots=argument_slots, operator_data=operator_data)

        # Check
        if not isinstance(operator_data["func"], types.FunctionType):
            raise TypeError("Expecting a FunctionType pointwise expression")
        expr_shape = operator_data["func"](*operands).ufl_shape
        if expr_shape != function_space.ufl_element().value_shape:
            raise ValueError("The dimension does not match with the dimension of the function space %s" % function_space)

    @property
    def expr(self):
        return self.operator_data["func"]

    # -- Assembly methods -- #

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
    """Helper function for instantiating the :class:`~.PointexprOperator` class.

    This function facilitates having a two-stage instantiation which dissociates between class arguments
    that are fixed, such as the function space and the input function, and the operands of the operator,
    which may change, e.g. when the operator is used in a time-loop.

    Example
    -------

    .. code-block:: python

        # Stage 1: Partially initialise the operator.
        N = point_expr(lambda x, y: x - y, function_space=V)
        # Stage 2: Define the operands and use the operator in a UFL expression.
        F = (inner(grad(u), grad(v)) + inner(N(u, f), v)) * dx

    Parameters
    ----------
    point_expr: collections.abc.Callable
                A function expression (e.g. lambda expression)
    function_space: firedrake.functionspaceimpl.WithGeometryBase
                    The function space into which the input expression needs to be interpolated.

    Returns
    -------
    collections.abc.Callable
        The partially initialised :class:`~.PointexprOperator` class.
    """
    return partial(PointexprOperator, operator_data={"func": point_expr}, function_space=function_space)
