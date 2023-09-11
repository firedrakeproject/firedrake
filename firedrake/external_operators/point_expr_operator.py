from functools import partial
import types
import sympy as sp

# from ufl.algorithms.apply_derivatives import VariableRuleset
from ufl.constantvalue import as_ufl

from firedrake.external_operators import AbstractExternalOperator, assemble_method

from pyop2.datatypes import ScalarType


class PointexprOperator(AbstractExternalOperator):
    r"""A :class:`PointexprOperator` is an implementation of ExternalOperator that is defined through
    a given function f (e.g. a lambda expression) and whose values are defined through the mere evaluation
    of f pointwise.
    """

    def __init__(self, *operands, function_space, derivatives=None, result_coefficient=None, argument_slots=(),
                 val=None, name=None, dtype=ScalarType, operator_data):
        r"""
        :param operands: operands on which act the :class:`PointexrOperator`.
        :param function_space: the :class:`.FunctionSpace`,
        or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        Alternatively, another :class:`Function` may be passed here and its function space
        will be used to build this :class:`Function`.  In this case, the function values are copied.
        :param derivatives: tuple specifiying the derivative multiindex.
        :param val: NumPy array-like (or :class:`pyop2.Dat`) providing initial values (optional).
            If val is an existing :class:`Function`, then the data will be shared.
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to ``ScalarType``).
        :param operator_data: dictionary containing the function defining how to evaluate the :class:`PointexprOperator`.
        """

        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                          result_coefficient=result_coefficient, argument_slots=argument_slots,
                                          val=val, name=name, dtype=dtype,
                                          operator_data=operator_data)

        # Check
        if not isinstance(operator_data, types.FunctionType):
            raise TypeError("Expecting a FunctionType pointwise expression")
        expr_shape = operator_data(*operands).ufl_shape
        if expr_shape != function_space.ufl_element().value_shape():
            raise ValueError("The dimension does not match with the dimension of the function space %s" % function_space)

    @property
    def expr(self):
        return self.operator_data

    # --- Symbolic computations ---

    def _symbolic_differentiation(self):
        symb = sp.symbols('s:%d' % len(self.ufl_operands))
        r = sp.diff(self.expr(*symb), *zip(symb, self.derivatives))
        return sp.lambdify(symb, r, dummify=True)

    # --- Evaluation ---

    @assemble_method(0, (0,))
    def assemble_operator(self, *args, **kwargs):
        return self._evaluate(*args, **kwargs)

    @assemble_method(1, (0, None))
    def assemble_Jacobian_action(self, *args, **kwargs):
        from firedrake.function import Function
        res = Function(self.function_space())
        w = self.argument_slots()[-1]
        # Get diagonal of the Jacobian
        dNdu = self._evaluate()
        # Multiply pointwise dNdu and w since the Jacobian is diagonal
        with res.dat.vec_wo as res_vec:
            with w.dat.vec_ro as u_vec:
                with dNdu.dat.vec_ro as v_vec:
                    res_vec.pointwiseMult(u_vec, v_vec)
        return res

    @assemble_method(1, (0, 1))
    def assemble_Jacobian(self, *args, assembly_opts, **kwargs):
        result = self._evaluate()

        # Construct the Jacobian matrix
        integral_types = set(['cell'])
        J = self._matrix_builder((), assembly_opts, integral_types)
        with result.dat.vec as vec:
            J.petscmat.setDiagonal(vec)
        return J

    @assemble_method(1, (1, 0))
    def assemble_Jacobian_adjoint(self, *args, assembly_opts, **kwargs):
        J = self.assemble_Jacobian(*args, assembly_opts=assembly_opts, **kwargs)
        J.petscmat.hermitianTranspose()
        return J

    @assemble_method(1, (None, 0))
    def assemble_Jacobian_adjoint_action(self, *args, **kwargs):
        from firedrake.cofunction import Cofunction
        res = Cofunction(self.function_space().dual())
        ustar = self.argument_slots()[0]
        # Get diagonal of the Jacobian
        dNdu = self._evaluate()
        # Multiply pointwise dNdu and ustar since the Jacobian is diagonal
        with res.dat.vec_wo as res_vec:
            with ustar.dat.vec_ro as u_vec:
                with dNdu.dat.vec_ro as v_vec:
                    res_vec.pointwiseMult(u_vec, v_vec)
        return res

    def _evaluate(self, *args, **kwargs):
        operands = self.ufl_operands
        operator = self._symbolic_differentiation()
        expr = as_ufl(operator(*operands))
        if expr.ufl_shape == () and expr != 0:
            return self.interpolate(expr)
            # var = VariableRuleset(self.ufl_operands[0])
            # expr = expr*var._Id
        elif expr == 0:
            return self.assign(expr)
        # TODO: Clean that once Interp branch got merged to this branch
        return self.assign(expr)  # self.interpolate(expr)


# Helper function #
def point_expr(point_expr, function_space):
    r"""The point_expr function returns the `PointexprOperator` class initialised with :
        - point_expr : a function expression (e.g. lambda expression)
        - function space
     """
    return partial(PointexprOperator, operator_data=point_expr, function_space=function_space)
