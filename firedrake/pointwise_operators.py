from abc import ABCMeta, abstractmethod
from functools import partial
import types
from ufl.core.external_operator import ExternalOperator
from ufl.utils.str import as_native_str
from firedrake.function import Function
from firedrake import utils
from pyop2.datatypes import ScalarType
from ufl.log import error


class AbstractPointwiseOperator(Function, ExternalOperator, metaclass=ABCMeta):

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, operator_data=None):
        Function.__init__(self, eval_space, val, name, dtype)
        ExternalOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count)

        self.operator_data = operator_data

    @abstractmethod
    def evaluate(self):
        """define the evaluation method for the ExternalOperator object"""

    @utils.cached_property
    def _split(self):
        return tuple(Function(V, val) for (V, val) in zip(self.function_space(), self.topological.split()))

    def _ufl_expr_reconstruct_(self, *operands, eval_space=None, derivatives=None, shape=None, operator_data=None):
        "Return a new object of the same type with new operands."
        return type(self)(*operands, eval_space=eval_space or self.eval_space, derivatives=derivatives, shape=shape, operator_data=operator_data)

    def __str__(self):
        "Default repr string construction for PointwiseOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (type(self).__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self._ufl_function_space), repr(self.derivatives), repr(self.ufl_shape), repr(self.operator_data))
        return as_native_str(r)


class PointexprOperator(AbstractPointwiseOperator):

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, point_expr):
        AbstractPointwiseOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, operator_data=point_expr)

        # Check
        if not isinstance(point_expr, types.FunctionType):
            error("Expecting a FunctionType pointwise expression")

    def evaluate(self):
        operands = self.ufl_operands
        expr = self.operator_data(*operands)
        return self.interpolate(expr)


class PointsolveOperator(AbstractPointwiseOperator):

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, point_expr):
        AbstractPointwiseOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, operator_data=point_expr)

        # Check
        if not isinstance(point_expr, types.FunctionType):
            error("Expecting a FunctionType pointwise expression")

    def evaluate(self):
        # TODO: Use Newton-type solver
        raise NotImplementedError("TODO !")


def PointExprOp(*operands, eval_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, point_expr):
    expr_shape = point_expr(*operands).ufl_shape
    return PointexprOperator(*operands, eval_space=eval_space, derivatives=derivatives, shape=expr_shape, count=count, val=val, name=name, dtype=dtype, point_expr=point_expr)


def point_expr(point_expr):
    return partial(PointExprOp, point_expr=point_expr)
