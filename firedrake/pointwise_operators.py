from abc import ABCMeta, abstractmethod
from functools import partial
import types
from ufl.core.external_operator import ExternalOperator
from ufl.utils.str import as_native_str
from firedrake.function import Function
from firedrake import utils
from pyop2.datatypes import ScalarType
from ufl.log import error
import sympy as sp


class AbstractPointwiseOperator(Function, ExternalOperator, metaclass=ABCMeta):

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, operator_data=None):
        Function.__init__(self, eval_space, val, name, dtype)
        ExternalOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count)

        self.operator_data = operator_data

    @abstractmethod
    def compute_derivatives(self):
        """apply the derivatives on operator_data"""

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
    r"""A :class:`PointexprOperator` ... TODO :
     """

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, point_expr):
        AbstractPointwiseOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, operator_data=point_expr)

        # Check
        if not isinstance(point_expr, types.FunctionType):
            error("Expecting a FunctionType pointwise expression")

    def compute_derivatives(self):
        symb = sp.symbols('s:%d' % len(self.ufl_operands))
        r = sp.diff(self.operator_data(*symb), *zip(symb, self.derivatives))
        return sp.lambdify(symb, r)

    def evaluate(self):
        operands = self.ufl_operands
        operator = self.compute_derivatives()
        expr = operator(*operands)
        return self.interpolate(expr)


class PointsolveOperator(AbstractPointwiseOperator):
    r"""A :class:`PointsolveOperator` ... TODO :

        scipy syntax : newton_params = {'fprime':None, 'args':(), 'tol':1.48e-08, 'maxiter':50, 'fprime2':None, 'x1':None,
                             'rtol':0.0, 'full_output':False, 'disp':True}
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
     """

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, point_solve, params, solver):
        AbstractPointwiseOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, operator_data=point_solve)

        # Check
        if not isinstance(point_solve, types.FunctionType):
            error("Expecting a FunctionType pointwise expression")

        if not (isinstance(params, tuple) or params <= point_solve.__code__.co_varnames):
            error("Expecting a tuple subset of %s instead of : %s" % (point_solve.__code__.co_varnames, params))
        if len(params) != len(operands):
            error("Expecting %s operands" % len(params))
        if point_solve.__code__.co_argcount - len(operands) > 1:
            error("Multidimensional Newton : Not yet impelemented")  # Needed feature ?

        self.params = params

        if not isinstance(solver, dict):
            error("Expecting a dict with the solver arguments instead of %s" % solver)
        elif 'x0' not in solver.keys():
            error("Expecting an initial condition x0")

        self.solver = solver

    def compute_derivatives(self, f):
        deriv_index = self.derivatives
        varnames = self.operator_data.__code__.co_varnames
        if deriv_index == (0,)*len(deriv_index):
            return f

        var = tuple(set(varnames)-set(self.params))
        symb = sp.symbols(' '.join(chr(i) for i in range(97, 97+len(varnames))))
        r = self.params+var
        symb2 = tuple(symb[r.index(i)] for i in varnames)

        for i, di in enumerate(deriv_index):
            if di != 0:
                res = sp.idiff(self.operator_data(*symb2), symb[-1], symb[i], di).simplify()
                if not all(v == 0 for v in deriv_index[:i]+deriv_index[i+1:]):
                    error("Cross-derivatives not handled : %s" % deriv_index)  # Needed feature ?

        df = sp.lambdify(symb, res)
        args = self.ufl_operands + (f,)
        expr = df(*args)
        return self.interpolate(expr)

    def evaluate(self):
        try:
            from scipy import optimize
        except ImportError:
            raise ImportError("Error when trying to import Scipy.optimize")

        space = self.ufl_function_space()
        solver = self.solver

        # Pre-processing to get the values of the initial guesses
        if isinstance(solver['x0'], Function):
            solver['x0'] = Function(space).interpolate(solver['x0']).dat.data
        if 'x1' in solver.keys() and isinstance(solver['x1'], Function):
            solver['x1'] = Function(space).interpolate(solver['x1']).dat.data

        # Vectorized Newton
        args = (Function(space).interpolate(pi) for pi in self.ufl_operands)
        vals = tuple(coeff.dat.data for coeff in args)
        g = partial(self.operator_data, **dict(zip(self.params, vals)))

        result = Function(space)
        result.dat.data[:] = optimize.newton(g, **self.solver)
        result = self.compute_derivatives(result)

        return self.assign(result)


class PointnetOperator(AbstractPointwiseOperator):
    r"""A :class:`PointnetOperator` ... TODO :
     """

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, framework, model):
        AbstractPointwiseOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype)

        self.framework = framework
        self.model = model


class PytorchOperator(PointnetOperator):
    r"""A :class:`PyTorchOperator` ... TODO :
     """

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, framework, model):
        PointnetOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, framework=framework, model=model)

        # Check
        try:
            import torch
        except ImportError:
            raise ImportError("Error when trying to import PyTorch")

        
class TensorFlowOperator(PointnetOperator):
    r"""A :class:`TensorFlowOperator` ... TODO :
     """

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, framework, model):
        PointnetOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, framework=framework, model=model)

        # Check
        try:
            import tensorflow
        except ImportError:
            raise ImportError("Error when trying to import TensorFlow")


class KerasOperator(PointnetOperator):
    r"""A :class:`KerasOperator` ... TODO :
     """

    def __init__(self, *operands, eval_space, derivatives=None, shape=None, count=None, val=None, name=None, dtype=ScalarType, framework, model):
        PointnetOperator.__init__(self, *operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, framework=framework, model=model)

        # Check
        try:
            from tensorflow import keras
        except ImportError:
            raise ImportError("Error when trying to import tensorflow.keras")


def PointExprOp(*operands, eval_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, point_expr):
    expr_shape = point_expr(*operands).ufl_shape
    return PointexprOperator(*operands, eval_space=eval_space, derivatives=derivatives, shape=expr_shape, count=count, val=val, name=name, dtype=dtype, point_expr=point_expr)


def point_expr(point_expr):
    return partial(PointExprOp, point_expr=point_expr)


def PointSolveOp(*operands, eval_space, derivatives=None, count=None, shape=(), val=None, name=None, dtype=ScalarType, point_solve, params, solver):
    return PointsolveOperator(*operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, point_solve=point_solve, params=params, solver=solver)


def point_solve(point_solve, solver, params=()):
    return partial(PointSolveOp, point_solve=point_solve, params=params, solver=solver)


def PointPyTorchOp(*operands, eval_space, derivatives=None, count=None, shape=(), val=None, name=None, dtype=ScalarType, framework, model):
    return PytorchOperator(*operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, framework=framework, model=model)


def PointTensorFlowOp(*operands, eval_space, derivatives=None, count=None, shape=(), val=None, name=None, dtype=ScalarType, framework, model):
    return TensorFlowOperator(*operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, framework=framework, model=model)


def PointKerasOp(*operands, eval_space, derivatives=None, count=None, shape=(), val=None, name=None, dtype=ScalarType, framework, model):
    return KerasOperator(*operands, eval_space=eval_space, derivatives=derivatives, shape=shape, count=count, val=val, name=name, dtype=dtype, framework=framework, model=model)


def neuralnet(framework):
    if framework == 'PyTorch':
        return partial(PointPyTorchOp, framework=framework)
    elif framework == 'TensorFlow':
        return partial(PointTensorFlowOp, framework=framework)
    elif framework == 'Keras':
        return partial(PointKerasOp, framework=framework)
    else:
        error("Expecting one of the following library : PyTorch, TensorFlow or Keras and not %s" % framework)
