from abc import ABCMeta, abstractmethod
from functools import partial
import types
from ufl.core.external_operator import ExternalOperator, find_initial_external_operator
from ufl.core.expr import Expr
from ufl.utils.str import as_native_str
from firedrake.function import Function
from firedrake import utils
from pyop2.datatypes import ScalarType
from ufl.log import error
import sympy as sp
from scipy import optimize


class AbstractPointwiseOperator(Function, ExternalOperator, metaclass=ABCMeta):

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data=None):
        Function.__init__(self, function_space, val, name, dtype)
        ExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count)

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

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, count=None, name=None, operator_data=None):
        "Return a new object of the same type with new operands."
        deriv_multiindex = derivatives or self.derivatives
        return type(self)(*operands, function_space=function_space or self._ufl_function_space,
                          derivatives=derivatives or self.derivatives,
                          count=count or self._count,
                          name=name or self.name(),
                          operator_data=operator_data or self.operator_data)
        """
        # We look up in the existing external operators and their derivatives
        key_e = find_initial_external_operator(self)
        if deriv_multiindex in type(self)._ufl_all_external_operators_[key_e].keys():
            corresponding_count = type(self)._ufl_all_external_operators_[key_e][deriv_multiindex]._count
            return type(self)(*operands, function_space=function_space or self._ufl_function_space,
                          derivatives=deriv_multiindex,
                          count=corresponding_count,
                          name=name or self.name(),
                          operator_data=operator_data or self.operator_data)
        else:
            reconstruct_pointop = type(self)(*operands, function_space=function_space or self._ufl_function_space,
                          derivatives=deriv_multiindex,
                          name=name or self.name(),
                          operator_data=operator_data or self.operator_data)
            del type(self)._ufl_all_external_operators_[reconstruct_pointop]
            type(self)._ufl_all_external_operators_[key_e][deriv_multiindex] = reconstruct_pointop
            return reconstruct_pointop
        """
    def __str__(self):
        "Default repr string construction for PointwiseOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (type(self).__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self._ufl_function_space), repr(self.derivatives), repr(self.ufl_shape), repr(self.operator_data))
        return as_native_str(r)


class PointexprOperator(AbstractPointwiseOperator):
    r"""A :class:`PointexprOperator` ... TODO :
     """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data):
        AbstractPointwiseOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, operator_data=operator_data)

        # Check
        if not isinstance(operator_data, types.FunctionType):
            error("Expecting a FunctionType pointwise expression")
        expr_shape = operator_data(*operands).ufl_shape
        if expr_shape != function_space.ufl_element().value_shape():
            error("The dimension does not match with the dimension of the function space %s" % function_space)

    @property
    def expr(self):
        return self.operator_data

    def compute_derivatives(self):
        symb = sp.symbols('s:%d' % len(self.ufl_operands))
        r = sp.diff(self.expr(*symb), *zip(symb, self.derivatives))
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

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data):
        AbstractPointwiseOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, operator_data=operator_data)

        # Check
        if not isinstance(operator_data['point_solve'], types.FunctionType):
            error("Expecting a FunctionType pointwise expression")

        if operator_data['point_solve'].__code__.co_argcount != len(operands) + 1:
            error("Expecting %s operands" % (operator_data['point_solve'].__code__.co_argcount-1))
        if operator_data['solver'] not in ('newton', 'secant', 'halley'):
            error("Expecting of the following method : %s" % ('newton', 'secant', 'halley'))
        if not isinstance(operator_data['solver_params'], dict):
            error("Expecting a dict with the solver arguments instead of %s" % operator_data['solver_params'])

    @property
    def operator_f(self):
        return self.operator_data['point_solve']

    @property
    def solver(self):
        return self.operator_data['solver']

    @property
    def solver_params(self):
        return self.operator_data['solver_params']

    def compute_derivatives(self, f):
        deriv_index = (0,) + self.derivatives
        symb = sp.symbols('s:%d' % len(deriv_index))
        if deriv_index == (0,)*len(deriv_index):
            return f

        for i, di in enumerate(deriv_index):
            if di != 0:
                res = sp.idiff(self.operator_f(*symb), symb[0], symb[i], di).simplify()
                if not all(v == 0 for v in deriv_index[:i]+deriv_index[i+1:]):
                    error("Cross-derivatives not handled : %s" % deriv_index)  # Needed feature ?
                break
        df = sp.lambdify(symb, res)
        expr = df(f, *self.ufl_operands)
        return self.interpolate(expr)

    def evaluate(self):
        space = self.ufl_function_space()
        solver_params = self.solver_params
        f = self.operator_f

        # Pre-processing to get the values of the initial guesses
        if 'x0' in solver_params.keys() and isinstance(solver_params['x0'], Expr):
            solver_params['x0'] = Function(space).interpolate(solver_params['x0']).dat.data
        if 'x1' in solver_params.keys() and isinstance(solver_params['x1'], Expr):
            solver_params['x1'] = Function(space).interpolate(solver_params['x1']).dat.data

        # Vectorized Newton
        args = tuple(Function(space).interpolate(pi) for pi in self.ufl_operands)
        vals = tuple(coeff.dat.data for coeff in args)

        if self.solver in ('newton', 'halley') and 'fprime' not in solver_params.keys():
            symb = sp.symbols('s:%d' % f.__code__.co_argcount)
            fprime = sp.diff(f(*symb), symb[0])
            gprime = sp.lambdify(symb, fprime)
            solver_params['fprime'] = partial(gprime, **dict(zip(gprime.__code__.co_varnames[1:], vals)))

        if self.solver == 'halley' and 'fprime2' not in solver_params.keys():
            symb = sp.symbols('s:%d' % f.__code__.co_argcount)
            fprime2 = sp.diff(f(*symb), symb[0], symb[0])
            gprime2 = sp.lambdify(symb, fprime2)
            solver_params['fprime2'] = partial(gprime2, **dict(zip(gprime2.__code__.co_varnames[1:], vals)))

        g = partial(f, **dict(zip(f.__code__.co_varnames[1:], vals)))

        if 'x0' not in solver_params.keys():
            self.dat.data[:] = optimize.newton(g, **solver_params, x0=self.dat.data)
        else:
            self.dat.data[:] = optimize.newton(g, **solver_params)
        return self.compute_derivatives(self)


class PointnetOperator(AbstractPointwiseOperator):
    r"""A :class:`PointnetOperator` ... TODO :
     """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, framework, model, ncontrols=None):
        AbstractPointwiseOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype)

        # Checks
        if not isinstance(ncontrols, int) or ncontrols > len(self.ufl_operands):
            error("Expecting for the number of controls an int type smaller or equal than the number of operands and not %s" % ncontrols)

        self.framework = framework
        self.model = model
        self.ncontrols = ncontrols
        self._controls = tuple(range(0,self.ncontrols))

    @property
    def controls(self):
        return dict(zip(self._controls, tuple(self.ufl_operands[i] for i in self._controls)))

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, count=None, name=None, model=None, ncontrols=None):
        "Return a new object of the same type with new operands."
        return type(self)(*operands, function_space=function_space or self._ufl_function_space,
                          derivatives=derivatives or self.derivatives,
                          count=count or self._count,
                          name=name or self.name(),
                          framework=self.framework,
                          model=model or self.model,
                          ncontrols=ncontrols or self.ncontrols)

    def __str__(self):
        "Default repr string construction for PointwiseOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (type(self).__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self._ufl_function_space), repr(self.derivatives), repr(self.ufl_shape), repr(self.framework))
        return as_native_str(r)

    # def compute_grad_inputs(self):
    #    "Compute the gradient of the neural net output with respect to the inputs."
    #    raise NotImplementedError(self.__class__.compute_grad_inputs)


class PytorchOperator(PointnetOperator):
    r"""A :class:`PyTorchOperator` ... TODO :
     """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, framework, model, ncontrols=None):
        PointnetOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, framework=framework, model=model, ncontrols=ncontrols)

        # Check
        #try:
        #    import torch
        #except ImportError:
        #    raise ImportError("Error when trying to import PyTorch")

    def compute_derivatives(self):
        """Compute the gradient of the network wrt inputs"""
        op = self.interpolate(self.ufl_operands[0])
        torch_op = torch.from_numpy(op.dat.data).type(torch.FloatTensor)
        model_output = self.evaluate().data.data
        res = []
        for i, e in enumerate(torch_op):
            xi = torch.unsqueeze(e, 0)
            yi = model_output[i]
            res.append(torch.autograd.grad(yi, xi)[0])
        return res

    def evaluate(self):
        import torch
        model = self.model.eval()
        op = self.interpolate(self.ufl_operands[0])
        torch_op = torch.from_numpy(op.dat.data).type(torch.FloatTensor)
        model_input = torch.unsqueeze(torch_op, 1)
        result = Function(self.ufl_function_space())
        result.dat.data[:] = model(model_input).detach().numpy().squeeze(1)
        return self.assign(result)
        # for i, e in enumerate(torch_op):
        #     model_input = torch.unsqueeze(e, 0)
        #     result.dat.data[i] = model(model_input).detach().numpy()
        # return self.assign(result)


class TensorFlowOperator(PointnetOperator):
    r"""A :class:`TensorFlowOperator` ... TODO :
     """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, framework, model, ncontrols=None):
        PointnetOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, framework=framework, model=model, ncontrols=ncontrols)

        # Check
        #try:
        #    import tensorflow
        #except ImportError:
        #    raise ImportError("Error when trying to import TensorFlow")


class KerasOperator(PointnetOperator):
    r"""A :class:`KerasOperator` ... TODO :
     """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, framework, model, ncontrols=None):
        PointnetOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, framework=framework, model=model, ncontrols=ncontrols)

        # Check
        #try:
        #    from tensorflow import keras
        #except ImportError:
        #    raise ImportError("Error when trying to import tensorflow.keras")


def point_expr(point_expr, function_space):
    return partial(PointexprOperator, operator_data=point_expr, function_space=function_space)


def point_solve(point_solve, function_space, solver='newton', solver_params=None):
    if solver_params is None:
        solver_params = {}
    operator_data = {'point_solve': point_solve, 'solver': solver, 'solver_params': solver_params}
    return partial(PointsolveOperator, operator_data=operator_data, function_space=function_space)


def neuralnet(model, function_space, ncontrols=1):

    torch_module = type(None); tensorflow_module = type(None); keras_module = type(None);

    # Checks
    try:
        import torch
        torch_module = torch.nn.modules.module.Module
    except ImportError:
        pass

    try:
        import tensorflow
        #tensorflow_module =
    except ImportError:
        pass

    try:
        from tensorflow import keras
        #keras_module =
    except ImportError:
        pass

    if isinstance(model, torch_module):
        return partial(PytorchOperator, function_space=function_space, framework='PyTorch', model=model, ncontrols=ncontrols)
    elif isinstance(model, tensorflow_module):
        return partial(TensorFlowOperator, function_space=function_space, framework='TensorFlow', model=model, ncontrols=ncontrols)
    elif isinstance(model, keras_module):
        return partial(KerasOperator, function_space=function_space, framework='Keras', model=model, ncontrols=ncontrols)
    else:
        error("Expecting one of the following library : PyTorch, TensorFlow or Keras and that the library has been installed")
