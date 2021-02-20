from abc import ABCMeta
from functools import partial, reduce
from operator import mul
from hashlib import md5
import types
import sympy as sp
import numpy as np

from ufl import zero, conj
from ufl.core.external_operator import ExternalOperator
from ufl.core.expr import Expr
from ufl.coefficient import Coefficient
from ufl.algorithms.apply_derivatives import VariableRuleset
from ufl.constantvalue import as_ufl
from ufl.referencevalue import ReferenceValue
from ufl.operators import transpose, inner
from ufl.log import error

import firedrake.assemble
from firedrake.function import Function
from firedrake.constant import Constant
from firedrake import utils, functionspaceimpl
from firedrake.adjoint import PointwiseOperatorsMixin
from pyop2.datatypes import ScalarType


class AbstractExternalOperator(ExternalOperator, PointwiseOperatorsMixin, metaclass=ABCMeta):
    r"""Abstract base class from which stem all the Firedrake practical implementations of the
    ExternalOperator, i.e. all the ExternalOperator subclasses that have mechanisms to be
    evaluated pointwise and to provide their own derivatives.
    This class inherits from firedrake.function.Function and ufl.core.external_operator.ExternalOperator
    Every subclass based on this class must provide the `_compute_derivatives` and '_evaluate' or `_evaluate_action` methods.
    """

    def __init__(self, *operands, function_space, derivatives=None, val=None, name=None, dtype=ScalarType, operator_data=None, coefficient=None, arguments=(), local_operands=()):
        ExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, arguments=arguments, local_operands=local_operands)
        fspace = self.ufl_function_space()
        if not isinstance(fspace, functionspaceimpl.WithGeometry):
            fspace = functionspaceimpl.FunctionSpace(function_space.mesh().topology, fspace.ufl_element())
            fspace = functionspaceimpl.WithGeometry(fspace, function_space.mesh())

        # Check arguments and action_coefficients
        # self._check_arguments_action_coefficients()

        if coefficient is None:
            coefficient = Function(fspace, val, name, dtype)
            self._val = coefficient.topological
        elif not isinstance(coefficient, (Coefficient, ReferenceValue)):
            raise TypeError('Expecting a Coefficient and not %s', type(coefficient))
        self._coefficient = coefficient
        self._val = val
        self._name = name

        self.operator_data = operator_data

    def _check_arguments_action_coefficients(self):
        if not any(self.is_type_global):
            return
        n_args = len(self._arguments)
        n_action_coefficients = len(self._action_coefficients)
        if n_args + n_action_coefficients != sum(self.derivatives):
            raise ValueError('Expecting number of arguments (%s) + number of action arguments (%s) to be equal to the number of derivatives taken (%s)!' % (n_args, n_action_coefficients, sum(self.derivatives)))

    def name(self):
        return getattr(self.get_coefficient(), '_name', self._name)

    def function_space(self):
        return self.get_coefficient().function_space()

    def _make_function_space_args(self, k, y, adjoint=False):
        """Make the function space of the Gateaux derivative: dN[x] = \frac{dN}{dOperands[k]} * y(x) if adjoint is False
        and of \frac{dN}{dOperands[k]}^{*} * y(x) if adjoint is True"""
        ufl_function_space = ExternalOperator._make_function_space_args(self, k, y, adjoint=adjoint)
        mesh = self.function_space().mesh()
        function_space = functionspaceimpl.FunctionSpace(mesh.topology, ufl_function_space.ufl_element())
        return functionspaceimpl.WithGeometry(function_space, mesh)

    @property
    def dat(self):
        return self.get_coefficient().dat

    @property
    def topological(self):
        # When we replace coefficients in _build_coefficient_replace_map
        # we replace firedrake.Function by ufl.Coefficient and we lose track of val
        return getattr(self.get_coefficient(), 'topological', self._val)

    def assign(self, *args, **kwargs):
        assign = self.get_coefficient().assign(*args, **kwargs)
        # Keep track of the function's value
        self._val = assign.topological
        return assign

    def interpolate(self, *args, **kwargs):
        interpolate = self.get_coefficient().interpolate(*args, **kwargs)
        # Keep track of the function's value
        self._val = interpolate.topological
        return interpolate

    def split(self):
        return self.get_coefficient().split()

    @property
    def block_variable(self):
        return self.get_coefficient().block_variable

    @property
    def _ad_floating_active(self):
        self.get_coefficient()._ad_floating_active

    def _compute_derivatives(self):
        """apply the derivatives on operator_data"""

    def _evaluate(self):
        raise NotImplementedError('The %s class requires an _evaluate method' % type(self).__name__)

    def _evaluate_action(self, *args, **kwargs):
        raise NotImplementedError('The %s class requires an _evaluate_action method' % type(self).__name__)

    def _evaluate_adjoint_action(self, *args, **kwargs):
        raise NotImplementedError('The %s class should have an _evaluate_adjoint_action method when needed' % type(self).__name__)

    def evaluate(self, *args, **kwargs):
        """define the evaluation method for the ExternalOperator object"""
        action_coefficients = self.action_coefficients()
        if any(self.is_type_global):  # Check if at least one operand is global
            x = tuple(e for e, _ in action_coefficients)
            if len(x) != sum(self.derivatives):
                raise ValueError('Global external operators cannot be evaluated, you need to take the action!')
            is_adjoint = action_coefficients[-1][1] if len(action_coefficients) > 0 else False
            if is_adjoint:
                return self._evaluate_adjoint_action(x)
            # If there are local operands (i.e. if not all the operands are global)
            # `_evaluate_action` needs to take care of handling them correctly
            return self._evaluate_action(x, *args, **kwargs)
        return self._evaluate(*args, **kwargs)

    def copy(self, deepcopy=False):
        r"""Return a copy of this CoordinatelessFunction.

        :kwarg deepcopy: If ``True``, the new
            :class:`CoordinatelessFunction` will allocate new space
            and copy values.  If ``False``, the default, then the new
            :class:`CoordinatelessFunction` will share the dof values.
        """
        if deepcopy:
            val = type(self.dat)(self.dat)
        else:
            val = self.dat
        return type(self)(*self.ufl_operands, function_space=self.function_space(), val=val,
                          name=self.name(), dtype=self.dat.dtype,
                          derivatives=self.derivatives,
                          operator_data=self.operator_data)

    # Computing the action of the adjoint derivative may require different procedures
    # depending on wrt what is taken the derivative
    # E.g. the neural network case: where the adjoint derivative computation leads us
    # to compute the gradient wrt the inputs of the network or the weights.
    def evaluate_adj_component_control(self, x, idx):
        r"""Starting from the residual form: F(N(u, m), u(m), m) = 0
            This method computes the action of (dN/dm)^{*} on x where m is the control
        """
        derivatives = tuple(dj + int(idx == j) for j, dj in enumerate(self.derivatives))
        """
        # This chunk of code assume that GLOBAL refers to being global with respect to the control as well
        if self.is_type_global[idx]:
            function_space = self._make_function_space_args(idx, x, adjoint=True)
            dNdq = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives,
                                               function_space=function_space)
            result = dNdq._evaluate_adjoint_action(x)
            return result.vector()
        """
        dNdq = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives)
        dNdq = dNdq._evaluate()
        dNdq_adj = conj(transpose(dNdq))
        result = firedrake.assemble(dNdq_adj)
        if isinstance(self.ufl_operands[idx], Constant):
            return result.vector().inner(x)
        return result.vector() * x

    def evaluate_adj_component_state(self, x, idx):
        r"""Starting from the residual form: F(N(u, m), u(m), m) = 0
            This method computes the action of (dN/du)^{*} on x where u is the state
        """
        derivatives = tuple(dj + int(idx == j) for j, dj in enumerate(self.derivatives))
        if self.is_type_global[idx]:
            new_args = self.arguments() + ((x, True),)
            function_space = self._make_function_space_args(idx, x, adjoint=True)
            return self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives,
                                               function_space=function_space, arguments=new_args)
        dNdq = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives)
        dNdq_adj = conj(transpose(dNdq))
        return inner(dNdq_adj, x)

    @utils.cached_property
    def _split(self):
        return tuple(Function(V, val) for (V, val) in zip(self.function_space(), self.topological.split()))

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, name=None, operator_data=None, val=None, coefficient=None, arguments=None, local_operands=None, add_kwargs={}):
        "Return a new object of the same type with new operands."
        deriv_multiindex = derivatives or self.derivatives

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            corresponding_coefficient = None
            e_master = self._extop_master
            for ext in e_master.coefficient_dict.values():
                if ext.derivatives == deriv_multiindex:
                    return ext._ufl_expr_reconstruct_(*operands, function_space=function_space,
                                                      derivatives=deriv_multiindex,
                                                      name=name,
                                                      coefficient=coefficient,
                                                      arguments=arguments,
                                                      local_operands=local_operands,
                                                      operator_data=operator_data,
                                                      add_kwargs=add_kwargs)
        else:
            corresponding_coefficient = coefficient or self._coefficient

        reconstruct_op = type(self)(*operands, function_space=function_space or self._extop_master.ufl_function_space(),
                                    derivatives=deriv_multiindex,
                                    name=name or self.name(),
                                    coefficient=corresponding_coefficient,
                                    arguments=arguments or (self.arguments()+self.action_coefficients()),
                                    local_operands=self.local_operands,
                                    operator_data=operator_data or self.operator_data,
                                    **add_kwargs)

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            self._extop_master.coefficient_dict.update({deriv_multiindex: reconstruct_op})
            reconstruct_op._extop_master = self._extop_master
        elif deriv_multiindex != (0,)*len(operands):
            reconstruct_op._extop_master = self._extop_master
        return reconstruct_op

    def __str__(self):
        "Default repr string construction for PointwiseOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (type(self).__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self.ufl_function_space()), repr(self.derivatives), repr(self.ufl_shape), repr(self.operator_data))
        return r


class PointexprOperator(AbstractExternalOperator):
    r"""A :class:`PointexprOperator` is an implementation of ExternalOperator that is defined through
    a given function f (e.g. a lambda expression) and whose values are defined through the mere evaluation
    of f pointwise.
    """

    def __init__(self, *operands, function_space, derivatives=None, val=None, name=None, coefficient=None, arguments=(), dtype=ScalarType, operator_data):
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

        local_operands = operands
        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, val=val, name=name, coefficient=coefficient, arguments=arguments, local_operands=local_operands, dtype=dtype, operator_data=operator_data)

        # Check
        if not isinstance(operator_data, types.FunctionType):
            error("Expecting a FunctionType pointwise expression")
        expr_shape = operator_data(*operands).ufl_shape
        if expr_shape != function_space.ufl_element().value_shape():
            error("The dimension does not match with the dimension of the function space %s" % function_space)

    @property
    def expr(self):
        return self.operator_data

    def _compute_derivatives(self):
        cplx = utils.complex_mode
        real = not cplx
        symb = sp.symbols('s:%d' % len(self.ufl_operands), real=real, complex=cplx)
        r = sp.diff(self.expr(*symb), *zip(symb, self.derivatives))
        return sp.lambdify(symb, r, dummify=True)

    def _evaluate(self):
        operands = self.ufl_operands
        operator = self._compute_derivatives()
        expr = as_ufl(operator(*operands))
        if expr.ufl_shape == () and expr != 0:
            var = VariableRuleset(self.ufl_operands[0])
            expr = expr*var._Id
        elif expr == 0:
            return self.assign(expr)
        return self.interpolate(expr)


class PointsolveOperator(AbstractExternalOperator):
    r"""A :class:`PointsolveOperator` is an implementation of ExternalOperator that is defined through
    a given function f (e.g. a lambda expression) and whose values correspond to the root of this function
    evaluated pointwise, i.e. x such that f(x) = 0.

    The vectorized newton implementation relies on scipy.optimize.newton, it has the same syntax for the choice of the parameters and the same default values :
    newton_params = {'fprime':None, 'args':(), 'tol':1.48e-08, 'maxiter':50, 'fprime2':None, 'x1':None,
                    'rtol':0.0, 'full_output':False, 'disp':True}
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
     """

    _cache = {}

    def __init__(self, *operands, function_space, derivatives=None, val=None, name=None, coefficient=None, arguments=(), dtype=ScalarType, operator_data, disp=False):
        r"""
        :param operands: operands on which act the :class:`PointsolveOperator`.
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
        :param operator_data: dictionary containing the:
                - function defining how to evaluate the :class:`PointsolveOperator`
                - solver_name
                - solver_parameters:
                            + x0: initial condition
                            + fprime: gradient of the function defining the :class:`PointsolveOperator`
                            + maxiter: max number of iterations
                            + tol: tolerance
                  More parameters are available (cf. documentation of the scipy.optimize.newton).
                  We have extended to implementation of `scipy.optimize.newton` to handle non-scalar case. If a more
                  precise or efficient solver is needed for the pointwise solves, you can subclass the `evaluate` method.
                  TODO: Generate C-code that will be faster.
        :param disp: boolean indication whether we display the max of the number of iterations taken over the pointwise solves.
        """

        local_operands = operands
        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, val=val, name=name, coefficient=coefficient, arguments=arguments, local_operands=local_operands, dtype=dtype, operator_data=operator_data)

        # Check
        if not isinstance(operator_data['point_solve'], types.FunctionType):
            error("Expecting a FunctionType pointwise expression")

        if operator_data['point_solve'].__code__.co_argcount != len(operands) + 1:
            error("Expecting %s operands" % (operator_data['point_solve'].__code__.co_argcount-1))
        if operator_data['solver_name'] not in ('newton', 'secant', 'halley'):
            error("Expecting of the following method : (%s, %s, %s) " % ('newton', 'secant', 'halley'))
        if not isinstance(operator_data['solver_params'], dict):
            error("Expecting a dict with the solver arguments instead of %s" % operator_data['solver_params'])

        self.disp = disp

        # per this stack overflow
        # https://stackoverflow.com/questions/63363308/can-a-ufuncify-function-be-set-up-to-take-complex-input-ufunc-wrapper-modul
        # Cython doesn't do the right thing for ufuncify, so use fortran instead
        if utils.complex_mode:
            self.ufuncify_language = 'F95'
            self.ufuncify_backend = 'f2py'
        else:
            self.ufuncify_language = 'C'
            self.ufuncify_backend = 'Cython'

    @property
    def operator_f(self):
        return self.operator_data['point_solve']

    @property
    def solver_name(self):
        return self.operator_data['solver_name']

    @property
    def solver_params(self):
        return self.operator_data['solver_params']

    @property
    def symbolic_sympy_optim(self):
        return self.operator_data['sympy_optim']

    def _cache_key(self, F, y, mode, same_shape):
        return md5((str(F) + str(y) + str(mode) + str(same_shape)).encode()).hexdigest()

    def _cache_symbolic(mode):
        def eval_function(eval_func):
            # _symbolic_inv_jac just depends on the expression and the shapes involved
            # The symbolic inversion by itself is not so problematic however the lambdification (sp.lambdify)
            # of the result can quickly be horrifically slow
            # TODO: we can generate C code using sp.codegen to compute the expression obtained (it will still be
            # associated to this caching mechanism)
            def wrapper(self, *args, **kwargs):
                F = args[1]
                y = kwargs.get('y')
                same_shape = kwargs.get('same_shape')
                key = self._cache_key(F, y, mode, same_shape)
                cache = self._cache
                try:
                    return cache[key]
                except KeyError:
                    result = eval_func(self, *args, **kwargs)
                    cache[key] = result
                    return result
            return wrapper
        return eval_function

    def _compute_derivatives(self, f):
        deriv_index = (0,) + self.derivatives
        ufl_space = self._extop_master.ufl_function_space()
        symb = (self._sympy_create_symbols(ufl_space.shape, 0),)
        symb += tuple(self._sympy_create_symbols(e.ufl_shape, i+1) for i, e in enumerate(self.ufl_operands))
        shape = f.ufl_shape

        # If no derivatives is taken
        if deriv_index == (0,)*len(deriv_index):
            return f

        # Preprocessing
        fexpr = self.operator_f(*symb)
        args = self._prepare_args_f(ufl_space)
        vals = tuple(coeff.dat.data_ro for coeff in args)
        fj, vj = self._reshape_args(args, (f.dat.data_ro,) + vals, f)

        # Symbols construction
        symb = (self._sympy_create_symbols(shape, 0),)
        symb += tuple(self._sympy_create_symbols(e.ufl_shape, i+1) for i, e in enumerate(args))
        # We need to define appropriate symbols to impose the right shape on the inputs when lambdifying the sympy expressions
        ops_f = (self._sympy_create_symbols(shape, 0, granular=False),)
        ops_f += tuple(self._sympy_create_symbols(e.ufl_shape, i+1, granular=False)
                       for i, e in enumerate(args))
        new_symb = tuple(e.free_symbols.pop() for e in ops_f)

        # Compute derivatives
        for i, di in enumerate(deriv_index):
            # -> Sympy idiff does not work for high dimension
            if di != 0:
                # We want to compute dsymb[0]/dsymb[i]
                # We know by Implicit Function Theorem that :
                # if f(x,y) = 0 =>  df/dx + dy/dx*df/dy = 0
                dfdsi = sp.diff(fexpr, symb[i])
                dydx_symb = sp.diff(fexpr, symb[0])
                dydx = self._sympy_inv_jacobian(symb, fexpr, new_symb, ops_f, -dfdsi, same_shape=True)

                # dy/dx
                res = np.array(dydx(fj, *vj))
                if len(self.ufl_shape) > 2:
                    # for sp.Array (i.e rank > 2)
                    res = res.reshape(-1, *self.ufl_shape)

                if di > 1:
                    # ImpDiff : (df/dy)*(d2y/dx) = -(d2f/dx)+(d2f/dy)*(dy/dx)**2
                    d2fds0 = sp.diff(fexpr, symb[0], 2)
                    d2fdsi = sp.diff(fexpr, symb[i], 2)

                    C = - d2fdsi + d2fds0*dydx_symb*dydx_symb
                    d2ydx_symb = sp.diff(fexpr, symb[0], symb[0])
                    d2ydx = self._sympy_inv_jacobian(symb, fexpr, new_symb, ops_f, C, same_shape=True)

                    # d2y/dx
                    res = np.array(d2ydx(fj, *vj))
                    if len(self.ufl_shape) > 2:
                        # for sp.Array (i.e rank > 2)
                        res = res.reshape(-1, *self.ufl_shape)

                    if di == 3:
                        # ImpDiff : (df/dy)*(d3y/dx) = -(d3f/dx)+(d2f/dy)*(d2y/dx)*(dy/dx)+(d3f/dy)*(dy/dx)**3
                        d3fds0 = sp.diff(fexpr, symb[0], 3)
                        d3fdsi = sp.diff(fexpr, symb[i], 3)

                        C = - d3fdsi + d2fds0*d2ydx_symb*dydx_symb + d3fds0*(dydx_symb)**3
                        d3ydx = self._sympy_inv_jacobian(symb, fexpr, new_symb, ops_f, C, same_shape=True)

                        # d3y/dx
                        res = np.array(d3ydx(fj, *vj))
                        if len(self.ufl_shape) > 2:
                            # for sp.Array (i.e rank > 2)
                            res = res.reshape(-1, *self.ufl_shape)

                    elif di != 2:
                        # The implicit differentiation order can be extended if needed
                        raise NotImplementedError("Implicit differentiation of order n is not handled for n>3")

                if not all(v == 0 for v in deriv_index[:i]+deriv_index[i+1:]):
                    # Needed feature ?
                    raise NotImplementedError("Cross-derivatives not handled : %s" % deriv_index)
                break
        self.dat.data[:] = res
        return self.get_coefficient()

    def _evaluate(self):
        r"""
        Let f(x, y_1, ..., y_k) = 0, where y_1, ..., y_k are parameters. We look for the solution x of this equation.
        This method returns the solution x satisfying this equation or its derivatives: \frac{\partial x}{\partial y_i},
        using implicit differentiation.
        The parameters can be Functions, Constants, Expressions or even other PointwiseOperators.
        For the solution: x belongs to self.function_space.
        For the parameters: y_1, ..., y_k must either:
                                                        - have the same shape than x (in which case we interpolate them)
                                                        - belongs to a suitable function space
        """
        ufl_space = self.ufl_function_space()
        shape = ufl_space.shape
        solver_params = self.solver_params.copy()  # To avoid breaking immutability
        f = self.operator_f

        # If we want to evaluate derivative, we first have to evaluate the solution and then use implicit differentiation
        if self.derivatives != (0,)*len(self.derivatives):
            e_master = self._extop_master
            return self._compute_derivatives(e_master)

        # Prepare the arguments of f
        args = self._prepare_args_f(ufl_space)
        vals = tuple(coeff.dat.data_ro for coeff in args)

        # Symbols construction
        symb = (self._sympy_create_symbols(shape, 0),)
        symb += tuple(self._sympy_create_symbols(e.ufl_shape, i+1) for i, e in enumerate(args))

        # Pre-processing to get the values of the initial guesses
        if 'x0' in solver_params.keys() and isinstance(solver_params['x0'], Expr):
            val_x0 = solver_params.pop('x0')
            solver_params_x0 = Function(ufl_space).interpolate(val_x0).dat.data_ro
        else:
            solver_params_x0 = self.dat.data_ro

        # We need to define appropriate symbols to impose the right shape on the inputs when lambdifying the sympy expressions
        ops_f = (self._sympy_create_symbols(shape, 0, granular=False),)
        ops_f += tuple(self._sympy_create_symbols(e.ufl_shape, i+1, granular=False)
                       for i, e in enumerate(args))
        new_symb = tuple(e.free_symbols.pop() for e in ops_f)
        new_f = self._sympy_eval_function(symb, f(*symb))

        # Computation of the jacobian
        if self.solver_name in ('newton', 'halley'):
            if 'fprime' not in solver_params.keys():
                fexpr = f(*symb)
                df = self._sympy_inv_jacobian(symb, fexpr, new_symb, ops_f)
                solver_params['fprime'] = df

        # Computation of the hessian
        if self.solver_name == 'halley':
            if 'fprime2' not in solver_params.keys():
                # TODO: use something like _sympy_inv_jacobian
                d2f = sp.diff(f(*symb), symb[0], symb[0])
                d2f = self._sympy_subs_symbols(symb, ops_f, d2f)
                fprime2 = sp.lambdify(new_symb, d2f, modules='numpy', dummify=True)
                solver_params['fprime2'] = fprime2

        # Reshape numpy arrays
        solver_params_x0, vals = self._reshape_args(args, (solver_params_x0,) + vals)

        # Vectorized nonlinear solver (e.g. Newton, Halley...)
        res = self._vectorized_newton(new_f, solver_params_x0, args=vals, **solver_params)
        self.dat.data[:] = res.squeeze()
        return self.get_coefficient()

    def _prepare_args_f(self, space):
        """
        Prepare the arguments for the function f satisfying f(x, y_1, ..., y_k) = 0
        x belongs to self.function_space().
        y_1, ..., y_k must either:
                                     - have the same shape than x (in which case we interpolate them)
                                     - belongs to a suitable function space
        """
        shape = space.shape
        args = ()
        for i, pi in enumerate(self.ufl_operands):
            if isinstance(pi, Constant):
                # TODO: Is it worth using `functools.partial` instead to fix the constant values later on?!
                opi = Function(pi._ad_function_space(self.ufl_domain())).assign(pi)
            elif pi.ufl_shape == shape:
                opi = Function(space).interpolate(pi)
            # We can have an argument with a different shape
            # The argument should however belong to an appropriate space
            # TODO: Should we perfom additional procedures for differently shaped arguments?
            elif len(pi.dat.data_ro) == len(self.dat.data_ro):
                opi = pi
            else:
                raise ValueError('Invalid arguments: incompatible function space for %s', pi)
            args += (opi,)
        return args

    @_cache_symbolic(mode='solve_jacobian')
    def _sympy_inv_jacobian(self, symb, F, new_symb, ops_f, y=None, same_shape=False):
        r"""
        Symbolically solves J_{F} \delta x = y, where J_{F} is the Jacobian of F wrt x
        (abuse of notation, i.e. it refers to \frac{\partial F}{\partial x}).
        and \delta x the solution we are solving for and y the rhs, it can either be:
                            - F(x): when evaluating the PointSolveOperator
                            - \frac{\partial F(x)}{\partial y_i}: when evaluating the derivatives of the the PointSolveOperator
        Solving symbolically leverages the fact that we are at the end solving the same equation over all the points,
        we then just have to replace the symbolic expression by the numerical values when needed.
        """
        x = symb[0]
        y = y or F
        df = sp.diff(F, x)
        self_shape = self.ufl_shape
        if len(self_shape) == 1:
            self_shape += (1,)

        if isinstance(df, sp.Matrix):
            sol = df.LUsolve(y)
        elif isinstance(x, sp.Symbol):
            sol = y/df
        else:
            # Sympy does not handle symbolic tensor inversion, as it does for rank <= 2, so we reshape the problem.
            # For instance, DF \delta x = y where DF.shape = (2,2,2,2) and x.shape/y.shape = (2,2)
            # will be turn into DF \delta x = y where DF.shape = (4,4) and x.shape/y.shape = (4,1)
            sol = df

            # Matrix
            shape = df.shape
            n_elts = int(reduce(mul, shape)**0.5)
            if n_elts**2 != reduce(mul, shape):
                raise ValueError('Incompatible dimension: %', shape)
            dfMat = np.array(df.tolist()).reshape(n_elts, n_elts)
            dfMat = sp.Matrix(dfMat)

            # Vector
            yVec = np.array(y.tolist()).flatten()
            yVec = sp.Matrix(yVec)
            if same_shape:
                # Ax=B with A.shape/B.shape/x.shape = (2,2) (happens frequently for implicit differentiation)
                yVec = yVec.reshape(n_elts, n_elts)

            # Compute solution and reshape (symbolically)
            sol = dfMat.LUsolve(yVec)
            """
            if len(self_shape) > 2:
                return sp.Array(sol, self_shape)
            sol = sol.reshape(*self_shape)
            """
            if len(self_shape) < 3:
                sol = sol.reshape(*self_shape)

        nsize = self.dat.data_ro.shape[0]
        if hasattr(sol, 'shape'):
            self_shape = sol.shape

        return self._generate_eval_sympy_C_kernel(sol, symb, self_shape, nsize)

    @_cache_symbolic(mode='eval_f')
    def _sympy_eval_function(self, symb, F):
        """TODO: """
        self_shape = self.ufl_shape
        if len(self_shape) == 1:
            self_shape += (1,)

        nsize = self.dat.data_ro.shape[0]
        return self._generate_eval_sympy_C_kernel(F, symb, self_shape, nsize)

    def _generate_eval_sympy_C_kernel(self, symbolic_expr, symb, shape, n):
        r"""TODO: Explain: only generated once call when performing solve and cache + on simple expressions does not make big difference but as an expr gets more complicated significant impact + optimization can considerably speed up (it basically introduces intermediate variables for redundant terms and appends the assignment lines in the C code) ...
        """

        from sympy.utilities.autowrap import ufuncify

        new_expr = symbolic_expr
        if self.symbolic_sympy_optim:
            rep, new_expr = sp.cse(symbolic_expr)
            new_expr = new_expr[0]

        # Scalar eval kernel
        def eval_ufunc_scalar(S, symb, n):
            r""" Wrapper for the C evaluation code generated for scalar
            """
            # Get symbols
            symbs = ()
            for e in symb:
                symbs += (e,)

            if self.symbolic_sympy_optim:
                # Helpers block for optimization
                H = ()
                extra_symbs = []
                for var, sub_expr in rep:
                    H += ((str(var), sub_expr, list(symbs) + extra_symbs),)
                    extra_symbs += [var]
                # TODO: define directory (add name path tempdir="[name_path]")
                gen_func = ufuncify(symbs, S,
                                    language=self.ufuncify_language,
                                    backend=self.ufuncify_backend,
                                    helpers=list(H))
            else:
                # Special case because helpers=list( () ) slows down the code generation
                # TODO: define directory (add name path tempdir="[name_path]")
                gen_func = ufuncify(symbs, S,
                                    language=self.ufuncify_language,
                                    backend=self.ufuncify_backend)

            def python_eval_wrapper(*args, n=n):
                result = np.empty((n,), dtype=utils.ScalarType)
                vals = ()
                for e in args:
                    if len(e.shape) == 1:
                        vals += (e.copy(),)
                    else:
                        vals += tuple(e[:, i, j].copy() for i in range(e.shape[1]) for j in range(e.shape[2]))

                result[:] = gen_func(*vals)
                return result
            return python_eval_wrapper

        # Vector, Matrix and Array eval kernel
        def eval_ufunc_Matrix(M, symb, shape, n):
            r""" Wrapper for the C evaluation code generated for sp.Matrix, this also encompasses Vectors and Arrays
            since both are are reshaped to matrix
            TODO: Explain that shape the python bit is shape dependent ()
            Dire ufuncify vectorized version of autowrap but does not handle non scalar expression so element wise kernel
            """
            nn = shape[0]
            mm = shape[1]

            # Get symbols
            symbs = ()
            for e in symb:
                symbs += tuple(np.array(e.tolist()).flatten())
            expr = [mi for mi in M]

            if self.symbolic_sympy_optim:
                # Helpers block for optimization
                H = ()
                extra_symbs = []
                for var, sub_expr in rep:
                    H += ((str(var), sub_expr, list(symbs) + extra_symbs),)
                    extra_symbs += [var]
                # TODO: define directory (add name path tempdir="[name_path]")
                ufuncs = [ufuncify(symbs, expr[i],
                                   language=self.ufuncify_language,
                                   backend=self.ufuncify_backend,
                                   helpers=list(H)) for i in range(nn*mm)]
            else:
                # Special case because helpers=list( () ) slows down the code generation
                # TODO: define directory (add name path tempdir="[name_path]")
                ufuncs = [ufuncify(symbs, expr[i],
                                   language=self.ufuncify_language,
                                   backend=self.ufuncify_backend) for i in range(nn*mm)]

            def python_eval_wrapper(*args, n=n, M=M, expr=expr, symbs=symbs):
                result = np.empty((n, nn, mm))
                vals = ()
                for e in args:
                    if len(e.shape) == 1:
                        vals += (e.copy(),)
                    else:
                        vals += tuple(e[:, i, j].copy() for i in range(e.shape[1]) for j in range(e.shape[2]))

                for idx, gen_func in enumerate(ufuncs):
                    i = int(idx/mm)
                    j = idx % mm
                    result[:, i, j] = gen_func(*vals)
                return result[:]
            return python_eval_wrapper

        # Return the eval kernel
        if len(shape) > 1:
            return eval_ufunc_Matrix(new_expr, symb, shape, n)
        return eval_ufunc_scalar(new_expr, symb, n)

    def _reshape_args(self, args, values, f=None):
        f = f or self
        res = ()
        args = (f,) + args
        for a, v in zip(args, values):
            if len(a.ufl_shape) == 1:
                v = np.expand_dims(v, -1)
            res += (v,)
        return res[0], res[1:]

    def _vectorized_newton(self, func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50, fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
        """
        A vectorized version of Newton and Halley methods for arrays
        This version is a modification of the 'optimize.newton' function
        from the SciPy library which handles non-scalar cases.
        """
        # Explicitly copy `x0` as `p` will be modified inplace, but, the
        # user's array should not be altered.
        try:
            p = np.array(x0, copy=True, dtype=utils.ScalarType)
        except TypeError:
            # can't convert complex to float
            p = np.array(x0, copy=True)

        failures = np.ones_like(p, dtype=bool)
        zero_val = np.zeros_like(p, dtype=bool)
        nz_der = np.ones_like(failures)
        if fprime is not None:
            # Newton-Raphson method
            for iteration in range(maxiter):
                # first evaluate fval
                fval = np.asarray(func(p, *args))
                # If all fval are 0, all roots have been found, then terminate
                # newton_time = time.time()
                if not fval.any():
                    failures = fval.astype(bool)
                    break
                fder = np.asarray(fprime(p, *args))
                nz_der = (fder != 0)
                # stop iterating if all derivatives are zero
                if not nz_der.any():
                    break
                # Newton step
                dp = fder[nz_der]

                if fprime2 is not None:
                    fder2 = np.asarray(fprime2(p, *args))
                    # dp = dp / (1.0 - 0.5 * dp * fder2[nz_der])
                    dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / fder[nz_der])
                # only update nonzero derivatives
                p[nz_der] -= dp
                failures[nz_der] = np.abs(dp) >= tol  # items not yet converged
                # stop iterating if there aren't any failures, not incl zero der
                if not failures[nz_der].any():
                    zero_val = np.abs(fval) < tol
                    break

        zero_der = ~nz_der & failures  # don't include converged with zero-ders
        if not zero_val.all():
            if zero_der.any():
                import warnings
                all_or_some = 'all' if zero_der.all() else 'some'
                msg = '{:s} derivatives were zero'.format(all_or_some)
                warnings.warn(msg, RuntimeWarning)
            elif failures.any():
                import warnings
                all_or_some = 'all' if failures.all() else 'some'
                msg = '{0:s} failed to converge after {1:d} iterations'.format(all_or_some, maxiter)
                if failures.all():
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning)

        return p

    def _sympy_create_symbols(self, xshape, i, granular=True):
        cplx = utils.complex_mode
        real = not cplx
        if len(xshape) == 0:
            return sp.symbols('s_'+str(i), real=real, complex=cplx)
        elif len(xshape) == 1:
            if not granular:
                return sp.Matrix(sp.MatrixSymbol('V_'+str(i), xshape[0], 1))
            symb = sp.symbols('v'+str(i)+'_:%d' % xshape[0], real=real, complex=cplx)
            # sp.Matrix are more flexible for the representation of vector than sp.Array (e.g it enables to use norms)
            return sp.Matrix(symb)
        elif len(xshape) == 2:
            if not granular:
                return sp.Matrix(sp.MatrixSymbol('M_'+str(i), *xshape))
            nm = xshape[0]*xshape[1]
            symb = sp.symbols('m'+str(i)+'_:%d' % nm, real=real, complex=cplx)
            coeffs = [symb[i:i+xshape[1]] for i in range(0, nm, xshape[1])]
            return sp.Matrix(coeffs)

    def _sympy_subs_symbols(self, old, new, fprime):
        s1 = ()
        s2 = ()
        for o, n in zip(old, new):
            if isinstance(o, sp.Symbol):
                s1 += (o,)
                s2 += (n,)
            else:
                s1 += tuple(o for o in np.array(o.tolist()).flatten())
                s2 += tuple(o for o in np.array(n.tolist()).flatten())
        if hasattr(fprime, 'subs'):
            return fprime.subs(dict(zip(s1, s2)))

        T = sp.tensor.array.Array(fprime)
        return T.subs(dict(zip(s1, s2)))


class PointnetOperator(AbstractExternalOperator):
    r"""A :class:`PointnetOperator`: is an implementation of ExternalOperator that is defined through
    a given neural network model N and whose values correspond to the output of the neural network represented by N.
     """

    def __init__(self, *operands, function_space, derivatives=None, val=None, name=None, coefficient=None, arguments=(), dtype=ScalarType, operator_data, weights_version=None, local_operands=()):
        r"""
        :param operands: operands on which act the :class:`PointnetOperator`.
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
        :param operator_data: dictionary containing the:
                - model: the machine learning model
                - framework: it specifies wich machine learning framework we are dealing with (e.g. Pytorch or Tensorflow)
                - ncontrols: the number of controls
        :param weights_version: a dictionary keeping track of the weights version, to inform if whether we need to update them.
        """

        # Add the weights in the operands list and update the derivatives multiindex
        last_op = operands[-1]
        init_weights = (isinstance(last_op, ReferenceValue) and isinstance(last_op.ufl_operands[0], Constant))
        init_weights = init_weights or isinstance(last_op, Constant)
        if not init_weights:
            weights_val = ml_get_weights(operator_data['model'], operator_data['framework'])
            cw = Constant(zero(*weights_val.shape))
            # Assign and convert (from torch to numpy)
            cw.dat.data[:] = weights_val
            operands += (cw,)
            # TODO: At the moment the Global Neural Net case is not handled!!
            local_operands += (cw,)
            # Type exception is caught later
            if isinstance(derivatives, tuple):
                derivatives += (0,)

        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, val=val, name=name, coefficient=coefficient, arguments=arguments, local_operands=local_operands, dtype=dtype, operator_data=operator_data)

        # Checks
        if 'ncontrols' not in self.operator_data.keys():
            self.operator_data['ncontrols'] = 1
        if not isinstance(operator_data['ncontrols'], int) or operator_data['ncontrols'] > len(self.ufl_operands):
            error("Expecting for the number of controls an int type smaller or equal \
                  than the number of operands and not %s" % operator_data['ncontrols'])

        self._controls = tuple(range(0, self.ncontrols))

        if weights_version is not None:
            self._weights_version = weights_version
        else:
            self._weights_version = {'version': 1, 'W': self.ufl_operands[-1]}

    @property
    def framework(self):
        return self.operator_data['framework']

    @property
    def model(self):
        return self.operator_data['model']

    @property
    def ncontrols(self):
        return self.operator_data['ncontrols']

    @property
    def controls(self):
        return dict(zip(self._controls, tuple(self.ufl_operands[i] for i in self._controls)))

    def copy(self, deepcopy=False):
        r"""Return a copy of this CoordinatelessFunction.

        :kwarg deepcopy: If ``True``, the new
            :class:`CoordinatelessFunction` will allocate new space
            and copy values.  If ``False``, the default, then the new
            :class:`CoordinatelessFunction` will share the dof values.
        """
        if deepcopy:
            val = type(self.dat)(self.dat)
        else:
            val = self.dat
        return type(self)(*self.ufl_operands, function_space=self.function_space(), val=val,
                          name=self.name(), dtype=self.dat.dtype,
                          derivatives=self.derivatives,
                          operator_data=self.operator_data,
                          weights_version=self._weights_version)

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, name=None, operator_data=None, val=None, coefficient=None, arguments=(), add_kwargs={}):
        "Overwrite _ufl_expr_reconstruct to pass on weights_version"
        add_kwargs['weights_version'] = self._weights_version
        return AbstractExternalOperator._ufl_expr_reconstruct_(self, *operands, function_space=function_space,
                                                               derivatives=derivatives,
                                                               val=val, name=name,
                                                               coefficient=coefficient,
                                                               arguments=arguments,
                                                               operator_data=operator_data,
                                                               add_kwargs=add_kwargs)


class PytorchOperator(PointnetOperator):
    r"""A :class:`PytorchOperator`: is an implementation of ExternalOperator that is defined through
    a given PyTorch model N and whose values correspond to the output of the neural network represented by N.
    The inputs of N are obtained by interpolating `self.ufl_operands[0]` into `self.function_space`.

    +   The evaluation of the PytorchOperator is done by performing a forward pass through the network N
        The first argument is considered as the input of the network, if one want to correlate different
        arguments (Functions, Constant, Expressions or even other PointwiseOperators) then he needs
        to either:
                    - subclass this method to specify how this correlation should be done
                    or
                    - construct another pointwise operator that will do this job and pass it in as argument
     +  The gradient of the PytorchOperator is done by taking the gradient of the outputs of N with respect to
        its inputs.
     """

    def __init__(self, *operands, function_space, derivatives=None, val=None, name=None, coefficient=None, arguments=(), dtype=ScalarType, operator_data, weights_version=None):
        r"""
        :param operands: operands on which act the :class:`PytorchOperator`.
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
        :param operator_data: dictionary containing the:
                - model: the Pytorch model
                - ncontrols: the number of controls
        :param weights_version: a dictionary keeping track of the weights version, to inform if whether we need to update them.
        """

        local_operands = operands
        PointnetOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, val=val, name=name, coefficient=coefficient, arguments=arguments, local_operands=local_operands, dtype=dtype, operator_data=operator_data, weights_version=weights_version)

        # Set datatype to double (torch.float64) as the firedrake.Function default data type is float64
        self.model.double()  # or torch.set_default_dtype(torch.float64)

    @utils.cached_property
    def ml_backend(self):
        try:
            import torch
        except ImportError:
            raise ImportError("Error when trying to import PyTorch")
        return torch

    def _compute_derivatives(self, N, x, model_tape=False):
        """Compute the gradient of the network wrt inputs"""
        if self.derivatives == (0,)*len(self.ufl_operands):
            N = N.squeeze(1)
            if model_tape:
                return N
            return N.detach()
        elif self.derivatives[-1] == 1:
            # When we want to compute: \frac{\partial{N}}{\partial{weights}}
            return self.ml_backend.zeros(len(x))

        gradient = self.ml_backend.autograd.grad(list(N), x, retain_graph=True)[0]
        return gradient.squeeze(1)

    def _eval_update_weights(evaluate):
        """Check if we need to update the weights"""
        def wrapper(self, *args, **kwargs):
            # Get Constants representing weights
            self_w = self._weights_version['W']
            w = self.ufl_operands[-1]
            # Get versions
            self_version = self._weights_version['version']
            w_version = w.dat.dat_version

            if self_version != w_version or w != self_w:
                if w._is_control:
                    self._weights_version['version'] = w_version
                    self._weights_version['W'] = w
                    self._update_model_weights()

            return evaluate(self, *args, **kwargs)
        return wrapper

    @_eval_update_weights
    def _evaluate(self, model_tape=False):
        """
        Evaluate the neural network by performing a forward pass through the network
        The first argument is considered as the input of the network, if one want to correlate different
        arguments (Functions, Constant, Expressions or even other PointwiseOperators) then he needs
        to either:
                    - subclass this method to specify how this correlation should be done
                    or
                    - construct another pointwise operator that will do this job and pass it in as argument
        """
        model = self.model

        # Explictly set the eval mode does matter for
        # networks having different behaviours for training/evaluating (e.g. Dropout)
        model.eval()

        space = self.ufl_function_space()
        op = Function(space).interpolate(self.ufl_operands[0])

        torch_op = self.ml_backend.tensor(op.dat.data_ro, requires_grad=True)
        torch_op = self.ml_backend.unsqueeze(torch_op, 1)

        # Vectorized forward pass
        val = model(torch_op)
        res = self._compute_derivatives(val, torch_op, model_tape)

        # We return a list instead of assigning to keep track of the PyTorch tape contained in the torch variables
        if model_tape:
            return res
        result = Function(space)
        result.dat.data[:] = res

        # Explictly set the train mode does matter for
        # networks having different behaviours for training/evaluating (e.g. Dropout)
        model.train()

        return self.assign(result)

    def get_weights(self):
        return ml_get_weights(self.model, self.framework)

    def evaluate_adj_component_control(self, x, idx):
        if idx == len(self.ufl_operands) - 1:
            outputs = self.evaluate(model_tape=True)
            weights = self.model.weight
            grad_W = self.ml_backend.autograd.grad(outputs, weights, grad_outputs=[self.ml_backend.tensor(x.dat.data_ro)], retain_graph=True)
            """
            for i, e in enumerate(outputs):
                v = self.ml_backend.tensor(x.dat.data_ro[i])
                res.dat.data[i] = self.ml_backend.autograd.grad(e, weights, grad_outputs=[v], retain_graph=True)[0]
            """
            w = self.ufl_operands[-1]
            cst_fct_space = w._ad_function_space(self.function_space().mesh())
            return Function(cst_fct_space, val=grad_W).vector()
        return AbstractExternalOperator.evaluate_adj_component_control(self, x, idx)

    def _assign_weights(self, weights):
        self.model.weight.data = weights

    def _update_model_weights(self):
        weights_op = self.ml_backend.tensor(self.ufl_operands[-1].dat.data_ro)
        self._assign_weights(weights_op.unsqueeze(0))


class TensorFlowOperator(PointnetOperator):
    r"""A :class:`TensorFlowOperator` ... TODO :
     """

    def __init__(self, *operands, function_space, derivatives=None, val=None, name=None, dtype=ScalarType, operator_data):
        PointnetOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, val=val, name=name, dtype=dtype, operator_data=operator_data)
        raise NotImplementedError('TensorFlowOperator not implemented yet!')


# Helper functions #
def point_expr(point_expr, function_space):
    r"""The point_expr function returns the `PointexprOperator` class initialised with :
        - point_expr : a function expression (e.g. lambda expression)
        - function space
     """
    return partial(PointexprOperator, operator_data=point_expr, function_space=function_space)


def point_solve(point_solve, function_space, solver_name='newton', solver_params=None, disp=False, sympy_optim=False):
    r"""The point_solve function returns the `PointsolveOperator` class initialised with :
        - point_solve : a function expression (e.g. lambda expression)
        - function space
        - solver_name
        - solver_params
        - disp : if you want to display the maximum number of iterations taken over the whol poinwise solves.
    """
    if solver_params is None:
        solver_params = {}
    operator_data = {'point_solve': point_solve, 'solver_name': solver_name,
                     'solver_params': solver_params, 'sympy_optim': sympy_optim}
    if disp not in (True, False):
        disp = False
    if sympy_optim not in (True, False):
        sympy_optim = True
    return partial(PointsolveOperator, operator_data=operator_data, function_space=function_space, disp=disp)


def neuralnet(model, function_space, ncontrols=1):

    torch_module = type(None)
    tensorflow_module = type(None)

    # Checks
    try:
        import torch
        torch_module = torch.nn.modules.module.Module
    except ImportError:
        pass

    if isinstance(model, torch_module):
        operator_data = {'framework': 'PyTorch', 'model': model, 'ncontrols': ncontrols}
        return partial(PytorchOperator, function_space=function_space, operator_data=operator_data)
    elif isinstance(model, tensorflow_module):
        operator_data = {'framework': 'TensorFlow', 'model': model, 'ncontrols': ncontrols}
        return partial(TensorFlowOperator, function_space=function_space, operator_data=operator_data)
    else:
        error("Expecting one of the following library : PyTorch, TensorFlow (or Keras) and that the library has been installed")


def weights(*args):
    res = []
    for e in args:
        w = e.ufl_operands[-1]
        if not isinstance(w, Constant):
            raise TypeError("Expecting a Constant and not $s", w)
        res += [w]
    if len(res) == 1:
        return res[0]
    return res


def ml_get_weights(model, framework):
    if framework == 'PyTorch':
        return model.weight.data
    else:
        raise NotImplementedError(framework + ' operator is not implemented yet.')
