from abc import ABCMeta, abstractmethod
from functools import partial
import types
import sympy as sp
import numpy as np

from ufl import zero
from ufl.core.external_operator import ExternalOperator
from ufl.core.expr import Expr
from ufl.algorithms.apply_derivatives import VariableRuleset
from ufl.constantvalue import as_ufl
from ufl.operators import transpose
from ufl.log import error

import firedrake.assemble
from firedrake.function import Function
from firedrake.constant import Constant
from firedrake import utils, functionspaceimpl
from firedrake.adjoint import PointwiseOperatorsMixin
from pyop2.datatypes import ScalarType


class AbstractPointwiseOperator(Function, ExternalOperator, PointwiseOperatorsMixin, metaclass=ABCMeta):
    r"""Abstract base class from which stem all the Firedrake practical implementations of the
    ExternalOperator, i.e. all the ExternalOperator subclasses that have mechanisms to be
    evaluated pointwise and to provide their own derivatives.
    This class inherits from firedrake.function.Function and ufl.core.external_operator.ExternalOperator
    Every subclass based on this class must provide the `compute_derivatives` and 'evaluate' methods.
    """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data=None):
        ExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count)
        aa = function_space
        ofs = self.original_function_space()
        if not isinstance(ofs, functionspaceimpl.WithGeometry):
            fs1 = functionspaceimpl.FunctionSpace(function_space.mesh().topology, ofs.ufl_element())
            fs = functionspaceimpl.WithGeometry(fs1, function_space.mesh())
        else:
            fs = ofs

        Function.__init__(self, fs, val, name, dtype, count=self._count)  # count has been initialised in ExternalOperator.__init__

        self._ufl_function_space = aa
        self._original_function_space = fs
        self.operator_data = operator_data

    @abstractmethod
    def compute_derivatives(self):
        """apply the derivatives on operator_data"""

    @abstractmethod
    def evaluate(self):
        """define the evaluation method for the ExternalOperator object"""

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
    def adjoint_action(self, x, idx):
        r"""Starting from the residual form: F(N(u, m), u(m), m) = 0
            This method computes the action of (dN/dq)^{*}
            where q \in \{u, m\}.
        """
        derivatives = tuple(dj + int(idx == j) for j, dj in enumerate(self.derivatives))
        dNdq = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives)
        dNdq = dNdq.evaluate()
        dNdq_adj = transpose(dNdq)
        result = firedrake.assemble(dNdq_adj)
        return result.vector() * x

    def _adjoint(self, idx):
        derivatives = tuple(dj + int(idx == j) for j, dj in enumerate(self.derivatives))
        dNdq = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives)
        dNdq = dNdq.evaluate()
        dNdq_adj = transpose(dNdq)
        result = firedrake.assemble(dNdq_adj)
        return result

    @utils.cached_property
    def _split(self):
        return tuple(Function(V, val) for (V, val) in zip(self.function_space(), self.topological.split()))

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, count=None, name=None, operator_data=None, val=None, add_kwargs={}):
        "Return a new object of the same type with new operands."
        deriv_multiindex = derivatives or self.derivatives

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            corresponding_count = None
            e_master = self._extop_master
            for ext in e_master._extop_dependencies:
                if ext.derivatives == deriv_multiindex:
                    return ext._ufl_expr_reconstruct_(*operands, function_space=function_space,
                                                      derivatives=deriv_multiindex, count=count,
                                                      val=ext.topological, name=name,
                                                      operator_data=operator_data,
                                                      add_kwargs=add_kwargs)
        else:
            corresponding_count = self._count
            val = self.topological

        reconstruct_op = type(self)(*operands, function_space=function_space or self._ufl_function_space,
                                    derivatives=deriv_multiindex,
                                    count=corresponding_count,
                                    val=val,
                                    name=name or self.name(),
                                    operator_data=operator_data or self.operator_data,
                                    **add_kwargs)

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            self._extop_master._extop_dependencies.append(reconstruct_op)
            reconstruct_op._extop_master = self._extop_master
        else:
            reconstruct_op._extop_master = self._extop_master
            reconstruct_op._extop_dependencies = self._extop_dependencies
        return reconstruct_op

    def __str__(self):
        "Default repr string construction for PointwiseOperator operators."
        # This should work for most cases
        r = "%s(%s,%s,%s,%s,%s)" % (type(self).__name__, ", ".join(repr(op) for op in self.ufl_operands), repr(self._ufl_function_space), repr(self.derivatives), repr(self.ufl_shape), repr(self.operator_data))
        return r


class PointexprOperator(AbstractPointwiseOperator):
    r"""A :class:`PointexprOperator` is an implementation of ExternalOperator that is defined through
    a given function f (e.g. a lambda expression) and whose values are defined through the mere evaluation
    of f pointwise.
    """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data):
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
        return sp.lambdify(symb, r, dummify=True)

    def evaluate(self):
        operands = self.ufl_operands
        operator = self.compute_derivatives()
        expr = as_ufl(operator(*operands))
        if expr.ufl_shape == () and expr != 0:
            var = VariableRuleset(self.ufl_operands[0])
            expr = expr*var._Id
        elif expr == 0:
            return self.assign(expr)
        return self.interpolate(expr)


class PointsolveOperator(AbstractPointwiseOperator):
    r"""A :class:`PointsolveOperator` is an implementation of ExternalOperator that is defined through
    a given function f (e.g. a lambda expression) and whose values correspond to the root of this function
    evaluated pointwise, i.e. x such that f(x) = 0.

    The vectorized newton implementation relies on scipy.optimize.newton, it has the same syntax for the choice of the parameters and the same default values :
    newton_params = {'fprime':None, 'args':(), 'tol':1.48e-08, 'maxiter':50, 'fprime2':None, 'x1':None,
                    'rtol':0.0, 'full_output':False, 'disp':True}
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
     """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data, disp=False):
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

        AbstractPointwiseOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, operator_data=operator_data)

        # Check
        if not isinstance(operator_data['point_solve'], types.FunctionType):
            error("Expecting a FunctionType pointwise expression")

        if operator_data['point_solve'].__code__.co_argcount != len(operands) + 1:
            error("Expecting %s operands" % (operator_data['point_solve'].__code__.co_argcount-1))
        if operator_data['solver_name'] not in ('newton', 'secant', 'halley'):
            error("Expecting of the following method : %s" % ('newton', 'secant', 'halley'))
        if not isinstance(operator_data['solver_params'], dict):
            error("Expecting a dict with the solver arguments instead of %s" % operator_data['solver_params'])

        self.disp = disp

    @property
    def operator_f(self):
        return self.operator_data['point_solve']

    @property
    def solver_name(self):
        return self.operator_data['solver_name']

    @property
    def solver_params(self):
        return self.operator_data['solver_params']

    def compute_derivatives(self, f):
        deriv_index = (0,) + self.derivatives
        symb = (self._sympy_create_symbols(self.ufl_function_space().shape, 0),)
        symb += tuple(self._sympy_create_symbols(e.ufl_shape, i+1) for i, e in enumerate(self.ufl_operands))
        ufl_space = self.ufl_function_space()
        ufl_shape = self.ufl_shape

        if deriv_index == (0,)*len(deriv_index):
            return f

        fexpr = self.operator_f(*symb)
        args, constants = self._prepare_args_f(ufl_space)
        vals = tuple(coeff.dat.data_ro for coeff in args)
        for i, di in enumerate(deriv_index):
            if di != 0:
                def implicit_differentiation(res):
                    # We want to compute dsymb[0]/dsymb[i]
                    # We know by Implicit Function Theorema that :
                    # if f(x,y) = 0 =>  df/dx + dy/dx*df/dy = 0
                    dfds0 = sp.diff(fexpr, symb[0])
                    dfdsi = sp.diff(fexpr, symb[i])

                    # -> Sympy idiff does not work for high dimension
                    # -> We could solve the system "dfds0*ds0/dsi = -dfdsi" symbolically and then
                    # for the n-th derivative just apply n-1th time the derivative on the result.
                    # However again sympy.linsolve() does not handle high dimensional case
                    dfds0l = sp.lambdify(symb, dfds0, modules='numpy', dummify=True)
                    dfdsil = sp.lambdify(symb, dfdsi, modules='numpy', dummify=True)

                    if di > 1:
                        d2fds0 = sp.diff(fexpr, symb[0], 2)
                        d2fdsi = sp.diff(fexpr, symb[i], 2)

                        d2fds0l = sp.lambdify(symb, d2fds0, modules='numpy', dummify=True)
                        d2fdsil = sp.lambdify(symb, d2fdsi, modules='numpy', dummify=True)
                        if di == 3:
                            d3fds0 = sp.diff(fexpr, symb[0], 3)
                            d3fdsi = sp.diff(fexpr, symb[i], 3)

                            d3fds0l = sp.lambdify(symb, d3fds0, modules='numpy', dummify=True)
                            d3fdsil = sp.lambdify(symb, d3fdsi, modules='numpy', dummify=True)
                        elif di != 2:
                            # The implicit differentiation order can be extended if needed
                            raise NotImplementedError("Implicit differentiation of order n is not handled for n>3")

                    def multidimensional_numpy_solve(shape):
                        if len(shape) == 0:
                            return lambda x: 1/x
                        elif len(shape) == 2:
                            return np.linalg.inv
                        else:
                            return np.linalg.tensorinv

                    # We store the function to avoid to have a if in the loop
                    solve_multid = multidimensional_numpy_solve(ufl_shape)
                    # TODO : Vectorized version ?
                    for j in range(len(res)):
                        fj = f.dat.data_ro[j]
                        val_ops = ()
                        for k, v in enumerate(vals):
                            if k in constants:
                                val_ops += (v,)
                            else:
                                val_ops += (v[j],)
                        A = dfds0l(fj.flatten(), *(voj.flatten() for voj in val_ops))
                        B = dfdsil(fj.flatten(), *(voj.flatten() for voj in val_ops))

                        # Conversion to numpy array and squeezing
                        A = np.squeeze(A)
                        B = np.squeeze(B)

                        # ImpDiff : df/dy * dy/dx = df/dx
                        C = -B
                        solve_multid = multidimensional_numpy_solve(A.shape)
                        InvA = solve_multid(A)
                        dydx = InvA*C
                        if di == 1:
                            res[j] = dydx
                        else:  # di = 2 or 3
                            A2 = d2fds0l(fj.flatten(), *(voj.flatten() for voj in val_ops))
                            B2 = d2fdsil(fj.flatten(), *(voj.flatten() for voj in val_ops))

                            # Conversion to numpy array and squeezing
                            A2 = np.squeeze(A2)
                            B2 = np.squeeze(B2)

                            # ImpDiff : (df/dy)*(d2y/dx) = -(d2f/dx)-(d2f/dy)*(dy/dx)**2
                            C = -B2 + A2*dydx*dydx
                            d2ydx = InvA*C
                            if di == 2:
                                res[j] = d2ydx
                            else:  # di = 3
                                A3 = d3fds0l(fj.flatten(), *(voj.flatten() for voj in val_ops))
                                B3 = d3fdsil(fj.flatten(), *(voj.flatten() for voj in val_ops))

                                # Conversion to numpy array and squeezing
                                A3 = np.squeeze(A3)
                                B3 = np.squeeze(B3)

                                # ImpDiff : (df/dy)*(d3y/dx) = -(d3f/dx)+(d2f/dy)*(d2y/dx)*(dy/dx)+(d3f/dy)*(dy/dx)**3
                                C = - B3 + A2*d2ydx*dydx + A3*(dydx)**3
                                res[j] = InvA*C
                implicit_differentiation(self.dat.data)
                if not all(v == 0 for v in deriv_index[:i]+deriv_index[i+1:]):
                    raise NotImplementedError("Cross-derivatives not handled : %s" % deriv_index)  # Needed feature ?
                break
        return self

    def evaluate(self):
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
            xstar = e_master.evaluate()
            return self.compute_derivatives(xstar)

        # Prepare the arguments of f
        args, constants = self._prepare_args_f(ufl_space)
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
        new_f = sp.lambdify(new_symb, f(*ops_f), modules='numpy', dummify=True)

        # Computation of the jacobian
        if self.solver_name in ('newton', 'halley'):
            if 'fprime' not in solver_params.keys():
                fexpr = f(*symb)
                df = self._sympy_inv_jacobian(symb[0], fexpr)
                df = self._sympy_subs_symbols(symb, ops_f, df, shape)
                fprime = sp.lambdify(new_symb, df, modules='numpy', dummify=True)
                solver_params['fprime'] = fprime

        # Computation of the hessian
        if self.solver_name == 'halley':
            if 'fprime2' not in solver_params.keys():
                # TODO: use something like _sympy_inv_jacobian
                d2f = sp.diff(f(*symb), symb[0], symb[0])
                d2f = self._sympy_subs_symbols(symb, ops_f, d2f, shape)
                fprime2 = sp.lambdify(new_symb, d2f, modules='numpy', dummify=True)
                solver_params['fprime2'] = fprime2

        offset = 0
        if len(shape) == 1:
            # Expand the dimension
            offset = 1
            solver_params_x0 = np.expand_dims(solver_params_x0, -1)
            vals = tuple(np.expand_dims(e, -1) for e in vals)
        elif len(shape) >= 2:
            # Sympy does not handle symbolic tensor inversion, as it does for rank <= 2
            # so we need to operate on the numpy vector
            df = solver_params['fprime']
            solver_params['fprime'] = self._numpy_tensor_solve(new_f, solver_params['fprime'], constants)

        # Reshape
        solver_params_x0 = np.rollaxis(solver_params_x0, 0, offset+len(shape)+1)
        vals = tuple(np.rollaxis(e, 0, offset+len(args[i].ufl_shape)+1) for i, e in enumerate(vals))

        # Vectorized nonlinear solver (e.g. Newton, Halley...)
        res = self._vectorized_newton(new_f, solver_params_x0, args=vals, **solver_params)
        self.dat.data[:] = np.rollaxis(res.squeeze(), -1)

        return self

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
        constants = ()
        for i, pi in enumerate(self.ufl_operands):
            if isinstance(pi, Constant):
                constants += (i,)
                opi = pi
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
        return args, constants

    def _sympy_inv_jacobian(self, x, F):
        """
        Symbolically performs the inversion of the Jacobian of F, except if x is an Array (i.e. rank>2)
        in which case we return the Jacobian of F and perform later on the inversion numerically.
        This is because sympy does not allow for symbolic inversion of Array objects.
        """
        if isinstance(x, sp.Matrix) and x.shape[-1] == 1:
            # Vector
            jac = F.jacobian(x)
            # Symbolically compute J_{F}^{-1}*F
            df = jac.inv() * F
        else:
            df = sp.diff(F, x)
            if isinstance(x, sp.Symbol):
                df = F/df
        return df

    def _numpy_tensor_solve(self, f, df, constants):
        """
        Solve the linear system involving the Jacobian, where df has a rank greater than 2.
        """
        if len(constants):
            def _tensor_solve_(X, *Y):
                ndat = X.shape[-1]
                f_X = f(X, *Y)
                # Constants
                C = tuple(Y[i] for i in constants)
                # Other arguments
                Y = np.array(tuple(Y[i] for i in range(Y.shape[0]) if i not in constants))
                df_X = np.array([df(X[..., i], *Y[..., i], *C) for i in range(ndat)]).squeeze()
                res = np.array([np.linalg.tensorsolve(df_X[i, ...], f_X[..., i]) for i in range(ndat)])
                return np.rollaxis(res, 0, len(f_X.shape))
        else:
            def _tensor_solve_(X, *Y):
                ndat = X.shape[-1]
                f_X = f(X, *Y)
                Y = np.array(Y)
                df_X = np.array([df(X[..., i], *Y[..., i]) for i in range(ndat)])
                res = np.array([np.linalg.tensorsolve(df_X[i, ...], f_X[..., i]) for i in range(ndat)])
                return np.rollaxis(res, 0, len(f_X.shape))
        return _tensor_solve_

    def _vectorized_newton(self, func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50, fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
        """
        A vectorized version of Newton, Halley, and secant methods for arrays
        This version is a modification of the 'optimize.newton' function
        from the SciPy library which handles non-scalar cases.
        """
        # Explicitly copy `x0` as `p` will be modified inplace, but, the
        # user's array should not be altered.
        try:
            p = np.array(x0, copy=True, dtype=float)
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

        # if full_output:
        #     result = namedtuple('result', ('root', 'converged', 'zero_der'))
        #     p = result(p, ~failures, zero_der)

        return p

    def _sympy_create_symbols(self, xshape, i, granular=True):
        if len(xshape) == 0:
            return sp.symbols('s_'+str(i))
        elif len(xshape) == 1:
            if not granular:
                return sp.Matrix(sp.MatrixSymbol('V_'+str(i), xshape[0], 1))
            symb = sp.symbols('v'+str(i)+'_:%d' % xshape[0], real=True)
            # sp.Matrix are more flexible for the representation of vector than sp.Array (e.g it enables to use norms)
            return sp.Matrix(symb)
        elif len(xshape) == 2:
            if not granular:
                return sp.Matrix(sp.MatrixSymbol('M_'+str(i), *xshape))
            nm = xshape[0]*xshape[1]
            symb = sp.symbols('m'+str(i)+'_:%d' % nm, real=True)
            coeffs = [symb[i:i+xshape[1]] for i in range(0, nm, xshape[1])]
            return sp.Matrix(coeffs)

    def _sympy_subs_symbols(self, old, new, fprime, shape):
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


class PointnetOperator(AbstractPointwiseOperator):
    r"""A :class:`PointnetOperator`: is an implementation of ExternalOperator that is defined through
    a given neural network model N and whose values correspond to the output of the neural network represented by N.
     """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data, weights_version=None):
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
        if not isinstance(operands[-1], Constant):
            weights_val = ml_get_weights(operator_data['model'], operator_data['framework'])
            cw = Constant(zero(*weights_val.shape))
            # Assign and convert (from torch to numpy)
            cw.dat.data[:] = weights_val
            operands += (cw,)
            # Type exception is caught later
            if isinstance(derivatives, tuple):
                derivatives += (0,)

        AbstractPointwiseOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, operator_data=operator_data)

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

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, count=None, name=None, operator_data=None, val=None, add_kwargs={}):
        "Overwrite _ufl_expr_reconstruct to pass on weights_version"
        add_kwargs['weights_version'] = self._weights_version
        return AbstractPointwiseOperator._ufl_expr_reconstruct_(self, *operands, function_space=function_space,
                                                                derivatives=derivatives, count=count,
                                                                val=val, name=name,
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

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data, weights_version=None):
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

        PointnetOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, operator_data=operator_data, weights_version=weights_version)

        # Set datatype to double (torch.float64) as the firedrake.Function default data type is float64
        self.model.double()  # or torch.set_default_dtype(torch.float64)

    @utils.cached_property
    def ml_backend(self):
        try:
            import torch
        except ImportError:
            raise ImportError("Error when trying to import PyTorch")
        return torch

    def compute_derivatives(self, N, x, model_tape=False):
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
    def evaluate(self, model_tape=False):
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
        res = self.compute_derivatives(val, torch_op, model_tape)

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

    def adjoint_action(self, x, idx):
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
        return AbstractPointwiseOperator.adjoint_action(self, x, idx)

    def _assign_weights(self, weights):
        self.model.weight.data = weights

    def _update_model_weights(self):
        weights_op = self.ml_backend.tensor(self.ufl_operands[-1].dat.data_ro)
        self._assign_weights(weights_op.unsqueeze(0))


class TensorFlowOperator(PointnetOperator):
    r"""A :class:`TensorFlowOperator` ... TODO :
     """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data):
        PointnetOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, operator_data=operator_data)
        raise NotImplementedError('TensorFlowOperator not implemented yet!')


# Helper functions #
def point_expr(point_expr, function_space):
    r"""The point_expr function returns the `PointexprOperator` class initialised with :
        - point_expr : a function expression (e.g. lambda expression)
        - function space
     """
    return partial(PointexprOperator, operator_data=point_expr, function_space=function_space)


def point_solve(point_solve, function_space, solver_name='newton', solver_params=None, disp=False):
    r"""The point_solve function returns the `PointsolveOperator` class initialised with :
        - point_solve : a function expression (e.g. lambda expression)
        - function space
        - solver_name
        - solver_params
        - disp : if you want to display the maximum number of iterations taken over the whol poinwise solves.
    """
    if solver_params is None:
        solver_params = {}
    operator_data = {'point_solve': point_solve, 'solver_name': solver_name, 'solver_params': solver_params}
    if disp not in (True, False):
        disp = False
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
