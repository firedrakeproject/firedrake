from abc import ABCMeta, abstractmethod
from functools import partial
import types
from ufl.core.external_operator import ExternalOperator
from ufl.core.expr import Expr
from ufl.algorithms.apply_derivatives import VariableRuleset
from ufl.constantvalue import as_ufl
from firedrake.function import Function
from firedrake import utils, functionspaceimpl
from pyop2.datatypes import ScalarType
from ufl.log import error
import sympy as sp
import numpy as np
from scipy import optimize


class AbstractPointwiseOperator(Function, ExternalOperator, metaclass=ABCMeta):
    r"""Asbtract base class from wich stem from all the firedrake practical implementation of the
    ExternalOperator, i.e. all the ExternalOperator subclasses that have mechanisms to be
    evaluated pointwise and to provide their own derivatives.
    This class inherits from firedrake.function.Function and ufl.core.external_operator.ExternalOperator
    Every subclass based on this class must provide the `compute_derivatives` and 'evaluate' methods.
    """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data=None, extop_id=None):
        ExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, extop_id=extop_id)
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

        if extop_id is not None:
            self.extop_id = extop_id

    @abstractmethod
    def compute_derivatives(self):
        """apply the derivatives on operator_data"""

    @abstractmethod
    def evaluate(self):
        """define the evaluation method for the ExternalOperator object"""

    @utils.cached_property
    def _split(self):
        return tuple(Function(V, val) for (V, val) in zip(self.function_space(), self.topological.split()))

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, count=None, name=None, operator_data=None, extop_id=None):
        "Return a new object of the same type with new operands."
        deriv_multiindex = derivatives or self.derivatives

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            corresponding_count = None
            e_master = self._extop_master
            for ext in e_master._extop_dependencies:
                if ext.derivatives == deriv_multiindex:
                    return ext._ufl_expr_reconstruct_(*operands, function_space=function_space,
                                                      derivatives=deriv_multiindex, count=count, name=name,
                                                      operator_data=operator_data)
        else:
            corresponding_count = self._count

        reconstruct_op = type(self)(*operands, function_space=function_space or self._ufl_function_space,
                                    derivatives=deriv_multiindex,
                                    count=corresponding_count,
                                    name=name or self.name(),
                                    operator_data=operator_data or self.operator_data)

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            self._extop_master._extop_dependencies.append(reconstruct_op)
            reconstruct_op._extop_master = self._extop_master
        else:
            reconstruct_op._extop_master = self._extop_master
            reconstruct_op._extop_dependencies = self._extop_dependencies
        return reconstruct_op

    def _reconstruct_extop_id(self, position, name_form):
        # self._extop_dependencies contains self as well
        for e in self._extop_dependencies:
            e._ufl_expr_reconstruct(*self.ufl_operands, extop_id={name_form: position})

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

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data, extop_id=None):
        r"""
        :param operands: operands on which act the :class:`PointexrOperator`.
        :param function_space: the :class:`.FunctionSpace`,
        or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        Alternatively, another :class:`Function` may be passed here and its function space
        will be used to build this :class:`Function`.  In this case, the function values are copied.
        :param derivatives: tuple scecifiying the derivative multiindex.
        :param val: NumPy array-like (or :class:`pyop2.Dat`) providing initial values (optional).
            If val is an existing :class:`Function`, then the data will be shared.
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to ``ScalarType``).
        :param extop_id: dictionary that store the position of the :class:`PointexrOperator` in the forms where it turns up.
        :param operator_data: dictionary containing the function defining how to evaluate the :class:`PointexprOperator`.
        """

        AbstractPointwiseOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, operator_data=operator_data, extop_id=extop_id)

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

    In 1d, it uses scipy.optimize.newton, therefore it has the same syntax for the choice of the parameters
    and the same default values :
    newton_params = {'fprime':None, 'args':(), 'tol':1.48e-08, 'maxiter':50, 'fprime2':None, 'x1':None,
                    'rtol':0.0, 'full_output':False, 'disp':True}
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
     """

    def __init__(self, *operands, function_space, derivatives=None, count=None, val=None, name=None, dtype=ScalarType, operator_data, extop_id=None, disp=False):
        r"""
        :param operands: operands on which act the :class:`PointsolveOperator`.
        :param function_space: the :class:`.FunctionSpace`,
        or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        Alternatively, another :class:`Function` may be passed here and its function space
        will be used to build this :class:`Function`.  In this case, the function values are copied.
        :param derivatives: tuple scecifiying the derivative multiindex.
        :param val: NumPy array-like (or :class:`pyop2.Dat`) providing initial values (optional).
            If val is an existing :class:`Function`, then the data will be shared.
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to ``ScalarType``).
        :param extop_id: dictionary that store the position of the :class:`PointsolveOperator` in the forms where it turns up.
        :param operator_data: dictionary containing the:
                - function defining how to evaluate the :class:`PointsolveOperator`
                - solver_name
                - solver_parameters:
                            + x0: initial condition
                            + fprime: gradient of the function defining the :class:`PointsolveOperator`
                            + maxiter: max number of iterations
                            + tol: tolerance
                  More parameters are available for the 1d case where we use scipy.optimize.newton. If a more
                  precise or efficient solver is needed for the pointwise solves, you can subclass the `solver` method.
        :param disp: boolean indication whether we display the max of the number of iterations taken over the pointwise solves.
        """

        AbstractPointwiseOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, count=count, val=val, name=name, dtype=dtype, operator_data=operator_data, extop_id=extop_id)

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
        symb = (create_symbols(self.ufl_function_space().shape, 0),)
        symb += tuple(create_symbols(e.ufl_shape, i+1) for i, e in enumerate(self.ufl_operands))
        ufl_space = self.ufl_function_space()
        ufl_shape = self.ufl_shape

        if deriv_index == (0,)*len(deriv_index):
            return f

        fexpr = self.operator_f(*symb)
        args = tuple(Function(ufl_space).interpolate(pi) for pi in self.ufl_operands)
        vals = tuple(coeff.dat.data for coeff in args)
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
                    dfds0l = sp.lambdify(symb, dfds0, modules='numpy')
                    dfdsil = sp.lambdify(symb, dfdsi, modules='numpy')

                    if di > 1:
                        d2fds0 = sp.diff(fexpr, symb[0], 2)
                        d2fdsi = sp.diff(fexpr, symb[i], 2)

                        d2fds0l = sp.lambdify(symb, d2fds0, modules='numpy')
                        d2fdsil = sp.lambdify(symb, d2fdsi, modules='numpy')
                        if di == 3:
                            d3fds0 = sp.diff(fexpr, symb[0], 3)
                            d3fdsi = sp.diff(fexpr, symb[i], 3)

                            d3fds0l = sp.lambdify(symb, d3fds0, modules='numpy')
                            d3fdsil = sp.lambdify(symb, d3fdsi, modules='numpy')
                        else:
                            # The implicit differentiation order can be extended if needed
                            error("Implicit differentiation of order n is not handled for n>3")

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
                        fj = f.dat.data[j]
                        val_ops = tuple(v[j] for v in vals)
                        A = dfds0l(fj.flatten(), *(voj.flatten() for voj in val_ops))
                        B = dfdsil(fj.flatten(), *(voj.flatten() for voj in val_ops))

                        # Conversion to numpy array and squeezing
                        A = np.squeeze(A)
                        B = np.squeeze(B)

                        # ImpDiff : df/dy * dy/dx = df/dx
                        C = -B
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
                    error("Cross-derivatives not handled : %s" % deriv_index)  # Needed feature ?
                break
        return self

    def evaluate(self):
        ufl_space = self.ufl_function_space()
        solver_params = self.solver_params.copy()  # To avoid breaking immutability
        nn = len(self.dat.data)
        f = self.operator_f

        # If we want to evaluate derivative, we first have to evaluate the solution and then use implicit differentiation
        if self.derivatives != (0,)*len(self.derivatives):
            e_master = self._extop_master
            xstar = e_master.evaluate()
            return self.compute_derivatives(xstar)

        # Symbols construction
        symb = (create_symbols(ufl_space.shape, 0),)
        symb += tuple(create_symbols(e.ufl_shape, i+1) for i, e in enumerate(self.ufl_operands))

        # Pre-processing to get the values of the initial guesses
        if 'x0' in solver_params.keys() and isinstance(solver_params['x0'], Expr):
            solver_params_x0 = Function(ufl_space).interpolate(solver_params['x0']).dat.data
        else:
            solver_params_x0 = self.dat.data
        if 'x1' in solver_params.keys() and isinstance(solver_params['x1'], Expr):
            solver_params_x1 = Function(ufl_space).interpolate(solver_params['x1']).dat.data
        else:
            solver_params_x1 = nn*[None]  # To avoid a "if" in the loop over the dofs

        # Newton
        args = tuple(Function(ufl_space).interpolate(pi) for pi in self.ufl_operands)
        vals = tuple(coeff.dat.data for coeff in args)
        pointwise_vals = tuple(dict(zip(f.__code__.co_varnames[1:], tuple(v[i] for v in vals))) for i in range(nn))

        solver_params_fprime = []
        solver_params_fprime2 = nn*[None]  # To avoid to have a if in the loop over the dofs

        # Computation of the jacobian
        if self.solver_name in ('newton', 'halley'):
            if 'fprime' not in solver_params.keys():
                fprime = sp.diff(f(*symb), symb[0])
                gprime = sp.lambdify(symb, fprime, modules='numpy')
            else:
                gprime = solver_params['fprime']
            for i in range(nn):
                def gprime_reshaped(x):
                    xx = x.flatten()
                    dg = partial(gprime, **dict(zip(gprime.__code__.co_varnames[1:], [v[i].flatten() for v in vals])))
                    return np.squeeze(dg(xx))
                solver_params_fprime += [gprime_reshaped]

        # Computation of the hessian
        if self.solver_name == 'halley':
            if 'fprime2' not in solver_params.keys():
                fprime2 = sp.diff(f(*symb), symb[0], symb[0])
                gprime2 = sp.lambdify(symb, fprime2, modules='numpy')
            else:
                gprime2 = solver_params['fprime2']
            solver_params_fprime2 = []
            for i in range(nn):
                def gprime2_reshaped(x):
                    xx = x.flatten()
                    d2g = partial(gprime2, **dict(zip(gprime2.__code__.co_varnames[1:], [v[i].flatten() for v in vals])))
                    return np.squeeze(d2g(xx))
                solver_params_fprime2 += [gprime2_reshaped]

        # Reshaping of the input/output of the function that is going to be optimised
        def f_lambdified_reshaped(x, **kwargs):
            func_args = (x,) + tuple(kwargs.values())
            xx = [e.flatten() for e in func_args]
            f_lambd = sp.lambdify(symb, f(*symb), modules='numpy')
            return np.squeeze(f_lambd(*xx))
        fl = f_lambdified_reshaped

        # Pointwise solver over the dofs
        glob_iter_counter = []
        for i in range(nn):
            solver_params['fprime'] = solver_params_fprime[i]
            solver_params['fprime2'] = solver_params_fprime2[i]
            g = partial(fl, **pointwise_vals[i])
            self.dat.data[i], iter_counter = self.solver(g, solver_params_x0[i], solver_params_x1[i], **solver_params)
            glob_iter_counter += [iter_counter]
        if self.disp:
            max_iter_counter = max(glob_iter_counter)+1
            print("\n The maximum number of iterations taken by the PointSolveOperator.solver is : %d " % max_iter_counter)
        return self

    def solver(self, *args, **kwargs):
        """
        Provide the solver : When the variable is a scalar, the optimize.newton function from scipy is used (it only handles the scalar case). In higher-dimensions, we provide an implementation of the Newton method.
        This method can be overwritten to provide a more appropriate method for the pointwise solving.
        """
        g = args[0]
        x0 = args[1]
        x1 = args[2]
        # To avoid having two x0, x1 arguments, the None is to avoid a KeyError
        kwargs.pop('x0', None)
        kwargs.pop('x1', None)
        if len(self.ufl_shape) == 0:
            res, out = optimize.newton(g, **kwargs, x0=x0, x1=x1, full_output=True)
            return res, out.iterations
        else:
            def newton_generalized(F, dF, x, tol, maxiter):
                """
                Solve nonlinear system F=0 by Newton's method.
                dF : Jacobian of F
                x : start value
                Stopping criterion :  ||F||_{2} < eps.
                """
                it = 0
                Fx = F(x)
                normF = np.linalg.norm(Fx, ord=2)
                while abs(normF) > tol and it < maxiter:
                    delta = np.linalg.tensorsolve(dF(x), -Fx)  # tensorsolve is consistent with the case where x is a vector
                    x = x + delta
                    Fx = F(x)
                    normF = np.linalg.norm(Fx, ord=2)
                    it += 1

                if abs(normF) > tol:
                    it = -1
                return x, it
            tol = kwargs.get('tol') or 1.48e-08
            maxiter = kwargs.get('maxiter') or 50
            jacobian = kwargs.get('fprime')
            return newton_generalized(g, jacobian, x0, tol, maxiter)

# Neural Net bit : Here !


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

# Neural Net bit 2 : Here !


def create_symbols(xshape, i):
    if len(xshape) == 0:
        return sp.symbols('s_'+str(i))
    elif len(xshape) == 1:
        symb = sp.symbols('v'+str(i)+'_:%d' % xshape[0], real=True)
        # sp.Matrix are more flexible for the representation of vector than sp.Array (e.g enables to use norms)
        return sp.Matrix(symb)
    elif len(xshape) == 2:
        nm = xshape[0]*xshape[1]
        symb = sp.symbols('m'+str(i)+'_:%d' % nm, real=True)
        coeffs = [symb[i:i+xshape[1]] for i in range(0, nm, xshape[1])]
        return sp.Matrix(coeffs)
