import ufl

from pyop2.logger import warning, RED
from pyop2.utils import as_tuple
from pyop2.profiling import timed_function, profile

import assemble
import function
import solving
import solving_utils
import ufl_expr
from petsc import PETSc
import firedrake.mg.utils
import firedrake.mg.ufl_utils


__all__ = ["LinearVariationalProblem",
           "LinearVariationalSolver",
           "NonlinearVariationalProblem",
           "NonlinearVariationalSolver"]


class NonlinearVariationalProblem(object):
    """Nonlinear variational problem F(u; v) = 0."""

    def __init__(self, F, u, bcs=None, J=None,
                 Jp=None,
                 form_compiler_parameters=None):
        """
        :param F: the nonlinear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param J: the Jacobian J = dF/du (optional)
        :param Jp: a form used for preconditioning the linear system,
                 optional, if not supplied then the Jacobian itself
                 will be used.
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        """

        # Extract and check arguments
        u = solving._extract_u(u)
        bcs = solving._extract_bcs(bcs)

        # Store input UFL forms and solution Function
        self.F = F
        # Use the user-provided Jacobian. If none is provided, derive
        # the Jacobian from the residual.
        self.J = J or ufl_expr.derivative(F, u)
        self.Jp = Jp
        self.u = u
        self.bcs = bcs

        # Store form compiler parameters
        self.form_compiler_parameters = form_compiler_parameters
        self._constant_jacobian = False


def get_dm(problem):
    return problem.u.function_space()._dm


class _SNESContext(object):
    """
    Context holding information for SNES callbacks.

    :arg problem: a :class:`NonlinearVariationalProblem`.

    The idea here is that the SNES holds a shell DM which contains
    this object as "user context".  When the SNES calls back to the
    user form_function code, we pull the DM out of the SNES and then
    get the context (which is one of these objects) to find the
    Firedrake level information.
    """
    def __init__(self, problems):
        problems = as_tuple(problems)
        self._problems = problems
        # Build the jacobian with the correct sparsity pattern.  Note
        # that since matrix assembly is lazy this doesn't actually
        # force an additional assembly of the matrix since in
        # form_jacobian we call assemble again which drops this
        # computation on the floor.
        self._jacs = tuple(assemble.assemble(problem.J, bcs=problem.bcs,
                                             form_compiler_parameters=problem.form_compiler_parameters)
                           for problem in problems)
        if problems[-1].Jp is not None:
            self._pjacs = tuple(assemble.assemble(problem.Jp, bcs=problem.bcs,
                                                  form_compiler_parameters=problem.form_compiler_parameters)
                                for problem in problems)
        else:
            self._pjacs = self._jacs
        # Function to hold current guess
        self._xs = tuple(function.Function(problem.u) for problem in problems)
        self.Fs = tuple(ufl.replace(problem.F, {problem.u: x}) for problem, x in zip(problems,
                                                                                     self._xs))
        self.Js = tuple(ufl.replace(problem.J, {problem.u: x}) for problem, x in zip(problems,
                                                                                     self._xs))
        if problems[-1].Jp is not None:
            self.Jps = tuple(ufl.replace(problem.Jp, {problem.u: x}) for problem, x in zip(problems,
                                                                                           self._xs))
        else:
            self.Jps = tuple(None for _ in problems)
        self._Fs = tuple(function.Function(F.arguments()[0].function_space())
                         for F in self.Fs)
        self._jacobians_assembled = [False for _ in problems]

    def set_function(self, snes):
        """Set the residual evaluation function"""
        with self._Fs[-1].dat.vec as v:
            snes.setFunction(self.form_function, v)

    def set_jacobian(self, snes):
        snes.setJacobian(self.form_jacobian, J=self._jacs[-1]._M.handle,
                         P=self._pjacs[-1]._M.handle)

    def set_fieldsplits(self, pc):
        test = self.Fs[-1].arguments()[0]
        pmat = self._pjacs[-1]._M
        names = [fs.name if fs.name else str(i)
                 for i, fs in enumerate(test.function_space())]
        return solving_utils.set_fieldsplits(pmat, pc, names=names)

    def set_nullspace(self, nullspace, ises=None):
        nullspace._apply(self._jacs[-1]._M, ises=ises)
        if self.Jps[-1] is not None:
            nullspace._apply(self._pjacs[-1]._M, ises=ises)

    @property
    def is_mixed(self):
        return self._jacs[-1]._M.sparsity.shape != (1, 1)

    @classmethod
    def create_matrix(cls, dm):
        _, lvl = firedrake.mg.utils.get_level(dm.getAttr("__fs__")())
        ctx = dm.getAppCtx()
        return ctx._jacs[lvl]._M.handle

    @classmethod
    def form_function(cls, snes, X, F):
        """Form the residual for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg F: the residual at X (a Vec)
        """
        dm = snes.getDM()
        ctx = dm.getAppCtx()
        _, lvl = firedrake.mg.utils.get_level(dm.getAttr("__fs__")())
        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._xs[lvl].dat.vec as v:
            if v != X:
                X.copy(v)

        assemble.assemble(ctx.Fs[lvl], tensor=ctx._Fs[lvl],
                          form_compiler_parameters=ctx._problems[lvl].form_compiler_parameters)
        for bc in ctx._problems[lvl].bcs:
            bc.zero(ctx._Fs[lvl])

        # F may not be the same vector as self._F, so copy
        # residual out to F.
        with ctx._Fs[lvl].dat.vec_ro as v:
            v.copy(F)

    @classmethod
    def form_jacobian(cls, snes, X, J, P):
        """Form the Jacobian for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        dm = snes.getDM()
        ctx = dm.getAppCtx()
        _, lvl = firedrake.mg.utils.get_level(dm.getAttr("__fs__")())

        if ctx._problems[lvl]._constant_jacobian and ctx._jacobians_assembled[lvl]:
            # Don't need to do any work with a constant jacobian
            # that's already assembled
            return
        ctx._jacobians_assembled[lvl] = True

        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._xs[lvl].dat.vec as v:
            X.copy(v)
        assemble.assemble(ctx.Js[lvl],
                          tensor=ctx._jacs[lvl],
                          bcs=ctx._problems[lvl].bcs,
                          form_compiler_parameters=ctx._problems[lvl].form_compiler_parameters)
        ctx._jacs[lvl].M._force_evaluation()
        if ctx.Jps[lvl] is not None:
            assemble.assemble(ctx.Jps[lvl],
                              tensor=ctx._pjacs[lvl],
                              bcs=ctx._problems[lvl].bcs,
                              form_compiler_parameters=ctx._problems[lvl].form_compiler_parameters)
            ctx._pjacs[lvl].M._force_evaluation()


def _extract_kwargs(**kwargs):
    parameters = kwargs.get('solver_parameters', None)
    if 'parameters' in kwargs:
        warning(RED % "The 'parameters' keyword is deprecated, use 'solver_parameters' instead.")
        parameters = kwargs['parameters']
        if 'solver_parameters' in kwargs:
            warning(RED % "'parameters' and 'solver_parameters' passed, using the latter")
            parameters = kwargs['solver_parameters']

    # Make sure we don't stomp on a dict the user has passed in.
    parameters = parameters.copy() if parameters is not None else {}
    nullspace = kwargs.get('nullspace', None)

    return parameters, nullspace


class NonlinearVariationalSolver(object):
    """Solves a :class:`NonlinearVariationalProblem`."""

    _id = 0

    def __init__(self, problem, **kwargs):
        """
        :arg problem: A :class:`NonlinearVariationalProblem` to solve.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.  For
            example, to set the nonlinear solver type to just use a linear
            solver:

        .. code-block:: python

            {'snes_type': 'ksponly'}

        PETSc flag options should be specified with `bool` values. For example:

        .. code-block:: python

            {'snes_monitor': True}
        """
        assert isinstance(problem, NonlinearVariationalProblem)

        parameters, nullspace = _extract_kwargs(**kwargs)

        ctx = _SNESContext(problem)

        # Don't ask the DM to provide fieldsplit splits
        parameters.setdefault('pc_fieldsplit_dm_splits', False)
        # Mixed problem, use jacobi pc if user has not supplied one.
        if ctx.is_mixed:
            parameters.setdefault('pc_type', 'jacobi')

        self.snes = PETSc.SNES().create()
        self._opt_prefix = 'firedrake_snes_%d_' % NonlinearVariationalSolver._id
        NonlinearVariationalSolver._id += 1
        self.snes.setOptionsPrefix(self._opt_prefix)

        self.parameters = parameters

        self._problem = problem

        self._ctx = ctx
        dm = get_dm(problem)
        self.snes.setDM(dm)

        ctx.set_function(self.snes)
        ctx.set_jacobian(self.snes)
        dm.setCreateMatrix(ctx.create_matrix)
        ises = ctx.set_fieldsplits(self.snes.ksp.pc)
        if nullspace is not None:
            ctx.set_nullspace(nullspace, ises=ises)

    def __del__(self):
        # Remove stuff from the options database
        # It's fixed size, so if we don't it gets too big.
        if hasattr(self, '_opt_prefix'):
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                del opts[self._opt_prefix + k]
            delattr(self, '_opt_prefix')

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        assert isinstance(val, dict), 'Must pass a dict to set parameters'
        self._parameters = val
        solving_utils.update_parameters(self, self.snes)

    @timed_function("SNES solver execution")
    @profile
    def solve(self):
        dm = self.snes.getDM()
        dm.setAppCtx(self._ctx)

        # Apply the boundary conditions to the initial guess.
        for bc in self._problem.bcs:
            bc.apply(self._problem.u)

        # User might have updated parameters dict before calling
        # solve, ensure these are passed through to the snes.
        solving_utils.update_parameters(self, self.snes)

        with self._problem.u.dat.vec as v:
            self.snes.solve(None, v)

        reasons = self.snes.ConvergedReason()
        reasons = dict([(getattr(reasons, r), r)
                        for r in dir(reasons) if not r.startswith('_')])
        r = self.snes.getConvergedReason()
        try:
            reason = reasons[r]
            inner = False
        except KeyError:
            kspreasons = self.snes.getKSP().ConvergedReason()
            kspreasons = dict([(getattr(kspreasons, kr), kr)
                               for kr in dir(kspreasons) if not kr.startswith('_')])
            r = self.snes.getKSP().getConvergedReason()
            try:
                reason = kspreasons[r]
                inner = True
            except KeyError:
                reason = 'unknown reason (petsc4py enum incomplete?)'
        if r < 0:
            if inner:
                msg = "Inner linear solve failed to converge after %d iterations with reason: %s" % \
                      (self.snes.getKSP().getIterationNumber(), reason)
            else:
                msg = reason
            raise RuntimeError("""Nonlinear solve failed to converge after %d nonlinear iterations.
Reason:
   %s""" % (self.snes.getIterationNumber(), msg))


class LinearVariationalProblem(NonlinearVariationalProblem):
    """Linear variational problem a(u, v) = L(v)."""

    def __init__(self, a, L, u, bcs=None, aP=None,
                 form_compiler_parameters=None,
                 constant_jacobian=True):
        """
        :param a: the bilinear form
        :param L: the linear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param aP: an optional operator to assemble to precondition
                 the system (if not provided a preconditioner may be
                 computed from ``a``)
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        :param constant_jacobian: (optional) flag indicating that the
                 Jacobian is constant (i.e. does not depend on
                 varying fields).  If your Jacobian can change, set
                 this flag to :data:`False`.
        """

        # In the linear case, the Jacobian is the equation LHS.
        J = a
        F = ufl.action(J, u) - L

        super(LinearVariationalProblem, self).__init__(F, u, bcs, J, aP,
                                                       form_compiler_parameters=form_compiler_parameters)
        self._constant_jacobian = constant_jacobian


class LinearVariationalSolver(NonlinearVariationalSolver):
    """Solves a :class:`LinearVariationalProblem`."""

    def __init__(self, *args, **kwargs):
        """
        :arg problem: A :class:`LinearVariationalProblem` to solve.
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.
        """
        super(LinearVariationalSolver, self).__init__(*args, **kwargs)

        self.parameters.setdefault('snes_type', 'ksponly')
        self.parameters.setdefault('ksp_rtol', 1.0e-7)
        solving_utils.update_parameters(self, self.snes)
