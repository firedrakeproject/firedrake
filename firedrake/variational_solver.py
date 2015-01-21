import ufl

from pyop2.logger import warning, RED
from pyop2.profiling import timed_function, profile

import assemble
import function
import solving
import solving_utils
import ufl_expr
from petsc import PETSc

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


class NonlinearVariationalSolver(object):
    """Solves a :class:`NonlinearVariationalProblem`."""

    _id = 0

    def __init__(self, *args, **kwargs):
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

        .. warning ::

            Since this object contains a circular reference and a
            custom ``__del__`` attribute, you *must* call :meth:`.destroy`
            on it when you are done, otherwise it will never be
            garbage collected.

        """
        assert isinstance(args[0], NonlinearVariationalProblem)
        self._problem = args[0]
        # Build the jacobian with the correct sparsity pattern.  Note
        # that since matrix assembly is lazy this doesn't actually
        # force an additional assembly of the matrix since in
        # form_jacobian we call assemble again which drops this
        # computation on the floor.
        self._jac = assemble.assemble(self._problem.J, bcs=self._problem.bcs,
                                      form_compiler_parameters=self._problem.form_compiler_parameters)
        if self._problem.Jp is not None:
            self._pjac = assemble.assemble(self._problem.Jp, bcs=self._problem.bcs,
                                           form_compiler_parameters=self._problem.form_compiler_parameters)
        else:
            self._pjac = self._jac
        test = self._problem.F.arguments()[0]
        self._F = function.Function(test.function_space())
        # Function to hold current guess
        self._x = function.Function(self._problem.u)
        self._problem.F = ufl.replace(self._problem.F, {self._problem.u: self._x})
        self._problem.J = ufl.replace(self._problem.J, {self._problem.u: self._x})
        if self._problem.Jp is not None:
            self._problem.Jp = ufl.replace(self._problem.Jp, {self._problem.u: self._x})
        self._jacobian_assembled = False
        self.snes = PETSc.SNES().create()
        self._opt_prefix = 'firedrake_snes_%d_' % NonlinearVariationalSolver._id
        NonlinearVariationalSolver._id += 1
        self.snes.setOptionsPrefix(self._opt_prefix)

        parameters = kwargs.get('solver_parameters', None)
        if 'parameters' in kwargs:
            warning(RED % "The 'parameters' keyword to %s is deprecated, use 'solver_parameters' instead.",
                    self.__class__.__name__)
            parameters = kwargs['parameters']
            if 'solver_parameters' in kwargs:
                warning(RED % "'parameters' and 'solver_parameters' passed to %s, using the latter",
                        self.__class__.__name__)
                parameters = kwargs['solver_parameters']

        # Make sure we don't stomp on a dict the user has passed in.
        parameters = parameters.copy() if parameters is not None else {}
        # Mixed problem, use jacobi pc if user has not supplied one.
        if self._jac._M.sparsity.shape != (1, 1):
            parameters.setdefault('pc_type', 'jacobi')

        self.parameters = parameters

        ksp = self.snes.getKSP()
        pc = ksp.getPC()
        pmat = self._pjac._M
        names = [fs.name if fs.name else str(i)
                 for i, fs in enumerate(test.function_space())]

        ises = solving_utils.set_fieldsplits(pmat, pc, names=names)

        with self._F.dat.vec as v:
            self.snes.setFunction(self.form_function, v)
        self.snes.setJacobian(self.form_jacobian, J=self._jac._M.handle,
                              P=self._pjac._M.handle)

        nullspace = kwargs.get('nullspace', None)
        if nullspace is not None:
            self.set_nullspace(nullspace, ises=ises)

    def set_nullspace(self, nullspace, ises=None):
        """Set the null space for this solver.

        :arg nullspace: a :class:`.VectorSpaceBasis` spanning the null
             space of the operator.

        This overwrites any existing null space."""
        nullspace._apply(self._jac._M, ises=ises)
        if self._problem.Jp is not None:
            nullspace._apply(self._pjac._M, ises=ises)

    def form_function(self, snes, X_, F_):
        # X_ may not be the same vector as the vec behind self._x, so
        # copy guess in from X_.
        with self._x.dat.vec as v:
            if v != X_:
                with v as _v, X_ as _x:
                    _v[:] = _x[:]
        assemble.assemble(self._problem.F, tensor=self._F,
                          form_compiler_parameters=self._problem.form_compiler_parameters)
        for bc in self._problem.bcs:
            bc.zero(self._F)

        # F_ may not be the same vector as self._F_tensor, so copy
        # residual out to F_.
        with self._F.dat.vec_ro as v:
            if F_ != v:
                with v as _v, F_ as _f:
                    _f[:] = _v[:]

    def form_jacobian(self, snes, X_, J_, P_):
        if self._problem._constant_jacobian and self._jacobian_assembled:
            # Don't need to do any work with a constant jacobian
            # that's already assembled
            return
        self._jacobian_assembled = True
        # X_ may not be the same vector as the vec behind self._x, so
        # copy guess in from X_.
        with self._x.dat.vec as v:
            if v != X_:
                with v as _v, X_ as _x:
                    _v[:] = _x[:]
        assemble.assemble(self._problem.J,
                          tensor=self._jac,
                          bcs=self._problem.bcs,
                          form_compiler_parameters=self._problem.form_compiler_parameters)
        self._jac.M._force_evaluation()
        if self._problem.Jp is not None:
            assemble.assemble(self._problem.Jp,
                              tensor=self._pjac,
                              bcs=self._problem.bcs,
                              form_compiler_parameters=self._problem.form_compiler_parameters)
            self._pjac.M._force_evaluation()
            return PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    def __del__(self):
        # Remove stuff from the options database
        # It's fixed size, so if we don't it gets too big.
        if hasattr(self, '_opt_prefix'):
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                del opts[self._opt_prefix + k]
            delattr(self, '_opt_prefix')

    def destroy(self):
        """Destroy the SNES object inside the solver.

        You must call this explicitly, because the SNES holds a
        reference to the solver it lives inside, defeating the garbage
        collector."""
        if self.snes is not None:
            self.snes.destroy()
            self.snes = None

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

        .. warning ::

            Since this object contains a circular reference and a
            custom ``__del__`` attribute, you *must* call :meth:`.destroy`
            on it when you are done, otherwise it will never be
            garbage collected.
        """
        super(LinearVariationalSolver, self).__init__(*args, **kwargs)

        self.parameters.setdefault('snes_type', 'ksponly')
        self.parameters.setdefault('ksp_rtol', 1.0e-7)
        solving_utils.update_parameters(self, self.snes)
