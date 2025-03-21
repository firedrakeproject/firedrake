from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.matrix import MatrixBase
from firedrake.petsc import PETSc
from pyop2.mpi import internal_comm
from firedrake.variational_solver import LinearVariationalProblem, LinearVariationalSolver

__all__ = ["LinearSolver"]


class LinearSolver(LinearVariationalSolver):

    @PETSc.Log.EventDecorator()
    def __init__(self, A, *, P=None, **kwargs):
        """A linear solver for assembled systems (Ax = b) with constant A.

        :arg A: a :class:`~.MatrixBase` (the operator).
        :arg P: an optional :class:`~.MatrixBase` to construct any
             preconditioner from; if none is supplied ``A`` is
             used to construct the preconditioner.
        :kwarg solver_parameters: (optional) dict of solver parameters.
        :kwarg nullspace: an optional :class:`~.VectorSpaceBasis` (or
            :class:`~.MixedVectorSpaceBasis` spanning the null space
            of the operator.
        :kwarg transpose_nullspace: as for the nullspace, but used to
               make the right hand side consistent.
        :kwarg near_nullspace: as for the nullspace, but used to set
               the near nullpace.
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.
        :kwarg pre_apply_bcs: If `True`, the bcs are applied before the solve.
               Otherwise, the bcs are included as part of the linear system.

        .. note::

          Any boundary conditions for this solve *must* have been
          applied when assembling the operator.
        """
        if not isinstance(A, MatrixBase):
            raise TypeError("Provided operator is a '%s', not a MatrixBase" % type(A).__name__)
        if P is not None and not isinstance(P, MatrixBase):
            raise TypeError("Provided preconditioner is a '%s', not a MatrixBase" % type(P).__name__)

        test, trial = A.arguments()
        self.x = Function(trial.function_space())
        self.b = Cofunction(test.function_space().dual())

        problem = LinearVariationalProblem(A, self.b, self.x, aP=P,
                                           form_compiler_parameters=A.form_compiler_parameters,
                                           constant_jacobian=True)
        super().__init__(problem, **kwargs)

        self.A = A
        self.comm = A.comm
        self._comm = internal_comm(self.comm, self)
        self.P = P if P is not None else A

        self.ksp = self.snes.ksp

    @PETSc.Log.EventDecorator()
    def solve(self, x, b):
        """Solve the linear system with RHS ``b`` and store the solution in ``x``.

        Parameters
        ----------
        x : firedrake.function.Function
            A Function to place the solution to the linear system in.
        b : firedrake.cofunction.Cofunction
            A Cofunction with the right-hand side of the linear system.
        """
        if not isinstance(x, Function):
            raise TypeError(f"Provided solution is a '{type(x).__name__}', not a Function")
        if not isinstance(b, Cofunction):
            raise TypeError(f"Provided RHS is a '{type(b).__name__}', not a Cofunction")

        # When solving `Ax = b`, with A: V x U -> R, or equivalently A: V -> U*,
        # we need to make sure that x and b belong to V and U*, respectively.
        if x.function_space() != self.x.function_space():
            raise ValueError(f"x must be a Function in {self.x.function_space()}.")
        if b.function_space() != self.b.function_space():
            raise ValueError(f"b must be a Cofunction in {self.b.function_space()}.")

        self.b.assign(b)
        if self.ksp.getInitialGuessNonzero():
            self.x.assign(x)
        else:
            self.x.zero()
        try:
            super().solve()
        finally:
            # Update x even when ConvergenceError is raised
            x.assign(self.x)
