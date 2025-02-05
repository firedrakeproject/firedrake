from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.vector import Vector
from firedrake.matrix import MatrixBase
from firedrake.petsc import PETSc
from pyop2.mpi import internal_comm
from firedrake.variational_solver import LinearVariationalProblem, LinearVariationalSolver

__all__ = ["LinearSolver"]


class LinearSolver:

    @PETSc.Log.EventDecorator()
    def __init__(self, A, *, P=None, **kwargs):
        """A linear solver for assembled systems (Ax = b).

        :arg A: a :class:`~.MatrixBase` (the operator).
        :arg P: an optional :class:`~.MatrixBase` to construct any
             preconditioner from; if none is supplied ``A`` is
             used to construct the preconditioner.
        :kwarg parameters: (optional) dict of solver parameters.
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
        :kwarg pre_apply_bcs: If `False`, the problem is linearised
               around the initial guess before imposing the boundary conditions.

        .. note::

          Any boundary conditions for this solve *must* have been
          applied when assembling the operator.
        """
        if not isinstance(A, MatrixBase):
            raise TypeError("Provided operator is a '%s', not a MatrixBase" % type(A).__name__)
        if P is not None and not isinstance(P, MatrixBase):
            raise TypeError("Provided preconditioner is a '%s', not a MatrixBase" % type(P).__name__)

        test, trial = A.a.arguments()
        x = Function(trial.function_space())
        b = Cofunction(test.function_space().dual())

        problem = LinearVariationalProblem(A, b, x, bcs=A.bcs, aP=P)
        solver = LinearVariationalSolver(problem, **kwargs)
        self.b = b
        self.x = x
        self.solver = solver

        self.A = A
        self.comm = A.comm
        self._comm = internal_comm(self.comm, self)
        self.P = P if P is not None else A

        self.ksp = self.solver.snes.ksp
        self.parameters = self.solver.parameters

    @PETSc.Log.EventDecorator()
    def solve(self, x, b):
        """Solve the linear system with RHS ``b`` and store the solution in ``x``.

        Parameters
        ----------
        x : firedrake.function.Function or firedrake.vector.Vector
            Existing Function or Vector to place the solution to the linear system in.
        b : firedrake.cofunction.Cofunction or firedrake.vector.Vector
            A Cofunction or Vector with the right-hand side of the linear system.
        """
        if not isinstance(x, (Function, Vector)):
            raise TypeError(f"Provided solution is a '{type(x).__name__}', not a Function or Vector")
        if isinstance(x, Vector):
            x = x.function
        if not isinstance(b, (Cofunction, Vector)):
            raise TypeError(f"Provided RHS is a '{type(b).__name__}', not a Cofunction or Vector")
        if isinstance(b, Vector):
            b = b.function

        # When solving `Ax = b`, with A: V x U -> R, or equivalently A: V -> U*,
        # we need to make sure that x and b belong to V and U*, respectively.
        if x.function_space() != self.x.function_space():
            raise ValueError(f"x must be a Function in {self.x.function_space()}.")
        if b.function_space() != self.b.function_space():
            raise ValueError(f"b must be a Cofunction in {self.b.function_space()}.")

        self.x.assign(x)
        self.b.assign(b)
        self.solver.solve()
        x.assign(self.x)
