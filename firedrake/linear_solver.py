from firedrake.exceptions import ConvergenceError
import firedrake.function as function
import firedrake.cofunction as cofunction
import firedrake.vector as vector
import firedrake.matrix as matrix
import firedrake.solving_utils as solving_utils
from firedrake import dmhooks
from firedrake.petsc import PETSc, OptionsManager, flatten_parameters
from firedrake.utils import cached_property
from firedrake.ufl_expr import action
from pyop2.mpi import internal_comm

__all__ = ["LinearSolver"]


class LinearSolver(OptionsManager):

    DEFAULT_KSP_PARAMETERS = solving_utils.DEFAULT_KSP_PARAMETERS

    @PETSc.Log.EventDecorator()
    def __init__(self, A, *, P=None, solver_parameters=None,
                 nullspace=None, transpose_nullspace=None,
                 near_nullspace=None, options_prefix=None,
                 pre_apply_bcs=True):
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
        :kwarg pre_apply_bcs: Ignored by this class.

        .. note::

          Any boundary conditions for this solve *must* have been
          applied when assembling the operator.
        """
        if not isinstance(A, matrix.MatrixBase):
            raise TypeError("Provided operator is a '%s', not a MatrixBase" % type(A).__name__)
        if P is not None and not isinstance(P, matrix.MatrixBase):
            raise TypeError("Provided preconditioner is a '%s', not a MatrixBase" % type(P).__name__)

        solver_parameters = flatten_parameters(solver_parameters or {})
        solver_parameters = solving_utils.set_defaults(solver_parameters,
                                                       A.arguments(),
                                                       ksp_defaults=self.DEFAULT_KSP_PARAMETERS)
        self.A = A
        self.comm = A.comm
        self._comm = internal_comm(self.comm, self)
        self.P = P if P is not None else A

        # Set up parameters mixin
        super().__init__(solver_parameters, options_prefix)

        self.A.petscmat.setOptionsPrefix(self.options_prefix)
        self.P.petscmat.setOptionsPrefix(self.options_prefix)

        # If preconditioning matrix is matrix-free, then default to jacobi
        if isinstance(self.P, matrix.ImplicitMatrix):
            self.set_default_parameter("pc_type", "jacobi")

        self.ksp = PETSc.KSP().create(comm=self._comm)

        W = self.test_space
        # DM provides fieldsplits (but not operators)
        self.ksp.setDM(W.dm)
        self.ksp.setDMActive(False)

        if nullspace is not None:
            nullspace._apply(self.A)
            if P is not None:
                nullspace._apply(self.P)

        if transpose_nullspace is not None:
            transpose_nullspace._apply(self.A, transpose=True)
            if P is not None:
                transpose_nullspace._apply(self.P, transpose=True)

        if near_nullspace is not None:
            near_nullspace._apply(self.A, near=True)
            if P is not None:
                near_nullspace._apply(self.P, near=True)

        self.nullspace = nullspace
        self.transpose_nullspace = transpose_nullspace
        self.near_nullspace = near_nullspace
        # Operator setting must come after null space has been
        # applied
        self.ksp.setOperators(A=self.A.petscmat, P=self.P.petscmat)
        # Set from options now (we're not allowed to change parameters
        # anyway).
        self.set_from_options(self.ksp)

    @cached_property
    def test_space(self):
        return self.A.arguments()[0].function_space()

    @cached_property
    def trial_space(self):
        return self.A.arguments()[1].function_space()

    @cached_property
    def _ulift(self):
        return function.Function(self.trial_space)

    @cached_property
    def _blift(self):
        return cofunction.Cofunction(self.test_space.dual())

    @cached_property
    def _update_rhs(self):
        from firedrake.assemble import get_assembler
        expr = -action(self.A.a, self._ulift)
        # needs_zeroing = False adds -A * ulift to the exisiting value of the output tensor
        # zero_bc_nodes = True writes the BC data on the boundary nodes of the output tensor
        return get_assembler(expr, bcs=self.A.bcs, zero_bc_nodes=False, needs_zeroing=False).assemble

    def _lifted(self, b):
        self._ulift.dat.zero()
        for bc in self.A.bcs:
            bc.apply(self._ulift)
        # Copy the input to the internal Cofunction so we do not modify the input
        self._blift.assign(b)
        self._update_rhs(tensor=self._blift)
        # self._blift is now b - A u_bc, and satisfies the boundary conditions
        return self._blift

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
        if not isinstance(x, (function.Function, vector.Vector)):
            raise TypeError(f"Provided solution is a '{type(x).__name__}', not a Function or Vector")
        if isinstance(x, vector.Vector):
            x = x.function
        if not isinstance(b, (cofunction.Cofunction, vector.Vector)):
            raise TypeError(f"Provided RHS is a '{type(b).__name__}', not a Cofunction or Vector")
        if isinstance(b, vector.Vector):
            b = b.function

        # When solving `Ax = b`, with A: V x U -> R, or equivalently A: V -> U*,
        # we need to make sure that x and b belong to V and U*, respectively.
        if x.function_space() != self.trial_space:
            raise ValueError(f"x must be a Function in {self.trial_space}.")
        if b.function_space() != self.test_space.dual():
            raise ValueError(f"b must be a Cofunction in {self.test_space.dual()}.")

        if len(self.trial_space) > 1 and self.nullspace is not None:
            self.nullspace._apply(self.trial_space.dof_dset.field_ises)
        if len(self.test_space) > 1 and self.transpose_nullspace is not None:
            self.transpose_nullspace._apply(self.test_space.dof_dset.field_ises,
                                            transpose=True)
        if len(self.trial_space) > 1 and self.near_nullspace is not None:
            self.near_nullspace._apply(self.trial_space.dof_dset.field_ises, near=True)

        if self.A.has_bcs:
            b = self._lifted(b)

        if self.ksp.getInitialGuessNonzero():
            acc = x.dat.vec
        else:
            acc = x.dat.vec_wo

        with self.inserted_options(), b.dat.vec_ro as rhs, acc as solution, dmhooks.add_hooks(self.ksp.dm, self):
            self.ksp.solve(rhs, solution)

        r = self.ksp.getConvergedReason()
        if r < 0:
            raise ConvergenceError("LinearSolver failed to converge after %d iterations with reason: %s", self.ksp.getIterationNumber(), solving_utils.KSPReasons[r])

        # Grab the comm associated with `x` and call PETSc's garbage cleanup routine
        comm = x.function_space().mesh()._comm
        PETSc.garbage_cleanup(comm=comm)
