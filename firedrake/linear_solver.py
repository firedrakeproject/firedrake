from __future__ import absolute_import, print_function, division
import ufl

from firedrake.exceptions import ConvergenceError
import firedrake.function as function
import firedrake.vector as vector
import firedrake.matrix as matrix
import firedrake.solving_utils as solving_utils
from firedrake.petsc import PETSc
from firedrake.slate import slate
from firedrake.utils import cached_property


__all__ = ["LinearSolver"]


class LinearSolver(solving_utils.ParametersMixin):

    def __init__(self, A, P=None, solver_parameters=None,
                 nullspace=None, transpose_nullspace=None,
                 near_nullspace=None, options_prefix=None):
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

        .. note::

          Any boundary conditions for this solve *must* have been
          applied when assembling the operator.
        """
        if not isinstance(A, matrix.MatrixBase):
            raise TypeError("Provided operator is a '%s', not a MatrixBase" % type(A).__name__)
        if P is not None and not isinstance(P, matrix.MatrixBase):
            raise TypeError("Provided preconditioner is a '%s', not a MatrixBase" % type(P).__name__)

        self.A = A
        self.comm = A.comm
        self.P = P if P is not None else A

        # Set up parameters mixin
        super(LinearSolver, self).__init__(solver_parameters, options_prefix)
        # Set some defaults
        self.set_default_parameter("ksp_rtol", "1e-7")
        # If preconditioning matrix is matrix-free, then default to no
        # preconditioning.
        if isinstance(self.P, matrix.ImplicitMatrix):
            self.set_default_parameter("pc_type", "none")
        elif self.P.block_shape != (1, 1):
            # Otherwise, mixed problems default to jacobi.
            self.set_default_parameter("pc_type", "jacobi")

        self.ksp = PETSc.KSP().create(comm=self.comm)

        W = self.A.a.arguments()[0].function_space()
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
        self._W = W
        # Operator setting must come after null space has been
        # applied
        # Force evaluation here
        self.A.force_evaluation()
        self.P.force_evaluation()
        self.ksp.setOperators(A=self.A.petscmat, P=self.P.petscmat)
        # Set from options now (we're not allowed to change parameters
        # anyway).
        self.set_from_options(self.ksp)

    @cached_property
    def _b(self):
        """A function to store the RHS.

        Used in presence of BCs."""
        return function.Function(self._W)

    @cached_property
    def _Abcs(self):
        """A function storing the action of the operator on a zero Function
        satisfying the BCs.

        Used in the presence of BCs.
        """
        b = function.Function(self._W)
        for bc in self.A.bcs:
            bc.apply(b)
        from firedrake.assemble import _assemble
        if isinstance(self.A.a, slate.TensorBase):
            return _assemble(self.A.a * b)
        else:
            return _assemble(ufl.action(self.A.a, b))

    def solve(self, x, b):
        if not isinstance(x, (function.Function, vector.Vector)):
            raise TypeError("Provided solution is a '%s', not a Function or Vector" % type(x).__name__)
        if not isinstance(b, function.Function):
            raise TypeError("Provided RHS is a '%s', not a Function" % type(b).__name__)

        if len(self._W) > 1 and self.nullspace is not None:
            self.nullspace._apply(self._W.dof_dset.field_ises)
        if len(self._W) > 1 and self.transpose_nullspace is not None:
            self.transpose_nullspace._apply(self._W.dof_dset.field_ises, transpose=True)
        if len(self._W) > 1 and self.near_nullspace is not None:
            self.near_nullspace._apply(self._W.dof_dset.field_ises, near=True)
        if self.A.has_bcs:
            b_bc = self._b
            # rhs = b - action(A, zero_function_with_bcs_applied)
            b_bc.assign(b - self._Abcs)
            # Now we need to apply the boundary conditions to the "RHS"
            for bc in self.A.bcs:
                bc.apply(b_bc)
            # don't want to write into b itself, because that would confuse user
            b = b_bc
        with self.inserted_options():
            with b.dat.vec_ro as rhs:
                if self.ksp.getInitialGuessNonzero():
                    acc = x.dat.vec
                else:
                    acc = x.dat.vec_wo
                with acc as solution:
                    self.ksp.solve(rhs, solution)

        r = self.ksp.getConvergedReason()
        if r < 0:
            raise ConvergenceError("LinearSolver failed to converge after %d iterations with reason: %s", self.ksp.getIterationNumber(), solving_utils.KSPReasons[r])
