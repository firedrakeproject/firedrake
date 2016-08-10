from __future__ import absolute_import
import ufl
import firedrake.function as function
import firedrake.vector as vector
import firedrake.matrix as matrix
import firedrake.solving_utils as solving_utils
from firedrake.petsc import PETSc
from firedrake.utils import cached_property


__all__ = ["LinearSolver"]


class LinearSolver(object):

    _id = 0

    def __init__(self, A, P=None, solver_parameters=None,
                 nullspace=None, transpose_nullspace=None,
                 options_prefix=None):
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
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.

        .. note::

          Any boundary conditions for this solve *must* have been
          applied when assembling the operator.
        """
        # Do this first so __del__ doesn't barf horribly if we get an
        # error in __init__
        if options_prefix is not None:
            self._opt_prefix = options_prefix
            self._auto_prefix = False
        else:
            self._opt_prefix = "firedrake_ksp_%d_" % LinearSolver._id
            self._auto_prefix = True
            LinearSolver._id += 1

        if not isinstance(A, matrix.MatrixBase):
            raise TypeError("Provided operator is a '%s', not an MatrixBase" % type(A).__name__)
        if P is not None and not isinstance(P, matrix.MatrixBase):
            raise TypeError("Provided preconditioner is a '%s', not an MatrixBase" % type(P).__name__)

        self.A = A
        self.comm = A.comm
        self.P = P if P is not None else A

        parameters = solver_parameters.copy() if solver_parameters is not None else {}
        parameters.setdefault("ksp_rtol", "1e-7")

        # Disabled: don't refer to the ._M, plus this won't work for saddle point systems!
        # if self.P._M.sparsity.shape != (1, 1):
        #     parameters.setdefault('pc_type', 'jacobi')

        self.ksp = PETSc.KSP().create(comm=self.comm)
        self.ksp.setOptionsPrefix(self._opt_prefix)

        # Allow command-line arguments to override dict parameters
        opts = PETSc.Options()
        for k, v in opts.getAll().iteritems():
            if k.startswith(self._opt_prefix):
                parameters[k[len(self._opt_prefix):]] = v

        self.parameters = parameters

        W = self.A.a.arguments()[0].function_space()
        # DM provides fieldsplits (but not operators)
        self.ksp.setDM(W._dm)
        self.ksp.setDMActive(False)

        if nullspace is not None:
            nullspace._apply(self.A)
            if P is not None:
                nullspace._apply(self.P)

        if transpose_nullspace is not None:
            transpose_nullspace._apply(self.A, transpose=True)
            if P is not None:
                transpose_nullspace._apply(self.P, transpose=True)

        self.nullspace = nullspace
        self.transpose_nullspace = transpose_nullspace
        self._W = W
        # Operator setting must come after null space has been
        # applied
        self.A.force_evaluation()
        self.P.force_evaluation()
        self.ksp.setOperators(A=self.A.PETScMatHandle, P=self.P.PETScMatHandle)

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
        return _assemble(ufl.action(self.A.a, b))

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        assert isinstance(val, dict), "Must pass a dict to set parameters"
        self._parameters = val
        solving_utils.update_parameters(self, self.ksp)

    def __del__(self):
        if self._auto_prefix and hasattr(self, '_opt_prefix'):
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                del opts[self._opt_prefix + k]
            delattr(self, '_opt_prefix')

    def solve(self, x, b):
        if not isinstance(x, (function.Function, vector.Vector)):
            raise TypeError("Provided solution is a '%s', not a Function or Vector" % type(x).__name__)
        if not isinstance(b, function.Function):
            raise TypeError("Provided RHS is a '%s', not a Function" % type(b).__name__)

        if len(self._W) > 1 and self.nullspace is not None:
            self.nullspace._apply(self._W.dof_dset.field_ises)
        if len(self._W) > 1 and self.transpose_nullspace is not None:
            self.transpose_nullspace._apply(self._W.dof_dset.field_ises, transpose=True)
        # User may have updated parameters
        solving_utils.update_parameters(self, self.ksp)
        if self.A.has_bcs:
            b_bc = self._b
            # rhs = b - action(A, zero_function_with_bcs_applied)
            b_bc.assign(b - self._Abcs)
            # Now we need to apply the boundary conditions to the "RHS"
            for bc in self.A.bcs:
                bc.apply(b_bc)
            # don't want to write into b itself, because that would confuse user
            b = b_bc
        with b.dat.vec_ro as rhs:
            with x.dat.vec as solution:
                self.ksp.solve(rhs, solution)

        r = self.ksp.getConvergedReason()
        if r < 0:
            raise RuntimeError("LinearSolver failed to converge after %d iterations with reason: %s", self.ksp.getIterationNumber(), solving_utils.KSPReasons[r])
