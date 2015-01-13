import function
import solving_utils
from petsc import PETSc


__all__ = ["LinearSolver"]


class LinearSolver(object):

    _id = 0

    def __init__(self, A, P=None, solver_parameters=None,
                 nullspace=None):
        """A linear solver for assembled systems (Ax = b).

        :arg A: a :class:`~.Matrix` (the operator).
        :arg P: an optional :class:`~.Matrix` to construct any
             preconditioner from; if none is supplied :data:`A` is
             used to construct the preconditioner.
        :kwarg parameters: (optional) dict of solver parameters.
        :kwarg nullspace: an optional :class:`~.VectorSpaceBasis` (or
            :class:`~.MixedVectorSpaceBasis` spanning the null space
            of the operator.

        .. note::

          Any boundary conditions for this solve *must* have been
          applied when assembling the operator.
        """
        self.A = A
        self.P = P if P is not None else A

        parameters = solver_parameters.copy() if solver_parameters is not None else {}
        parameters.setdefault("ksp_rtol", "1e-7")

        if self.P._M.sparsity.shape != (1, 1):
            parameters.setdefault('pc_type', 'jacobi')

        self._opt_prefix = "firedrake_ksp_%d_" % LinearSolver._id
        LinearSolver._id += 1

        self.ksp = PETSc.KSP().create()
        self.ksp.setOptionsPrefix(self._opt_prefix)

        self.parameters = parameters

        pc = self.ksp.getPC()

        pmat = self.P._M
        ises = solving_utils.set_fieldsplits(pmat, pc)

        if nullspace is not None:
            nullspace._apply(self.A.M, ises=ises)
            if P is not None:
                nullspace._apply(self.P.M, ises=ises)
        # Operator setting must come after null space has been
        # applied
        # Force evaluation here
        self.ksp.setOperators(A=self.A.M.handle, P=self.P.M.handle)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        assert isinstance(val, dict), "Must pass a dict to set parameters"
        self._parameters = val
        solving_utils.update_parameters(self, self.ksp)

    def __del__(self):
        if hasattr(self, '_opt_prefix'):
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                del opts[self._opt_prefix + k]
            delattr(self, '_opt_prefix')

    def solve(self, x, b):
        # User may have updated parameters
        solving_utils.update_parameters(self, self.ksp)
        if self.A.has_bcs:
            b_bc = function.Function(b.function_space())
            for bc in self.A.bcs:
                bc.apply(b_bc)
            # rhs = b - action(A, b_bc)
            b_bc.assign(b - self.A._form_action(b_bc))
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
            reasons = self.ksp.ConvergedReason()
            reasons = dict([(getattr(reasons, reason), reason)
                            for reason in dir(reasons) if not reason.startswith('_')])
            raise RuntimeError("LinearSolver failed to converge after %d iterations with reason: %s", self.ksp.getIterationNumber(), reasons[r])
