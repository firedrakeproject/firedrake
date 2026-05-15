import abc

from firedrake.preconditioners import PCBase
from firedrake.petsc import PETSc


class SCBase(PCBase):
    """A general-purpose base class for static condensation
    interfaces.
    """

    @abc.abstractmethod
    def forward_elimination(self, pc, x):
        """Perform the forward elimination of fields and
        provide the reduced right-hand side for the condensed
        system.

        :arg pc: a Preconditioner instance.
        :arg x: a PETSc vector containing the incoming right-hand side.
        """

        raise NotImplementedError("Forward elimination not implemented")

    @abc.abstractmethod
    def backward_substitution(self, pc, y):
        """Perform the backwards recovery of eliminated fields.

        :arg pc: a Preconditioner instance.
        :arg y: a PETSc vector for placing the resulting fields.
        """

        raise NotImplementedError("Backward substitution not implemented")

    @abc.abstractmethod
    def sc_solve(self, pc):
        """Solve the condensed linear system for the
        condensed field.

        :arg pc: a Preconditioner instance.
        """

        raise NotImplementedError("Solve not implemented")

    def apply(self, pc, x, y):
        """Applies the static condensation preconditioner.

        :arg pc: a Preconditioner instance.
        :arg x: A PETSc vector containing the incoming right-hand side.
        :arg y: A PETSc vector for the result.
        """

        with PETSc.Log.Event("SCForwardElim"):
            self.forward_elimination(pc, x)

        with PETSc.Log.Event("SCSolve"):
            self.sc_solve(pc)

        with PETSc.Log.Event("SCBackSub"):
            self.backward_substitution(pc, y)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner."""

        raise NotImplementedError("Transpose application is not implemented.")
