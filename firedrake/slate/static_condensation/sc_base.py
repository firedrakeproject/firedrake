import abc

from firedrake.preconditioners import PCBase
from pyop2.profiling import timed_region


class SCBase(PCBase):
    """
    """

    @abc.abstractmethod
    def forward_elimination(self, pc, x):
        """
        """

        raise NotImplementedError("Forward elimination not implemented")

    @abc.abstractmethod
    def backward_substitution(self, pc, y):
        """
        """

        raise NotImplementedError("Backward substitution not implemented")

    @abc.abstractmethod
    def sc_solve(self, pc):
        """
        """

        raise NotImplementedError("Solve not implemented")

    def apply(self, pc, x, y):
        """
        """

        with timed_region("SCForwardElim"):
            self.forward_elimination(pc, x)

        with timed_region("SCSolve"):
            self.sc_solve(pc)

        with timed_region("SCBackSub"):
            self.backward_substitution(pc, y)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner."""

        raise NotImplementedError("Transpose application is not implemented.")
