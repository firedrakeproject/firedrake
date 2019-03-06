import abc

from firedrake_citations import Citations
from firedrake.petsc import PETSc

__all__ = ("PCBase", "SNESBase", "PCSNESBase")


class PCSNESBase(object, metaclass=abc.ABCMeta):

    needs_python_amat = False
    """Set this to True if the A matrix needs to be Python (matfree)."""

    needs_python_pmat = True
    """Set this to False if the P matrix needs to be Python (matfree).

    If the preconditioner also works with assembled matrices, then use False here.
    """

    def __init__(self):
        """Create a PC context suitable for PETSc.

        Matrix free preconditioners should inherit from this class and
        implement:

        - :meth:`initialize`
        - :meth:`update`
        - :meth:`apply`
        - :meth:`applyTranspose`

        """
        Citations().register("Kirby2017")
        self.initialized = False
        super(PCSNESBase, self).__init__()

    @abc.abstractmethod
    def update(self, pc):
        """Update any state in this preconditioner."""
        pass

    @abc.abstractmethod
    def initialize(self, pc):
        """Initialize any state in this preconditioner."""
        pass

    def setUp(self, pc):
        """Setup method called by PETSc.

        Subclasses should probably not override this and instead
        implement :meth:`update` and :meth:`initialize`."""
        A, P = pc.getOperators()
        Atype = A.getType()
        Ptype = P.getType()

        if self.needs_python_amat and Atype != PETSc.Mat.Type.PYTHON:
            raise ValueError("PC needs amat to have type python, but it is %s" % Atype)
        if self.needs_python_pmat and Ptype != PETSc.Mat.Type.PYTHON:
            raise ValueError("PC needs pmat to have type python, but it is %s" % Ptype)

        if self.initialized:
            self.update(pc)
        else:
            self.initialize(pc)
            self.initialized = True

    def view(self, pc, viewer=None):
        if viewer is None:
            return
        typ = viewer.getType()
        if typ != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Firedrake custom %s %s\n" %
                           (self._asciiname, type(self).__name__))

    @staticmethod
    def get_appctx(pc):
        from firedrake.dmhooks import get_appctx
        return get_appctx(pc.getDM()).appctx


class PCBase(PCSNESBase):

    _asciiname = "preconditioner"
    _objectname = "pc"

    @abc.abstractmethod
    def apply(self, pc, X, Y):
        """Apply the preconditioner to X, putting the result in Y.

        Both X and Y are PETSc Vecs, Y is not guaranteed to be zero on entry.
        """
        pass

    @abc.abstractmethod
    def applyTranspose(self, pc, X, Y):
        """Apply the transpose of the preconditioner to X, putting the result in Y.

        Both X and Y are PETSc Vecs, Y is not guaranteed to be zero on entry.

        """
        pass


class SNESBase(PCSNESBase):

    _asciiname = "nonlinear solver"
    _objectname = "snes"
