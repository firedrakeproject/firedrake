import abc

from firedrake_citations import Citations
from firedrake.petsc import PETSc

__all__ = ("PCBase", "SNESBase")


class PCBase(object, metaclass=abc.ABCMeta):

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
        super(PCBase, self).__init__()

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
        viewer.printfASCII("Firedrake matrix-free preconditioner %s\n" %
                           type(self).__name__)

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

    @staticmethod
    def get_appctx(pc):
        from firedrake.dmhooks import get_appctx
        return get_appctx(pc.getDM()).appctx

class SNESBase(object, metaclass=abc.ABCMeta):

    def __init__(self):
        """Create a SNES context suitable for PETSc.

        Custom nonlinear solvers should inherit from this class and
        implement:

        - :meth:`initialize`
        - :meth:`update`
        - :meth:`solve`

        """
        self.initialized = False
        super(SNESBase, self).__init__()

    @abc.abstractmethod
    def update(self, snes):
        """Update any state in this preconditioner."""
        pass

    @abc.abstractmethod
    def initialize(self, snes):
        """Initialize any state in this preconditioner."""
        pass

    def setUp(self, snes):
        """Setup method called by PETSc.

        Subclasses should probably not override this and instead
        implement :meth:`update` and :meth:`initialize`."""
        if self.initialized:
            self.update(snes)
        else:
            self.initialize(snes)
            self.initialized = True

    def view(self, snes, viewer=None):
        if viewer is None:
            return
        typ = viewer.getType()
        if typ != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Firedrake custom nonlinear solver %s\n" %
                           type(self).__name__)

    @abc.abstractmethod
    def solve(self, snes, B, X):
        """Approximately solve F(X) = B.

        Both B and X are PETSc Vecs, B is often not zero.
        """
        pass

    @staticmethod
    def get_appctx(snes):
        from firedrake.dmhooks import get_appctx
        return get_appctx(snes.getDM()).appctx
