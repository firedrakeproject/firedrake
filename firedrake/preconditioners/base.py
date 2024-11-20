import abc

from firedrake_citations import Citations
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx

__all__ = ("PCBase", "SNESBase", "PCSNESBase")


class PCSNESBase(object, metaclass=abc.ABCMeta):
    def __init__(self):
        """Create a PC context suitable for PETSc.

        Matrix free preconditioners should inherit from this class and
        implement:

        - :meth:`~.PCSNESBase.initialize`
        - :meth:`~.PCSNESBase.update`
        - :meth:`~.PCBase.apply`
        - :meth:`~.PCBase.applyTranspose`

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
        implement :meth:`~.PCSNESBase.update` and :meth:`~.PCSNESBase.initialize`."""

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

    def destroy(self, pc):
        if hasattr(self, "pc"):
            self.pc.destroy()

    def form(self, obj, *args):
        """Return the preconditioning bilinear form and boundary conditions.

        Parameters
        ----------
        obj : PETSc.PC or PETSc.SNES
            The PETSc solver object.
        test : ufl.TestFunction
            The test function.
        trial : ufl.TrialFunction
            The trial function.

        Returns
        -------
        a : ufl.Form
            The preconditioning bilinear form.
        bcs : DirichletBC[] or None
            The boundary conditions.

        Notes
        -----
        Subclasses may override this function to provide an auxiliary bilinear
        form. Use `self.get_appctx(obj)` to get the user-supplied
        application-context, if desired.
        """
        if isinstance(obj, PETSc.PC):
            pc = obj
        elif isinstance(obj, PETSc.SNES):
            pc = obj.ksp.pc
        else:
            raise ValueError("Not a PC or SNES?")

        _, P = pc.getOperators()
        if P.getType() == "python":
            ctx = P.getPythonContext()
            a = ctx.a
            bcs = tuple(ctx.bcs)
        else:
            ctx = get_appctx(pc.getDM())
            a = ctx.Jp or ctx.J
            bcs = ctx.bcs_Jp
        if len(args):
            a = a(*args)
        return a, bcs

    @staticmethod
    def get_appctx(pc):
        return get_appctx(pc.getDM()).appctx

    @staticmethod
    def new_snes_ctx(pc, op, bcs, mat_type, fcp=None, options_prefix=None):
        """ Create a new SNES contex for nested preconditioning
        """
        from firedrake.variational_solver import NonlinearVariationalProblem
        from firedrake.function import Function
        from firedrake.ufl_expr import action
        from firedrake.solving_utils import _SNESContext

        dm = pc.getDM()
        old_appctx = get_appctx(dm).appctx
        u = Function(op.arguments()[-1].function_space())
        F = action(op, u)
        if bcs:
            bcs = tuple(bc._as_nonlinear_variational_problem_arg() for bc in bcs)
        nprob = NonlinearVariationalProblem(F, u,
                                            bcs=bcs,
                                            J=op,
                                            form_compiler_parameters=fcp)
        return _SNESContext(nprob, mat_type, mat_type, old_appctx, options_prefix=options_prefix)


class PCBase(PCSNESBase):

    _asciiname = "preconditioner"
    _objectname = "pc"

    needs_python_amat = False
    """Set this to True if the A matrix needs to be Python (matfree)."""

    needs_python_pmat = False
    """Set this to False if the P matrix needs to be Python (matfree).

    If the preconditioner also works with assembled matrices, then use False here.
    """

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

    def setUp(self, pc):
        A, P = pc.getOperators()
        Atype = A.getType()
        Ptype = P.getType()

        pcname = type(self).__module__ + "." + type(self).__name__
        if self.needs_python_amat and Atype != PETSc.Mat.Type.PYTHON:
            raise ValueError("PC '%s' needs amat to have type python, but it is %s" % (pcname, Atype))
        if self.needs_python_pmat and Ptype != PETSc.Mat.Type.PYTHON:
            raise ValueError("PC '%s' needs pmat to have type python, but it is %s" % (pcname, Ptype))

        super().setUp(pc)


class SNESBase(PCSNESBase):

    _asciiname = "nonlinear solver"
    _objectname = "snes"
