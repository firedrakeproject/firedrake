import abc

import petsctools
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx, get_function_space
from firedrake.bcs import BCBase
import ufl

__all__ = ("PCBase", "SNESBase", "PCSNESBase")


class PCSNESBase(object, metaclass=abc.ABCMeta):
    """Create a PC or SNES python context suitable for PETSc.

    Both Python PC and SNES classes should inherit from this class and implement:

    - :meth:`~.PCSNESBase.initialize`
    - :meth:`~.PCSNESBase.update`

    Python PC classes should additionally implement:

    - :meth:`~.PCBase.apply`
    - :meth:`~.PCBase.applyTranspose`

    Python SNES classes should additionally implement _either_:

    - ``SNESBase.step``
    - ``SNESBase.solve``
    """
    def __init__(self):
        petsctools.cite("Kirby2017")
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
    def get_appctx(obj):
        return get_appctx(obj.getDM()).appctx

    @staticmethod
    def get_function_space(obj):
        return get_function_space(obj.getDM())

    @staticmethod
    def new_snes_ctx(
            pc: PETSc.PC,
            Jp: ufl.BaseForm,
            bcs: [BCBase],
            mat_type: str,
            fcp: dict | None = None,
            **kwargs):
        """Create a new `_SNESContext` for nested (linear) preconditioning

        Parameters
        ----------
        pc
            The PC object.
        Jp
            A bilinear form for preconditioning.
        bcs
            The boundary conditions.
        mat_type
            The matrix type for the assembly of ``Jp``.
        fcp
            The form compiler parameters.
        kwargs
            Any extra kwargs are passed on to the new _SNESContext.
            For details see `firedrake.solving_utils._SNESContext`.

        Returns
        -------
        A new `_SNESContext`
        """
        from firedrake.variational_solver import LinearVariationalProblem
        from firedrake.function import Function
        from firedrake.solving_utils import _SNESContext

        dm = pc.getDM()
        old_appctx = get_appctx(dm).appctx
        u = Function(Jp.arguments()[-1].function_space())
        L = 0
        if bcs:
            bcs = tuple(bc._as_nonlinear_variational_problem_arg(is_linear=True) for bc in bcs)

        nprob = LinearVariationalProblem(Jp, L, u, bcs=bcs, form_compiler_parameters=fcp)
        return _SNESContext(nprob, mat_type, mat_type, appctx=old_appctx, **kwargs)


class PCBase(PCSNESBase):
    """Create a PC python context suitable for PETSc.

    Matrix free preconditioners should inherit from this class and
    implement:

    - :meth:`~.PCSNESBase.initialize`
    - :meth:`~.PCSNESBase.update`
    - :meth:`~.PCBase.apply`
    - :meth:`~.PCBase.applyTranspose`
    """

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
    """Create SNES python context suitable for PETSc.

    Python SNES classes should inherit from this class and implement:

    - :meth:`~.PCSNESBase.initialize`
    - :meth:`~.PCSNESBase.update`

    Inheriting classes should additionally implement *either*:

    - ``SNESBase.step``
    - ``SNESBase.solve``

    The required function signatures for each method are shown below:

    .. code-block:: python3

        def solve(self, snes, b, x):
            '''Solve the nonlinear problem using the Vec x as the initial guess and
            putting the solution back into x. The Vec b is constant forcing term which
            may be None.
            '''
            pass

        def step(self, snes, X, F, Y):
            '''Apply one iteration of the SNES to the current iterate X,
            using the function residual F, and putting the update in Y.

            X, F and Y are PETSc Vecs, Y is not guaranteed to be zero on entry.
            '''
            pass

    Notes
    -----
    The function signatures for the ``solve`` and ``step`` methods are shown in
    the docstring rather than being implemented as abstract methods because
    petsc4py will test whether the SNES python context has either a ``step``
    or ``solve`` method to decide what to do when ``snes.solve()`` is called.
    """

    _asciiname = "nonlinear solver"
    _objectname = "snes"
