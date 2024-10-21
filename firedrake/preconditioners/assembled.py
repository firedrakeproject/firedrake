from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.ufl_expr import TestFunction, TrialFunction
import firedrake.dmhooks as dmhooks
from firedrake.dmhooks import get_function_space

__all__ = ("AssembledPC", "AuxiliaryOperatorPC")


class AssembledPC(PCBase):
    """A matrix-free PC that assembles the operator.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``assembled_``.
    """

    _prefix = "assembled_"

    def initialize(self, pc):
        from firedrake.assemble import get_assembler
        A, P = pc.getOperators()

        if pc.type != "python":
            raise ValueError("Expecting PC type python")
        opc = pc
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        V = get_function_space(pc.getDM()).collapse()
        test = TestFunction(V)
        trial = TrialFunction(V)

        if P.type == "python":
            context = P.getPythonContext()
            # It only makes sense to preconditioner/invert a diagonal
            # block in general.  That's all we're going to allow.
            if not context.on_diag:
                raise ValueError("Only makes sense to invert diagonal block")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix

        mat_type = PETSc.Options().getString(options_prefix + "mat_type", "aij")

        (a, bcs) = self.form(pc, test, trial)

        form_assembler = get_assembler(a, bcs=bcs, form_compiler_parameters=fcp, mat_type=mat_type, options_prefix=options_prefix)
        self.P = form_assembler.allocate()
        self._assemble_P = form_assembler.assemble
        self._assemble_P(tensor=self.P)

        # Transfer nullspace over
        self.set_nullspaces(pc)

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -assembled_pc_type ksp.
        pc = PETSc.PC().create(comm=opc.comm)
        pc.incrementTabLevel(1, parent=opc)

        # We set a DM and an appropriate SNESContext on the constructed PC so one
        # can do e.g. multigrid or patch solves.
        dm = opc.getDM()
        self._ctx_ref = self.new_snes_ctx(opc, a, bcs, mat_type,
                                          fcp=fcp, options_prefix=options_prefix)

        pc.setDM(dm)
        pc.setOptionsPrefix(options_prefix)
        pc.setOperators(A, self.P.petscmat)
        self.pc = pc
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False):
            pc.setFromOptions()

    def update(self, pc):
        self._assemble_P(tensor=self.P)

    def set_nullspaces(self, pc):
        # Copy nullspaces over from parent P matrix
        _, P = pc.getOperators()
        Pmat = self.P.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)
        Pmat.setNearNullSpace(P.getNearNullSpace())

    def apply(self, pc, x, y):
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        super(AssembledPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)


class AuxiliaryOperatorPC(AssembledPC):
    """A preconditioner that builds a PC on a specified form.
    Mainly used for describing approximations to Schur complements.
    """

    _prefix = "aux_"
