from firedrake.preconditioners.base import PCBase
from firedrake.functionspace import FunctionSpace, MixedFunctionSpace
from firedrake.petsc import PETSc
from firedrake.ufl_expr import TestFunction, TrialFunction
import firedrake.dmhooks as dmhooks
from firedrake.dmhooks import get_function_space

__all__ = ("OffloadPC",)


class OffloadPC(PCBase):
    """Offload PC from CPU to GPU and back.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``offload_``.
    """

    _prefix = "offload_"

    def initialize(self, pc):
        with PETSc.Log.Event("Event: initialize offload"):  #
            A, P = pc.getOperators()

            outer_pc = pc
            appctx = self.get_appctx(pc)
            fcp = appctx.get("form_compiler_parameters")

            V = get_function_space(pc.getDM())
            if len(V) == 1:
                V = FunctionSpace(V.mesh(), V.ufl_element())
            else:
                V = MixedFunctionSpace([V_ for V_ in V])
            test = TestFunction(V)
            trial = TrialFunction(V)

            (a, bcs) = self.form(pc, test, trial)

            if P.type == "assembled":
                context = P.getPythonContext()
                # It only makes sense to preconditioner/invert a diagonal
                # block in general.  That's all we're going to allow.
                if not context.on_diag:
                    raise ValueError("Only makes sense to invert diagonal block")

            prefix = pc.getOptionsPrefix()
            options_prefix = prefix + self._prefix

            mat_type = PETSc.Options().getString(options_prefix + "mat_type", "cusparse")

            # Convert matrix to ajicusparse
            with PETSc.Log.Event("Event: matrix offload"):
                P_cu = P.convert(mat_type='aijcusparse')  # todo

            # Transfer nullspace
            P_cu.setNullSpace(P.getNullSpace())
            tnullsp = P.getTransposeNullSpace()
            if tnullsp.handle != 0:
                P_cu.setTransposeNullSpace(tnullsp)
            P_cu.setNearNullSpace(P.getNearNullSpace())

            # PC object set-up
            pc = PETSc.PC().create(comm=outer_pc.comm)
            pc.incrementTabLevel(1, parent=outer_pc)

            # We set a DM and an appropriate SNESContext on the constructed PC
            # so one can do e.g. multigrid or patch solves.
            dm = outer_pc.getDM()
            self._ctx_ref = self.new_snes_ctx(
                outer_pc, a, bcs, mat_type,
                fcp=fcp, options_prefix=options_prefix
            )

            pc.setDM(dm)
            pc.setOptionsPrefix(options_prefix)
            pc.setOperators(A, P_cu)
            self.pc = pc
            with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False):
                pc.setFromOptions()

    def update(self, pc):
        _, P = pc.getOperators()
        _, P_cu = self.pc.getOperators()
        P.copy(P_cu)

    def form(self, pc, test, trial):
        _, P = pc.getOperators()
        if P.getType() == "python":
            context = P.getPythonContext()
            return (context.a, context.row_bcs)
        else:
            context = dmhooks.get_appctx(pc.getDM())
            return (context.Jp or context.J, context._problem.bcs)

    # Convert vectors to CUDA, solve and get solution on CPU back
    def apply(self, pc, x, y):
        with PETSc.Log.Event("Event: apply offload"):  #
            dm = pc.getDM()
            with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
                with PETSc.Log.Event("Event: vectors offload"):
                    y_cu = PETSc.Vec()  # begin
                    y_cu.createCUDAWithArrays(y)
                    x_cu = PETSc.Vec()
                    # Passing a vec into another vec doesnt work because original is locked
                    x_cu.createCUDAWithArrays(x.array_r)
                with PETSc.Log.Event("Event: solve"):
                    self.pc.apply(x_cu, y_cu)  #
            with PETSc.Log.Event("Event: vectors copy back"):
                y.copy(y_cu)  #

    def applyTranspose(self, pc, X, Y):
        raise NotImplementedError

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        print("viewing PC")
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to solve on GPU\n")
            self.pc.view(viewer)
