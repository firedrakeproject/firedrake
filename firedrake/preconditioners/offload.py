from firedrake.preconditioners.assembled import AssembledPC
from firedrake.petsc import PETSc
import firedrake.dmhooks as dmhooks

__all__ = ("OffloadPC",)


class OffloadPC(AssembledPC):
    """Offload PC from CPU to GPU and back.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``offload_``.
    """

    _prefix = "offload_"

    def initialize(self, pc):
        super().initialize(pc)

        with PETSc.Log.Event("Event: initialize offload"):
            A, P = pc.getOperators()

            # Convert matrix to ajicusparse
            mat_type = PETSc.Options().getString(self._prefix + "mat_type", "cusparse")
            with PETSc.Log.Event("Event: matrix offload"):
                P_cu = P.convert(mat_type='aijcusparse')  # todo

            # Transfer nullspace
            P_cu.setNullSpace(P.getNullSpace())
            P_cu.setTransposeNullSpace(P.getTransposeNullSpace())
            P_cu.setNearNullSpace(P.getNearNullSpace())

            # Update preconditioner with GPU matrix
            self.pc.setOperators(A, P_cu)

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
                    self.pc.apply(x_cu, y_cu)
                    # Calling data to synchronize vector
                    tmp = y_cu.array_r  # noqa: F841
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
