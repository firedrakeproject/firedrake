from firedrake.preconditioners.assembled import AssembledPC
from firedrake.petsc import PETSc
from firedrake.utils import device_matrix_type
from functools import cache
import warnings

import firedrake.dmhooks as dmhooks

__all__ = ("OffloadPC",)



@cache
def offload_mat_type(pc_comm_rank) -> str | None:
    mat_type = device_matrix_type()
    if mat_type is None:
        if pc_comm_rank == 0:
            warnings.warn(
                "This installation of Firedrake is not GPU-enabled, therefore OffloadPC"
                "will do nothing. For this preconditioner to function correctly PETSc"
                "will need to be rebuilt with some GPU capability (e.g. '--with-cuda=1')."
            )
        return None
    try:
        dev = PETSc.Device.create()
    except PETSc.Error:
        if pc_comm_rank == 0:
            warnings.warn(
                "This installation of Firedrake is GPU-enabled, but no GPU device has"
                "been detected. OffloadPC will do nothing on this host"
            )
        return None
    if dev.getDeviceType() == "HOST":
        raise RuntimeError(
            "A GPU-enabled Firedrake build has been detected, and GPU hardware has been"
            "detected but a GPU device was unable to be initialised."
        )
    dev.destroy()
    return mat_type


class OffloadPC(AssembledPC):
    """Offload PC from CPU to GPU and back.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``offload_``.
    """

    _prefix = "offload_"

    def initialize(self, pc):
        # Check if our PETSc installation is GPU enabled
        super().initialize(pc)
        self.offload_mat_type = offload_mat_type(pc.comm.rank)
        if self.offload_mat_type is not None:
            with PETSc.Log.Event("Event: initialize offload"):
                A, P = pc.getOperators()

                # Convert matrix to ajicusparse
                with PETSc.Log.Event("Event: matrix offload"):
                    P_cu = P.convert(self.offload_mat_type)  # todo

            # Transfer nullspace
            P_cu.setNullSpace(P.getNullSpace())
            P_cu.setTransposeNullSpace(P.getTransposeNullSpace())
            P_cu.setNearNullSpace(P.getNearNullSpace())

            # Update preconditioner with GPU matrix
            self.pc.setOperators(A, P_cu)

    # Convert vectors to CUDA, solve and get solution on CPU back
    def apply(self, pc, x, y):
        if self.offload_mat_type is None:
            self.pc.apply(x, y)
        else:
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
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to solve on GPU\n")
            self.pc.view(viewer)
