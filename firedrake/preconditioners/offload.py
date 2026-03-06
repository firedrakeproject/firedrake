from firedrake.preconditioners.assembled import AssembledPC
from firedrake.petsc import PETSc

import petsc4py
import petsctools
import firedrake.dmhooks as dmhooks

__all__ = ("OffloadPC",)


class OffloadPC(AssembledPC):
    """Offload PC from CPU to GPU and back.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``offload_``.
    """

    _prefix = "offload_"

    device_mat_type_map = {"cuda": "aijcusparse"}

    def initialize(self, pc):
        # Check if our PETSc installation is GPU enabled
        for device, mat_type in self.device_mat_type_map.items():
            if device in petsctools.get_external_packages():
                break
        else:
            raise NotImplementedError(
                "This installation of Firedrake is not GPU-compatible, therefore "
                "OffloadPC can not be used. PETSc will need to be rebuilt with "
                "some GPU capability (e.g. '--with-cuda=1') to use this functionality."
            )
        # Check if we are we on a machine with a GPU.
        try:
            dev = PETSc.Device.Create()
        except petsc4py.PETSc.Error:
            raise RuntimeError(
                "This installation of Firedrake is GPU-Compatible, but no GPU device "
                "has been detected. OffloadPC can not be used on this host."
            )

        # Some other reason we could not initialise a device.
        if dev.getDeviceType() == "HOST":
            raise RuntimeError(
                "A GPU-enabled Firedrake build has been detected, but a GPU device was "
                "unable to be initialised. Cannot use OffloadPC."
            )
        dev.destroy()

        super().initialize(pc)

        with PETSc.Log.Event("Event: initialize offload"):
            A, P = pc.getOperators()

            # Convert matrix to ajicusparse
            with PETSc.Log.Event("Event: matrix offload"):
                P_cu = P.convert(mat_type)  # todo

            # Transfer nullspace
            P_cu.setNullSpace(P.getNullSpace())
            P_cu.setTransposeNullSpace(P.getTransposeNullSpace())
            P_cu.setNearNullSpace(P.getNearNullSpace())

            # Update preconditioner with GPU matrix
            self.pc.setOperators(A, P_cu)

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
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to solve on GPU\n")
            self.pc.view(viewer)
