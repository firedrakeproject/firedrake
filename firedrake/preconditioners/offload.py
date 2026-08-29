from firedrake.preconditioners.assembled import PCBase
from firedrake.petsc import PETSc
from firedrake.utils import device_matrix_type, get_device_type

import firedrake.dmhooks as dmhooks

__all__ = ("OffloadPC",)


_device_vector_impls = {
    "CUDA": {
        "createWithArrays": "createCUDAWithArrays",
    },
    "HIP": {
        "createWithArrays": "createHIPWithArrays",
    },
}

# These matrix types require an expensive implicit -> dense -> sparse conversion when
# offloaded to a GPU. The A matrix does not need to be offloaded, therefore if the A
# matrix is any of these types, do not offload it.
_no_offload_mat_types = ("python", "schurcomplement")


class OffloadPC(PCBase):
    """Offload PC from CPU to GPU and back.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``offload_``.
    """

    _prefix = "offload_"

    def call_device_vec_impl(self, x, func_name: str, *args, **kwargs):
        return getattr(x, _device_vector_impls[self.device_type][func_name])(
            *args, **kwargs
        )

    def initialize(self, pc):
        A, P = pc.getOperators()

        if pc.type != "python":
            raise ValueError("Expecting PC type python")
        opc = pc
        if P.type == "python":
            context = P.getPythonContext()
            # It only makes sense to precondition/invert a diagonal
            # block in general.  That's all we're going to allow.
            if not context.on_diag:
                raise ValueError("Only makes sense to invert diagonal block")

        prefix = pc.getOptionsPrefix() or ""
        options_prefix = prefix + self._prefix

        self.device_mat = device_matrix_type(warn=(pc.comm.rank == 0))
        self.device_type = get_device_type()
        dm = opc.getDM()

        pc = PETSc.PC().create(comm=opc.comm)
        pc.setDM(dm)
        pc.setOptionsPrefix(options_prefix)
        if self.device_mat is not None:
            with PETSc.Log.Event("Event: initialize offload"):
                P_dev = PETSc.Mat()
                P_dev = P.convert(mat_type=self.device_mat, out=P_dev)
                if A.handle == P.handle:
                    A_dev = P_dev
                elif A.type in _no_offload_mat_types:
                    A_dev = A
                else:
                    A_dev = PETSc.Mat()
                    A_dev = A.convert(mat_type=self.device_mat, out=A_dev)
            P_dev.setNullSpace(P.getNullSpace())
            P_dev.setTransposeNullSpace(P.getTransposeNullSpace())
            P_dev.setNearNullSpace(P.getNearNullSpace())
            pc.setOperators(A_dev, P_dev)
        else:
            pc.setOperators(A, P)

        # Simplest reconstruction we can manage
        octx = dmhooks.get_appctx(dm)
        self._ctx_ref = octx.reconstruct(mat_type=self.device_mat, pmat_type=self.device_mat)
        self.pc = pc

        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False):
            self.pc.setFromOptions()

    def update(self, pc):
        A, P = pc.getOperators()
        A_dev, P_dev = self.pc.getOperators()
        # Perform a value-only copy
        P.copy(P_dev, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        if A_dev.handle != P_dev.handle and A.type not in _no_offload_mat_types:
            # Perform a value-only copy
            A.copy(A_dev, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)

    # Convert vectors to CUDA, solve and get solution on CPU back
    def apply(self, pc, x, y, transpose=False):
        pc_apply = self.pc.applyTranspose if transpose else self.pc.apply
        dm = pc.getDM()
        if self.device_mat is None:
            with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
                pc_apply(x, y)
        else:
            with PETSc.Log.Event("Event: apply offload"):
                with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
                    with PETSc.Log.Event("Event: vectors offload"):
                        # Create the to-be-offloaded vector
                        y_dev = PETSc.Vec()
                        # Use device implementation of 'createWithArrays' function
                        self.call_device_vec_impl(y_dev, "createWithArrays", y.array_r, None)
                        # Create the to-be-offloaded vector
                        x_dev = PETSc.Vec()
                        # Use device implementation of 'createWithArrays' function
                        self.call_device_vec_impl(x_dev, "createWithArrays", x.array_r, None)
                    with PETSc.Log.Event("Event: solve"):
                        pc_apply(x_dev, y_dev)
                    with PETSc.Log.Event("Event: vectors copy back"):
                        # y is already designated as host storage for y_dev, so calling
                        # getArray is sufficient to synchronise the vector on the device
                        # with y on the host
                        y_dev.getArray(True)

    def applyTranspose(self, pc, x, y):
        self.apply(pc, x, y, transpose=True)

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to solve on GPU\n")
            self.pc.view(viewer)
