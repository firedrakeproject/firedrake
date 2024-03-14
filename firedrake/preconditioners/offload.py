import abc

from firedrake.preconditioners.base import PCBase
from firedrake.functionspace import FunctionSpace, MixedFunctionSpace
from firedrake.petsc import PETSc
from firedrake.ufl_expr import TestFunction, TrialFunction
import firedrake.dmhooks as dmhooks
from firedrake.dmhooks import get_function_space

import petsc4py.PETSc # in firedrake.petsc?

# outside: ksp.setOperators(A)
# todo: for densecuda later- now only cusparse

__all__ = ("OffloadPC",)


class OffloadPC(PCBase):
    """Offload to GPU as PC to solve.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``offload_``.
    """

    _prefix = "offload_"

    def initialize(self, pc):
        A, P = pc.getOperators()  # P preconditioner

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

        if P.type == "assembled":  # not python value error - only assembled (preconditioner)
            context = P.getPythonContext()
            # It only makes sense to preconditioner/invert a diagonal
            # block in general.  That's all we're going to allow.
            if not context.on_diag:  # still? diagonal block?
                raise ValueError("Only makes sense to invert diagonal block")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix

        mat_type = PETSc.Options().getString(options_prefix + "mat_type", "cusparse")  # cuda?

        # matrix to cuda
        P_cu = P.convert(mat_type='aijcusparse')
        # eventually change allocate_matrix from assembled.py

        P_cu.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            P_cu.setTransposeNullSpace(tnullsp)
        P_cu.setNearNullSpace(P.getNearNullSpace())

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

# vectors and solve
    def apply(self, pc, x, y):  # y=b?
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            b_cu = PETSc.Vec()
            b_cu.createCUDAWithArrays(y)
            u = PETSc.Vec()
            u.createCUDAWithArrays(x)
            self.pc.apply(x, y)  # solve is here
            u.getArray()  # return vector

    def applyTranspose(self, pc, X, Y):
        raise NotImplementedError

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        print("viewing PC")
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to solve on GPU\n")
            self.pc.view(viewer)
