from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
import firedrake.dmhooks as dmhooks


__all__ = ['GTMGPC']


class GTMGPC(PCBase):

    needs_python_pmat = False
    _prefix = "gt_"

    def initialize(self, pc):
        from firedrake import TestFunction, parameters
        from firedrake.assemble import get_assembler
        from firedrake.interpolation import Interpolator
        from firedrake.solving_utils import _SNESContext
        from firedrake.matrix_free.operators import ImplicitMatrixContext

        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        ctx = dmhooks.get_appctx(pc.getDM())
        if ctx is None:
            raise ValueError("No context found.")
        if not isinstance(ctx, _SNESContext):
            raise ValueError("Don't know how to get form from %r" % ctx)

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options()

        # Handle the fine operator if type is python
        if P.getType() == "python":
            ictx = P.getPythonContext()
            if ictx is None:
                raise ValueError("No context found on matrix")
            if not isinstance(ictx, ImplicitMatrixContext):
                raise ValueError("Don't know how to get form from %r" % ictx)

            fine_operator = ictx.a
            fine_bcs = ictx.row_bcs
            if fine_bcs != ictx.col_bcs:
                raise NotImplementedError("Row and column bcs must match")

            fine_mat_type = opts.getString(options_prefix + "mat_type",
                                           parameters["default_matrix_type"])
            fine_form_assembler = get_assembler(fine_operator, bcs=fine_bcs, form_compiler_parameters=fcp, mat_type=fine_mat_type, options_prefix=options_prefix)
            self.fine_op = fine_form_assembler.allocate()
            self._assemble_fine_op = fine_form_assembler.assemble
            self._assemble_fine_op(tensor=self.fine_op)
            fine_petscmat = self.fine_op.petscmat
        else:
            fine_petscmat = P

        # Transfer fine operator null space
        fine_petscmat.setNullSpace(P.getNullSpace())
        fine_transpose_nullspace = P.getTransposeNullSpace()
        if fine_transpose_nullspace.handle != 0:
            fine_petscmat.setTransposeNullSpace(fine_transpose_nullspace)

        # Handle the coarse operator
        coarse_options_prefix = options_prefix + "mg_coarse_"
        coarse_mat_type = opts.getString(coarse_options_prefix + "mat_type",
                                         parameters["default_matrix_type"])

        get_coarse_space = appctx.get("get_coarse_space", None)
        if not get_coarse_space:
            raise ValueError("Need to provide a callback which provides the coarse space.")
        coarse_space = get_coarse_space()

        get_coarse_operator = appctx.get("get_coarse_operator", None)
        if not get_coarse_operator:
            raise ValueError("Need to provide a callback which provides the coarse operator.")
        coarse_operator = get_coarse_operator()

        coarse_space_bcs = appctx.get("coarse_space_bcs", None)

        # These should be callbacks which return the relevant nullspaces
        get_coarse_nullspace = appctx.get("get_coarse_op_nullspace", None)
        get_coarse_transpose_nullspace = appctx.get("get_coarse_op_transpose_nullspace", None)

        coarse_form_assembler = get_assembler(coarse_operator, bcs=coarse_space_bcs, form_compiler_parameters=fcp, mat_type=coarse_mat_type, options_prefix=coarse_options_prefix)
        self.coarse_op = coarse_form_assembler.allocate()
        self._assemble_coarse_op = coarse_form_assembler.assemble
        self._assemble_coarse_op(tensor=self.coarse_op)
        coarse_opmat = self.coarse_op.petscmat

        # Set nullspace if provided
        if get_coarse_nullspace:
            nsp = get_coarse_nullspace()
            coarse_opmat.setNullSpace(nsp.nullspace())

        if get_coarse_transpose_nullspace:
            tnsp = get_coarse_transpose_nullspace()
            coarse_opmat.setTransposeNullSpace(tnsp.nullspace())

        interp_petscmat = appctx.get("interpolation_matrix", None)
        if interp_petscmat is None:
            # Create interpolation matrix from coarse space to fine space
            fine_space = ctx.J.arguments()[0].function_space()
            interpolator = Interpolator(TestFunction(coarse_space), fine_space)
            interpolation_matrix = interpolator.callable()
            interp_petscmat = interpolation_matrix.handle

        # We set up a PCMG object that uses the constructed interpolation
        # matrix to generate the restriction/prolongation operators.
        # This is a two-level multigrid preconditioner.
        pcmg = PETSc.PC().create(comm=pc.comm)
        pcmg.incrementTabLevel(1, parent=pc)

        pcmg.setType(pc.Type.MG)
        pcmg.setOptionsPrefix(options_prefix)
        pcmg.setMGLevels(2)
        pcmg.setMGCycleType(pc.MGCycleType.V)
        pcmg.setMGInterpolation(1, interp_petscmat)
        pcmg.setOperators(A=fine_petscmat, P=fine_petscmat)

        coarse_solver = pcmg.getMGCoarseSolve()
        coarse_solver.setOperators(A=coarse_opmat, P=coarse_opmat)
        # coarse space dm
        coarse_dm = coarse_space.dm
        coarse_solver.setDM(coarse_dm)
        coarse_solver.setDMActive(False)
        pcmg.setDM(pc.getDM())
        pcmg.setFromOptions()
        self.pc = pcmg
        self._dm = coarse_dm

        prefix = coarse_solver.getOptionsPrefix()
        # Create new appctx
        self._ctx_ref = self.new_snes_ctx(pc,
                                          coarse_operator,
                                          coarse_space_bcs,
                                          coarse_mat_type,
                                          fcp,
                                          options_prefix=prefix)

        with dmhooks.add_hooks(coarse_dm, self,
                               appctx=self._ctx_ref,
                               save=False):
            coarse_solver.setFromOptions()

    def update(self, pc):
        if hasattr(self, "fine_op"):
            self._assemble_fine_op(tensor=self.fine_op)

        self._assemble_coarse_op(tensor=self.coarse_op)
        self.pc.setUp()

    def apply(self, pc, X, Y):
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(X, Y)

    def view(self, pc, viewer=None):
        super(GTMGPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC using Gopalakrishnan and Tan algorithm\n")
            self.pc.view(viewer)
