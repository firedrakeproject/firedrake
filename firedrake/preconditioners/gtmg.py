from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx


__all__ = ['GTMGPC']


class GTMGPC(PCBase):

    needs_python_pmat = True
    _prefix = "gt_"

    def initialize(self, pc):

        from firedrake import TestFunction, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        from firedrake.interpolation import Interpolator

        _, P = pc.getOperators()
        context = P.getPythonContext()

        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        opc = pc
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options()

        # Handle the fine operator
        fine_mat_type = opts.getString(options_prefix + "mat_type",
                                       parameters["default_matrix_type"])
        fine_operator = context.a
        fine_bcs = context.row_bcs
        self.fine_op = allocate_matrix(fine_operator,
                                       bcs=fine_bcs,
                                       form_compiler_parameters=fcp,
                                       mat_type=fine_mat_type,
                                       options_prefix=options_prefix)
        self._assemble_fine_op = create_assembly_callable(fine_operator,
                                                          tensor=self.fine_op,
                                                          bcs=fine_bcs,
                                                          form_compiler_parameters=fcp,
                                                          mat_type=fine_mat_type)
        self._assemble_fine_op()
        self.fine_op.force_evaluation()
        fine_petscmat = self.fine_op.petscmat

        # Transfer fine operator null space
        fine_petscmat.setNullSpace(P.getNullSpace())
        fine_transpose_nullspace = P.getTransposeNullSpace()
        if fine_transpose_nullspace.handle != 0:
            fine_petscmat.setTransposeNullSpace(fine_transpose_nullspace)

        # Handle the coarse operator
        coarse_options_prefix = options_prefix + "mg_coarse"
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

        self.coarse_op = allocate_matrix(coarse_operator,
                                         bcs=coarse_space_bcs,
                                         form_compiler_parameters=fcp,
                                         mat_type=coarse_mat_type,
                                         options_prefix=coarse_options_prefix)
        self._assemble_coarse_op = create_assembly_callable(coarse_operator,
                                                            tensor=self.coarse_op,
                                                            bcs=coarse_space_bcs,
                                                            form_compiler_parameters=fcp)
        self._assemble_coarse_op()
        self.coarse_op.force_evaluation()
        coarse_opmat = self.coarse_op.petscmat

        # Set nullspace if provided
        if get_coarse_nullspace:
            coarse_opmat.setNullSpace(get_coarse_nullspace())

        if get_coarse_transpose_nullspace:
            coarse_opmat.setTransposeNullSpace(get_coarse_transpose_nullspace())

        interp_petscmat = appctx.get("interpolation_matrix", None)
        if interp_petscmat is None:
            # Create interpolation matrix from coarse space to fine space
            fine_space = context.a.arguments()[0].function_space()
            interpolator = Interpolator(TestFunction(coarse_space), fine_space)
            interpolation_matrix = interpolator.callable()
            interpolation_matrix._force_evaluation()
            interp_petscmat = interpolation_matrix.handle

        # We set up a PCMG object that uses the constructed interpolation
        # matrix to generate the restriction/prolongation operators.
        # This is a two-level multigrid preconditioner.
        pc = PETSc.PC().create(comm=opc.comm)
        pc.incrementTabLevel(1, parent=opc)

        pc.setType(pc.Type.MG)
        pc.setOptionsPrefix(options_prefix)
        pc.setMGLevels(2)
        pc.setMGInterpolation(1, interp_petscmat)
        pc.setOperators(fine_petscmat, fine_petscmat)

        coarse_solver = pc.getMGCoarseSolve()
        coarse_solver.setOperators(coarse_opmat, coarse_opmat)

        # We set a DM and an appropriate SNESContext for the coarse solver
        # so we can do multigrid or patch solves.
        from firedrake.variational_solver import NonlinearVariationalProblem
        from firedrake.solving_utils import _SNESContext
        from firedrake.function import Function
        from firedrake.ufl_expr import action

        dm = opc.getDM()
        octx = get_appctx(dm)
        coarse_tmp = Function(coarse_space)
        F = action(coarse_operator, coarse_tmp)
        nprob = NonlinearVariationalProblem(F, coarse_tmp,
                                            bcs=coarse_space_bcs,
                                            J=coarse_operator,
                                            form_compiler_parameters=fcp)
        nctx = _SNESContext(nprob, coarse_mat_type, coarse_mat_type, octx.appctx)
        self._ctx_ref = nctx

        # Push new context onto the coarse space dm
        coarse_dm = coarse_space.dm
        push_appctx(coarse_dm, nctx)

        coarse_solver.setDM(coarse_dm)
        coarse_solver.setDMActive(False)
        pc.setFromOptions()
        pc.setUp()
        self.pc = pc
        self.coarse_solver = coarse_solver
        pop_appctx(coarse_dm)

    def update(self, pc):
        self._assemble_fine_op()
        self.fine_op.force_evaluation()
        self._assemble_coarse_op()
        self.coarse_op.force_evaluation()

    def apply(self, pc, X, Y):
        dm = self.coarse_solver.getDM()
        push_appctx(dm, self._ctx_ref)
        self.pc.apply(X, Y)
        pop_appctx(dm)

    def applyTranspose(self, pc, X, Y):
        dm = self.coarse_solver.getDM()
        push_appctx(dm, self._ctx_ref)
        self.pc.applyTranspose(X, Y)
        pop_appctx(dm)

    def view(self, pc, viewer=None):
        super(GTMGPC, self).view(pc, viewer)
        self.pc.view(viewer)
