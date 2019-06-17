from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.dmhooks import get_function_space, get_appctx, push_appctx, pop_appctx


__all__ = ['GTMGPC']


class GTMGPC(PCBase):

    needs_python_pmat = True
    _prefix = "gt_"

    def initialize(self, pc):

        from firedrake import TestFunction, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        from firedrake.interpolation import Interpolator

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options()

        _, P = pc.getOperators()
        opc = pc
        context = P.getPythonContext()
        appctx = context.appctx
        fcp = appctx.get("form_compiler_parameters")

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

        # Create interpolation matrix from coarse space to fine space
        # fine_space = get_function_space(opc.getDM())
        # import ipdb; ipdb.set_trace()
        fine_space = context.a.arguments()[0].function_space()
        interpolator = Interpolator(TestFunction(coarse_space), fine_space)
        interpolation_matrix = interpolator.callable()
        interpolation_matrix._force_evaluation()
        interp_petscmat = interpolation_matrix.handle

        # We set up a PCMG object that uses the constructed interpolation
        # matrix to generate the restriction/prolongation operators.
        # This is a two-level multigrid preconditioner.
        pcmg = PETSc.PC().create(comm=pc.comm)
        pcmg.incrementTabLevel(1, parent=opc)
        pcmg.setType(pcmg.Type.MG)
        pcmg.setOptionsPrefix(options_prefix)
        pcmg.setMGLevels(2)
        pcmg.setMGInterpolation(1, interp_petscmat)
        pcmg.setOperators(fine_petscmat)

        # Now we configure the coarse solver. We need to set a DM
        # and an appropriate SNESContext on the pc so we can do, for example,
        # geometric multigrid on the coarse solve.
        from firedrake.variational_solver import NonlinearVariationalProblem
        from firedrake.solving_utils import _SNESContext
        from firedrake.function import Function
        from firedrake.ufl_expr import action
        import ipdb

        ipdb.set_trace()
        dm = opc.getDM()
        oappctx = get_appctx(dm)
        ipdb.set_trace()
        cu = Function(coarse_space)
        cF = action(coarse_operator, cu)
        nproblem = NonlinearVariationalProblem(cF, cu, coarse_space_bcs,
                                               J=coarse_operator,
                                               form_compiler_parameters=fcp)
        ncontext = _SNESContext(nproblem, coarse_mat_type, coarse_mat_type,
                                oappctx.appctx)
        push_appctx(dm, ncontext)
        self._ctx_ref = ncontext

        ipdb.set_trace()

        pcmg.setDM(dm)
        coarse_solver = pcmg.getMGCoarseSolve()
        coarse_solver.setOperators(coarse_opmat)
        pcmg.setFromOptions()
        pcmg.setUp()
        self._pcmg = pcmg
        pop_appctx(dm)

    def update(self, pc):
        self._assemble_fine_op()
        self.fine_op.force_evaluation()
        self._assemble_coarse_op()
        self.coarse_op.force_evaluation()

    def apply(self, pc, X, Y):
        self._pcmg.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        self._pcmg.applyTranspose(X, Y)

    def view(self, pc, viewer=None):
        super(GTMGPC, self).view(pc, viewer)
        self._pcmg.view(viewer)
