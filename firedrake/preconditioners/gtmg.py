from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase


__all__ = ['GTMGPC']


class GTMGPC(PCBase):

    needs_python_pmat = True

    def initialize(self, pc):

        from firedrake import TestFunction, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        from firedrake.interpolation import Interpolator

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + "gt_"
        opts = PETSc.Options()

        A, P = pc.getOperators()
        context = P.getPythonContext()
        # ipdb.set_trace()

        ctx_form = context.a
        fine_space = ctx_form.arguments()[0].function_space()

        appctx = context.appctx
        # ipdb.set_trace()

        get_coarse_space = appctx.get("get_coarse_space", None)
        if not get_coarse_space:
            raise ValueError("Need to specify a coarse space.")
        coarse_space = get_coarse_space()

        get_coarse_operator = appctx.get("get_coarse_operator", None)
        if not get_coarse_operator:
            raise ValueError("Need to specify a coarse operator")
        coarse_operator = get_coarse_operator()

        coarse_space_bcs = appctx.get("coarse_space_bcs", None)

        coarse_mat_type = opts.getString(options_prefix + "mat_type",
                                         parameters["default_matrix_type"])

        self.coarse_op = allocate_matrix(coarse_operator,
                                         bcs=coarse_space_bcs,
                                         form_compiler_parameters=context.fc_params,
                                         mat_type=coarse_mat_type,
                                         options_prefix=options_prefix)
        self._assemble_coarse_op = create_assembly_callable(coarse_operator,
                                                            tensor=self.coarse_op,
                                                            bcs=coarse_space_bcs,
                                                            form_compiler_parameters=context.fc_params)
        self._assemble_coarse_op()
        self.coarse_op.force_evaluation()
        coarse_opmat = self.coarse_op.petscmat
        # ipdb.set_trace()

        interpolator = Interpolator(TestFunction(coarse_space), fine_space)
        interpolation_matrix = interpolator.callable().handle
        # ipdb.set_trace()

        mgpc = PETSc.PC().create(comm=pc.comm)
        mgpc.setType(mgpc.Type.MG)
        mgpc.setOptionsPrefix(prefix)
        mgpc.setMGLevels(2)
        mgpc.setMGInterpolation(1, interpolation_matrix)
        # A = assemble(context.a,
        #              bcs=context.row_bcs,
        #              mat_type=fine_mat_type).M.handle

        mgpc.setOperators(A, P)
        coarse = mgpc.getMGCoarseSolve()
        coarse.setOperators(coarse_opmat)
        mgpc.setFromOptions()
        self.mgpc = mgpc

    def update(self, pc):
        self._assemble_coarse_op()
        self.coarse_op.force_evaluation()

    def apply(self, pc, X, Y):
        self.mgpc.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        self.mgpc.applyTranspose(X, Y)

    def view(self, pc, viewer=None):
        super(GTMGPC, self).view(pc, viewer)
        self.mgpc.view(viewer)
