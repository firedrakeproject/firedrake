from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc

__all__ = ("AssembledPC", )


class AssembledPC(PCBase):
    """A matrix-free PC that assembles the operator.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``assembled_``.
    """
    def initialize(self, pc):
        from firedrake.assemble import allocate_matrix, create_assembly_callable

        _, P = pc.getOperators()

        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        opc = pc
        context = P.getPythonContext()
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + "assembled_"

        # It only makes sense to preconditioner/invert a diagonal
        # block in general.  That's all we're going to allow.
        if not context.on_diag:
            raise ValueError("Only makes sense to invert diagonal block")

        mat_type = PETSc.Options().getString(options_prefix + "mat_type", "aij")
        self.P = allocate_matrix(context.a, bcs=context.row_bcs,
                                 form_compiler_parameters=context.fc_params,
                                 mat_type=mat_type,
                                 options_prefix=options_prefix)
        self._assemble_P = create_assembly_callable(context.a, tensor=self.P,
                                                    bcs=context.row_bcs,
                                                    form_compiler_parameters=context.fc_params,
                                                    mat_type=mat_type)
        self._assemble_P()
        self.P.force_evaluation()

        # Transfer nullspace over
        Pmat = self.P.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -assembled_pc_type ksp.
        pc = PETSc.PC().create(comm=opc.comm)
        pc.incrementTabLevel(1, parent=opc)
        pc.setOptionsPrefix(options_prefix)
        pc.setOperators(Pmat, Pmat)
        pc.setFromOptions()
        pc.setUp()
        self.pc = pc

    def update(self, pc):
        self._assemble_P()
        self.P.force_evaluation()

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        super(AssembledPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)
