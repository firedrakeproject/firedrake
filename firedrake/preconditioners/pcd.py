from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc

__all__ = ("PCDPC", )


class PCDPC(PCBase):

    needs_python_pmat = True

    r"""A Pressure-Convection-Diffusion preconditioner for Navier-Stokes.

    This preconditioner approximates the inverse of the pressure schur
    complement for the Navier-Stokes equations by.

    .. math::

       S^{-1} \sim K^{-1} F_p M^{-1}

    Where :math:`K = \nabla^2`,
    :math:`F_p = (1/\mathrm{Re}) \nabla^2 + u\cdot\nabla`
    and :math:`M = \mathbb{I}`.

    The inverse of :math:`K` is approximated by a KSP which can be
    controlled using the options prefix ``pcd_Kp_``.

    The inverse of :math:`M` is similarly approximated by a KSP which
    can be controlled using the options prefix ``pcd_Mp_``.

    :math:`F_p` requires both the Reynolds number and the current
    velocity.  You must provide these with options using the glbaol
    option ``Re`` for the Reynolds number and the prefixed option
    ``pcd_velocity_space`` which should be the index into the full
    space that gives velocity field.

    .. note::

       Currently, the boundary conditions applied to the PCD operator
       are correct for characteristic velocity boundary conditions,
       but sub-optimal for in and outflow boundaries.
    """
    def initialize(self, pc):
        from firedrake import (TrialFunction, TestFunction, dx, inner,
                               grad, split, Constant, parameters)
        from firedrake.assemble import assemble, get_assembler
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        prefix = pc.getOptionsPrefix() + "pcd_"

        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()

        test, trial = context.a.arguments()
        if test.function_space() != trial.function_space():
            raise ValueError("Pressure space test and trial space differ")

        Q = test.function_space()

        p = TrialFunction(Q)
        q = TestFunction(Q)

        mass = p*q*dx

        # Regularisation to avoid having to think about nullspaces.
        stiffness = inner(grad(p), grad(q))*dx + Constant(1e-6)*p*q*dx

        opts = PETSc.Options()
        # we're inverting Mp and Kp, so default them to assembled.
        # Fp only needs its action, so default it to mat-free.
        # These can of course be overridden.
        # only Fp is referred to in update, so that's the only
        # one we stash.
        default = parameters["default_matrix_type"]
        Mp_mat_type = opts.getString(prefix+"Mp_mat_type", default)
        Kp_mat_type = opts.getString(prefix+"Kp_mat_type", default)
        self.Fp_mat_type = opts.getString(prefix+"Fp_mat_type", "matfree")

        Mp = assemble(mass, form_compiler_parameters=context.fc_params,
                      mat_type=Mp_mat_type,
                      options_prefix=prefix + "Mp_")
        Kp = assemble(stiffness, form_compiler_parameters=context.fc_params,
                      mat_type=Kp_mat_type,
                      options_prefix=prefix + "Kp_")

        # FIXME: Should we transfer nullspaces over.  I think not.

        Mksp = PETSc.KSP().create(comm=pc.comm)
        Mksp.incrementTabLevel(1, parent=pc)
        Mksp.setOptionsPrefix(prefix + "Mp_")
        Mksp.setOperators(Mp.petscmat)
        Mksp.setUp()
        Mksp.setFromOptions()
        self.Mksp = Mksp

        Kksp = PETSc.KSP().create(comm=pc.comm)
        Kksp.incrementTabLevel(1, parent=pc)
        Kksp.setOptionsPrefix(prefix + "Kp_")
        Kksp.setOperators(Kp.petscmat)
        Kksp.setUp()
        Kksp.setFromOptions()
        self.Kksp = Kksp

        state = context.appctx["state"]

        Re = context.appctx.get("Re", 1.0)

        velid = context.appctx["velocity_space"]

        u0 = split(state)[velid]
        fp = 1.0/Re * inner(grad(p), grad(q))*dx + inner(u0, grad(p))*q*dx

        self.Re = Re
        form_assembler = get_assembler(fp, bcs=None, form_compiler_parameters=context.fc_params, mat_type=self.Fp_mat_type, options_prefix=prefix + "Fp_")
        self.Fp = form_assembler.allocate()
        self._assemble_Fp = form_assembler.assemble
        self._assemble_Fp(tensor=self.Fp)
        Fpmat = self.Fp.petscmat
        self.workspace = [Fpmat.createVecLeft() for i in (0, 1)]

    def update(self, pc):
        self._assemble_Fp(tensor=self.Fp)

    def apply(self, pc, x, y):
        a, b = self.workspace
        self.Mksp.solve(x, a)
        self.Fp.petscmat.mult(a, b)
        self.Kksp.solve(b, y)

    def applyTranspose(self, pc, x, y):
        a, b = self.workspace
        self.Kksp.solveTranspose(x, b)
        self.Fp.petscmat.multTranspose(b, a)
        self.Mksp.solveTranspose(y, a)

    def view(self, pc, viewer=None):
        super(PCDPC, self).view(pc, viewer)
        viewer.printfASCII("Pressure-Convection-Diffusion inverse K^-1 F_p M^-1:\n")
        viewer.printfASCII("Reynolds number in F_p (applied matrix-free) is %s\n" %
                           str(self.Re))
        viewer.printfASCII("KSP solver for K^-1:\n")
        self.Kksp.view(viewer)
        viewer.printfASCII("KSP solver for M^-1:\n")
        self.Mksp.view(viewer)

    def destroy(self, pc):
        if hasattr(self, "workspace"):
            for vec in self.workspace:
                vec.destroy()
        if hasattr(self, "Kksp"):
            self.Kksp.destroy()
        if hasattr(self, "Fp"):
            self.Fp.petscmat.destroy()
        if hasattr(self, "Mksp"):
            self.Mksp.destroy()
