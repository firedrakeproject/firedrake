from __future__ import absolute_import
import abc

from firedrake.petsc import PETSc

__all__ = ("AssembledPC", "MassInvPC", "PCDPC", "PCBase")


class PCBase(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Create a PC context suitable for PETSc.

        Matrix free preconditioners should inherit from this class and
        implement:

        - :meth:`initialize`
        - :meth:`update`
        - :meth:`apply`
        - :meth:`applyTranspose`

        """
        self.initialized = False
        super(PCBase, self).__init__()

    @abc.abstractmethod
    def update(self, pc):
        """Update any state in this preconditioner."""
        pass

    @abc.abstractmethod
    def initialize(self, pc):
        """Initialize any state in this preconditioner."""
        pass

    def setUp(self, pc):
        """Setup method called by PETSc.

        Subclasses should probably not override this and instead
        implement :meth:`update` and :meth:`initialize`."""
        if self.initialized:
            self.update(pc)
        else:
            self.initialize(pc)
            self.initialized = True

    def view(self, pc, viewer=None):
        if viewer is None:
            return
        typ = viewer.getType()
        if typ != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Firedrake matrix-free preconditioner %s\n" %
                           type(self).__name__)

    @abc.abstractmethod
    def apply(self, pc, X, Y):
        """Apply the preconditioner to X, putting the result in Y.

        Both X and Y are PETSc Vecs, Y is not guaranteed to be zero on entry.
        """
        pass

    @abc.abstractmethod
    def applyTranspose(self, pc, X, Y):
        """Apply the transpose of the preconditioner to X, putting the result in Y.

        Both X and Y are PETSc Vecs, Y is not guaranteed to be zero on entry.

        """
        pass


class AssembledPC(PCBase):
    """A matrix-free PC that assembles the operator.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``assembled_``.
    """
    def initialize(self, pc):
        from firedrake.assemble import allocate_matrix, create_assembly_callable

        _, P = pc.getOperators()

        context = P.getPythonContext()
        prefix = pc.getOptionsPrefix()

        # It only makes sense to preconditioner/invert a diagonal
        # block in general.  That's all we're going to allow.
        if not context.on_diag:
            raise ValueError("Only makes sense to invert diagonal block")

        mat_type = PETSc.Options().getString(prefix + "assembled_mat_type", "aij")
        self.P = allocate_matrix(context.a, bcs=context.row_bcs,
                                 form_compiler_parameters=context.fc_params,
                                 mat_type=mat_type)
        self._assemble_P = create_assembly_callable(context.a, tensor=self.P,
                                                    bcs=context.row_bcs,
                                                    form_compiler_parameters=context.fc_params,
                                                    mat_type=mat_type)
        self._assemble_P()
        self.mat_type = mat_type
        self.P.force_evaluation()

        # Transfer nullspace over
        Pmat = self.P.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        Pmat.setTransposeNullSpace(P.getTransposeNullSpace())

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -assembled_pc_type ksp.
        pc = PETSc.PC().create()
        pc.setOptionsPrefix(prefix+"assembled_")
        pc.setOperators(Pmat, Pmat)
        pc.setUp()
        pc.setFromOptions()
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
        viewer.printfASCII("PC to apply inverse\n")
        viewer.pushASCIITab()
        self.pc.view(viewer)
        viewer.popASCIITab()


class MassInvPC(PCBase):

    """A matrix free operator that inverts the mass matrix in the provided space.

    Internally this creates a PETSc KSP object that can be controlled
    by options using the extra options prefix ``Mp_``.

    For Stokes problems, to be spectrally equivalent to the Schur
    complement, the mass matrix should be weighted by the viscosity.
    This can be provided (defaulting to constant viscosity) by
    providing a field defining the viscosity in the application
    context, keyed on ``"mu"``.
    """
    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble, inner, parameters
        prefix = pc.getOptionsPrefix()

        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()

        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("MassInvPC only makes sense if test and trial space are the same")

        V = test.function_space()

        mu = context.appctx.get("mu", 1.0)

        u = TrialFunction(V)
        v = TestFunction(V)
        # Handle vector and tensor-valued spaces.

        # 1/mu goes into the inner product in case it varies spatially.
        a = inner(1/mu * u, v)*dx

        opts = PETSc.Options()
        mat_type = opts.getString(prefix+"Mp_mat_type", parameters["default_matrix_type"])

        A = assemble(a, form_compiler_parameters=context.fc_params,
                     mat_type=mat_type)
        A.force_evaluation()

        Pmat = A.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        Pmat.setTransposeNullSpace(P.getTransposeNullSpace())

        ksp = PETSc.KSP().create()
        ksp.setOperators(Pmat)
        ksp.setOptionsPrefix(prefix + "Mp_")
        ksp.setUp()
        ksp.setFromOptions()
        self.ksp = ksp

    def update(self, pc):
        pass

    def apply(self, pc, X, Y):
        self.ksp.solve(X, Y)

    # Mass matrix is symmetric
    applyTranspose = apply

    def view(self, pc, viewer=None):
        super(MassInvPC, self).view(pc, viewer)
        viewer.printfASCII("KSP solver for M^-1\n")
        viewer.pushASCIITab()
        self.ksp.view(viewer)
        viewer.popASCIITab()


class PCDPC(PCBase):
    """A Pressure-Convection-Diffusion preconditioner for Navier-Stokes.

    This preconditioner approximates the inverse of the pressure schur
    complement for the Navier-Stokes equations by.

    .. math::

       S^{-1} \sim K^{-1} F_p M^{-1}

    Where :math:`K = \\nabla^2`, :math:`M = \mathbb{I}` and
    :math:`F_p = 1/\mathrm{Re} \\nabla^2 + u\cdot\\nabla`.

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
        from firedrake import TrialFunction, TestFunction, dx, \
            assemble, inner, grad, split, Constant, parameters
        from firedrake.assemble import allocate_matrix, create_assembly_callable
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
                      mat_type=Mp_mat_type)
        Kp = assemble(stiffness, form_compiler_parameters=context.fc_params,
                      mat_type=Kp_mat_type)

        Mp.force_evaluation()
        Kp.force_evaluation()

        # FIXME: Should we transfer nullspaces over.  I think not.

        Mksp = PETSc.KSP().create()
        Mksp.setOptionsPrefix(prefix + "Mp_")
        Mksp.setOperators(Mp.petscmat)
        Mksp.setUp()
        Mksp.setFromOptions()
        self.Mksp = Mksp

        Kksp = PETSc.KSP().create()
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
        self.Fp = allocate_matrix(fp, form_compiler_parameters=context.fc_params,
                                  mat_type=self.Fp_mat_type)
        self._assemble_Fp = create_assembly_callable(fp, tensor=self.Fp,
                                                     form_compiler_parameters=context.fc_params,
                                                     mat_type=self.Fp_mat_type)
        self._assemble_Fp()
        self.Fp.force_evaluation()
        Fpmat = self.Fp.petscmat
        self.workspace = [Fpmat.createVecLeft() for i in (0, 1)]

    def update(self, pc):
        self._assemble_Fp()
        self.Fp.force_evaluation()

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
        viewer.pushASCIITab()
        viewer.printfASCII("Reynolds number in F_p (applied matrix-free) is %g\n" %
                           str(self.Re))
        viewer.printfASCII("KSP solver for K^-1:\n")
        viewer.pushASCIITab()
        self.Kksp.view(viewer)
        viewer.popASCIITab()
        viewer.printfASCII("KSP solver for M^-1:\n")
        viewer.pushASCIITab()
        self.Mksp.view(viewer)
        viewer.popASCIITab()
        viewer.popASCIITab()
