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
        from firedrake.assemble import assemble

        _, P = pc.getOperators()

        context = P.getPythonContext()

        # It only makes sense to preconditioner/invert a diagonal
        # block in general.  That's all we're going to allow.
        if not context.on_diag:
            raise ValueError("Only makes sense to invert diagonal block")

        self.P = assemble(context.a, bcs=context.row_bcs,
                          form_compiler_parameters=context.fc_params, nest=False)
        self.P.force_evaluation()

        # Transfer nullspace over
        Pmat = self.P.PETScMatHandle
        Pmat.setNullSpace(P.getNullSpace())
        Pmat.setTransposeNullSpace(P.getTransposeNullSpace())

        prefix = pc.getOptionsPrefix()

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
        from firedrake import assemble
        P = self.P
        P = assemble(P.a, tensor=P, bcs=P.bcs)
        P.force_evaluation()

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)


class MassInvPC(PCBase):

    """A matrix free operator that inverts the mass matrix in the provided space.

    Internally this creates a PETSc KSP object that can be controlled
    by options using the extra options prefix ``Mp_``.
    """
    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble, inner
        prefix = pc.getOptionsPrefix()

        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()

        test, trial = context.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("MassInvPC only makes sense if test and trial space are the same")

        V = test.function_space()

        u = TrialFunction(V)
        v = TestFunction(V)
        # Handle vector and tensor-valued spaces.
        a = inner(u, v)*dx

        A = assemble(a, form_compiler_parameters=context.fc_params)
        A.force_evaluation()

        Pmat = A.PETScMatHandle
        Pmat.setNullSpace(P.getNullSpace())
        Pmat.setTranposeNullSpace(P.getTransposeNullSpace())

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


class PCDPC(PCBase):
    """A Pressure-Convection-Diffusion preconditioner for Navier-Stokes.

    This preconditioner approximates the inverse of the pressure schur
    complement for the Navier-Stokes equations by.

    .. math::

       S^{-1} \approx K^{-1} F_p M^{-1}

    Where :math:`K = \nabla^2`, :math:`M = \mathbb{I}` and
    :math:`F_p = 1/\mathrm{Re} \nabla^2 + u\cdot\grad`.

    The inverse of :math:`K` is approximated by a KSP which can be
    controlled using the options prefix ``pcd_Kp_``.

    The inverse of :math:`M` is similarly approximated by a KSP which
    can be controlled using the options prefix ``pcd_Mp_``.

    :math:`F_p` requires both the Reynolds number and the current
    velocity.  You must provide these with options using the glbaol
    option ``Re`` for the Reynolds number and the prefixed option
    ``pcd_velocity_space`` which should be the index into the full
    space that gives velocity field.

    """
    def initialize(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, \
            assemble, inner, grad, split, Constant
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

        # FIXME: allow these guys to be matrix-free
        Mp = assemble(mass, form_compiler_parameters=context.fc_params)
        Kp = assemble(stiffness, form_compiler_parameters=context.fc_params)

        Mp.force_evaluation()
        Kp.force_evaluation()

        # FIXME: Should we transfer nullspaces over.  I think not.

        Mksp = PETSc.KSP().create()
        Mksp.setOptionsPrefix(prefix + "Mp_")
        Mksp.setOperators(Mp.PETScMatHandle)
        Mksp.setUp()
        Mksp.setFromOptions()
        self.Mksp = Mksp

        Kksp = PETSc.KSP().create()
        Kksp.setOptionsPrefix(prefix + "Kp_")
        Kksp.setOperators(Kp.PETScMatHandle)
        Kksp.setUp()
        Kksp.setFromOptions()
        self.Kksp = Kksp

        state = context.context["state"]

        OptDB = PETSc.Options()
        # FIXME: spatially-varying Reynolds number.  Should probably
        # gets these from the context dict.
        Re_num = OptDB.getReal("Re", 1.0)
        Re = Constant(Re_num)
        velid = OptDB.getInt(prefix + "velocity_space")

        u0 = split(state)[velid]
        fp = 1.0/Re * inner(grad(p), grad(q))*dx + inner(u0, grad(p))*q*dx

        # FIXME, allow assembled matrix here
        self.Fp = assemble(fp, form_compiler_parameters=context.fc_params,
                           matfree=True)
        self.Fp.force_evaluation()
        Fpmat = self.Fp.PETScMatHandle
        self.workspace = [Fpmat.createVecLeft() for i in (0, 1)]

    def update(self, pc):
        # FIXME, support assembled matrix too
        from firedrake import assemble
        assemble(self.Fp.a, tensor=self.Fp, matfree=True)
        self.Fp.force_evaluation()

    def apply(self, pc, x, y):
        a, b = self.workspace
        self.Mksp.solve(x, a)
        self.Fp.PETScMatHandle.mult(a, b)
        self.Kksp.solve(b, y)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
