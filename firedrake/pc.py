# Defines some preconditioners to be used in connection with
# ImplicitMatrix (matrix-free matrices).
# The general pattern is that each PC will provide a setUp method
# that can extract the input matrix' Python context and do with it
# what it will.  It sets some local state that the apply and (if
# desired, applyTranspose) methods will use.

# The first such preconditioner is, although somewhat against the
# spirit of matrix-free methods, the simplest way to illustrate the
# flow pattern.  It's called "AssembledPC" and just forces the
# assembly of the ImplicitMatrix into a PETSc aij matrix and sets up a
# PC context to do what we like with.

from firedrake.petsc import PETSc
from pcinit import InitializedPC
from ssc_schwarz import PatchPC, P1PC

__all__ = ["AssembledPC", "MassInvPC", "IdentityPC", "PCDPC", "PatchPC", "P1PC", "InitializedPC"]


class AssembledPC(InitializedPC):
    def initialSetUp(self, pc):
        from firedrake.assemble import assemble

        _, P = pc.getOperators()
        P_ufl = P.getPythonContext()

        # It only makes sense to preconditioner/invert a diagonal
        # block in general.  That's all we're going to allow.
        assert P_ufl.on_diag

        self.P_fd = assemble(P_ufl.a, bcs=P_ufl.row_bcs,
                             form_compiler_parameters=P_ufl.fc_params, nest=False)
        self.P_fd.force_evaluation()

        Pmat = self.P_fd.PETScMatHandle
        Pmat.setNullSpace(P.getNullSpace())

        optpre = pc.getOptionsPrefix()

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -assembled_pc_type ksp.::

        pc = PETSc.PC().create()
        pc.setOptionsPrefix(optpre+"assembled_")
        pc.setOperators(Pmat, Pmat)
        pc.setUp()
        pc.setFromOptions()
        self.pc = pc

    def subsequentSetUp(self, pc):
        from firedrake import assemble
        _, P = pc.getOperators()
        P_ufl = P.getPythonContext()
        assemble(P_ufl.a, bcs=P_ufl.row_bcs,
                 form_compiler_parameters=P_ufl.fc_params, nest=False,
                 tensor=self.P_fd)
        self.P_fd.force_evaluation()

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)


class IdentityPC(object):
    def setUp(self, pc):
        return

    def apply(self, pc, X, Y):
        X.copy(Y)
        return

    def applyTranspose(self, pc, X, Y):
        X.copy(Y)
        return


class MassInvPC(InitializedPC):
    def initialSetUp(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble
        optpre = pc.getOptionsPrefix()

        # we assume A has things stuffed inside of it
        A, P = pc.getOperators()
        Aufl = P.getPythonContext()

        pressure_space = Aufl.a.arguments()[0].function_space()

        pp = TrialFunction(pressure_space)
        qq = TestFunction(pressure_space)
        mp = pp*qq*dx

        Mfd = assemble(mp)
        Mfd.force_evaluation()
        M = Mfd.PETScMatHandle
        M.setNullSpace(P.getNullSpace())

        Mksp = PETSc.KSP().create()
        Mksp.setOperators(M)
        Mksp.setOptionsPrefix(optpre + "Mp_")
        Mksp.setUp()
        Mksp.setFromOptions()
        self.Mksp = Mksp

    def subsequentSetUp(self, pc):
        pass

    def apply(self, pc, X, Y):
        self.Mksp.solve(X, Y)
        return

    def applyTranspose(self, pc, X, Y):
        # Mass matrix is symmetric.  Don't need to solveTranspose
        # on subKSP
        self.Mksp.solve(X, Y)
        return


class PCDPC(InitializedPC):
    def initialSetUp(self, pc):
        from firedrake import TrialFunction, TestFunction, dx, assemble, inner, grad, split, Constant
        optpre = pc.getOptionsPrefix()

        # we assume A has things stuffed inside of it
        A, P = pc.getOperators()
        Pctx = P.getPythonContext()

        pressure_space = Pctx.a.arguments()[0].function_space()

        pp = TrialFunction(pressure_space)
        qq = TestFunction(pressure_space)
        mp = pp*qq*dx

        Mfd = assemble(mp)
        Mfd.force_evaluation()
        M = Mfd.PETScMatHandle
        M.setNullSpace(P.getNullSpace())

        Mksp = PETSc.KSP().create()
        Mksp.setOperators(M)
        Mksp.setOptionsPrefix(optpre + "Mp_")
        Mksp.setUp()
        Mksp.setFromOptions()
        self.Mksp = Mksp

        kp = inner(grad(pp), grad(qq))*dx + Constant(1.e-6)*pp*qq*dx
        Kfd = assemble(kp)
        Kfd.force_evaluation()
        K = Kfd.PETScMatHandle
        K.setNullSpace(P.getNullSpace())

#        K.getNullSpace().view()
#        print dir(K.getNullSpace())
#        K.view()
        Kksp = PETSc.KSP().create()
        Kksp.setOperators(K)
        Kksp.setOptionsPrefix(optpre + "Kp_")
        Kksp.setUp()
        Kksp.setFromOptions()
        self.Kksp = Kksp

        ctx = Pctx.extra
        x0 = ctx["state"]

        # Re = ctx.get("Re", Constant(1.0))
        # velid = ctx["velocity_space"]
        OptDB = PETSc.Options()
        Re_num = OptDB.getReal("Re", 1.0)
        Re = Constant(Re_num)
        velid = OptDB.getInt(optpre + "pcd_velocity_space")

        u0 = split(x0)[velid]
        fp = 1.0/Re * inner(grad(pp), grad(qq))*dx + inner(u0, grad(pp))*qq*dx
        self.Fpfd = assemble(fp, matfree=True)
        self.Fpfd.force_evaluation()
        self.Fp = self.Fpfd.PETScMatHandle

        self.tmp = [self.Fp.createVecLeft() for i in (0, 1)]

    def subsequentSetUp(self, pc):
        # self.Fpfd.
        self.Fpfd.assemble()

    def apply(self, pc, x, y):
        self.Mksp.solve(x, self.tmp[0])
        self.Fp.mult(self.tmp[0], self.tmp[1])
        self.Kksp.solve(self.tmp[1], y)
