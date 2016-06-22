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

__all__ = ["AssembledPC", "MassInvPC", "IdentityPC"]


class AssembledPC(object):
    def setUp(self, pc):
        from firedrake.assemble import assemble

        _, P = pc.getOperators()
        P_ufl = P.getPythonContext()

        if not self.initialized:  # First time, build new matrix, set up internal PC
            # It only makes sense to preconditioner/invert a diagonal
            # block in general.  That's all we're going to allow.
            assert P_ufl.on_diag

            self.P_fd = assemble(P_ufl.a, bcs=P_ufl.row_bcs,
                                 form_compiler_parameters=P_ufl.fc_params, nest=False)
            self.P_fd.force_evaluation()

            Pmat = self.P_fd.PETScMatHandle
            Pmat.setNullSpace(P.getNullSpace())

            optpre = pc.getOptionsPrefix()
            self.initialized = True

            # Internally, we just set up a PC object that the user can configure
            # however from the PETSc command line.  Since PC allows the user to specify
            # a KSP, we can do iterative by -assembled_pc_type ksp.::

            pc = PETSc.PC().create()
            pc.setOptionsPrefix(optpre+"assembled_")
            pc.setOperators(Pmat, Pmat)
            pc.setUp()
            pc.setFromOptions()
            self.pc = pc
            self.initialized = True
        else:  # reassemble in existing tensor.  Everything else is fine
            print self.P_fd
            assemble(P_ufl.a, bcs=P_ufl.row_bcs,
                     form_compiler_parameteres=P_ufl.fc_params, nest=False,
                     tensor=self.P_fd)

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.apply(x, y)

    @property
    def initialized(self):
        return hasattr(self, "_initialized") and self._initialized

    @initialized.setter
    def initialized(self, value):
        assert type(value) is bool
        self._initialized = value


class IdentityPC(object):
    def setUp(self, pc):
        return

    def apply(self, pc, X, Y):
        X.copy(Y)
        return

    def applyTranspose(self, pc, X, Y):
        X.copy(Y)
        return


class MassInvPC(object):
    def setUp(self, pc):
        # This only builds the matrix once since it is constant over all iterations.
        if not self.initialized:
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
            self.initialized = True
        else:
            pass

    def apply(self, pc, X, Y):
        self.Mksp.solve(X, Y)
        return

    def applyTranspose(self, pc, X, Y):
        # Mass matrix is symmetric.  Don't need to solveTranspose
        # on subKSP
        self.Mksp.solve(X, Y)
        return

    @property
    def initialized(self):
        return hasattr(self, "_initialized") and self._initialized

    @initialized.setter
    def initialized(self, value):
        assert type(value) is bool
        self._initialized = value
