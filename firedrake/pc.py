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

__all__ = ["AssembledPC"]


class AssembledPC(object):
    def setUp(self, pc):
        from firedrake.assemble import assemble

        _, P = pc.getOperators()
        P_ufl = P.getPythonContext()

        # It only makes sense to preconditioner/invert a diagonal
        # block in general.  That's all we're going to allow.
        assert P_ufl.on_diag

        P_fd = assemble(P_ufl.a, bcs=P_ufl.row_bcs,
                        form_compiler_parameters=P_ufl.fc_params, nest=False)
        Pmat = P_fd.M.handle
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

# Applying this preconditioner is easy.::
    def apply(self, pc, x, y):
        self.pc.apply(x, y)

# And so is applying the transpose (if the internal PC supports it):
    def applyTranspose(self, pc, x, y):
        self.pc.apply(x, y)
