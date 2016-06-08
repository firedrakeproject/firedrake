Now, currently there is not much available as far as matrix-free
preconditioners in Firedrake, although Lawrence & Rob are working on
an additive Schwarz subspace correction method in a branch.   We
always need to be able to drop down to assemble the matrix and use
PETSc algebraic methods on the matrix.  Although it might make poor
sense to think of an algebraic solver as a preconditioner, this is
idiomatic for PETSc -- LU factorization is a preconditioner used in
conjuction with -ksp_type preonly.::

  from firedrake.assemble import assemble
  from firedrake.petsc import PETSc

  
Note that, like the UFLMatrix, this does not inherit from PETSc's PC
type.  Instead, it is a class that gets instantiated by PETSc and then
stuffed inside of a PC object.  When the PETSc PC type is "Python",
PETSc forwards its calls to methods implemented inside of this.::
  
  class AssembledPC(object):

This plays the role of a construcor, it is called when the enclosing
KSP context is set up.  In it, we just assemble into a firedrake
matrix, with nesting turned off.  We assume that if somebody wants an
algebraic matrix, they want to do something to it that requires
non-nested storage.  Later, we will extend the UFL matrix type to
support customized field splits.::
  
      def setUp(self, pc):
          _, P = pc.getOperators()
          P_ufl = P.getPythonContext()
          P_fd = assemble(P_ufl.a, bcs=P_ufl.bcs,
	                  form_compiler_parameters=P_ufl.fc_params, nest=False)
          Pmat = P_fd.M.handle
          optpre = pc.getOptionsPrefix()

Internally, we just set up a KSP object that the user can configure
however from the PETSc command line.::
  
          ksp = PETSc.KSP().create()
          ksp.setOptionsPrefix(optpre+"assembled_")
          ksp.setOperators(Pmat, Pmat)
          ksp.setUp()
          ksp.setFromOptions()
          self.ksp = ksp

Applying this preconditioner is relatively easy.::
  
      def apply(self, pc, x, y):
          ksp = self.ksp
          ksp.solve(x, y)
        
        
        
        
