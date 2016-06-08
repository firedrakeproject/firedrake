How the sausage is made, where sausage == matrix-free action::

  from firedrake.petsc import PETSc

  class UFLMatrix(object):
      def __init__(self, a, bcs=[], state=None, fc_params = {}, extra={}):
          from firedrake import function
	  from ufl import action

We just stuff pointers to the bilinear form (Jacobian), boundary
conditions, form compiler parameters, and anything extra that user
wants passed in (from the solver setup routine).  This likely includes
physical parameters that aren't directly visible from UFL.  Since this
will typically be embedded in nonlinear loop, the `state` variable
allows us to insert the current linearization state.::
  
          self.a = a
          self.bcs = bcs
          self.fc_params = fc_params
          self.extra = extra
	  self.newton_state = state

This creates some functions for the source and target of 1-form assembly::
  
          # from test space
          self._x = function.Function(a.arguments()[1].function_space())

          # from trial space
          self._y = function.Function(a.arguments()[0].function_space())

We need to get the local and global sizes from these so the Python matrix
knows how to set itself up.  This could be done better?::
  
          with self._x.dat.vec_ro as xx:
              self.row_sizes = xx.getSizes()
          with self._y.dat.vec_ro as yy:
              self.col_sizes = yy.getSizes()

We will stash the UFL business for the action so we don't have to reconstruct
it at each matrix-vector product.::

          self.action = action(self.a, self._x)
	      
This defins how the PETSc matrix applies itself to a vector.  In our
case, it's just assembling a 1-form and applying boundary conditions.::
  
      def mult(self, X, Y):
          from firedrake.assemble import assemble
          
          with self._x.dat.vec as v:
              if v != X:
                  X.copy(v)

          assemble(self.action, self._y,
                   form_compiler_parameters = self.fc_params)
  
          for bc in self.bcs:
              bc.apply(self._y)

          with self._y.dat.vec_ro as v:
              v.copy(Y)

          return

Now, to enable fieldsplit preconditioners, we need to enable submatrix
extraction for our custom matrix type.  Note that we are splitting UFL
and index sets rather than an assembled matrix, keeping matrix
assembly deferred as long as possible.::
  
      def getSubMatrix(self, mat, row_is, col_is, target=None):
          from utils import find_sub_block

	  assert target is None
	  
These are the sets of ISes of which the the row and column space consist.::

	  row_ises = self._y.function_space().dof_dset.field_ises
	  col_ises = self._x.function_space().dof_dset.field_ises

This uses a nifty utility Lawrence provided to map the index sets into
tuples of integers indicating which field ids (hence logical sub-blocks).::

	  row_inds = find_sub_block(row_is, row_ises)
	  col_inds = find_sub_block(col_is, col_ises)

Now, actually extracting the right UFL bit will occur inside a special
class, which is a Python object that needs to be stuffed inside
a PETSc matrix::

          submat_ufl = UFLSubMatrix(mat.getPythonContext(), row_inds, col_inds)
          submat = PETSc.Mat().create()
	  submat.setType("python")
	  submat.setSizes((submat_ufl.row_sizes, submat_ufl.col_sizes))
	  submat.setPythonContext(submat_ufl)
	  return submat
  
And now for the sub matrix class.::

  class UFLSubMatrix(UFLMatrix):
      def __init__(self, A, row_inds, col_inds):
          from utils import ExtractSubBlock
          self.parent = A
	  asub, = ExtractSubBlock(row_inds, col_inds).split(A.a)
	  self.a = asub
	  
          UFLMatrix.__init__(self, asub,
	                     bcs=A.bcs,
			     state=A.newton_state,
			     fc_params=A.fc_params,
			     extra=A.extra)

The multiplication should just inherit, no?  But we need to be careful
when we extract submatrices.  Let's make sure one level works for now
and disable submatrices of submatrices.::

      def getSubMatrix(self, mat, row_is, col_is):
          1/0

          



