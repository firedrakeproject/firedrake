How the sausage is made, where sausage == matrix-free action::


  class UFLMatrix(object):
      def __init__(self, a, bcs=[], fc_params = {}, extra={}):
          from firedrake import function

We just stuff pointers to the bilinear form (Jacobian), boundary
conditions, form compiler parameters, and anything extra that user
wants passed in (from the solver setup routine).  This likely includes
physical parameters that aren't directly visible from UFL::
  
          self.a = a
          self.bcs = bcs
          self.fc_params = fc_params
          self.extra = extra

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

	      
This defins how the PETSc matrix applies itself to a vector.  In our
case, it's just assembling a 1-form and applying boundary conditions.::
  
      def mult(self, mat, X, Y):
          from firedrake.assemble import assemble
          from ufl import action
          
          with self._x.dat.vec as v:
              if v != X:
                  X.copy(v)

          assemble(action(self.a, self._x), self._y,
                   form_compiler_parameters = self.fc_params)
  
          for bc in self.bcs:
              bc.apply(self._y)

          with self._y.dat.vec_ro as v:
              v.copy(Y)

          return

