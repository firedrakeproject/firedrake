This just confirms that we can use the mult method
In a UFL matrix and get the same answer as if we use the
assembled matrix.::

  from firedrake import *
  from firedrake.frankensolve import UFLMatrix

  M = UnitSquareMesh(100, 100)
  V = FunctionSpace(M, "CG", 1)
  W = V

  v = TrialFunction(V)
  w = TestFunction(W)

  a = inner(grad(v), grad(w)) * dx

  bcs = [DirichletBC(V, Constant(0.0), (1,2,3,4))] 


  x = Function(V).interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])"))
  bcs[0].apply(x)

  y0 = Function(W)
  y1 = Function(W)

  A = assemble(a)
  bcs[0].apply(A)


  with x.dat.vec as xx:
      with y0.dat.vec as yy:
          A.M.handle.mult(xx, yy)

AA needs to be dropped into a PETSc matrix for full functionality, but here goes.::

  AA = UFLMatrix(a, bcs=bcs)
  with x.dat.vec as xx:
      with y1.dat.vec as yy:
          AA.mult(AA, xx, yy)


  err = (assemble((y0-y1)**2*dx))**0.5


Note it works in parallel if we're so inclined.::
  
  from mpi4py import MPI
  if MPI.COMM_WORLD.Get_rank() == 0:
      print err


