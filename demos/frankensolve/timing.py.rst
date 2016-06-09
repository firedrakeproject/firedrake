This is not a full-fledged suite, but it does show how to embed the
UFL Matrix into a PETSc matrix.  I'm learning that there is quite a high
overhead for the matvec, partly Python and partly Firedrake?  More
information will be needed, but that for high order and on a coarse mesh,
we have a small constant multiple for the matvec and win big on the assembly costs.::

  from firedrake import *
  from firedrake.frankensolve import UFLMatrix
  from firedrake.petsc import PETSc
  import time

  M = UnitCubeMesh(16, 16, 16)
  V = FunctionSpace(M, "CG", 1)
  W = V

  v = TrialFunction(V)
  w = TestFunction(W)
  f = Function(W)

  a = inner(grad(v), grad(w)) * dx

  bcs = [DirichletBC(V, Constant(0.0), (1,2,3,4))] 


  x = Function(V).interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])"))
  bcs[0].apply(x)

  y0 = Function(W)
  y1 = Function(W)

This times the assembly + BCs of the Firedrake fully assembled matrix.::
  
  t1 = time.time()
  A = assemble(a)
  bcs[0].apply(A)
  t_assemble = time.time() - t1

This times the creation of a PETSc matrix & UFLMatrix creation as its PETSc context.::

  t1 = time.time()
  A_ufl = PETSc.Mat().create()
  with x.dat.vec_ro as xx:
      xxsz = xx.getSizes()

  with y0.dat.vec_ro as yy:
      yysz = yy.getSizes()
  A_ufl.setSizes((yysz, xxsz))

  A_ufl.setType(PETSc.Mat.Type.PYTHON)
  A_ufl.setPythonContext(UFLMatrix(a, bcs=bcs))

  t_assembleMF = time.time() - t1

Now we time the matvecs of the two.::

  Aaij = A.M.handle
  t1 = time.time()
  with x.dat.vec_ro as xx:
      with y0.dat.vec as yy:
          Aaij.mult(xx, yy)

  taij = time.time() - t1


  t1 = time.time()
  with x.dat.vec_ro as xx:
      with y1.dat.vec as yy:
          A_ufl.mult(xx, yy)
  tmf = time.time() - t1

and display the results to the screen.::
  
  print "Assembly: "
  print "%.2e \t %.2e" % (t_assemble, t_assembleMF)
  print "Action: "
  print "%.2e \t %.2e" % (taij, tmf)


