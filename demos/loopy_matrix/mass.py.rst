Mass matrix in loopy
================
We are going to compute the action of the mass matrix via loopy and make
sure we get the same answer as the assembled sparse matrix.

  from firedrake import *

  M = UnitSquareMesh(10, 10)
  V = VectorFunctionSpace(M, "CG", 3)

  u = TrialFunction(V)
  v = TestFunction(V)

  a = inner(grad(u),grad(v))*dx

  A = assemble(a)
  Alp = assemble(a, mat_type="loopy")

  x = Function(V)
  y0 = Function(V)
  y1 = Function(V)

  A.force_evaluation()

  with x.dat.vec as xvec:
      xvec.setRandom()

  print "multiplying by A"
  with x.dat.vec as vin:
      with y0.dat.vec as vout:
          A.petscmat.mult(vin, vout)


  print "multiplying by Alp"
  with x.dat.vec as vin:
      with y1.dat.vec as vout:
          Alp.petscmat.mult(vin, vout)


  print sqrt(assemble((y0-y1)**2*dx))

A runnable python script implementing this demo file is available
`here <mass.py>`__.

If you want to loop at or monkey with the loopy kernel, you should get/set
knl = Alp.petscmat.getPythonContext().knl

