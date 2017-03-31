Mass matrix in loopy
================
We are going to compute the action of the stokes matrix via loopy and make
sure we get the same answer as the assembled sparse matrix.::

  from firedrake import *

  M = UnitSquareMesh(10, 10)

  k = 1
  W = FunctionSpace(M, "CG", k)
  V = VectorFunctionSpace(M, "CG", k+1)
  Z = V*W
  
  u, p = TrialFunctions(Z)
  v, q = TestFunctions(Z)

  a = (
      inner(grad(u),grad(v))*dx
      - inner(p, div(v))*dx
      + inner(div(u), q)*dx
      )

  A = assemble(a)
  Alp = assemble(a, mat_type="loopy")

  x = Function(Z)
  y0 = Function(Z)
  y1 = Function(Z)

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


  with y0.dat.vec as v0:
      with y1.dat.vec as v1:
          v2 = v1 - v0
          print v2.norm()

A runnable python script implementing this demo file is available
`here <stokes.py>`__.

If you want to look at or monkey with the loopy kernel, you should get/set
knl = Alp.petscmat.getPythonContext().knl

