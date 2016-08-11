Navier-Stokes
==============
Driven cavity with Taylor-Hood.::

  from firedrake import *

  N = 64

  M = UnitSquareMesh(N, N)

  V = VectorFunctionSpace(M, "CG", 2)
  W = FunctionSpace(M, "CG", 1)
  Z = V * W

  up = Function(Z)
  u, p = split(up)
  v, q = TestFunctions(Z)

  Re = Constant(100.0)

  F = (
      1.0 / Re * inner(grad(u), grad(v)) * dx +
      inner(dot(grad(u), u), v) * dx -
      p * div(v) * dx +
      div(u) * q * dx
  )

  bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
         DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

  nullspace = MixedVectorSpaceBasis(
      Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

  prob = NonlinearVariationalProblem(F, up, bcs=bcs, nest=False)

  solver = NonlinearVariationalSolver(prob, options_prefix='',
                                      nullspace=nullspace)

  solver.solve()

  File("cavity.pvd").write(*up.split())
