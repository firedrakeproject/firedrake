Let's test a simple case of field splitting.::

  from firedrake import *
  import nlvs
  M = UnitSquareMesh(10, 10)
  V = FunctionSpace(M, "CG", 1)
  W = FunctionSpace(M, "CG", 2)
  Z = V*W

  (u0, u1) = TrialFunctions(Z)
  (v0, v1) = TestFunctions(Z)

  a = inner(grad(u0), grad(v0))*dx + u1*v1*dx
  bcs = [DirichletBC(Z.sub(0), Constant(0), (1,2,3,4))]

  L = v0*dx - v1*dx

  u01 = Function(Z)

  prob = LinearVariationalProblem(a, L, u01, bcs=bcs)

  solver = nlvs.NonlinearVariationalSolver(prob, options_prefix='')

  solver.solve()

  u00, u10 = u01.split()
  File("u0.pvd").write(u00)
  File("u1.pvd").write(u10)
  

