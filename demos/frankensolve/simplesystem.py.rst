Let's test a simple case of field splitting.::

  from firedrake import *
  import nlvs
  M = UnitSquareMesh(3, 3)
  V = FunctionSpace(M, "CG", 1)
  W = FunctionSpace(M, "CG", 2)
  Z = V*W

  uu = TrialFunction(Z)
  vv = TestFunction(Z)
  
  u0, u1 = split(uu)
  v0, v1 = split(vv)

  a = inner(grad(u0), grad(v0)) * dx + inner(grad(u1), grad(v1))*dx
  L = v0*dx+v1*dx
 
  bcs = [DirichletBC(Z.sub(0), Constant(0), (1,2,3,4)),
         DirichletBC(Z.sub(1), Constant(0), (1,2,3,4)),]

  uu0 = Function(Z)

  prob = LinearVariationalProblem(a, L, uu0, bcs=bcs)
  
  solver = nlvs.NonlinearVariationalSolver(prob, options_prefix='')

  solver.solve()

  u00, u10 = uu0.split()
  File("u0.pvd").write(u00)
  File("u1.pvd").write(u10)
  

