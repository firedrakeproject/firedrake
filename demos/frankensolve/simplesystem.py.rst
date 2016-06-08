Let's test a simple case of field splitting.::

  from firedrake import *
  from firedrake.frankensolve import FrankenSolver
  M = UnitSquareMesh(3, 3)

  V = FunctionSpace(M, "CG", 1)
  W = FunctionSpace(M, "CG", 2)
  Z = V*W

  uu = TrialFunction(Z)
  vv = TestFunction(Z)
  
  u0, u1 = split(uu)
  v0, v1 = split(vv)

  ff = Function(Z)
  ff.split()[0].interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])"))
  ff.split()[1].interpolate(Expression("sin(2*pi*x[0])*sin(3*pi*x[1])"))
  
  a = inner(grad(u0), grad(v0))*dx +u0*v0*dx + inner(grad(u1), grad(v1))*dx + u1*v1*dx
  L = inner(ff,vv)*dx
 
  bcs=[]

  uu0 = Function(Z)

  prob = LinearVariationalProblem(a, L, uu0, bcs=bcs)
  
  solver = FrankenSolver(prob, options_prefix='')

  solver.solve()

  u00, u10 = uu0.split()
  File("u0.pvd").write(u00)
  File("u1.pvd").write(u10)
 

