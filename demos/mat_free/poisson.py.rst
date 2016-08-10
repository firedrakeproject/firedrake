Poisson
=======
It is what it is.  Conforming discretization on a regular mesh.
Try out "python poisson.py -options_file opts.poisson.<foo>.txt::

  from firedrake import *  # noqa


  N = 128

  M = UnitSquareMesh(N, N)

  V = FunctionSpace(M, "CG", 2)

  u = TrialFunction(V)
  v = TestFunction(V)

  a = inner(grad(u), grad(v)) * dx

  F = Function(V)
  F.interpolate(Expression("sin(x[0]*pi)*sin(2*x[1]*pi)"))
  L = F*v*dx

  bcs = [DirichletBC(V, Constant(2.0), (1,))]

  uu = Function(V)

  prob = LinearVariationalProblem(a, L, uu, bcs)

  solver = LinearVariationalSolver(prob, options_prefix='')

  solver.solve()

  File("poisson.pvd").write(uu)
