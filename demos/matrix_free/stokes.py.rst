==================
 Stokes Equations
==================
Again, they are what they are.  This is the driven cavity example.
We have several options files showing how to do some field-splitting
and such for this.::

  from firedrake import *

  N = 128

  M = UnitSquareMesh(N, N)

  V = VectorFunctionSpace(M, "CG", 2)
  W = FunctionSpace(M, "CG", 1)
  Z = V * W

  u, p = TrialFunctions(Z)
  v, q = TestFunctions(Z)

  a = (inner(grad(u), grad(v)) -
       p * div(v) +
       div(u) * q + p*q)*dx

  F = Function(V)
  G = Function(W)

  L = inner(F, v) * dx + G*q*dx

  bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
         DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

  up = Function(Z)

  nullspace = MixedVectorSpaceBasis(
      Z, [Z.sub(0), VectorSpaceBasis(constant=True)])


We set ``nest=False`` inside the problem so that assembled matrices
will be monolithic and we can use a direct solver.  The ``nest``
parameter is ignored if we turn on implicit matrices and so can still
field-split them efficiently.::
      
  prob = LinearVariationalProblem(a, L, up, bcs=bcs, nest=False)

  solver = LinearVariationalSolver(prob, options_prefix='', nullspace=nullspace)

  solver.solve()

  File("stokes.pvd").write(*up.split())
