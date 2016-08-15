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

  prob = NonlinearVariationalProblem(F, up, bcs=bcs)


We set extra information in the solver.  Certain Python-based preconditioners
(e.g. pressure convection-diffusion) need access to the Reynolds number,
which can't be fished out of the UFL bilinear form.  PCD also needs to know which
piece of the mixed function space is the velocity.  We pass this into the solver,
but otherwise the preconditioners are set from the options.::

  extra_ctx = {"Re": Re, "velocity_space": 0}
  
  solver = NonlinearVariationalSolver(prob, options_prefix='',
                                      nullspace=nullspace,
				      extra_ctx=extra_ctx)

  solver.solve()

  File("cavity.pvd").write(*up.split())
