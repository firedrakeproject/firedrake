Rayleigh-Benard Convection
==========================
This problem involves a variable-temperature incompressible fluid.
Variations in the fluid temperature are assumed to affect the momentum
balance through a buoyant term (the Boussinesq approximation), leading
to a Navier-Stokes equation with a nonlinear coupling to a
convection-diffusion equation for temperature.

Two key physical parameters are the Rayleigh numbe (Ra) and the
Prandtl number (Pr).::


  from firedrake import *

  N = 8

  M = UnitSquareMesh(N, N)

  V = VectorFunctionSpace(M, "CG", 2)
  W = FunctionSpace(M, "CG", 1)
  Q = FunctionSpace(M, "CG", 1)
  Z = V * W * Q

  upT = Function(Z)
  u, p, T = split(upT)
  v, q, S = TestFunctions(Z)

  Ra = Constant(200.)
  Pr = Constant(6.8)

  g = Constant((0, -1))

  F = (
      inner(grad(u), grad(v))*dx
      + inner(dot(grad(u), u), v)*dx
      - inner(p, div(v))*dx
      - Ra*Pr*inner(T*g, v)*dx
      + inner(div(u), q)*dx
      + inner(dot(grad(T), u), S)*dx
      + 1/Pr * inner(grad(T), grad(S))*dx
  )

  bcs = [
      DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4)),
      DirichletBC(Z.sub(2), Constant(1.0), (1,)),
      DirichletBC(Z.sub(2), Constant(0.0), (2,))
  ]


  nullspace = MixedVectorSpaceBasis(
      Z, [Z.sub(0), VectorSpaceBasis(constant=True), Z.sub(2)])


  prob = NonlinearVariationalProblem(F, upT, bcs=bcs, nest=False)

  solver = NonlinearVariationalSolver(prob, options_prefix='',
                                      nullspace=nullspace,
                                      extra_ctx={"velocity_space": 0}
                                      )

  solver.solve()

  File("benard.pvd").write(*upT.split())
