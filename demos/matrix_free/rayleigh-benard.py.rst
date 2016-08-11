Rayleigh-Benard Convection
==========================
This problem involves a variable-temperature incompressible fluid.
Variations in the fluid temperature are assumed to affect the momentum
balance through a buoyant term (the Boussinesq approximation), leading
to a Navier-Stokes equation with a nonlinear coupling to a
convection-diffusion equation for temperature.

Two key physical parameters are the Rayleigh number (Ra), which
measures the ratio of energy from buoyant forces to viscous
dissipation and heat conduction and the
Prandtl number (Pr), which measures the ratio of viscosity to heat
conduction::

  from firedrake import *
  from firedrake.petsc import PETSc

  N = 8

  M = UnitSquareMesh(N, N)

  V = VectorFunctionSpace(M, "CG", 2)
  W = FunctionSpace(M, "CG", 1)
  Q = FunctionSpace(M, "CG", 1)
  Z = V * W * Q

  upT = Function(Z)
  u, p, T = split(upT)
  v, q, S = TestFunctions(Z)

  OptsDB = PETSc.Options()
  Ra = Constant(OptsDB.getReal("Ra", 200.0))
  Pr = Constant(OptsDB.getReal("Pr", 6.8))

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

There are two common versions of this problem.  In one case, heat is
applied from bottom to top so that the temperature gradient is
enforced parallel to the gravitation.  In this case, the temperature
difference is applied horizontally, perpendicular to gravity.  It
tends to make prettier pictures for low Rayleigh numbers, but also
tends to take more Newton iterations since the coupling terms in the
Jacobian are a bit stronger.  Switching to the first case would be a
simple change of bits of the boundary associated with the second and
third boundary conditions below::
  
  bcs = [
      DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4)),
      DirichletBC(Z.sub(2), Constant(1.0), (1,)),
      DirichletBC(Z.sub(2), Constant(0.0), (2,))
  ]

Like Navier-Stokes, the pressure is only defined up to a constant.::

  nullspace = MixedVectorSpaceBasis(
      Z, [Z.sub(0), VectorSpaceBasis(constant=True), Z.sub(2)])


  prob = NonlinearVariationalProblem(F, upT, bcs=bcs, nest=False)

Also, if we desire to use PCD preconditioning for the Navier-Stokes
block, we need to make sure the variational solver has access to where
the velocity block lies.  The PCD preconditioner, if it cannot find a
Reynolds number in the extra context, defaults to 1.0, which works for
the Benard convection case.  We haven't added the Rayleigh or Prandtl
numbers to the dictionary since our known preconditioners don't
actually require them, although doing so would be quite easy.::
  
  solver = NonlinearVariationalSolver(prob, options_prefix='',
                                      nullspace=nullspace)

  solver.solve()

  File("benard.pvd").write(*upT.split())
