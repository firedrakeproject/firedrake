Rayleigh-Benard Convection
==========================
This problem involves a variable-temperature incompressible fluid.
Variations in the fluid temperature are assumed to affect the momentum
balance through a buoyant term (the Boussinesq approximation), leading
to a Navier-Stokes equation with a nonlinear coupling to a
convection-diffusion equation for temperature.

We will set up the problem using Taylor-Hood elements for
the Navier-Stokes part, and piecewise linear elements for the
temperature. ::

  from firedrake import *

  N = 128

  M = UnitSquareMesh(N, N)

  V = VectorFunctionSpace(M, "CG", 2)
  W = FunctionSpace(M, "CG", 1)
  Q = FunctionSpace(M, "CG", 1)
  Z = V * W * Q

  upT = Function(Z)
  u, p, T = split(upT)
  v, q, S = TestFunctions(Z)

Two key physical parameters are the Rayleigh number (Ra), which
measures the ratio of energy from buoyant forces to viscous
dissipation and heat conduction and the
Prandtl number (Pr), which measures the ratio of viscosity to heat
conduction. ::

  Ra = Constant(200.0)
  Pr = Constant(6.8)

Along with gravity, which points down. ::

  g = Constant((0, -1))

  F = (
      inner(grad(u), grad(v))*dx
      + inner(dot(grad(u), u), v)*dx
      - inner(p, div(v))*dx
      - (Ra/Pr)*inner(T*g, v)*dx
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


First off, we'll solve the full system using a direct solver.  As
previously, we use MUMPS, so wrap the solve in ``try/except`` to avoid
errors if it is not available. ::

  from firedrake.petsc import PETSc

  try:
     solve(F == 0, upT, bcs=bcs, nullspace=nullspace,
           solver_parameters={"mat_type": "aij",
                              "snes_monitor": None,
                              "ksp_type": "gmres",
                              "pc_type": "lu",
                              "pc_factor_mat_solver_type": "mumps"})
  except PETSc.Error as e:
      if e.ierr == 92:
          warning("MUMPS not installed, skipping direct solve")
      else:
          raise e

For our next trick, we will use a fieldsplit preconditioner.  This
time, rather than using a Schur complement, we will use a
multiplicative type (effectively block Gauss-Seidel).  As ever, this
has more options, so we'll use a parameters dictionary.  We use
matrix-free actions for the coupled operator, and solve the linearised
system with GMRES preconditioned with a multiplicative fieldsplit. ::

  parameters = {"mat_type": "matfree",
                "snes_monitor": None,
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "multiplicative",

We want to split the Navier-Stokes part off from the temperature
variable. ::

                "pc_fieldsplit_0_fields": "0,1",
                "pc_fieldsplit_1_fields": "2",

We'll invert the Navier-Stokes block with MUMPS::

                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "python",
                "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
                "fieldsplit_0_assembled_pc_type": "lu",
                "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",

the temperature block will also be inverted directly, but with plain
LU.::

                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "python",
                "fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
                "fieldsplit_1_assembled_pc_type": "lu"}

Now for the solve. ::

  upT.assign(0)
  try:
      solve(F == 0, upT, bcs=bcs, nullspace=nullspace,
            solver_parameters=parameters)
  except PETSc.Error as e:
      if e.ierr == 92:
          warning("MUMPS not installed, skipping assembled fieldsplit solve")
      else:
          raise e

Finally, we'll demonstrate recursive fieldsplitting.  We'll use the
same multiplicative fieldsplit preconditioner for the
velocity-pressure and temperature blocks, but we'll precondition the
Navier-Stokes part with :class:`~.PCDPC` using a lower Schur
complement factorisation, and approximately invert the temperature
block using algebraic multigrid.  There are lots of parameters here,
so let's run through them.  Since there are many options here, in
particular for the nested subsolves, we :ref:`specify options using
nested <nested_options_blocks>`, rather than flat, dictionaries.  The
solver parameters dictionary can either be a flat dictionary of
key-value pairs, where both the keys and the values are strings, or it
can be nested.  In the latter case, the value should be a dictionary,
of options and the key is `prepended` to all keys in the dictionary
before passing to the solver. ::

  parameters = {"mat_type": "matfree",
                "snes_monitor": None,

We'll use inexact GMRES solves to invert the Navier-Stokes block, so
the preconditioner as a whole is not stationary, hence we need
flexible GMRES. ::

               "ksp_type": "fgmres",
               "ksp_gmres_modifiedgramschmidt": True,
               "pc_type": "fieldsplit",
               "pc_fieldsplit_type": "multiplicative",

Again we split off Navier-Stokes from the temperature block ::

               "pc_fieldsplit_0_fields": "0,1",
               "pc_fieldsplit_1_fields": "2",

which we solve inexactly using preconditioned GMRES. ::

               "fieldsplit_0": {
                   "ksp_type": "gmres",
                   "ksp_gmres_modifiedgramschmidt": True,
                   "ksp_rtol": 1e-2,
                   "pc_type": "fieldsplit",
                   "pc_fieldsplit_type": "schur",
                   "pc_fieldsplit_schur_fact_type": "lower",

Invert the velocity block with a single V-cycle of algebraic
multigrid::

                   "fieldsplit_0": {
                       "ksp_type": "preonly",
                       "pc_type": "python",
                       "pc_python_type": "firedrake.AssembledPC",
                       "assembled_pc_type": "hypre"
                   },

and approximate the Schur complement inverse with PCD. ::

                   "fieldsplit_1": {
                        "ksp_type": "preonly",
                        "pc_type": "python",
                        "pc_python_type": "firedrake.PCDPC",

We need to configure the pressure mass and Poisson solves, along with
how to apply the convection-diffusion operator.  For the latter, we
will use an assembled operator this time round. ::

                        "pcd_Mp_ksp_type": "preonly",
                        "pcd_Mp_pc_type": "ilu",
                        "pcd_Kp_ksp_type": "preonly",
                        "pcd_Kp_pc_type": "hypre",
                        "pcd_Fp_mat_type": "aij"
                   }
               },

Now for the temperature block, we use a moderately coarse tolerance
for algebraic multigrid preconditioned GMRES. ::

              "fieldsplit_1": {
                   "ksp_type": "gmres",
                   "ksp_rtol": "1e-4",
                   "pc_type": "python",
                   "pc_python_type": "firedrake.AssembledPC",
                   "assembled_pc_type": "hypre"
              }
         }

And we're done with all the options.  All that's left is to solve the
problem.  Recall that the PCD preconditioner needs to know where the
velocity space lives in the velocity-pressure block, which we provide
through the application context argument.  It also needs to know the
Reynolds number, which defaults to 1.0, which happens to work for our
problem setup.  We haven't added the Rayleigh or Prandtl numbers to
the dictionary since our known preconditioners don't actually require
them, although doing so would be quite easy.::

  appctx = {"velocity_space": 0}
  upT.assign(0)

  solve(F == 0, upT, bcs=bcs, nullspace=nullspace,
        solver_parameters=parameters, appctx=appctx)

Finally, we'll output the results for visualisation. ::

  u, p, T = upT.subfunctions
  u.rename("Velocity")
  p.rename("Pressure")
  T.rename("Temperature")

  VTKFile("benard.pvd").write(u, p, T)

A runnable python script implementing this demo file is available
:demo:`here <rayleigh-benard.py>`.
