Navier-Stokes equations
=======================

We solve the Navier-Stokes equations using Taylor-Hood elements.  The
example is that of a lid-driven cavity.

.. code-block:: python

  import os
  from firedrake import *

  if os.getenv("FIREDRAKE_CI") == "1":
      # trick to speed up the Firedrake test suite
      N = 8
  else:
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

Now we'll solve the problem.  First, using a direct solver.  Again, if
MUMPS is not installed, this solve will not work, so we wrap the solve
in a ``try/except`` block.

.. code-block:: python

  from firedrake.petsc import PETSc

  try:
      solve(F == 0, up, bcs=bcs, nullspace=nullspace,
            solver_parameters={"snes_monitor": None,
                               "ksp_type": "gmres",
                               "mat_type": "aij",
                               "pc_type": "lu",
                               "pc_factor_mat_solver_type": "mumps"})
  except PETSc.Error as e:
      if e.ierr == 92:
          warning("MUMPS not installed, skipping direct solve")
      else:
          raise e

Now we'll show an example using the :class:`~.PCDPC` preconditioner
that implements the pressure convection-diffusion approximation to the
pressure Schur complement.  We'll need more solver parameters this
time, so again we'll set those up in a dictionary.

.. code-block:: python

  parameters = {"mat_type": "matfree",
                "snes_monitor": None,

We'll use a non-stationary Krylov solve for the Schur complement, so
we need to use a flexible Krylov method on the outside.

.. code-block:: python

               "ksp_type": "fgmres",
               "ksp_gmres_modifiedgramschmidt": None,
               "ksp_monitor_true_residual": None,

Now to configure the preconditioner:

.. code-block:: python

               "pc_type": "fieldsplit",
               "pc_fieldsplit_type": "schur",
               "pc_fieldsplit_schur_fact_type": "lower",

we invert the velocity block with LU:

.. code-block:: python

               "fieldsplit_0_ksp_type": "preonly",
               "fieldsplit_0_pc_type": "python",
               "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
               "fieldsplit_0_assembled_pc_type": "lu",

and invert the schur complement inexactly using GMRES, preconditioned
with PCD.

.. code-block:: python

               "fieldsplit_1_ksp_type": "gmres",
               "fieldsplit_1_ksp_rtol": 1e-4,
               "fieldsplit_1_pc_type": "python",
               "fieldsplit_1_pc_python_type": "firedrake.PCDPC",

This preconditioner requires information about the the problem that
is not easily accessible from the bilinear form. Specifically, we need
the Reynolds number and which part of the mixed velocity-pressure space
the velocity corresponds to.

.. code-block:: python

               "fieldsplit_1_pcd_Re": Re,
               "fieldsplit_1_pcd_velocity_space": u,

We now need to configure the mass and stiffness solvers in the PCD
preconditioner.  For this example, we will just invert them with LU,
although of course we can use a scalable method if we wish. First the
mass solve:

.. code-block:: python

               "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
               "fieldsplit_1_pcd_Mp_pc_type": "lu",

and the stiffness solve.

.. code-block:: python

               "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
               "fieldsplit_1_pcd_Kp_pc_type": "lu",

Finally, we just need to decide whether to apply the action of the
pressure-space convection-diffusion operator with an assembled matrix
or matrix free.  Here we will use matrix-free:

.. code-block:: python

               "fieldsplit_1_pcd_Fp_mat_type": "matfree"}

With the parameters set up, we can solve the problem, remembering to
pass in the application context so that the PCD preconditioner can
find the Reynolds number.

.. code-block:: python

  up.assign(0)

  solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters)

And finally we write the results to a file for visualisation.

.. code-block:: python

  u, p = up.subfunctions
  u.rename("Velocity")
  p.rename("Pressure")

  VTKFile("cavity.pvd").write(u, p)

A runnable python script implementing this demo file is available
:demo:`here <navier_stokes.py>`.
