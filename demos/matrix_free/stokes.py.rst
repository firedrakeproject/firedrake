Stokes Equations
================

A simple example of a saddle-point system, we will use the Stokes
equations to demonstrate some of the ways we can do field-splitting
with matrix-free operators.  We set up the problem as a lid-driven
cavity.

As ever, we import firedrake and define a mesh.::

  from firedrake import *

  N = 64

  M = UnitSquareMesh(N, N)

  V = VectorFunctionSpace(M, "CG", 2)
  W = FunctionSpace(M, "CG", 1)
  Z = V * W

  u, p = TrialFunctions(Z)
  v, q = TestFunctions(Z)

  a = (inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q))*dx

  L = inner(Constant((0, 0)), v) * dx

The boundary conditions are defined on the velocity space.  Zero
Dirichlet conditions on the bottom and side walls, a constant :math:`u
= (1, 0)` condition on the lid.::

  bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
         DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

  up = Function(Z)

Since we do not specify boundary conditions on the pressure space, it
is only defined up to a constant.  We will remove this component of
the solution in the solver by providing the appropriate nullspace.::

  nullspace = MixedVectorSpaceBasis(
      Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

First up, we will solve the problem directly.  For this to work, the
sparse direct solver MUMPS must be installed.  Hence this solve is
wrapped in a ``try/except`` block so that an error is not raised in
the case that it is not, to do this we must import ``PETSc``::

  from firedrake.petsc import PETSc
  
To factor the matrix from this mixed system, we must specify
a ``mat_type`` of ``aij`` to the solve call.::

  try:
      solve(a == L, up, bcs=bcs, nullspace=nullspace,
            solver_parameters={"ksp_type": "gmres",
                               "mat_type": "aij",
                               "pc_type": "lu",
                               "pc_factor_mat_solver_type": "mumps"})
  except PETSc.Error as e:
      if e.ierr == 92:
          warning("MUMPS not installed, skipping direct solve")
      else:
          raise e

Now we'll use a Schur complement preconditioner using unassembled
matrices.  We can do all of this purely by changing the solver
options.  We'll define the parameters separately to run through the
options.::

  parameters = {

First up we select the unassembled matrix type::

      "mat_type": "matfree",

Now we configure the solver, using GMRES using the diagonal part of
the Schur complement factorisation to approximate the inverse.  We'll
also monitor the convergence of the residual, and ask PETSc to view
the configured Krylov solver object.::

      "ksp_type": "gmres",
      "ksp_monitor_true_residual": None,
      "ksp_view": None,
      "pc_type": "fieldsplit",
      "pc_fieldsplit_type": "schur",
      "pc_fieldsplit_schur_fact_type": "diag",

Next we configure the solvers for the blocks.  For the velocity block,
we use an :class:`.AssembledPC` and approximate the inverse of the
vector laplacian using a single multigrid V-cycle.::

      "fieldsplit_0_ksp_type": "preonly",
      "fieldsplit_0_pc_type": "python",
      "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
      "fieldsplit_0_assembled_pc_type": "hypre",

For the Schur complement block, we approximate the inverse of the
schur complement with a pressure mass inverse.  For constant viscosity
this works well.  For variable, but low-contrast viscosity, one should
use a viscosity-weighted mass-matrix.  This is achievable by passing a
dictionary with "mu" associated with the viscosity into solve.  The
MassInvPC will choose a default value of 1.0 if not set.  For high viscosity
contrasts, this preconditioner is mesh-dependent and should be replaced
by some form of approximate commutator.::

      "fieldsplit_1_ksp_type": "preonly",
      "fieldsplit_1_pc_type": "python",
      "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",

The mass inverse is dense, and therefore approximated with an incomplete
LU factorization, which we configure now::

      "fieldsplit_1_Mp_mat_type": "aij",
      "fieldsplit_1_Mp_pc_type": "ilu"
   }

Having set up the parameters, we can now go ahead and solve the
problem.::

  up.assign(0)
  solve(a == L, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters)

Last, but not least, we'll write the solution to a file for later
visualisation.  We split the function into its velocity and pressure
parts and give them reasonable names, then write them to a paraview
file.::

  u, p = up.subfunctions
  u.rename("Velocity")
  p.rename("Pressure")

  File("stokes.pvd").write(u, p)

By default, the mass matrix is assembled in the :class:`~.MassInvPC`
preconditioner, however, this can be controlled using a ``mat_type``
argument.  To do this, we must specify the ``mat_type`` inside the
preconditioner.  We can use the previous set of parameters and just
modify them slightly. ::

  parameters["fieldsplit_1_Mp_mat_type"] = "matfree"

With an unassembled matrix, of course, we are not able to use standard
preconditioners, so for this example, we will just invert the mass
matrix using unpreconditioned conjugate gradients. ::

  parameters["fieldsplit_1_Mp_pc_type"] = "ksp"
  parameters["fieldsplit_1_Mp_ksp_ksp_type"] = "cg"
  parameters["fieldsplit_1_Mp_ksp_pc_type"] = "none"

  up.assign(0)
  solve(a == L, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters)

A runnable python script implementing this demo file is available
:demo:`here <stokes.py>`.
