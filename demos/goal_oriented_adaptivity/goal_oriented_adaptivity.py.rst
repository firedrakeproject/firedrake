Goal-oriented mesh adaptivity
=============================

This demo uses a dual-weighted residual (DWR) estimator to adapt a mesh for
one particular quantity of interest.  The example is the weakly symmetric
Hellinger--Reissner elasticity problem studied by Rognes and Logg in
`Automated Goal-Oriented Error Control I
<https://doi.org/10.1137/100795008>`__.  It is a useful test because its mixed
space contains both :math:`H(\mathrm{div})`- and :math:`L^2`-conforming fields.

The unknowns are the two rows of the stress :math:`\sigma`, the displacement
:math:`u`, and a scalar rotation :math:`\gamma`.  We use first-order BDM
elements for each stress row, piecewise constants for the displacement, and
continuous linears for the rotation.  The deliberately small initial mesh
keeps the demo quick; increasing its resolution and the number of refinement
steps produces the longer adaptive sequences used in the paper. ::

  from firedrake import *

  mesh = UnitSquareMesh(2, 2)
  stress_row = FunctionSpace(mesh, "BDM", 1)
  displacement = VectorFunctionSpace(mesh, "DG", 0)
  rotation = FunctionSpace(mesh, "CG", 1)
  W = stress_row * stress_row * displacement * rotation

  w = Function(W, name="elasticity solution")
  sigma0, sigma1, u, gamma = split(w)
  tau0, tau1, v, eta = TestFunctions(W)
  sigma = as_tensor((sigma0, sigma1))
  tau = as_tensor((tau0, tau1))

For shear modulus :math:`\mu` and Lamé parameter :math:`\lambda`, the
compliance tensor is

.. math::

   A\sigma = \frac{1}{2\mu}\left(\sigma
   - \frac{\lambda}{2(\mu + \lambda)}\operatorname{tr}(\sigma)I\right).

We manufacture the displacement
:math:`u_0=(xy\sin(\pi y),0)` and use the corresponding body force from the
paper.  Prescribed displacement is a natural boundary condition in this
mixed formulation. ::

  x, y = SpatialCoordinate(mesh)
  mu = Constant(1.0)
  lmbda = Constant(100.0)
  compliance = (
      sigma - lmbda/(2*(mu + lmbda))*tr(sigma)*Identity(2)
  )/(2*mu)

  body_force = as_vector((
      pi*mu*(2*x*cos(pi*y) - pi*x*y*sin(pi*y)),
      (mu + lmbda)*(pi*y*cos(pi*y) + sin(pi*y)),
  ))
  u0 = as_vector((x*y*sin(pi*y), 0))
  n = FacetNormal(mesh)

  F = (
      inner(compliance, tau)
      + dot(div(sigma), v)
      + dot(u, div(tau))
      + (sigma[0, 1] - sigma[1, 0])*eta
      + gamma*(tau[0, 1] - tau[1, 0])
  )*dx - dot(body_force, v)*dx - dot(u0, dot(tau, n))*ds

Our goal is the weighted average shear traction on the right boundary.  Its
exact value is approximately :math:`-0.06029761071`.  The DWR callback
linearises this functional, solves the low- and enriched-order dual problems,
localises the weak residual with bubble and cone functions, and performs
global Dörfler marking. ::

  psi = y*(y - 1)
  tangent = as_vector((0, 1))
  goal = psi*dot(dot(n, sigma), tangent)*ds(2)

The outer solve and all four auxiliary solver families are configured through
PETSc options.  Their option prefixes are ``dwr_primal_``, ``dwr_dual_``,
``dwr_cell_``, and ``dwr_facet_``.  Here we use direct solves throughout so
that the example focuses on adaptivity. ::

  direct = {
      "ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps",
  }
  solver_parameters = {
      **direct,
      "snes_adapt_sequence": 1,
      "dwr_marking_fraction": 0.5,
      **{
          f"dwr_{kind}_{key}": value
          for kind in ("primal", "dual")
          for key, value in direct.items()
      },
      **{
          f"dwr_{kind}_{key}": value
          for kind in ("cell", "facet")
          for key, value in {
              "ksp_type": "preonly",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps",
          }.items()
      },
  }

  initial_dofs = W.dim()
  w_adapt = solve(
      F == 0,
      w,
      solver_parameters=solver_parameters,
      marking_callback=dwr_marking_callback(goal),
  )

``solve`` returns the solution on the final adapted mesh.  Since the goal's
measure and spatial coordinate belong to the original mesh, we write the same
functional on the final mesh when reporting its value. ::

  adapted_mesh = w_adapt.function_space().mesh().unique()
  adapted_sigma0, adapted_sigma1, _, _ = split(w_adapt)
  adapted_sigma = as_tensor((adapted_sigma0, adapted_sigma1))
  _, adapted_y = SpatialCoordinate(adapted_mesh)
  adapted_normal = FacetNormal(adapted_mesh)
  adapted_goal = (
      adapted_y*(adapted_y - 1)
      * dot(dot(adapted_normal, adapted_sigma), tangent)
      * ds(2, domain=adapted_mesh)
  )

  print(f"degrees of freedom: {initial_dofs} -> {w_adapt.function_space().dim()}")
  print(f"weighted shear traction: {assemble(adapted_goal):.8f}")

The refinement is driven by the error in this boundary traction, rather than
by a global energy norm.  More refinement steps can be requested by increasing
``snes_adapt_sequence`` without writing an explicit solve--estimate--refine
loop.
