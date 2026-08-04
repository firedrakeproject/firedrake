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

The auxiliary cell and facet solvers accept their own options under the
``dwr_cell_`` and ``dwr_facet_`` prefixes, but here we leave them at their
PETSc defaults.  The enriched primal solve has no options of its own either,
so it inherits the outer solver's options below; we call
``ksp.solveTranspose`` on it, so the dual solve reuses those options too. ::

  solver_parameters = {
      "ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps",
      "snes_adapt_sequence": 5,
      "dwr_marking_fraction": 0.5,
  }

  initial_dofs = W.dim()
  problem = NonlinearVariationalProblem(F, w)
  solver = NonlinearVariationalSolver(
      problem,
      solver_parameters=solver_parameters,
      marking_callback=DWRMarkingCallback(goal),
  )
  w_adapt = solver.solve()

``solver.solve()`` returns the solution on the final adapted mesh, and
``solver.get_goal_functional()`` gives the goal functional already
reconstructed on that mesh. ::

  adapted_goal = solver.get_goal_functional()

  print(f"degrees of freedom: {initial_dofs} -> {w_adapt.function_space().dim()}")
  print(f"weighted shear traction: {assemble(adapted_goal):.8f}")

The refinement is driven by the error in this boundary traction, rather than
by a global energy norm.  Cells near the right boundary, where the goal
functional is defined, are refined preferentially over the rest of the
domain. ::

  import matplotlib.pyplot as plt
  from firedrake.pyplot import triplot

  fig, axes = plt.subplots()
  triplot(w_adapt.function_space().mesh().unique(), axes=axes)
  axes.set_aspect("equal")
  axes.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
  fig.savefig("goal_oriented_adaptivity_mesh.png", bbox_inches="tight")

.. image:: goal_oriented_adaptivity_mesh.png
   :width: 60%
   :alt: The adapted mesh, refined near the right boundary.
   :align: center

More refinement steps can be requested by increasing ``snes_adapt_sequence``
without writing an explicit solve--estimate--refine loop.
