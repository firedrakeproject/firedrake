Nonlinear preconditioning applied to the Allen-Cahn equation
============================================================

In the other demonstrations on nonlinear PDE, we've used Newton's method to
solve finite- dimensional systems of nonlinear equations. Suppose we want to
solve a system

.. math::
  F(u) = 0.

At each step of Newton's method, we first compute a *search direction*
:math:`v` by solving the linear system
.. math::

  dF(u)v = -F(u).

We can then use a line search along :math:`v` to obtain the next candidate
solution. When we talked about preconditioning, we have always meant applying
a preconditioner to the linear system for the search direction. A linear
preconditioner transforms a linear system into an equivalent one that is
easier to solve.

For some highly nonlinear problems, however, Newton's method can stagnate or
fail to converge, even with globalization strategies such as a line search.
For example, if :math:`dF(u)` behaves badly, we might not be able to compute a
search direction at all.

An alternative strategy is to use *nonlinear preconditioning*, abbreviated as
NPC in the following. Nonlinear preconditioning does the same thing as linear
preconditioning: it transforms the problem into an equivalent one that is
(hopefully) easier to solve. Rather than work on the linear system for the
search direction, however, NPC works directly on the nonlinear system itself.
Typically NPC is applied within a higher-level nonlinear solver. For example,
there is a nonlinear analogue of GMRES. For a review of nonlinear solver and
preconditioning strategies beyond Newton, see :cite:`brune2015composing`.

Here we will demonstrate how to use nonlinear preconditioning for the steady-
state Allen-Cahn equation

.. math::

  -\epsilon\Delta u - u + u^3 = 0

on the unit interval with Dirichlet boundary conditions :math:`u(0) = -1` and
:math:`u(1) = +1`. The Jacobian of this residual is indefinite wherever
:math:`|u| < 1/\sqrt{3}`. Newton's method can fail to compute a search
direciton when starting from an initial guess that crosses this region.

This demo is adapted from this `Chebfun example`_.

.. _Chebfun example: https://www.chebfun.org/examples/ode-nonlin/AllenCahn.html

::

  import numpy as np
  from firedrake import *

Here we use a domain of length 10, a small diffusion coefficient of 0.003,
and an initial guess that ramps from +1 at the left-hand boundary to -1 at
the right.

::

  nx = 128
  lx = 10.0
  eps = Constant(3e-3)

  mesh = IntervalMesh(nx, lx)
  Q = FunctionSpace(mesh, "CG", 1)

  x, = SpatialCoordinate(mesh)
  u_1 = Constant(1)
  u_2 = Constant(-1)
  Lx = Constant(lx)
  initial_guess = (1 - x / Lx) * u_1 + x / Lx * u_2

  bcs = [DirichletBC(Q, u_1, [1]), DirichletBC(Q, u_2, [2])]

  u = Function(Q)
  u.interpolate(initial_guess)

In order to have a good baseline to compare against, we want to use as good
a nonlinear solution strategy as possible. Here we use Newton's method with
the *critical point* line search, which is specially adapted for problems
like Allen-Cahn which can be derived from minimizing some free energy. Even
with 10 iterations of a line search, Newton's method will fail. You can try
other line search methods (secant, backtracking, etc.) and find similar
outcomes. If you pump the number of line search iterations way up, you can
make Newton's method converge... to the wrong solution!

::

  v = TestFunction(Q)
  F = (eps * inner(grad(u), grad(v)) - (u - u**3) * v) * dx

  problem = NonlinearVariationalProblem(F, u, bcs)

  newton_parameters = {
      "snes_type": "newtonls",
      "snes_monitor": None,
      "snes_converged_reason": None,
      "snes_linesearch_type": "cp",
      "snes_linesearch_max_it": 10,
      "snes_linesearch_monitor": None,
  }
  solver = NonlinearVariationalSolver(problem, solver_parameters=newton_parameters)
  try:
      solver.solve()
  except ConvergenceError as err:
      print(err)
      print("--------------------------")
      print("Solver failed to converge!")
      print("Resetting `u`")
      u.interpolate(initial_guess)

To salvage the wreck, we'll use nonlinear preconditioning. Here we define
a custom preconditioner which inherits from `AuxiliaryOperatorSNES`. The
arguments that it takes are first the nonlinear equation solver; then the
value `u_k` of the solution at the previous iteration; the current value `u`
for the solution; and a test function. It outputs the variational form which
will determine the value of `u`.

Our approach here is to solve for a candidate guess by holding back the
linear part of the reaction term in the Allen-Cahn equation. In other words,
at each step, we solve the PDE

.. math::

  -\epsilon\Delta u - u_k + u^3 = 0.

The Jacobian for this problem w.r.t. :math:`u` is symmetric and positive-
definite. We have good guarantees about the convergence of Newton's method
in that case, so we can reuse the solver parameters that we tried the first
time around for this inner problem.

::

  class AllenCahnAuxSNES(firedrake.AuxiliaryOperatorSNES):
      def form(self, snes, u_k, u, v):
          return (eps * inner(grad(u), grad(v)) + (u**3 - u_k) * v) * dx, bcs

Here we use the simplest solution strategy possible: nonlinear Richardson
iteration. We then specify the additional parameters under the key `"npc"`.

::

  solver_parameters = {
      "snes_type": "nrichardson",
      "snes_monitor": None,
      "snes_converged_reason": None,
      "npc": {
          "snes_type": "python",
          "snes_python_type": f"{__name__}.AllenCahnAuxSNES",
          "aux": newton_parameters,
      },
  }

  solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)
  solver.solve()

The Allen-Cahn equation is derivable through minimization of the free energy
functional

.. math::

  G(u) = \int\_\Omega\left(\frac{\epsilon}{2}|\nabla u|^2 + (1 - u^2)^2/4\right)dx.

To close, let's evaluate the free energy at the starting guess and at the
computed solution.

::

  G = (0.5 * eps * inner(grad(u), grad(u)) + 0.25 * (1 - u**2) ** 2) * dx
  G_initial = firedrake.assemble(firedrake.replace(G, {u: initial_guess}))
  G_final = firedrake.assemble(G)
  print(f"Initial free energy: {G_initial:0.04f}")
  print(f"Final:               {G_final:0.04f}")
